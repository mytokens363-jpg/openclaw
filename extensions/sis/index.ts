/**
 * SIS — Self-Improving System for OpenClaw
 *
 * Learns from every agent run. Extracts lessons using local Qwen3 inference,
 * stores them in SQLite with FTS5, injects relevant lessons before each run.
 *
 * Lesson lifecycle:
 *   agent_end  → extract lessons from conversation using local model
 *             → store in SQLite with confidence=0.5, tags, related_tools
 *   before_prompt_build → FTS5 search relevant lessons by current prompt
 *                       → inject top N lessons into system context
 *   feedback   → confidence rises (positive) / falls (negative)
 *   decay      → confidence drops 0.05 per 30 days without validation
 *   promotion  → confidence caps at 0.95 after 5+ confirmed occurrences
 *
 * Lesson types: mistake | success | workaround | discovery
 * Pattern types: temporal | sequential | failure | success
 */

import { randomUUID } from "node:crypto";
import { Type } from "@sinclair/typebox";
import OpenAI from "openai";
import type Database from "better-sqlite3";
import { definePluginEntry, type OpenClawPluginApi } from "./api.js";

// ============================================================================
// Config defaults
// ============================================================================

const DEFAULT_DB_PATH = "~/.openclaw/sis.db";
const DEFAULT_INFERENCE_URL = "http://localhost:8080/v1";
const DEFAULT_MODEL = "qwen3.5-122b";
const DEFAULT_MAX_LESSONS = 5;
const DEFAULT_MIN_CONFIDENCE = 0.5;
const DECAY_AFTER_DAYS = 30;
const DECAY_RATE = 0.05;

// ============================================================================
// Types
// ============================================================================

type LessonType = "mistake" | "success" | "workaround" | "discovery";
type PatternType = "temporal" | "sequential" | "failure" | "success";

type Lesson = {
  id: string;
  type: LessonType;
  context: string;
  action: string;
  outcome: string;
  lesson: string;
  correction: string | null;
  confidence: number;
  occurrences: number;
  last_seen: number;
  tags: string;       // JSON array
  related_tools: string; // JSON array
  created_at: number;
};

type Pattern = {
  id: string;
  type: PatternType;
  description: string;
  occurrences: number;
  last_seen: number;
  created_at: number;
};

type LessonExtraction = {
  lessons: Array<{
    type: LessonType;
    context: string;
    action: string;
    outcome: string;
    lesson: string;
    correction?: string;
    tags?: string[];
    related_tools?: string[];
  }>;
  patterns: Array<{
    type: PatternType;
    description: string;
  }>;
};

// ============================================================================
// SIS SQLite Store
// ============================================================================

class SISStore {
  private db: Database.Database | null = null;

  constructor(private readonly dbPath: string) {}

  private getDb(): Database.Database {
    if (this.db) return this.db;
    throw new Error("SISStore not initialized. Call init() first.");
  }

  async init(): Promise<void> {
    const { default: BetterSQLite3 } = await import("better-sqlite3");
    // Resolve ~ in path
    const resolvedPath = this.dbPath.replace(/^~/, process.env["HOME"] ?? "~");
    this.db = new BetterSQLite3(resolvedPath);
    this.db.pragma("journal_mode = WAL");
    this.db.pragma("foreign_keys = ON");
    this.runMigrations();
  }

  private runMigrations(): void {
    const db = this.getDb();

    // Lessons table
    db.exec(`
      CREATE TABLE IF NOT EXISTS sis_lessons (
        id             TEXT PRIMARY KEY,
        type           TEXT NOT NULL,
        context        TEXT NOT NULL,
        action         TEXT NOT NULL,
        outcome        TEXT NOT NULL,
        lesson         TEXT NOT NULL,
        correction     TEXT,
        confidence     REAL NOT NULL DEFAULT 0.5,
        occurrences    INTEGER NOT NULL DEFAULT 1,
        last_seen      INTEGER NOT NULL,
        tags           TEXT NOT NULL DEFAULT '[]',
        related_tools  TEXT NOT NULL DEFAULT '[]',
        created_at     INTEGER NOT NULL
      )
    `);

    // FTS5 virtual table for full-text search over lesson content
    db.exec(`
      CREATE VIRTUAL TABLE IF NOT EXISTS sis_lessons_fts
      USING fts5(
        id UNINDEXED,
        context,
        lesson,
        tags,
        content='sis_lessons',
        content_rowid='rowid'
      )
    `);

    // Triggers to keep FTS in sync
    db.exec(`
      CREATE TRIGGER IF NOT EXISTS sis_lessons_ai
        AFTER INSERT ON sis_lessons BEGIN
          INSERT INTO sis_lessons_fts(rowid, id, context, lesson, tags)
          VALUES (new.rowid, new.id, new.context, new.lesson, new.tags);
        END;

      CREATE TRIGGER IF NOT EXISTS sis_lessons_ad
        AFTER DELETE ON sis_lessons BEGIN
          INSERT INTO sis_lessons_fts(sis_lessons_fts, rowid, id, context, lesson, tags)
          VALUES ('delete', old.rowid, old.id, old.context, old.lesson, old.tags);
        END;

      CREATE TRIGGER IF NOT EXISTS sis_lessons_au
        AFTER UPDATE ON sis_lessons BEGIN
          INSERT INTO sis_lessons_fts(sis_lessons_fts, rowid, id, context, lesson, tags)
          VALUES ('delete', old.rowid, old.id, old.context, old.lesson, old.tags);
          INSERT INTO sis_lessons_fts(rowid, id, context, lesson, tags)
          VALUES (new.rowid, new.id, new.context, new.lesson, new.tags);
        END;
    `);

    // Patterns table
    db.exec(`
      CREATE TABLE IF NOT EXISTS sis_patterns (
        id          TEXT PRIMARY KEY,
        type        TEXT NOT NULL,
        description TEXT NOT NULL,
        occurrences INTEGER NOT NULL DEFAULT 1,
        last_seen   INTEGER NOT NULL,
        created_at  INTEGER NOT NULL
      )
    `);
  }

  storeLessons(lessons: LessonExtraction["lessons"]): number {
    const db = this.getDb();
    const stmt = db.prepare(`
      INSERT INTO sis_lessons
        (id, type, context, action, outcome, lesson, correction, confidence, occurrences,
         last_seen, tags, related_tools, created_at)
      VALUES
        (@id, @type, @context, @action, @outcome, @lesson, @correction, @confidence,
         @occurrences, @last_seen, @tags, @related_tools, @created_at)
      ON CONFLICT(id) DO NOTHING
    `);

    const insertMany = db.transaction((rows: Lesson[]) => {
      for (const row of rows) stmt.run(row);
    });

    const now = Date.now();
    const rows: Lesson[] = lessons.map((l) => ({
      id: randomUUID(),
      type: l.type,
      context: l.context,
      action: l.action,
      outcome: l.outcome,
      lesson: l.lesson,
      correction: l.correction ?? null,
      confidence: 0.5,
      occurrences: 1,
      last_seen: now,
      tags: JSON.stringify(l.tags ?? []),
      related_tools: JSON.stringify(l.related_tools ?? []),
      created_at: now,
    }));

    insertMany(rows);
    return rows.length;
  }

  storePatterns(patterns: LessonExtraction["patterns"]): void {
    const db = this.getDb();
    const now = Date.now();

    const stmt = db.prepare(`
      INSERT INTO sis_patterns (id, type, description, occurrences, last_seen, created_at)
      VALUES (@id, @type, @description, @occurrences, @last_seen, @created_at)
      ON CONFLICT(id) DO UPDATE SET
        occurrences = occurrences + 1,
        last_seen = excluded.last_seen
    `);

    for (const p of patterns) {
      stmt.run({
        id: randomUUID(),
        type: p.type,
        description: p.description,
        occurrences: 1,
        last_seen: now,
        created_at: now,
      });
    }
  }

  searchLessons(query: string, limit: number, minConfidence: number): Lesson[] {
    const db = this.getDb();

    // FTS5 search — strip special chars to avoid FTS syntax errors
    const safeQuery = query.replace(/[^a-zA-Z0-9 ]/g, " ").trim();

    if (!safeQuery) {
      // Fallback: return highest-confidence lessons
      return db
        .prepare(
          `SELECT * FROM sis_lessons
           WHERE confidence >= ?
           ORDER BY confidence DESC, occurrences DESC
           LIMIT ?`,
        )
        .all(minConfidence, limit) as Lesson[];
    }

    try {
      return db
        .prepare(
          `SELECT l.* FROM sis_lessons l
           JOIN sis_lessons_fts fts ON fts.id = l.id
           WHERE sis_lessons_fts MATCH ?
             AND l.confidence >= ?
           ORDER BY l.confidence DESC, l.occurrences DESC
           LIMIT ?`,
        )
        .all(`${safeQuery}`, minConfidence, limit) as Lesson[];
    } catch {
      // FTS syntax error — fallback to confidence sort
      return db
        .prepare(
          `SELECT * FROM sis_lessons
           WHERE confidence >= ?
           ORDER BY confidence DESC
           LIMIT ?`,
        )
        .all(minConfidence, limit) as Lesson[];
    }
  }

  boostLesson(id: string, delta = 0.1): void {
    const db = this.getDb();
    db.prepare(
      `UPDATE sis_lessons SET
         confidence = MIN(0.95, confidence + ?),
         occurrences = occurrences + 1,
         last_seen = ?
       WHERE id = ?`,
    ).run(delta, Date.now(), id);
  }

  decayOldLessons(): number {
    const db = this.getDb();
    const cutoff = Date.now() - DECAY_AFTER_DAYS * 24 * 60 * 60 * 1000;
    const result = db.prepare(
      `UPDATE sis_lessons SET
         confidence = MAX(0.1, confidence - ?)
       WHERE last_seen < ? AND occurrences < 5`,
    ).run(DECAY_RATE, cutoff);
    return result.changes;
  }

  count(): { lessons: number; patterns: number } {
    const db = this.getDb();
    const lessons = (db.prepare("SELECT COUNT(*) as n FROM sis_lessons").get() as { n: number }).n;
    const patterns = (db.prepare("SELECT COUNT(*) as n FROM sis_patterns").get() as { n: number }).n;
    return { lessons, patterns };
  }

  listLessons(limit = 20): Lesson[] {
    const db = this.getDb();
    return db
      .prepare("SELECT * FROM sis_lessons ORDER BY confidence DESC, last_seen DESC LIMIT ?")
      .all(limit) as Lesson[];
  }

  close(): void {
    this.db?.close();
    this.db = null;
  }
}

// ============================================================================
// Lesson Extractor — local Qwen3 inference
// ============================================================================

class LessonExtractor {
  private client: OpenAI;

  constructor(
    baseUrl: string,
    private readonly model: string,
    apiKey: string,
  ) {
    this.client = new OpenAI({
      apiKey: apiKey || "none",
      baseURL: baseUrl,
    });
  }

  async extract(conversation: string): Promise<LessonExtraction | null> {
    const systemPrompt = `You are a lesson extraction system. Analyze the AI agent conversation and extract actionable lessons.

Return ONLY a JSON object with this exact schema — no markdown, no preamble:
{
  "lessons": [
    {
      "type": "mistake|success|workaround|discovery",
      "context": "What was being attempted (1 sentence)",
      "action": "What action was taken (1 sentence)",
      "outcome": "What happened as a result (1 sentence)",
      "lesson": "What should be remembered for next time (1-2 sentences)",
      "correction": "How to do it correctly (if type=mistake, else null)",
      "tags": ["tag1", "tag2"],
      "related_tools": ["tool_name1"]
    }
  ],
  "patterns": [
    {
      "type": "temporal|sequential|failure|success",
      "description": "Short description of the pattern (1 sentence)"
    }
  ]
}

Rules:
- Only extract lessons that would genuinely help future runs
- Leave lessons array empty if nothing significant happened
- Leave patterns array empty if no clear patterns
- Max 3 lessons and 2 patterns per extraction
- Be specific — generic lessons like "be careful" are useless
- Tags should be lowercase, hyphenated (e.g. "whatsapp", "tool-call", "api-error")`;

    try {
      const response = await this.client.chat.completions.create({
        model: this.model,
        messages: [
          { role: "system", content: systemPrompt },
          {
            role: "user",
            content: `Extract lessons from this conversation:\n\n${conversation.slice(0, 6000)}`,
          },
        ],
        max_tokens: 1500,
        temperature: 0.2,
      });

      const raw = response.choices[0]?.message?.content ?? "";
      const cleaned = raw
        .replace(/```json\s*/gi, "")
        .replace(/```\s*/g, "")
        .trim();

      const parsed = JSON.parse(cleaned) as LessonExtraction;

      // Validate structure
      if (!Array.isArray(parsed.lessons) || !Array.isArray(parsed.patterns)) {
        return null;
      }

      return parsed;
    } catch {
      return null;
    }
  }
}

// ============================================================================
// Conversation formatter
// ============================================================================

function formatConversation(messages: unknown[]): string {
  const lines: string[] = [];
  for (const msg of messages) {
    if (!msg || typeof msg !== "object") continue;
    const m = msg as Record<string, unknown>;
    const role = String(m["role"] ?? "");
    if (!["user", "assistant"].includes(role)) continue;

    const content = m["content"];
    let text = "";
    if (typeof content === "string") {
      text = content;
    } else if (Array.isArray(content)) {
      text = content
        .filter(
          (b): b is Record<string, unknown> =>
            !!b && typeof b === "object" && (b as Record<string, unknown>)["type"] === "text",
        )
        .map((b) => String(b["text"] ?? ""))
        .join(" ");
    }

    if (text.trim()) {
      lines.push(`[${role.toUpperCase()}]: ${text.trim().slice(0, 500)}`);
    }
  }
  return lines.join("\n\n");
}

// ============================================================================
// Prompt formatter
// ============================================================================

function formatLessonsForPrompt(lessons: Lesson[]): string {
  if (lessons.length === 0) return "";

  const lines = lessons.map((l) => {
    const prefix = {
      mistake: "MISTAKE",
      success: "SUCCESS",
      workaround: "WORKAROUND",
      discovery: "DISCOVERY",
    }[l.type];

    const parts = [`- [${prefix}] ${l.lesson}`];
    if (l.correction) parts.push(`  → Fix: ${l.correction}`);
    if (l.related_tools !== "[]") {
      const tools = JSON.parse(l.related_tools) as string[];
      if (tools.length > 0) parts.push(`  → Tools: ${tools.join(", ")}`);
    }
    return parts.join("\n");
  });

  return [
    "## Lessons from past experience",
    "Apply these lessons to avoid repeating mistakes and reinforce what works:",
    ...lines,
  ].join("\n");
}

// ============================================================================
// Plugin Definition
// ============================================================================

export default definePluginEntry({
  id: "sis",
  name: "Self-Improving System (SIS)",
  description:
    "Learns from every agent run. Extracts lessons using local Qwen3, " +
    "stores patterns in SQLite, injects relevant knowledge before each session. " +
    "Grows smarter the more it runs. Zero cloud dependency.",

  register(api: OpenClawPluginApi) {
    const cfg = api.pluginConfig as {
      inferenceUrl?: string;
      inferenceModel?: string;
      inferenceApiKey?: string;
      dbPath?: string;
      maxLessonsInPrompt?: number;
      minConfidenceToInject?: number;
      enabled?: boolean;
    };

    if (cfg.enabled === false) {
      api.logger.info("sis: disabled via config");
      return;
    }

    const inferenceUrl = cfg.inferenceUrl ?? DEFAULT_INFERENCE_URL;
    const model = cfg.inferenceModel ?? DEFAULT_MODEL;
    const apiKey = cfg.inferenceApiKey ?? "none";
    const dbPath = cfg.dbPath ?? DEFAULT_DB_PATH;
    const maxLessons = cfg.maxLessonsInPrompt ?? DEFAULT_MAX_LESSONS;
    const minConfidence = cfg.minConfidenceToInject ?? DEFAULT_MIN_CONFIDENCE;

    const store = new SISStore(dbPath);
    const extractor = new LessonExtractor(inferenceUrl, model, apiKey);

    let initialized = false;

    api.logger.info(
      `sis: registered (inference: ${inferenceUrl}, model: ${model}, db: ${dbPath})`,
    );

    // ==========================================================================
    // Hook: inject lessons before each agent run
    // ==========================================================================

    api.on("before_prompt_build", async (event) => {
      if (!initialized) return;
      if (!event.prompt || event.prompt.length < 5) return;

      try {
        const lessons = store.searchLessons(event.prompt, maxLessons, minConfidence);
        if (lessons.length === 0) return;

        const context = formatLessonsForPrompt(lessons);
        if (!context) return;

        api.logger.info(`sis: injecting ${lessons.length} lessons into context`);

        return { appendSystemContext: context };
      } catch (err) {
        api.logger.warn(`sis: lesson injection failed: ${String(err)}`);
      }
    });

    // ==========================================================================
    // Hook: extract lessons after each agent run
    // ==========================================================================

    api.on("agent_end", async (event) => {
      if (!initialized) return;
      if (!event.success || !event.messages || event.messages.length === 0) return;

      try {
        const conversation = formatConversation(event.messages as unknown[]);
        if (!conversation || conversation.length < 50) return;

        api.logger.info("sis: extracting lessons from completed run");

        const extraction = await extractor.extract(conversation);
        if (!extraction) {
          api.logger.warn("sis: lesson extraction returned null (model parse error?)");
          return;
        }

        let stored = 0;
        if (extraction.lessons.length > 0) {
          stored = store.storeLessons(extraction.lessons);
        }
        if (extraction.patterns.length > 0) {
          store.storePatterns(extraction.patterns);
        }

        if (stored > 0 || extraction.patterns.length > 0) {
          api.logger.info(
            `sis: stored ${stored} lessons, ${extraction.patterns.length} patterns`,
          );
        }
      } catch (err) {
        api.logger.warn(`sis: lesson extraction failed: ${String(err)}`);
      }
    });

    // ==========================================================================
    // Tools
    // ==========================================================================

    api.registerTool(
      {
        name: "sis_lessons",
        label: "SIS Lessons",
        description: "Search and inspect lessons the agent has learned from past runs.",
        parameters: Type.Object({
          query: Type.Optional(Type.String({ description: "Search query (empty = show top lessons)" })),
          limit: Type.Optional(Type.Number({ description: "Max results (default: 10)" })),
        }),
        async execute(_id, params) {
          if (!initialized) {
            return {
              content: [{ type: "text", text: "SIS not yet initialized." }],
              details: {},
            };
          }

          const { query, limit = 10 } = params as { query?: string; limit?: number };

          const lessons = query
            ? store.searchLessons(query, limit, 0)
            : store.listLessons(limit);

          if (lessons.length === 0) {
            return {
              content: [{ type: "text", text: "No lessons found yet. SIS learns from agent runs." }],
              details: { count: 0 },
            };
          }

          const text = lessons
            .map(
              (l, i) =>
                `${i + 1}. [${l.type.toUpperCase()}] conf:${l.confidence.toFixed(2)} occ:${l.occurrences}\n` +
                `   ${l.lesson}${l.correction ? `\n   → ${l.correction}` : ""}`,
            )
            .join("\n\n");

          const { lessons: lCount, patterns: pCount } = store.count();
          return {
            content: [
              {
                type: "text",
                text: `SIS Knowledge Base: ${lCount} lessons, ${pCount} patterns\n\n${text}`,
              },
            ],
            details: { count: lessons.length, total: lCount },
          };
        },
      },
      { name: "sis_lessons" },
    );

    api.registerTool(
      {
        name: "sis_boost",
        label: "SIS Boost Lesson",
        description: "Confirm a lesson was helpful, boosting its confidence score.",
        parameters: Type.Object({
          lessonId: Type.String({ description: "Lesson ID to boost" }),
        }),
        async execute(_id, params) {
          const { lessonId } = params as { lessonId: string };
          store.boostLesson(lessonId);
          return {
            content: [{ type: "text", text: `Lesson ${lessonId} confidence boosted.` }],
            details: { action: "boosted", id: lessonId },
          };
        },
      },
      { name: "sis_boost" },
    );

    // ==========================================================================
    // CLI Commands
    // ==========================================================================

    api.registerCli(
      ({ program }) => {
        const sis = program.command("sis").description("Self-Improving System commands");

        sis
          .command("stats")
          .description("Show SIS knowledge base statistics")
          .action(() => {
            if (!initialized) {
              console.log("SIS not initialized. Start the gateway first.");
              return;
            }
            const { lessons, patterns } = store.count();
            console.log(`Lessons: ${lessons}`);
            console.log(`Patterns: ${patterns}`);
          });

        sis
          .command("lessons")
          .description("List top lessons")
          .option("--limit <n>", "Max results", "20")
          .action((opts: { limit: string }) => {
            if (!initialized) {
              console.log("SIS not initialized.");
              return;
            }
            const lessons = store.listLessons(parseInt(opts.limit, 10));
            for (const l of lessons) {
              console.log(`[${l.type.toUpperCase()}] conf:${l.confidence.toFixed(2)} occ:${l.occurrences}`);
              console.log(`  ${l.lesson}`);
              if (l.correction) console.log(`  → ${l.correction}`);
              console.log(`  id: ${l.id}`);
              console.log();
            }
          });

        sis
          .command("search <query>")
          .description("Search lessons by keyword")
          .action((query: string) => {
            if (!initialized) {
              console.log("SIS not initialized.");
              return;
            }
            const lessons = store.searchLessons(query, 10, 0);
            for (const l of lessons) {
              console.log(`[${l.type.toUpperCase()}] ${l.lesson}`);
              console.log(`  id: ${l.id} conf:${l.confidence.toFixed(2)}`);
            }
          });

        sis
          .command("decay")
          .description("Run manual decay pass on stale lessons")
          .action(() => {
            if (!initialized) {
              console.log("SIS not initialized.");
              return;
            }
            const changed = store.decayOldLessons();
            console.log(`Decayed ${changed} stale lessons.`);
          });
      },
      { commands: ["sis"] },
    );

    // ==========================================================================
    // Service
    // ==========================================================================

    api.registerService({
      id: "sis",
      start: async () => {
        await store.init();
        initialized = true;
        const { lessons, patterns } = store.count();

        // Run decay on startup to prune stale lessons
        const decayed = store.decayOldLessons();

        api.logger.info(
          `sis: ready — ${lessons} lessons, ${patterns} patterns` +
          (decayed > 0 ? `, ${decayed} decayed` : ""),
        );
      },
      stop: () => {
        store.close();
        initialized = false;
        api.logger.info("sis: closed");
      },
    });
  },
});
