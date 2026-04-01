/**
 * Global Plugin Hook Runner
 *
 * Singleton hook runner that's initialized when plugins are loaded
 * and can be called from anywhere in the codebase.
 */

import { createSubsystemLogger } from "../logging/subsystem.js";
import { resolveGlobalSingleton } from "../shared/global-singleton.js";
import type { GlobalHookRunnerRegistry } from "./hook-registry.types.js";
import type { PluginHookGatewayContext, PluginHookGatewayStopEvent } from "./hook-types.js";
import { createHookRunner, type HookRunner } from "./hooks.js";

type HookRunnerGlobalState = {
  hookRunner: HookRunner | null;
  registry: GlobalHookRunnerRegistry | null;
};

const hookRunnerGlobalStateKey = Symbol.for("openclaw.plugins.hook-runner-global-state");
const getState = () =>
  resolveGlobalSingleton<HookRunnerGlobalState>(hookRunnerGlobalStateKey, () => ({
    hookRunner: null,
    registry: null,
  }));

const getLog = () => createSubsystemLogger("plugins");

/**
 * Initialize the global hook runner with a plugin registry.
 * Called once when plugins are loaded during gateway startup.
 */
export function initializeGlobalHookRunner(registry: GlobalHookRunnerRegistry): void {
  const state = getState();
  const log = getLog();
  state.registry = registry;
  state.hookRunner = createHookRunner(registry, {
    logger: {
      debug: (msg) => log.debug(msg),
      warn: (msg) => log.warn(msg),
      error: (msg) => log.error(msg),
    },
    catchErrors: true,
    failurePolicyByHook: {
      before_tool_call: "fail-closed",
    },
  });

  // Count typed hooks (api.on() registrations) not legacy hooks (registry.hooks).
  // registry.hooks is the old internal hook system; registry.typedHooks is what
  // plugins register via api.on(). Reading the wrong array caused the log to always
  // report 0 hooks even when typed hooks were successfully registered, making it
  // impossible to verify hook initialization from logs (issue #5513).
  const hookCount = registry.typedHooks.length;
  if (hookCount > 0) {
    log.info(`hook runner initialized with ${hookCount} registered typed hooks`);
  } else {
    log.debug("hook runner initialized (no typed hooks registered)");
  }
}

/**
 * Get the global hook runner.
 * Returns null if plugins haven't been loaded yet.
 */
export function getGlobalHookRunner(): HookRunner | null {
  return getState().hookRunner;
}

/**
 * Get the global plugin registry.
 * Returns null if plugins haven't been loaded yet.
 */
export function getGlobalPluginRegistry(): GlobalHookRunnerRegistry | null {
  return getState().registry;
}

/**
 * Check if any hooks are registered for a given hook name.
 *
 * Returns false (not throws) when the hook runner hasn't been initialized yet,
 * which is the correct behavior for call sites that guard with this function.
 * A debug log is emitted so unexpected null states surface in verbose mode.
 */
export function hasGlobalHooks(hookName: Parameters<HookRunner["hasHooks"]>[0]): boolean {
  const { hookRunner } = getState();
  if (!hookRunner) {
    // Hook runner not yet initialized — plugins may not have loaded yet.
    // This is expected during early gateway startup but not during agent runs.
    getLog().debug(`hasGlobalHooks("${hookName}") called before hook runner initialized`);
    return false;
  }
  return hookRunner.hasHooks(hookName);
}

export async function runGlobalGatewayStopSafely(params: {
  event: PluginHookGatewayStopEvent;
  ctx: PluginHookGatewayContext;
  onError?: (err: unknown) => void;
}): Promise<void> {
  const log = getLog();
  const hookRunner = getGlobalHookRunner();
  if (!hookRunner?.hasHooks("gateway_stop")) {
    return;
  }
  try {
    await hookRunner.runGatewayStop(params.event, params.ctx);
  } catch (err) {
    if (params.onError) {
      params.onError(err);
      return;
    }
    log.warn(`gateway_stop hook failed: ${String(err)}`);
  }
}

/**
 * Reset the global hook runner (for testing).
 */
export function resetGlobalHookRunner(): void {
  const state = getState();
  state.hookRunner = null;
  state.registry = null;
}
