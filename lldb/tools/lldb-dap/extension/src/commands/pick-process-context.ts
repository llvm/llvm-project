import * as vscode from "vscode";

/**
 * Context captured at debug-configuration-resolution time so that the
 * `lldb-dap.pickProcess` command (invoked later by VS Code's variable
 * substitution for `${command:PickProcess}`) can target the correct lldb-dap
 * binary and platform.
 *
 * VS Code's command substitution does not pass the in-flight debug
 * configuration to the command handler, so we stash it here instead. The flow
 * is:
 *
 *   1. `resolveDebugConfiguration(folder, config)`   → `set(...)`
 *   2. VS Code expands `${command:PickProcess}`      → `take()`
 *   3. `resolveDebugConfigurationWithSubstitutedVariables(...)`
 *
 * This is a single-slot singleton: we rely on VS Code serializing debug
 * session starts, which is the observed behavior but not formally documented.
 * If two sessions ever race, the second one wins and the first falls back to
 * the default context — an acceptable degradation.
 */
export interface PickProcessContext {
  folder: vscode.WorkspaceFolder | undefined;
  debugConfiguration: vscode.DebugConfiguration;
}

let pending: PickProcessContext | undefined;

export function setPickProcessContext(ctx: PickProcessContext): void {
  pending = ctx;
}

/** Returns the stashed context (if any) and clears the slot. */
export function takePickProcessContext(): PickProcessContext | undefined {
  const ctx = pending;
  pending = undefined;
  return ctx;
}

/**
 * Discards any stashed context. Called at the start of each configuration
 * resolution so that a cancelled prior session cannot leak stale context
 * into the next one.
 */
export function clearPickProcessContext(): void {
  pending = undefined;
}
