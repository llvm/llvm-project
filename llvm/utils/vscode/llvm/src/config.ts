/**
 * This file was copied from /mlir/utils/vscode/src/config.ts and adapted for use in LLVM
 */

import * as vscode from 'vscode';

/**
 *  Gets the config value `llvm.<key>`, with an optional workspace folder.
 */
export function get<T>(key: string,
  workspaceFolder: vscode.WorkspaceFolder = null,
  defaultValue: T = undefined): T {
  return vscode.workspace.getConfiguration('llvm', workspaceFolder)
    .get<T>(key, defaultValue);
}

/**
 *  Sets the config value `llvm.<key>`.
 */
export function update<T>(key: string, value: T,
  target?: vscode.ConfigurationTarget) {
  return vscode.workspace.getConfiguration('llvm').update(key, value, target);
}
