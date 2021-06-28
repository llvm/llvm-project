import * as vscode from 'vscode';

/**
 *  Gets the config value `mlir.<key>`.
 */
export function get<T>(key: string): T {
  return vscode.workspace.getConfiguration('mlir').get<T>(key);
}

/**
 *  Sets the config value `mlir.<key>`.
 */
export function update<T>(key: string, value: T,
                          target?: vscode.ConfigurationTarget) {
  return vscode.workspace.getConfiguration('mlir').update(key, value, target);
}
