import * as vscode from 'vscode';

/**
 *  Gets the config value `mlir.<key>`, with an optional workspace folder.
 */
export function get<T>(key: string,
                       workspaceFolder: vscode.WorkspaceFolder = null,
                       defaultValue: T = undefined): T {
  return vscode.workspace.getConfiguration('mlir', workspaceFolder)
      .get<T>(key, defaultValue);
}

/**
 *  Sets the config value `mlir.<key>`.
 */
export function update<T>(key: string, value: T,
                          target?: vscode.ConfigurationTarget) {
  return vscode.workspace.getConfiguration('mlir').update(key, value, target);
}
