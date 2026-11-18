import * as os from "os";
import * as path from "path";
import * as vscode from "vscode";

/**
 * Expands the character `~` to the user's home directory
 */
export function expandUser(file_path: string): string {
  if (os.platform() === "win32") {
    return file_path;
  }

  if (!file_path) {
    return "";
  }

  if (!file_path.startsWith("~")) {
    return file_path;
  }

  const path_len = file_path.length;
  if (path_len === 1) {
    return os.homedir();
  }

  if (file_path.charAt(1) === path.sep) {
    return path.join(os.homedir(), file_path.substring(1));
  }

  const sep_index = file_path.indexOf(path.sep);
  const user_name_end = sep_index === -1 ? file_path.length : sep_index;
  const user_name = file_path.substring(1, user_name_end);
  try {
    if (user_name === os.userInfo().username) {
      return path.join(os.homedir(), file_path.substring(user_name_end));
    }
  } catch (error) {
    return file_path;
  }

  return file_path;
}

// Traverse a JSON value, replacing placeholders in all strings.
export async function substitute<T>(val: T): Promise<T> {
  if (typeof val === "string") {
    const replacementPattern = /\$\{(.*?)\}/g;
    const replacementPromises: Promise<string|undefined>[] = [];
    const matches = val.matchAll(replacementPattern);
    for (const match of matches) {
      // match[1] is the first captured group
      replacementPromises.push(replacement(match[1]));
    }
    const replacements = await Promise.all(replacementPromises);
    val = val.replace(
              replacementPattern,
              // If there's no replacement available, keep the placeholder.
              match => replacements.shift() ?? match) as unknown as T;
  } else if (Array.isArray(val)) {
    val = await Promise.all(val.map(substitute)) as T;
  } else if (typeof val === "object") {
    // Substitute values but not keys, so we don't deal with collisions.
    const result = {} as {[k: string]: any};
    for (const key in val) {
      result[key] = await substitute(val[key]);
    }
    val = result as T;
  }
  return val;
}

// Subset of substitution variables that are most likely to be useful.
// https://code.visualstudio.com/docs/editor/variables-reference
async function replacement(name: string): Promise<string|undefined> {
  if (name === "userHome") {
    return os.homedir();
  }
  if (name === "workspaceRoot" || name === "workspaceFolder" ||
      name === "cwd") {
    if (vscode.workspace.rootPath !== undefined)
      return vscode.workspace.rootPath;
    if (vscode.window.activeTextEditor !== undefined)
      return path.dirname(vscode.window.activeTextEditor.document.uri.fsPath);
    return process.cwd();
  }
  if (name === "workspaceFolderBasename" &&
      vscode.workspace.rootPath !== undefined) {
    return path.basename(vscode.workspace.rootPath);
  }
  const envPrefix = "env:";
  if (name.startsWith(envPrefix))
    return process.env[name.substr(envPrefix.length)] ?? "";
  const configPrefix = "config:";
  if (name.startsWith(configPrefix)) {
    const config = vscode.workspace.getConfiguration().get(
        name.substr(configPrefix.length));
    return (typeof config === "string") ? config : undefined;
  }
  const commandPrefix = "command:";
  if (name.startsWith(commandPrefix)) {
    const commandId = name.substr(commandPrefix.length);
    try {
      return await vscode.commands.executeCommand(commandId);
    } catch (error) {
      console.warn(`LLDB DAP: Error resolving command '${commandId}':`, error);
      vscode.window.showWarningMessage(
          `LLDB DAP: Failed to resolve ${commandId}`);

      return undefined;
    }
  }

  return undefined;
}
