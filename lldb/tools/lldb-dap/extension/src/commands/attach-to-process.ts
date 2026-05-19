import * as vscode from "vscode";

import { LogFilePathProvider } from "../logging";

import { pickProcess } from "./pick-process";

/**
 * Prompts the user to select a running process and starts a new lldb-dap attach
 * debug session against it. Invoked from the command palette, without requiring
 * a launch.json configuration.
 */
export async function attachToProcess(
  logger: vscode.LogOutputChannel,
  logFilePath: LogFilePathProvider,
): Promise<boolean> {
  const pid = await pickProcess(logger, logFilePath);
  if (pid === undefined) {
    return false;
  }
  return vscode.debug.startDebugging(undefined, {
    type: "lldb-dap",
    request: "attach",
    name: "Attach",
    pid,
  });
}
