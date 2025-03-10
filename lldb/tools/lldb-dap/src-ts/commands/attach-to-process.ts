import * as vscode from "vscode";

export async function attachToProcess(): Promise<boolean> {
  return await vscode.debug.startDebugging(undefined, {
    type: "lldb-dap",
    request: "attach",
    name: "Attach to Process",
    pid: "${command:pickProcess}",
  });
}
