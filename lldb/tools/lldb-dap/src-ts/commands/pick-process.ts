import * as path from "path";
import * as vscode from "vscode";
import { createProcessTree } from "../process-tree";

interface ProcessQuickPick extends vscode.QuickPickItem {
  processId: number;
}

/**
 * Prompts the user to select a running process.
 *
 * @returns The pid of the process as a string or undefined if cancelled.
 */
export async function pickProcess(): Promise<string | undefined> {
  const processTree = createProcessTree();
  const selectedProcess = await vscode.window.showQuickPick<ProcessQuickPick>(
    processTree.listAllProcesses().then((processes): ProcessQuickPick[] => {
      return processes
        .sort((a, b) => b.start - a.start) // Sort by start date in descending order
        .map((proc) => {
          return {
            processId: proc.id,
            label: path.basename(proc.command),
            description: proc.id.toString(),
            detail: proc.arguments,
          } satisfies ProcessQuickPick;
        });
    }),
    {
      placeHolder: "Select a process to attach the debugger to",
    },
  );
  if (!selectedProcess) {
    return;
  }
  return selectedProcess.processId.toString();
}
