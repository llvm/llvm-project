import * as path from "path";
import * as vscode from "vscode";
import { createProcessTree } from "../process-tree";

interface ProcessQuickPick extends vscode.QuickPickItem {
  processId?: number;
}

/**
 * Prompts the user to select a running process.
 *
 * The return value must be a string so that it is compatible with VS Code's
 * string substitution infrastructure. The value will eventually be converted
 * to a number by the debug configuration provider.
 *
 * @param configuration The related debug configuration, if any
 * @returns The pid of the process as a string or undefined if cancelled.
 */
export async function pickProcess(
  configuration?: vscode.DebugConfiguration,
): Promise<string | undefined> {
  const processTree = createProcessTree();
  const selectedProcess = await vscode.window.showQuickPick<ProcessQuickPick>(
    processTree.listAllProcesses().then((processes): ProcessQuickPick[] => {
      // Sort by start date in descending order
      processes.sort((a, b) => b.start - a.start);
      // Filter by program if requested
      if (typeof configuration?.program === "string") {
        processes = processes.filter(
          (proc) => proc.command === configuration.program,
        );
        // Show a better message if all processes were filtered out
        if (processes.length === 0) {
          return [
            {
              label: "No processes matched the debug configuration's program",
            },
          ];
        }
      }
      // Convert to a QuickPickItem
      return processes.map((proc) => {
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
      matchOnDetail: true,
    },
  );
  return selectedProcess?.processId?.toString();
}
