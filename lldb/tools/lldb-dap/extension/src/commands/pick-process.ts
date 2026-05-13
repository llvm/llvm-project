import { execFile } from "node:child_process";
import * as path from "node:path";
import { promisify } from "node:util";
import * as vscode from "vscode";

import { createDebugAdapterExecutable } from "../debug-adapter-factory";
import { LogFilePathProvider } from "../logging";
import { LldbDapProcessTree, Process } from "../process-tree";

import { takePickProcessContext } from "./pick-process-context";

const exec = promisify(execFile);

/**
 * Cache of which lldb-dap binaries have been observed to support the
 * `--list-processes` flag, keyed by executable path. We probe once per path.
 */
const listProcessesSupportCache = new Map<string, Promise<boolean>>();

/** Probes `<exe> --help` to see whether `--list-processes` is advertised. */
function isListProcessesSupported(exe: string): Promise<boolean> {
  const cached = listProcessesSupportCache.get(exe);
  if (cached) {
    return cached;
  }
  const probe = exec(exe, ["--help"])
    .then(({ stdout }) => /--list-processes/.test(stdout))
    .catch(() => false);
  listProcessesSupportCache.set(exe, probe);
  return probe;
}

interface ProcessQuickPick extends vscode.QuickPickItem {
  processId: number;
}

/**
 * Prompts the user to select a running process, enumerated by `lldb-dap
 * --list-processes`. When invoked as `${command:PickProcess}` from a
 * launch.json the matching {@link PickProcessContext} stashed by
 * `resolveDebugConfiguration` lets us target the right lldb-dap binary and
 * platform (for remote attach).
 *
 * The return value must be a string so that it is compatible with VS Code's
 * variable substitution infrastructure. The debug configuration provider
 * converts it to a number before the attach request reaches lldb-dap.
 *
 * @returns the pid of the process as a string, or `undefined` if cancelled.
 */
export async function pickProcess(
  logger: vscode.LogOutputChannel,
  logFilePath: LogFilePathProvider,
): Promise<string | undefined> {
  const ctx = takePickProcessContext();
  const executable = await createDebugAdapterExecutable(
    logger,
    logFilePath,
    ctx?.folder,
    ctx?.debugConfiguration ?? { type: "lldb-dap", request: "attach", name: "Attach" },
  );

  if (!(await isListProcessesSupported(executable.command))) {
    await vscode.window.showErrorMessage(
      "The lldb-dap binary does not support the --list-processes flag " +
        "required by the process picker. Please update to a newer version.",
      { modal: true, detail: executable.command },
    );
    return undefined;
  }

  const tree = new LldbDapProcessTree(executable.command, {
    platformName: ctx?.debugConfiguration.platformName,
    platformUrl: ctx?.debugConfiguration.platformUrl,
  });

  const selectedProcess = await vscode.window.showQuickPick<ProcessQuickPick>(
    tree.listAllProcesses().then(toQuickPickItems),
    {
      placeHolder: "Select a process to attach the debugger to",
      matchOnDescription: true,
      matchOnDetail: true,
    },
  );
  return selectedProcess?.processId.toString();
}

export function toQuickPickItems(processes: Process[]): ProcessQuickPick[] {
  return processes.map((proc) => ({
    processId: proc.id,
    label: path.basename(proc.command) || proc.id.toString(),
    description: proc.id.toString(),
    detail: proc.arguments,
  }));
}
