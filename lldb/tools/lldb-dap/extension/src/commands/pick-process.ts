import { execFile } from "node:child_process";
import * as path from "node:path";
import { promisify } from "node:util";
import * as vscode from "vscode";

import { createDebugAdapterExecutable } from "../debug-adapter-factory";
import { LogFilePathProvider } from "../logging";
import { LldbDapProcessTree, Process } from "../process-tree";

const exec = promisify(execFile);

/**
 * Cache of which lldb-dap binaries have been observed to support the
 * `--list-processes` flag, keyed by executable path. We probe once per path,
 * but only remember successful probes — a transient failure (binary still
 * building, EBUSY, etc.) should not poison the cache for the rest of the
 * session.
 */
const listProcessesSupportCache = new Map<string, Promise<boolean>>();

/** Probes `<exe> --help` to see whether `--list-processes` is advertised. */
function isListProcessesSupported(exe: string): Promise<boolean> {
  const cached = listProcessesSupportCache.get(exe);
  if (cached) {
    return cached;
  }
  const probe = exec(exe, ["--help"]).then(({ stdout }) =>
    /--list-processes/.test(stdout),
  );
  listProcessesSupportCache.set(exe, probe);
  // Drop failed probes so the next attempt can retry.
  probe.catch(() => {
    if (listProcessesSupportCache.get(exe) === probe) {
      listProcessesSupportCache.delete(exe);
    }
  });
  return probe.catch(() => false);
}

interface ProcessQuickPick extends vscode.QuickPickItem {
  processId: number;
}

/**
 * Prompts the user to select a running process, enumerated by `lldb-dap
 * --list-processes`. When invoked from `resolveDebugConfiguration` the caller
 * forwards the in-flight configuration so we can target the right lldb-dap
 * binary and platform (for remote attach).
 *
 * @returns the pid of the selected process, or `undefined` if the user
 *   cancelled or the picker failed (in which case an error has already been
 *   shown).
 */
export async function pickProcess(
  logger: vscode.LogOutputChannel,
  logFilePath: LogFilePathProvider,
  folder?: vscode.WorkspaceFolder,
  debugConfiguration?: vscode.DebugConfiguration,
): Promise<number | undefined> {
  const executable = await createDebugAdapterExecutable(
    logger,
    logFilePath,
    folder,
    debugConfiguration ?? {
      type: "lldb-dap",
      request: "attach",
      name: "Attach",
    },
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
    platformName: debugConfiguration?.platformName,
    platformUrl: debugConfiguration?.platformUrl,
  });

  let processes: Process[];
  try {
    processes = await tree.listAllProcesses();
  } catch (error) {
    logger.error(error as Error);
    await vscode.window.showErrorMessage(
      "Failed to list processes from lldb-dap.",
      { modal: true, detail: describeExecError(error) },
    );
    return undefined;
  }

  const selectedProcess = await vscode.window.showQuickPick<ProcessQuickPick>(
    toQuickPickItems(processes),
    {
      placeHolder: "Select a process to attach the debugger to",
      matchOnDescription: true,
      matchOnDetail: true,
    },
  );
  return selectedProcess?.processId;
}

/**
 * Extracts a user-visible message from an error raised by `execFile`.
 * Node attaches `stderr` to the rejection; prefer that over the generic
 * "Command failed" message so e.g. a bad --platform-url surfaces instead of
 * silently failing.
 */
function describeExecError(error: unknown): string {
  const stderr = (error as { stderr?: unknown } | undefined)?.stderr;
  if (typeof stderr === "string" && stderr.trim() !== "") {
    return stderr.trim();
  }
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

export function toQuickPickItems(processes: Process[]): ProcessQuickPick[] {
  return processes.map((proc) => ({
    processId: proc.id,
    label: path.basename(proc.command) || proc.id.toString(),
    description: proc.id.toString(),
    detail: `${proc.command} ${proc.arguments}`,
  }));
}
