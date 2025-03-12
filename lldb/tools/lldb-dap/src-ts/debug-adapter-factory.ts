import * as path from "path";
import * as util from "util";
import * as vscode from "vscode";
import * as child_process from "child_process";
import * as fs from "node:fs/promises";

export async function isExecutable(path: string): Promise<Boolean> {
  try {
    await fs.access(path, fs.constants.X_OK);
  } catch {
    return false;
  }
  return true;
}

async function findWithXcrun(executable: string): Promise<string | undefined> {
  if (process.platform === "darwin") {
    try {
      const exec = util.promisify(child_process.execFile);
      let { stdout, stderr } = await exec("/usr/bin/xcrun", [
        "-find",
        executable,
      ]);
      if (stdout) {
        return stdout.toString().trimEnd();
      }
    } catch (error) {}
  }
  return undefined;
}

async function findInPath(executable: string): Promise<string | undefined> {
  const env_path =
    process.platform === "win32" ? process.env["Path"] : process.env["PATH"];
  if (!env_path) {
    return undefined;
  }

  const paths = env_path.split(path.delimiter);
  for (const p of paths) {
    const exe_path = path.join(p, executable);
    if (await isExecutable(exe_path)) {
      return exe_path;
    }
  }
  return undefined;
}

async function findDAPExecutable(): Promise<string | undefined> {
  const executable = process.platform === "win32" ? "lldb-dap.exe" : "lldb-dap";

  // Prefer lldb-dap from Xcode on Darwin.
  const xcrun_dap = findWithXcrun(executable);
  if (xcrun_dap) {
    return xcrun_dap;
  }

  // Find lldb-dap in the user's path.
  const path_dap = findInPath(executable);
  if (path_dap) {
    return path_dap;
  }

  return undefined;
}

async function getDAPExecutable(
  session: vscode.DebugSession,
): Promise<string | undefined> {
  const config = vscode.workspace.getConfiguration(
    "lldb-dap",
    session.workspaceFolder,
  );

  // Prefer the explicitly specified path in the extension's configuration.
  const configPath = config.get<string>("executable-path");
  if (configPath && configPath.length !== 0) {
    return configPath;
  }

  // Try finding the lldb-dap binary.
  const foundPath = await findDAPExecutable();
  if (foundPath) {
    return foundPath;
  }

  return undefined;
}

/**
 * This class defines a factory used to find the lldb-dap binary to use
 * depending on the session configuration.
 */
export class LLDBDapDescriptorFactory
  implements vscode.DebugAdapterDescriptorFactory
{
  async createDebugAdapterDescriptor(
    session: vscode.DebugSession,
    executable: vscode.DebugAdapterExecutable | undefined,
  ): Promise<vscode.DebugAdapterDescriptor | undefined> {
    const config = vscode.workspace.getConfiguration(
      "lldb-dap",
      session.workspaceFolder,
    );

    const log_path = config.get<string>("log-path");
    let env: { [key: string]: string } = {};
    if (log_path) {
      env["LLDBDAP_LOG"] = log_path;
    }
    const configEnvironment =
      config.get<{ [key: string]: string }>("environment") || {};
    const dapPath = await getDAPExecutable(session);
    const dbgOptions = {
      env: {
        ...executable?.options?.env,
        ...configEnvironment,
        ...env,
      },
    };
    if (dapPath) {
      if (!(await isExecutable(dapPath))) {
        LLDBDapDescriptorFactory.showLLDBDapNotFoundMessage(dapPath);
        return undefined;
      }
      return new vscode.DebugAdapterExecutable(dapPath, [], dbgOptions);
    } else if (executable) {
      if (!(await isExecutable(executable.command))) {
        LLDBDapDescriptorFactory.showLLDBDapNotFoundMessage(executable.command);
        return undefined;
      }
      return new vscode.DebugAdapterExecutable(
        executable.command,
        executable.args,
        dbgOptions,
      );
    }
    return undefined;
  }

  /**
   * Shows a message box when the debug adapter's path is not found
   */
  static async showLLDBDapNotFoundMessage(path: string) {
    const openSettingsAction = "Open Settings";
    const callbackValue = await vscode.window.showErrorMessage(
      `Debug adapter path: ${path} is not a valid file`,
      openSettingsAction,
    );

    if (openSettingsAction === callbackValue) {
      vscode.commands.executeCommand(
        "workbench.action.openSettings",
        "lldb-dap.executable-path",
      );
    }
  }
}
