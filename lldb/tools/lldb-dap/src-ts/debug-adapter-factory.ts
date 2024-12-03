import * as path from "path";
import * as util from "util";
import * as vscode from "vscode";

import { LLDBDapOptions } from "./types";

/**
 * This class defines a factory used to find the lldb-dap binary to use
 * depending on the session configuration.
 */
export class LLDBDapDescriptorFactory
  implements vscode.DebugAdapterDescriptorFactory
{
  static async isValidFile(pathUri: vscode.Uri): Promise<Boolean> {
    try {
      const fileStats = await vscode.workspace.fs.stat(pathUri);
      if (!(fileStats.type & vscode.FileType.File)) {
        return false;
      }
    } catch (err) {
      return false;
    }
    return true;
  }

  static async findDAPExecutable(): Promise<string | undefined> {
    let executable = "lldb-dap";
    if (process.platform === "win32") {
      executable = "lldb-dap.exe";
    }

    // Prefer lldb-dap from Xcode on Darwin.
    if (process.platform === "darwin") {
      try {
        const exec = util.promisify(require("child_process").execFile);
        let { stdout, stderr } = await exec("/usr/bin/xcrun", [
          "-find",
          executable,
        ]);
        if (stdout) {
          return stdout.toString().trimEnd();
        }
      } catch (error) {}
    }

    // Find lldb-dap in the user's path.
    let env_path =
      process.env["PATH"] ||
      (process.platform === "win32" ? process.env["Path"] : null);
    if (!env_path) {
      return undefined;
    }

    const paths = env_path.split(path.delimiter);
    for (const p of paths) {
      const exe_path = path.join(p, executable);
      if (
        await LLDBDapDescriptorFactory.isValidFile(vscode.Uri.file(exe_path))
      ) {
        return exe_path;
      }
    }

    return undefined;
  }

  static async getDAPExecutable(
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
    const foundPath = await LLDBDapDescriptorFactory.findDAPExecutable();
    if (foundPath) {
      return foundPath;
    }

    return undefined;
  }

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
    const dapPath = await LLDBDapDescriptorFactory.getDAPExecutable(session);
    const dbgOptions = {
      env: {
        ...executable?.options?.env,
        ...configEnvironment,
        ...env,
      },
    };
    if (dapPath) {
      const fileUri = vscode.Uri.file(dapPath);
      if (!(await LLDBDapDescriptorFactory.isValidFile(fileUri))) {
        LLDBDapDescriptorFactory.showLLDBDapNotFoundMessage(fileUri.path);
        return undefined;
      }
      return new vscode.DebugAdapterExecutable(dapPath, [], dbgOptions);
    } else if (executable) {
      const fileUri = vscode.Uri.file(executable.command);
      if (!(await LLDBDapDescriptorFactory.isValidFile(fileUri))) {
        LLDBDapDescriptorFactory.showLLDBDapNotFoundMessage(fileUri.path);
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
