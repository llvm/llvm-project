import * as path from "path";
import * as util from "util";
import * as vscode from "vscode";
import * as child_process from "child_process";
import * as fs from "node:fs/promises";
import {
  showErrorWithConfigureButton,
  showLLDBDapNotFoundMessage,
} from "./ui/error-messages";

const exec = util.promisify(child_process.execFile);

async function isExecutable(path: string): Promise<Boolean> {
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
  const xcrun_dap = await findWithXcrun(executable);
  if (xcrun_dap) {
    return xcrun_dap;
  }

  // Find lldb-dap in the user's path.
  const path_dap = await findInPath(executable);
  if (path_dap) {
    return path_dap;
  }

  return undefined;
}

async function getDAPExecutable(
  folder: vscode.WorkspaceFolder | undefined,
  configuration: vscode.DebugConfiguration,
): Promise<string | undefined> {
  // Check if the executable was provided in the launch configuration.
  const launchConfigPath = configuration["debugAdapterExecutable"];
  if (typeof launchConfigPath === "string" && launchConfigPath.length !== 0) {
    return launchConfigPath;
  }

  // Check if the executable was provided in the extension's configuration.
  const config = vscode.workspace.getConfiguration("lldb-dap", folder);
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

async function getDAPArguments(
  folder: vscode.WorkspaceFolder | undefined,
  configuration: vscode.DebugConfiguration,
  userInteractive?: boolean,
): Promise<string[] | null | undefined> {
  // Check the debug configuration for arguments first
  const debugConfigArgs = configuration.debugAdapterArgs;
  if (debugConfigArgs) {
    if (
      !Array.isArray(debugConfigArgs) ||
      debugConfigArgs.findIndex((entry) => typeof entry !== "string") !== -1
    ) {
      if (!userInteractive) {
        return undefined;
      }
      return showErrorWithConfigureButton(
        "The debugAdapterArgs property must be an array of string values.",
      );
    }
    return debugConfigArgs;
  }
  if (
    Array.isArray(debugConfigArgs) &&
    debugConfigArgs.findIndex((entry) => typeof entry !== "string") === -1
  ) {
    return debugConfigArgs;
  }
  // Fall back on the workspace configuration
  return vscode.workspace
    .getConfiguration("lldb-dap", folder)
    .get<string[]>("arguments", []);
}

/**
 * Creates a new {@link vscode.DebugAdapterExecutable} based on the provided workspace folder and
 * debug configuration. Assumes that the given debug configuration is for a local launch of lldb-dap.
 *
 * @param folder The {@link vscode.WorkspaceFolder} that the debug session will be launched within
 * @param configuration The {@link vscode.DebugConfiguration}
 * @param userInteractive Whether or not this was called due to user interaction (determines if modals should be shown)
 * @returns
 */
export async function createDebugAdapterExecutable(
  folder: vscode.WorkspaceFolder | undefined,
  configuration: vscode.DebugConfiguration,
  userInteractive?: boolean,
): Promise<vscode.DebugAdapterExecutable | undefined> {
  const config = vscode.workspace.getConfiguration("lldb-dap", folder);
  const log_path = config.get<string>("log-path");
  let env: { [key: string]: string } = {};
  if (log_path) {
    env["LLDBDAP_LOG"] = log_path;
  }
  const configEnvironment =
    config.get<{ [key: string]: string }>("environment") || {};
  const dapPath = await getDAPExecutable(folder, configuration);

  if (!dapPath) {
    if (userInteractive) {
      showLLDBDapNotFoundMessage();
    }
    return undefined;
  }

  if (!(await isExecutable(dapPath))) {
    if (userInteractive) {
      showLLDBDapNotFoundMessage(dapPath);
    }
    return undefined;
  }

  const dbgOptions = {
    env: {
      ...configEnvironment,
      ...env,
    },
  };
  const dbgArgs = await getDAPArguments(folder, configuration, userInteractive);
  if (!dbgArgs) {
    return undefined;
  }

  return new vscode.DebugAdapterExecutable(dapPath, dbgArgs, dbgOptions);
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
    if (executable) {
      throw new Error(
        "Setting the debug adapter executable in the package.json is not supported.",
      );
    }

    // Use a server connection if the debugAdapterPort is provided
    if (session.configuration.debugAdapterPort) {
      return new vscode.DebugAdapterServer(
        session.configuration.debugAdapterPort,
        session.configuration.debugAdapterHost,
      );
    }

    return createDebugAdapterExecutable(
      session.workspaceFolder,
      session.configuration,
    );
  }
}
