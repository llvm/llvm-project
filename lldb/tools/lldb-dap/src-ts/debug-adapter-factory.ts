import * as path from "path";
import * as util from "util";
import * as vscode from "vscode";
import * as child_process from "child_process";
import * as fs from "node:fs/promises";
import { ConfigureButton, OpenSettingsButton } from "./ui/show-error-message";
import { ErrorWithNotification } from "./ui/error-with-notification";
import { LogFilePathProvider, LogType } from "./logging";

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

/**
 * Retrieves the lldb-dap executable path either from settings or the provided
 * {@link vscode.DebugConfiguration}.
 *
 * @param workspaceFolder The {@link vscode.WorkspaceFolder} that the debug session will be launched within
 * @param configuration The {@link vscode.DebugConfiguration} that will be launched
 * @throws An {@link ErrorWithNotification} if something went wrong
 * @returns The path to the lldb-dap executable
 */
async function getDAPExecutable(
  workspaceFolder: vscode.WorkspaceFolder | undefined,
  configuration: vscode.DebugConfiguration,
): Promise<string> {
  // Check if the executable was provided in the launch configuration.
  const launchConfigPath = configuration["debugAdapterExecutable"];
  if (typeof launchConfigPath === "string" && launchConfigPath.length !== 0) {
    if (!(await isExecutable(launchConfigPath))) {
      throw new ErrorWithNotification(
        `Debug adapter path "${launchConfigPath}" is not a valid file. The path comes from your launch configuration.`,
        new ConfigureButton(),
      );
    }
    return launchConfigPath;
  }

  // Check if the executable was provided in the extension's configuration.
  const config = vscode.workspace.getConfiguration("lldb-dap", workspaceFolder);
  const configPath = config.get<string>("executable-path");
  if (configPath && configPath.length !== 0) {
    if (!(await isExecutable(configPath))) {
      throw new ErrorWithNotification(
        `Debug adapter path "${configPath}" is not a valid file. The path comes from your settings.`,
        new OpenSettingsButton("lldb-dap.executable-path"),
      );
    }
    return configPath;
  }

  // Try finding the lldb-dap binary.
  const foundPath = await findDAPExecutable();
  if (foundPath) {
    if (!(await isExecutable(foundPath))) {
      throw new ErrorWithNotification(
        `Found a potential debug adapter on your system at "${configPath}", but it is not a valid file.`,
        new OpenSettingsButton("lldb-dap.executable-path"),
      );
    }
    return foundPath;
  }

  throw new ErrorWithNotification(
    "Unable to find the path to the LLDB debug adapter executable.",
    new OpenSettingsButton("lldb-dap.executable-path"),
  );
}

/**
 * Retrieves the arguments that will be provided to lldb-dap either from settings or the provided
 * {@link vscode.DebugConfiguration}.
 *
 * @param workspaceFolder The {@link vscode.WorkspaceFolder} that the debug session will be launched within
 * @param configuration The {@link vscode.DebugConfiguration} that will be launched
 * @throws An {@link ErrorWithNotification} if something went wrong
 * @returns The arguments that will be provided to lldb-dap
 */
async function getDAPArguments(
  workspaceFolder: vscode.WorkspaceFolder | undefined,
  configuration: vscode.DebugConfiguration,
): Promise<string[]> {
  // Check the debug configuration for arguments first.
  const debugConfigArgs = configuration.debugAdapterArgs;
  if (debugConfigArgs) {
    if (
      !Array.isArray(debugConfigArgs) ||
      debugConfigArgs.findIndex((entry) => typeof entry !== "string") !== -1
    ) {
      throw new ErrorWithNotification(
        "The debugAdapterArgs property must be an array of string values. Please update your launch configuration",
        new ConfigureButton(),
      );
    }
    return debugConfigArgs;
  }
  // Fall back on the workspace configuration.
  return vscode.workspace
    .getConfiguration("lldb-dap", workspaceFolder)
    .get<string[]>("arguments", []);
}

/**
 * Creates a new {@link vscode.DebugAdapterExecutable} based on the provided workspace folder and
 * debug configuration. Assumes that the given debug configuration is for a local launch of lldb-dap.
 *
 * @param logger The {@link vscode.LogOutputChannel} to log setup diagnostics
 * @param logFilePath The {@link LogFilePathProvider} for determining where to put session logs
 * @param workspaceFolder The {@link vscode.WorkspaceFolder} that the debug session will be launched within
 * @param configuration The {@link vscode.DebugConfiguration} that will be launched
 * @throws An {@link ErrorWithNotification} if something went wrong
 * @returns The {@link vscode.DebugAdapterExecutable} that can be used to launch lldb-dap
 */
export async function createDebugAdapterExecutable(
  logger: vscode.LogOutputChannel,
  logFilePath: LogFilePathProvider,
  workspaceFolder: vscode.WorkspaceFolder | undefined,
  configuration: vscode.DebugConfiguration,
): Promise<vscode.DebugAdapterExecutable> {
  const config = vscode.workspace.workspaceFile
    ? vscode.workspace.getConfiguration("lldb-dap")
    : vscode.workspace.getConfiguration("lldb-dap", workspaceFolder);
  const log_path = config.get<string>("log-path");
  let env: { [key: string]: string } = {};
  if (log_path) {
    env["LLDBDAP_LOG"] = log_path;
  } else if (
    vscode.workspace.getConfiguration("lldb-dap").get("captureSessionLogs", false)
  ) {
    env["LLDBDAP_LOG"] = logFilePath.get(LogType.DEBUG_SESSION);
  }
  const configEnvironment =
    config.get<{ [key: string]: string }>("environment") || {};
  const dapPath = await getDAPExecutable(workspaceFolder, configuration);

  const dbgOptions = {
    env: {
      ...configEnvironment,
      ...env,
    },
    cwd: configuration.cwd ?? workspaceFolder?.uri.fsPath,
  };
  const dbgArgs = await getDAPArguments(workspaceFolder, configuration);

  logger.info(`lldb-dap path: ${dapPath}`);
  logger.info(`lldb-dap args: ${dbgArgs}`);
  logger.info(`cwd: ${dbgOptions.cwd}`);
  logger.info(`env: ${JSON.stringify(dbgOptions.env)}`);

  return new vscode.DebugAdapterExecutable(dapPath, dbgArgs, dbgOptions);
}

/**
 * This class defines a factory used to find the lldb-dap binary to use
 * depending on the session configuration.
 */
export class LLDBDapDescriptorFactory
  implements vscode.DebugAdapterDescriptorFactory
{
  constructor(
    private readonly logger: vscode.LogOutputChannel,
    private logFilePath: LogFilePathProvider,
  ) {
    vscode.commands.registerCommand(
      "lldb-dap.createDebugAdapterDescriptor",
      (
        session: vscode.DebugSession,
        executable: vscode.DebugAdapterExecutable | undefined,
      ) => this.createDebugAdapterDescriptor(session, executable),
    );
  }

  async createDebugAdapterDescriptor(
    session: vscode.DebugSession,
    executable: vscode.DebugAdapterExecutable | undefined,
  ): Promise<vscode.DebugAdapterDescriptor | undefined> {
    this.logger.info(`Creating debug adapter for session "${session.name}"`);
    this.logger.info(
      `Session "${session.name}" debug configuration:\n` +
        JSON.stringify(session.configuration, undefined, 2),
    );
    if (executable) {
      const error = new Error(
        "Setting the debug adapter executable in the package.json is not supported.",
      );
      this.logger.error(error);
      throw error;
    }

    // Use a server connection if the debugAdapterPort is provided
    if (session.configuration.debugAdapterPort) {
      this.logger.info(
        `Spawning debug adapter server on port ${session.configuration.debugAdapterPort}`,
      );
      return new vscode.DebugAdapterServer(
        session.configuration.debugAdapterPort,
        session.configuration.debugAdapterHostname,
      );
    }

    return createDebugAdapterExecutable(
      this.logger,
      this.logFilePath,
      session.workspaceFolder,
      session.configuration,
    );
  }
}
