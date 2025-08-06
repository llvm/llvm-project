import * as vscode from "vscode";
import * as child_process from "child_process";
import * as util from "util";
import { LLDBDapServer } from "./lldb-dap-server";
import { createDebugAdapterExecutable } from "./debug-adapter-factory";
import { ConfigureButton, showErrorMessage } from "./ui/show-error-message";
import { ErrorWithNotification } from "./ui/error-with-notification";
import { LogFilePathProvider } from "./logging";

const exec = util.promisify(child_process.execFile);

/**
 * Determines whether or not the given lldb-dap executable supports executing
 * in server mode.
 *
 * @param exe the path to the lldb-dap executable
 * @returns a boolean indicating whether or not lldb-dap supports server mode
 */
async function isServerModeSupported(exe: string): Promise<boolean> {
  const { stdout } = await exec(exe, ["--help"]);
  return /--connection/.test(stdout);
}

interface BoolConfig {
  type: "boolean";
  default: boolean;
}
interface StringConfig {
  type: "string";
  default: string;
}
interface NumberConfig {
  type: "number";
  default: number;
}
interface StringArrayConfig {
  type: "stringArray";
  default: string[];
}
type DefaultConfig =
  | BoolConfig
  | NumberConfig
  | StringConfig
  | StringArrayConfig;

const configurations: Record<string, DefaultConfig> = {
  // Keys for debugger configurations.
  commandEscapePrefix: { type: "string", default: "`" },
  customFrameFormat: { type: "string", default: "" },
  customThreadFormat: { type: "string", default: "" },
  detachOnError: { type: "boolean", default: false },
  disableASLR: { type: "boolean", default: true },
  disableSTDIO: { type: "boolean", default: false },
  displayExtendedBacktrace: { type: "boolean", default: false },
  enableAutoVariableSummaries: { type: "boolean", default: false },
  enableSyntheticChildDebugging: { type: "boolean", default: false },
  timeout: { type: "number", default: 30 },

  // Keys for platform / target configuration.
  platformName: { type: "string", default: "" },
  targetTriple: { type: "string", default: "" },

  // Keys for debugger command hooks.
  initCommands: { type: "stringArray", default: [] },
  preRunCommands: { type: "stringArray", default: [] },
  postRunCommands: { type: "stringArray", default: [] },
  stopCommands: { type: "stringArray", default: [] },
  exitCommands: { type: "stringArray", default: [] },
  terminateCommands: { type: "stringArray", default: [] },
};

export class LLDBDapConfigurationProvider
  implements vscode.DebugConfigurationProvider
{
  constructor(
    private readonly server: LLDBDapServer,
    private readonly logger: vscode.LogOutputChannel,
    private readonly logFilePath: LogFilePathProvider,
  ) {
    vscode.commands.registerCommand(
      "lldb-dap.resolveDebugConfiguration",
      (
        folder: vscode.WorkspaceFolder | undefined,
        debugConfiguration: vscode.DebugConfiguration,
        token?: vscode.CancellationToken,
      ) => this.resolveDebugConfiguration(folder, debugConfiguration, token),
    );
    vscode.commands.registerCommand(
      "lldb-dap.resolveDebugConfigurationWithSubstitutedVariables",
      (
        folder: vscode.WorkspaceFolder | undefined,
        debugConfiguration: vscode.DebugConfiguration,
        token?: vscode.CancellationToken,
      ) =>
        this.resolveDebugConfigurationWithSubstitutedVariables(
          folder,
          debugConfiguration,
          token,
        ),
    );
  }

  async resolveDebugConfiguration(
    folder: vscode.WorkspaceFolder | undefined,
    debugConfiguration: vscode.DebugConfiguration,
    token?: vscode.CancellationToken,
  ): Promise<vscode.DebugConfiguration> {
    this.logger.info(
      `Resolving debug configuration for "${debugConfiguration.name}"`,
    );
    this.logger.debug(
      "Initial debug configuration:\n" +
        JSON.stringify(debugConfiguration, undefined, 2),
    );
    let config = vscode.workspace.getConfiguration("lldb-dap");
    for (const [key, cfg] of Object.entries(configurations)) {
      if (Reflect.has(debugConfiguration, key)) {
        continue;
      }
      const value = config.get(key);
      if (value === undefined || value === cfg.default) {
        continue;
      }
      switch (cfg.type) {
        case "string":
          if (typeof value !== "string") {
            throw new Error(`Expected ${key} to be a string, got ${value}`);
          }
          break;
        case "number":
          if (typeof value !== "number") {
            throw new Error(`Expected ${key} to be a number, got ${value}`);
          }
          break;
        case "boolean":
          if (typeof value !== "boolean") {
            throw new Error(`Expected ${key} to be a boolean, got ${value}`);
          }
          break;
        case "stringArray":
          if (typeof value !== "object" && Array.isArray(value)) {
            throw new Error(
              `Expected ${key} to be a array of strings, got ${value}`,
            );
          }
          if ((value as string[]).length === 0) {
            continue;
          }
          break;
      }

      debugConfiguration[key] = value;
    }

    return debugConfiguration;
  }

  async resolveDebugConfigurationWithSubstitutedVariables(
    folder: vscode.WorkspaceFolder | undefined,
    debugConfiguration: vscode.DebugConfiguration,
    _token?: vscode.CancellationToken,
  ): Promise<vscode.DebugConfiguration | null | undefined> {
    try {
      if (
        "debugAdapterHostname" in debugConfiguration &&
        !("debugAdapterPort" in debugConfiguration)
      ) {
        throw new ErrorWithNotification(
          "A debugAdapterPort must be provided when debugAdapterHostname is set. Please update your launch configuration.",
          new ConfigureButton(),
        );
      }

      // Check if we're going to launch a debug session or use an existing process
      if ("debugAdapterPort" in debugConfiguration) {
        if (
          "debugAdapterExecutable" in debugConfiguration ||
          "debugAdapterArgs" in debugConfiguration
        ) {
          throw new ErrorWithNotification(
            "The debugAdapterPort property is incompatible with debugAdapterExecutable and debugAdapterArgs. Please update your launch configuration.",
            new ConfigureButton(),
          );
        }
      } else {
        // Always try to create the debug adapter executable as this will show the user errors
        // if there are any.
        const executable = await createDebugAdapterExecutable(
          this.logger,
          this.logFilePath,
          folder,
          debugConfiguration,
        );
        if (!executable) {
          return undefined;
        }

        // Server mode needs to be handled here since DebugAdapterDescriptorFactory
        // will show an unhelpful error if it returns undefined. We'd rather show a
        // nicer error message here and allow stopping the debug session gracefully.
        const config = vscode.workspace.getConfiguration("lldb-dap", folder);
        if (
          config.get<boolean>("serverMode", false) &&
          (await isServerModeSupported(executable.command))
        ) {
          const serverInfo = await this.server.start(
            executable.command,
            executable.args,
            executable.options,
          );
          if (!serverInfo) {
            return undefined;
          }
          // Use a debug adapter host and port combination rather than an executable
          // and list of arguments.
          delete debugConfiguration.debugAdapterExecutable;
          delete debugConfiguration.debugAdapterArgs;
          debugConfiguration.debugAdapterHostname = serverInfo.host;
          debugConfiguration.debugAdapterPort = serverInfo.port;
        }
      }

      this.logger.info(
        "Resolved debug configuration:\n" +
          JSON.stringify(debugConfiguration, undefined, 2),
      );

      return debugConfiguration;
    } catch (error) {
      this.logger.error(error as Error);
      // Show a better error message to the user if possible
      if (!(error instanceof ErrorWithNotification)) {
        throw error;
      }
      return await error.showNotification({
        modal: true,
        showConfigureButton: true,
      });
    }
  }
}
