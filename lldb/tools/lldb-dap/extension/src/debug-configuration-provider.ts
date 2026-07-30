import * as child_process from "child_process";
import * as os from "os";
import * as path from "path";
import * as util from "util";
import * as vscode from "vscode";
import { pickProcess } from "./commands/pick-process";
import { convertToInteger } from "./commands/pid-helpers";
import { createDebugAdapterExecutable } from "./debug-adapter-factory";
import {
  getEnvironmentValue,
  supportsCliFlag,
} from "./compatibility-utils";
import { LLDBDapServer } from "./lldb-dap-server";
import { LogFilePathProvider } from "./logging";
import { ErrorWithNotification } from "./ui/error-with-notification";
import { ConfigureButton } from "./ui/show-error-message";

const exec = util.promisify(child_process.execFile);
const PROBE_TIMEOUT_MS = 2000;

interface PythonProbeResult {
  status: "ok" | "failed" | "timeout" | "error";
  detail?: string;
}

function buildWindowsPythonRuntimeSearchHint(
  env: { [key: string]: string } | undefined,
): string {
  const lldbPythonLibrary =
    getEnvironmentValue(env, "LLDB_PYTHON_LIBRARY") ?? "<unset>";
  const pythonHome = getEnvironmentValue(env, "PYTHONHOME") ?? "<unset>";
  const pathValue = getEnvironmentValue(env, "PATH") ?? "";
  const pathEntries = pathValue
    .split(path.delimiter)
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 0);
  const sampleEntries = pathEntries.slice(0, 3);
  const remainingEntries = Math.max(0, pathEntries.length - sampleEntries.length);

  const pathSummary =
    sampleEntries.length > 0
      ? `${sampleEntries.join(path.delimiter)}${
          remainingEntries > 0 ? `${path.delimiter}... (+${remainingEntries} more)` : ""
        }`
      : "<empty>";

  return (
    "Windows Python runtime search hint: " +
    `LLDB_PYTHON_LIBRARY=${lldbPythonLibrary}; ` +
    `PYTHONHOME=${pythonHome}; ` +
    `PATH(sample)=${pathSummary}`
  );
}

/**
 * Fetches the `--help` output of the given lldb-dap executable, used to
 * detect which optional CLI flags (e.g. `--connection`, `--check-python`)
 * this build supports.
 *
 * @param exe the path to the lldb-dap executable
 * @returns the help text, or undefined if the probe failed or timed out
 */
async function getHelpOutput(exe: string): Promise<string | undefined> {
  try {
    const { stdout } = await exec(exe, ["--help"], {
      timeout: PROBE_TIMEOUT_MS,
    });
    return stdout;
  } catch {
    return undefined;
  }
}

async function runPythonRuntimeProbe(
  executable: vscode.DebugAdapterExecutable,
): Promise<PythonProbeResult> {
  return new Promise((resolve) => {
    const child = child_process.spawn(executable.command, ["--check-python"], {
      cwd: executable.options?.cwd,
      env: executable.options?.env,
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
    });

    let resolved = false;
    let timedOut = false;
    let stdout = "";
    let stderr = "";

    const finish = (result: PythonProbeResult) => {
      if (resolved) {
        return;
      }
      resolved = true;
      clearTimeout(timer);
      resolve(result);
    };

    child.stdout?.on("data", (chunk: Buffer | string) => {
      stdout += chunk.toString();
    });
    child.stderr?.on("data", (chunk: Buffer | string) => {
      stderr += chunk.toString();
    });

    child.on("error", (error) => {
      finish({ status: "error", detail: error.message });
    });

    child.on("close", (code) => {
      if (timedOut) {
        return;
      }
      const detail = stderr || stdout;
      if (code === 0) {
        finish({ status: "ok", detail });
      } else {
        finish({ status: "failed", detail });
      }
    });

    const timer = setTimeout(() => {
      timedOut = true;
      try {
        child.kill();
      } catch {
        // Ignore kill errors and continue.
      }
      finish({ status: "timeout", detail: stderr || stdout });
    }, PROBE_TIMEOUT_MS);
  });
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
  platformUrl: { type: "string", default: "" },
  targetTriple: { type: "string", default: "" },

  // Keys for debugger command hooks.
  initCommands: { type: "stringArray", default: [] },
  preRunCommands: { type: "stringArray", default: [] },
  postRunCommands: { type: "stringArray", default: [] },
  stopCommands: { type: "stringArray", default: [] },
  exitCommands: { type: "stringArray", default: [] },
  terminateCommands: { type: "stringArray", default: [] },
};

export function getDefaultConfigKey(
  key: string,
): string | number | boolean | string[] | undefined {
  return configurations[key]?.default;
}

export class LLDBDapConfigurationProvider
  implements vscode.DebugConfigurationProvider
{
  /** Tracks Python pre-check warnings already shown, per adapter and reason. */
  private readonly shownPythonCheckWarnings = new Set<string>();

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

  /**
   * Informs the user that the Python runtime pre-check was skipped and why.
   *
   * The explanation is shown as a non-blocking warning notification at most
   * once per adapter executable and reason, to avoid nagging on every
   * launch. The full runtime search hint is always written to the logs.
   *
   * This is only used for the "error" and "timeout" reasons: a supported
   * check that misbehaves is genuinely anomalous. A build that simply lacks
   * `--check-python` (the "unsupported" reason) is a normal, healthy older
   * adapter and is logged only, via {@link logPythonCheckUnsupported}, to
   * avoid nagging every user of an older lldb-dap on every launch.
   */
  private warnPythonCheckSkipped(
    executablePath: string,
    reason: "error" | "timeout",
    explanation: string,
    runtimeSearchHint: string,
  ): void {
    this.logger.warn(`${explanation}\n${runtimeSearchHint}`);

    const dedupeKey = `${executablePath}:${reason}`;
    if (this.shownPythonCheckWarnings.has(dedupeKey)) {
      return;
    }
    this.shownPythonCheckWarnings.add(dedupeKey);
    // Fire-and-forget: the notification must not block the launch.
    vscode.window
      .showWarningMessage(explanation, "Show Logs")
      .then((selection) => {
        if (selection === "Show Logs") {
          this.logger.show();
        }
      });
  }

  /** Logs (without a visible notification) that --check-python is unsupported. */
  private logPythonCheckUnsupported(
    explanation: string,
    runtimeSearchHint: string,
  ): void {
    this.logger.info(`${explanation}\n${runtimeSearchHint}`);
  }

  async resolveDebugConfiguration(
    folder: vscode.WorkspaceFolder | undefined,
    debugConfiguration: vscode.DebugConfiguration,
    token?: vscode.CancellationToken,
  ): Promise<vscode.DebugConfiguration | null | undefined> {
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

    // If the user asked for the process picker, run it here — while we still
    // have the workspace folder and platform fields — rather than deferring
    // to VS Code's variable substitution, which doesn't pass the
    // configuration to the command handler.
    if (debugConfiguration.pid === "${command:pickProcess}") {
      const pid = await pickProcess(
        this.logger,
        this.logFilePath,
        folder,
        debugConfiguration,
      );
      if (pid === undefined) {
        // User cancelled, or the picker surfaced its own error.
        return null;
      }
      debugConfiguration.pid = pid;
    }

    return debugConfiguration;
  }

  async resolveDebugConfigurationWithSubstitutedVariables(
    folder: vscode.WorkspaceFolder | undefined,
    debugConfiguration: vscode.DebugConfiguration,
    _token?: vscode.CancellationToken,
  ): Promise<vscode.DebugConfiguration | null | undefined> {
    try {
      // Convert "pid" to a number if it came in as a string (e.g. via the
      // ${command:pickProcess} variable substitution).
      if ("pid" in debugConfiguration) {
        const pid = convertToInteger(debugConfiguration.pid);
        if (pid === undefined) {
          throw new ErrorWithNotification(
            "Invalid debug configuration: property 'pid' must either be an integer or a string containing an integer value.",
            new ConfigureButton(),
          );
        }
        debugConfiguration.pid = pid;
      }

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

        // Probe --help at most once per resolution; the output determines
        // which optional CLI flags this lldb-dap build supports.
        let helpOutputPromise: Promise<string | undefined> | undefined;
        const getCachedHelpOutput = () =>
          (helpOutputPromise ??= getHelpOutput(executable.command));

        if (os.platform() === "win32") {
          const runtimeSearchHint = buildWindowsPythonRuntimeSearchHint(
            executable.options?.env,
          );
          const pythonCheckSupported = supportsCliFlag(
            (await getCachedHelpOutput()) ?? "",
            "--check-python",
          );
          if (pythonCheckSupported) {
            const result = await runPythonRuntimeProbe(executable);
            if (result.status === "error") {
              this.warnPythonCheckSkipped(
                executable.command,
                "error",
                "Skipped the Python runtime check: running " +
                  `"lldb-dap --check-python" failed (${result.detail ?? "unknown error"}). ` +
                  "Debugging will continue, but if lldb-dap fails to start " +
                  "or Python scripting is unavailable, verify your Python " +
                  "installation.",
                runtimeSearchHint,
              );
            } else if (result.status === "timeout") {
              this.warnPythonCheckSkipped(
                executable.command,
                "timeout",
                "Skipped the Python runtime check: " +
                  `"lldb-dap --check-python" did not finish within ` +
                  `${PROBE_TIMEOUT_MS / 1000} seconds. Debugging will ` +
                  "continue without verifying the Python runtime.",
                runtimeSearchHint,
              );
            } else if (result.status === "failed") {
              const failureDetail = (result.detail ?? "").trim();
              throw new ErrorWithNotification(
                "LLDB-DAP reported an unusable Python runtime while running --check-python." +
                  (failureDetail.length > 0
                    ? `\n\n${failureDetail}`
                    : "") +
                  `\n\n${runtimeSearchHint}`,
                new ConfigureButton(),
              );
            }
          } else {
            this.logPythonCheckUnsupported(
              "Skipped the Python runtime check: this lldb-dap does not " +
                'support "--check-python". Debugging will continue, but the ' +
                "extension cannot verify your Python runtime before launch. " +
                "If the debugger fails to start, ensure the Python version " +
                "LLDB was built against is installed and on PATH.",
              runtimeSearchHint,
            );
          }
        }

        // Server mode needs to be handled here since DebugAdapterDescriptorFactory
        // will show an unhelpful error if it returns undefined. We'd rather show a
        // nicer error message here and allow stopping the debug session gracefully.
        const config = vscode.workspace.getConfiguration("lldb-dap", folder);
        // Match only the standalone --connection flag and avoid matching
        // related options such as --connection-timeout.
        if (
          config.get<boolean>("serverMode", false) &&
          supportsCliFlag((await getCachedHelpOutput()) ?? "", "--connection")
        ) {
          const connectionTimeoutSeconds = config.get<number | undefined>(
            "connectionTimeout",
            undefined,
          );
          const serverInfo = await this.server.start(
            executable.command,
            executable.args,
            executable.options,
            connectionTimeoutSeconds,
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
