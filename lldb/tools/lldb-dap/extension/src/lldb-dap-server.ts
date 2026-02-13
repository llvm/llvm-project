import { FSWatcher, watch as chokidarWatch } from "chokidar";
import * as child_process from "node:child_process";
import { isDeepStrictEqual } from "util";
import * as vscode from "vscode";

/**
 * The information needed to spawn a lldb-dap process: e.g. arguments, binary
 * path, etc.
 */
class ServerSpawnInfo {
  constructor(
    readonly path: string,
    readonly args: string[],
    readonly env: { [key: string]: string },
  ) {}

  static create(
    path: string,
    args: string[],
    env: NodeJS.ProcessEnv | { [key: string]: string } | undefined,
  ): ServerSpawnInfo {
    // Convert env to a plain object for later comparison. We filter out
    // `LLDBDAP_LOG` because we ignore it when considering reusing lldb-dap
    // server.
    const normalizedEnv: { [key: string]: string } = {};
    for (const [key, value] of Object.entries(env ?? {})) {
      if (key !== "LLDBDAP_LOG") {
        normalizedEnv[key] = String(value);
      }
    }
    return new ServerSpawnInfo(path, args, normalizedEnv);
  }

  /**
   * Compares this spawn info with another for equality.
   */
  equals(other: ServerSpawnInfo): boolean {
    return isDeepStrictEqual(
      [this.path, this.args, this.env],
      [other.path, other.args, other.env],
    );
  }

  /**
   * Returns a human-readable string representation of this spawn info.
   */
  toDisplay(): string {
    const cmd = [this.path, ...this.args];

    const env = Object.entries(this.env)
      .map(([k, v]) => `${k}=${v}`)
      // Sort to make it easier to compare. Sorting isn't necessary for the
      // `isDeepStrictEqual` check.
      .sort()
      // Joining with newlines makes it difficult to truncate by string length,
      // so we opt for space.
      .join(" ");

    let display = cmd.join(" ");
    if (env) {
      display += `\n${env}`;
    }

    return display;
  }

  /**
   * Returns a truncated representation of `toDisplay`.
   */
  toDisplayTruncated(maxLength: number = 300): string {
    const full = this.toDisplay();
    if (full.length <= maxLength) {
      return full;
    }
    return full.substring(0, maxLength) + "...";
  }
}

/**
 * Represents a running lldb-dap process that is accepting connections (i.e. in "server mode").
 *
 * Handles startup of the process if it isn't running already as well as prompting the user
 * to restart when arguments have changed.
 */
export class LLDBDapServer implements vscode.Disposable {
  private serverProcess?: child_process.ChildProcessWithoutNullStreams;
  private serverInfo?: Promise<{ host: string; port: number }>;
  private serverSpawnInfo?: ServerSpawnInfo;
  // Detects changes to the lldb-dap executable file since the server's startup.
  private serverFileWatcher?: FSWatcher;
  // Indicates whether the lldb-dap executable file has changed since the server's startup.
  private serverFileChanged?: boolean;

  constructor() {
    vscode.commands.registerCommand(
      "lldb-dap.getServerProcess",
      () => this.serverProcess,
    );
  }

  /**
   * Starts the server with the provided options. The server will be restarted or reused as
   * necessary.
   *
   * @param dapPath the path to the debug adapter executable
   * @param args the list of arguments to provide to the debug adapter
   * @param options the options to provide to the debug adapter process
   * @returns a promise that resolves with the host and port information or `undefined` if unable to launch the server.
   */
  async start(
    dapPath: string,
    args: string[],
    options?: child_process.SpawnOptionsWithoutStdio,
    connectionTimeoutSeconds?: number,
  ): Promise<{ host: string; port: number } | undefined> {
    // Both the --connection and --connection-timeout arguments are subject to the shouldContinueStartup() check.
    const connectionTimeoutArgs =
      connectionTimeoutSeconds && connectionTimeoutSeconds > 0
        ? ["--connection-timeout", `${connectionTimeoutSeconds}`]
        : [];
    const dapArgs = [
      ...args,
      "--connection",
      "listen://localhost:0",
      ...connectionTimeoutArgs,
    ];
    if (!(await this.shouldContinueStartup(dapPath, dapArgs, options?.env))) {
      return undefined;
    }

    if (this.serverInfo) {
      return this.serverInfo;
    }

    this.serverInfo = new Promise((resolve, reject) => {
      const process = child_process.spawn(dapPath, dapArgs, options);
      process.on("error", (error) => {
        reject(error);
        this.cleanUp(process);
      });
      process.on("exit", (code, signal) => {
        let errorMessage = "Server process exited early";
        if (code !== undefined) {
          errorMessage += ` with code ${code}`;
        } else if (signal !== undefined) {
          errorMessage += ` due to signal ${signal}`;
        }
        reject(new Error(errorMessage));
        this.cleanUp(process);
      });
      process.stdout.setEncoding("utf8").on("data", (data) => {
        const connection = /connection:\/\/\[([^\]]+)\]:(\d+)/.exec(
          data.toString(),
        );
        if (connection) {
          const host = connection[1];
          const port = Number(connection[2]);
          resolve({ host, port });
          process.stdout.removeAllListeners();
        }
      });
      this.serverProcess = process;
      this.serverSpawnInfo = ServerSpawnInfo.create(dapPath, dapArgs, options?.env);
      this.serverFileChanged = false;
      this.serverFileWatcher = chokidarWatch(dapPath);
      this.serverFileWatcher
        .on("change", () => (this.serverFileChanged = true))
        .on("unlink", () => (this.serverFileChanged = true));
    });
    return this.serverInfo;
  }

  /**
   * Checks to see if the server needs to be restarted. If so, it will prompt the user
   * to ask if they wish to restart.
   *
   * @param dapPath the path to the debug adapter
   * @param args the arguments for the debug adapter
   * @returns whether or not startup should continue depending on user input
   */
  private async shouldContinueStartup(
    dapPath: string,
    args: string[],
    env: NodeJS.ProcessEnv | { [key: string]: string } | undefined,
  ): Promise<boolean> {
    if (
      !this.serverProcess ||
      !this.serverInfo ||
      !this.serverSpawnInfo ||
      !this.serverFileWatcher ||
      this.serverFileChanged === undefined
    ) {
      return true;
    }

    const changeTLDR = [];
    const changeDetails = [];

    if (this.serverFileChanged) {
      changeTLDR.push("an old binary");
    }

    const newSpawnInfo = ServerSpawnInfo.create(dapPath, args, env);
    if (!this.serverSpawnInfo.equals(newSpawnInfo)) {
      changeTLDR.push("different arguments");
      changeDetails.push(`
The previous lldb-dap server was started with:

${this.serverSpawnInfo.toDisplayTruncated()}

The new lldb-dap server will be started with:

${newSpawnInfo.toDisplayTruncated()}
`);
    }

    // If the server hasn't changed, continue startup without killing it.
    if (changeTLDR.length === 0) {
      return true;
    }

    // The server has changed. Prompt the user to restart it.
    const userInput = await vscode.window.showInformationMessage(
      "The lldb-dap server has changed. Would you like to restart the server?",
      {
        modal: true,
        detail: `An existing lldb-dap server (${this.serverProcess.pid}) is running with ${changeTLDR.map((s) => `*${s}*`).join(" and ")}.
${changeDetails.join("\n")}
Restarting the server will interrupt any existing debug sessions and start a new server.`,
      },
      "Restart",
      "Use Existing",
      "Show Full Details",
    );
    switch (userInput) {
      case "Restart":
        this.dispose();
        return true;
      case "Use Existing":
        return true;
      case "Show Full Details":
        await this.showFullSpawnInfoComparison(this.serverSpawnInfo, newSpawnInfo);
        return false;
      case undefined:
        return false;
    }
  }

  dispose() {
    if (!this.serverProcess) {
      return;
    }
    this.serverProcess.kill();
    this.cleanUp(this.serverProcess);
  }

  cleanUp(process: child_process.ChildProcessWithoutNullStreams) {
    // If the following don't equal, then the fields have already been updated
    // (either a new process has started, or the fields were already cleaned
    // up), and so the cleanup should be skipped.
    if (this.serverProcess === process) {
      this.serverProcess = undefined;
      this.serverInfo = undefined;
      this.serverSpawnInfo = undefined;
      this.serverFileWatcher?.close();
      this.serverFileWatcher = undefined;
      this.serverFileChanged = undefined;
    }
  }

  private async showFullSpawnInfoComparison(
    oldSpawnInfo: ServerSpawnInfo,
    newSpawnInfo: ServerSpawnInfo,
  ): Promise<void> {
    const content = `Comparing lldb-dap env and args
===============================
Each section contains the binary and args on the first line, and then env on the
following lines.

Existing
--------
${oldSpawnInfo.toDisplay()}

New
--------
${newSpawnInfo.toDisplay()}
`;
    const doc = await vscode.workspace.openTextDocument({
      content,
      language: "text",
    });
    await vscode.window.showTextDocument(doc, { preview: true });
  }
}
