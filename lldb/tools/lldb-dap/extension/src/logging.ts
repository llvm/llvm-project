import * as path from "path";
import * as vscode from "vscode";

/**
 * Formats the given date as a string in the form "YYYYMMddTHHMMSS".
 *
 * @param date The date to format as a string.
 * @returns The formatted date.
 */
function formatDate(date: Date): string {
  const year = date.getFullYear().toString().padStart(4, "0");
  const month = (date.getMonth() + 1).toString().padStart(2, "0");
  const day = date.getDate().toString().padStart(2, "0");
  const hour = date.getHours().toString().padStart(2, "0");
  const minute = date.getMinutes().toString().padStart(2, "0");
  const seconds = date.getSeconds().toString().padStart(2, "0");
  return `${year}${month}${day}T${hour}${minute}${seconds}`;
}

export enum LogType {
  DEBUG_SESSION,
}

export class LogFilePathProvider {
  private logFolder: string = "";

  constructor(
    private context: vscode.ExtensionContext,
    private logger: vscode.LogOutputChannel,
  ) {
    this.updateLogFolder();
    context.subscriptions.push(
      vscode.workspace.onDidChangeConfiguration((e) => {
        if (e.affectsConfiguration("lldb-dap.logFolder")) {
          this.updateLogFolder();
        }
      }),
    );
  }

  get(type: LogType): string {
    const logFolder = this.logFolder || this.context.logUri.fsPath;
    switch (type) {
      case LogType.DEBUG_SESSION:
        return path.join(
          logFolder,
          `lldb-dap-${formatDate(new Date())}-${vscode.env.sessionId.split("-")[0]}.log`,
        );
        break;
    }
  }

  private updateLogFolder() {
    const config = vscode.workspace.getConfiguration("lldb-dap");
    let logFolder =
      config.get<string>("logFolder") || this.context.logUri.fsPath;
    vscode.workspace.fs
      .createDirectory(vscode.Uri.file(logFolder))
      .then(undefined, (error) => {
        this.logger.error(`Failed to create log folder ${logFolder}: ${error}`);
        logFolder = this.context.logUri.fsPath;
      })
      .then(() => {
        this.logFolder = logFolder;
        this.logger.info(`Persisting lldb-dap logs to ${logFolder}`);
      });
  }
}
