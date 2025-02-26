import * as path from "path";
import { BaseProcessTree, ProcessTreeParser } from "../base-process-tree";
import { ChildProcessWithoutNullStreams, spawn } from "child_process";

export class WindowsProcessTree extends BaseProcessTree {
  protected override spawnProcess(): ChildProcessWithoutNullStreams {
    const wmic = path.join(
      process.env["WINDIR"] || "C:\\Windows",
      "System32",
      "wbem",
      "WMIC.exe",
    );
    return spawn(
      wmic,
      ["process", "get", "CommandLine,CreationDate,ProcessId"],
      { stdio: "pipe" },
    );
  }

  protected override createParser(): ProcessTreeParser {
    const lineRegex = /^(.*)\s+([0-9]+)\.[0-9]+[+-][0-9]+\s+([0-9]+)$/;

    return (line) => {
      const matches = lineRegex.exec(line.trim());
      if (!matches || matches.length !== 4) {
        return;
      }

      const id = Number(matches[3]);
      const start = Number(matches[2]);
      let fullCommandLine = matches[1].trim();
      if (isNaN(id) || !fullCommandLine) {
        return;
      }
      // Extract the command from the full command line
      let command = fullCommandLine;
      if (fullCommandLine[0] === '"') {
        const end = fullCommandLine.indexOf('"', 1);
        if (end > 0) {
          command = fullCommandLine.slice(1, end - 1);
        }
      } else {
        const end = fullCommandLine.indexOf(" ");
        if (end > 0) {
          command = fullCommandLine.slice(0, end);
        }
      }

      return { id, command, arguments: fullCommandLine, start };
    };
  }
}
