import * as path from "path";
import { BaseProcessTree, ProcessTreeParser } from "../base-process-tree";

export class WindowsProcessTree extends BaseProcessTree {
  protected override getCommand(): string {
    return path.join(
      process.env["WINDIR"] || "C:\\Windows",
      "System32",
      "wbem",
      "WMIC.exe",
    );
  }

  protected override getCommandArguments(): string[] {
    return ["process", "get", "CommandLine,CreationDate,ProcessId"];
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
