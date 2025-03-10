import * as path from "path";
import { BaseProcessTree, ProcessTreeParser } from "../base-process-tree";

export class WindowsProcessTree extends BaseProcessTree {
  protected override getCommand(): string {
    return "PowerShell";
  }

  protected override getCommandArguments(): string[] {
    return [
      "-Command",
      'Get-CimInstance -ClassName Win32_Process | Format-Table ProcessId, @{Label="CreationDate";Expression={"{0:yyyyMddHHmmss}" -f $_.CreationDate}}, CommandLine | Out-String -width 9999',
    ];
  }

  protected override createParser(): ProcessTreeParser {
    const lineRegex = /^([0-9]+)\s+([0-9]+)\s+(.*)$/;

    return (line) => {
      const matches = lineRegex.exec(line.trim());
      if (!matches || matches.length !== 4) {
        return;
      }

      const id = Number(matches[1]);
      const start = Number(matches[2]);
      let fullCommandLine = matches[3].trim();
      if (isNaN(id) || !fullCommandLine) {
        return;
      }
      // Extract the command from the full command line
      let command = fullCommandLine;
      if (fullCommandLine[0] === '"') {
        const end = fullCommandLine.indexOf('"', 1);
        if (end > 0) {
          command = fullCommandLine.slice(1, end);
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
