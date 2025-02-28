import { BaseProcessTree, ProcessTreeParser } from "../base-process-tree";

export class LinuxProcessTree extends BaseProcessTree {
  protected override getCommand(): string {
    return "ps";
  }

  protected override getCommandArguments(): string[] {
    return ["-axo", "pid=PID,lstart=START,comm:128=COMMAND,command=ARGUMENTS"];
  }

  protected override createParser(): ProcessTreeParser {
    let commandOffset: number | undefined;
    let argumentsOffset: number | undefined;
    return (line) => {
      if (!commandOffset || !argumentsOffset) {
        commandOffset = line.indexOf("COMMAND");
        argumentsOffset = line.indexOf("ARGUMENTS");
        return;
      }

      const pid = /^\s*([0-9]+)\s*/.exec(line);
      if (!pid) {
        return;
      }

      return {
        id: Number(pid[1]),
        command: line.slice(commandOffset, argumentsOffset).trim(),
        arguments: line.slice(argumentsOffset).trim(),
        start: Date.parse(line.slice(pid[0].length, commandOffset).trim()),
      };
    };
  }
}
