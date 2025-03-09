import { BaseProcessTree, ProcessTreeParser } from "../base-process-tree";

export class LinuxProcessTree extends BaseProcessTree {
  protected override getCommand(): string {
    return "ps";
  }

  protected override getCommandArguments(): string[] {
    return [
      "-axo",
      // The length of exe must be large enough or data will be truncated.
      `pid=PID,state=STATE,lstart=START,exe:128=COMMAND,args=ARGUMENTS`,
    ];
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

      const pidAndState = /^\s*([0-9]+)\s+([a-zA-Z<>+]+)\s+/.exec(line);
      if (!pidAndState) {
        return;
      }

      // Make sure the process isn't in a trace/debug or zombie state as we cannot attach to them
      const state = pidAndState[2];
      if (state.includes("X") || state.includes("Z")) {
        return;
      }

      // ps will list "-" as the command if it does not know where the executable is located
      const command = line.slice(commandOffset, argumentsOffset).trim();
      if (command === "-") {
        return;
      }

      return {
        id: Number(pidAndState[1]),
        command,
        arguments: line.slice(argumentsOffset).trim(),
        start: Date.parse(
          line.slice(pidAndState[0].length, commandOffset).trim(),
        ),
      };
    };
  }
}
