import { LinuxProcessTree } from "./linux-process-tree";

export class DarwinProcessTree extends LinuxProcessTree {
  protected override getCommandArguments(): string[] {
    return [
      "-axo",
      // The length of comm must be large enough or data will be truncated.
      `pid=PID,lstart=START,comm=${"COMMAND".padEnd(256, "-")},args=ARGUMENTS`,
    ];
  }
}
