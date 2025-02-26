import { ChildProcessWithoutNullStreams, spawn } from "child_process";
import { LinuxProcessTree } from "./linux-process-tree";

function fill(prefix: string, suffix: string, length: number): string {
  return prefix + suffix.repeat(length - prefix.length);
}

export class DarwinProcessTree extends LinuxProcessTree {
  protected override spawnProcess(): ChildProcessWithoutNullStreams {
    return spawn("ps", [
      "-xo",
      // The length of comm must be large enough or data will be truncated.
      `pid=PID,lstart=START,comm=${fill("COMMAND", "-", 256)},command=ARGUMENTS`,
    ]);
  }
}
