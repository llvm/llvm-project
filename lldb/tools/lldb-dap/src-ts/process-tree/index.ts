import { DarwinProcessTree } from "./platforms/darwin-process-tree";
import { LinuxProcessTree } from "./platforms/linux-process-tree";
import { WindowsProcessTree } from "./platforms/windows-process-tree";

/**
 * Represents a single process running on the system.
 */
export interface Process {
  /** Process ID */
  id: number;

  /** Command that was used to start the process */
  command: string;

  /** The full command including arguments that was used to start the process */
  arguments: string;

  /** The date when the process was started */
  start: number;
}

export interface ProcessTree {
  listAllProcesses(): Promise<Process[]>;
}

/** Returns a {@link ProcessTree} based on the current platform. */
export function createProcessTree(): ProcessTree {
  switch (process.platform) {
    case "darwin":
      return new DarwinProcessTree();
    case "win32":
      return new WindowsProcessTree();
    default:
      return new LinuxProcessTree();
  }
}
