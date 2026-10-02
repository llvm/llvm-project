import { execFile } from "node:child_process";
import { promisify } from "node:util";

/** Represents a single process running on the system. */
export interface Process {
  /** Process ID. */
  id: number;

  /** Command that was used to start the process. */
  command: string;

  /** The full command including arguments that was used to start the process. */
  arguments: string;
}

export interface ProcessTree {
  listAllProcesses(): Promise<Process[]>;
}

/**
 * Options passed to {@link LldbDapProcessTree} to target a specific platform.
 */
export interface LldbDapProcessTreeOptions {
  /** Name of the LLDB platform to select (e.g. "host", "remote-linux"). */
  platformName?: string;
  /** URL to connect the platform to, for remote platforms. */
  platformUrl?: string;
}

/**
 * Runs a command and captures its stdout. Abstracted so tests can inject a
 * stub without spawning a real process.
 */
export type ExecFn = (
  exe: string,
  args: string[],
) => Promise<{ stdout: string }>;

const defaultExec: ExecFn = (() => {
  const promisified = promisify(execFile);
  return async (exe, args) => {
    // Remote platforms with many processes can emit a lot of JSON; give
    // ourselves headroom over node's 1 MB default.
    const { stdout } = await promisified(exe, args, {
      maxBuffer: 32 * 1024 * 1024,
    });
    return { stdout };
  };
})();

/**
 * Shape of a single entry in the JSON emitted by `lldb-dap --list-processes`.
 * Keys other than `pid` are optional — absent keys mean "not available" rather
 * than "empty".
 */
interface LldbDapProcessEntry {
  pid: number;
  name?: string;
  executable?: string;
  arguments?: string;
  triple?: string;
  user?: number;
}

/**
 * Lists processes by invoking `lldb-dap --list-processes`. The LLDB platform
 * layer enumerates the processes, so this works for remote platforms too when
 * `--platform` / `--platform-url` are supplied.
 */
export class LldbDapProcessTree implements ProcessTree {
  constructor(
    private readonly lldbDapPath: string,
    private readonly options: LldbDapProcessTreeOptions = {},
    private readonly exec: ExecFn = defaultExec,
  ) {}

  async listAllProcesses(): Promise<Process[]> {
    const args = ["--list-processes"];
    if (this.options.platformName) {
      args.push("--platform", this.options.platformName);
    }
    if (this.options.platformUrl) {
      args.push("--platform-url", this.options.platformUrl);
    }

    const { stdout } = await this.exec(this.lldbDapPath, args);
    return parseListProcessesOutput(stdout);
  }
}

/** Parses the JSON array produced by `lldb-dap --list-processes`. */
export function parseListProcessesOutput(stdout: string): Process[] {
  const parsed = JSON.parse(stdout);
  if (!Array.isArray(parsed)) {
    throw new Error(
      "Unexpected output from lldb-dap --list-processes (not a JSON array)",
    );
  }

  return parsed.map((entry: LldbDapProcessEntry) => {
    if (typeof entry?.pid !== "number") {
      throw new Error(
        "Unexpected output from lldb-dap --list-processes (entry missing numeric pid)",
      );
    }
    return {
      id: entry.pid,
      command: entry.executable ?? entry.name ?? "",
      arguments: entry.arguments ?? entry.name ?? "",
    };
  });
}
