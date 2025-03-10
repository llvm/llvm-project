import * as util from "util";
import * as child_process from "child_process";
import { Process, ProcessTree } from ".";

const exec = util.promisify(child_process.execFile);

/** Parses process information from a given line of process output. */
export type ProcessTreeParser = (line: string) => Process | undefined;

/**
 * Implements common behavior between the different {@link ProcessTree} implementations.
 */
export abstract class BaseProcessTree implements ProcessTree {
  /**
   * Get the command responsible for collecting all processes on the system.
   */
  protected abstract getCommand(): string;

  /**
   * Get the list of arguments used to launch the command.
   */
  protected abstract getCommandArguments(): string[];

  /**
   * Create a new parser that can read the process information from stdout of the process
   * spawned by {@link spawnProcess spawnProcess()}.
   */
  protected abstract createParser(): ProcessTreeParser;

  async listAllProcesses(): Promise<Process[]> {
    const execCommand = exec(this.getCommand(), this.getCommandArguments());
    const parser = this.createParser();
    return (await execCommand).stdout.split("\n").flatMap((line) => {
      const process = parser(line.toString());
      if (!process || process.id === execCommand.child.pid) {
        return [];
      }
      return [process];
    });
  }
}
