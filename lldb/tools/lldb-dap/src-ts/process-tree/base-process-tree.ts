import { ChildProcessWithoutNullStreams } from "child_process";
import { Process, ProcessTree } from ".";
import { Transform } from "stream";

/** Parses process information from a given line of process output. */
export type ProcessTreeParser = (line: string) => Process | undefined;

/**
 * Implements common behavior between the different {@link ProcessTree} implementations.
 */
export abstract class BaseProcessTree implements ProcessTree {
  /**
   * Spawn the process responsible for collecting all processes on the system.
   */
  protected abstract spawnProcess(): ChildProcessWithoutNullStreams;

  /**
   * Create a new parser that can read the process information from stdout of the process
   * spawned by {@link spawnProcess spawnProcess()}.
   */
  protected abstract createParser(): ProcessTreeParser;

  listAllProcesses(): Promise<Process[]> {
    return new Promise<Process[]>((resolve, reject) => {
      const proc = this.spawnProcess();
      const parser = this.createParser();

      // Capture processes from stdout
      const processes: Process[] = [];
      proc.stdout.pipe(new LineBasedStream()).on("data", (line) => {
        const process = parser(line.toString());
        if (process && process.id !== proc.pid) {
          processes.push(process);
        }
      });

      // Resolve or reject the promise based on exit code/signal/error
      proc.on("error", reject);
      proc.on("exit", (code, signal) => {
        if (code === 0) {
          resolve(processes);
        } else if (signal) {
          reject(
            new Error(
              `Unable to list processes: process exited due to signal ${signal}`,
            ),
          );
        } else {
          reject(
            new Error(
              `Unable to list processes: process exited with code ${code}`,
            ),
          );
        }
      });
    });
  }
}

/**
 * A stream that emits each line as a single chunk of data. The end of a line is denoted
 * by the newline character '\n'.
 */
export class LineBasedStream extends Transform {
  private readonly newline: number = "\n".charCodeAt(0);
  private buffer: Buffer = Buffer.alloc(0);

  override _transform(
    chunk: Buffer,
    _encoding: string,
    callback: () => void,
  ): void {
    let currentIndex = 0;
    while (currentIndex < chunk.length) {
      const newlineIndex = chunk.indexOf(this.newline, currentIndex);
      if (newlineIndex === -1) {
        this.buffer = Buffer.concat([
          this.buffer,
          chunk.subarray(currentIndex),
        ]);
        break;
      }

      const newlineChunk = chunk.subarray(currentIndex, newlineIndex);
      const line = Buffer.concat([this.buffer, newlineChunk]);
      this.push(line);
      this.buffer = Buffer.alloc(0);

      currentIndex = newlineIndex + 1;
    }

    callback();
  }

  override _flush(callback: () => void): void {
    if (this.buffer.length) {
      this.push(this.buffer);
    }

    callback();
  }
}
