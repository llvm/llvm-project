import * as assert from "assert";

import {
  ExecFn,
  LldbDapProcessTree,
  parseListProcessesOutput,
} from "../../src/process-tree";

/** Builds an {@link ExecFn} stub that records invocations and returns canned stdout. */
function stubExec(stdout: string): {
  exec: ExecFn;
  calls: { exe: string; args: string[] }[];
} {
  const calls: { exe: string; args: string[] }[] = [];
  const exec: ExecFn = async (exe, args) => {
    calls.push({ exe, args });
    return { stdout };
  };
  return { exec, calls };
}

suite("LldbDapProcessTree", function () {
  test("invokes the given lldb-dap with --list-processes", async function () {
    const { exec, calls } = stubExec("[]");
    const tree = new LldbDapProcessTree("/opt/bin/lldb-dap", {}, exec);

    const processes = await tree.listAllProcesses();

    assert.deepStrictEqual(processes, []);
    assert.strictEqual(calls.length, 1);
    assert.strictEqual(calls[0].exe, "/opt/bin/lldb-dap");
    assert.deepStrictEqual(calls[0].args, ["--list-processes"]);
  });

  test("forwards platformName and platformUrl", async function () {
    const { exec, calls } = stubExec("[]");
    const tree = new LldbDapProcessTree(
      "lldb-dap",
      {
        platformName: "remote-linux",
        platformUrl: "connect://localhost:1234",
      },
      exec,
    );

    await tree.listAllProcesses();

    assert.deepStrictEqual(calls[0].args, [
      "--list-processes",
      "--platform",
      "remote-linux",
      "--platform-url",
      "connect://localhost:1234",
    ]);
  });

  test("parses a realistic multi-process listing", async function () {
    const stdout = JSON.stringify([
      {
        pid: 501,
        name: "bash",
        executable: "/bin/bash",
        triple: "arm64-apple-macosx",
        user: 501,
      },
      {
        pid: 502,
        name: "vim",
        executable: "/usr/bin/vim",
        triple: "arm64-apple-macosx",
        user: 501,
      },
      {
        pid: 1,
        name: "launchd",
        executable: "/sbin/launchd",
        triple: "arm64-apple-macosx",
        user: 0,
      },
    ]);
    const tree = new LldbDapProcessTree("lldb-dap", {}, stubExec(stdout).exec);

    const processes = await tree.listAllProcesses();

    assert.strictEqual(processes.length, 3);
    assert.deepStrictEqual(processes[0], {
      id: 501,
      command: "/bin/bash",
      arguments: "bash",
    });
    assert.deepStrictEqual(processes[1], {
      id: 502,
      command: "/usr/bin/vim",
      arguments: "vim",
    });
    assert.deepStrictEqual(processes[2], {
      id: 1,
      command: "/sbin/launchd",
      arguments: "launchd",
    });
  });

  test("propagates exec errors", async function () {
    const failing: ExecFn = async () => {
      throw new Error("lldb-dap: error: unknown argument '--list-processes'");
    };
    const tree = new LldbDapProcessTree("lldb-dap", {}, failing);

    await assert.rejects(() => tree.listAllProcesses(), /unknown argument/);
  });
});

suite("parseListProcessesOutput", function () {
  test("parses a full entry", function () {
    const stdout = JSON.stringify([
      {
        pid: 1234,
        name: "bash",
        executable: "/bin/bash",
        triple: "arm64-apple-macosx",
        user: 501,
      },
    ]);
    const [proc] = parseListProcessesOutput(stdout);
    assert.strictEqual(proc.id, 1234);
    assert.strictEqual(proc.command, "/bin/bash");
    assert.strictEqual(proc.arguments, "bash");
  });

  test("falls back to name when executable is missing", function () {
    const stdout = JSON.stringify([{ pid: 42, name: "kernel_task" }]);
    const [proc] = parseListProcessesOutput(stdout);
    assert.strictEqual(proc.command, "kernel_task");
    assert.strictEqual(proc.arguments, "kernel_task");
  });

  test("handles an entry with only a pid", function () {
    const [proc] = parseListProcessesOutput(JSON.stringify([{ pid: 7 }]));
    assert.strictEqual(proc.id, 7);
    assert.strictEqual(proc.command, "");
    assert.strictEqual(proc.arguments, "");
  });

  test("returns an empty array for an empty JSON list", function () {
    assert.deepStrictEqual(parseListProcessesOutput("[]"), []);
  });

  test("rejects non-array JSON", function () {
    assert.throws(
      () => parseListProcessesOutput('{"pid": 1}'),
      /not a JSON array/,
    );
  });

  test("rejects entry without numeric pid", function () {
    assert.throws(
      () => parseListProcessesOutput(JSON.stringify([{ name: "orphan" }])),
      /missing numeric pid/,
    );
  });

  test("rejects malformed JSON", function () {
    assert.throws(() => parseListProcessesOutput("not json"));
  });
});
