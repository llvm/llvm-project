import * as assert from "assert";

import {
  clearPickProcessContext,
  setPickProcessContext,
  takePickProcessContext,
} from "../../src/commands/pick-process-context";

suite("pick-process-context", function () {
  // The module holds a single-slot singleton; make sure each test starts clean.
  setup(function () {
    clearPickProcessContext();
  });

  test("take returns undefined when nothing is stashed", function () {
    assert.strictEqual(takePickProcessContext(), undefined);
  });

  test("take returns the last set value", function () {
    const cfg = { type: "lldb-dap", request: "attach", name: "A" };
    setPickProcessContext({ folder: undefined, debugConfiguration: cfg });

    const ctx = takePickProcessContext();
    assert.ok(ctx);
    assert.strictEqual(ctx!.debugConfiguration, cfg);
  });

  test("take clears the slot", function () {
    setPickProcessContext({
      folder: undefined,
      debugConfiguration: { type: "lldb-dap", request: "attach", name: "A" },
    });

    assert.notStrictEqual(takePickProcessContext(), undefined);
    assert.strictEqual(takePickProcessContext(), undefined);
  });

  test("set overwrites a pending value", function () {
    const first = { type: "lldb-dap", request: "attach", name: "first" };
    const second = { type: "lldb-dap", request: "attach", name: "second" };

    setPickProcessContext({ folder: undefined, debugConfiguration: first });
    setPickProcessContext({ folder: undefined, debugConfiguration: second });

    const ctx = takePickProcessContext();
    assert.strictEqual(ctx!.debugConfiguration, second);
  });

  test("clear discards pending context", function () {
    setPickProcessContext({
      folder: undefined,
      debugConfiguration: { type: "lldb-dap", request: "attach", name: "A" },
    });

    clearPickProcessContext();

    assert.strictEqual(takePickProcessContext(), undefined);
  });
});
