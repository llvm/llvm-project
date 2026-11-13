import * as assert from "assert";

import type { dap } from "@protocol/dap";
import type * as vscode from "vscode";
import { VariablesTrackerFactory } from "../../src/ui/variables-tracker";

function response<T extends dap.Response>(
  command: string,
  body: T["body"],
  success = true,
): T {
  return {
    seq: 2,
    type: "response",
    request_seq: 1,
    command,
    success,
    body,
  } as T;
}

function variablesResponse(...variables: dap.Variable[]) {
  return response<dap.VariablesResponse>("variables", { variables });
}

function variable(value: string, type?: string): dap.Variable {
  return { name: "pt", value, type, variablesReference: 0 };
}

suite("VariablesTracker", function () {
  // Mimics `debug.showVariableTypes` setting.
  const settings = { showVariableTypes: false };
  const factory = new VariablesTrackerFactory(() => settings.showVariableTypes);
  let tracker: vscode.DebugAdapterTracker;

  setup(function () {
    settings.showVariableTypes = false;
    tracker = factory.createDebugAdapterTracker();
  });

  test("puts the type in an empty variable value", function () {
    const message = variablesResponse(variable("", "PointType"));
    tracker.onDidSendMessage!(message);
    assert.strictEqual(message.body.variables[0].value, "PointType");
  });

  test("keeps a variable value", function () {
    const message = variablesResponse(
      variable("11", "int"),
      variable("hello world", "std::string"),
    );
    tracker.onDidSendMessage!(message);

    const [var1, var2] = message.body.variables;
    assert.strictEqual(var1.value, "11");
    assert.strictEqual(var1.type, "int");

    assert.strictEqual(var2.value, "hello world");
    assert.strictEqual(var2.type, "std::string");
  });

  test("keeps an empty value without a type", function () {
    const message = variablesResponse(variable(""));
    tracker.onDidSendMessage!(message);
    assert.strictEqual(message.body.variables[0].value, "");
  });

  test("puts the type in an empty setVariable value", function () {
    const message = response<dap.SetVariableResponse>("setVariable", {
      value: "",
      type: "PointType",
    });
    tracker.onDidSendMessage!(message);
    assert.strictEqual(message.body.value, "PointType");
  });

  test("puts the type in an empty evaluate result", function () {
    const message = response<dap.EvaluateResponse>("evaluate", {
      result: "",
      type: "PointType",
      variablesReference: 1,
    });
    tracker.onDidSendMessage!(message);
    assert.strictEqual(message.body.result, "PointType");
  });

  test("keeps an empty evaluate result without a type", function () {
    // e.g. a debug console command with no output.
    const message = response<dap.EvaluateResponse>("evaluate", {
      result: "",
      variablesReference: 0,
    });
    tracker.onDidSendMessage!(message);
    assert.strictEqual(message.body.result, "");
  });

  test("follows `showVariableTypes` setting during a session", function () {
    const typesHidden = variablesResponse(variable("", "PointType"));
    tracker.onDidSendMessage!(typesHidden);

    settings.showVariableTypes = true;
    const typesShown = variablesResponse(variable("", "PointType"));
    tracker.onDidSendMessage!(typesShown);

    settings.showVariableTypes = false;
    const typesHiddenAgain = variablesResponse(variable("", "PointType"));
    tracker.onDidSendMessage!(typesHiddenAgain);

    assert.strictEqual(typesHidden.body.variables[0].value, "PointType");
    assert.strictEqual(typesShown.body.variables[0].value, "");
    assert.strictEqual(typesHiddenAgain.body.variables[0].value, "PointType");
  });

  test("ignores failed responses", function () {
    const message = response<dap.VariablesResponse>(
      "variables",
      { variables: [variable("", "PointType")] },
      /*success=*/ false,
    );
    tracker.onDidSendMessage!(message);
    assert.strictEqual(message.body.variables[0].value, "");
  });

  test("ignores messages that are not responses", function () {
    const message = {
      seq: 1,
      type: "event",
      event: "output",
      body: { output: "some text" },
    } as dap.OutputEvent;
    tracker.onDidSendMessage!(message);
    assert.deepStrictEqual(message.body, { output: "some text" });
  });
});

suite("VariablesTrackerFactory", function () {
  test("creates a tracker regardless of lambda return value", function () {
    const factoryFalse = new VariablesTrackerFactory(() => false);
    assert.ok(factoryFalse.createDebugAdapterTracker());

    const factoryTrue = new VariablesTrackerFactory(() => true);
    assert.ok(factoryTrue.createDebugAdapterTracker());
  });
});
