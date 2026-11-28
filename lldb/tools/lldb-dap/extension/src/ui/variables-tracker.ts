import type { dap } from "@protocol/dap";
import type * as vscode from "vscode";

export class VariablesTrackerFactory
  implements vscode.DebugAdapterTrackerFactory
{
  constructor(private readonly showVariableTypes: () => boolean) {}

  createDebugAdapterTracker(): vscode.DebugAdapterTracker {
    return new VariablesTracker(this.showVariableTypes);
  }
}

/**
 * Sets the variables's type as it's displayed value when the variable has
 * no value. e.g. a struct without a value.
 */
class VariablesTracker implements vscode.DebugAdapterTracker {
  constructor(private readonly showVariableTypes: () => boolean) {}

  onDidSendMessage(message: dap.ProtocolMessage) {
    if (message.type !== "response") {
      return;
    }

    const response = message as dap.Response;
    if (!response.success || this.showVariableTypes()) {
      return;
    }

    switch (response.command) {
      case "variables":
        const var_response = response as dap.VariablesResponse;
        const variables = var_response.body.variables;
        for (const variable of variables) {
          if (variable.value === "" && variable.type) {
            variable.value = variable.type;
          }
        }
        break;
      case "setVariable": {
        const body = (response as dap.SetVariableResponse).body;
        if (body.value === "" && body.type) {
          body.value = body.type;
        }
        break;
      }
      case "evaluate": {
        const body = (response as dap.EvaluateResponse).body;
        if (body.result === "" && body.type) {
          body.result = body.type;
        }
        break;
      }
      // TODO: update if we ever support "setExpression" command.
    }
  }
}
