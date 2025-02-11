import * as path from "path";
import * as util from "util";
import * as vscode from "vscode";

import {
  LLDBDapDescriptorFactory,
  isExecutable,
} from "./debug-adapter-factory";
import { DisposableContext } from "./disposable-context";

/**
 * This class represents the extension and manages its life cycle. Other extensions
 * using it as as library should use this class as the main entry point.
 */
export class LLDBDapExtension extends DisposableContext {
  constructor() {
    super();
    this.pushSubscription(
      vscode.debug.registerDebugAdapterDescriptorFactory(
        "lldb-dap",
        new LLDBDapDescriptorFactory(),
      ),
    );

    this.pushSubscription(
      vscode.workspace.onDidChangeConfiguration(async (event) => {
        if (event.affectsConfiguration("lldb-dap.executable-path")) {
          const dapPath = vscode.workspace
            .getConfiguration("lldb-dap")
            .get<string>("executable-path");

          if (dapPath) {
            if (await isExecutable(dapPath)) {
              return;
            }
          }
          LLDBDapDescriptorFactory.showLLDBDapNotFoundMessage(dapPath || "");
        }
      }),
    );
  }
}

/**
 * This is the entry point when initialized by VS Code.
 */
export function activate(context: vscode.ExtensionContext) {
  context.subscriptions.push(new LLDBDapExtension());
}
