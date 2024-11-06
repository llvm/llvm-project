import * as vscode from "vscode";
import { LLDBDapOptions } from "./types";
import { DisposableContext } from "./disposable-context";
import { LLDBDapDescriptorFactory } from "./debug-adapter-factory";

/**
 * This creates the configurations for this project if used as a standalone
 * extension.
 */
function createDefaultLLDBDapOptions(): LLDBDapOptions {
  return {
    debuggerType: "lldb-dap",
    async createDapExecutableCommand(
      session: vscode.DebugSession,
      packageJSONExecutable: vscode.DebugAdapterExecutable | undefined,
    ): Promise<vscode.DebugAdapterExecutable | undefined> {
      const config = vscode.workspace.getConfiguration(
        "lldb-dap",
        session.workspaceFolder,
      );
      const path = config.get<string>("executable-path");
      const log_path = config.get<string>("log-path");

      let env: { [key: string]: string } = {};
      if (log_path) {
        env["LLDBDAP_LOG"] = log_path;
      }
      const configEnvironment = config.get<{ [key: string]: string }>("environment") || {};
      if (path) {
        const dbgOptions = {
          env: {
            ...configEnvironment,
            ...env,
          }
        };
        return new vscode.DebugAdapterExecutable(path, [], dbgOptions);
      } else if (packageJSONExecutable) {
        return new vscode.DebugAdapterExecutable(
          packageJSONExecutable.command,
          packageJSONExecutable.args,
          {
            ...packageJSONExecutable.options,
            env: {
              ...packageJSONExecutable.options?.env,
              ...configEnvironment,
              ...env,
            },
          },
        );
      } else {
        return undefined;
      }
    },
  };
}

/**
 * This class represents the extension and manages its life cycle. Other extensions
 * using it as as library should use this class as the main entry point.
 */
export class LLDBDapExtension extends DisposableContext {
  private lldbDapOptions: LLDBDapOptions;

  constructor(lldbDapOptions: LLDBDapOptions) {
    super();
    this.lldbDapOptions = lldbDapOptions;

    this.pushSubscription(
      vscode.debug.registerDebugAdapterDescriptorFactory(
        this.lldbDapOptions.debuggerType,
        new LLDBDapDescriptorFactory(this.lldbDapOptions),
      ),
    );

    this.pushSubscription(
      vscode.workspace.onDidChangeConfiguration(async (event) => {
        if (event.affectsConfiguration("lldb-dap.executable-path")) {
          const dapPath = vscode.workspace
            .getConfiguration("lldb-dap")
            .get<string>("executable-path");

          if (dapPath) {
            const fileUri = vscode.Uri.file(dapPath);
            if (
              await LLDBDapDescriptorFactory.isValidDebugAdapterPath(fileUri)
            ) {
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
  context.subscriptions.push(
    new LLDBDapExtension(createDefaultLLDBDapOptions()),
  );
}
