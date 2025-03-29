import * as vscode from "vscode";

import { LLDBDapDescriptorFactory } from "./debug-adapter-factory";
import { DisposableContext } from "./disposable-context";
import { LaunchUriHandler } from "./uri-launch-handler";
import { LLDBDapConfigurationProvider } from "./debug-configuration-provider";
import { LLDBDapServer } from "./lldb-dap-server";

/**
 * This class represents the extension and manages its life cycle. Other extensions
 * using it as as library should use this class as the main entry point.
 */
export class LLDBDapExtension extends DisposableContext {
  constructor() {
    super();

    const lldbDapServer = new LLDBDapServer();
    this.pushSubscription(lldbDapServer);

    this.pushSubscription(
      vscode.debug.registerDebugConfigurationProvider(
        "lldb-dap",
        new LLDBDapConfigurationProvider(lldbDapServer),
      ),
    );

    this.pushSubscription(
      vscode.debug.registerDebugAdapterDescriptorFactory(
        "lldb-dap",
        new LLDBDapDescriptorFactory(),
      ),
    );

    this.pushSubscription(
      vscode.window.registerUriHandler(new LaunchUriHandler()),
    );
  }
}

/**
 * This is the entry point when initialized by VS Code.
 */
export function activate(context: vscode.ExtensionContext) {
  context.subscriptions.push(new LLDBDapExtension());
}
