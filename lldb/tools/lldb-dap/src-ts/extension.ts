import * as vscode from "vscode";

import { LLDBDapDescriptorFactory } from "./debug-adapter-factory";
import { DisposableContext } from "./disposable-context";
import { LaunchUriHandler } from "./uri-launch-handler";
import { LLDBDapConfigurationProvider } from "./debug-configuration-provider";
import { LLDBDapServer } from "./lldb-dap-server";
import { DebugSessionTracker } from "./debug-session-tracker";
import { ModulesDataProvider } from "./ui/modules-data-provider";

/**
 * This class represents the extension and manages its life cycle. Other extensions
 * using it as as library should use this class as the main entry point.
 */
export class LLDBDapExtension extends DisposableContext {
  constructor() {
    super();

    const lldbDapServer = new LLDBDapServer();
    const sessionTracker = new DebugSessionTracker();

    this.pushSubscription(
      lldbDapServer,
      sessionTracker,
      vscode.debug.registerDebugConfigurationProvider(
        "lldb-dap",
        new LLDBDapConfigurationProvider(lldbDapServer),
      ),
      vscode.debug.registerDebugAdapterDescriptorFactory(
        "lldb-dap",
        new LLDBDapDescriptorFactory(),
      ),
      vscode.debug.registerDebugAdapterTrackerFactory(
        "lldb-dap",
        sessionTracker,
      ),
      vscode.window.registerTreeDataProvider(
        "lldb-dap.modules",
        new ModulesDataProvider(sessionTracker),
      ),
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
