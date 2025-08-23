import * as path from "path";
import * as vscode from "vscode";

import { LLDBDapDescriptorFactory } from "./debug-adapter-factory";
import { DisposableContext } from "./disposable-context";
import { LaunchUriHandler } from "./uri-launch-handler";
import { LLDBDapConfigurationProvider } from "./debug-configuration-provider";
import { LLDBDapServer } from "./lldb-dap-server";
import { DebugSessionTracker } from "./debug-session-tracker";
import {
  ModulesDataProvider,
  ModuleProperty,
} from "./ui/modules-data-provider";
import { LogFilePathProvider } from "./logging";
import { SymbolsProvider } from "./ui/symbols-provider";

/**
 * This class represents the extension and manages its life cycle. Other extensions
 * using it as as library should use this class as the main entry point.
 */
export class LLDBDapExtension extends DisposableContext {
  constructor(
    context: vscode.ExtensionContext,
    logger: vscode.LogOutputChannel,
    logFilePath: LogFilePathProvider,
    outputChannel: vscode.OutputChannel,
  ) {
    super();

    const lldbDapServer = new LLDBDapServer();
    const sessionTracker = new DebugSessionTracker(logger);

    this.pushSubscription(
      logger,
      outputChannel,
      lldbDapServer,
      sessionTracker,
      vscode.debug.registerDebugConfigurationProvider(
        "lldb-dap",
        new LLDBDapConfigurationProvider(lldbDapServer, logger, logFilePath),
      ),
      vscode.debug.registerDebugAdapterDescriptorFactory(
        "lldb-dap",
        new LLDBDapDescriptorFactory(logger, logFilePath),
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

    this.pushSubscription(vscode.commands.registerCommand(
      "lldb-dap.modules.copyProperty",
      (node: ModuleProperty) => vscode.env.clipboard.writeText(node.value),
    ));

    this.pushSubscription(new SymbolsProvider(sessionTracker, context));
  }
}

/**
 * This is the entry point when initialized by VS Code.
 */
export async function activate(context: vscode.ExtensionContext) {
  const outputChannel = vscode.window.createOutputChannel("LLDB-DAP", { log: true });
  outputChannel.info("LLDB-DAP extension activating...");
  const logFilePath = new LogFilePathProvider(context, outputChannel);
  context.subscriptions.push(
    new LLDBDapExtension(context, outputChannel, logFilePath, outputChannel),
  );
  outputChannel.info("LLDB-DAP extension activated");
}
