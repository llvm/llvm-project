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
import { Logger } from "./logger";

/**
 * This class represents the extension and manages its life cycle. Other extensions
 * using it as as library should use this class as the main entry point.
 */
export class LLDBDapExtension extends DisposableContext {
  constructor(logger: Logger, outputChannel: vscode.OutputChannel) {
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
        new LLDBDapConfigurationProvider(lldbDapServer, logger),
      ),
      vscode.debug.registerDebugAdapterDescriptorFactory(
        "lldb-dap",
        new LLDBDapDescriptorFactory(logger),
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

    vscode.commands.registerCommand(
      "lldb-dap.modules.copyProperty",
      (node: ModuleProperty) => vscode.env.clipboard.writeText(node.value),
    );
  }
}

/**
 * This is the entry point when initialized by VS Code.
 */
export async function activate(context: vscode.ExtensionContext) {
  await vscode.workspace.fs.createDirectory(context.logUri);
  const outputChannel = vscode.window.createOutputChannel("LLDB-DAP");
  const logger = new Logger((name) => path.join(context.logUri.fsPath, name), outputChannel);
  logger.info("LLDB-Dap extension activating...");
  context.subscriptions.push(new LLDBDapExtension(logger, outputChannel));
  logger.info("LLDB-Dap extension activated");
}
