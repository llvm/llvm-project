import * as vscode from "vscode";
import { DebugProtocol } from "@vscode/debugprotocol";

import { DebugSessionTracker } from "../debug-session-tracker";
import { DisposableContext } from "../disposable-context";

import { SymbolType } from "..";
import { getSymbolsTableHTMLContent } from "./symbols-webview-html";
import { getDefaultConfigKey } from "../debug-configuration-provider";

export class SymbolsProvider extends DisposableContext {
  constructor(
    private readonly tracker: DebugSessionTracker,
    private readonly extensionContext: vscode.ExtensionContext,
  ) {
    super();

    this.pushSubscription(vscode.commands.registerCommand(
      "lldb-dap.debug.showSymbols",
      () => {
        const session = vscode.debug.activeDebugSession;
        if (!session) return;

        this.SelectModuleAndShowSymbols(session);
      },
    ));

    this.pushSubscription(vscode.commands.registerCommand(
      "lldb-dap.modules.showSymbols",
      (moduleItem: DebugProtocol.Module) => {
        const session = vscode.debug.activeDebugSession;
        if (!session) return;

        this.showSymbolsForModule(session, moduleItem);
      },
    ));

    this.tracker.onDidReceiveSessionCapabilities(([ _session, capabilities ]) => {
      if (capabilities.supportsModuleSymbolsRequest) {
        vscode.commands.executeCommand(
            "setContext", "lldb-dap.supportsModuleSymbolsRequest", true);
      }
    });

    this.tracker.onDidExitSession((_session) => {
      vscode.commands.executeCommand("setContext", "lldb-dap.supportsModuleSymbolsRequest", false);
    });
  }

  private async SelectModuleAndShowSymbols(session: vscode.DebugSession) {
    const modules = this.tracker.debugSessionModules(session);
    if (!modules || modules.length === 0) {
      return;
    }

    // Let the user select a module to show symbols for
    const selectedModule = await vscode.window.showQuickPick(modules.map(m => new ModuleQuickPickItem(m)), {
        placeHolder: "Select a module to show symbols for"
    });
    if (!selectedModule) {
      return;
    }

    this.showSymbolsForModule(session, selectedModule.module);
  }

  private async showSymbolsForModule(session: vscode.DebugSession, module: DebugProtocol.Module) {
    try {
      const symbols = await this.getSymbolsForModule(session, module.id.toString());
      this.showSymbolsInNewTab(module.name.toString(), symbols);
    } catch (error) {
      if (error instanceof Error) {
        vscode.window.showErrorMessage("Failed to retrieve symbols: " + error.message);
      } else {
        vscode.window.showErrorMessage("Failed to retrieve symbols due to an unknown error.");
      }
      
      return;
    }
  }

  private async getSymbolsForModule(session: vscode.DebugSession, moduleId: string): Promise<SymbolType[]> {
    const symbols_response: { symbols: Array<SymbolType> } = await session.customRequest("__lldb_moduleSymbols", { moduleId, moduleName: '' });
    return symbols_response?.symbols || [];
  }

  private async showSymbolsInNewTab(moduleName: string, symbols: SymbolType[]) {
    const panel = vscode.window.createWebviewPanel(
      "lldb-dap.symbols",
      `Symbols for ${moduleName}`,
      vscode.ViewColumn.Active,
      {
        enableScripts: true,
        localResourceRoots: [
          this.getExtensionResourcePath()
        ]
      }
    );

    let tabulatorJsFilename = "tabulator_simple.min.css";
    if (vscode.window.activeColorTheme.kind === vscode.ColorThemeKind.Dark || vscode.window.activeColorTheme.kind === vscode.ColorThemeKind.HighContrast) {
      tabulatorJsFilename = "tabulator_midnight.min.css";
    }
    const tabulatorCssPath = panel.webview.asWebviewUri(vscode.Uri.joinPath(this.getExtensionResourcePath(), tabulatorJsFilename));
    const tabulatorJsPath = panel.webview.asWebviewUri(vscode.Uri.joinPath(this.getExtensionResourcePath(), "tabulator.min.js"));
    const symbolsTableScriptPath = panel.webview.asWebviewUri(vscode.Uri.joinPath(this.getExtensionResourcePath(), "symbols-table-view.js"));

    panel.webview.html = getSymbolsTableHTMLContent(tabulatorJsPath, tabulatorCssPath, symbolsTableScriptPath);
    panel.webview.postMessage({ command: "updateSymbols", symbols: symbols });
  }

  private getExtensionResourcePath(): vscode.Uri {
    return vscode.Uri.joinPath(this.extensionContext.extensionUri, "out", "webview");
  }
}

class ModuleQuickPickItem implements vscode.QuickPickItem {
  constructor(public readonly module: DebugProtocol.Module) {}

  get label(): string {
    return this.module.name;
  }

  get description(): string {
    return this.module.id.toString();
  }
}
