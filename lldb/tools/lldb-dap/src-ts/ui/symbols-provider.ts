import * as vscode from "vscode";
import { DebugProtocol } from "@vscode/debugprotocol";

import { DebugSessionTracker } from "../debug-session-tracker";
import { DisposableContext } from "../disposable-context";

export class SymbolsProvider extends DisposableContext {
  constructor(private readonly tracker: DebugSessionTracker) {
    super();

    this.pushSubscription(vscode.commands.registerCommand(
      "lldb-dap.modules.showSymbols",
      () => {
        this.SelectModuleAndShowSymbols();
      },
    ));
  }

  static async doesServerSupportSymbolsRequest(session: vscode.DebugSession): Promise<boolean> {
    try {
      const dummyArguments = { _dummy: true };
      const _result = await session.customRequest("dapGetModuleSymbols", dummyArguments);
      return true;
    } catch (_error) {
      return false;
    }
  }

  private async SelectModuleAndShowSymbols() {
    const session = vscode.debug.activeDebugSession;
    if (!session) {
        return;
    }

    if (!await SymbolsProvider.doesServerSupportSymbolsRequest(session)) {
        vscode.window.showErrorMessage("The debug adapter does not support symbol requests.");
        return;
    }

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

    try {
      const symbols = await this.getSymbolsForModule(session, selectedModule.module.id.toString());
      this.showSymbolsForModule(selectedModule.module.name.toString(), symbols);
    } catch (error) {
      if (error instanceof Error) {
        vscode.window.showErrorMessage("Failed to retrieve symbols: " + error.message);
      } else {
        vscode.window.showErrorMessage("Failed to retrieve symbols due to an unknown error.");
      }
      
      return;
    }
  }

  private async getSymbolsForModule(session: vscode.DebugSession, moduleId: string): Promise<DAPSymbol[]> {
    console.log(`Getting symbols for module: ${moduleId}`);
    const symbols_response: { symbols: Array<DAPSymbolType> } = await session.customRequest("dapGetModuleSymbols", { moduleId });


    return symbols_response?.symbols.map(symbol => new DAPSymbol(
      symbol.userId,
      symbol.isDebug,
      symbol.isSynthetic,
      symbol.isExternal,
      symbol.type,
      symbol.fileAddress,
      symbol.loadAddress,
      symbol.size,
      symbol.name,
    )) || [];
  }

  private async showSymbolsForModule(moduleName: string, symbols: DAPSymbol[]) {
    const panel = vscode.window.createWebviewPanel(
      "lldb-dap.symbols",
      `Symbols for ${moduleName}`,
      vscode.ViewColumn.Active,
      {}
    );

    panel.webview.html = SymbolsProvider.getHTMLContentForSymbols(moduleName, symbols);
  }

  private static getHTMLContentForSymbols(moduleName: string, symbols: DAPSymbol[]): string {
    const symbolLines = symbols.map(symbol => ` - ${symbol.name} (${symbol.type})`);
    return `Symbols for module: ${moduleName}\n${symbolLines.join("\n")}`;
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

/// The symbol type we get from the lldb-dap server
type DAPSymbolType = {
  userId: number;
  isDebug: boolean;
  isSynthetic: boolean;
  isExternal: boolean;
  type: string;
  fileAddress: number;
  loadAddress?: number;
  size: number;
  name: string;
};

class DAPSymbol {
  constructor(
    public readonly userId: number,
    public readonly isDebug: boolean,
    public readonly isSynthetic: boolean,
    public readonly isExternal: boolean,
    public readonly type: string,
    public readonly fileAddress: number,
    public readonly loadAddress: number | undefined,
    public readonly size: number,
    public readonly name: string,
  ) {}
}
