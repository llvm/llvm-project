import * as vscode from "vscode";
import { DebugProtocol } from "@vscode/debugprotocol";
import { DebugSessionTracker } from "../debug-session-tracker";

/** A tree data provider for listing loaded modules for the active debug session. */
export class ModulesDataProvider
  implements vscode.TreeDataProvider<DebugProtocol.Module>
{
  private changeTreeData = new vscode.EventEmitter<void>();
  readonly onDidChangeTreeData = this.changeTreeData.event;

  constructor(private readonly tracker: DebugSessionTracker) {
    tracker.onDidChangeModules(() => this.changeTreeData.fire());
    vscode.debug.onDidChangeActiveDebugSession(() =>
      this.changeTreeData.fire(),
    );
  }

  getTreeItem(module: DebugProtocol.Module): vscode.TreeItem {
    let treeItem = new vscode.TreeItem(/*label=*/ module.name);
    if (module.path) {
      treeItem.description = `${module.id} -- ${module.path}`;
    } else {
      treeItem.description = `${module.id}`;
    }

    const tooltip = new vscode.MarkdownString();
    tooltip.appendMarkdown(`# ${module.name}\n\n`);
    tooltip.appendMarkdown(`- **ID**: ${module.id}\n`);
    if (module.addressRange) {
      tooltip.appendMarkdown(
        `- **Load address**: 0x${Number(module.addressRange).toString(16)}\n`,
      );
    }
    if (module.path) {
      tooltip.appendMarkdown(`- **Path**: ${module.path}\n`);
    }
    if (module.version) {
      tooltip.appendMarkdown(`- **Version**: ${module.version}\n`);
    }
    if (module.symbolStatus) {
      tooltip.appendMarkdown(`- **Symbol status**: ${module.symbolStatus}\n`);
    }
    if (module.symbolFilePath) {
      tooltip.appendMarkdown(
        `- **Symbol file path**: ${module.symbolFilePath}\n`,
      );
    }

    treeItem.tooltip = tooltip;
    return treeItem;
  }

  getChildren(): DebugProtocol.Module[] {
    if (!vscode.debug.activeDebugSession) {
      return [];
    }

    return this.tracker.debugSessionModules(vscode.debug.activeDebugSession);
  }
}
