import * as vscode from "vscode";
import { DebugProtocol } from "@vscode/debugprotocol";
import { DebugSessionTracker } from "../debug-session-tracker";

export class ModuleDataProvider
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
    tooltip.appendMarkdown(`# Module '${module.name}'\n\n`);
    tooltip.appendMarkdown(`- **id** : ${module.id}\n`);
    if (module.addressRange) {
      tooltip.appendMarkdown(`- **load address** : ${module.addressRange}\n`);
    }
    if (module.path) {
      tooltip.appendMarkdown(`- **path** : ${module.path}\n`);
    }
    if (module.version) {
      tooltip.appendMarkdown(`- **version** : ${module.version}\n`);
    }
    if (module.symbolStatus) {
      tooltip.appendMarkdown(`- **symbol status** : ${module.symbolStatus}\n`);
    }
    if (module.symbolFilePath) {
      tooltip.appendMarkdown(
        `- **symbol file path** : ${module.symbolFilePath}\n`,
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
