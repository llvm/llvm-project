import * as vscode from "vscode";
import { DebugProtocol } from "@vscode/debugprotocol";
import { DebugSessionTracker } from "../debug-session-tracker";

export interface ModuleProperty {
  key: string;
  value: string;
}

/** Type to represent both Module and ModuleProperty since TreeDataProvider
 * expects one concrete type */
type TreeData = DebugProtocol.Module | ModuleProperty;

function isModule(type: TreeData): type is DebugProtocol.Module {
  return (type as DebugProtocol.Module).id !== undefined;
}

class ModuleItem extends vscode.TreeItem {
  constructor(module: DebugProtocol.Module) {
    super(module.name, vscode.TreeItemCollapsibleState.Collapsed);
    this.description = module.symbolStatus;
  }

  static getProperties(module: DebugProtocol.Module): ModuleProperty[] {
    // does not include the name and symbol status as it is show in the parent.
    let children: ModuleProperty[] = [];
    children.push({ key: "id:", value: module.id.toString() });

    if (module.addressRange) {
      children.push({
        key: "load address:",
        value: module.addressRange,
      });
    }
    if (module.path) {
      children.push({ key: "path:", value: module.path });
    }
    if (module.version) {
      children.push({ key: "version:", value: module.version });
    }
    if (module.symbolFilePath) {
      children.push({ key: "symbol filepath:", value: module.symbolFilePath });
    }
    return children;
  }
}

/** A tree data provider for listing loaded modules for the active debug session. */
export class ModulesDataProvider implements vscode.TreeDataProvider<TreeData> {
  private changeTreeData = new vscode.EventEmitter<void>();
  readonly onDidChangeTreeData = this.changeTreeData.event;

  constructor(private readonly tracker: DebugSessionTracker) {
    tracker.onDidChangeModules(() => this.changeTreeData.fire());
  }

  getTreeItem(module: TreeData): vscode.TreeItem {
    if (isModule(module)) {
      return new ModuleItem(module);
    }

    let item = new vscode.TreeItem(module.key);
    item.description = module.value;
    item.tooltip = `${module.key} ${module.value}`;
    item.contextValue = "property";
    return item;
  }

  getChildren(element?: TreeData): TreeData[] {
    if (!vscode.debug.activeDebugSession) {
      return [];
    }

    if (!element) {
      return this.tracker.debugSessionModules(vscode.debug.activeDebugSession);
    }

    if (isModule(element)) {
      return ModuleItem.getProperties(element);
    }

    return [];
  }
}
