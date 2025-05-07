import { DebugProtocol } from "@vscode/debugprotocol";
import * as vscode from "vscode";

interface EventMap {
  module: DebugProtocol.ModuleEvent;
}

function isEvent(
  message: DebugProtocol.ProtocolMessage,
): message is DebugProtocol.Event;
function isEvent<K extends keyof EventMap>(
  message: DebugProtocol.ProtocolMessage,
  event: K,
): message is EventMap[K];
function isEvent(
  message: DebugProtocol.ProtocolMessage,
  event?: string,
): boolean {
  return (
    message.type === "event" &&
    (!event || (message as DebugProtocol.Event).event === event)
  );
}

export class DebugSessionTracker
  implements vscode.DebugAdapterTrackerFactory, vscode.Disposable
{
  private modules = new Map<vscode.DebugSession, DebugProtocol.Module[]>();
  private modulesChanged = new vscode.EventEmitter<void>();
  onDidChangeModules: vscode.Event<void> = this.modulesChanged.event;

  dispose() {
    this.modules.clear();
    this.modulesChanged.dispose();
  }

  createDebugAdapterTracker(
    session: vscode.DebugSession,
  ): vscode.ProviderResult<vscode.DebugAdapterTracker> {
    return {
      onDidSendMessage: (message) => this.onDidSendMessage(session, message),
      onExit: () => this.onExit(session),
    };
  }

  debugSessionModules(session: vscode.DebugSession): DebugProtocol.Module[] {
    return this.modules.get(session) ?? [];
  }

  private onExit(session: vscode.DebugSession) {
    this.modules.delete(session);
  }

  private onDidSendMessage(
    session: vscode.DebugSession,
    message: DebugProtocol.ProtocolMessage,
  ) {
    if (isEvent(message, "module")) {
      const { module, reason } = message.body;
      const modules = this.modules.get(session) ?? [];
      switch (reason) {
        case "new":
        case "changed": {
          const index = modules.findIndex((m) => m.id === module.id);
          if (index !== -1) {
            modules[index] = module;
          } else {
            modules.push(module);
          }
          break;
        }
        case "removed": {
          const index = modules.findIndex((m) => m.id === module.id);
          if (index !== -1) {
            modules.splice(index, 1);
          }
          break;
        }
        default:
          console.error("unexpected module event reason");
          break;
      }
      this.modules.set(session, modules);
      this.modulesChanged.fire();
    }
  }
}
