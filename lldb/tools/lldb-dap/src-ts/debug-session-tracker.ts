import { DebugProtocol } from "@vscode/debugprotocol";
import * as vscode from "vscode";

/** A helper type for mapping event types to their corresponding data type. */
// prettier-ignore
interface EventMap {
  "module": DebugProtocol.ModuleEvent;
  "exited": DebugProtocol.ExitedEvent;
}

/** A type assertion to check if a ProtocolMessage is an event or if it is a specific event. */
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

/** Tracks lldb-dap sessions for data visualizers. */
export class DebugSessionTracker
  implements vscode.DebugAdapterTrackerFactory, vscode.Disposable
{
  /**
   * Tracks active modules for each debug sessions.
   *
   * The modules are kept in an array to maintain the load order of the modules.
   */
  private modules = new Map<vscode.DebugSession, DebugProtocol.Module[]>();
  private modulesChanged = new vscode.EventEmitter<
    vscode.DebugSession | undefined
  >();

  /**
   * Fired when modules are changed for any active debug session.
   *
   * Use `debugSessionModules` to retieve the active modules for a given debug session.
   */
  onDidChangeModules: vscode.Event<vscode.DebugSession | undefined> =
    this.modulesChanged.event;

  constructor(private logger: vscode.LogOutputChannel) {
    this.onDidChangeModules(this.moduleChangedListener, this);
    vscode.debug.onDidChangeActiveDebugSession((session) =>
      this.modulesChanged.fire(session),
    );
  }

  dispose() {
    this.modules.clear();
    this.modulesChanged.dispose();
  }

  createDebugAdapterTracker(
    session: vscode.DebugSession,
  ): vscode.ProviderResult<vscode.DebugAdapterTracker> {
    this.logger.info(`Starting debug session "${session.name}"`);
    let stopping = false;
    return {
      onError: (error) => !stopping && this.logger.error(error), // Can throw benign read errors when shutting down.
      onDidSendMessage: (message) => this.onDidSendMessage(session, message),
      onWillStopSession: () => (stopping = true),
      onExit: () => this.onExit(session),
    };
  }

  /**
   * Retrieves the modules for the given debug session.
   *
   * Modules are returned in load order.
   */
  debugSessionModules(session: vscode.DebugSession): DebugProtocol.Module[] {
    return this.modules.get(session) ?? [];
  }

  /** Clear information from the active session. */
  private onExit(session: vscode.DebugSession) {
    this.modules.delete(session);
    this.modulesChanged.fire(undefined);
  }

  private showModulesTreeView(showModules: boolean) {
    vscode.commands.executeCommand(
      "setContext",
      "lldb-dap.showModules",
      showModules,
    );
  }

  private moduleChangedListener(session: vscode.DebugSession | undefined) {
    if (!session) {
      this.showModulesTreeView(false);
      return;
    }

    if (session == vscode.debug.activeDebugSession) {
      const sessionHasModules = this.modules.get(session) != undefined;
      this.showModulesTreeView(sessionHasModules);
    }
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
      this.modulesChanged.fire(session);
    } else if (isEvent(message, "exited")) {
      // The vscode.DebugAdapterTracker#onExit event is sometimes called with
      // exitCode = undefined but the exit event from LLDB-DAP always has the "exitCode"
      const { exitCode } = message.body;
      this.logger.info(
        `Session "${session.name}" exited with code ${exitCode}`,
      );
    }
  }
}
