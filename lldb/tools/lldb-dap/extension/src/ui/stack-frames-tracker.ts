import { DebugProtocol as dap } from "@vscode/debugprotocol";
import * as vscode from "vscode";

const OPTIMIZED_SUFFIX = " [opt]";

interface LLDBStackFrame extends dap.StackFrame {
  $__lldb_isOptimized?: boolean;
}

export function registerStackFramesTracker(context: vscode.ExtensionContext) {
  const tracker = vscode.debug.registerDebugAdapterTrackerFactory(
    "lldb-dap",
    new StackFramesTrackerFactory(),
  );
  context.subscriptions.push(tracker);
}

class StackFramesTrackerFactory implements vscode.DebugAdapterTrackerFactory {
  createDebugAdapterTracker(
    session: vscode.DebugSession,
  ): vscode.DebugAdapterTracker | undefined {
    // Don't change the user specified frame format.
    if (session.configuration.customFrameFormat) {
      return undefined;
    }
    return new OptimizedFramesTracker();
  }
}

class OptimizedFramesTracker implements vscode.DebugAdapterTracker {
  // The `seq` of `stackTrace` requests that specified a format.
  private readonly formattedRequests = new Set<number>();

  onWillReceiveMessage(message: dap.ProtocolMessage) {
    if (isStackTraceRequest(message) && message.arguments?.format) {
      this.formattedRequests.add(message.seq);
    }
  }

  onDidSendMessage(message: dap.ProtocolMessage) {
    if (!isStackTraceResponse(message)) {
      return;
    }

    // Skip formatted stackTrace and error responses.
    const hadFormat = this.formattedRequests.delete(message.request_seq);
    if (hadFormat || !message.success) {
      return;
    }

    const frames = (message.body.stackFrames ?? []) as LLDBStackFrame[];
    for (const frame of frames) {
      if (
        frame.$__lldb_isOptimized === true &&
        !frame.name.endsWith(OPTIMIZED_SUFFIX)
      ) {
        frame.name += OPTIMIZED_SUFFIX;
      }
    }
  }
}

function isStackTraceRequest(
  message: dap.ProtocolMessage,
): message is dap.StackTraceRequest {
  return (
    message.type === "request" &&
    (message as dap.Request).command === "stackTrace"
  );
}

function isStackTraceResponse(
  message: dap.ProtocolMessage,
): message is dap.StackTraceResponse {
  return (
    message.type === "response" &&
    (message as dap.Response).command === "stackTrace"
  );
}
