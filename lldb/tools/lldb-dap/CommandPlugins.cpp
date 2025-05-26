//===-- CommandPlugins.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CommandPlugins.h"
#include "Handler/ResponseHandler.h"
#include "JSONUtils.h"
#include "lldb/API/SBStream.h"

using namespace lldb_dap;

bool StartDebuggingCommand::DoExecute(lldb::SBDebugger debugger, char **command,
                                      lldb::SBCommandReturnObject &result) {
  // Command format like: `start-debugging <launch|attach> <configuration>`
  if (!command) {
    result.SetError("Invalid use of start-debugging, expected format "
                    "`start-debugging <launch|attach> <configuration>`.");
    return false;
  }

  if (!command[0] || llvm::StringRef(command[0]).empty()) {
    result.SetError("start-debugging request type missing.");
    return false;
  }

  if (!command[1] || llvm::StringRef(command[1]).empty()) {
    result.SetError("start-debugging debug configuration missing.");
    return false;
  }

  llvm::StringRef request{command[0]};
  std::string raw_configuration{command[1]};

  llvm::Expected<llvm::json::Value> configuration =
      llvm::json::parse(raw_configuration);

  if (!configuration) {
    llvm::Error err = configuration.takeError();
    std::string msg = "Failed to parse json configuration: " +
                      llvm::toString(std::move(err)) + "\n\n" +
                      raw_configuration;
    result.SetError(msg.c_str());
    return false;
  }

  dap.SendReverseRequest<LogFailureResponseHandler>(
      "startDebugging",
      llvm::json::Object{{"request", request},
                         {"configuration", std::move(*configuration)}});

  result.SetStatus(lldb::eReturnStatusSuccessFinishNoResult);

  return true;
}

bool ReplModeCommand::DoExecute(lldb::SBDebugger debugger, char **command,
                                lldb::SBCommandReturnObject &result) {
  // Command format like: `repl-mode <variable|command|auto>?`
  // If a new mode is not specified report the current mode.
  if (!command || llvm::StringRef(command[0]).empty()) {
    std::string mode;
    switch (dap.repl_mode) {
    case ReplMode::Variable:
      mode = "variable";
      break;
    case ReplMode::Command:
      mode = "command";
      break;
    case ReplMode::Auto:
      mode = "auto";
      break;
    }

    result.Printf("lldb-dap repl-mode %s.\n", mode.c_str());
    result.SetStatus(lldb::eReturnStatusSuccessFinishResult);

    return true;
  }

  llvm::StringRef new_mode{command[0]};

  if (new_mode == "variable") {
    dap.repl_mode = ReplMode::Variable;
  } else if (new_mode == "command") {
    dap.repl_mode = ReplMode::Command;
  } else if (new_mode == "auto") {
    dap.repl_mode = ReplMode::Auto;
  } else {
    lldb::SBStream error_message;
    error_message.Printf("Invalid repl-mode '%s'. Expected one of 'variable', "
                         "'command' or 'auto'.\n",
                         new_mode.data());
    result.SetError(error_message.GetData());
    return false;
  }

  result.Printf("lldb-dap repl-mode %s set.\n", new_mode.data());
  result.SetStatus(lldb::eReturnStatusSuccessFinishNoResult);
  return true;
}

/// Sends a DAP event with an optional body.
///
/// https://code.visualstudio.com/api/references/vscode-api#debug.onDidReceiveDebugSessionCustomEvent
bool SendEventCommand::DoExecute(lldb::SBDebugger debugger, char **command,
                                 lldb::SBCommandReturnObject &result) {
  // Command format like: `send-event <name> <body>?`
  if (!command || !command[0] || llvm::StringRef(command[0]).empty()) {
    result.SetError("Not enough arguments found, expected format "
                    "`lldb-dap send-event <name> <body>?`.");
    return false;
  }

  llvm::StringRef name{command[0]};
  // Events that are stateful and should be handled by lldb-dap internally.
  const std::array internal_events{"breakpoint", "capabilities", "continued",
                                   "exited",     "initialize",   "loadedSource",
                                   "module",     "process",      "stopped",
                                   "terminated", "thread"};
  if (llvm::is_contained(internal_events, name)) {
    std::string msg =
        llvm::formatv("Invalid use of lldb-dap send-event, event \"{0}\" "
                      "should be handled by lldb-dap internally.",
                      name)
            .str();
    result.SetError(msg.c_str());
    return false;
  }

  llvm::json::Object event(CreateEventObject(name));

  if (command[1] && !llvm::StringRef(command[1]).empty()) {
    // See if we have unused arguments.
    if (command[2]) {
      result.SetError(
          "Additional arguments found, expected `lldb-dap send-event "
          "<name> <body>?`.");
      return false;
    }

    llvm::StringRef raw_body{command[1]};

    llvm::Expected<llvm::json::Value> body = llvm::json::parse(raw_body);

    if (!body) {
      llvm::Error err = body.takeError();
      std::string msg = "Failed to parse custom event body: " +
                        llvm::toString(std::move(err));
      result.SetError(msg.c_str());
      return false;
    }

    event.try_emplace("body", std::move(*body));
  }

  dap.SendJSON(llvm::json::Value(std::move(event)));
  result.SetStatus(lldb::eReturnStatusSuccessFinishNoResult);
  return true;
}
