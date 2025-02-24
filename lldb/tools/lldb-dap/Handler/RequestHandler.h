//===-- Request.h ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_HANDLER_HANDLER_H
#define LLDB_TOOLS_LLDB_DAP_HANDLER_HANDLER_H

#include "lldb/API/SBError.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"

namespace lldb_dap {
struct DAP;

class RequestHandler {
public:
  RequestHandler(DAP &dap) : dap(dap) {}

  /// RequestHandler are not copyable.
  /// @{
  RequestHandler(const RequestHandler &) = delete;
  RequestHandler &operator=(const RequestHandler &) = delete;
  /// @}

  virtual ~RequestHandler() = default;

  virtual void operator()(const llvm::json::Object &request) = 0;

protected:
  /// Helpers used by multiple request handlers.
  /// FIXME: Move these into the DAP class?
  /// @{

  /// Both attach and launch take a either a sourcePath or sourceMap
  /// argument (or neither), from which we need to set the target.source-map.
  void SetSourceMapFromArguments(const llvm::json::Object &arguments);

  /// Prints a welcome message on the editor if the preprocessor variable
  /// LLDB_DAP_WELCOME_MESSAGE is defined.
  void PrintWelcomeMessage();

  // Takes a LaunchRequest object and launches the process, also handling
  // runInTerminal if applicable. It doesn't do any of the additional
  // initialization and bookkeeping stuff that is needed for `request_launch`.
  // This way we can reuse the process launching logic for RestartRequest too.
  lldb::SBError LaunchProcess(const llvm::json::Object &request);

  // Check if the step-granularity is `instruction`.
  bool HasInstructionGranularity(const llvm::json::Object &request);

  /// @}

  DAP &dap;
};

class AttachRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "attach"; }
  void operator()(const llvm::json::Object &request) override;
};

class BreakpointLocationsRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "breakpointLocations"; }
  void operator()(const llvm::json::Object &request) override;
};

class CompletionsRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "completions"; }
  void operator()(const llvm::json::Object &request) override;
};

class ContinueRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "continue"; }
  void operator()(const llvm::json::Object &request) override;
};

class ConfigurationDoneRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "configurationDone"; }
  void operator()(const llvm::json::Object &request) override;
};

class DisconnectRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "disconnect"; }
  void operator()(const llvm::json::Object &request) override;
};

class EvaluateRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "evaluate"; }
  void operator()(const llvm::json::Object &request) override;
};

class ExceptionInfoRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "exceptionInfo"; }
  void operator()(const llvm::json::Object &request) override;
};

class InitializeRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "initialize"; }
  void operator()(const llvm::json::Object &request) override;
};

class LaunchRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "launch"; }
  void operator()(const llvm::json::Object &request) override;
};

class RestartRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "restart"; }
  void operator()(const llvm::json::Object &request) override;
};

class NextRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "next"; }
  void operator()(const llvm::json::Object &request) override;
};

class StepInRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "stepIn"; }
  void operator()(const llvm::json::Object &request) override;
};

class StepInTargetsRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "stepInTargets"; }
  void operator()(const llvm::json::Object &request) override;
};

class StepOutRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "stepOut"; }
  void operator()(const llvm::json::Object &request) override;
};

} // namespace lldb_dap

#endif
