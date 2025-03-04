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

  virtual void operator()(const llvm::json::Object &request) const = 0;

protected:
  /// Helpers used by multiple request handlers.
  /// FIXME: Move these into the DAP class?
  /// @{

  /// Both attach and launch take a either a sourcePath or sourceMap
  /// argument (or neither), from which we need to set the target.source-map.
  void SetSourceMapFromArguments(const llvm::json::Object &arguments) const;

  /// Prints a welcome message on the editor if the preprocessor variable
  /// LLDB_DAP_WELCOME_MESSAGE is defined.
  void PrintWelcomeMessage() const;

  // Takes a LaunchRequest object and launches the process, also handling
  // runInTerminal if applicable. It doesn't do any of the additional
  // initialization and bookkeeping stuff that is needed for `request_launch`.
  // This way we can reuse the process launching logic for RestartRequest too.
  lldb::SBError LaunchProcess(const llvm::json::Object &request) const;

  // Check if the step-granularity is `instruction`.
  bool HasInstructionGranularity(const llvm::json::Object &request) const;

  /// @}

  DAP &dap;
};

class AttachRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "attach"; }
  void operator()(const llvm::json::Object &request) const override;
};

class BreakpointLocationsRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "breakpointLocations"; }
  void operator()(const llvm::json::Object &request) const override;
};

class CompletionsRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "completions"; }
  void operator()(const llvm::json::Object &request) const override;
};

class ContinueRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "continue"; }
  void operator()(const llvm::json::Object &request) const override;
};

class ConfigurationDoneRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "configurationDone"; }
  void operator()(const llvm::json::Object &request) const override;
};

class DisconnectRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "disconnect"; }
  void operator()(const llvm::json::Object &request) const override;
};

class EvaluateRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "evaluate"; }
  void operator()(const llvm::json::Object &request) const override;
};

class ExceptionInfoRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "exceptionInfo"; }
  void operator()(const llvm::json::Object &request) const override;
};

class InitializeRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "initialize"; }
  void operator()(const llvm::json::Object &request) const override;
};

class LaunchRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "launch"; }
  void operator()(const llvm::json::Object &request) const override;
};

class RestartRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "restart"; }
  void operator()(const llvm::json::Object &request) const override;
};

class NextRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "next"; }
  void operator()(const llvm::json::Object &request) const override;
};

class StepInRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "stepIn"; }
  void operator()(const llvm::json::Object &request) const override;
};

class StepInTargetsRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "stepInTargets"; }
  void operator()(const llvm::json::Object &request) const override;
};

class StepOutRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "stepOut"; }
  void operator()(const llvm::json::Object &request) const override;
};

class SetBreakpointsRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "setBreakpoints"; }
  void operator()(const llvm::json::Object &request) const override;
};

class SetExceptionBreakpointsRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "setExceptionBreakpoints"; }
  void operator()(const llvm::json::Object &request) const override;
};

class SetFunctionBreakpointsRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "setFunctionBreakpoints"; }
  void operator()(const llvm::json::Object &request) const override;
};

class DataBreakpointInfoRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "dataBreakpointInfo"; }
  void operator()(const llvm::json::Object &request) const override;
};

class SetDataBreakpointsRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "setDataBreakpoints"; }
  void operator()(const llvm::json::Object &request) const override;
};

class SetInstructionBreakpointsRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() {
    return "setInstructionBreakpoints";
  }
  void operator()(const llvm::json::Object &request) const override;
};

class CompileUnitsRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "compileUnits"; }
  void operator()(const llvm::json::Object &request) const override;
};

class ModulesRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "modules"; }
  void operator()(const llvm::json::Object &request) const override;
};

class PauseRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "pause"; }
  void operator()(const llvm::json::Object &request) const override;
};

class ScopesRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "scopes"; }
  void operator()(const llvm::json::Object &request) const override;
};

class SetVariableRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "setVariable"; }
  void operator()(const llvm::json::Object &request) const override;
};

class SourceRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "source"; }
  void operator()(const llvm::json::Object &request) const override;
};

class StackTraceRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "stackTrace"; }
  void operator()(const llvm::json::Object &request) const override;
};

class ThreadsRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "threads"; }
  void operator()(const llvm::json::Object &request) const override;
};

class VariablesRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "variables"; }
  void operator()(const llvm::json::Object &request) const override;
};

class LocationsRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "locations"; }
  void operator()(const llvm::json::Object &request) const override;
};

class DisassembleRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "disassemble"; }
  void operator()(const llvm::json::Object &request) const override;
};

class ReadMemoryRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "readMemory"; }
  void operator()(const llvm::json::Object &request) const override;
};

/// A request used in testing to get the details on all breakpoints that are
/// currently set in the target. This helps us to test "setBreakpoints" and
/// "setFunctionBreakpoints" requests to verify we have the correct set of
/// breakpoints currently set in LLDB.
class TestGetTargetBreakpointsRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() {
    return "_testGetTargetBreakpoints";
  }
  void operator()(const llvm::json::Object &request) const override;
};

} // namespace lldb_dap

#endif
