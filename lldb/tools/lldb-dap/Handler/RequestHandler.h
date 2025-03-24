//===-- Request.h ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_HANDLER_HANDLER_H
#define LLDB_TOOLS_LLDB_DAP_HANDLER_HANDLER_H

#include "DAP.h"
#include "DAPError.h"
#include "DAPLog.h"
#include "Protocol/ProtocolBase.h"
#include "Protocol/ProtocolRequests.h"
#include "lldb/API/SBError.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include <optional>
#include <type_traits>

template <typename T> struct is_optional : std::false_type {};

template <typename T> struct is_optional<std::optional<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_optional_v = is_optional<T>::value;

namespace lldb_dap {
struct DAP;

/// Base class for request handlers. Do not extend this directly: Extend
/// the RequestHandler template subclass instead.
class BaseRequestHandler {
public:
  BaseRequestHandler(DAP &dap) : dap(dap) {}

  /// BaseRequestHandler are not copyable.
  /// @{
  BaseRequestHandler(const BaseRequestHandler &) = delete;
  BaseRequestHandler &operator=(const BaseRequestHandler &) = delete;
  /// @}

  virtual ~BaseRequestHandler() = default;

  virtual void operator()(const protocol::Request &request) const = 0;

  virtual llvm::StringMap<bool> GetCapabilities() const { return {}; }

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

/// FIXME: Migrate callers to typed RequestHandler for improved type handling.
class LegacyRequestHandler : public BaseRequestHandler {
  using BaseRequestHandler::BaseRequestHandler;
  virtual void operator()(const llvm::json::Object &request) const = 0;
  void operator()(const protocol::Request &request) const override {
    auto req = toJSON(request);
    (*this)(*req.getAsObject());
  }
};

/// Base class for handling DAP requests. Handlers should declare their
/// arguments and response body types like:
///
/// class MyRequestHandler : public RequestHandler<Arguments, ResponseBody> {
///   ....
/// };
template <typename Args, typename Body>
class RequestHandler : public BaseRequestHandler {
  using BaseRequestHandler::BaseRequestHandler;

  void operator()(const protocol::Request &request) const override {
    protocol::Response response;
    response.request_seq = request.seq;
    response.command = request.command;

    if (!is_optional_v<Args> && !request.arguments) {
      DAP_LOG(dap.log,
              "({0}) malformed request {1}, expected arguments but got none",
              dap.transport.GetClientName(), request.command);
      response.success = false;
      response.message = llvm::formatv("arguments required for command '{0}' "
                                       "but none received",
                                       request.command)
                             .str();
      dap.Send(response);
      return;
    }

    Args arguments;
    llvm::json::Path::Root root;
    if (request.arguments && !fromJSON(request.arguments, arguments, root)) {
      std::string parse_failure;
      llvm::raw_string_ostream OS(parse_failure);
      root.printErrorContext(request.arguments, OS);

      protocol::ErrorMessage error_message;
      error_message.format = parse_failure;

      protocol::ErrorResponseBody body;
      body.error = error_message;

      response.success = false;
      response.body = std::move(body);

      dap.Send(response);
      return;
    }

    llvm::Expected<Body> body = Run(arguments);
    if (auto Err = body.takeError()) {
      protocol::ErrorMessage error_message;
      error_message.sendTelemetry = false;
      if (llvm::Error unhandled = llvm::handleErrors(
              std::move(Err), [&](const DAPError &E) -> llvm::Error {
                error_message.format = E.getMessage();
                error_message.showUser = E.getShowUser();
                error_message.id = E.convertToErrorCode().value();
                error_message.url = E.getURL();
                error_message.urlLabel = E.getURLLabel();
                return llvm::Error::success();
              }))
        error_message.format = llvm::toString(std::move(unhandled));
      protocol::ErrorResponseBody body;
      body.error = error_message;
      response.success = false;
      response.body = std::move(body);
    } else {
      response.success = true;
      if constexpr (!std::is_same_v<Body, std::monostate>)
        response.body = std::move(*body);
    }

    dap.Send(response);
  };

  virtual llvm::Expected<Body> Run(const Args &) const = 0;
};

class AttachRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "attach"; }
  void operator()(const llvm::json::Object &request) const override;
};

class BreakpointLocationsRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "breakpointLocations"; }
  llvm::StringMap<bool> GetCapabilities() const override {
    return {{"supportsBreakpointLocationsRequest", true}};
  }
  void operator()(const llvm::json::Object &request) const override;
};

class CompletionsRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "completions"; }
  llvm::StringMap<bool> GetCapabilities() const override {
    return {{"supportsCompletionsRequest", true}};
  }
  void operator()(const llvm::json::Object &request) const override;
};

class ContinueRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "continue"; }
  void operator()(const llvm::json::Object &request) const override;
};

class ConfigurationDoneRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "configurationDone"; }
  llvm::StringMap<bool> GetCapabilities() const override {
    return {{"supportsConfigurationDoneRequest", true}};
  }
  void operator()(const llvm::json::Object &request) const override;
};

class DisconnectRequestHandler
    : public RequestHandler<std::optional<protocol::DisconnectArguments>,
                            protocol::DisconnectResponse> {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral GetCommand() { return "disconnect"; }
  llvm::Expected<protocol::DisconnectResponse>
  Run(const std::optional<protocol::DisconnectArguments> &args) const override;
};

class EvaluateRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "evaluate"; }
  void operator()(const llvm::json::Object &request) const override;
};

class ExceptionInfoRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "exceptionInfo"; }
  llvm::StringMap<bool> GetCapabilities() const override {
    return {{"supportsExceptionInfoRequest", true}};
  }
  void operator()(const llvm::json::Object &request) const override;
};

class InitializeRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "initialize"; }
  llvm::StringMap<bool> GetCapabilities() const override {
    return {{"supportsRunInTerminalRequest", true}};
  }
  void operator()(const llvm::json::Object &request) const override;
};

class LaunchRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "launch"; }
  void operator()(const llvm::json::Object &request) const override;
};

class RestartRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "restart"; }
  llvm::StringMap<bool> GetCapabilities() const override {
    return {{"supportsRestartRequest", true}};
  }
  void operator()(const llvm::json::Object &request) const override;
};

class NextRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "next"; }
  void operator()(const llvm::json::Object &request) const override;
};

class StepInRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "stepIn"; }
  void operator()(const llvm::json::Object &request) const override;
};

class StepInTargetsRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "stepInTargets"; }
  llvm::StringMap<bool> GetCapabilities() const override {
    return {{"supportsStepInTargetsRequest", true}};
  }
  void operator()(const llvm::json::Object &request) const override;
};

class StepOutRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "stepOut"; }
  void operator()(const llvm::json::Object &request) const override;
};

class SetBreakpointsRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "setBreakpoints"; }
  llvm::StringMap<bool> GetCapabilities() const override {
    return {{"supportsConditionalBreakpoints", true},
            {"supportsHitConditionalBreakpoints", true}};
  }
  void operator()(const llvm::json::Object &request) const override;
};

class SetExceptionBreakpointsRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "setExceptionBreakpoints"; }
  void operator()(const llvm::json::Object &request) const override;
};

class SetFunctionBreakpointsRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "setFunctionBreakpoints"; }
  llvm::StringMap<bool> GetCapabilities() const override {
    return {{"supportsFunctionBreakpoints", true}};
  }
  void operator()(const llvm::json::Object &request) const override;
};

class DataBreakpointInfoRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "dataBreakpointInfo"; }
  void operator()(const llvm::json::Object &request) const override;
};

class SetDataBreakpointsRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "setDataBreakpoints"; }
  void operator()(const llvm::json::Object &request) const override;
};

class SetInstructionBreakpointsRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() {
    return "setInstructionBreakpoints";
  }
  void operator()(const llvm::json::Object &request) const override;
};

class CompileUnitsRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "compileUnits"; }
  void operator()(const llvm::json::Object &request) const override;
};

class ModulesRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "modules"; }
  llvm::StringMap<bool> GetCapabilities() const override {
    return {{"supportsModulesRequest", true}};
  }
  void operator()(const llvm::json::Object &request) const override;
};

class PauseRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "pause"; }
  void operator()(const llvm::json::Object &request) const override;
};

class ScopesRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "scopes"; }
  void operator()(const llvm::json::Object &request) const override;
};

class SetVariableRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "setVariable"; }
  llvm::StringMap<bool> GetCapabilities() const override {
    return {{"supportsSetVariable", true}};
  }
  void operator()(const llvm::json::Object &request) const override;
};

class SourceRequestHandler
    : public RequestHandler<protocol::SourceArguments,
                            protocol::SourceResponseBody> {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral GetCommand() { return "source"; }
  llvm::Expected<protocol::SourceResponseBody>
  Run(const protocol::SourceArguments &args) const override;
};

class StackTraceRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "stackTrace"; }
  void operator()(const llvm::json::Object &request) const override;
};

class ThreadsRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "threads"; }
  void operator()(const llvm::json::Object &request) const override;
};

class VariablesRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "variables"; }
  void operator()(const llvm::json::Object &request) const override;
};

class LocationsRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "locations"; }
  void operator()(const llvm::json::Object &request) const override;
};

class DisassembleRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "disassemble"; }
  llvm::StringMap<bool> GetCapabilities() const override {
    return {{"supportsDisassembleRequest", true}};
  }
  void operator()(const llvm::json::Object &request) const override;
};

class ReadMemoryRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "readMemory"; }
  llvm::StringMap<bool> GetCapabilities() const override {
    return {{"supportsReadMemoryRequest", true}};
  }
  void operator()(const llvm::json::Object &request) const override;
};

/// A request used in testing to get the details on all breakpoints that are
/// currently set in the target. This helps us to test "setBreakpoints" and
/// "setFunctionBreakpoints" requests to verify we have the correct set of
/// breakpoints currently set in LLDB.
class TestGetTargetBreakpointsRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() {
    return "_testGetTargetBreakpoints";
  }
  void operator()(const llvm::json::Object &request) const override;
};

} // namespace lldb_dap

#endif
