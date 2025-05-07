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
#include "Protocol/ProtocolTypes.h"
#include "lldb/API/SBError.h"
#include "llvm/ADT/DenseSet.h"
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

  void Run(const protocol::Request &);

  virtual void operator()(const protocol::Request &request) const = 0;

  using FeatureSet = llvm::SmallDenseSet<AdapterFeature, 1>;
  virtual FeatureSet GetSupportedFeatures() const { return {}; }

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
  llvm::Error
  LaunchProcess(const protocol::LaunchRequestArguments &request) const;

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
/// class MyRequestHandler : public RequestHandler<Arguments, Response> {
///   ....
/// };
template <typename Args, typename Resp>
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
    llvm::json::Path::Root root("arguments");
    if (request.arguments && !fromJSON(*request.arguments, arguments, root)) {
      std::string parse_failure;
      llvm::raw_string_ostream OS(parse_failure);
      OS << "invalid arguments for request '" << request.command
         << "': " << llvm::toString(root.getError()) << "\n";
      root.printErrorContext(*request.arguments, OS);

      response.success = false;
      response.body = ToResponse(llvm::make_error<DAPError>(parse_failure));

      dap.Send(response);
      return;
    }

    if constexpr (std::is_same_v<Resp, llvm::Error>) {
      if (llvm::Error err = Run(arguments)) {
        response.success = false;
        response.body = ToResponse(std::move(err));
      } else {
        response.success = true;
      }
    } else {
      Resp body = Run(arguments);
      if (llvm::Error err = body.takeError()) {
        response.success = false;
        response.body = ToResponse(std::move(err));
      } else {
        response.success = true;
        response.body = std::move(*body);
      }
    }

    // Mark the request as 'cancelled' if the debugger was interrupted while
    // evaluating this handler.
    if (dap.debugger.InterruptRequested()) {
      dap.debugger.CancelInterruptRequest();
      response.success = false;
      response.message = protocol::eResponseMessageCancelled;
      response.body = std::nullopt;
    }

    dap.Send(response);

    PostRun();
  };

  virtual Resp Run(const Args &) const = 0;

  /// A hook for a request handler to run additional operations after the
  /// request response is sent but before the next request handler.
  virtual void PostRun() const {};

  protocol::ErrorResponseBody ToResponse(llvm::Error err) const {
    protocol::ErrorMessage error_message;
    // Default to showing the user errors unless otherwise specified by a
    // DAPError.
    error_message.showUser = true;
    error_message.sendTelemetry = false;
    if (llvm::Error unhandled = llvm::handleErrors(
            std::move(err), [&](const DAPError &E) -> llvm::Error {
              error_message.format = E.getMessage();
              error_message.showUser = E.getShowUser();
              error_message.id = E.convertToErrorCode().value();
              error_message.url = E.getURL();
              error_message.urlLabel = E.getURLLabel();
              return llvm::Error::success();
            })) {
      error_message.format = llvm::toString(std::move(unhandled));
    }
    protocol::ErrorResponseBody body;
    body.error = error_message;
    return body;
  }
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
  FeatureSet GetSupportedFeatures() const override {
    return {protocol::eAdapterFeatureBreakpointLocationsRequest};
  }
  void operator()(const llvm::json::Object &request) const override;
};

class CompletionsRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "completions"; }
  FeatureSet GetSupportedFeatures() const override {
    return {protocol::eAdapterFeatureCompletionsRequest};
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
  FeatureSet GetSupportedFeatures() const override {
    return {protocol::eAdapterFeatureConfigurationDoneRequest};
  }
  void operator()(const llvm::json::Object &request) const override;
};

class DisconnectRequestHandler
    : public RequestHandler<std::optional<protocol::DisconnectArguments>,
                            protocol::DisconnectResponse> {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral GetCommand() { return "disconnect"; }
  FeatureSet GetSupportedFeatures() const override {
    return {protocol::eAdapterFeatureTerminateDebuggee};
  }
  llvm::Error
  Run(const std::optional<protocol::DisconnectArguments> &args) const override;
};

class EvaluateRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "evaluate"; }
  void operator()(const llvm::json::Object &request) const override;
  FeatureSet GetSupportedFeatures() const override {
    return {protocol::eAdapterFeatureEvaluateForHovers};
  }
};

class ExceptionInfoRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "exceptionInfo"; }
  FeatureSet GetSupportedFeatures() const override {
    return {protocol::eAdapterFeatureExceptionInfoRequest};
  }
  void operator()(const llvm::json::Object &request) const override;
};

class InitializeRequestHandler
    : public RequestHandler<protocol::InitializeRequestArguments,
                            llvm::Expected<protocol::InitializeResponseBody>> {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral GetCommand() { return "initialize"; }
  llvm::Expected<protocol::InitializeResponseBody>
  Run(const protocol::InitializeRequestArguments &args) const override;
};

class LaunchRequestHandler
    : public RequestHandler<protocol::LaunchRequestArguments,
                            protocol::LaunchResponseBody> {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral GetCommand() { return "launch"; }
  llvm::Error
  Run(const protocol::LaunchRequestArguments &arguments) const override;
  void PostRun() const override;
};

class RestartRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "restart"; }
  FeatureSet GetSupportedFeatures() const override {
    return {protocol::eAdapterFeatureRestartRequest};
  }
  void operator()(const llvm::json::Object &request) const override;
};

class NextRequestHandler
    : public RequestHandler<protocol::NextArguments, protocol::NextResponse> {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral GetCommand() { return "next"; }
  llvm::Error Run(const protocol::NextArguments &args) const override;
};

class StepInRequestHandler : public RequestHandler<protocol::StepInArguments,
                                                   protocol::StepInResponse> {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral GetCommand() { return "stepIn"; }
  llvm::Error Run(const protocol::StepInArguments &args) const override;
};

class StepInTargetsRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "stepInTargets"; }
  FeatureSet GetSupportedFeatures() const override {
    return {protocol::eAdapterFeatureStepInTargetsRequest};
  }
  void operator()(const llvm::json::Object &request) const override;
};

class StepOutRequestHandler : public RequestHandler<protocol::StepOutArguments,
                                                    protocol::StepOutResponse> {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral GetCommand() { return "stepOut"; }
  llvm::Error Run(const protocol::StepOutArguments &args) const override;
};

class SetBreakpointsRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "setBreakpoints"; }
  FeatureSet GetSupportedFeatures() const override {
    return {protocol::eAdapterFeatureConditionalBreakpoints,
            protocol::eAdapterFeatureHitConditionalBreakpoints};
  }
  void operator()(const llvm::json::Object &request) const override;
};

class SetExceptionBreakpointsRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "setExceptionBreakpoints"; }
  FeatureSet GetSupportedFeatures() const override {
    return {protocol::eAdapterFeatureExceptionOptions};
  }
  void operator()(const llvm::json::Object &request) const override;
};

class SetFunctionBreakpointsRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "setFunctionBreakpoints"; }
  FeatureSet GetSupportedFeatures() const override {
    return {protocol::eAdapterFeatureFunctionBreakpoints};
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
  FeatureSet GetSupportedFeatures() const override {
    return {protocol::eAdapterFeatureDataBreakpoints};
  }
};

class SetInstructionBreakpointsRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() {
    return "setInstructionBreakpoints";
  }
  void operator()(const llvm::json::Object &request) const override;
  FeatureSet GetSupportedFeatures() const override {
    return {protocol::eAdapterFeatureInstructionBreakpoints};
  }
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
  FeatureSet GetSupportedFeatures() const override {
    return {protocol::eAdapterFeatureModulesRequest};
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

class SetVariableRequestHandler final
    : public RequestHandler<protocol::SetVariableArguments,
                            llvm::Expected<protocol::SetVariableResponseBody>> {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral GetCommand() { return "setVariable"; }
  FeatureSet GetSupportedFeatures() const override {
    return {protocol::eAdapterFeatureSetVariable};
  }
  llvm::Expected<protocol::SetVariableResponseBody>
  Run(const protocol::SetVariableArguments &args) const override;
};

class SourceRequestHandler final
    : public RequestHandler<protocol::SourceArguments,
                            llvm::Expected<protocol::SourceResponseBody>> {
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
  FeatureSet GetSupportedFeatures() const override {
    return {protocol::eAdapterFeatureDelayedStackTraceLoading};
  }
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
  FeatureSet GetSupportedFeatures() const override {
    return {protocol::eAdapterFeatureDisassembleRequest};
  }
  void operator()(const llvm::json::Object &request) const override;
};

class ReadMemoryRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "readMemory"; }
  FeatureSet GetSupportedFeatures() const override {
    return {protocol::eAdapterFeatureReadMemoryRequest};
  }
  void operator()(const llvm::json::Object &request) const override;
};

class CancelRequestHandler
    : public RequestHandler<protocol::CancelArguments,
                            protocol::CancelResponseBody> {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral GetCommand() { return "cancel"; }
  FeatureSet GetSupportedFeatures() const override {
    return {protocol::eAdapterFeatureCancelRequest};
  }
  llvm::Error Run(const protocol::CancelArguments &args) const override;
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

class WriteMemoryRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "writeMemory"; }
  FeatureSet GetSupportedFeatures() const override {
    return {protocol::eAdapterFeatureWriteMemoryRequest};
  }
  void operator()(const llvm::json::Object &request) const override;
};

} // namespace lldb_dap

#endif
