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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include <optional>
#include <type_traits>
#include <variant>
#include <vector>

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

template <typename Args>
llvm::Expected<Args> parseArgs(const protocol::Request &request) {
  if (!is_optional_v<Args> && !request.arguments)
    return llvm::make_error<DAPError>(
        llvm::formatv("arguments required for command '{0}' "
                      "but none received",
                      request.command)
            .str());

  Args arguments;
  llvm::json::Path::Root root("arguments");
  if (request.arguments && !fromJSON(*request.arguments, arguments, root)) {
    std::string parse_failure;
    llvm::raw_string_ostream OS(parse_failure);
    OS << "invalid arguments for request '" << request.command
       << "': " << llvm::toString(root.getError()) << "\n";
    root.printErrorContext(*request.arguments, OS);
    return llvm::make_error<DAPError>(parse_failure);
  }

  return arguments;
}
template <>
inline llvm::Expected<protocol::EmptyArguments>
parseArgs(const protocol::Request &request) {
  return std::nullopt;
}

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

    llvm::Expected<Args> arguments = parseArgs<Args>(request);
    if (llvm::Error err = arguments.takeError()) {
      HandleErrorResponse(std::move(err), response);
      dap.Send(response);
      return;
    }

    if constexpr (std::is_same_v<Resp, llvm::Error>) {
      if (llvm::Error err = Run(*arguments)) {
        HandleErrorResponse(std::move(err), response);
      } else {
        response.success = true;
      }
    } else {
      Resp body = Run(*arguments);
      if (llvm::Error err = body.takeError()) {
        HandleErrorResponse(std::move(err), response);
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
  ///
  /// *NOTE*: PostRun will be invoked even if the `Run` operation returned an
  /// error.
  virtual void PostRun() const {};

  void HandleErrorResponse(llvm::Error err,
                           protocol::Response &response) const {
    response.success = false;
    llvm::handleAllErrors(
        std::move(err),
        [&](const NotStoppedError &err) {
          response.message = lldb_dap::protocol::eResponseMessageNotStopped;
        },
        [&](const DAPError &err) {
          protocol::ErrorMessage error_message;
          error_message.sendTelemetry = false;
          error_message.format = err.getMessage();
          error_message.showUser = err.getShowUser();
          error_message.id = err.convertToErrorCode().value();
          error_message.url = err.getURL();
          error_message.urlLabel = err.getURLLabel();
          protocol::ErrorResponseBody body;
          body.error = error_message;
          response.body = body;
        },
        [&](const llvm::ErrorInfoBase &err) {
          protocol::ErrorMessage error_message;
          error_message.showUser = true;
          error_message.sendTelemetry = false;
          error_message.format = err.message();
          error_message.id = err.convertToErrorCode().value();
          protocol::ErrorResponseBody body;
          body.error = error_message;
          response.body = body;
        });
  }
};

class AttachRequestHandler
    : public RequestHandler<protocol::AttachRequestArguments,
                            protocol::AttachResponse> {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral GetCommand() { return "attach"; }
  llvm::Error Run(const protocol::AttachRequestArguments &args) const override;
  void PostRun() const override;
};

class BreakpointLocationsRequestHandler
    : public RequestHandler<
          protocol::BreakpointLocationsArguments,
          llvm::Expected<protocol::BreakpointLocationsResponseBody>> {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral GetCommand() { return "breakpointLocations"; }
  FeatureSet GetSupportedFeatures() const override {
    return {protocol::eAdapterFeatureBreakpointLocationsRequest};
  }
  llvm::Expected<protocol::BreakpointLocationsResponseBody>
  Run(const protocol::BreakpointLocationsArguments &args) const override;

  std::vector<std::pair<uint32_t, uint32_t>>
  GetSourceBreakpointLocations(std::string path, uint32_t start_line,
                               uint32_t start_column, uint32_t end_line,
                               uint32_t end_column) const;
  std::vector<std::pair<uint32_t, uint32_t>>
  GetAssemblyBreakpointLocations(int64_t source_reference, uint32_t start_line,
                                 uint32_t end_line) const;
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

class ContinueRequestHandler
    : public RequestHandler<protocol::ContinueArguments,
                            llvm::Expected<protocol::ContinueResponseBody>> {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral GetCommand() { return "continue"; }
  llvm::Expected<protocol::ContinueResponseBody>
  Run(const protocol::ContinueArguments &args) const override;
};

class ConfigurationDoneRequestHandler
    : public RequestHandler<protocol::ConfigurationDoneArguments,
                            protocol::ConfigurationDoneResponse> {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral GetCommand() { return "configurationDone"; }
  FeatureSet GetSupportedFeatures() const override {
    return {protocol::eAdapterFeatureConfigurationDoneRequest};
  }
  protocol::ConfigurationDoneResponse
  Run(const protocol::ConfigurationDoneArguments &) const override;
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
                            llvm::Expected<protocol::InitializeResponse>> {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral GetCommand() { return "initialize"; }
  llvm::Expected<protocol::InitializeResponse>
  Run(const protocol::InitializeRequestArguments &args) const override;
};

class LaunchRequestHandler
    : public RequestHandler<protocol::LaunchRequestArguments,
                            protocol::LaunchResponse> {
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

class StepInTargetsRequestHandler
    : public RequestHandler<
          protocol::StepInTargetsArguments,
          llvm::Expected<protocol::StepInTargetsResponseBody>> {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral GetCommand() { return "stepInTargets"; }
  llvm::Expected<protocol::StepInTargetsResponseBody>
  Run(const protocol::StepInTargetsArguments &args) const override;
};

class StepOutRequestHandler : public RequestHandler<protocol::StepOutArguments,
                                                    protocol::StepOutResponse> {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral GetCommand() { return "stepOut"; }
  llvm::Error Run(const protocol::StepOutArguments &args) const override;
};

class SetBreakpointsRequestHandler
    : public RequestHandler<
          protocol::SetBreakpointsArguments,
          llvm::Expected<protocol::SetBreakpointsResponseBody>> {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral GetCommand() { return "setBreakpoints"; }
  FeatureSet GetSupportedFeatures() const override {
    return {protocol::eAdapterFeatureConditionalBreakpoints,
            protocol::eAdapterFeatureHitConditionalBreakpoints};
  }
  llvm::Expected<protocol::SetBreakpointsResponseBody>
  Run(const protocol::SetBreakpointsArguments &args) const override;
};

class SetExceptionBreakpointsRequestHandler
    : public RequestHandler<
          protocol::SetExceptionBreakpointsArguments,
          llvm::Expected<protocol::SetExceptionBreakpointsResponseBody>> {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral GetCommand() { return "setExceptionBreakpoints"; }
  FeatureSet GetSupportedFeatures() const override {
    /// Prefer the `filterOptions` feature over the `exceptionOptions`.
    /// exceptionOptions is not supported in VSCode, while `filterOptions` is
    /// supported.
    return {protocol::eAdapterFeatureExceptionFilterOptions};
  }
  llvm::Expected<protocol::SetExceptionBreakpointsResponseBody>
  Run(const protocol::SetExceptionBreakpointsArguments &args) const override;
};

class SetFunctionBreakpointsRequestHandler
    : public RequestHandler<
          protocol::SetFunctionBreakpointsArguments,
          llvm::Expected<protocol::SetFunctionBreakpointsResponseBody>> {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral GetCommand() { return "setFunctionBreakpoints"; }
  FeatureSet GetSupportedFeatures() const override {
    return {protocol::eAdapterFeatureFunctionBreakpoints};
  }
  llvm::Expected<protocol::SetFunctionBreakpointsResponseBody>
  Run(const protocol::SetFunctionBreakpointsArguments &args) const override;
};

class DataBreakpointInfoRequestHandler
    : public RequestHandler<
          protocol::DataBreakpointInfoArguments,
          llvm::Expected<protocol::DataBreakpointInfoResponseBody>> {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral GetCommand() { return "dataBreakpointInfo"; }
  llvm::Expected<protocol::DataBreakpointInfoResponseBody>
  Run(const protocol::DataBreakpointInfoArguments &args) const override;
};

class SetDataBreakpointsRequestHandler
    : public RequestHandler<
          protocol::SetDataBreakpointsArguments,
          llvm::Expected<protocol::SetDataBreakpointsResponseBody>> {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral GetCommand() { return "setDataBreakpoints"; }
  FeatureSet GetSupportedFeatures() const override {
    return {protocol::eAdapterFeatureDataBreakpoints};
  }
  llvm::Expected<protocol::SetDataBreakpointsResponseBody>
  Run(const protocol::SetDataBreakpointsArguments &args) const override;
};

class SetInstructionBreakpointsRequestHandler
    : public RequestHandler<
          protocol::SetInstructionBreakpointsArguments,
          llvm::Expected<protocol::SetInstructionBreakpointsResponseBody>> {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral GetCommand() {
    return "setInstructionBreakpoints";
  }
  FeatureSet GetSupportedFeatures() const override {
    return {protocol::eAdapterFeatureInstructionBreakpoints};
  }
  llvm::Expected<protocol::SetInstructionBreakpointsResponseBody>
  Run(const protocol::SetInstructionBreakpointsArguments &args) const override;
};

class CompileUnitsRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "compileUnits"; }
  void operator()(const llvm::json::Object &request) const override;
};

class ModulesRequestHandler final
    : public RequestHandler<std::optional<protocol::ModulesArguments>,
                            llvm::Expected<protocol::ModulesResponseBody>> {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral GetCommand() { return "modules"; }
  FeatureSet GetSupportedFeatures() const override {
    return {protocol::eAdapterFeatureModulesRequest};
  }
  llvm::Expected<protocol::ModulesResponseBody>
  Run(const std::optional<protocol::ModulesArguments> &args) const override;
};

class PauseRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "pause"; }
  void operator()(const llvm::json::Object &request) const override;
};

class ScopesRequestHandler final
    : public RequestHandler<protocol::ScopesArguments,
                            llvm::Expected<protocol::ScopesResponseBody>> {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral GetCommand() { return "scopes"; }

  llvm::Expected<protocol::ScopesResponseBody>
  Run(const protocol::ScopesArguments &args) const override;
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

class ThreadsRequestHandler
    : public RequestHandler<protocol::ThreadsArguments,
                            llvm::Expected<protocol::ThreadsResponseBody>> {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral GetCommand() { return "threads"; }
  llvm::Expected<protocol::ThreadsResponseBody>
  Run(const protocol::ThreadsArguments &) const override;
};

class VariablesRequestHandler
    : public RequestHandler<protocol::VariablesArguments,
                            llvm::Expected<protocol::VariablesResponseBody>> {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral GetCommand() { return "variables"; }
  llvm::Expected<protocol::VariablesResponseBody>
  Run(const protocol::VariablesArguments &) const override;
};

class LocationsRequestHandler : public LegacyRequestHandler {
public:
  using LegacyRequestHandler::LegacyRequestHandler;
  static llvm::StringLiteral GetCommand() { return "locations"; }
  void operator()(const llvm::json::Object &request) const override;
};

class DisassembleRequestHandler final
    : public RequestHandler<protocol::DisassembleArguments,
                            llvm::Expected<protocol::DisassembleResponseBody>> {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral GetCommand() { return "disassemble"; }
  FeatureSet GetSupportedFeatures() const override {
    return {protocol::eAdapterFeatureDisassembleRequest};
  }
  llvm::Expected<protocol::DisassembleResponseBody>
  Run(const protocol::DisassembleArguments &args) const override;
};

class ReadMemoryRequestHandler final
    : public RequestHandler<protocol::ReadMemoryArguments,
                            llvm::Expected<protocol::ReadMemoryResponseBody>> {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral GetCommand() { return "readMemory"; }
  FeatureSet GetSupportedFeatures() const override {
    return {protocol::eAdapterFeatureReadMemoryRequest};
  }
  llvm::Expected<protocol::ReadMemoryResponseBody>
  Run(const protocol::ReadMemoryArguments &args) const override;
};

class CancelRequestHandler : public RequestHandler<protocol::CancelArguments,
                                                   protocol::CancelResponse> {
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

class WriteMemoryRequestHandler final
    : public RequestHandler<protocol::WriteMemoryArguments,
                            llvm::Expected<protocol::WriteMemoryResponseBody>> {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral GetCommand() { return "writeMemory"; }
  FeatureSet GetSupportedFeatures() const override {
    return {protocol::eAdapterFeatureWriteMemoryRequest};
  }
  llvm::Expected<protocol::WriteMemoryResponseBody>
  Run(const protocol::WriteMemoryArguments &args) const override;
};

} // namespace lldb_dap

#endif
