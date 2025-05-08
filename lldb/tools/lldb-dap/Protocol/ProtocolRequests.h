//===-- ProtocolTypes.h ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains POD structs based on the DAP specification at
// https://microsoft.github.io/debug-adapter-protocol/specification
//
// This is not meant to be a complete implementation, new interfaces are added
// when they're needed.
//
// Each struct has a toJSON and fromJSON function, that converts between
// the struct and a JSON representation. (See JSON.h)
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_PROTOCOL_PROTOCOL_REQUESTS_H
#define LLDB_TOOLS_LLDB_DAP_PROTOCOL_PROTOCOL_REQUESTS_H

#include "Protocol/ProtocolBase.h"
#include "Protocol/ProtocolTypes.h"
#include "lldb/lldb-defines.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>

namespace lldb_dap::protocol {

/// Arguments for `cancel` request.
struct CancelArguments {
  /// The ID (attribute `seq`) of the request to cancel. If missing no request
  /// is cancelled.
  ///
  /// Both a `requestId` and a `progressId` can be specified in one request.
  std::optional<int64_t> requestId;

  /// The ID (attribute `progressId`) of the progress to cancel. If missing no
  /// progress is cancelled.
  ///
  /// Both a `requestId` and a `progressId` can be specified in one request.
  std::optional<int64_t> progressId;
};
bool fromJSON(const llvm::json::Value &, CancelArguments &, llvm::json::Path);

/// Response to `cancel` request. This is just an acknowledgement, so no body
/// field is required.
using CancelResponseBody = VoidResponse;

/// Arguments for `disconnect` request.
struct DisconnectArguments {
  /// A value of true indicates that this `disconnect` request is part of a
  /// restart sequence.
  std::optional<bool> restart;

  /// Indicates whether the debuggee should be terminated when the debugger is
  /// disconnected. If unspecified, the debug adapter is free to do whatever it
  /// thinks is best. The attribute is only honored by a debug adapter if the
  /// corresponding capability `supportTerminateDebuggee` is true.
  std::optional<bool> terminateDebuggee;

  /// Indicates whether the debuggee should stay suspended when the debugger is
  /// disconnected. If unspecified, the debuggee should resume execution. The
  /// attribute is only honored by a debug adapter if the corresponding
  /// capability `supportSuspendDebuggee` is true.
  std::optional<bool> suspendDebuggee;
};
bool fromJSON(const llvm::json::Value &, DisconnectArguments &,
              llvm::json::Path);

/// Response to `disconnect` request. This is just an acknowledgement, so no
/// body field is required.
using DisconnectResponse = VoidResponse;

/// Features supported by DAP clients.
enum ClientFeature : unsigned {
  eClientFeatureVariableType,
  eClientFeatureVariablePaging,
  eClientFeatureRunInTerminalRequest,
  eClientFeatureMemoryReferences,
  eClientFeatureProgressReporting,
  eClientFeatureInvalidatedEvent,
  eClientFeatureMemoryEvent,
  /// Client supports the `argsCanBeInterpretedByShell` attribute on the
  /// `runInTerminal` request.
  eClientFeatureArgsCanBeInterpretedByShell,
  eClientFeatureStartDebuggingRequest,
  /// The client will interpret ANSI escape sequences in the display of
  /// `OutputEvent.output` and `Variable.value` fields when
  /// `Capabilities.supportsANSIStyling` is also enabled.
  eClientFeatureANSIStyling,
};

/// Format of paths reported by the debug adapter.
enum PathFormat : unsigned { ePatFormatPath, ePathFormatURI };

/// Arguments for `initialize` request.
struct InitializeRequestArguments {
  /// The ID of the debug adapter.
  std::string adapterID;

  /// The ID of the client using this adapter.
  std::optional<std::string> clientID;

  /// The human-readable name of the client using this adapter.
  std::optional<std::string> clientName;

  /// The ISO-639 locale of the client using this adapter, e.g. en-US or de-CH.
  std::optional<std::string> locale;

  /// Determines in what format paths are specified. The default is `path`,
  /// which is the native format.
  PathFormat pathFormat = ePatFormatPath;

  /// If true all line numbers are 1-based (default).
  std::optional<bool> linesStartAt1;

  /// If true all column numbers are 1-based (default).
  std::optional<bool> columnsStartAt1;

  /// The set of supported features reported by the client.
  llvm::DenseSet<ClientFeature> supportedFeatures;

  /// lldb-dap Extensions
  /// @{

  /// Source init files when initializing lldb::SBDebugger.
  std::optional<bool> lldbExtSourceInitFile;

  /// @}
};
bool fromJSON(const llvm::json::Value &, InitializeRequestArguments &,
              llvm::json::Path);

/// Response to `initialize` request. The capabilities of this debug adapter.
using InitializeResponseBody = std::optional<Capabilities>;

/// DAP Launch and Attach common configurations.
struct Configuration {
  /// Specify a working directory to use when launching `lldb-dap`. If the debug
  /// information in your executable contains relative paths, this option can be
  /// used so that `lldb-dap` can find source files and object files that have
  /// relative paths.
  std::optional<std::string> debuggerRoot;

  /// Enable auto generated summaries for variables when no summaries exist for
  /// a given type. This feature can cause performance delays in large projects
  /// when viewing variables.
  bool enableAutoVariableSummaries = false;

  /// If a variable is displayed using a synthetic children, also display the
  /// actual contents of the variable at the end under a [raw] entry. This is
  /// useful when creating sythetic child plug-ins as it lets you see the actual
  /// contents of the variable.
  bool enableSyntheticChildDebugging = false;

  /// Enable language specific extended backtraces.
  bool displayExtendedBacktrace = false;

  /// The escape prefix to use for executing regular LLDB commands in the Debug
  /// Console, instead of printing variables. Defaults to a backtick. If it's an
  /// empty string, then all expression in the Debug Console are treated as
  /// regular LLDB commands.
  std::string commandEscapePrefix = "`";

  /// If non-empty, stack frames will have descriptions generated based on the
  /// provided format. See https://lldb.llvm.org/use/formatting.html for an
  /// explanation on format strings for frames. If the format string contains
  /// errors, an error message will be displayed on the Debug Console and the
  /// default frame names will be used. This might come with a performance cost
  /// because debug information might need to be processed to generate the
  /// description.
  std::optional<std::string> customFrameFormat;

  /// Same as `customFrameFormat`, but for threads instead of stack frames.
  std::optional<std::string> customThreadFormat;

  /// Specify a source path to remap "./" to allow full paths to be used when
  /// setting breakpoints in binaries that have relative source paths.
  std::optional<std::string> sourcePath;

  /// Specify an array of path re-mappings. Each element in the array must be a
  /// two element array containing a source and destination pathname. Overrides
  /// sourcePath.
  std::vector<std::pair<std::string, std::string>> sourceMap;

  /// LLDB commands executed upon debugger startup prior to creating the LLDB
  /// target.
  std::vector<std::string> preInitCommands;

  /// LLDB commands executed upon debugger startup prior to creating the LLDB
  /// target.
  std::vector<std::string> initCommands;

  /// LLDB commands executed just before launching/attaching, after the LLDB
  /// target has been created.
  std::vector<std::string> preRunCommands;

  /// LLDB commands executed just after launching/attaching, after the LLDB
  /// target has been created.
  std::vector<std::string> postRunCommands;

  /// LLDB commands executed just after each stop.
  std::vector<std::string> stopCommands;

  /// LLDB commands executed when the program exits.
  std::vector<std::string> exitCommands;

  /// LLDB commands executed when the debugging session ends.
  std::vector<std::string> terminateCommands;

  /// Path to the executable.
  ///
  /// *NOTE:* When launching, either `launchCommands` or `program` must be
  /// configured. If both are configured then `launchCommands` takes priority.
  std::optional<std::string> program;

  /// Target triple for the program (arch-vendor-os). If not set, inferred from
  /// the binary.
  std::optional<std::string> targetTriple;

  /// Specify name of the platform to use for this target, creating the platform
  /// if necessary.
  std::optional<std::string> platformName;
};

/// lldb-dap specific launch arguments.
struct LaunchRequestArguments {
  /// Common lldb-dap configuration values for launching/attaching operations.
  Configuration configuration;

  /// If true, the launch request should launch the program without enabling
  /// debugging.
  bool noDebug = false;

  /// Launch specific operations.
  /// @{

  /// LLDB commands executed to launch the program.
  ///
  /// *NOTE:* Either launchCommands or program must be configured.
  ///
  /// If set, takes priority over the 'program' when launching the target.
  std::vector<std::string> launchCommands;

  /// The program working directory.
  std::optional<std::string> cwd;

  /// An array of command line argument strings to be passed to the program
  /// being launched.
  std::vector<std::string> args;

  /// Environment variables to set when launching the program. The format of
  /// each environment variable string is "VAR=VALUE" for environment variables
  /// with values or just "VAR" for environment variables with no values.
  llvm::StringMap<std::string> env;

  /// If set, then the client stub should detach rather than killing the debugee
  /// if it loses connection with lldb.
  bool detachOnError = false;

  /// Disable ASLR (Address Space Layout Randomization) when launching the
  /// process.
  bool disableASLR = true;

  /// Do not set up for terminal I/O to go to running process.
  bool disableSTDIO = false;

  /// Set whether to shell expand arguments to the process when launching.
  bool shellExpandArguments = false;

  /// Stop at the entry point of the program when launching a process.
  bool stopOnEntry = false;

  /// Launch the program inside an integrated terminal in the IDE. Useful for
  /// debugging interactive command line programs.
  bool runInTerminal = false;

  /// Optional timeout for `runInTerminal` requests.
  std::chrono::seconds timeout = std::chrono::seconds(30);

  /// @}
};
bool fromJSON(const llvm::json::Value &, LaunchRequestArguments &,
              llvm::json::Path);

/// Response to `launch` request. This is just an acknowledgement, so no body
/// field is required.
using LaunchResponseBody = VoidResponse;

/// Arguments for `setVariable` request.
struct SetVariableArguments {
  /// The reference of the variable container. The `variablesReference` must
  /// have been obtained in the current suspended state. See 'Lifetime of Object
  ///  References' in the Overview section for details.
  uint64_t variablesReference = UINT64_MAX;

  /// The name of the variable in the container.
  std::string name;

  /// The value of the variable.
  std::string value;

  /// Specifies details on how to format the response value.
  ValueFormat format;
};
bool fromJSON(const llvm::json::Value &, SetVariableArguments &,
              llvm::json::Path);

/// Response to `setVariable` request.
struct SetVariableResponseBody {

  /// The new value of the variable.
  std::string value;

  /// The type of the new value. Typically shown in the UI when hovering over
  /// the value.
  std::optional<std::string> type;

  /// If `variablesReference` is > 0, the new value is structured and its
  /// children can be retrieved by passing `variablesReference` to the
  /// `variables` request as long as execution remains suspended. See 'Lifetime
  /// of Object References' in the Overview section for details.
  ///
  /// If this property is included in the response, any `variablesReference`
  /// previously associated with the updated variable, and those of its
  /// children, are no longer valid.
  std::optional<uint64_t> variablesReference;

  /// The number of named child variables.
  /// The client can use this information to present the variables in a paged
  /// UI and fetch them in chunks.
  /// The value should be less than or equal to 2147483647 (2^31-1).
  std::optional<uint32_t> namedVariables;

  /// The number of indexed child variables.
  /// The client can use this information to present the variables in a paged
  /// UI and fetch them in chunks.
  /// The value should be less than or equal to 2147483647 (2^31-1).
  std::optional<uint32_t> indexedVariables;

  /// A memory reference to a location appropriate for this result.
  /// For pointer type eval results, this is generally a reference to the
  /// memory address contained in the pointer.
  /// This attribute may be returned by a debug adapter if corresponding
  /// capability `supportsMemoryReferences` is true.
  std::optional<std::string> memoryReference;

  /// A reference that allows the client to request the location where the new
  /// value is declared. For example, if the new value is function pointer, the
  /// adapter may be able to look up the function's location. This should be
  /// present only if the adapter is likely to be able to resolve the location.
  ///
  /// This reference shares the same lifetime as the `variablesReference`. See
  /// 'Lifetime of Object References' in the Overview section for details.
  std::optional<uint64_t> valueLocationReference;
};
llvm::json::Value toJSON(const SetVariableResponseBody &);

/// Arguments for `source` request.
struct SourceArguments {
  /// Specifies the source content to load. Either `source.path` or
  /// `source.sourceReference` must be specified.
  std::optional<Source> source;

  /// The reference to the source. This is the same as `source.sourceReference`.
  /// This is provided for backward compatibility since old clients do not
  /// understand the `source` attribute.
  int64_t sourceReference;
};
bool fromJSON(const llvm::json::Value &, SourceArguments &, llvm::json::Path);

/// Response to `source` request.
struct SourceResponseBody {
  /// Content of the source reference.
  std::string content;

  /// Content type (MIME type) of the source.
  std::optional<std::string> mimeType;
};
llvm::json::Value toJSON(const SourceResponseBody &);

/// Arguments for `next` request.
struct NextArguments {
  /// Specifies the thread for which to resume execution for one step (of the
  /// given granularity).
  uint64_t threadId = LLDB_INVALID_THREAD_ID;

  /// If this flag is true, all other suspended threads are not resumed.
  bool singleThread = false;

  /// Stepping granularity. If no granularity is specified, a granularity of
  /// `statement` is assumed.
  SteppingGranularity granularity = eSteppingGranularityStatement;
};
bool fromJSON(const llvm::json::Value &, NextArguments &, llvm::json::Path);

/// Response to `next` request. This is just an acknowledgement, so no
/// body field is required.
using NextResponse = VoidResponse;

/// Arguments for `stepIn` request.
struct StepInArguments {
  /// Specifies the thread for which to resume execution for one step-into (of
  /// the given granularity).
  uint64_t threadId = LLDB_INVALID_THREAD_ID;

  /// If this flag is true, all other suspended threads are not resumed.
  bool singleThread = false;

  /// Id of the target to step into.
  std::optional<uint64_t> targetId;

  /// Stepping granularity. If no granularity is specified, a granularity of
  /// `statement` is assumed.
  SteppingGranularity granularity = eSteppingGranularityStatement;
};
bool fromJSON(const llvm::json::Value &, StepInArguments &, llvm::json::Path);

/// Response to `stepIn` request. This is just an acknowledgement, so no
/// body field is required.
using StepInResponse = VoidResponse;

/// Arguments for `stepOut` request.
struct StepOutArguments {
  /// Specifies the thread for which to resume execution for one step-out (of
  /// the given granularity).
  uint64_t threadId = LLDB_INVALID_THREAD_ID;

  /// If this flag is true, all other suspended threads are not resumed.
  std::optional<bool> singleThread;

  /// Stepping granularity. If no granularity is specified, a granularity of
  /// `statement` is assumed.
  SteppingGranularity granularity = eSteppingGranularityStatement;
};
bool fromJSON(const llvm::json::Value &, StepOutArguments &, llvm::json::Path);

/// Response to `stepOut` request. This is just an acknowledgement, so no
/// body field is required.
using StepOutResponse = VoidResponse;

} // namespace lldb_dap::protocol

#endif
