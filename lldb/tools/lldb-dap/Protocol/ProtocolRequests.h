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
#include "lldb/lldb-types.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

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
using CancelResponse = VoidResponse;

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
using InitializeResponse = std::optional<Capabilities>;

/// DAP Launch and Attach common configurations.
///
/// See package.json debuggers > configurationAttributes > launch or attach >
/// properties for common configurations.
struct Configuration {
  /// Specify a working directory to use when launching `lldb-dap`. If the debug
  /// information in your executable contains relative paths, this option can be
  /// used so that `lldb-dap` can find source files and object files that have
  /// relative paths.
  std::string debuggerRoot;

  /// Enable auto generated summaries for variables when no summaries exist for
  /// a given type. This feature can cause performance delays in large projects
  /// when viewing variables.
  bool enableAutoVariableSummaries = false;

  /// If a variable is displayed using a synthetic children, also display the
  /// actual contents of the variable at the end under a [raw] entry. This is
  /// useful when creating synthetic child plug-ins as it lets you see the
  /// actual contents of the variable.
  bool enableSyntheticChildDebugging = false;

  /// Enable language specific extended backtraces.
  bool displayExtendedBacktrace = false;

  /// Stop at the entry point of the program when launching or attaching.
  bool stopOnEntry = false;

  /// Optional timeout when waiting for the program to `runInTerminal` or
  /// attach.
  std::chrono::seconds timeout = std::chrono::seconds(30);

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
  std::string sourcePath;

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
  std::string program;

  /// Target triple for the program (arch-vendor-os). If not set, inferred from
  /// the binary.
  std::string targetTriple;

  /// Specify name of the platform to use for this target, creating the platform
  /// if necessary.
  std::string platformName;
};

enum Console : unsigned {
  eConsoleInternal,
  eConsoleIntegratedTerminal,
  eConsoleExternalTerminal
};

/// lldb-dap specific launch arguments.
struct LaunchRequestArguments {
  /// Common lldb-dap configuration values for launching/attaching operations.
  Configuration configuration;

  /// If true, the launch request should launch the program without enabling
  /// debugging.
  bool noDebug = false;

  /// Launch specific operations.
  ///
  /// See package.json debuggers > configurationAttributes > launch >
  /// properties.
  /// @{

  /// LLDB commands executed to launch the program.
  ///
  /// *NOTE:* Either launchCommands or program must be configured.
  ///
  /// If set, takes priority over the 'program' when launching the target.
  std::vector<std::string> launchCommands;

  /// The program working directory.
  std::string cwd;

  /// An array of command line argument strings to be passed to the program
  /// being launched.
  std::vector<std::string> args;

  /// Environment variables to set when launching the program. The format of
  /// each environment variable string is "VAR=VALUE" for environment variables
  /// with values or just "VAR" for environment variables with no values.
  llvm::StringMap<std::string> env;

  /// If set, then the client stub should detach rather than killing the
  /// debuggee if it loses connection with lldb.
  bool detachOnError = false;

  /// Disable ASLR (Address Space Layout Randomization) when launching the
  /// process.
  bool disableASLR = true;

  /// Do not set up for terminal I/O to go to running process.
  bool disableSTDIO = false;

  /// Set whether to shell expand arguments to the process when launching.
  bool shellExpandArguments = false;

  /// Specify where to launch the program: internal console, integrated
  /// terminal or external terminal.
  Console console = eConsoleInternal;

  /// @}
};
bool fromJSON(const llvm::json::Value &, LaunchRequestArguments &,
              llvm::json::Path);

/// Response to `launch` request. This is just an acknowledgement, so no body
/// field is required.
using LaunchResponse = VoidResponse;

#define LLDB_DAP_INVALID_PORT -1
/// An invalid 'frameId' default value.
#define LLDB_DAP_INVALID_FRAME_ID UINT64_MAX

/// lldb-dap specific attach arguments.
struct AttachRequestArguments {
  /// Common lldb-dap configuration values for launching/attaching operations.
  Configuration configuration;

  /// Attach specific operations.
  ///
  /// See package.json debuggers > configurationAttributes > attach >
  /// properties.
  /// @{

  /// Custom commands that are executed instead of attaching to a process ID or
  /// to a process by name. These commands may optionally create a new target
  /// and must perform an attach. A valid process must exist after these
  /// commands complete or the `"attach"` will fail.
  std::vector<std::string> attachCommands;

  /// System process ID to attach to.
  lldb::pid_t pid = LLDB_INVALID_PROCESS_ID;

  /// Wait for the process to launch.
  bool waitFor = false;

  /// TCP/IP port to attach to a remote system. Specifying both pid and port is
  /// an error.
  int32_t gdbRemotePort = LLDB_DAP_INVALID_PORT;

  /// The hostname to connect to a remote system. The default hostname being
  /// used `localhost`.
  std::string gdbRemoteHostname = "localhost";

  /// Path to the core file to debug.
  std::string coreFile;

  /// @}
};
bool fromJSON(const llvm::json::Value &, AttachRequestArguments &,
              llvm::json::Path);

/// Response to `attach` request. This is just an acknowledgement, so no body
/// field is required.
using AttachResponse = VoidResponse;

/// Arguments for `continue` request.
struct ContinueArguments {
  /// Specifies the active thread. If the debug adapter supports single thread
  /// execution (see `supportsSingleThreadExecutionRequests`) and the argument
  /// `singleThread` is true, only the thread with this ID is resumed.
  lldb::tid_t threadId = LLDB_INVALID_THREAD_ID;

  /// If this flag is true, execution is resumed only for the thread with given
  /// `threadId`.
  bool singleThread = false;
};
bool fromJSON(const llvm::json::Value &, ContinueArguments &, llvm::json::Path);

/// Response to `continue` request.
struct ContinueResponseBody {
  // If omitted or set to `true`, this response signals to the client that all
  // threads have been resumed. The value `false` indicates that not all threads
  // were resumed.
  bool allThreadsContinued = true;
};
llvm::json::Value toJSON(const ContinueResponseBody &);

/// Arguments for `completions` request.
struct CompletionsArguments {
  /// Returns completions in the scope of this stack frame. If not specified,
  /// the completions are returned for the global scope.
  uint64_t frameId = LLDB_DAP_INVALID_FRAME_ID;

  /// One or more source lines. Typically this is the text users have typed into
  /// the debug console before they asked for completion.
  std::string text;

  /// The position within `text` for which to determine the completion
  /// proposals. It is measured in UTF-16 code units and the client capability
  /// `columnsStartAt1` determines whether it is 0- or 1-based.
  int64_t column = 0;

  /// A line for which to determine the completion proposals. If missing the
  /// first line of the text is assumed.
  int64_t line = 0;
};
bool fromJSON(const llvm::json::Value &, CompletionsArguments &,
              llvm::json::Path);

/// Response to `completions` request.
struct CompletionsResponseBody {
  /// The possible completions for a given caret position and text.
  std::vector<CompletionItem> targets;
};
llvm::json::Value toJSON(const CompletionsResponseBody &);

/// Arguments for `configurationDone` request.
using ConfigurationDoneArguments = EmptyArguments;

/// Response to `configurationDone` request. This is just an acknowledgement, so
/// no body field is required.
using ConfigurationDoneResponse = VoidResponse;

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
  std::string type;

  /// If `variablesReference` is > 0, the new value is structured and its
  /// children can be retrieved by passing `variablesReference` to the
  /// `variables` request as long as execution remains suspended. See 'Lifetime
  /// of Object References' in the Overview section for details.
  ///
  /// If this property is included in the response, any `variablesReference`
  /// previously associated with the updated variable, and those of its
  /// children, are no longer valid.
  uint64_t variablesReference = 0;

  /// The number of named child variables.
  /// The client can use this information to present the variables in a paged
  /// UI and fetch them in chunks.
  /// The value should be less than or equal to 2147483647 (2^31-1).
  uint32_t namedVariables = 0;

  /// The number of indexed child variables.
  /// The client can use this information to present the variables in a paged
  /// UI and fetch them in chunks.
  /// The value should be less than or equal to 2147483647 (2^31-1).
  uint32_t indexedVariables = 0;

  /// A memory reference to a location appropriate for this result.
  /// For pointer type eval results, this is generally a reference to the
  /// memory address contained in the pointer.
  /// This attribute may be returned by a debug adapter if corresponding
  /// capability `supportsMemoryReferences` is true.
  lldb::addr_t memoryReference = LLDB_INVALID_ADDRESS;

  /// A reference that allows the client to request the location where the new
  /// value is declared. For example, if the new value is function pointer, the
  /// adapter may be able to look up the function's location. This should be
  /// present only if the adapter is likely to be able to resolve the location.
  ///
  /// This reference shares the same lifetime as the `variablesReference`. See
  /// 'Lifetime of Object References' in the Overview section for details.
  uint64_t valueLocationReference = 0;
};
llvm::json::Value toJSON(const SetVariableResponseBody &);

struct ScopesArguments {
  /// Retrieve the scopes for the stack frame identified by `frameId`. The
  /// `frameId` must have been obtained in the current suspended state. See
  /// 'Lifetime of Object References' in the Overview section for details.
  uint64_t frameId = LLDB_DAP_INVALID_FRAME_ID;
};
bool fromJSON(const llvm::json::Value &, ScopesArguments &, llvm::json::Path);

struct ScopesResponseBody {
  std::vector<Scope> scopes;
};
llvm::json::Value toJSON(const ScopesResponseBody &);

/// Arguments for `source` request.
struct SourceArguments {
  /// Specifies the source content to load. Either `source.path` or
  /// `source.sourceReference` must be specified.
  std::optional<Source> source;

  /// The reference to the source. This is the same as `source.sourceReference`.
  /// This is provided for backward compatibility since old clients do not
  /// understand the `source` attribute.
  int64_t sourceReference = LLDB_DAP_INVALID_SRC_REF;
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

/// Arguments for the `threads` request, no arguments.
using ThreadsArguments = EmptyArguments;

/// Response to `threads` request.
struct ThreadsResponseBody {
  /// All threads.
  std::vector<Thread> threads;
};
llvm::json::Value toJSON(const ThreadsResponseBody &);

/// Arguments for `next` request.
struct NextArguments {
  /// Specifies the thread for which to resume execution for one step (of the
  /// given granularity).
  lldb::tid_t threadId = LLDB_INVALID_THREAD_ID;

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
  lldb::tid_t threadId = LLDB_INVALID_THREAD_ID;

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

/// Arguments for `stepInTargets` request.
struct StepInTargetsArguments {
  /// The stack frame for which to retrieve the possible step-in targets.
  uint64_t frameId = LLDB_DAP_INVALID_FRAME_ID;
};
bool fromJSON(const llvm::json::Value &, StepInTargetsArguments &,
              llvm::json::Path);

/// Response to `stepInTargets` request.
struct StepInTargetsResponseBody {
  /// The possible step-in targets of the specified source location.
  std::vector<StepInTarget> targets;
};
llvm::json::Value toJSON(const StepInTargetsResponseBody &);

/// Arguments for `stepOut` request.
struct StepOutArguments {
  /// Specifies the thread for which to resume execution for one step-out (of
  /// the given granularity).
  lldb::tid_t threadId = LLDB_INVALID_THREAD_ID;

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

/// Arguments for `breakpointLocations` request.
struct BreakpointLocationsArguments {
  /// The source location of the breakpoints; either `source.path` or
  /// `source.sourceReference` must be specified.
  Source source;

  /// Start line of range to search possible breakpoint locations in. If only
  /// the line is specified, the request returns all possible locations in that
  /// line.
  uint32_t line;

  /// Start position within `line` to search possible breakpoint locations in.
  /// It is measured in UTF-16 code units and the client capability
  /// `columnsStartAt1` determines whether it is 0- or 1-based. If no column is
  /// given, the first position in the start line is assumed.
  std::optional<uint32_t> column;

  /// End line of range to search possible breakpoint locations in. If no end
  /// line is given, then the end line is assumed to be the start line.
  std::optional<uint32_t> endLine;

  /// End position within `endLine` to search possible breakpoint locations in.
  /// It is measured in UTF-16 code units and the client capability
  /// `columnsStartAt1` determines whether it is 0- or 1-based. If no end column
  /// is given, the last position in the end line is assumed.
  std::optional<uint32_t> endColumn;
};
bool fromJSON(const llvm::json::Value &, BreakpointLocationsArguments &,
              llvm::json::Path);

/// Response to `breakpointLocations` request.
struct BreakpointLocationsResponseBody {
  /// Content of the source reference.
  std::vector<BreakpointLocation> breakpoints;
};
llvm::json::Value toJSON(const BreakpointLocationsResponseBody &);

/// Arguments for `setBreakpoints` request.
struct SetBreakpointsArguments {
  /// The source location of the breakpoints; either `source.path` or
  /// `source.sourceReference` must be specified.
  Source source;

  /// The code locations of the breakpoints.
  std::optional<std::vector<SourceBreakpoint>> breakpoints;

  /// Deprecated: The code locations of the breakpoints.
  std::optional<std::vector<uint32_t>> lines;

  /// A value of true indicates that the underlying source has been modified
  /// which results in new breakpoint locations.
  std::optional<bool> sourceModified;
};
bool fromJSON(const llvm::json::Value &, SetBreakpointsArguments &,
              llvm::json::Path);

/// Response to `setBreakpoints` request.
/// Returned is information about each breakpoint created by this request.
/// This includes the actual code location and whether the breakpoint could be
/// verified. The breakpoints returned are in the same order as the elements of
/// the breakpoints (or the deprecated lines) array in the arguments.
struct SetBreakpointsResponseBody {
  /// Information about the breakpoints.
  /// The array elements are in the same order as the elements of the
  /// `breakpoints` (or the deprecated `lines`) array in the arguments.
  std::vector<Breakpoint> breakpoints;
};
llvm::json::Value toJSON(const SetBreakpointsResponseBody &);

/// Arguments for `setFunctionBreakpoints` request.
struct SetFunctionBreakpointsArguments {
  /// The function names of the breakpoints.
  std::vector<FunctionBreakpoint> breakpoints;
};
bool fromJSON(const llvm::json::Value &, SetFunctionBreakpointsArguments &,
              llvm::json::Path);

/// Response to `setFunctionBreakpoints` request.
/// Returned is information about each breakpoint created by this request.
struct SetFunctionBreakpointsResponseBody {
  /// Information about the breakpoints. The array elements correspond to the
  /// elements of the `breakpoints` array.
  std::vector<Breakpoint> breakpoints;
};
llvm::json::Value toJSON(const SetFunctionBreakpointsResponseBody &);

/// Arguments for `setInstructionBreakpoints` request.
struct SetInstructionBreakpointsArguments {
  /// The instruction references of the breakpoints.
  std::vector<InstructionBreakpoint> breakpoints;
};
bool fromJSON(const llvm::json::Value &, SetInstructionBreakpointsArguments &,
              llvm::json::Path);

/// Response to `setInstructionBreakpoints` request.
struct SetInstructionBreakpointsResponseBody {
  /// Information about the breakpoints. The array elements correspond to the
  /// elements of the `breakpoints` array.
  std::vector<Breakpoint> breakpoints;
};
llvm::json::Value toJSON(const SetInstructionBreakpointsResponseBody &);

/// Arguments for `dataBreakpointInfo` request.
struct DataBreakpointInfoArguments {
  /// Reference to the variable container if the data breakpoint is requested
  /// for a child of the container. The `variablesReference` must have been
  /// obtained in the current suspended state.See 'Lifetime of Object
  /// References' in the Overview section for details.
  std::optional<int64_t> variablesReference;

  /// The name of the variable's child to obtain data breakpoint information
  /// for. If `variablesReference` isn't specified, this can be an expression,
  /// or an address if `asAddress` is also true.
  std::string name;

  /// When `name` is an expression, evaluate it in the scope of this stack
  /// frame. If not specified, the expression is evaluated in the global scope.
  /// When `asAddress` is true, the `frameId` is ignored.
  uint64_t frameId = LLDB_DAP_INVALID_FRAME_ID;

  /// If specified, a debug adapter should return information for the range of
  /// memory extending `bytes` number of bytes from the address or variable
  /// specified by `name`. Breakpoints set using the resulting data ID should
  /// pause on data access anywhere within that range.
  /// Clients may set this property only if the `supportsDataBreakpointBytes`
  /// capability is true.
  std::optional<int64_t> bytes;

  /// If `true`, the `name` is a memory address and the debugger should
  /// interpret it as a decimal value, or hex value if it is prefixed with `0x`.
  /// Clients may set this property only if the `supportsDataBreakpointBytes`
  /// capability is true.
  std::optional<bool> asAddress;

  /// The mode of the desired breakpoint. If defined, this must be one of the
  /// `breakpointModes` the debug adapter advertised in its `Capabilities`.
  std::optional<std::string> mode;
};
bool fromJSON(const llvm::json::Value &, DataBreakpointInfoArguments &,
              llvm::json::Path);

/// Response to `dataBreakpointInfo` request.
struct DataBreakpointInfoResponseBody {
  /// An identifier for the data on which a data breakpoint can be registered
  /// with the `setDataBreakpoints` request or null if no data breakpoint is
  /// available. If a `variablesReference` or `frameId` is passed, the `dataId`
  /// is valid in the current suspended state, otherwise it's valid
  /// indefinitely. See 'Lifetime of Object References' in the Overview section
  /// for details. Breakpoints set using the `dataId` in the
  /// `setDataBreakpoints` request may outlive the lifetime of the associated
  /// `dataId`.
  std::optional<std::string> dataId;

  /// UI string that describes on what data the breakpoint is set on or why a
  /// data breakpoint is not available.
  std::string description;

  /// Attribute lists the available access types for a potential data
  /// breakpoint. A UI client could surface this information.
  std::optional<std::vector<DataBreakpointAccessType>> accessTypes;

  /// Attribute indicates that a potential data breakpoint could be persisted
  /// across sessions.
  std::optional<bool> canPersist;
};
llvm::json::Value toJSON(const DataBreakpointInfoResponseBody &);

/// Arguments for `setDataBreakpoints` request.
struct SetDataBreakpointsArguments {
  /// The contents of this array replaces all existing data breakpoints. An
  /// empty array clears all data breakpoints.
  std::vector<DataBreakpoint> breakpoints;
};
bool fromJSON(const llvm::json::Value &, SetDataBreakpointsArguments &,
              llvm::json::Path);

/// Response to `setDataBreakpoints` request.
struct SetDataBreakpointsResponseBody {
  /// Information about the data breakpoints. The array elements correspond to
  /// the elements of the input argument `breakpoints` array.
  std::vector<Breakpoint> breakpoints;
};
llvm::json::Value toJSON(const SetDataBreakpointsResponseBody &);

/// Arguments for `setExceptionBreakpoints` request.
struct SetExceptionBreakpointsArguments {
  /// Set of exception filters specified by their ID. The set of all possible
  /// exception filters is defined by the `exceptionBreakpointFilters`
  /// capability. The `filter` and `filterOptions` sets are additive.
  std::vector<std::string> filters;

  /// Set of exception filters and their options. The set of all possible
  /// exception filters is defined by the `exceptionBreakpointFilters`
  /// capability. This attribute is only honored by a debug adapter if the
  /// corresponding capability `supportsExceptionFilterOptions` is true. The
  /// `filter` and `filterOptions` sets are additive.
  std::vector<ExceptionFilterOptions> filterOptions;

  // unsupported keys: exceptionOptions
};
bool fromJSON(const llvm::json::Value &, SetExceptionBreakpointsArguments &,
              llvm::json::Path);

/// Response to `setExceptionBreakpoints` request.
///
/// The response contains an array of `Breakpoint` objects with information
/// about each exception breakpoint or filter. The `Breakpoint` objects are in
/// the same order as the elements of the `filters`, `filterOptions`,
/// `exceptionOptions` arrays given as arguments. If both `filters` and
/// `filterOptions` are given, the returned array must start with `filters`
/// information first, followed by `filterOptions` information.
///
/// The `verified` property of a `Breakpoint` object signals whether the
/// exception breakpoint or filter could be successfully created and whether the
/// condition is valid. In case of an error the `message` property explains the
/// problem. The `id` property can be used to introduce a unique ID for the
/// exception breakpoint or filter so that it can be updated subsequently by
/// sending breakpoint events.
///
/// For backward compatibility both the `breakpoints` array and the enclosing
/// `body` are optional. If these elements are missing a client is not able to
/// show problems for individual exception breakpoints or filters.
struct SetExceptionBreakpointsResponseBody {
  /// Information about the exception breakpoints or filters.
  ///
  /// The breakpoints returned are in the same order as the elements of the
  /// `filters`, `filterOptions`, `exceptionOptions` arrays in the arguments. If
  /// both `filters` and `filterOptions` are given, the returned array must
  /// start with `filters` information first, followed by `filterOptions`
  /// information.
  std::vector<Breakpoint> breakpoints;
};
llvm::json::Value toJSON(const SetExceptionBreakpointsResponseBody &);

/// Arguments to `disassemble` request.
struct DisassembleArguments {
  /// Memory reference to the base location containing the instructions to
  /// disassemble.
  lldb::addr_t memoryReference = LLDB_INVALID_ADDRESS;

  /// Offset (in bytes) to be applied to the reference location before
  /// disassembling. Can be negative.
  int64_t offset = 0;

  /// Offset (in instructions) to be applied after the byte offset (if any)
  /// before disassembling. Can be negative.
  int64_t instructionOffset = 0;

  /// Number of instructions to disassemble starting at the specified location
  /// and offset.
  /// An adapter must return exactly this number of instructions - any
  /// unavailable instructions should be replaced with an implementation-defined
  /// 'invalid instruction' value.
  uint32_t instructionCount = 0;

  /// If true, the adapter should attempt to resolve memory addresses and other
  /// values to symbolic names.
  bool resolveSymbols = false;
};
bool fromJSON(const llvm::json::Value &, DisassembleArguments &,
              llvm::json::Path);
llvm::json::Value toJSON(const DisassembleArguments &);

/// Response to `disassemble` request.
struct DisassembleResponseBody {
  /// The list of disassembled instructions.
  std::vector<DisassembledInstruction> instructions;
};
bool fromJSON(const llvm::json::Value &, DisassembleResponseBody &,
              llvm::json::Path);
llvm::json::Value toJSON(const DisassembleResponseBody &);

/// Arguments for `readMemory` request.
struct ReadMemoryArguments {
  /// Memory reference to the base location from which data should be read.
  lldb::addr_t memoryReference = LLDB_INVALID_ADDRESS;

  /// Offset (in bytes) to be applied to the reference location before reading
  /// data. Can be negative.
  int64_t offset = 0;

  /// Number of bytes to read at the specified location and offset.
  uint64_t count = 0;
};
bool fromJSON(const llvm::json::Value &, ReadMemoryArguments &,
              llvm::json::Path);

/// Response to `readMemory` request.
struct ReadMemoryResponseBody {
  /// The address of the first byte of data returned.
  /// Treated as a hex value if prefixed with `0x`, or as a decimal value
  /// otherwise.
  lldb::addr_t address = LLDB_INVALID_ADDRESS;

  /// The number of unreadable bytes encountered after the last successfully
  /// read byte.
  /// This can be used to determine the number of bytes that should be skipped
  /// before a subsequent `readMemory` request succeeds.
  uint64_t unreadableBytes = 0;

  /// The bytes read from memory, encoded using base64. If the decoded length
  /// of `data` is less than the requested `count` in the original `readMemory`
  /// request, and `unreadableBytes` is zero or omitted, then the client should
  /// assume it's reached the end of readable memory.
  std::vector<std::byte> data;
};
llvm::json::Value toJSON(const ReadMemoryResponseBody &);

/// Arguments for `modules` request.
struct ModulesArguments {
  /// The index of the first module to return; if omitted modules start at 0.
  uint32_t startModule = 0;

  /// The number of modules to return. If `moduleCount` is not specified or 0,
  /// all modules are returned.
  uint32_t moduleCount = 0;
};
bool fromJSON(const llvm::json::Value &, ModulesArguments &, llvm::json::Path);

/// Response to `modules` request.
struct ModulesResponseBody {
  /// All modules or range of modules.
  std::vector<Module> modules;

  /// The total number of modules available.
  uint32_t totalModules = 0;
};
llvm::json::Value toJSON(const ModulesResponseBody &);

/// Arguments for `variables` request.
struct VariablesArguments {
  /// The variable for which to retrieve its children. The `variablesReference`
  /// must have been obtained in the current suspended state. See 'Lifetime of
  /// Object References' in the Overview section for details.
  uint64_t variablesReference;

  enum VariablesFilter : unsigned {
    eVariablesFilterBoth = 0,
    eVariablesFilterIndexed = 1 << 0,
    eVariablesFilterNamed = 1 << 1,
  };

  /// Filter to limit the child variables to either named or indexed. If
  /// omitted, both types are fetched.
  VariablesFilter filter = eVariablesFilterBoth;

  /// The index of the first variable to return; if omitted children start at 0.
  ///
  /// The attribute is only honored by a debug adapter if the corresponding
  /// capability `supportsVariablePaging` is true.
  uint64_t start = 0;

  /// The number of variables to return. If count is missing or 0, all variables
  /// are returned.
  ///
  /// The attribute is only honored by a debug adapter if the corresponding
  /// capability `supportsVariablePaging` is true.
  uint64_t count = 0;

  /// Specifies details on how to format the Variable values.
  ///
  /// The attribute is only honored by a debug adapter if the corresponding
  /// capability `supportsValueFormattingOptions` is true.
  std::optional<ValueFormat> format;
};
bool fromJSON(const llvm::json::Value &Param,
              VariablesArguments::VariablesFilter &VA, llvm::json::Path Path);
bool fromJSON(const llvm::json::Value &, VariablesArguments &,
              llvm::json::Path);

/// Response to `variables` request.
struct VariablesResponseBody {
  /// All (or a range) of variables for the given variable reference.
  std::vector<Variable> variables;
};
llvm::json::Value toJSON(const VariablesResponseBody &);

/// Arguments for `writeMemory` request.
struct WriteMemoryArguments {
  /// Memory reference to the base location to which data should be written.
  lldb::addr_t memoryReference = LLDB_INVALID_ADDRESS;

  /// Offset (in bytes) to be applied to the reference location before writing
  /// data. Can be negative.
  int64_t offset = 0;

  /// Property to control partial writes. If true, the debug adapter should
  /// attempt to write memory even if the entire memory region is not writable.
  /// In such a case the debug adapter should stop after hitting the first byte
  /// of memory that cannot be written and return the number of bytes written in
  /// the response via the `offset` and `bytesWritten` properties.
  /// If false or missing, a debug adapter should attempt to verify the region
  /// is writable before writing, and fail the response if it is not.
  bool allowPartial = false;

  /// Bytes to write, encoded using base64.
  std::string data;
};
bool fromJSON(const llvm::json::Value &, WriteMemoryArguments &,
              llvm::json::Path);

/// Response to writeMemory request.
struct WriteMemoryResponseBody {
  /// Property that should be returned when `allowPartial` is true to indicate
  /// the number of bytes starting from address that were successfully written.
  uint64_t bytesWritten = 0;
};
llvm::json::Value toJSON(const WriteMemoryResponseBody &);

} // namespace lldb_dap::protocol

#endif
