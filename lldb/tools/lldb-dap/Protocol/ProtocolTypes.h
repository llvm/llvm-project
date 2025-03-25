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

#ifndef LLDB_TOOLS_LLDB_DAP_PROTOCOL_PROTOCOL_TYPES_H
#define LLDB_TOOLS_LLDB_DAP_PROTOCOL_PROTOCOL_TYPES_H

#include "llvm/Support/JSON.h"
#include <cstdint>
#include <optional>
#include <string>

namespace lldb_dap::protocol {

/// An `ExceptionBreakpointsFilter` is shown in the UI as an filter option for
/// configuring how exceptions are dealt with.
struct ExceptionBreakpointsFilter {
  /// The internal ID of the filter option. This value is passed to the
  /// `setExceptionBreakpoints` request.
  std::string filter;

  /// The name of the filter option. This is shown in the UI.
  std::string label;

  /// A help text providing additional information about the exception filter.
  /// This string is typically shown as a hover and can be translated.
  std::optional<std::string> description;

  /// Initial value of the filter option. If not specified a value false is
  /// assumed.
  std::optional<bool> defaultState;

  /// Controls whether a condition can be specified for this filter option. If
  /// false or missing, a condition can not be set.
  std::optional<bool> supportsCondition;

  /// A help text providing information about the condition. This string is
  /// shown as the placeholder text for a text box and can be translated.
  std::optional<std::string> conditionDescription;
};
llvm::json::Value toJSON(const ExceptionBreakpointsFilter &);

/// A ColumnDescriptor specifies what module attribute to show in a column of
/// the modules view, how to format it, and what the column’s label should be.
///
/// It is only used if the underlying UI actually supports this level of
/// customization.
struct ColumnDescriptor {
  /// Name of the attribute rendered in this column.
  std::string attributeName;

  /// Header UI label of column.
  std::string label;

  /// Format to use for the rendered values in this column. TBD how the format
  /// strings looks like.
  std::optional<std::string> format;

  enum class Type { String, Number, Boolean, Timestamp };

  /// Datatype of values in this column. Defaults to `string` if not specified.
  /// Values: 'string', 'number', 'boolean', 'unixTimestampUTC'.
  std::optional<Type> type;

  /// Width of this column in characters (hint only).
  std::optional<int> width;
};
llvm::json::Value toJSON(const ColumnDescriptor &);

/// Names of checksum algorithms that may be supported by a debug adapter.
/// Values: ‘MD5’, ‘SHA1’, ‘SHA256’, ‘timestamp’.
enum class ChecksumAlgorithm { md5, sha1, sha256, timestamp };
llvm::json::Value toJSON(const ChecksumAlgorithm &);

/// Describes one or more type of breakpoint a BreakpointMode applies to. This
/// is a non-exhaustive enumeration and may expand as future breakpoint types
/// are added.
enum class BreakpointModeApplicability {
  /// In `SourceBreakpoint`'s.
  source,
  /// In exception breakpoints applied in the `ExceptionFilterOptions`.
  exception,
  /// In data breakpoints requested in the `DataBreakpointInfo` request.
  data,
  /// In `InstructionBreakpoint`'s.
  instruction
};
llvm::json::Value toJSON(const BreakpointModeApplicability &);

/// A `BreakpointMode` is provided as a option when setting breakpoints on
/// sources or instructions.
struct BreakpointMode {
  /// The internal ID of the mode. This value is passed to the `setBreakpoints`
  /// request.
  std::string mode;

  /// The name of the breakpoint mode. This is shown in the UI.
  std::string label;

  /// A help text providing additional information about the breakpoint mode.
  /// This string is typically shown as a hover and can be translated.
  std::optional<std::string> description;

  /// Describes one or more type of breakpoint this mode applies to.
  std::vector<BreakpointModeApplicability> appliesTo;
};
llvm::json::Value toJSON(const BreakpointMode &);

/// Information about the capabilities of a debug adapter.
struct Capabilities {
  /// The debug adapter supports the `configurationDone` request.
  std::optional<bool> supportsConfigurationDoneRequest;

  /// The debug adapter supports function breakpoints.
  std::optional<bool> supportsFunctionBreakpoints;

  /// The debug adapter supports conditional breakpoints.
  std::optional<bool> supportsConditionalBreakpoints;

  /// The debug adapter supports breakpoints that break execution after a
  /// specified number of hits.
  std::optional<bool> supportsHitConditionalBreakpoints;

  /// The debug adapter supports a (side effect free) `evaluate` request for
  /// data hovers.
  std::optional<bool> supportsEvaluateForHovers;

  /// Available exception filter options for the `setExceptionBreakpoints`
  /// request.
  std::optional<std::vector<ExceptionBreakpointsFilter>>
      exceptionBreakpointFilters;

  /// The debug adapter supports stepping back via the `stepBack` and
  /// `reverseContinue` requests.
  std::optional<bool> supportsStepBack;

  /// The debug adapter supports setting a variable to a value.
  std::optional<bool> supportsSetVariable;

  /// The debug adapter supports restarting a frame.
  std::optional<bool> supportsRestartFrame;

  /// The debug adapter supports the `gotoTargets` request.
  std::optional<bool> supportsGotoTargetsRequest;

  /// The debug adapter supports the `stepInTargets` request.
  std::optional<bool> supportsStepInTargetsRequest;

  /// The debug adapter supports the `completions` request.
  std::optional<bool> supportsCompletionsRequest;

  /// The set of characters that should trigger completion in a REPL. If not
  /// specified, the UI should assume the `.` character.
  std::optional<std::vector<std::string>> completionTriggerCharacters;

  /// The debug adapter supports the `modules` request.
  std::optional<bool> supportsModulesRequest;

  /// The set of additional module information exposed by the debug adapter.
  std::optional<std::vector<ColumnDescriptor>> additionalModuleColumns;

  /// Checksum algorithms supported by the debug adapter.
  std::optional<std::vector<ChecksumAlgorithm>> supportedChecksumAlgorithms;

  /// The debug adapter supports the `restart` request. In this case a client
  /// should not implement `restart` by terminating and relaunching the adapter
  /// but by calling the `restart` request.
  std::optional<bool> supportsRestartRequest;

  /// The debug adapter supports `exceptionOptions` on the
  /// `setExceptionBreakpoints` request.
  std::optional<bool> supportsExceptionOptions;

  /// The debug adapter supports a `format` attribute on the `stackTrace`,
  /// `variables`, and `evaluate` requests.
  std::optional<bool> supportsValueFormattingOptions;

  /// The debug adapter supports the `exceptionInfo` request.
  std::optional<bool> supportsExceptionInfoRequest;

  /// The debug adapter supports the `terminateDebuggee` attribute on the
  /// `disconnect` request.
  std::optional<bool> supportTerminateDebuggee;

  /// The debug adapter supports the `suspendDebuggee` attribute on the
  /// `disconnect` request.
  std::optional<bool> supportSuspendDebuggee;

  /// The debug adapter supports the delayed loading of parts of the stack,
  /// which requires that both the `startFrame` and `levels` arguments and the
  /// `totalFrames` result of the `stackTrace` request are supported.
  std::optional<bool> supportsDelayedStackTraceLoading;

  /// The debug adapter supports the `loadedSources` request.
  std::optional<bool> supportsLoadedSourcesRequest;

  /// The debug adapter supports log points by interpreting the `logMessage`
  /// attribute of the `SourceBreakpoint`.
  std::optional<bool> supportsLogPoints;

  /// The debug adapter supports the `terminateThreads` request.
  std::optional<bool> supportsTerminateThreadsRequest;

  /// The debug adapter supports the `setExpression` request.
  std::optional<bool> supportsSetExpression;

  /// The debug adapter supports the `terminate` request.
  std::optional<bool> supportsTerminateRequest;

  /// The debug adapter supports data breakpoints.
  std::optional<bool> supportsDataBreakpoints;

  /// The debug adapter supports the `readMemory` request.
  std::optional<bool> supportsReadMemoryRequest;

  /// The debug adapter supports the `writeMemory` request.
  std::optional<bool> supportsWriteMemoryRequest;

  /// The debug adapter supports the `disassemble` request.
  std::optional<bool> supportsDisassembleRequest;

  /// The debug adapter supports the `cancel` request.
  std::optional<bool> supportsCancelRequest;

  /// The debug adapter supports the `breakpointLocations` request.
  std::optional<bool> supportsBreakpointLocationsRequest;

  /// The debug adapter supports the `clipboard` context value in the `evaluate`
  /// request.
  std::optional<bool> supportsClipboardContext;

  /// The debug adapter supports stepping granularities (argument `granularity`)
  /// for the stepping requests.
  std::optional<bool> supportsSteppingGranularity;

  /// The debug adapter supports adding breakpoints based on instruction
  /// references.
  std::optional<bool> supportsInstructionBreakpoints;

  /// The debug adapter supports `filterOptions` as an argument on the
  /// `setExceptionBreakpoints` request.
  std::optional<bool> supportsExceptionFilterOptions;

  /// The debug adapter supports the `singleThread` property on the execution
  /// requests (`continue`, `next`, `stepIn`, `stepOut`, `reverseContinue`,
  /// `stepBack`).
  std::optional<bool> supportsSingleThreadExecutionRequests;

  /// The debug adapter supports the `asAddress` and `bytes` fields in the
  /// `dataBreakpointInfo` request.
  std::optional<bool> supportsDataBreakpointBytes;

  /// Modes of breakpoints supported by the debug adapter, such as 'hardware' or
  /// 'software'. If present, the client may allow the user to select a mode and
  /// include it in its `setBreakpoints` request.
  ///
  /// Clients may present the first applicable mode in this array as the
  /// 'default' mode in gestures that set breakpoints.
  std::optional<std::vector<BreakpointMode>> breakpointModes;

  /// The debug adapter supports ANSI escape sequences in styling of
  /// `OutputEvent.output` and `Variable.value` fields.
  std::optional<bool> supportsANSIStyling;

  /// lldb-dap Extensions
  std::optional<std::string> lldbVersion;
};
llvm::json::Value toJSON(const Capabilities &);

/// A `Source` is a descriptor for source code. It is returned from the debug
/// adapter as part of a `StackFrame` and it is used by clients when specifying
/// breakpoints.
struct Source {
  enum class PresentationHint { normal, emphasize, deemphasize };

  /// The short name of the source. Every source returned from the debug adapter
  /// has a name. When sending a source to the debug adapter this name is
  /// optional.
  std::optional<std::string> name;

  /// The path of the source to be shown in the UI. It is only used to locate
  /// and load the content of the source if no `sourceReference` is specified
  /// (or its value is 0).
  std::optional<std::string> path;

  /// If the value > 0 the contents of the source must be retrieved through the
  /// `source` request (even if a path is specified). Since a `sourceReference`
  /// is only valid for a session, it can not be used to persist a source. The
  /// value should be less than or equal to 2147483647 (2^31-1).
  std::optional<int64_t> sourceReference;

  /// A hint for how to present the source in the UI. A value of `deemphasize`
  /// can be used to indicate that the source is not available or that it is
  /// skipped on stepping.
  std::optional<PresentationHint> presentationHint;

  // unsupported keys: origin, sources, adapterData, checksums
};
bool fromJSON(const llvm::json::Value &, Source &, llvm::json::Path);

} // namespace lldb_dap::protocol

#endif
