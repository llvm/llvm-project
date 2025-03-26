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
#include <set>
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
enum class ChecksumAlgorithm { MD5, SHA1, SHA256, timestamp };
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
  enum class Feature {
    /// The debug adapter supports ANSI escape sequences in styling of
    /// `OutputEvent.output` and `Variable.value` fields.
    supportsANSIStyling,
    /// The debug adapter supports the `breakpointLocations` request.
    supportsBreakpointLocationsRequest,
    /// The debug adapter supports the `cancel` request.
    supportsCancelRequest,
    /// The debug adapter supports the `clipboard` context value in the
    /// `evaluate` request.
    supportsClipboardContext,
    /// The debug adapter supports the `completions` request.
    supportsCompletionsRequest,
    /// The debug adapter supports conditional breakpoints.
    supportsConditionalBreakpoints,
    /// The debug adapter supports the `configurationDone` request.
    supportsConfigurationDoneRequest,
    /// The debug adapter supports the `asAddress` and `bytes` fields in the
    /// `dataBreakpointInfo` request.
    supportsDataBreakpointBytes,
    /// The debug adapter supports data breakpoints.
    supportsDataBreakpoints,
    /// The debug adapter supports the delayed loading of parts of the stack,
    /// which requires that both the `startFrame` and `levels` arguments and the
    /// `totalFrames` result of the `stackTrace` request are supported.
    supportsDelayedStackTraceLoading,
    /// The debug adapter supports the `disassemble` request.
    supportsDisassembleRequest,
    /// The debug adapter supports a (side effect free) `evaluate` request for
    /// data hovers.
    supportsEvaluateForHovers,
    /// The debug adapter supports `filterOptions` as an argument on the
    /// `setExceptionBreakpoints` request.
    supportsExceptionFilterOptions,
    /// The debug adapter supports the `exceptionInfo` request.
    supportsExceptionInfoRequest,
    /// The debug adapter supports `exceptionOptions` on the
    /// `setExceptionBreakpoints` request.
    supportsExceptionOptions,
    /// The debug adapter supports function breakpoints.
    supportsFunctionBreakpoints,
    /// The debug adapter supports the `gotoTargets` request.
    supportsGotoTargetsRequest,
    /// The debug adapter supports breakpoints that break execution after a
    /// specified number of hits.
    supportsHitConditionalBreakpoints,
    /// The debug adapter supports adding breakpoints based on instruction
    /// references.
    supportsInstructionBreakpoints,
    /// The debug adapter supports the `loadedSources` request.
    supportsLoadedSourcesRequest,
    /// The debug adapter supports log points by interpreting the `logMessage`
    /// attribute of the `SourceBreakpoint`.
    supportsLogPoints,
    /// The debug adapter supports the `modules` request.
    supportsModulesRequest,
    /// The debug adapter supports the `readMemory` request.
    supportsReadMemoryRequest,
    /// The debug adapter supports restarting a frame.
    supportsRestartFrame,
    /// The debug adapter supports the `restart` request. In this case a client
    /// should not implement `restart` by terminating and relaunching the
    /// adapter but by calling the `restart` request.
    supportsRestartRequest,
    /// The debug adapter supports the `setExpression` request.
    supportsSetExpression,
    /// The debug adapter supports setting a variable to a value.
    supportsSetVariable,
    /// The debug adapter supports the `singleThread` property on the execution
    /// requests (`continue`, `next`, `stepIn`, `stepOut`, `reverseContinue`,
    /// `stepBack`).
    supportsSingleThreadExecutionRequests,
    /// The debug adapter supports stepping back via the `stepBack` and
    /// `reverseContinue` requests.
    supportsStepBack,
    /// The debug adapter supports the `stepInTargets` request.
    supportsStepInTargetsRequest,
    /// The debug adapter supports stepping granularities (argument
    /// `granularity`) for the stepping requests.
    supportsSteppingGranularity,
    /// The debug adapter supports the `terminate` request.
    supportsTerminateRequest,
    /// The debug adapter supports the `terminateThreads` request.
    supportsTerminateThreadsRequest,
    /// The debug adapter supports the `suspendDebuggee` attribute on the
    /// `disconnect` request.
    supportSuspendDebuggee,
    /// The debug adapter supports a `format` attribute on the `stackTrace`,
    /// `variables`, and `evaluate` requests.
    supportsValueFormattingOptions,
    /// The debug adapter supports the `writeMemory` request.
    supportsWriteMemoryRequest,
    /// The debug adapter supports the `terminateDebuggee` attribute on the
    /// `disconnect` request.
    supportTerminateDebuggee,
  };

  /// The supported features for this adapter.
  std::set<Feature> supportedFeatures;

  /// Available exception filter options for the `setExceptionBreakpoints`
  /// request.
  std::optional<std::vector<ExceptionBreakpointsFilter>>
      exceptionBreakpointFilters;

  /// The set of characters that should trigger completion in a REPL. If not
  /// specified, the UI should assume the `.` character.
  std::optional<std::vector<std::string>> completionTriggerCharacters;

  /// The set of additional module information exposed by the debug adapter.
  std::optional<std::vector<ColumnDescriptor>> additionalModuleColumns;

  /// Checksum algorithms supported by the debug adapter.
  std::optional<std::vector<ChecksumAlgorithm>> supportedChecksumAlgorithms;

  /// Modes of breakpoints supported by the debug adapter, such as 'hardware' or
  /// 'software'. If present, the client may allow the user to select a mode and
  /// include it in its `setBreakpoints` request.
  ///
  /// Clients may present the first applicable mode in this array as the
  /// 'default' mode in gestures that set breakpoints.
  std::optional<std::vector<BreakpointMode>> breakpointModes;

  /// lldb-dap Extensions
  /// @{

  /// The version of the adapter.
  std::optional<std::string> lldbExtVersion;

  /// @}
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
