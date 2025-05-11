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

#include "lldb/lldb-defines.h"
#include "llvm/ADT/DenseSet.h"
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

enum ColumnType : unsigned {
  eColumnTypeString,
  eColumnTypeNumber,
  eColumnTypeBoolean,
  eColumnTypeTimestamp
};

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

  /// Datatype of values in this column. Defaults to `string` if not specified.
  /// Values: 'string', 'number', 'boolean', 'unixTimestampUTC'.
  std::optional<ColumnType> type;

  /// Width of this column in characters (hint only).
  std::optional<int> width;
};
llvm::json::Value toJSON(const ColumnDescriptor &);

/// Names of checksum algorithms that may be supported by a debug adapter.
/// Values: ‘MD5’, ‘SHA1’, ‘SHA256’, ‘timestamp’.
enum ChecksumAlgorithm : unsigned {
  eChecksumAlgorithmMD5,
  eChecksumAlgorithmSHA1,
  eChecksumAlgorithmSHA256,
  eChecksumAlgorithmTimestamp
};
llvm::json::Value toJSON(const ChecksumAlgorithm &);

/// Describes one or more type of breakpoint a BreakpointMode applies to. This
/// is a non-exhaustive enumeration and may expand as future breakpoint types
/// are added.
enum BreakpointModeApplicability : unsigned {
  /// In `SourceBreakpoint`'s.
  eBreakpointModeApplicabilitySource,
  /// In exception breakpoints applied in the `ExceptionFilterOptions`.
  eBreakpointModeApplicabilityException,
  /// In data breakpoints requested in the `DataBreakpointInfo` request.
  eBreakpointModeApplicabilityData,
  /// In `InstructionBreakpoint`'s.
  eBreakpointModeApplicabilityInstruction
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

/// Debug Adapter Features flags supported by lldb-dap.
enum AdapterFeature : unsigned {
  /// The debug adapter supports ANSI escape sequences in styling of
  /// `OutputEvent.output` and `Variable.value` fields.
  eAdapterFeatureANSIStyling,
  /// The debug adapter supports the `breakpointLocations` request.
  eAdapterFeatureBreakpointLocationsRequest,
  /// The debug adapter supports the `cancel` request.
  eAdapterFeatureCancelRequest,
  /// The debug adapter supports the `clipboard` context value in the
  /// `evaluate` request.
  eAdapterFeatureClipboardContext,
  /// The debug adapter supports the `completions` request.
  eAdapterFeatureCompletionsRequest,
  /// The debug adapter supports conditional breakpoints.
  eAdapterFeatureConditionalBreakpoints,
  /// The debug adapter supports the `configurationDone` request.
  eAdapterFeatureConfigurationDoneRequest,
  /// The debug adapter supports the `asAddress` and `bytes` fields in the
  /// `dataBreakpointInfo` request.
  eAdapterFeatureDataBreakpointBytes,
  /// The debug adapter supports data breakpoints.
  eAdapterFeatureDataBreakpoints,
  /// The debug adapter supports the delayed loading of parts of the stack,
  /// which requires that both the `startFrame` and `levels` arguments and the
  /// `totalFrames` result of the `stackTrace` request are supported.
  eAdapterFeatureDelayedStackTraceLoading,
  /// The debug adapter supports the `disassemble` request.
  eAdapterFeatureDisassembleRequest,
  /// The debug adapter supports a (side effect free) `evaluate` request for
  /// data hovers.
  eAdapterFeatureEvaluateForHovers,
  /// The debug adapter supports `filterOptions` as an argument on the
  /// `setExceptionBreakpoints` request.
  eAdapterFeatureExceptionFilterOptions,
  /// The debug adapter supports the `exceptionInfo` request.
  eAdapterFeatureExceptionInfoRequest,
  /// The debug adapter supports `exceptionOptions` on the
  /// `setExceptionBreakpoints` request.
  eAdapterFeatureExceptionOptions,
  /// The debug adapter supports function breakpoints.
  eAdapterFeatureFunctionBreakpoints,
  /// The debug adapter supports the `gotoTargets` request.
  eAdapterFeatureGotoTargetsRequest,
  /// The debug adapter supports breakpoints that break execution after a
  /// specified number of hits.
  eAdapterFeatureHitConditionalBreakpoints,
  /// The debug adapter supports adding breakpoints based on instruction
  /// references.
  eAdapterFeatureInstructionBreakpoints,
  /// The debug adapter supports the `loadedSources` request.
  eAdapterFeatureLoadedSourcesRequest,
  /// The debug adapter supports log points by interpreting the `logMessage`
  /// attribute of the `SourceBreakpoint`.
  eAdapterFeatureLogPoints,
  /// The debug adapter supports the `modules` request.
  eAdapterFeatureModulesRequest,
  /// The debug adapter supports the `readMemory` request.
  eAdapterFeatureReadMemoryRequest,
  /// The debug adapter supports restarting a frame.
  eAdapterFeatureRestartFrame,
  /// The debug adapter supports the `restart` request. In this case a client
  /// should not implement `restart` by terminating and relaunching the
  /// adapter but by calling the `restart` request.
  eAdapterFeatureRestartRequest,
  /// The debug adapter supports the `setExpression` request.
  eAdapterFeatureSetExpression,
  /// The debug adapter supports setting a variable to a value.
  eAdapterFeatureSetVariable,
  /// The debug adapter supports the `singleThread` property on the execution
  /// requests (`continue`, `next`, `stepIn`, `stepOut`, `reverseContinue`,
  /// `stepBack`).
  eAdapterFeatureSingleThreadExecutionRequests,
  /// The debug adapter supports stepping back via the `stepBack` and
  /// `reverseContinue` requests.
  eAdapterFeatureStepBack,
  /// The debug adapter supports the `stepInTargets` request.
  eAdapterFeatureStepInTargetsRequest,
  /// The debug adapter supports stepping granularities (argument
  /// `granularity`) for the stepping requests.
  eAdapterFeatureSteppingGranularity,
  /// The debug adapter supports the `terminate` request.
  eAdapterFeatureTerminateRequest,
  /// The debug adapter supports the `terminateThreads` request.
  eAdapterFeatureTerminateThreadsRequest,
  /// The debug adapter supports the `suspendDebuggee` attribute on the
  /// `disconnect` request.
  eAdapterFeatureSuspendDebuggee,
  /// The debug adapter supports a `format` attribute on the `stackTrace`,
  /// `variables`, and `evaluate` requests.
  eAdapterFeatureValueFormattingOptions,
  /// The debug adapter supports the `writeMemory` request.
  eAdapterFeatureWriteMemoryRequest,
  /// The debug adapter supports the `terminateDebuggee` attribute on the
  /// `disconnect` request.
  eAdapterFeatureTerminateDebuggee,
};

/// Information about the capabilities of a debug adapter.
struct Capabilities {
  /// The supported features for this adapter.
  llvm::DenseSet<AdapterFeature> supportedFeatures;

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

enum PresentationHint : unsigned {
  ePresentationHintNormal,
  ePresentationHintEmphasize,
  ePresentationHintDeemphasize,
};
llvm::json::Value toJSON(PresentationHint hint);

/// A `Source` is a descriptor for source code. It is returned from the debug
/// adapter as part of a `StackFrame` and it is used by clients when specifying
/// breakpoints.
struct Source {
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
llvm::json::Value toJSON(const Source &);

/// The granularity of one `step` in the stepping requests `next`, `stepIn`,
/// `stepOut` and `stepBack`.
enum SteppingGranularity : unsigned {
  /// The step should allow the program to run until the current statement has
  /// finished executing. The meaning of a statement is determined by the
  /// adapter and it may be considered equivalent to a line. For example
  /// `for(int i = 0; i < 10; i++)` could be considered to have 3 statements
  /// `int i = 0`, `i < 10`, and `i++`.
  eSteppingGranularityStatement,
  /// The step should allow the program to run until the current source line has
  /// executed.
  eSteppingGranularityLine,
  /// The step should allow one instruction to execute (e.g. one x86
  /// instruction).
  eSteppingGranularityInstruction,
};
bool fromJSON(const llvm::json::Value &, SteppingGranularity &,
              llvm::json::Path);

/// Provides formatting information for a value.
struct ValueFormat {
  /// Display the value in hex.
  std::optional<bool> hex;
};
bool fromJSON(const llvm::json::Value &, ValueFormat &, llvm::json::Path);

/// Properties of a breakpoint location returned from the `breakpointLocations`
/// request.
struct BreakpointLocation {
  /// Start line of breakpoint location.
  uint32_t line;

  /// The start position of a breakpoint location. Position is measured in
  /// UTF-16 code units and the client capability `columnsStartAt1` determines
  /// whether it is 0- or 1-based.
  std::optional<uint32_t> column;

  /// The end line of breakpoint location if the location covers a range.
  std::optional<uint32_t> endLine;

  /// The end position of a breakpoint location (if the location covers a
  /// range). Position is measured in UTF-16 code units and the client
  /// capability `columnsStartAt1` determines whether it is 0- or 1-based.
  std::optional<uint32_t> endColumn;
};
llvm::json::Value toJSON(const BreakpointLocation &);

/// A machine-readable explanation of why a breakpoint may not be verified.
enum class BreakpointReason : unsigned {
  /// Indicates a breakpoint might be verified in the future, but
  /// the adapter cannot verify it in the current state.
  eBreakpointReasonPending,
  /// Indicates a breakpoint was not able to be verified, and the
  /// adapter does not believe it can be verified without intervention.
  eBreakpointReasonFailed,
};
llvm::json::Value toJSON(const BreakpointReason &);

/// Information about a breakpoint created in `setBreakpoints`,
/// `setFunctionBreakpoints`, `setInstructionBreakpoints`, or
/// `setDataBreakpoints` requests.
struct Breakpoint {
  /// The identifier for the breakpoint. It is needed if breakpoint events are
  /// used to update or remove breakpoints.
  std::optional<int> id;

  /// If true, the breakpoint could be set (but not necessarily at the desired
  /// location).
  bool verified = false;

  /// A message about the state of the breakpoint.
  /// This is shown to the user and can be used to explain why a breakpoint
  /// could not be verified.
  std::optional<std::string> message;

  /// The source where the breakpoint is located.
  std::optional<Source> source;

  /// The start line of the actual range covered by the breakpoint.
  std::optional<uint32_t> line;

  /// Start position of the source range covered by the breakpoint. It is
  /// measured in UTF-16 code units and the client capability `columnsStartAt1`
  /// determines whether it is 0- or 1-based.
  std::optional<uint32_t> column;

  /// The end line of the actual range covered by the breakpoint.
  std::optional<uint32_t> endLine;

  /// End position of the source range covered by the breakpoint. It is measured
  /// in UTF-16 code units and the client capability `columnsStartAt1`
  /// determines whether it is 0- or 1-based. If no end line is given, then the
  /// end column is assumed to be in the start line.
  std::optional<uint32_t> endColumn;

  /// A memory reference to where the breakpoint is set.
  std::optional<std::string> instructionReference;

  /// The offset from the instruction reference.
  /// This can be negative.
  std::optional<int32_t> offset;

  /// A machine-readable explanation of why a breakpoint may not be verified. If
  /// a breakpoint is verified or a specific reason is not known, the adapter
  /// should omit this property.
  std::optional<BreakpointReason> reason;
};
llvm::json::Value toJSON(const Breakpoint &);

/// Properties of a breakpoint or logpoint passed to the `setBreakpoints`
/// request
struct SourceBreakpoint {
  /// The source line of the breakpoint or logpoint.
  uint32_t line = LLDB_INVALID_LINE_NUMBER;

  /// Start position within source line of the breakpoint or logpoint. It is
  /// measured in UTF-16 code units and the client capability `columnsStartAt1`
  /// determines whether it is 0- or 1-based.
  std::optional<uint32_t> column;

  /// The expression for conditional breakpoints.
  /// It is only honored by a debug adapter if the corresponding capability
  /// `supportsConditionalBreakpoints` is true.
  std::optional<std::string> condition;

  /// The expression that controls how many hits of the breakpoint are ignored.
  /// The debug adapter is expected to interpret the expression as needed.
  /// The attribute is only honored by a debug adapter if the corresponding
  /// capability `supportsHitConditionalBreakpoints` is true.
  /// If both this property and `condition` are specified, `hitCondition` should
  /// be evaluated only if the `condition` is met, and the debug adapter should
  /// stop only if both conditions are met.
  std::optional<std::string> hitCondition;

  /// If this attribute exists and is non-empty, the debug adapter must not
  /// 'break' (stop)
  /// but log the message instead. Expressions within `{}` are interpolated.
  /// The attribute is only honored by a debug adapter if the corresponding
  /// capability `supportsLogPoints` is true.
  /// If either `hitCondition` or `condition` is specified, then the message
  /// should only be logged if those conditions are met.
  std::optional<std::string> logMessage;

  /// The mode of this breakpoint. If defined, this must be one of the
  /// `breakpointModes` the debug adapter advertised in its `Capabilities`.
  std::optional<std::string> mode;
};
bool fromJSON(const llvm::json::Value &, SourceBreakpoint &, llvm::json::Path);

/// Properties of a breakpoint passed to the `setFunctionBreakpoints` request.
struct FunctionBreakpoint {
  /// The name of the function.
  std::string name;

  /// An expression for conditional breakpoints.
  /// It is only honored by a debug adapter if the corresponding capability
  /// `supportsConditionalBreakpoints` is true.
  std::optional<std::string> condition;

  /// An expression that controls how many hits of the breakpoint are ignored.
  /// The debug adapter is expected to interpret the expression as needed.
  /// The attribute is only honored by a debug adapter if the corresponding
  /// capability `supportsHitConditionalBreakpoints` is true.
  std::optional<std::string> hitCondition;
};
bool fromJSON(const llvm::json::Value &, FunctionBreakpoint &,
              llvm::json::Path);

/// This enumeration defines all possible access types for data breakpoints.
/// Values: ‘read’, ‘write’, ‘readWrite’
enum DataBreakpointAccessType : unsigned {
  eDataBreakpointAccessTypeRead,
  eDataBreakpointAccessTypeWrite,
  eDataBreakpointAccessTypeReadWrite
};
bool fromJSON(const llvm::json::Value &, DataBreakpointAccessType &,
              llvm::json::Path);
llvm::json::Value toJSON(const DataBreakpointAccessType &);

/// Properties of a data breakpoint passed to the `setDataBreakpoints` request.
struct DataBreakpointInfo {
  /// An id representing the data. This id is returned from the
  /// `dataBreakpointInfo` request.
  std::string dataId;

  /// The access type of the data.
  std::optional<DataBreakpointAccessType> accessType;

  /// An expression for conditional breakpoints.
  std::optional<std::string> condition;

  /// An expression that controls how many hits of the breakpoint are ignored.
  /// The debug adapter is expected to interpret the expression as needed.
  std::optional<std::string> hitCondition;
};
bool fromJSON(const llvm::json::Value &, DataBreakpointInfo &,
              llvm::json::Path);

/// Properties of a breakpoint passed to the `setInstructionBreakpoints` request
struct InstructionBreakpoint {
  /// The instruction reference of the breakpoint.
  /// This should be a memory or instruction pointer reference from an
  /// `EvaluateResponse`, `Variable`, `StackFrame`, `GotoTarget`, or
  /// `Breakpoint`.
  std::string instructionReference;

  /// The offset from the instruction reference in bytes.
  /// This can be negative.
  std::optional<int32_t> offset;

  /// An expression for conditional breakpoints.
  /// It is only honored by a debug adapter if the corresponding capability
  /// `supportsConditionalBreakpoints` is true.
  std::optional<std::string> condition;

  /// An expression that controls how many hits of the breakpoint are ignored.
  /// The debug adapter is expected to interpret the expression as needed.
  /// The attribute is only honored by a debug adapter if the corresponding
  /// capability `supportsHitConditionalBreakpoints` is true.
  std::optional<std::string> hitCondition;

  /// The mode of this breakpoint. If defined, this must be one of the
  /// `breakpointModes` the debug adapter advertised in its `Capabilities`.
  std::optional<std::string> mode;
};
bool fromJSON(const llvm::json::Value &, InstructionBreakpoint &,
              llvm::json::Path);

} // namespace lldb_dap::protocol

#endif
