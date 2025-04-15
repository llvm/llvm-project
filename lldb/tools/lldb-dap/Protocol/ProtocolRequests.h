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
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/JSON.h"
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
};

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

} // namespace lldb_dap::protocol

#endif
