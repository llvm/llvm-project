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
#include "llvm/Support/JSON.h"
#include <cstdint>
#include <optional>
#include <string>

namespace lldb_dap::protocol {

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

/// Arguments for `goto` request.
struct GotoArguments {
  /// Set the goto target for this thread.
  uint64_t threadId;

  /// The location where the debuggee will continue to run.
  uint64_t targetId;
};
bool fromJSON(const llvm::json::Value &, GotoArguments &, llvm::json::Path);

/// Response to goto request. This is just an acknowledgement, so no
/// body field is required.
using GotoResponseBody = VoidResponse;

/// Arguments for `gotoTargets` request.
struct GotoTargetsArguments {
  /// The source location for which the goto targets are determined.
  Source source;

  /// The line location for which the goto targets are determined.
  uint64_t line;

  /// The position within `line` for which the goto targets are determined. It
  /// is
  ///  measured in UTF-16 code units and the client capability `columnsStartAt1`
  ///  determines whether it is 0- or 1-based.
  std::optional<uint64_t> column;
};
bool fromJSON(const llvm::json::Value &, GotoTargetsArguments &,
              llvm::json::Path);

struct GotoTargetsResponseBody {
  /// The possible goto targets of the specified location.
  llvm::SmallVector<GotoTarget> targets;
};
llvm::json::Value toJSON(const GotoTargetsResponseBody &);

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
