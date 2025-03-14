//===-- Protocol.h --------------------------------------------------------===//
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

#ifndef LLDB_TOOLS_LLDB_DAP_PROTOCOL_H
#define LLDB_TOOLS_LLDB_DAP_PROTOCOL_H

#include "llvm/Support/JSON.h"
#include <cstdint>
#include <optional>
#include <string>
#include <variant>

namespace lldb_dap::protocol {

// MARK: Base Protocol

/// A client or debug adapter initiated request.
struct Request {
  /// Sequence number of the message (also known as message ID). The `seq` for
  /// the first message sent by a client or debug adapter is 1, and for each
  /// subsequent message is 1 greater than the previous message sent by that
  /// actor. `seq` can be used to order requests, responses, and events, and to
  /// associate requests with their corresponding responses. For protocol
  /// messages of type `request` the sequence number can be used to cancel the
  /// request.
  int64_t seq;

  /// The command to execute.
  std::string command;

  /// Object containing arguments for the command.
  std::optional<llvm::json::Value> rawArguments;
};
llvm::json::Value toJSON(const Request &);
bool fromJSON(const llvm::json::Value &, Request &, llvm::json::Path);

/// A debug adapter initiated event.
struct Event {
  /// Type of event.
  std::string event;

  /// Event-specific information.
  std::optional<llvm::json::Value> rawBody;
};
llvm::json::Value toJSON(const Event &);
bool fromJSON(const llvm::json::Value &, Event &, llvm::json::Path);

/// Response for a request.
struct Response {
  enum class Message {
    /// The request was cancelled
    cancelled,
    /// The request may be retried once the adapter is in a 'stopped' state
    notStopped,
  };

  /// Sequence number of the corresponding request.
  int64_t request_seq;

  /// The command requested.
  std::string command;

  /// Outcome of the request. If true, the request was successful and the `body`
  /// attribute may contain the result of the request. If the value is false,
  /// the attribute `message` contains the error in short form and the `body`
  /// may contain additional information (see `ErrorMessage`).
  bool success;

  // FIXME: Migrate usage of fallback string to ErrorMessage

  /// Contains the raw error in short form if `success` is false. This raw error
  /// might be interpreted by the client and is not shown in the UI. Some
  /// predefined values exist.
  std::optional<std::variant<Message, std::string>> message;

  /// Contains request result if success is true and error details if success is
  /// false.
  std::optional<llvm::json::Value> rawBody;
};
bool fromJSON(const llvm::json::Value &, Response &, llvm::json::Path);
llvm::json::Value toJSON(const Response &);

/// A structured message object. Used to return errors from requests.
struct ErrorMessage {
  /// Unique (within a debug adapter implementation) identifier for the message.
  /// The purpose of these error IDs is to help extension authors that have the
  /// requirement that every user visible error message needs a corresponding
  /// error number, so that users or customer support can find information about
  /// the specific error more easily.
  uint64_t id;

  /// A format string for the message. Embedded variables have the form
  /// `{name}`. If variable name starts with an underscore character, the
  /// variable does not contain user data (PII) and can be safely used for
  /// telemetry purposes.
  std::string format;

  /// An object used as a dictionary for looking up the variables in the format
  /// string.
  std::optional<std::map<std::string, std::string>> variables;

  /// If true send to telemetry.
  bool sendTelemetry;

  /// If true show user.
  bool showUser;

  /// A url where additional information about this message can be found.
  std::optional<std::string> url;

  /// A label that is presented to the user as the UI for opening the url.
  std::optional<std::string> urlLabel;
};
bool fromJSON(const llvm::json::Value &, ErrorMessage &, llvm::json::Path);
llvm::json::Value toJSON(const ErrorMessage &);

/// An individual protocol message of requests, responses, and events.
using Message = std::variant<Request, Response, Event>;
bool fromJSON(const llvm::json::Value &, Message &, llvm::json::Path);
llvm::json::Value toJSON(const Message &);

// MARK: Types

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

// MARK: Requests

/// This is just an acknowledgement, so no body field is required.
using VoidResponse = std::monostate;

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
