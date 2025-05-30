//===-- ProtocolBase.h ----------------------------------------------------===//
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
  ///
  /// Request handlers are expected to validate the arguments, which is handled
  /// by `RequestHandler`.
  std::optional<llvm::json::Value> arguments;
};
llvm::json::Value toJSON(const Request &);
bool fromJSON(const llvm::json::Value &, Request &, llvm::json::Path);

/// A debug adapter initiated event.
struct Event {
  /// Type of event.
  std::string event;

  /// Event-specific information.
  std::optional<llvm::json::Value> body;
};
llvm::json::Value toJSON(const Event &);
bool fromJSON(const llvm::json::Value &, Event &, llvm::json::Path);

enum ResponseMessage : unsigned {
  /// The request was cancelled
  eResponseMessageCancelled,
  /// The request may be retried once the adapter is in a 'stopped' state
  eResponseMessageNotStopped,
};

/// Response for a request.
struct Response {
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
  std::optional<std::variant<ResponseMessage, std::string>> message;

  /// Contains request result if success is true and error details if success is
  /// false.
  ///
  /// Request handlers are expected to build an appropriate body, see
  /// `RequestHandler`.
  std::optional<llvm::json::Value> body;
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
  uint64_t id = 0;

  /// A format string for the message. Embedded variables have the form
  /// `{name}`. If variable name starts with an underscore character, the
  /// variable does not contain user data (PII) and can be safely used for
  /// telemetry purposes.
  std::string format;

  /// An object used as a dictionary for looking up the variables in the format
  /// string.
  std::optional<std::map<std::string, std::string>> variables;

  /// If true send to telemetry.
  bool sendTelemetry = false;

  /// If true show user.
  bool showUser = false;

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

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Message &V) {
  OS << toJSON(V);
  return OS;
}

/// On error (whenever `success` is false), the body can provide more details.
struct ErrorResponseBody {
  /// A structured error message.
  std::optional<ErrorMessage> error;
};
llvm::json::Value toJSON(const ErrorResponseBody &);

/// This is a placehold for requests with an empty, null or undefined arguments.
using EmptyArguments = std::optional<std::monostate>;

/// This is just an acknowledgement, so no body field is required.
using VoidResponse = llvm::Error;

} // namespace lldb_dap::protocol

#endif
