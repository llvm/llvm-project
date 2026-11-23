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

#ifndef LLDB_TOOLS_LLDB_DAP_PROTOCOL_PROTOCOL_BASE_H
#define LLDB_TOOLS_LLDB_DAP_PROTOCOL_PROTOCOL_BASE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <optional>
#include <string>
#include <variant>

namespace lldb_dap::protocol {

// MARK: Base Protocol

/// Message unique identifier type.
using Id = uint64_t;

/// A unique identifier that indicates the `seq` field should be calculated by
/// the current session.
static constexpr Id kCalculateSeq = UINT64_MAX;

/// A wrapper around a 'std::string' to ensure the contents are valid utf8
/// during serialization.
class String {
public:
  String() = default;
  String(const std::string &str) : m_str(str) {}
  String(llvm::StringRef str) : m_str(str.str()) {}
  String(const char *str) : m_str(str) {}
  String(const llvm::formatv_object_base &payload) : m_str(payload.str()) {}
  String(const String &) = default;
  String(String &&str) : m_str(std::move(str.m_str)) {}
  String(std::string &&str) : m_str(std::move(str)) {}

  ~String() = default;

  String &operator=(const String &) = default;
  String &operator=(String &&Other) {
    m_str = std::move(Other.m_str);
    return *this;
  }

  /// Conversion Operators
  /// @{
  operator llvm::Twine() const { return m_str; }
  operator std::string() const { return m_str; }
  operator llvm::StringRef() const { return {m_str}; }
  /// @}

  void clear() { m_str.clear(); }
  bool empty() const { return m_str.empty(); }
  const char *c_str() const { return m_str.c_str(); }
  const char *data() const { return m_str.data(); }
  std::string str() const { return m_str; }

  inline String &operator+=(const String &RHS) {
    m_str += RHS.m_str;
    return *this;
  }

  friend String operator+(const String &LHS, const String &RHS) {
    return {LHS.m_str + RHS.m_str};
  }

  /// @name String Comparision Operators
  /// @{

  friend bool operator==(const String &LHS, const String &RHS) {
    return llvm::StringRef(LHS) == llvm::StringRef(RHS);
  }

  friend bool operator!=(const String &LHS, const String &RHS) {
    return !(LHS == RHS);
  }

  friend bool operator<(const String &LHS, const String &RHS) {
    return llvm::StringRef(LHS) < llvm::StringRef(RHS);
  }

  friend bool operator<=(const String &LHS, const String &RHS) {
    return llvm::StringRef(LHS) <= llvm::StringRef(RHS);
  }

  friend bool operator>(const String &LHS, const String &RHS) {
    return llvm::StringRef(LHS) > llvm::StringRef(RHS);
  }

  friend bool operator>=(const String &LHS, const String &RHS) {
    return llvm::StringRef(LHS) >= llvm::StringRef(RHS);
  }

  /// @}

private:
  std::string m_str;
};
llvm::json::Value toJSON(const String &s);
bool fromJSON(const llvm::json::Value &, String &, llvm::json::Path);

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const String &S) {
  OS << S.str();
  return OS;
}

/// A client or debug adapter initiated request.
struct Request {
  /// The command to execute.
  String command;

  /// Object containing arguments for the command.
  ///
  /// Request handlers are expected to validate the arguments, which is handled
  /// by `RequestHandler`.
  std::optional<llvm::json::Value> arguments = std::nullopt;

  /// Sequence number of the message (also known as message ID). The `seq` for
  /// the first message sent by a client or debug adapter is 1, and for each
  /// subsequent message is 1 greater than the previous message sent by that
  /// actor. `seq` can be used to order requests, responses, and events, and to
  /// associate requests with their corresponding responses. For protocol
  /// messages of type `request` the sequence number can be used to cancel the
  /// request.
  Id seq = kCalculateSeq;
};
llvm::json::Value toJSON(const Request &);
bool fromJSON(const llvm::json::Value &, Request &, llvm::json::Path);
bool operator==(const Request &, const Request &);

/// A debug adapter initiated event.
struct Event {
  /// Type of event.
  String event;

  /// Event-specific information.
  std::optional<llvm::json::Value> body = std::nullopt;

  /// Sequence number of the message (also known as message ID). The `seq` for
  /// the first message sent by a client or debug adapter is 1, and for each
  /// subsequent message is 1 greater than the previous message sent by that
  /// actor. `seq` can be used to order requests, responses, and events, and to
  /// associate requests with their corresponding responses. For protocol
  /// messages of type `request` the sequence number can be used to cancel the
  /// request.
  Id seq = kCalculateSeq;
};
llvm::json::Value toJSON(const Event &);
bool fromJSON(const llvm::json::Value &, Event &, llvm::json::Path);
bool operator==(const Event &, const Event &);

enum ResponseMessage : unsigned {
  /// The request was cancelled
  eResponseMessageCancelled,
  /// The request may be retried once the adapter is in a 'stopped' state
  eResponseMessageNotStopped,
};

/// Response for a request.
struct Response {
  /// Sequence number of the corresponding request.
  Id request_seq = 0;

  /// The command requested.
  String command;

  /// Outcome of the request. If true, the request was successful and the `body`
  /// attribute may contain the result of the request. If the value is false,
  /// the attribute `message` contains the error in short form and the `body`
  /// may contain additional information (see `ErrorMessage`).
  bool success = false;

  // FIXME: Migrate usage of fallback string to ErrorMessage

  /// Contains the raw error in short form if `success` is false. This raw error
  /// might be interpreted by the client and is not shown in the UI. Some
  /// predefined values exist.
  std::optional<std::variant<ResponseMessage, String>> message = std::nullopt;

  /// Contains request result if success is true and error details if success is
  /// false.
  ///
  /// Request handlers are expected to build an appropriate body, see
  /// `RequestHandler`.
  std::optional<llvm::json::Value> body = std::nullopt;

  /// Sequence number of the message (also known as message ID). The `seq` for
  /// the first message sent by a client or debug adapter is 1, and for each
  /// subsequent message is 1 greater than the previous message sent by that
  /// actor. `seq` can be used to order requests, responses, and events, and to
  /// associate requests with their corresponding responses. For protocol
  /// messages of type `request` the sequence number can be used to cancel the
  /// request.
  Id seq = kCalculateSeq;
};
bool fromJSON(const llvm::json::Value &, Response &, llvm::json::Path);
llvm::json::Value toJSON(const Response &);
bool operator==(const Response &, const Response &);

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
  String format;

  /// An object used as a dictionary for looking up the variables in the format
  /// string.
  std::map<String, String> variables;

  /// If true send to telemetry.
  bool sendTelemetry = false;

  /// If true show user.
  bool showUser = false;

  /// A url where additional information about this message can be found.
  String url;

  /// A label that is presented to the user as the UI for opening the url.
  String urlLabel;
};
bool fromJSON(const llvm::json::Value &, ErrorMessage &, llvm::json::Path);
llvm::json::Value toJSON(const ErrorMessage &);

/// An individual protocol message of requests, responses, and events.
using Message = std::variant<Request, Response, Event>;
bool fromJSON(const llvm::json::Value &, Message &, llvm::json::Path);
llvm::json::Value toJSON(const Message &);
bool operator==(const Message &, const Message &);

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
