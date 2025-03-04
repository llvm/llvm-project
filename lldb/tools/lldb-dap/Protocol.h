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

// "Request": {
//   "allOf": [ { "$ref": "#/definitions/ProtocolMessage" }, {
//     "type": "object",
//     "description": "A client or debug adapter initiated request.",
//     "properties": {
//       "type": {
//         "type": "string",
//         "enum": [ "request" ]
//       },
//       "command": {
//         "type": "string",
//         "description": "The command to execute."
//       },
//       "arguments": {
//         "type": [ "array", "boolean", "integer", "null", "number" , "object",
//         "string" ], "description": "Object containing arguments for the
//         command."
//       }
//     },
//     "required": [ "type", "command" ]
//   }]
// },
struct Request {
  int64_t seq;
  std::string command;
  std::optional<llvm::json::Value> rawArguments;
};
llvm::json::Value toJSON(const Request &);
bool fromJSON(const llvm::json::Value &, Request &, llvm::json::Path);

// "Event": {
//   "allOf": [ { "$ref": "#/definitions/ProtocolMessage" }, {
//     "type": "object",
//     "description": "A debug adapter initiated event.",
//     "properties": {
//       "type": {
//         "type": "string",
//         "enum": [ "event" ]
//       },
//       "event": {
//         "type": "string",
//         "description": "Type of event."
//       },
//       "body": {
//         "type": [ "array", "boolean", "integer", "null", "number" , "object",
//         "string" ], "description": "Event-specific information."
//       }
//     },
//     "required": [ "type", "event" ]
//   }]
// },
struct Event {
  std::string event;
  std::optional<llvm::json::Value> rawBody;
};
llvm::json::Value toJSON(const Event &);
bool fromJSON(const llvm::json::Value &, Event &, llvm::json::Path);

// "Response" : {
//   "allOf" : [
//     {"$ref" : "#/definitions/ProtocolMessage"}, {
//       "type" : "object",
//       "description" : "Response for a request.",
//       "properties" : {
//         "type" : {"type" : "string", "enum" : ["response"]},
//         "request_seq" : {
//           "type" : "integer",
//           "description" : "Sequence number of the corresponding request."
//         },
//         "success" : {
//           "type" : "boolean",
//           "description" :
//               "Outcome of the request.\nIf true, the request was successful "
//               "and the `body` attribute may contain the result of the "
//               "request.\nIf the value is false, the attribute `message` "
//               "contains the error in short form and the `body` may contain "
//               "additional information (see `ErrorResponse.body.error`)."
//         },
//         "command" :
//             {"type" : "string", "description" : "The command requested."},
//         "message" : {
//           "type" : "string",
//           "description" :
//               "Contains the raw error in short form if `success` is "
//               "false.\nThis raw error might be interpreted by the client and
//               " "is not shown in the UI.\nSome predefined values exist.",
//           "_enum" : [ "cancelled", "notStopped" ],
//           "enumDescriptions" : [
//             "the request was cancelled.",
//             "the request may be retried once the adapter is in a 'stopped'"
//             "state."
//           ]
//         },
//         "body" : {
//           "type" : [
//             "array", "boolean", "integer", "null", "number", "object",
//             "string"
//           ],
//           "description" : "Contains request result if success is true and "
//                           "error details if success is false."
//         }
//       },
//       "required" : [ "type", "request_seq", "success", "command" ]
//     }
//   ]
// }
struct Response {
  enum class Message {
    cancelled,
    notStopped,
  };

  int64_t request_seq;
  std::string command;
  bool success;
  // FIXME: Migrate usage of fallback string to ErrorMessage
  std::optional<std::variant<Message, std::string>> message;
  std::optional<llvm::json::Value> rawBody;
};
bool fromJSON(const llvm::json::Value &, Response &, llvm::json::Path);
llvm::json::Value toJSON(const Response &);

// "Message": {
//   "type": "object",
//   "description": "A structured message object. Used to return errors from
//   requests.", "properties": {
//     "id": {
//       "type": "integer",
//       "description": "Unique (within a debug adapter implementation)
//       identifier for the message. The purpose of these error IDs is to help
//       extension authors that have the requirement that every user visible
//       error message needs a corresponding error number, so that users or
//       customer support can find information about the specific error more
//       easily."
//     },
//     "format": {
//       "type": "string",
//       "description": "A format string for the message. Embedded variables
//       have the form `{name}`.\nIf variable name starts with an underscore
//       character, the variable does not contain user data (PII) and can be
//       safely used for telemetry purposes."
//     },
//     "variables": {
//       "type": "object",
//       "description": "An object used as a dictionary for looking up the
//       variables in the format string.", "additionalProperties": {
//         "type": "string",
//         "description": "All dictionary values must be strings."
//       }
//     },
//     "sendTelemetry": {
//       "type": "boolean",
//       "description": "If true send to telemetry."
//     },
//     "showUser": {
//       "type": "boolean",
//       "description": "If true show user."
//     },
//     "url": {
//       "type": "string",
//       "description": "A url where additional information about this message
//       can be found."
//     },
//     "urlLabel": {
//       "type": "string",
//       "description": "A label that is presented to the user as the UI for
//       opening the url."
//     }
//   },
//   "required": [ "id", "format" ]
// },
struct ErrorMessage {
  uint64_t id;
  std::string format;
  std::optional<std::map<std::string, std::string>> variables;
  bool sendTelemetry;
  bool showUser;
  std::optional<std::string> url;
  std::optional<std::string> urlLabel;
};
bool fromJSON(const llvm::json::Value &, ErrorMessage &, llvm::json::Path);
llvm::json::Value toJSON(const ErrorMessage &);

// "ProtocolMessage": {
//   "type": "object",
//   "title": "Base Protocol",
//   "description": "Base class of requests, responses, and events.",
//   "properties": {
//     "seq": {
//       "type": "integer",
//       "description": "Sequence number of the message (also known as
//       message ID). The `seq` for the first message sent by a client or
//       debug adapter is 1, and for each subsequent message is 1 greater
//       than the previous message sent by that actor. `seq` can be used to
//       order requests, responses, and events, and to associate requests
//       with their corresponding responses. For protocol messages of type
//       `request` the sequence number can be used to cancel the request."
//     },
//     "type": {
//       "type": "string",
//       "description": "Message type.",
//       "_enum": [ "request", "response", "event" ]
//     }
//   },
//   "required": [ "seq", "type" ]
// },
using Message = std::variant<Request, Response, Event>;
bool fromJSON(const llvm::json::Value &, Message &, llvm::json::Path);
llvm::json::Value toJSON(const Message &);

} // namespace lldb_dap::protocol

#endif
