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
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>

namespace lldb_dap {
namespace protocol {

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

  /// lldb-dap specific extension on the 'terminated' event specifically.
  std::optional<llvm::json::Value> statistics;
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
//             "the request was cancelled.", "the request may be retried once
//             the "
//                                           "adapter is in a 'stopped' state."
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
  int64_t request_seq;
  bool success;
  std::string command;
  std::optional<std::string> message;
  std::optional<llvm::json::Value> rawBody;
};
bool fromJSON(const llvm::json::Value &, Response &, llvm::json::Path);
llvm::json::Value toJSON(const Response &);

// A void response body for any response without a specific value.
using VoidResponseBody = std::nullptr_t;

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

} // namespace protocol
} // namespace lldb_dap

#endif
