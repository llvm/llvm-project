//===-- Protocol.h ----------------------------------------------*- C++ -*-===//
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
#include <vector>
#include <map>

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
// struct ProtocolMessage {
//   MessageType type;
//   int64_t seq;
// };
// enum class MessageType { request, event, response };
using ProtocolMessage = std::variant<Request, Response, Event>;
bool fromJSON(const llvm::json::Value &, ProtocolMessage &, llvm::json::Path);
llvm::json::Value toJSON(const ProtocolMessage &);

// "CancelRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "The `cancel` request is used by the client in two
//     situations:\n- to indicate that it is no longer interested in the result
//     produced by a specific request issued earlier\n- to cancel a progress
//     sequence.\nClients should only call this request if the corresponding
//     capability `supportsCancelRequest` is true.\nThis request has a hint
//     characteristic: a debug adapter can only be expected to make a 'best
//     effort' in honoring this request but there are no guarantees.\nThe
//     `cancel` request may return an error if it could not cancel an operation
//     but a client should refrain from presenting this error to end users.\nThe
//     request that got cancelled still needs to send a response back. This can
//     either be a normal result (`success` attribute true) or an error response
//     (`success` attribute false and the `message` set to
//     `cancelled`).\nReturning partial results from a cancelled request is
//     possible but please note that a client has no generic way for detecting
//     that a response is partial or not.\nThe progress that got cancelled still
//     needs to send a `progressEnd` event back.\n A client should not assume
//     that progress just got cancelled after sending the `cancel` request.",
//     "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "cancel" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/CancelArguments"
//       }
//     },
//     "required": [ "command" ]
//   }]
// },
// "CancelArguments": {
//   "type": "object",
//   "description": "Arguments for `cancel` request.",
//   "properties": {
//     "requestId": {
//       "type": "integer",
//       "description": "The ID (attribute `seq`) of the request to cancel. If
//       missing no request is cancelled.\nBoth a `requestId` and a `progressId`
//       can be specified in one request."
//     },
//     "progressId": {
//       "type": "string",
//       "description": "The ID (attribute `progressId`) of the progress to
//       cancel. If missing no progress is cancelled.\nBoth a `requestId` and a
//       `progressId` can be specified in one request."
//     }
//   }
// },
struct CancelArguments {
  std::optional<int64_t> requestId;
  std::optional<int64_t> progressId;
};
bool fromJSON(const llvm::json::Value &, CancelArguments &, llvm::json::Path);

// "CancelResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to `cancel` request. This is just an
//     acknowledgement, so no body field is required."
//   }]
// },
using CancelResponseBody = VoidResponseBody;

// "Message": {
//   "type": "object",
//   "description": "A structured message object. Used to return errors from
//   requests.", "properties": {
//     "id": {
//       "type": "integer",
//       "description": "Unique (within a debug adapter implementation)
//       identifier for the message. The purpose of these error IDs is to
//       help extension authors that have the requirement that every user
//       visible error message needs a corresponding error number, so that
//       users or customer support can find information about the specific
//       error more easily."
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
//       "description": "A url where additional information about this
//       message can be found."
//     },
//     "urlLabel": {
//       "type": "string",
//       "description": "A label that is presented to the user as the UI for
//       opening the url."
//     }
//   },
//   "required": [ "id", "format" ]
// },
struct Message {
  int id;
  std::string format;
  std::optional<bool> showUser;
  std::optional<std::map<std::string, std::string>> variables;
  std::optional<bool> sendTelemetry;
  std::optional<std::string> url;
  std::optional<std::string> urlLabel;
};
llvm::json::Value toJSON(const Message &);

// "ErrorResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "On error (whenever `success` is false), the body can
//     provide more details.", "properties": {
//       "body": {
//         "type": "object",
//         "properties": {
//           "error": {
//             "$ref": "#/definitions/Message",
//             "description": "A structured error message."
//           }
//         }
//       }
//     },
//     "required": [ "body" ]
//   }]
// },
struct ErrorResponseBody {
  std::optional<Message> error;
};
llvm::json::Value toJSON(const ErrorResponseBody &);

// MARK: Types

// "Source" : {
//   "type" : "object",
//   "description" : "A `Source` is a descriptor for source
//   code.\nIt is returned "
//                   "from the debug adapter as part of a
//                   `StackFrame` and it is " "used by clients when
//                   specifying breakpoints.",
//   "properties" : {
//     "name" : {
//       "type" : "string",
//       "description" : "The short name of the source. Every
//       source returned "
//                       "from the debug adapter has a name.\nWhen
//                       sending a " "source to the debug adapter
//                       this name is optional."
//     },
//     "path" : {
//       "type" : "string",
//       "description" :
//           "The path of the source to be shown in the UI.\nIt is
//           only used to " "locate and load the content of the
//           source if no `sourceReference` " "is specified (or its
//           value is 0)."
//     },
//     "sourceReference" : {
//       "type" : "integer",
//       "description" :
//           "If the value > 0 the contents of the source must be
//           retrieved " "through the `source` request (even if a
//           path is specified).\nSince " "a `sourceReference` is
//           only valid for a session, it can not be used " "to
//           persist a source.\nThe value should be less than or
//           equal to " "2147483647 (2^31-1)."
//     },
//     "presentationHint" : {
//       "type" : "string",
//       "description" :
//           "A hint for how to present the source in the UI.\nA
//           value of "
//           "`deemphasize` can be used to indicate that the source
//           is not " "available or that it is skipped on
//           stepping.",
//       "enum" : [ "normal", "emphasize", "deemphasize" ]
//     },
//     "origin" : {
//       "type" : "string",
//       "description" : "The origin of this source. For example,
//       'internal "
//                       "module', 'inlined content from source
//                       map', etc."
//     },
//     "sources" : {
//       "type" : "array",
//       "items" : {"$ref" : "#/definitions/Source"},
//       "description" : "A list of sources that are related to
//       this source. "
//                       "These may be the source that generated
//                       this source."
//     },
//     "adapterData" : {
//       "type" : [
//         "array", "boolean", "integer", "null", "number",
//         "object", "string"
//       ],
//       "description" :
//           "Additional data that a debug adapter might want to
//           loop through the " "client.\nThe client should leave
//           the data intact and persist it " "across sessions. The
//           client should not interpret the data."
//     },
//     "checksums" : {
//       "type" : "array",
//       "items" : {"$ref" : "#/definitions/Checksum"},
//       "description" : "The checksums associated with this file."
//     }
//   }
struct Source {
  enum class PresentationHint { normal, emphasize, deemphasize };

  std::optional<std::string> name;
  std::optional<std::string> path;
  std::optional<int64_t> sourceReference;
  std::optional<PresentationHint> presentationHint;
  std::optional<std::string> origin;

  // Unsupported fields: adapterData, checksums
};
bool fromJSON(const llvm::json::Value &, Source &, llvm::json::Path);
llvm::json::Value toJSON(const Source &);

// "StackFrame" : {
//   "type" : "object",
//   "description" : "A Stackframe contains the source location.",
//   "properties" : {
//     "id" : {
//       "type" : "integer",
//       "description" : "An identifier for the stack frame. It must be unique "
//                       "across all threads.\nThis id can be used to retrieve "
//                       "the scopes of the frame with the `scopes` request or
//                       to " "restart the execution of a stack frame."
//     },
//     "name" : {
//       "type" : "string",
//       "description" : "The name of the stack frame, typically a method name."
//     },
//     "source" : {
//       "$ref" : "#/definitions/Source",
//       "description" : "The source of the frame."
//     },
//     "line" : {
//       "type" : "integer",
//       "description" : "The line within the source of the frame. If the source
//       "
//                       "attribute is missing or doesn't exist, `line` is 0 and
//                       " "should be ignored by the client."
//     },
//     "column" : {
//       "type" : "integer",
//       "description" :
//           "Start position of the range covered by the stack frame. It is "
//           "measured in UTF-16 code units and the client capability "
//           "`columnsStartAt1` determines whether it is 0- or 1-based. If "
//           "attribute `source` is missing or doesn't exist, `column` is 0 and
//           " "should be ignored by the client."
//     },
//     "endLine" : {
//       "type" : "integer",
//       "description" : "The end line of the range covered by the stack frame."
//     },
//     "endColumn" : {
//       "type" : "integer",
//       "description" :
//           "End position of the range covered by the stack frame. It is "
//           "measured in UTF-16 code units and the client capability "
//           "`columnsStartAt1` determines whether it is 0- or 1-based."
//     },
//     "canRestart" : {
//       "type" : "boolean",
//       "description" :
//           "Indicates whether this frame can be restarted with the "
//           "`restartFrame` request. Clients should only use this if the debug
//           " "adapter supports the `restart` request and the corresponding "
//           "capability `supportsRestartFrame` is true. If a debug adapter has
//           " "this capability, then `canRestart` defaults to `true` if the "
//           "property is absent."
//     },
//     "instructionPointerReference" : {
//       "type" : "string",
//       "description" : "A memory reference for the current instruction pointer
//       "
//                       "in this frame."
//     },
//     "moduleId" : {
//       "type" : [ "integer", "string" ],
//       "description" : "The module associated with this frame, if any."
//     },
//     "presentationHint" : {
//       "type" : "string",
//       "enum" : [ "normal", "label", "subtle" ],
//       "description" :
//           "A hint for how to present this frame in the UI.\nA value of
//           `label` " "can be used to indicate that the frame is an artificial
//           frame that " "is used as a visual label or separator. A value of
//           `subtle` can be " "used to change the appearance of a frame in a
//           'subtle' way."
//     }
//   },
//   "required" : [ "id", "name", "line", "column" ]
// }
struct StackFrame {
  enum class PresentationHint { normal, label, subtle };
  int64_t id = 0;
  std::string name = "";
  uint32_t line = 0;
  uint32_t column = 0;
  std::optional<Source> source;
  std::optional<uint32_t> endLine;
  std::optional<uint32_t> endColumn;
  std::optional<bool> canRestart;
  std::optional<std::string> instructionPointerReference;
  std::optional<PresentationHint> presentationHint;
  // Unsupported fields: "moduleId"
};
llvm::json::Value toJSON(const StackFrame &);

// "VariablePresentationHint": {
//   "type": "object",
//   "description": "Properties of a variable that can be used to determine how
//   to render the variable in the UI.", "properties": {
//     "kind": {
//       "description": "The kind of variable. Before introducing additional
//       values, try to use the listed values.", "type": "string",
//       "_enum": [ "property", "method", "class", "data", "event", "baseClass",
//       "innerClass", "interface", "mostDerivedClass", "virtual",
//       "dataBreakpoint" ], "enumDescriptions": [
//         "Indicates that the object is a property.",
//         "Indicates that the object is a method.",
//         "Indicates that the object is a class.",
//         "Indicates that the object is data.",
//         "Indicates that the object is an event.",
//         "Indicates that the object is a base class.",
//         "Indicates that the object is an inner class.",
//         "Indicates that the object is an interface.",
//         "Indicates that the object is the most derived class.",
//         "Indicates that the object is virtual, that means it is a synthetic
//         object introduced by the adapter for rendering purposes, e.g. an
//         index range for large arrays.", "Deprecated: Indicates that a data
//         breakpoint is registered for the object. The `hasDataBreakpoint`
//         attribute should generally be used instead."
//       ]
//     },
//     "attributes": {
//       "description": "Set of attributes represented as an array of strings.
//       Before introducing additional values, try to use the listed values.",
//       "type": "array",
//       "items": {
//         "type": "string",
//         "_enum": [ "static", "constant", "readOnly", "rawString",
//         "hasObjectId", "canHaveObjectId", "hasSideEffects",
//         "hasDataBreakpoint" ], "enumDescriptions": [
//           "Indicates that the object is static.",
//           "Indicates that the object is a constant.",
//           "Indicates that the object is read only.",
//           "Indicates that the object is a raw string.",
//           "Indicates that the object can have an Object ID created for it.
//           This is a vestigial attribute that is used by some clients; 'Object
//           ID's are not specified in the protocol.", "Indicates that the
//           object has an Object ID associated with it. This is a vestigial
//           attribute that is used by some clients; 'Object ID's are not
//           specified in the protocol.", "Indicates that the evaluation had
//           side effects.", "Indicates that the object has its value tracked by
//           a data breakpoint."
//         ]
//       }
//     },
//     "visibility": {
//       "description": "Visibility of variable. Before introducing additional
//       values, try to use the listed values.", "type": "string",
//       "_enum": [ "public", "private", "protected", "internal", "final" ]
//     },
//     "lazy": {
//       "description": "If true, clients can present the variable with a UI
//       that supports a specific gesture to trigger its evaluation.\nThis
//       mechanism can be used for properties that require executing code when
//       retrieving their value and where the code execution can be expensive
//       and/or produce side-effects. A typical example are properties based on
//       a getter function.\nPlease note that in addition to the `lazy` flag,
//       the variable's `variablesReference` is expected to refer to a variable
//       that will provide the value through another `variable` request.",
//       "type": "boolean"
//     }
//   }
// },
struct VariablePresentationHint {
  enum class Kind {
    property_,
    method_,
    class_,
    data_,
    event_,
    baseClass_,
    innerClass_,
    interface_,
    mostDerivedClass_,
    virtual_,
    dataBreakpoint_,
  };
  enum class Attributes {
    static_,
    constant,
    readOnly,
    rawString,
    hasObjectId,
    canHaveObjectId,
    hasSideEffects,
    hasDataBreakpoint,
  };
  enum class Visibility {
    public_,
    private_,
    protected_,
    internal_,
    final_,
  };

  std::optional<Kind> kind;
  std::optional<std::vector<Attributes>> attributes;
  std::optional<Visibility> visibility;
  std::optional<bool> lazy;
};
llvm::json::Value toJSON(const VariablePresentationHint &);
llvm::json::Value toJSON(const VariablePresentationHint::Kind &);
llvm::json::Value toJSON(const VariablePresentationHint::Attributes &);
llvm::json::Value toJSON(const VariablePresentationHint::Visibility &);

// MARK: Events

// "ExitedEvent": {
//   "allOf": [ { "$ref": "#/definitions/Event" }, {
//     "type": "object",
//     "description": "The event indicates that the debuggee has exited and
//     returns its exit code.", "properties": {
//       "event": {
//         "type": "string",
//         "enum": [ "exited" ]
//       },
//       "body": {
//         "type": "object",
//         "properties": {
//           "exitCode": {
//             "type": "integer",
//             "description": "The exit code returned from the debuggee."
//           }
//         },
//         "required": [ "exitCode" ]
//       }
//     },
//     "required": [ "event", "body" ]
//   }]
// }
struct ExitedEventBody {
  int exitCode;
};
llvm::json::Value toJSON(const ExitedEventBody &);

// MARK: Requests

// "EvaluateRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "Evaluates the given expression in the context of a stack
//     frame.\nThe expression has access to any variables and arguments that are
//     in scope.", "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "evaluate" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/EvaluateArguments"
//       }
//     },
//     "required": [ "command", "arguments"  ]
//   }]
// },
// "EvaluateArguments": {
//   "type": "object",
//   "description": "Arguments for `evaluate` request.",
//   "properties": {
//     "expression": {
//       "type": "string",
//       "description": "The expression to evaluate."
//     },
//     "frameId": {
//       "type": "integer",
//       "description": "Evaluate the expression in the scope of this stack
//       frame. If not specified, the expression is evaluated in the global
//       scope."
//     },
//     "line": {
//       "type": "integer",
//       "description": "The contextual line where the expression should be
//       evaluated. In the 'hover' context, this should be set to the start of
//       the expression being hovered."
//     },
//     "column": {
//       "type": "integer",
//       "description": "The contextual column where the expression should be
//       evaluated. This may be provided if `line` is also provided.\n\nIt is
//       measured in UTF-16 code units and the client capability
//       `columnsStartAt1` determines whether it is 0- or 1-based."
//     },
//     "source": {
//       "$ref": "#/definitions/Source",
//       "description": "The contextual source in which the `line` is found.
//       This must be provided if `line` is provided."
//     },
//     "context": {
//       "type": "string",
//       "_enum": [ "watch", "repl", "hover", "clipboard", "variables" ],
//       "enumDescriptions": [
//         "evaluate is called from a watch view context.",
//         "evaluate is called from a REPL context.",
//         "evaluate is called to generate the debug hover contents.\nThis value
//         should only be used if the corresponding capability
//         `supportsEvaluateForHovers` is true.", "evaluate is called to
//         generate clipboard contents.\nThis value should only be used if the
//         corresponding capability `supportsClipboardContext` is true.",
//         "evaluate is called from a variables view context."
//       ],
//       "description": "The context in which the evaluate request is used."
//     },
//     "format": {
//       "$ref": "#/definitions/ValueFormat",
//       "description": "Specifies details on how to format the result.\nThe
//       attribute is only honored by a debug adapter if the corresponding
//       capability `supportsValueFormattingOptions` is true."
//     }
//   },
//   "required": [ "expression" ]
// },
struct EvaluateArguments {
  enum class Context { watch, repl, hover, clipboard, variables };

  std::string expression;
  std::optional<int64_t> frameId;
  std::optional<int> line;
  std::optional<int> column;
  std::optional<Source> source;
  std::optional<Context> context;
  // std::optional<ValueFormat> format; // unsupported
};
bool fromJSON(const llvm::json::Value &, EvaluateArguments &, llvm::json::Path);

// "EvaluateResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to `evaluate` request.",
//     "properties": {
//       "body": {
//         "type": "object",
//         "properties": {
//           "result": {
//             "type": "string",
//             "description": "The result of the evaluate request."
//           },
//           "type": {
//             "type": "string",
//             "description": "The type of the evaluate result.\nThis attribute
//             should only be returned by a debug adapter if the corresponding
//             capability `supportsVariableType` is true."
//           },
//           "presentationHint": {
//             "$ref": "#/definitions/VariablePresentationHint",
//             "description": "Properties of an evaluate result that can be used
//             to determine how to render the result in the UI."
//           },
//           "variablesReference": {
//             "type": "integer",
//             "description": "If `variablesReference` is > 0, the evaluate
//             result is structured and its children can be retrieved by passing
//             `variablesReference` to the `variables` request as long as
//             execution remains suspended. See 'Lifetime of Object References'
//             in the Overview section for details."
//           },
//           "namedVariables": {
//             "type": "integer",
//             "description": "The number of named child variables.\nThe client
//             can use this information to present the variables in a paged UI
//             and fetch them in chunks.\nThe value should be less than or equal
//             to 2147483647 (2^31-1)."
//           },
//           "indexedVariables": {
//             "type": "integer",
//             "description": "The number of indexed child variables.\nThe
//             client can use this information to present the variables in a
//             paged UI and fetch them in chunks.\nThe value should be less than
//             or equal to 2147483647 (2^31-1)."
//           },
//           "memoryReference": {
//             "type": "string",
//             "description": "A memory reference to a location appropriate for
//             this result.\nFor pointer type eval results, this is generally a
//             reference to the memory address contained in the pointer.\nThis
//             attribute may be returned by a debug adapter if corresponding
//             capability `supportsMemoryReferences` is true."
//           },
//           "valueLocationReference": {
//             "type": "integer",
//             "description": "A reference that allows the client to request the
//             location where the returned value is declared. For example, if a
//             function pointer is returned, the adapter may be able to look up
//             the function's location. This should be present only if the
//             adapter is likely to be able to resolve the location.\n\nThis
//             reference shares the same lifetime as the `variablesReference`.
//             See 'Lifetime of Object References' in the Overview section for
//             details."
//           }
//         },
//         "required": [ "result", "variablesReference" ]
//       }
//     },
//     "required": [ "body" ]
//   }]
// },
struct EvaluateResponseBody {
  std::string result = "";
  std::optional<std::string> type;
  std::optional<VariablePresentationHint> presentationHint;
  int64_t variablesReference = 0;
  std::optional<int> namedVariables;
  std::optional<int> indexedVariables;
  std::optional<std::string> memoryReference;
  std::optional<int> valueLocationReference;
};
llvm::json::Value toJSON(const EvaluateResponseBody &);

// "SourceRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "The request retrieves the source code for a given source
//     reference.", "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "source" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/SourceArguments"
//       }
//     },
//     "required": [ "command", "arguments"  ]
//   }]
// },
// "SourceArguments": {
//   "type": "object",
//   "description": "Arguments for 'source' request.",
//   "properties": {
//     "source": {
//       "$ref": "#/definitions/Source",
//       "description": "Specifies the source content to load. Either
//       source.path or source.sourceReference must be specified."
//     },
//     "sourceReference": {
//       "type": "integer",
//       "description": "The reference to the source. This is the same as
//       source.sourceReference. This is provided for backward compatibility
//       since old backends do not understand the 'source' attribute."
//     }
//   },
//   "required": [ "sourceReference" ]
// }
struct SourceArguments {
  std::optional<Source> source;
  int64_t sourceReference;
};
bool fromJSON(const llvm::json::Value &, SourceArguments &, llvm::json::Path);

// "SourceResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to 'source' request.",
//     "properties": {
//       "body": {
//         "type": "object",
//         "properties": {
//           "content": {
//             "type": "string",
//             "description": "Content of the source reference."
//           },
//           "mimeType": {
//             "type": "string",
//             "description": "Optional content type (mime type) of the source."
//           }
//         },
//         "required": [ "content" ]
//       }
//     },
//     "required": [ "body" ]
//   }]
// }
struct SourceResponseBody {
  std::string content = "";
  std::optional<std::string> mimeType;
};
llvm::json::Value toJSON(const SourceResponseBody &);

// MARK: Reverse Requests

} // namespace protocol
} // namespace lldb_dap

#endif // LLDB_TOOLS_LLDB_DAP_PROTOCOL_H
