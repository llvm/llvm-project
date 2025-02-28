//===-- ExceptionInfoRequestHandler.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "EventHelper.h"
#include "JSONUtils.h"
#include "RequestHandler.h"
#include "lldb/API/SBStream.h"

namespace lldb_dap {

// "ExceptionInfoRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "Retrieves the details of the exception that
//     caused this event to be raised. Clients should only call this request if
//     the corresponding capability `supportsExceptionInfoRequest` is true.",
//     "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "exceptionInfo" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/ExceptionInfoArguments"
//       }
//     },
//     "required": [ "command", "arguments"  ]
//   }]
// },
// "ExceptionInfoArguments": {
//   "type": "object",
//   "description": "Arguments for `exceptionInfo` request.",
//   "properties": {
//     "threadId": {
//       "type": "integer",
//       "description": "Thread for which exception information should be
//       retrieved."
//     }
//   },
//   "required": [ "threadId" ]
// },
// "ExceptionInfoResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to `exceptionInfo` request.",
//     "properties": {
//       "body": {
//         "type": "object",
//         "properties": {
//           "exceptionId": {
//             "type": "string",
//             "description": "ID of the exception that was thrown."
//           },
//           "description": {
//             "type": "string",
//             "description": "Descriptive text for the exception."
//           },
//           "breakMode": {
//          "$ref": "#/definitions/ExceptionBreakMode",
//            "description": "Mode that caused the exception notification to
//            be raised."
//           },
//           "details": {
//             "$ref": "#/definitions/ExceptionDetails",
//            "description": "Detailed information about the exception."
//           }
//         },
//         "required": [ "exceptionId", "breakMode" ]
//       }
//     },
//     "required": [ "body" ]
//   }]
// }
// "ExceptionDetails": {
//   "type": "object",
//   "description": "Detailed information about an exception that has
//   occurred.", "properties": {
//     "message": {
//       "type": "string",
//       "description": "Message contained in the exception."
//     },
//     "typeName": {
//       "type": "string",
//       "description": "Short type name of the exception object."
//     },
//     "fullTypeName": {
//       "type": "string",
//       "description": "Fully-qualified type name of the exception object."
//     },
//     "evaluateName": {
//       "type": "string",
//       "description": "An expression that can be evaluated in the current
//       scope to obtain the exception object."
//     },
//     "stackTrace": {
//       "type": "string",
//       "description": "Stack trace at the time the exception was thrown."
//     },
//     "innerException": {
//       "type": "array",
//       "items": {
//         "$ref": "#/definitions/ExceptionDetails"
//       },
//       "description": "Details of the exception contained by this exception,
//       if any."
//     }
//   }
// },
void ExceptionInfoRequestHandler::operator()(
    const llvm::json::Object &request) const {
  llvm::json::Object response;
  FillResponse(request, response);
  const auto *arguments = request.getObject("arguments");
  llvm::json::Object body;
  lldb::SBThread thread = dap.GetLLDBThread(*arguments);
  if (thread.IsValid()) {
    auto stopReason = thread.GetStopReason();
    if (stopReason == lldb::eStopReasonSignal)
      body.try_emplace("exceptionId", "signal");
    else if (stopReason == lldb::eStopReasonBreakpoint) {
      ExceptionBreakpoint *exc_bp = dap.GetExceptionBPFromStopReason(thread);
      if (exc_bp) {
        EmplaceSafeString(body, "exceptionId", exc_bp->filter);
        EmplaceSafeString(body, "description", exc_bp->label);
      } else {
        body.try_emplace("exceptionId", "exception");
      }
    } else {
      body.try_emplace("exceptionId", "exception");
    }
    if (!ObjectContainsKey(body, "description")) {
      char description[1024];
      if (thread.GetStopDescription(description, sizeof(description))) {
        EmplaceSafeString(body, "description", std::string(description));
      }
    }
    body.try_emplace("breakMode", "always");
    auto exception = thread.GetCurrentException();
    if (exception.IsValid()) {
      llvm::json::Object details;
      lldb::SBStream stream;
      if (exception.GetDescription(stream)) {
        EmplaceSafeString(details, "message", stream.GetData());
      }

      auto exceptionBacktrace = thread.GetCurrentExceptionBacktrace();
      if (exceptionBacktrace.IsValid()) {
        lldb::SBStream stream;
        exceptionBacktrace.GetDescription(stream);
        for (uint32_t i = 0; i < exceptionBacktrace.GetNumFrames(); i++) {
          lldb::SBFrame frame = exceptionBacktrace.GetFrameAtIndex(i);
          frame.GetDescription(stream);
        }
        EmplaceSafeString(details, "stackTrace", stream.GetData());
      }

      body.try_emplace("details", std::move(details));
    }
    // auto excInfoCount = thread.GetStopReasonDataCount();
    // for (auto i=0; i<excInfoCount; ++i) {
    //   uint64_t exc_data = thread.GetStopReasonDataAtIndex(i);
    // }
  } else {
    response["success"] = llvm::json::Value(false);
  }
  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}
} // namespace lldb_dap
