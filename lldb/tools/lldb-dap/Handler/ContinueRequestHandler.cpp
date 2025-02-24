//===-- ContinueRequestHandler.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "JSONUtils.h"
#include "RequestHandler.h"

namespace lldb_dap {

// "ContinueRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "Continue request; value of command field is 'continue'.
//                     The request starts the debuggee to run again.",
//     "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "continue" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/ContinueArguments"
//       }
//     },
//     "required": [ "command", "arguments"  ]
//   }]
// },
// "ContinueArguments": {
//   "type": "object",
//   "description": "Arguments for 'continue' request.",
//   "properties": {
//     "threadId": {
//       "type": "integer",
//       "description": "Continue execution for the specified thread (if
//                       possible). If the backend cannot continue on a single
//                       thread but will continue on all threads, it should
//                       set the allThreadsContinued attribute in the response
//                       to true."
//     }
//   },
//   "required": [ "threadId" ]
// },
// "ContinueResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to 'continue' request.",
//     "properties": {
//       "body": {
//         "type": "object",
//         "properties": {
//           "allThreadsContinued": {
//             "type": "boolean",
//             "description": "If true, the continue request has ignored the
//                             specified thread and continued all threads
//                             instead. If this attribute is missing a value
//                             of 'true' is assumed for backward
//                             compatibility."
//           }
//         }
//       }
//     },
//     "required": [ "body" ]
//   }]
// }
void ContinueRequestHandler::operator()(const llvm::json::Object &request) {
  llvm::json::Object response;
  FillResponse(request, response);
  lldb::SBProcess process = dap.target.GetProcess();
  lldb::SBError error = process.Continue();
  llvm::json::Object body;
  body.try_emplace("allThreadsContinued", true);
  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}
} // namespace lldb_dap
