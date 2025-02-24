//===-- SourceRequestHandler.cpp ------------------------------------------===//
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

namespace lldb_dap {

// "SourceRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "Source request; value of command field is 'source'. The
//     request retrieves the source code for a given source reference.",
//     "properties": {
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
// },
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
void SourceRequestHandler::operator()(const llvm::json::Object &request) {
  llvm::json::Object response;
  FillResponse(request, response);
  llvm::json::Object body{{"content", ""}};
  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

} // namespace lldb_dap
