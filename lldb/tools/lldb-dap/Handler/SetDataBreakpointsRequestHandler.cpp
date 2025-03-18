//===-- SetDataBreakpointsRequestHandler.cpp ------------------------------===//
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
#include "Watchpoint.h"
#include <set>

namespace lldb_dap {

// "SetDataBreakpointsRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "Replaces all existing data breakpoints with new data
//     breakpoints.\nTo clear all data breakpoints, specify an empty
//     array.\nWhen a data breakpoint is hit, a `stopped` event (with reason
//     `data breakpoint`) is generated.\nClients should only call this request
//     if the corresponding capability `supportsDataBreakpoints` is true.",
//     "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "setDataBreakpoints" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/SetDataBreakpointsArguments"
//       }
//     },
//     "required": [ "command", "arguments"  ]
//   }]
// },
// "SetDataBreakpointsArguments": {
//   "type": "object",
//   "description": "Arguments for `setDataBreakpoints` request.",
//   "properties": {
//     "breakpoints": {
//       "type": "array",
//       "items": {
//         "$ref": "#/definitions/DataBreakpoint"
//       },
//       "description": "The contents of this array replaces all existing data
//       breakpoints. An empty array clears all data breakpoints."
//     }
//   },
//   "required": [ "breakpoints" ]
// },
// "SetDataBreakpointsResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to `setDataBreakpoints` request.\nReturned is
//     information about each breakpoint created by this request.",
//     "properties": {
//       "body": {
//         "type": "object",
//         "properties": {
//           "breakpoints": {
//             "type": "array",
//             "items": {
//               "$ref": "#/definitions/Breakpoint"
//             },
//             "description": "Information about the data breakpoints. The array
//             elements correspond to the elements of the input argument
//             `breakpoints` array."
//           }
//         },
//         "required": [ "breakpoints" ]
//       }
//     },
//     "required": [ "body" ]
//   }]
// }
void SetDataBreakpointsRequestHandler::operator()(
    const llvm::json::Object &request) const {
  llvm::json::Object response;
  lldb::SBError error;
  FillResponse(request, response);
  const auto *arguments = request.getObject("arguments");
  const auto *breakpoints = arguments->getArray("breakpoints");
  llvm::json::Array response_breakpoints;
  dap.target.DeleteAllWatchpoints();
  std::vector<Watchpoint> watchpoints;
  if (breakpoints) {
    for (const auto &bp : *breakpoints) {
      const auto *bp_obj = bp.getAsObject();
      if (bp_obj)
        watchpoints.emplace_back(dap, *bp_obj);
    }
  }
  // If two watchpoints start at the same address, the latter overwrite the
  // former. So, we only enable those at first-seen addresses when iterating
  // backward.
  std::set<lldb::addr_t> addresses;
  for (auto iter = watchpoints.rbegin(); iter != watchpoints.rend(); ++iter) {
    if (addresses.count(iter->addr) == 0) {
      iter->SetWatchpoint();
      addresses.insert(iter->addr);
    }
  }
  for (auto wp : watchpoints)
    AppendBreakpoint(&wp, response_breakpoints);

  llvm::json::Object body;
  body.try_emplace("breakpoints", std::move(response_breakpoints));
  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

} // namespace lldb_dap
