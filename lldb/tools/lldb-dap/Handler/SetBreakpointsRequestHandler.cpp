//===-- SetBreakpointsRequestHandler.cpp ----------------------------------===//
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

// "SetBreakpointsRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "SetBreakpoints request; value of command field is
//     'setBreakpoints'. Sets multiple breakpoints for a single source and
//     clears all previous breakpoints in that source. To clear all breakpoint
//     for a source, specify an empty array. When a breakpoint is hit, a
//     StoppedEvent (event type 'breakpoint') is generated.", "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "setBreakpoints" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/SetBreakpointsArguments"
//       }
//     },
//     "required": [ "command", "arguments"  ]
//   }]
// },
// "SetBreakpointsArguments": {
//   "type": "object",
//   "description": "Arguments for 'setBreakpoints' request.",
//   "properties": {
//     "source": {
//       "$ref": "#/definitions/Source",
//       "description": "The source location of the breakpoints; either
//       source.path or source.reference must be specified."
//     },
//     "breakpoints": {
//       "type": "array",
//       "items": {
//         "$ref": "#/definitions/SourceBreakpoint"
//       },
//       "description": "The code locations of the breakpoints."
//     },
//     "lines": {
//       "type": "array",
//       "items": {
//         "type": "integer"
//       },
//       "description": "Deprecated: The code locations of the breakpoints."
//     },
//     "sourceModified": {
//       "type": "boolean",
//       "description": "A value of true indicates that the underlying source
//       has been modified which results in new breakpoint locations."
//     }
//   },
//   "required": [ "source" ]
// },
// "SetBreakpointsResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to 'setBreakpoints' request. Returned is
//     information about each breakpoint created by this request. This includes
//     the actual code location and whether the breakpoint could be verified.
//     The breakpoints returned are in the same order as the elements of the
//     'breakpoints' (or the deprecated 'lines') in the
//     SetBreakpointsArguments.", "properties": {
//       "body": {
//         "type": "object",
//         "properties": {
//           "breakpoints": {
//             "type": "array",
//             "items": {
//               "$ref": "#/definitions/Breakpoint"
//             },
//             "description": "Information about the breakpoints. The array
//             elements are in the same order as the elements of the
//             'breakpoints' (or the deprecated 'lines') in the
//             SetBreakpointsArguments."
//           }
//         },
//         "required": [ "breakpoints" ]
//       }
//     },
//     "required": [ "body" ]
//   }]
// },
// "SourceBreakpoint": {
//   "type": "object",
//   "description": "Properties of a breakpoint or logpoint passed to the
//   setBreakpoints request.", "properties": {
//     "line": {
//       "type": "integer",
//       "description": "The source line of the breakpoint or logpoint."
//     },
//     "column": {
//       "type": "integer",
//       "description": "An optional source column of the breakpoint."
//     },
//     "condition": {
//       "type": "string",
//       "description": "An optional expression for conditional breakpoints."
//     },
//     "hitCondition": {
//       "type": "string",
//       "description": "An optional expression that controls how many hits of
//       the breakpoint are ignored. The backend is expected to interpret the
//       expression as needed."
//     },
//     "logMessage": {
//       "type": "string",
//       "description": "If this attribute exists and is non-empty, the backend
//       must not 'break' (stop) but log the message instead. Expressions within
//       {} are interpolated."
//     }
//   },
//   "required": [ "line" ]
// }
void SetBreakpointsRequestHandler::operator()(
    const llvm::json::Object &request) const {
  llvm::json::Object response;
  lldb::SBError error;
  FillResponse(request, response);
  const auto *arguments = request.getObject("arguments");
  const auto *source = arguments->getObject("source");
  const auto path = GetString(source, "path");
  const auto *breakpoints = arguments->getArray("breakpoints");
  llvm::json::Array response_breakpoints;

  // Decode the source breakpoint infos for this "setBreakpoints" request
  SourceBreakpointMap request_bps;
  // "breakpoints" may be unset, in which case we treat it the same as being set
  // to an empty array.
  if (breakpoints) {
    for (const auto &bp : *breakpoints) {
      const auto *bp_obj = bp.getAsObject();
      if (bp_obj) {
        SourceBreakpoint src_bp(dap, *bp_obj);
        std::pair<uint32_t, uint32_t> bp_pos(src_bp.line, src_bp.column);
        request_bps.try_emplace(bp_pos, src_bp);
        const auto [iv, inserted] =
            dap.source_breakpoints[path].try_emplace(bp_pos, src_bp);
        // We check if this breakpoint already exists to update it
        if (inserted)
          iv->getSecond().SetBreakpoint(path.data());
        else
          iv->getSecond().UpdateBreakpoint(src_bp);
        AppendBreakpoint(&iv->getSecond(), response_breakpoints, path,
                         src_bp.line);
      }
    }
  }

  // Delete any breakpoints in this source file that aren't in the
  // request_bps set. There is no call to remove breakpoints other than
  // calling this function with a smaller or empty "breakpoints" list.
  auto old_src_bp_pos = dap.source_breakpoints.find(path);
  if (old_src_bp_pos != dap.source_breakpoints.end()) {
    for (auto &old_bp : old_src_bp_pos->second) {
      auto request_pos = request_bps.find(old_bp.first);
      if (request_pos == request_bps.end()) {
        // This breakpoint no longer exists in this source file, delete it
        dap.target.BreakpointDelete(old_bp.second.bp.GetID());
        old_src_bp_pos->second.erase(old_bp.first);
      }
    }
  }

  llvm::json::Object body;
  body.try_emplace("breakpoints", std::move(response_breakpoints));
  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

} // namespace lldb_dap
