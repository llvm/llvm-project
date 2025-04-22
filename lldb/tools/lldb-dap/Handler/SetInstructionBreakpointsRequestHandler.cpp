//===-- SetInstructionBreakpointsRequestHandler.cpp -----------------------===//
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

// "SetInstructionBreakpointsRequest": {
//   "allOf": [
//     {"$ref": "#/definitions/Request"},
//     {
//       "type": "object",
//       "description" :
//           "Replaces all existing instruction breakpoints. Typically, "
//           "instruction breakpoints would be set from a disassembly window. "
//           "\nTo clear all instruction breakpoints, specify an empty "
//           "array.\nWhen an instruction breakpoint is hit, a `stopped` event "
//           "(with reason `instruction breakpoint`) is generated.\nClients "
//           "should only call this request if the corresponding capability "
//           "`supportsInstructionBreakpoints` is true.",
//       "properties": {
//         "command": { "type": "string", "enum": ["setInstructionBreakpoints"]
//         }, "arguments": {"$ref":
//         "#/definitions/SetInstructionBreakpointsArguments"}
//       },
//       "required": [ "command", "arguments" ]
//     }
//   ]
// },
// "SetInstructionBreakpointsArguments": {
//   "type": "object",
//   "description": "Arguments for `setInstructionBreakpoints` request",
//   "properties": {
//     "breakpoints": {
//       "type": "array",
//       "items": {"$ref": "#/definitions/InstructionBreakpoint"},
//       "description": "The instruction references of the breakpoints"
//     }
//   },
//   "required": ["breakpoints"]
// },
// "SetInstructionBreakpointsResponse": {
//   "allOf": [
//     {"$ref": "#/definitions/Response"},
//     {
//       "type": "object",
//       "description": "Response to `setInstructionBreakpoints` request",
//       "properties": {
//         "body": {
//           "type": "object",
//           "properties": {
//             "breakpoints": {
//               "type": "array",
//               "items": {"$ref": "#/definitions/Breakpoint"},
//               "description":
//                   "Information about the breakpoints. The array elements
//                   " "correspond to the elements of the `breakpoints`
//                   array."
//             }
//           },
//           "required": ["breakpoints"]
//         }
//       },
//       "required": ["body"]
//     }
//   ]
// },
// "InstructionBreakpoint": {
//   "type": "object",
//   "description": "Properties of a breakpoint passed to the "
//                   "`setInstructionBreakpoints` request",
//   "properties": {
//     "instructionReference": {
//       "type": "string",
//       "description" :
//           "The instruction reference of the breakpoint.\nThis should be a "
//           "memory or instruction pointer reference from an
//           `EvaluateResponse`, "
//           "`Variable`, `StackFrame`, `GotoTarget`, or `Breakpoint`."
//     },
//     "offset": {
//       "type": "integer",
//       "description": "The offset from the instruction reference in "
//                       "bytes.\nThis can be negative."
//     },
//     "condition": {
//       "type": "string",
//       "description": "An expression for conditional breakpoints.\nIt is only
//       "
//                       "honored by a debug adapter if the corresponding "
//                       "capability `supportsConditionalBreakpoints` is true."
//     },
//     "hitCondition": {
//       "type": "string",
//       "description": "An expression that controls how many hits of the "
//                       "breakpoint are ignored.\nThe debug adapter is expected
//                       " "to interpret the expression as needed.\nThe
//                       attribute " "is only honored by a debug adapter if the
//                       corresponding " "capability
//                       `supportsHitConditionalBreakpoints` is true."
//     },
//     "mode": {
//       "type": "string",
//       "description": "The mode of this breakpoint. If defined, this must be
//       "
//                       "one of the `breakpointModes` the debug adapter "
//                       "advertised in its `Capabilities`."
//     }
//   },
//   "required": ["instructionReference"]
// },
// "Breakpoint": {
//   "type": "object",
//   "description" :
//       "Information about a breakpoint created in `setBreakpoints`, "
//       "`setFunctionBreakpoints`, `setInstructionBreakpoints`, or "
//       "`setDataBreakpoints` requests.",
//   "properties": {
//     "id": {
//       "type": "integer",
//       "description" :
//           "The identifier for the breakpoint. It is needed if breakpoint
//           " "events are used to update or remove breakpoints."
//     },
//     "verified": {
//       "type": "boolean",
//       "description": "If true, the breakpoint could be set (but not "
//                       "necessarily at the desired location)."
//     },
//     "message": {
//       "type": "string",
//       "description": "A message about the state of the breakpoint.\nThis
//       "
//                       "is shown to the user and can be used to explain
//                       why " "a breakpoint could not be verified."
//     },
//     "source": {
//       "$ref": "#/definitions/Source",
//       "description": "The source where the breakpoint is located."
//     },
//     "line": {
//       "type": "integer",
//       "description" :
//           "The start line of the actual range covered by the breakpoint."
//     },
//     "column": {
//       "type": "integer",
//       "description" :
//           "Start position of the source range covered by the breakpoint.
//           " "It is measured in UTF-16 code units and the client
//           capability "
//           "`columnsStartAt1` determines whether it is 0- or 1-based."
//     },
//     "endLine": {
//       "type": "integer",
//       "description" :
//           "The end line of the actual range covered by the breakpoint."
//     },
//     "endColumn": {
//       "type": "integer",
//       "description" :
//           "End position of the source range covered by the breakpoint. It
//           " "is measured in UTF-16 code units and the client capability "
//           "`columnsStartAt1` determines whether it is 0- or 1-based.\nIf
//           " "no end line is given, then the end column is assumed to be
//           in " "the start line."
//     },
//     "instructionReference": {
//       "type": "string",
//       "description": "A memory reference to where the breakpoint is
//       set."
//     },
//     "offset": {
//       "type": "integer",
//       "description": "The offset from the instruction reference.\nThis "
//                       "can be negative."
//     },
//     "reason": {
//       "type": "string",
//       "description" :
//           "A machine-readable explanation of why a breakpoint may not be
//           " "verified. If a breakpoint is verified or a specific reason
//           is " "not known, the adapter should omit this property.
//           Possible " "values include:\n\n- `pending`: Indicates a
//           breakpoint might be " "verified in the future, but the adapter
//           cannot verify it in the " "current state.\n - `failed`:
//           Indicates a breakpoint was not " "able to be verified, and the
//           adapter does not believe it can be " "verified without
//           intervention.",
//       "enum": [ "pending", "failed" ]
//     }
//   },
//   "required": ["verified"]
// },
void SetInstructionBreakpointsRequestHandler::operator()(
    const llvm::json::Object &request) const {
  llvm::json::Object response;
  llvm::json::Array response_breakpoints;
  llvm::json::Object body;
  FillResponse(request, response);

  const auto *arguments = request.getObject("arguments");
  const auto *breakpoints = arguments->getArray("breakpoints");

  // Disable any instruction breakpoints that aren't in this request.
  // There is no call to remove instruction breakpoints other than calling this
  // function with a smaller or empty "breakpoints" list.
  llvm::DenseSet<lldb::addr_t> seen(
      llvm::from_range, llvm::make_first_range(dap.instruction_breakpoints));

  for (const auto &bp : *breakpoints) {
    const auto *bp_obj = bp.getAsObject();
    if (!bp_obj)
      continue;
    // Read instruction breakpoint request.
    InstructionBreakpoint inst_bp(dap, *bp_obj);
    const auto [iv, inserted] = dap.instruction_breakpoints.try_emplace(
        inst_bp.GetInstructionAddressReference(), dap, *bp_obj);
    if (inserted)
      iv->second.SetBreakpoint();
    else
      iv->second.UpdateBreakpoint(inst_bp);
    AppendBreakpoint(&iv->second, response_breakpoints);
    seen.erase(inst_bp.GetInstructionAddressReference());
  }

  for (const auto &addr : seen) {
    auto inst_bp = dap.instruction_breakpoints.find(addr);
    if (inst_bp == dap.instruction_breakpoints.end())
      continue;
    dap.target.BreakpointDelete(inst_bp->second.GetID());
    dap.instruction_breakpoints.erase(addr);
  }

  body.try_emplace("breakpoints", std::move(response_breakpoints));
  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

} // namespace lldb_dap
