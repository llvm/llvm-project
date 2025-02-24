//===-- StepInTargetsRequestHandler.cpp -----------------------------------===//
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
#include "lldb/API/SBInstruction.h"

namespace lldb_dap {

// "StepInTargetsRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "This request retrieves the possible step-in targets for
//     the specified stack frame.\nThese targets can be used in the `stepIn`
//     request.\nClients should only call this request if the corresponding
//     capability `supportsStepInTargetsRequest` is true.", "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "stepInTargets" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/StepInTargetsArguments"
//       }
//     },
//     "required": [ "command", "arguments"  ]
//   }]
// },
// "StepInTargetsArguments": {
//   "type": "object",
//   "description": "Arguments for `stepInTargets` request.",
//   "properties": {
//     "frameId": {
//       "type": "integer",
//       "description": "The stack frame for which to retrieve the possible
//       step-in targets."
//     }
//   },
//   "required": [ "frameId" ]
// },
// "StepInTargetsResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to `stepInTargets` request.",
//     "properties": {
//       "body": {
//         "type": "object",
//         "properties": {
//           "targets": {
//             "type": "array",
//             "items": {
//               "$ref": "#/definitions/StepInTarget"
//             },
//             "description": "The possible step-in targets of the specified
//             source location."
//           }
//         },
//         "required": [ "targets" ]
//       }
//     },
//     "required": [ "body" ]
//   }]
// }
void StepInTargetsRequestHandler::operator()(
    const llvm::json::Object &request) {
  llvm::json::Object response;
  FillResponse(request, response);
  const auto *arguments = request.getObject("arguments");

  dap.step_in_targets.clear();
  lldb::SBFrame frame = dap.GetLLDBFrame(*arguments);
  if (frame.IsValid()) {
    lldb::SBAddress pc_addr = frame.GetPCAddress();
    lldb::SBAddress line_end_addr =
        pc_addr.GetLineEntry().GetSameLineContiguousAddressRangeEnd(true);
    lldb::SBInstructionList insts = dap.target.ReadInstructions(
        pc_addr, line_end_addr, /*flavor_string=*/nullptr);

    if (!insts.IsValid()) {
      response["success"] = false;
      response["message"] = "Failed to get instructions for frame.";
      dap.SendJSON(llvm::json::Value(std::move(response)));
      return;
    }

    llvm::json::Array step_in_targets;
    const auto num_insts = insts.GetSize();
    for (size_t i = 0; i < num_insts; ++i) {
      lldb::SBInstruction inst = insts.GetInstructionAtIndex(i);
      if (!inst.IsValid())
        break;

      lldb::addr_t inst_addr = inst.GetAddress().GetLoadAddress(dap.target);

      // Note: currently only x86/x64 supports flow kind.
      lldb::InstructionControlFlowKind flow_kind =
          inst.GetControlFlowKind(dap.target);
      if (flow_kind == lldb::eInstructionControlFlowKindCall) {
        // Use call site instruction address as id which is easy to debug.
        llvm::json::Object step_in_target;
        step_in_target["id"] = inst_addr;

        llvm::StringRef call_operand_name = inst.GetOperands(dap.target);
        lldb::addr_t call_target_addr;
        if (call_operand_name.getAsInteger(0, call_target_addr))
          continue;

        lldb::SBAddress call_target_load_addr =
            dap.target.ResolveLoadAddress(call_target_addr);
        if (!call_target_load_addr.IsValid())
          continue;

        // The existing ThreadPlanStepInRange only accept step in target
        // function with debug info.
        lldb::SBSymbolContext sc = dap.target.ResolveSymbolContextForAddress(
            call_target_load_addr, lldb::eSymbolContextFunction);

        // The existing ThreadPlanStepInRange only accept step in target
        // function with debug info.
        std::string step_in_target_name;
        if (sc.IsValid() && sc.GetFunction().IsValid())
          step_in_target_name = sc.GetFunction().GetDisplayName();

        // Skip call sites if we fail to resolve its symbol name.
        if (step_in_target_name.empty())
          continue;

        dap.step_in_targets.try_emplace(inst_addr, step_in_target_name);
        step_in_target.try_emplace("label", step_in_target_name);
        step_in_targets.emplace_back(std::move(step_in_target));
      }
    }
    llvm::json::Object body;
    body.try_emplace("targets", std::move(step_in_targets));
    response.try_emplace("body", std::move(body));
  } else {
    response["success"] = llvm::json::Value(false);
    response["message"] = "Failed to get frame for input frameId.";
  }
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

} // namespace lldb_dap
