//===-- StepInTargetsRequestHandler.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "Protocol/ProtocolRequests.h"
#include "RequestHandler.h"
#include "lldb/API/SBInstruction.h"
#include "lldb/lldb-defines.h"

using namespace lldb_dap::protocol;
namespace lldb_dap {

// This request retrieves the possible step-in targets for the specified stack
// frame.
// These targets can be used in the `stepIn` request.
// Clients should only call this request if the corresponding capability
// `supportsStepInTargetsRequest` is true.
llvm::Expected<StepInTargetsResponseBody>
StepInTargetsRequestHandler::Run(const StepInTargetsArguments &args) const {
  dap.step_in_targets.clear();
  const lldb::SBFrame frame = dap.GetLLDBFrame(args.frameId);
  if (!frame.IsValid())
    return llvm::make_error<DAPError>("Failed to get frame for input frameId.");

  lldb::SBAddress pc_addr = frame.GetPCAddress();
  lldb::SBAddress line_end_addr =
      pc_addr.GetLineEntry().GetSameLineContiguousAddressRangeEnd(true);
  lldb::SBInstructionList insts = dap.target.ReadInstructions(
      pc_addr, line_end_addr, /*flavor_string=*/nullptr);

  if (!insts.IsValid())
    return llvm::make_error<DAPError>("Failed to get instructions for frame.");

  StepInTargetsResponseBody body;
  const size_t num_insts = insts.GetSize();
  for (size_t i = 0; i < num_insts; ++i) {
    lldb::SBInstruction inst = insts.GetInstructionAtIndex(i);
    if (!inst.IsValid())
      break;

    const lldb::addr_t inst_addr = inst.GetAddress().GetLoadAddress(dap.target);
    if (inst_addr == LLDB_INVALID_ADDRESS)
      break;

    // Note: currently only x86/x64 supports flow kind.
    const lldb::InstructionControlFlowKind flow_kind =
        inst.GetControlFlowKind(dap.target);

    if (flow_kind == lldb::eInstructionControlFlowKindCall) {

      const llvm::StringRef call_operand_name = inst.GetOperands(dap.target);
      lldb::addr_t call_target_addr = LLDB_INVALID_ADDRESS;
      if (call_operand_name.getAsInteger(0, call_target_addr))
        continue;

      const lldb::SBAddress call_target_load_addr =
          dap.target.ResolveLoadAddress(call_target_addr);
      if (!call_target_load_addr.IsValid())
        continue;

      // The existing ThreadPlanStepInRange only accept step in target
      // function with debug info.
      lldb::SBSymbolContext sc = dap.target.ResolveSymbolContextForAddress(
          call_target_load_addr, lldb::eSymbolContextFunction);

      // The existing ThreadPlanStepInRange only accept step in target
      // function with debug info.
      llvm::StringRef step_in_target_name;
      if (sc.IsValid() && sc.GetFunction().IsValid())
        step_in_target_name = sc.GetFunction().GetDisplayName();

      // Skip call sites if we fail to resolve its symbol name.
      if (step_in_target_name.empty())
        continue;

      StepInTarget target;
      target.id = inst_addr;
      target.label = step_in_target_name;
      dap.step_in_targets.try_emplace(inst_addr, step_in_target_name);
      body.targets.emplace_back(std::move(target));
    }
  }
  return body;
}

} // namespace lldb_dap
