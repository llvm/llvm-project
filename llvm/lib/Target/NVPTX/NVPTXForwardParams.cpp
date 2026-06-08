//- NVPTXForwardParams.cpp - NVPTX Forward Device Params Removing Local Copy -//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// PTX supports 2 methods of accessing device function parameters:
//
//   - "simple" case: If a parameters is only loaded, and all loads can address
//     the parameter via a constant offset, then the parameter may be loaded via
//     the ".param" address space. This case is not possible if the parameters
//     is stored to or has it's address taken. This method is preferable when
//     possible. Ex:
//
//            ld.param.u32    %r1, [foo_param_1];
//            ld.param.u32    %r2, [foo_param_1+4];
//
//   - "move param" case: For more complex cases the address of the param may be
//     placed in a register via a "mov" instruction. This "mov" also implicitly
//     moves the param to the ".local" address space and allows for it to be
//     written to. This essentially defers the responsibilty of the byval copy
//     to the PTX calling convention.
//
//            mov.b64         %rd1, foo_param_0;
//            st.local.u32    [%rd1], 42;
//            add.u64         %rd3, %rd1, %rd2;
//            ld.local.u32    %r2, [%rd3];
//
// In NVPTXLowerArgs and SelectionDAG, we pessimistically assume that all
// parameters will use the "move param" case and the local address space. This
// pass is responsible for switching to the "simple" case when possible, as it
// is more efficient.
//
// We do this by simply traversing uses of the param "mov" instructions an
// trivially checking if they are all loads.
//
// The same "mov" also appears for grid constant kernel parameters: those are
// lowered to a bare parameter symbol that SelectionDAG can fold directly into a
// `ld.param` only within a single basic block. When such a parameter is used
// outside its defining block its address is instead materialized with a `mov`,
// and this pass forwards it back into the loads just like the device parameter
// case above.
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "NVPTXSubtarget.h"
#include "NVVMProperties.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

static bool traverseMoveUse(MachineInstr &U, const MachineRegisterInfo &MRI,
                            SmallVectorImpl<MachineInstr *> &RemoveList,
                            SmallVectorImpl<MachineInstr *> &LoadInsts) {
  switch (U.getOpcode()) {
  case NVPTX::LD_i16:
  case NVPTX::LD_i32:
  case NVPTX::LD_i64:
  case NVPTX::LDV_i16_v2:
  case NVPTX::LDV_i16_v4:
  case NVPTX::LDV_i32_v2:
  case NVPTX::LDV_i32_v4:
  case NVPTX::LDV_i64_v2:
  case NVPTX::LDV_i64_v4: {
    LoadInsts.push_back(&U);
    return true;
  }
  case NVPTX::cvta_local:
  case NVPTX::cvta_local_64:
  case NVPTX::cvta_to_local:
  case NVPTX::cvta_to_local_64: {
    for (auto &U2 : MRI.use_instructions(U.operands_begin()->getReg()))
      if (!traverseMoveUse(U2, MRI, RemoveList, LoadInsts))
        return false;

    RemoveList.push_back(&U);
    return true;
  }
  default:
    return false;
  }
}

static bool eliminateMove(MachineInstr &Mov, const MachineRegisterInfo &MRI,
                          SmallVectorImpl<MachineInstr *> &RemoveList,
                          unsigned ParamAddrSpace) {
  SmallVector<MachineInstr *, 16> MaybeRemoveList;
  SmallVector<MachineInstr *, 16> LoadInsts;

  for (auto &U : MRI.use_instructions(Mov.operands_begin()->getReg()))
    if (!traverseMoveUse(U, MRI, MaybeRemoveList, LoadInsts))
      return false;

  RemoveList.append(MaybeRemoveList);
  RemoveList.push_back(&Mov);

  const MachineOperand *ParamSymbol = Mov.uses().begin();
  assert(ParamSymbol->isSymbol());

  constexpr unsigned LDInstBasePtrOpIdx = 6;
  constexpr unsigned LDInstAddrSpaceOpIdx = 2;
  for (auto *LI : LoadInsts) {
    (LI->uses().begin() + LDInstBasePtrOpIdx)
        ->ChangeToES(ParamSymbol->getSymbolName());
    (LI->uses().begin() + LDInstAddrSpaceOpIdx)
        ->ChangeToImmediate(ParamAddrSpace);
  }
  return true;
}

// Return true if \p MI is a `mov` that materializes the address of a byval
// parameter and is a candidate for forwarding. Device function parameters are
// lowered through NVPTXISD::MoveParam (MOV{32,64}_PARAM), while grid constant
// kernel parameters are lowered to a bare parameter symbol that ISel
// materializes with the generic symbol `mov` (MOV_B{32,64}_sym) when the
// parameter is used outside its defining block. The latter opcode is shared
// with global address materialization, so check that the operand actually
// names a parameter of this function.
static bool isForwardableParamMove(const MachineInstr &MI,
                                   const StringSet<> &ParamSymbols) {
  switch (MI.getOpcode()) {
  case NVPTX::MOV32_PARAM:
  case NVPTX::MOV64_PARAM:
    return true;
  case NVPTX::MOV_B32_sym:
  case NVPTX::MOV_B64_sym: {
    const MachineOperand &Src = MI.getOperand(1);
    return Src.isSymbol() && ParamSymbols.contains(Src.getSymbolName());
  }
  default:
    return false;
  }
}

static bool forwardDeviceParams(MachineFunction &MF) {
  const auto &MRI = MF.getRegInfo();
  const Function &F = MF.getFunction();

  // Kernel parameters live in the read-only ".param::entry" space, device
  // function parameters in ".param::func"; both are read with `ld.param`.
  const bool IsKernel = isKernelFunction(F);
  const unsigned ParamAddrSpace = IsKernel ? NVPTX::AddressSpace::EntryParam
                                           : NVPTX::AddressSpace::DeviceParam;

  // Grid constant kernel parameters are materialized with the generic symbol
  // `mov`, which is also used for global addresses. Collect the parameter
  // symbol names so they can be told apart (see isForwardableParamMove).
  StringSet<> ParamSymbols;
  if (IsKernel) {
    const NVPTXTargetLowering *TLI =
        MF.getSubtarget<NVPTXSubtarget>().getTargetLowering();
    for (const Argument &Arg : F.args())
      ParamSymbols.insert(TLI->getParamName(&F, Arg.getArgNo()));
  }

  // The `mov` may have been sunk out of the entry block (e.g. to the single
  // block that uses it), so scan the whole function rather than just the entry.
  bool Changed = false;
  SmallVector<MachineInstr *, 16> RemoveList;
  for (MachineBasicBlock &MBB : MF)
    for (auto &MI : make_early_inc_range(MBB))
      if (isForwardableParamMove(MI, ParamSymbols))
        Changed |= eliminateMove(MI, MRI, RemoveList, ParamAddrSpace);

  for (auto *MI : RemoveList)
    MI->eraseFromParent();

  return Changed;
}

/// ----------------------------------------------------------------------------
///                       Pass (Manager) Boilerplate
/// ----------------------------------------------------------------------------

namespace {
struct NVPTXForwardParamsPass : public MachineFunctionPass {
  static char ID;
  NVPTXForwardParamsPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
} // namespace

char NVPTXForwardParamsPass::ID = 0;

INITIALIZE_PASS(NVPTXForwardParamsPass, "nvptx-forward-params",
                "NVPTX Forward Params", false, false)

bool NVPTXForwardParamsPass::runOnMachineFunction(MachineFunction &MF) {
  return forwardDeviceParams(MF);
}

MachineFunctionPass *llvm::createNVPTXForwardParamsPass() {
  return new NVPTXForwardParamsPass();
}
