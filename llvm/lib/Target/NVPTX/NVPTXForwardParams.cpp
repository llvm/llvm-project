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
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

static bool traverseMoveUse(MachineInstr &U, const MachineRegisterInfo &MRI,
                            SmallVectorImpl<MachineInstr *> &RemoveList,
                            SmallVectorImpl<MachineInstr *> &LoadInsts) {
  switch (U.getOpcode()) {
  case NVPTX::LD_f32:
  case NVPTX::LD_f64:
  case NVPTX::LD_i16:
  case NVPTX::LD_i32:
  case NVPTX::LD_i64:
  case NVPTX::LD_i8:
  case NVPTX::LDV_f32_v2:
  case NVPTX::LDV_f32_v4:
  case NVPTX::LDV_f64_v2:
  case NVPTX::LDV_f64_v4:
  case NVPTX::LDV_i16_v2:
  case NVPTX::LDV_i16_v4:
  case NVPTX::LDV_i32_v2:
  case NVPTX::LDV_i32_v4:
  case NVPTX::LDV_i64_v2:
  case NVPTX::LDV_i64_v4:
  case NVPTX::LDV_i8_v2:
  case NVPTX::LDV_i8_v4: {
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
                          SmallVectorImpl<MachineInstr *> &RemoveList) {
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
        ->ChangeToImmediate(NVPTX::AddressSpace::Param);
  }
  return true;
}

static bool forwardDeviceParams(MachineFunction &MF) {
  const auto &MRI = MF.getRegInfo();

  bool Changed = false;
  SmallVector<MachineInstr *, 16> RemoveList;
  for (auto &MI : make_early_inc_range(*MF.begin()))
    if (MI.getOpcode() == NVPTX::MOV32_PARAM ||
        MI.getOpcode() == NVPTX::MOV64_PARAM)
      Changed |= eliminateMove(MI, MRI, RemoveList);

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
