//===- NVPTXForwardKernelParams.cpp - Forward Grid Constant Kernel Params -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A grid constant kernel parameter is lowered to a bare parameter symbol, which
// SelectionDAG can fold directly into a `ld.param [sym+offset]`. That fold is
// limited to a single basic block: when a load lives in another block the
// symbol is carried across the block boundary in a register, materialized with
// a generic `mov`, and the loads address parameter space through that register:
//
//     mov.b64       %rd1, kernel_param_0;
//     ld.param.b64  %rd2, [%rd1];
//     ld.param.b64  %rd3, [%rd1+8];
//
// When every use of such a `mov` is a load, the register is just redundant
// address arithmetic and the symbol can be folded back into the loads:
//
//     ld.param.b64  %rd2, [kernel_param_0];
//     ld.param.b64  %rd3, [kernel_param_0+8];
//
// This is the same simplification NVPTXForwardParams performs for device
// function parameters, which are represented differently (the dedicated
// MOV{32,64}_PARAM produced by NVPTXISD::MoveParam).
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
#include "llvm/IR/Function.h"

using namespace llvm;

static bool isParamLoad(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case NVPTX::LD_i16:
  case NVPTX::LD_i32:
  case NVPTX::LD_i64:
  case NVPTX::LDV_i16_v2:
  case NVPTX::LDV_i16_v4:
  case NVPTX::LDV_i32_v2:
  case NVPTX::LDV_i32_v4:
  case NVPTX::LDV_i64_v2:
  case NVPTX::LDV_i64_v4:
    return true;
  default:
    return false;
  }
}

// If every use of the parameter address materialized by \p Mov is a load, fold
// the parameter symbol back into those loads (reading directly from
// `.param::entry`) and mark the now-dead `mov` for removal. Returns false,
// leaving everything in place, if any use is not a load, e.g. because the
// address also feeds arithmetic or escapes.
static bool forwardKernelParam(MachineInstr &Mov,
                               const MachineRegisterInfo &MRI,
                               SmallVectorImpl<MachineInstr *> &RemoveList) {
  const Register AddrReg = Mov.getOperand(0).getReg();

  SmallVector<MachineInstr *, 16> Loads;
  for (MachineInstr &U : MRI.use_instructions(AddrReg)) {
    if (!isParamLoad(U))
      return false;
    Loads.push_back(&U);
  }

  const MachineOperand &ParamSymbol = Mov.getOperand(1);
  assert(ParamSymbol.isSymbol());

  // Operand indices into the LD/LDV instructions, counted from the first use
  // operand (uses() skips the def).
  constexpr unsigned LDInstAddrSpaceOpIdx = 2;
  constexpr unsigned LDInstBasePtrOpIdx = 6;
  for (MachineInstr *LI : Loads) {
    (LI->uses().begin() + LDInstBasePtrOpIdx)
        ->ChangeToES(ParamSymbol.getSymbolName());
    (LI->uses().begin() + LDInstAddrSpaceOpIdx)
        ->ChangeToImmediate(NVPTX::AddressSpace::EntryParam);
  }
  RemoveList.push_back(&Mov);
  return true;
}

static bool forwardKernelParams(MachineFunction &MF) {
  const Function &F = MF.getFunction();
  // Only kernels have grid constant parameters; device function parameters use
  // the dedicated `mov` handled by NVPTXForwardParams.
  if (!isKernelFunction(F))
    return false;

  // The generic symbol `mov` (MOV_B{32,64}_sym) is also used to materialize
  // global addresses, so collect the parameter symbol names to tell them apart.
  const NVPTXTargetLowering *TLI =
      MF.getSubtarget<NVPTXSubtarget>().getTargetLowering();
  StringSet<> ParamSymbols;
  for (const Argument &Arg : F.args())
    ParamSymbols.insert(TLI->getParamName(&F, Arg.getArgNo()));

  const MachineRegisterInfo &MRI = MF.getRegInfo();
  bool Changed = false;
  SmallVector<MachineInstr *, 16> RemoveList;
  // The `mov` is usually sunk into the block that uses it, so scan the whole
  // function rather than just the entry block.
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : make_early_inc_range(MBB)) {
      if (MI.getOpcode() != NVPTX::MOV_B32_sym &&
          MI.getOpcode() != NVPTX::MOV_B64_sym)
        continue;
      const MachineOperand &Src = MI.getOperand(1);
      if (Src.isSymbol() && ParamSymbols.contains(Src.getSymbolName()))
        Changed |= forwardKernelParam(MI, MRI, RemoveList);
    }

  for (MachineInstr *MI : RemoveList)
    MI->eraseFromParent();

  return Changed;
}

/// ----------------------------------------------------------------------------
///                       Pass (Manager) Boilerplate
/// ----------------------------------------------------------------------------

namespace {
struct NVPTXForwardKernelParamsPass : public MachineFunctionPass {
  static char ID;
  NVPTXForwardKernelParamsPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
} // namespace

char NVPTXForwardKernelParamsPass::ID = 0;

INITIALIZE_PASS(NVPTXForwardKernelParamsPass, "nvptx-forward-kernel-params",
                "NVPTX Forward Kernel Params", false, false)

bool NVPTXForwardKernelParamsPass::runOnMachineFunction(MachineFunction &MF) {
  return forwardKernelParams(MF);
}

MachineFunctionPass *llvm::createNVPTXForwardKernelParamsPass() {
  return new NVPTXForwardKernelParamsPass();
}
