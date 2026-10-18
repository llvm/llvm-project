//===-- AGXFoldOperands.cpp - Fold uniform registers into operands --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Kernel arguments live in uniform registers, which are never allocated:
// instruction selection copies them into virtual general purpose registers.
// Like SIFoldOperands, this pass folds such copies into the operands that
// can read uniform registers directly (ALU sources and memory bases), which
// saves both the moves and the general purpose registers.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "AGXInstrInfo.h"
#include "AGXRegisterInfo.h"
#include "AGXSubtarget.h"
#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "agx-fold-operands"

namespace {

class AGXFoldOperands : public MachineFunctionPass {
public:
  static char ID;
  AGXFoldOperands() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "AGX Fold Operands"; }
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // namespace

char AGXFoldOperands::ID = 0;

INITIALIZE_PASS(AGXFoldOperands, DEBUG_TYPE, "AGX Fold Operands", false,
                false)

FunctionPass *llvm::createAGXFoldOperandsPass() {
  return new AGXFoldOperands();
}

/// Memory base registers encode uniform pairs below u128 only.
static constexpr unsigned MaxUniformBaseHalf = 0x100;

bool AGXFoldOperands::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;
  const AGXSubtarget &ST = MF.getSubtarget<AGXSubtarget>();
  const AGXInstrInfo *TII = ST.getInstrInfo();
  const AGXRegisterInfo *TRI = ST.getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  auto IsUniformCopy = [&](const MachineInstr &MI) {
    if (!MI.isCopy())
      return false;
    Register Dst = MI.getOperand(0).getReg();
    Register Src = MI.getOperand(1).getReg();
    return Dst.isVirtual() && Src.isPhysical() &&
           !MI.getOperand(1).getSubReg() &&
           AGX::getRegInfo(Src, *TRI).isUniform();
  };

  SmallVector<MachineInstr *, 32> Worklist;
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      if (IsUniformCopy(MI))
        Worklist.push_back(&MI);

  bool Changed = false;
  while (!Worklist.empty()) {
    MachineInstr &Copy = *Worklist.pop_back_val();
    Register Dst = Copy.getOperand(0).getReg();
    Register Src = Copy.getOperand(1).getReg();

    for (MachineOperand &Use :
         make_early_inc_range(MRI.use_nodbg_operands(Dst))) {
      MachineInstr &UseMI = *Use.getParent();
      MCRegister Folded = Src;
      if (Use.getSubReg())
        Folded = TRI->getSubReg(Src, Use.getSubReg());
      if (!Folded)
        continue;

      // Copies of parts of a uniform tuple become uniform copies themselves.
      if (UseMI.isCopy()) {
        if (Use.getSubReg() && UseMI.getOperand(0).getReg().isVirtual()) {
          Use.setReg(Folded);
          Use.setSubReg(0);
          Use.setIsKill(false);
          Worklist.push_back(&UseMI);
          Changed = true;
        }
        continue;
      }
      if (UseMI.isPHI() || UseMI.isRegSequence() || UseMI.isInlineAsm())
        continue;
      unsigned OpIdx = UseMI.getOperandNo(&Use);
      if (OpIdx >= UseMI.getDesc().getNumOperands())
        continue;
      const TargetRegisterClass *RC = TII->getRegClass(UseMI.getDesc(), OpIdx);
      if (!RC || !RC->contains(Folded))
        continue;
      if (RC == &AGX::BASE64RegClass &&
          AGX::getRegInfo(Folded, *TRI).HalfIndex >= MaxUniformBaseHalf)
        continue;
      Use.setReg(Folded);
      Use.setSubReg(0);
      Use.setIsKill(false);
      Changed = true;
    }

    if (MRI.use_nodbg_empty(Dst)) {
      Copy.eraseFromParent();
      Changed = true;
    }
  }
  return Changed;
}
