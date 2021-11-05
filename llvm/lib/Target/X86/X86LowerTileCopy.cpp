//===-- X86LowerTileCopy.cpp - Expand Tile Copy Instructions---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the pass which lower AMX tile copy instructions. Since
// there is no tile copy instruction, we need store tile register to stack
// and load from stack to another tile register. We need extra GR to hold
// the stride, and we need stack slot to hold the tile data register.
// We would run this pass after copy propagation, so that we don't miss copy
// optimization. And we would run this pass before prolog/epilog insertion,
// so that we can allocate stack slot.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrBuilder.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "x86-lower-tile-copy"

namespace {

class X86LowerTileCopy : public MachineFunctionPass {
public:
  static char ID;

  X86LowerTileCopy() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "X86 Lower Tile Copy"; }
};

} // namespace

char X86LowerTileCopy::ID = 0;

INITIALIZE_PASS_BEGIN(X86LowerTileCopy, "lowertilecopy", "Tile Copy Lowering",
                      false, false)
INITIALIZE_PASS_END(X86LowerTileCopy, "lowertilecopy", "Tile Copy Lowering",
                    false, false)

void X86LowerTileCopy::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

FunctionPass *llvm::createX86LowerTileCopyPass() {
  return new X86LowerTileCopy();
}

bool X86LowerTileCopy::runOnMachineFunction(MachineFunction &MF) {
  const X86Subtarget &ST = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = ST.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
      if (!MI.isCopy())
        continue;
      MachineOperand &DstMO = MI.getOperand(0);
      MachineOperand &SrcMO = MI.getOperand(1);
      Register SrcReg = SrcMO.getReg();
      Register DstReg = DstMO.getReg();
      if (!X86::TILERegClass.contains(DstReg, SrcReg))
        continue;

      const TargetRegisterInfo *TRI = ST.getRegisterInfo();
      // Allocate stack slot for tile register
      unsigned Size = TRI->getSpillSize(X86::TILERegClass);
      Align Alignment = TRI->getSpillAlign(X86::TILERegClass);
      int TileSS = MF.getFrameInfo().CreateSpillStackObject(Size, Alignment);
      // Allocate stack slot for stride register
      Size = TRI->getSpillSize(X86::GR64RegClass);
      Alignment = TRI->getSpillAlign(X86::GR64RegClass);
      int StrideSS = MF.getFrameInfo().CreateSpillStackObject(Size, Alignment);

      // TODO: Pick a killed regiter to avoid save/reload. There is problem
      // to get live interval in this stage.
      Register GR64Cand = X86::RAX;

      const DebugLoc &DL = MI.getDebugLoc();
      // mov %rax (%sp)
      BuildMI(MBB, MI, DL, TII->get(X86::IMPLICIT_DEF), GR64Cand);
      addFrameReference(BuildMI(MBB, MI, DL, TII->get(X86::MOV64mr)), StrideSS)
          .addReg(GR64Cand);
      // mov 64 %rax
      BuildMI(MBB, MI, DL, TII->get(X86::MOV64ri), GR64Cand).addImm(64);
      // tilestored %tmm, (%sp, %idx)
      unsigned Opc = X86::TILESTORED;
      MachineInstr *NewMI =
          addFrameReference(BuildMI(MBB, MI, DL, TII->get(Opc)), TileSS)
              .addReg(SrcReg, getKillRegState(SrcMO.isKill()));
      MachineOperand &MO = NewMI->getOperand(2);
      MO.setReg(GR64Cand);
      MO.setIsKill(true);
      // tileloadd (%sp, %idx), %tmm
      Opc = X86::TILELOADD;
      NewMI = addFrameReference(BuildMI(MBB, MI, DL, TII->get(Opc), DstReg),
                                TileSS);
      // restore %rax
      // mov (%sp) %rax
      addFrameReference(BuildMI(MBB, MI, DL, TII->get(X86::MOV64rm), GR64Cand),
                        StrideSS);
      MI.eraseFromParent();
      Changed = true;
    }
  }
  return Changed;
}
