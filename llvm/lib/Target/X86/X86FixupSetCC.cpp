//===- X86FixupSetCC.cpp - fix zero-extension of setcc and movbe ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a pass that fixes zero-extension of setcc patterns and
// movbe patterns.
// X86 setcc instructions are modeled to have no input arguments, and a single
// GR8 output argument. This is consistent with other similar instructions
// (e.g. movb), but means it is impossible to directly generate a setcc into
// the lower GR8 of a specified GR32.
// This means that ISel must select (zext (setcc)) into something like
// seta %al; movzbl %al, %eax.
// Unfortunately, this can cause a stall due to the partial register write
// performed by the setcc. Instead, we can use:
// xor %eax, %eax; seta %al
// This both avoids the stall, and encodes shorter.
//
// Furthurmore, we can use:
// setzua %al
// if feature zero-upper is available. It's faster than the xor+setcc sequence.
// When r16-r31 is used, it even encodes shorter.
//
// The same applies to a 16-bit MOVBE load, which also writes only the low
// part of its destination:
// movbew (%rdi), %ax; movzwl %ax, %eax
// becomes
// xorl %eax, %eax; movbew (%rdi), %ax
// If EFLAGS is live at the load, the xor goes before the instruction that
// defines EFLAGS, as for setcc.
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"

using namespace llvm;

#define DEBUG_TYPE "x86-fixup-setcc"

STATISTIC(NumSubstZexts, "Number of setcc + zext pairs substituted");
STATISTIC(NumSubstMovbeZexts, "Number of movbe + zext pairs substituted");

namespace {
class X86FixupSetCCLegacy : public MachineFunctionPass {
public:
  static char ID;

  X86FixupSetCCLegacy() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "X86 Fixup SetCC"; }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addPreserved<MachineRegisterClassInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
} // end anonymous namespace

char X86FixupSetCCLegacy::ID = 0;

INITIALIZE_PASS(X86FixupSetCCLegacy, DEBUG_TYPE, DEBUG_TYPE, false, false)

FunctionPass *llvm::createX86FixupSetCCLegacyPass() {
  return new X86FixupSetCCLegacy();
}

static bool fixupSetCC(MachineFunction &MF) {
  bool Changed = false;
  MachineRegisterInfo *MRI = &MF.getRegInfo();
  const X86Subtarget *ST = &MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = ST->getInstrInfo();

  SmallVector<MachineInstr*, 4> ToErase;

  for (auto &MBB : MF) {
    MachineInstr *FlagsDefMI = nullptr;
    for (auto &MI : MBB) {
      // Remember the most recent preceding eflags defining instruction.
      if (MI.definesRegister(X86::EFLAGS, /*TRI=*/nullptr))
        FlagsDefMI = &MI;

      // Find a setcc/setzucc (if ZU is enabled) that is used by a zext.
      // This doesn't have to be the only use, the transformation is safe
      // regardless.
      if (MI.getOpcode() != X86::SETCCr && MI.getOpcode() != X86::SETZUCCr)
        continue;

      MachineInstr *ZExt = nullptr;
      Register Reg0 = MI.getOperand(0).getReg();
      for (auto &Use : MRI->use_instructions(Reg0))
        if (Use.getOpcode() == X86::MOVZX32rr8)
          ZExt = &Use;

      if (!ZExt)
        continue;

      if (!FlagsDefMI)
        continue;

      // When ZU is available and not disabled by tuning, we rewrite to
      // setzucc, which doesn't clobber eflags; otherwise we insert a MOV32r0
      // (which does clobber eflags) before FlagsDefMI.
      bool UseSetZUCC = ST->hasZU() && !ST->preferLegacySetCC();

      // We'd like to put something that clobbers eflags directly before
      // FlagsDefMI. This can't hurt anything after FlagsDefMI, because
      // it, itself, by definition, clobbers eflags. But it may happen that
      // FlagsDefMI also *uses* eflags, in which case the transformation is
      // invalid.
      if (!UseSetZUCC &&
          FlagsDefMI->readsRegister(X86::EFLAGS, /*TRI=*/nullptr))
        continue;

      // On 32-bit, we need to be careful to force an ABCD register.
      const TargetRegisterClass *RC =
          ST->is64Bit() ? &X86::GR32RegClass : &X86::GR32_ABCDRegClass;
      if (!MRI->constrainRegClass(ZExt->getOperand(0).getReg(), RC)) {
        // If we cannot constrain the register, we would need an additional copy
        // and are better off keeping the MOVZX32rr8 we have now.
        continue;
      }

      ++NumSubstZexts;
      Changed = true;

      // X86 setcc/setzucc only takes an output GR8, so fake a GR32 input by
      // inserting the setcc/setzucc result into the low byte of the zeroed
      // register.
      Register ZeroReg = MRI->createVirtualRegister(RC);
      if (UseSetZUCC) {
        MI.setDesc(TII->get(X86::SETZUCCr));
        BuildMI(*ZExt->getParent(), ZExt, ZExt->getDebugLoc(),
                TII->get(TargetOpcode::IMPLICIT_DEF), ZeroReg);
      } else {
        // Initialize a register with 0. This must go before the eflags def
        BuildMI(MBB, FlagsDefMI, MI.getDebugLoc(), TII->get(X86::MOV32r0),
                ZeroReg)
            .setOperandDead(1);
      }

      BuildMI(*ZExt->getParent(), ZExt, ZExt->getDebugLoc(),
              TII->get(X86::INSERT_SUBREG), ZExt->getOperand(0).getReg())
          .addReg(ZeroReg)
          .addReg(Reg0)
          .addImm(X86::sub_8bit);

      // Redirect the debug-instr-number to the setcc.
      if (unsigned InstrNum = ZExt->peekDebugInstrNum())
        MF.makeDebugValueSubstitution({InstrNum, 0},
                                      {MI.getDebugInstrNum(), 0});

      ToErase.push_back(ZExt);
    }
  }

  for (auto &I : ToErase)
    I->eraseFromParent();

  return Changed;
}

static bool fixupMovbeZExt(MachineFunction &MF) {
  bool Changed = false;
  MachineRegisterInfo *MRI = &MF.getRegInfo();
  const X86Subtarget *ST = &MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo *TII = ST->getInstrInfo();
  const TargetRegisterInfo *TRI = ST->getRegisterInfo();

  SmallVector<MachineInstr *, 4> ToErase;

  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      if (MI.getOpcode() != X86::MOVZX32rr16)
        continue;

      // Look for a zero-extension whose source is a 16-bit MOVBE load
      // (legacy or EVEX encoded).
      Register SrcReg = MI.getOperand(1).getReg();
      if (!SrcReg.isVirtual())
        continue;
      auto *MovbeInstruction = MRI->getVRegDef(SrcReg);
      if (!MovbeInstruction ||
          (MovbeInstruction->getOpcode() != X86::MOVBE16rm &&
           MovbeInstruction->getOpcode() != X86::MOVBE16rm_EVEX))
        continue;

      MachineBasicBlock *MovbeMBB = MovbeInstruction->getParent();
      MachineInstr *InsertBefore = nullptr;
      auto Liveness =
          MovbeMBB->computeRegisterLiveness(TRI, X86::EFLAGS, MovbeInstruction);
      if (Liveness == MachineBasicBlock::LQR_Dead) {
        InsertBefore = MovbeInstruction;
      } else {
        // EFLAGS may be live at the load, or its liveness is unknown. Walk back
        // to the instruction that defines EFLAGS; if it does not read EFLAGS,
        // put the xor right before it, where the flags it clobbers are
        // overwritten at once.
        MachineBasicBlock::iterator I = MovbeInstruction->getIterator();
        while (I != MovbeMBB->begin()) {
          --I;
          if (I->modifiesRegister(X86::EFLAGS, TRI)) {
            // If this instruction also reads EFLAGS (e.g. ADC), give up, like
            // fixupSetCC does; walking further back is possible but rare.
            if (!I->readsRegister(X86::EFLAGS, TRI))
              InsertBefore = &*I;
            break;
          }
        }
      }
      if (!InsertBefore)
        continue;
      Register ZeroReg = MRI->createVirtualRegister(&X86::GR32RegClass);
      BuildMI(*InsertBefore->getParent(), InsertBefore,
              MovbeInstruction->getDebugLoc(), TII->get(X86::MOV32r0), ZeroReg)
          .setOperandDead(1);
      BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(X86::INSERT_SUBREG),
              MI.getOperand(0).getReg())
          .addReg(ZeroReg)
          .addReg(SrcReg)
          .addImm(X86::sub_16bit);
      // Redirect the debug-instr-number to the movbe.
      if (unsigned InstrNum = MI.peekDebugInstrNum())
        MF.makeDebugValueSubstitution(
            {InstrNum, 0}, {MovbeInstruction->getDebugInstrNum(), 0});
      ToErase.push_back(&MI);
      Changed = true;
      ++NumSubstMovbeZexts;
    }
  }
  for (MachineInstr *ZExt : ToErase)
    ZExt->eraseFromParent();

  return Changed;
}

bool X86FixupSetCCLegacy::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = fixupSetCC(MF);
  Changed |= fixupMovbeZExt(MF);
  return Changed;
}

PreservedAnalyses X86FixupSetCCPass::run(MachineFunction &MF,
                                         MachineFunctionAnalysisManager &MFAM) {
  bool Changed = fixupSetCC(MF);
  Changed |= fixupMovbeZExt(MF);
  return Changed ? getMachineFunctionPassPreservedAnalyses()
                       .preserveSet<CFGAnalyses>()
                 : PreservedAnalyses::all();
}
