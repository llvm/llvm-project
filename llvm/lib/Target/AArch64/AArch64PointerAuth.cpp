//===-- AArch64PointerAuth.cpp -- Harden code using PAuth ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64MachineFunctionInfo.h"
#include "AArch64Subtarget.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"

using namespace llvm;

#define AARCH64_POINTER_AUTH_NAME "AArch64 Pointer Authentication"

namespace {

class AArch64PointerAuth : public MachineFunctionPass {
public:
  static char ID;

  AArch64PointerAuth() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return AARCH64_POINTER_AUTH_NAME; }

private:
  const AArch64Subtarget *Subtarget = nullptr;
  const AArch64InstrInfo *TII = nullptr;

  void signLR(MachineFunction &MF, MachineBasicBlock::iterator MBBI) const;

  void authenticateLR(MachineFunction &MF,
                      MachineBasicBlock::iterator MBBI) const;
};

} // end anonymous namespace

INITIALIZE_PASS(AArch64PointerAuth, "aarch64-ptrauth",
                AARCH64_POINTER_AUTH_NAME, false, false)

FunctionPass *llvm::createAArch64PointerAuthPass() {
  return new AArch64PointerAuth();
}

char AArch64PointerAuth::ID = 0;

void AArch64PointerAuth::signLR(MachineFunction &MF,
                                MachineBasicBlock::iterator MBBI) const {
  const AArch64FunctionInfo *MFnI = MF.getInfo<AArch64FunctionInfo>();
  bool UseBKey = MFnI->shouldSignWithBKey();
  bool EmitCFI = MFnI->needsDwarfUnwindInfo(MF);
  bool NeedsWinCFI = MF.hasWinCFI();

  MachineBasicBlock &MBB = *MBBI->getParent();

  // Debug location must be unknown, see AArch64FrameLowering::emitPrologue.
  DebugLoc DL;

  if (UseBKey) {
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::EMITBKEY))
        .setMIFlag(MachineInstr::FrameSetup);
  }

  // No SEH opcode for this one; it doesn't materialize into an
  // instruction on Windows.
  BuildMI(MBB, MBBI, DL,
          TII->get(UseBKey ? AArch64::PACIBSP : AArch64::PACIASP))
      .setMIFlag(MachineInstr::FrameSetup);

  if (EmitCFI) {
    unsigned CFIIndex =
        MF.addFrameInst(MCCFIInstruction::createNegateRAState(nullptr));
    BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
        .addCFIIndex(CFIIndex)
        .setMIFlags(MachineInstr::FrameSetup);
  } else if (NeedsWinCFI) {
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::SEH_PACSignLR))
        .setMIFlag(MachineInstr::FrameSetup);
  }
}

void AArch64PointerAuth::authenticateLR(
    MachineFunction &MF, MachineBasicBlock::iterator MBBI) const {
  const AArch64FunctionInfo *MFnI = MF.getInfo<AArch64FunctionInfo>();
  bool UseBKey = MFnI->shouldSignWithBKey();
  bool EmitAsyncCFI = MFnI->needsAsyncDwarfUnwindInfo(MF);
  bool NeedsWinCFI = MF.hasWinCFI();

  MachineBasicBlock &MBB = *MBBI->getParent();
  DebugLoc DL = MBBI->getDebugLoc();
  // MBBI points to a PAUTH_EPILOGUE instruction to be replaced and
  // TI points to a terminator instruction that may or may not be combined.
  // Note that inserting new instructions "before MBBI" and "before TI" is
  // not the same because if ShadowCallStack is enabled, its instructions
  // are placed between MBBI and TI.
  MachineBasicBlock::iterator TI = MBB.getFirstInstrTerminator();

  // The AUTIASP instruction assembles to a hint instruction before v8.3a so
  // this instruction can safely used for any v8a architecture.
  // From v8.3a onwards there are optimised authenticate LR and return
  // instructions, namely RETA{A,B}, that can be used instead. In this case the
  // DW_CFA_AARCH64_negate_ra_state can't be emitted.
  bool TerminatorIsCombinable =
      TI != MBB.end() && TI->getOpcode() == AArch64::RET;
  if (Subtarget->hasPAuth() && TerminatorIsCombinable && !NeedsWinCFI &&
      !MF.getFunction().hasFnAttribute(Attribute::ShadowCallStack)) {
    unsigned CombinedRetOpcode = UseBKey ? AArch64::RETAB : AArch64::RETAA;
    BuildMI(MBB, TI, DL, TII->get(CombinedRetOpcode)).copyImplicitOps(*TI);
    MBB.erase(TI);
  } else {
    unsigned AutOpcode = UseBKey ? AArch64::AUTIBSP : AArch64::AUTIASP;
    BuildMI(MBB, MBBI, DL, TII->get(AutOpcode))
        .setMIFlag(MachineInstr::FrameDestroy);

    if (EmitAsyncCFI) {
      unsigned CFIIndex =
          MF.addFrameInst(MCCFIInstruction::createNegateRAState(nullptr));
      BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
          .addCFIIndex(CFIIndex)
          .setMIFlags(MachineInstr::FrameDestroy);
    }
    if (NeedsWinCFI) {
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::SEH_PACSignLR))
          .setMIFlag(MachineInstr::FrameDestroy);
    }
  }
}

bool AArch64PointerAuth::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getInfo<AArch64FunctionInfo>()->shouldSignReturnAddress(true))
    return false;

  Subtarget = &MF.getSubtarget<AArch64Subtarget>();
  TII = Subtarget->getInstrInfo();

  SmallVector<MachineBasicBlock::iterator> DeletedInstrs;
  bool Modified = false;

  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      auto It = MI.getIterator();
      switch (MI.getOpcode()) {
      default:
        // do nothing
        break;
      case AArch64::PAUTH_PROLOGUE:
        signLR(MF, It);
        DeletedInstrs.push_back(It);
        Modified = true;
        break;
      case AArch64::PAUTH_EPILOGUE:
        authenticateLR(MF, It);
        DeletedInstrs.push_back(It);
        Modified = true;
        break;
      }
    }
  }

  for (auto MI : DeletedInstrs)
    MI->eraseFromParent();

  return Modified;
}
