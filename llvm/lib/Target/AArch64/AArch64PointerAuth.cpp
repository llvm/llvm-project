//===-- AArch64PointerAuth.cpp -- Harden code using PAuth ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AArch64PointerAuth.h"

#include "AArch64.h"
#include "AArch64InstrInfo.h"
#include "AArch64MachineFunctionInfo.h"
#include "AArch64Subtarget.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"

using namespace llvm;
using namespace llvm::AArch64PAuth;

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

  /// Stores blend(AddrDisc, IntDisc) to the Result register.
  void emitBlend(MachineBasicBlock::iterator MBBI, Register Result,
                 Register AddrDisc, unsigned IntDisc) const;

  /// Expands PAUTH_BLEND pseudo instruction.
  void expandPAuthBlend(MachineBasicBlock::iterator MBBI) const;

  bool checkAuthenticatedLR(MachineBasicBlock::iterator TI) const;
};

} // end anonymous namespace

INITIALIZE_PASS(AArch64PointerAuth, "aarch64-ptrauth",
                AARCH64_POINTER_AUTH_NAME, false, false)

FunctionPass *llvm::createAArch64PointerAuthPass() {
  return new AArch64PointerAuth();
}

char AArch64PointerAuth::ID = 0;

static void emitPACSymOffsetIntoX16(const TargetInstrInfo &TII,
                                    MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator I, DebugLoc DL,
                                    MCSymbol *PACSym) {
  BuildMI(MBB, I, DL, TII.get(AArch64::ADRP), AArch64::X16)
      .addSym(PACSym, AArch64II::MO_PAGE);
  BuildMI(MBB, I, DL, TII.get(AArch64::ADDXri), AArch64::X16)
      .addReg(AArch64::X16)
      .addSym(PACSym, AArch64II::MO_PAGEOFF | AArch64II::MO_NC)
      .addImm(0);
}

// Where PAuthLR support is not known at compile time, it is supported using
// PACM. PACM is in the hint space so has no effect when PAuthLR is not
// supported by the hardware, but will alter the behaviour of PACI*SP, AUTI*SP
// and RETAA/RETAB if the hardware supports PAuthLR.
static void BuildPACM(const AArch64Subtarget &Subtarget, MachineBasicBlock &MBB,
                      MachineBasicBlock::iterator MBBI, DebugLoc DL,
                      MachineInstr::MIFlag Flags, MCSymbol *PACSym = nullptr) {
  const TargetInstrInfo *TII = Subtarget.getInstrInfo();
  auto &MFnI = *MBB.getParent()->getInfo<AArch64FunctionInfo>();

  // Offset to PAC*SP using ADRP + ADD.
  if (PACSym) {
    assert(Flags == MachineInstr::FrameDestroy);
    emitPACSymOffsetIntoX16(*TII, MBB, MBBI, DL, PACSym);
  }

  // Only emit PACM if -mbranch-protection has +pc and the target does not
  // have feature +pauth-lr.
  if (MFnI.branchProtectionPAuthLR() && !Subtarget.hasPAuthLR())
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::PACM)).setMIFlag(Flags);
}

static void emitPACCFI(const AArch64Subtarget &Subtarget,
                       MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                       DebugLoc DL, MachineInstr::MIFlag Flags, bool EmitCFI) {
  if (!EmitCFI)
    return;

  const TargetInstrInfo *TII = Subtarget.getInstrInfo();
  auto &MF = *MBB.getParent();
  auto &MFnI = *MF.getInfo<AArch64FunctionInfo>();

  auto CFIInst = MFnI.branchProtectionPAuthLR()
                     ? MCCFIInstruction::createNegateRAStateWithPC(nullptr)
                     : MCCFIInstruction::createNegateRAState(nullptr);

  unsigned CFIIndex = MF.addFrameInst(CFIInst);
  BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
      .addCFIIndex(CFIIndex)
      .setMIFlags(Flags);
}

void AArch64PointerAuth::signLR(MachineFunction &MF,
                                MachineBasicBlock::iterator MBBI) const {
  auto &MFnI = *MF.getInfo<AArch64FunctionInfo>();
  bool UseBKey = MFnI.shouldSignWithBKey();
  bool EmitCFI = MFnI.needsDwarfUnwindInfo(MF);
  bool NeedsWinCFI = MF.hasWinCFI();

  MachineBasicBlock &MBB = *MBBI->getParent();

  // Debug location must be unknown, see AArch64FrameLowering::emitPrologue.
  DebugLoc DL;

  if (UseBKey) {
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::EMITBKEY))
        .setMIFlag(MachineInstr::FrameSetup);
  }

  // PAuthLR authentication instructions need to know the value of PC at the
  // point of signing (PACI*).
  if (MFnI.branchProtectionPAuthLR()) {
    MCSymbol *PACSym = MF.getContext().createTempSymbol();
    MFnI.setSigningInstrLabel(PACSym);
  }

  // No SEH opcode for this one; it doesn't materialize into an
  // instruction on Windows.
  if (MFnI.branchProtectionPAuthLR() && Subtarget->hasPAuthLR()) {
    emitPACCFI(*Subtarget, MBB, MBBI, DL, MachineInstr::FrameSetup, EmitCFI);
    BuildMI(MBB, MBBI, DL,
            TII->get(MFnI.shouldSignWithBKey() ? AArch64::PACIBSPPC
                                               : AArch64::PACIASPPC))
        .setMIFlag(MachineInstr::FrameSetup)
        ->setPreInstrSymbol(MF, MFnI.getSigningInstrLabel());
  } else {
    BuildPACM(*Subtarget, MBB, MBBI, DL, MachineInstr::FrameSetup);
    emitPACCFI(*Subtarget, MBB, MBBI, DL, MachineInstr::FrameSetup, EmitCFI);
    BuildMI(MBB, MBBI, DL,
            TII->get(MFnI.shouldSignWithBKey() ? AArch64::PACIBSP
                                               : AArch64::PACIASP))
        .setMIFlag(MachineInstr::FrameSetup)
        ->setPreInstrSymbol(MF, MFnI.getSigningInstrLabel());
  }

  if (!EmitCFI && NeedsWinCFI) {
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
  MCSymbol *PACSym = MFnI->getSigningInstrLabel();

  if (Subtarget->hasPAuth() && TerminatorIsCombinable && !NeedsWinCFI &&
      !MF.getFunction().hasFnAttribute(Attribute::ShadowCallStack)) {
    if (MFnI->branchProtectionPAuthLR() && Subtarget->hasPAuthLR()) {
      assert(PACSym && "No PAC instruction to refer to");
      emitPACSymOffsetIntoX16(*TII, MBB, MBBI, DL, PACSym);
      BuildMI(MBB, TI, DL,
              TII->get(UseBKey ? AArch64::RETABSPPCi : AArch64::RETAASPPCi))
          .addSym(PACSym)
          .copyImplicitOps(*MBBI)
          .setMIFlag(MachineInstr::FrameDestroy);
    } else {
      BuildPACM(*Subtarget, MBB, TI, DL, MachineInstr::FrameDestroy, PACSym);
      BuildMI(MBB, TI, DL, TII->get(UseBKey ? AArch64::RETAB : AArch64::RETAA))
          .copyImplicitOps(*MBBI)
          .setMIFlag(MachineInstr::FrameDestroy);
    }
    MBB.erase(TI);
  } else {
    if (MFnI->branchProtectionPAuthLR() && Subtarget->hasPAuthLR()) {
      assert(PACSym && "No PAC instruction to refer to");
      emitPACSymOffsetIntoX16(*TII, MBB, MBBI, DL, PACSym);
      emitPACCFI(*Subtarget, MBB, MBBI, DL, MachineInstr::FrameDestroy,
                 EmitAsyncCFI);
      BuildMI(MBB, MBBI, DL,
              TII->get(UseBKey ? AArch64::AUTIBSPPCi : AArch64::AUTIASPPCi))
          .addSym(PACSym)
          .setMIFlag(MachineInstr::FrameDestroy);
    } else {
      BuildPACM(*Subtarget, MBB, MBBI, DL, MachineInstr::FrameDestroy, PACSym);
      emitPACCFI(*Subtarget, MBB, MBBI, DL, MachineInstr::FrameDestroy,
                 EmitAsyncCFI);
      BuildMI(MBB, MBBI, DL,
              TII->get(UseBKey ? AArch64::AUTIBSP : AArch64::AUTIASP))
          .setMIFlag(MachineInstr::FrameDestroy);
    }

    if (NeedsWinCFI) {
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::SEH_PACSignLR))
          .setMIFlag(MachineInstr::FrameDestroy);
    }
  }
}

unsigned llvm::AArch64PAuth::getCheckerSizeInBytes(AuthCheckMethod Method) {
  switch (Method) {
  case AuthCheckMethod::None:
    return 0;
  case AuthCheckMethod::DummyLoad:
    return 4;
  case AuthCheckMethod::HighBitsNoTBI:
    return 12;
  case AuthCheckMethod::XPACHint:
  case AuthCheckMethod::XPAC:
    return 20;
  }
  llvm_unreachable("Unknown AuthCheckMethod enum");
}

void AArch64PointerAuth::emitBlend(MachineBasicBlock::iterator MBBI,
                                   Register Result, Register AddrDisc,
                                   unsigned IntDisc) const {
  MachineBasicBlock &MBB = *MBBI->getParent();
  DebugLoc DL = MBBI->getDebugLoc();

  if (Result != AddrDisc)
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::ORRXrs), Result)
        .addReg(AArch64::XZR)
        .addReg(AddrDisc)
        .addImm(0);

  BuildMI(MBB, MBBI, DL, TII->get(AArch64::MOVKXi), Result)
      .addReg(Result)
      .addImm(IntDisc)
      .addImm(48);
}

void AArch64PointerAuth::expandPAuthBlend(
    MachineBasicBlock::iterator MBBI) const {
  Register ResultReg = MBBI->getOperand(0).getReg();
  Register AddrDisc = MBBI->getOperand(1).getReg();
  unsigned IntDisc = MBBI->getOperand(2).getImm();
  emitBlend(MBBI, ResultReg, AddrDisc, IntDisc);
}

bool AArch64PointerAuth::runOnMachineFunction(MachineFunction &MF) {
  Subtarget = &MF.getSubtarget<AArch64Subtarget>();
  TII = Subtarget->getInstrInfo();

  SmallVector<MachineBasicBlock::instr_iterator> PAuthPseudoInstrs;

  bool Modified = false;

  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      switch (MI.getOpcode()) {
      default:
        break;
      case AArch64::PAUTH_PROLOGUE:
      case AArch64::PAUTH_EPILOGUE:
      case AArch64::PAUTH_BLEND:
        PAuthPseudoInstrs.push_back(MI.getIterator());
        break;
      }
    }
  }

  for (auto It : PAuthPseudoInstrs) {
    switch (It->getOpcode()) {
    case AArch64::PAUTH_PROLOGUE:
      signLR(MF, It);
      break;
    case AArch64::PAUTH_EPILOGUE:
      authenticateLR(MF, It);
      break;
    case AArch64::PAUTH_BLEND:
      expandPAuthBlend(It);
      break;
    default:
      llvm_unreachable("Unhandled opcode");
    }
    It->eraseFromParent();
    Modified = true;
  }

  return Modified;
}
