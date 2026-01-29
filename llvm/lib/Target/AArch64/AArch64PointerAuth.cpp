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
#include "AArch64RegisterInfo.h"
#include "AArch64Subtarget.h"
#include "llvm/CodeGen/CFIInstBuilder.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"

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

  bool emitSignReturnAddressHardening(MachineFunction &MF);

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

static void emitPACCFI(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                       MachineInstr::MIFlag Flags, bool EmitCFI) {
  if (!EmitCFI)
    return;

  auto &MF = *MBB.getParent();
  auto &MFnI = *MF.getInfo<AArch64FunctionInfo>();

  CFIInstBuilder CFIBuilder(MBB, MBBI, Flags);
  MFnI.branchProtectionPAuthLR() ? CFIBuilder.buildNegateRAStateWithPC()
                                 : CFIBuilder.buildNegateRAState();
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
    emitPACCFI(MBB, MBBI, MachineInstr::FrameSetup, EmitCFI);
    BuildMI(MBB, MBBI, DL,
            TII->get(MFnI.shouldSignWithBKey() ? AArch64::PACIBSPPC
                                               : AArch64::PACIASPPC))
        .setMIFlag(MachineInstr::FrameSetup)
        ->setPreInstrSymbol(MF, MFnI.getSigningInstrLabel());
  } else {
    BuildPACM(*Subtarget, MBB, MBBI, DL, MachineInstr::FrameSetup);
    if (MFnI.branchProtectionPAuthLR())
      emitPACCFI(MBB, MBBI, MachineInstr::FrameSetup, EmitCFI);
    BuildMI(MBB, MBBI, DL,
            TII->get(MFnI.shouldSignWithBKey() ? AArch64::PACIBSP
                                               : AArch64::PACIASP))
        .setMIFlag(MachineInstr::FrameSetup)
        ->setPreInstrSymbol(MF, MFnI.getSigningInstrLabel());
    if (!MFnI.branchProtectionPAuthLR())
      emitPACCFI(MBB, MBBI, MachineInstr::FrameSetup, EmitCFI);
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
  //
  // If the PAC-RET hardening based on load of return address is enabled,
  // fallback to the use of AUTIASP/AUTIBSP and RET.
  bool TerminatorIsCombinable = TI != MBB.end() &&
                                TI->getOpcode() == AArch64::RET &&
                                !MFnI->shouldHardenSignReturnAddress();

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
      emitPACCFI(MBB, MBBI, MachineInstr::FrameDestroy, EmitAsyncCFI);
      BuildMI(MBB, MBBI, DL,
              TII->get(UseBKey ? AArch64::AUTIBSPPCi : AArch64::AUTIASPPCi))
          .addSym(PACSym)
          .setMIFlag(MachineInstr::FrameDestroy);
    } else {
      BuildPACM(*Subtarget, MBB, MBBI, DL, MachineInstr::FrameDestroy, PACSym);
      if (MFnI->branchProtectionPAuthLR())
        emitPACCFI(MBB, MBBI, MachineInstr::FrameDestroy, EmitAsyncCFI);
      BuildMI(MBB, MBBI, DL,
              TII->get(UseBKey ? AArch64::AUTIBSP : AArch64::AUTIASP))
          .setMIFlag(MachineInstr::FrameDestroy);
      if (!MFnI->branchProtectionPAuthLR())
        emitPACCFI(MBB, MBBI, MachineInstr::FrameDestroy, EmitAsyncCFI);
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
    default:
      llvm_unreachable("Unhandled opcode");
    }
    It->eraseFromParent();
    Modified = true;
  }

  Modified |= emitSignReturnAddressHardening(MF);

  return Modified;
}

bool AArch64PointerAuth::emitSignReturnAddressHardening(MachineFunction &MF) {
  const auto *FI = MF.getInfo<AArch64FunctionInfo>();
  assert(FI && "FI can't be null");
  if (!FI->shouldSignReturnAddress(MF) || !FI->shouldHardenSignReturnAddress())
    return false;
  assert(Subtarget && "Subtarget must be initialized");

  RegScavenger RS;
  bool Modified = false;
  for (MachineBasicBlock &MBB : MF) {
    if (!MBB.isReturnBlock())
      continue;

    MachineBasicBlock::iterator MBBI = MBB.getFirstTerminator();

    if (MBBI == MBB.end() || MBBI->getOpcode() != AArch64::RET)
      continue;

    RS.enterBasicBlockEnd(*MBBI->getParent());
    Register XReg = RS.scavengeRegisterBackwards(
        AArch64::GPR64RegClass, MBBI,
        /*RestoreAfter=*/false, /*SPAdj=*/0, /*AllowSpill=*/false);
    if (XReg == AArch64::NoRegister)
      // Couldn't find a free register to use for the hardening. Skip.
      continue;
    RS.setRegUsed(XReg);

    DebugLoc DL = MBBI->getDebugLoc();

    // Register copies are done using ORRXrs directly instead of using the
    // pseudo-instruction COPY because this function can be called after
    // pseudo-instruction expansion takes place, for example via the machine
    // outliner pass.
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::ORRXrs), XReg)
        .addUse(AArch64::XZR)
        .addUse(AArch64::LR)
        .addImm(0)
        .setMIFlag(MachineInstr::FrameDestroy);

    // The XPACI instruction is only available with FEAT_PAUTH. So if the
    // subtarget does not have it, the alternative XPACLRI instruction must be
    // used instead. The latter is in hint space, therefore can be present even
    // if FEAT_PAUTH is absent.
    if (Subtarget->hasPAuth()) {
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::XPACI), XReg)
          .addUse(XReg)
          .setMIFlag(MachineInstr::FrameDestroy);
      Register WReg =
          Subtarget->getRegisterInfo()->getSubReg(XReg, AArch64::sub_32);
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::LDRWui), WReg)
          .addUse(XReg)
          .addImm(0)
          .setMIFlag(MachineInstr::FrameDestroy);
    } else {
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::XPACLRI))
          .setMIFlag(MachineInstr::FrameDestroy);
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::LDRWui), AArch64::W30)
          .addUse(AArch64::LR)
          .addImm(0)
          .setMIFlag(MachineInstr::FrameDestroy);
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::ORRXrs), AArch64::LR)
          .addUse(AArch64::XZR)
          .addUse(XReg)
          .addImm(0)
          .setMIFlag(MachineInstr::FrameDestroy);
      if (MBBI != MBB.end() && (MBBI->getOpcode() == AArch64::RET_ReallyLR ||
                                MBBI->getOpcode() == AArch64::RET)) {
        BuildMI(MBB, MBBI, DL, TII->get(AArch64::RET))
            .addUse(AArch64::LR)
            .copyImplicitOps(*MBBI);
        MBB.erase(MBBI);
      }
    }
    Modified = true;
  }

  return Modified;
}
