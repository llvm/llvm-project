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
  /// An immediate operand passed to BRK instruction, if it is ever emitted.
  const unsigned BrkOperand = 0xc471;

  const AArch64Subtarget *Subtarget = nullptr;
  const AArch64InstrInfo *TII = nullptr;
  const AArch64RegisterInfo *TRI = nullptr;

  void signLR(MachineFunction &MF, MachineBasicBlock::iterator MBBI) const;

  void authenticateLR(MachineFunction &MF,
                      MachineBasicBlock::iterator MBBI) const;

  bool checkAuthenticatedLR(MachineBasicBlock::iterator TI) const;
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

namespace {

// Mark dummy LDR instruction as volatile to prevent removing it as dead code.
MachineMemOperand *createCheckMemOperand(MachineFunction &MF,
                                         const AArch64Subtarget &Subtarget) {
  MachinePointerInfo PointerInfo(Subtarget.getAddressCheckPSV());
  auto MOVolatileLoad =
      MachineMemOperand::MOLoad | MachineMemOperand::MOVolatile;

  return MF.getMachineMemOperand(PointerInfo, MOVolatileLoad, 4, Align(4));
}

} // namespace

MachineBasicBlock &llvm::AArch64PAuth::checkAuthenticatedRegister(
    MachineBasicBlock::iterator MBBI, AuthCheckMethod Method,
    Register AuthenticatedReg, Register TmpReg, bool UseIKey, unsigned BrkImm) {

  MachineBasicBlock &MBB = *MBBI->getParent();
  MachineFunction &MF = *MBB.getParent();
  const AArch64Subtarget &Subtarget = MF.getSubtarget<AArch64Subtarget>();
  const AArch64InstrInfo *TII = Subtarget.getInstrInfo();
  DebugLoc DL = MBBI->getDebugLoc();

  // First, handle the methods not requiring creating extra MBBs.
  switch (Method) {
  default:
    break;
  case AuthCheckMethod::None:
    return MBB;
  case AuthCheckMethod::DummyLoad:
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::LDRWui), getWRegFromXReg(TmpReg))
        .addReg(AArch64::LR)
        .addImm(0)
        .addMemOperand(createCheckMemOperand(MF, Subtarget));
    return MBB;
  }

  // Control flow has to be changed, so arrange new MBBs.

  // At now, at least an AUT* instruction is expected before MBBI
  assert(MBBI != MBB.begin() &&
         "Cannot insert the check at the very beginning of MBB");
  // The block to insert check into.
  MachineBasicBlock *CheckBlock = &MBB;
  // The remaining part of the original MBB that is executed on success.
  MachineBasicBlock *SuccessBlock = MBB.splitAt(*std::prev(MBBI));

  // The block that explicitly generates a break-point exception on failure.
  MachineBasicBlock *BreakBlock =
      MF.CreateMachineBasicBlock(MBB.getBasicBlock());
  MF.push_back(BreakBlock);
  MBB.splitSuccessor(SuccessBlock, BreakBlock);

  assert(CheckBlock->getFallThrough() == SuccessBlock);
  BuildMI(BreakBlock, DL, TII->get(AArch64::BRK)).addImm(BrkImm);

  switch (Method) {
  case AuthCheckMethod::None:
  case AuthCheckMethod::DummyLoad:
    llvm_unreachable("Should be handled above");
  case AuthCheckMethod::HighBitsNoTBI:
    BuildMI(CheckBlock, DL, TII->get(AArch64::EORXrs), TmpReg)
        .addReg(AuthenticatedReg)
        .addReg(AuthenticatedReg)
        .addImm(1);
    BuildMI(CheckBlock, DL, TII->get(AArch64::TBNZX))
        .addReg(TmpReg)
        .addImm(62)
        .addMBB(BreakBlock);
    return *SuccessBlock;
  case AuthCheckMethod::XPACHint:
    assert(AuthenticatedReg == AArch64::LR &&
           "XPACHint mode is only compatible with checking the LR register");
    assert(UseIKey && "XPACHint mode is only compatible with I-keys");
    BuildMI(CheckBlock, DL, TII->get(AArch64::ORRXrs), TmpReg)
        .addReg(AArch64::XZR)
        .addReg(AArch64::LR)
        .addImm(0);
    BuildMI(CheckBlock, DL, TII->get(AArch64::XPACLRI));
    BuildMI(CheckBlock, DL, TII->get(AArch64::SUBSXrs), AArch64::XZR)
        .addReg(TmpReg)
        .addReg(AArch64::LR)
        .addImm(0);
    BuildMI(CheckBlock, DL, TII->get(AArch64::Bcc))
        .addImm(AArch64CC::NE)
        .addMBB(BreakBlock);
    return *SuccessBlock;
  }
  llvm_unreachable("Unknown AuthCheckMethod enum");
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
    return 20;
  }
  llvm_unreachable("Unknown AuthCheckMethod enum");
}

bool AArch64PointerAuth::checkAuthenticatedLR(
    MachineBasicBlock::iterator TI) const {
  AuthCheckMethod Method = Subtarget->getAuthenticatedLRCheckMethod();

  if (Method == AuthCheckMethod::None)
    return false;

  // FIXME If FEAT_FPAC is implemented by the CPU, this check can be skipped.

  assert(!TI->getMF()->hasWinCFI() && "WinCFI is not yet supported");

  // The following code may create a signing oracle:
  //
  //   <authenticate LR>
  //   TCRETURN          ; the callee may sign and spill the LR in its prologue
  //
  // To avoid generating a signing oracle, check the authenticated value
  // before possibly re-signing it in the callee, as follows:
  //
  //   <authenticate LR>
  //   <check if LR contains a valid address>
  //   b.<cond> break_block
  // ret_block:
  //   TCRETURN
  // break_block:
  //   brk <BrkOperand>
  //
  // or just
  //
  //   <authenticate LR>
  //   ldr tmp, [lr]
  //   TCRETURN

  // TmpReg is chosen assuming X16 and X17 are dead after TI.
  assert(AArch64InstrInfo::isTailCallReturnInst(*TI) &&
         "Tail call is expected");
  Register TmpReg =
      TI->readsRegister(AArch64::X16, TRI) ? AArch64::X17 : AArch64::X16;
  assert(!TI->readsRegister(TmpReg, TRI) &&
         "More than a single register is used by TCRETURN");

  checkAuthenticatedRegister(TI, Method, AArch64::LR, TmpReg, /*UseIKey=*/true,
                             BrkOperand);

  return true;
}

bool AArch64PointerAuth::runOnMachineFunction(MachineFunction &MF) {
  const auto *MFnI = MF.getInfo<AArch64FunctionInfo>();
  if (!MFnI->shouldSignReturnAddress(true))
    return false;

  Subtarget = &MF.getSubtarget<AArch64Subtarget>();
  TII = Subtarget->getInstrInfo();
  TRI = Subtarget->getRegisterInfo();

  SmallVector<MachineBasicBlock::iterator> DeletedInstrs;
  SmallVector<MachineBasicBlock::iterator> TailCallInstrs;

  bool Modified = false;
  bool HasAuthenticationInstrs = false;

  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      auto It = MI.getIterator();
      switch (MI.getOpcode()) {
      default:
        if (AArch64InstrInfo::isTailCallReturnInst(MI))
          TailCallInstrs.push_back(It);
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
        HasAuthenticationInstrs = true;
        break;
      }
    }
  }

  // FIXME Do we need to emit any PAuth-related epilogue code at all
  //       when SCS is enabled?
  if (HasAuthenticationInstrs &&
      !MFnI->needsShadowCallStackPrologueEpilogue(MF)) {
    for (auto TailCall : TailCallInstrs)
      Modified |= checkAuthenticatedLR(TailCall);
  }

  for (auto MI : DeletedInstrs)
    MI->eraseFromParent();

  return Modified;
}
