//===-- AArch64PointerAuth.cpp -- Harden code using PAuth ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AArch64PointerAuth.h"

#include "AArch64.h"
#include "AArch64FrameLowering.h"
#include "AArch64InstrInfo.h"
#include "AArch64MachineFunctionInfo.h"
#include "AArch64Subtarget.h"
#include "llvm/CodeGen/CFIInstBuilder.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"

using namespace llvm;
using namespace llvm::AArch64PAuth;

#define AARCH64_POINTER_AUTH_NAME "AArch64 Pointer Authentication"

namespace {

class AArch64PointerAuthImpl {
public:
  bool run(MachineFunction &MF);

private:
  const AArch64Subtarget *Subtarget = nullptr;
  const AArch64InstrInfo *TII = nullptr;

  void signLR(MachineFunction &MF, MachineBasicBlock::iterator MBBI) const;

  void authenticateLR(MachineFunction &MF,
                      MachineBasicBlock::iterator MBBI) const;
};

class AArch64PointerAuthLegacy : public MachineFunctionPass {
public:
  static char ID;

  AArch64PointerAuthLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return AARCH64_POINTER_AUTH_NAME; }
};

} // end anonymous namespace

INITIALIZE_PASS(AArch64PointerAuthLegacy, "aarch64-ptrauth",
                AARCH64_POINTER_AUTH_NAME, false, false)

FunctionPass *llvm::createAArch64PointerAuthPass() {
  return new AArch64PointerAuthLegacy();
}

char AArch64PointerAuthLegacy::ID = 0;

static void emitEpiloguePACSymOffsetIntoReg(const TargetInstrInfo &TII,
                                            MachineBasicBlock &MBB,
                                            MachineBasicBlock::iterator I,
                                            DebugLoc DL, MCSymbol *PACSym,
                                            Register Reg) {
  BuildMI(MBB, I, DL, TII.get(AArch64::ADRP), Reg)
      .addSym(PACSym, AArch64II::MO_PAGE)
      .setMIFlag(MachineInstr::FrameDestroy);
  BuildMI(MBB, I, DL, TII.get(AArch64::ADDXri), Reg)
      .addReg(Reg)
      .addSym(PACSym, AArch64II::MO_PAGEOFF | AArch64II::MO_NC)
      .addImm(0)
      .setMIFlag(MachineInstr::FrameDestroy);
}

// Wrap a given PAC instruction in CFI that describes it.
// Depending on the type of CFI required, we may need to emit the directive
// either before or after the instruction, so that unwinders can correctly
// interpret the location of the signing instruction.
template <typename BuildPACMIFn>
static void decoratePACWithCFI(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator MBBI, bool EmitCFI,
                               BuildPACMIFn BuildPACMI) {
  if (!EmitCFI) {
    BuildPACMI();
    return;
  }

  auto &MF = *MBB.getParent();
  auto &MFnI = *MF.getInfo<AArch64FunctionInfo>();

  CFIInstBuilder CFIBuilder(MBB, MBBI, MachineInstr::FrameSetup);
  if (MFnI.branchProtectionPAuthLR()) {
    CFIBuilder.buildNegateRAStateWithPC();
    BuildPACMI();
  } else {
    BuildPACMI();
    if (!MF.getTarget().getTargetTriple().isOSBinFormatMachO()) {
      CFIBuilder.buildNegateRAState();
    }
  }
}

static void emitAUTCFI(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                       bool EmitCFI) {
  if (!EmitCFI)
    return;

  auto &MF = *MBB.getParent();
  auto &MFnI = *MF.getInfo<AArch64FunctionInfo>();
  CFIInstBuilder CFIBuilder(MBB, MBBI, MachineInstr::FrameDestroy);
  const Triple &TT = MF.getTarget().getTargetTriple();

  if (MFnI.branchProtectionPAuthLR()) {
    // DW_CFA_AARCH64_negate_ra_state_with_pc is semantically broken for
    // functions where shrinkwrapping places signing/authenticating pairs on
    // distinct CFG paths.
    //
    // DWARF CFI is evaluated linearly over the byte stream, not along control
    // flow edges. The toggle semantics of this directive therefore cannot
    // faithfully represent the signed/unsigned RA state for all possible CFG
    // paths. The added complexity versus DW_CFA_AARCH64_negate_ra_state is
    // that an unwinder must also reconstruct the PC of the PACI[AB]SPPC in
    // order to verify the signed LR, and that address is derived from the
    // location of this directive in the linear CFI stream.
    //
    // The correct fix is to use DW_CFA_AARCH64_set_ra_state_with_pc, which
    // sets the RA state and signing address absolutely rather than toggling
    // them. An unwinder that supports this directive can reconstruct the
    // correct state on any CFG path, regardless of how many
    // signing/authenticating pairs exist in the function. However, not all
    // unwinders support this directive, so we cannot rely on it exclusively.
    //
    // For unwinders that only support DW_CFA_AARCH64_negate_ra_state_with_pc,
    // libunwind exploits a loophole: it records the address at the
    // DW_CFA_AARCH64_negate_ra_state_with_pc site to authenticate the LR, but
    // does not care that the CFI state remains "signed with pc" after
    // authentication has occurred. This means we can safely omit the
    // FrameDestroy emission of this directive, treating it solely as a marker
    // for the signing site, as long as each function has at most one such
    // signing location. That invariant holds today because shrinkwrapping
    // does not yet hoist or sink PAuth_LR frame code across CFG join/split
    // points; once it does, we must avoid those transformations on platforms
    // that have this limitation.
    //
    // https://github.com/ARM-software/abi-aa/issues/327
    // https://github.com/ARM-software/abi-aa/pull/346
  } else if (!TT.isOSBinFormatMachO()) {
    CFIBuilder.buildNegateRAState();
  }
}

void AArch64PointerAuthImpl::signLR(MachineFunction &MF,
                                    MachineBasicBlock::iterator MBBI) const {
  auto &MFnI = *MF.getInfo<AArch64FunctionInfo>();
  bool UseBKey = MFnI.shouldSignWithBKey();
  bool EmitCFI = MFnI.needsDwarfUnwindInfo(MF);
  bool NeedsWinCFI = MF.hasWinCFI();

  MachineBasicBlock &MBB = *MBBI->getParent();

  // Debug location must be unknown, see AArch64FrameLowering::emitPrologue.
  DebugLoc DL;

  if (UseBKey && !MF.getTarget().getTargetTriple().isOSBinFormatMachO()) {
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
    decoratePACWithCFI(MBB, MBBI, EmitCFI, [&]() {
      BuildMI(MBB, MBBI, DL,
              TII->get(UseBKey ? AArch64::PACIBSPPC : AArch64::PACIASPPC))
          .setMIFlag(MachineInstr::FrameSetup)
          ->setPreInstrSymbol(MF, MFnI.getSigningInstrLabel());
    });
  } else {
    if (MFnI.branchProtectionPAuthLR()) {
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::PACM))
          .setMIFlag(MachineInstr::FrameSetup);
    }
    decoratePACWithCFI(MBB, MBBI, EmitCFI, [&]() {
      BuildMI(MBB, MBBI, DL,
              TII->get(UseBKey ? AArch64::PACIBSP : AArch64::PACIASP))
          .setMIFlag(MachineInstr::FrameSetup)
          ->setPreInstrSymbol(MF, MFnI.getSigningInstrLabel());
    });
  }

  if (!EmitCFI && NeedsWinCFI) {
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::SEH_PACSignLR))
        .setMIFlag(MachineInstr::FrameSetup);
  }
}

void AArch64PointerAuthImpl::authenticateLR(
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
      BuildMI(MBB, TI, DL,
              TII->get(UseBKey ? AArch64::RETABSPPCi : AArch64::RETAASPPCi))
          .addSym(PACSym)
          .copyImplicitOps(*MBBI)
          .setMIFlag(MachineInstr::FrameDestroy);
    } else {
      if (MFnI->branchProtectionPAuthLR()) {
        emitEpiloguePACSymOffsetIntoReg(*TII, MBB, MBBI, DL, PACSym,
                                        AArch64::X16);
        BuildMI(MBB, MBBI, DL, TII->get(AArch64::PACM))
            .setMIFlag(MachineInstr::FrameDestroy);
      }
      BuildMI(MBB, TI, DL, TII->get(UseBKey ? AArch64::RETAB : AArch64::RETAA))
          .copyImplicitOps(*MBBI)
          .setMIFlag(MachineInstr::FrameDestroy);
    }
    MBB.erase(TI);
    return;
  }

  auto &AFL = *static_cast<const AArch64FrameLowering *>(
      MF.getSubtarget().getFrameLowering());
  int64_t ArgumentStackToRestore = AFL.getArgumentStackToRestore(MF, MBB);

  // When ArgumentStackToRestore < 0, the tail callee pops more argument space
  // than this function received, so after the frame teardown SP is below the
  // entry SP used as the signing modifier. Reconstruct entry SP in x16 and
  // authenticate using AUTI[AB]1716 (x17=LR, x16=entry_SP).
  if (ArgumentStackToRestore < 0) {
    emitFrameOffset(MBB, MBBI, DL, AArch64::X16, AArch64::SP,
                    StackOffset::getFixed(-ArgumentStackToRestore), TII,
                    MachineInstr::FrameDestroy);

    BuildMI(MBB, MBBI, DL, TII->get(AArch64::ORRXrs), AArch64::X17)
        .addReg(AArch64::XZR)
        .addReg(AArch64::LR)
        .addImm(0)
        .setMIFlag(MachineInstr::FrameDestroy);

    if (MFnI->branchProtectionPAuthLR() && Subtarget->hasPAuthLR()) {
      assert(PACSym && "No PAC instruction to refer to");
      emitEpiloguePACSymOffsetIntoReg(*TII, MBB, MBBI, DL, PACSym,
                                      AArch64::X15);

      emitAUTCFI(MBB, MBBI, EmitAsyncCFI);
      unsigned AutOpc = UseBKey ? AArch64::AUTIB171615 : AArch64::AUTIA171615;
      BuildMI(MBB, MBBI, DL, TII->get(AutOpc))
          .setMIFlag(MachineInstr::FrameDestroy);
    } else if (MFnI->branchProtectionPAuthLR()) {
      assert(PACSym && "No PAC instruction to refer to");
      emitEpiloguePACSymOffsetIntoReg(*TII, MBB, MBBI, DL, PACSym,
                                      AArch64::X15);

      // The PACM hint-space instruction modifies the following AUTI[AB]1716
      // to optionally take x15 as an extra operand depending on the
      // presence of +pauth-lr at runtime. On machines without +pauth-lr, it
      // behaves as a nop, and the address of the PACI[AB]SP in x15 is
      // ignored.
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::PACM))
          .setMIFlag(MachineInstr::FrameDestroy);

      emitAUTCFI(MBB, MBBI, EmitAsyncCFI);
      unsigned AutOpc = UseBKey ? AArch64::AUTIB1716 : AArch64::AUTIA1716;
      BuildMI(MBB, MBBI, DL, TII->get(AutOpc))
          .setMIFlag(MachineInstr::FrameDestroy);
    } else {
      unsigned AutOpc = UseBKey ? AArch64::AUTIB1716 : AArch64::AUTIA1716;
      BuildMI(MBB, MBBI, DL, TII->get(AutOpc))
          .setMIFlag(MachineInstr::FrameDestroy);
      emitAUTCFI(MBB, MBBI, EmitAsyncCFI);
    }

    BuildMI(MBB, MBBI, DL, TII->get(AArch64::ORRXrs), AArch64::LR)
        .addReg(AArch64::XZR)
        .addReg(AArch64::X17)
        .addImm(0)
        .setMIFlag(MachineInstr::FrameDestroy);
    return;
  }

  // When ArgumentStackToRestore > 0, this function received more argument
  // space than the tail callee pops. The epilogue contains an SP adjustment
  // (e.g. "add sp, sp, #N") to discard the leftover argument space. We must
  // authenticate *before* that adjustment so that AUTI[AB]SP sees the entry
  // SP discriminator. Move any such SP-adjusting instructions to after the
  // authentication instruction.
  //
  // We cannot simply bump SP first and then use AUTI[AB]SP with the bumped
  // value, because the live arguments would fall below SP and potentially
  // outside the red-zone.
  SmallVector<MachineInstr *, 2> SPMods;
  if (ArgumentStackToRestore > 0) {
    for (auto I = MBBI; I->getFlag(MachineInstr::FrameDestroy); --I) {
      if ((I->getOpcode() == AArch64::ADDXri ||
           I->getOpcode() == AArch64::SUBXri) &&
          I->getOperand(0).getReg() == AArch64::SP &&
          I->getOperand(1).getReg() == AArch64::SP)
        SPMods.push_back(&*I);
    }
  }
  for (auto *MI : SPMods)
    MI->removeFromParent();

  if (MFnI->branchProtectionPAuthLR() && Subtarget->hasPAuthLR()) {
    assert(PACSym && "No PAC instruction to refer to");
    emitAUTCFI(MBB, MBBI, EmitAsyncCFI);
    BuildMI(MBB, MBBI, DL,
            TII->get(UseBKey ? AArch64::AUTIBSPPCi : AArch64::AUTIASPPCi))
        .addSym(PACSym)
        .setMIFlag(MachineInstr::FrameDestroy);
  } else {
    if (MFnI->branchProtectionPAuthLR()) {
      emitEpiloguePACSymOffsetIntoReg(*TII, MBB, MBBI, DL, PACSym,
                                      AArch64::X16);

      BuildMI(MBB, MBBI, DL, TII->get(AArch64::PACM))
          .setMIFlag(MachineInstr::FrameDestroy);
      emitAUTCFI(MBB, MBBI, EmitAsyncCFI);
    }
    BuildMI(MBB, MBBI, DL,
            TII->get(UseBKey ? AArch64::AUTIBSP : AArch64::AUTIASP))
        .setMIFlag(MachineInstr::FrameDestroy);
    if (!MFnI->branchProtectionPAuthLR())
      emitAUTCFI(MBB, MBBI, EmitAsyncCFI);
  }

  if (NeedsWinCFI) {
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::SEH_PACSignLR))
        .setMIFlag(MachineInstr::FrameDestroy);
  }

  for (auto *MI : SPMods)
    MBB.insert(MBBI, MI);
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

bool AArch64PointerAuthImpl::run(MachineFunction &MF) {
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

  return Modified;
}

bool AArch64PointerAuthLegacy::runOnMachineFunction(MachineFunction &MF) {
  return AArch64PointerAuthImpl().run(MF);
}

PreservedAnalyses
AArch64PointerAuthPass::run(MachineFunction &MF,
                            MachineFunctionAnalysisManager &MFAM) {
  const bool Changed = AArch64PointerAuthImpl().run(MF);
  if (!Changed)
    return PreservedAnalyses::all();
  PreservedAnalyses PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
