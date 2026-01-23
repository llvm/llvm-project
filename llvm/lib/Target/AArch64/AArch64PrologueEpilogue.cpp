//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AArch64PrologueEpilogue.h"
#include "AArch64FrameLowering.h"
#include "AArch64MachineFunctionInfo.h"
#include "AArch64Subtarget.h"
#include "MCTargetDesc/AArch64AddressingModes.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/CodeGen/CFIInstBuilder.h"
#include "llvm/MC/MCContext.h"

#define DEBUG_TYPE "frame-info"

STATISTIC(NumRedZoneFunctions, "Number of functions using red zone");

namespace llvm {

static bool matchLibcall(const TargetLowering &TLI, const MachineOperand &MO,
                         RTLIB::Libcall LC) {
  return MO.isSymbol() &&
         StringRef(TLI.getLibcallName(LC)) == MO.getSymbolName();
}

bool AArch64PrologueEpilogueCommon::requiresGetVGCall() const {
  return AFI->hasStreamingModeChanges() &&
         !MF.getSubtarget<AArch64Subtarget>().hasSVE();
}

bool AArch64PrologueEpilogueCommon::isVGInstruction(
    MachineBasicBlock::iterator MBBI, const TargetLowering &TLI) const {
  unsigned Opc = MBBI->getOpcode();
  if (Opc == AArch64::CNTD_XPiI)
    return true;

  if (!requiresGetVGCall())
    return false;

  if (Opc == AArch64::BL)
    return matchLibcall(TLI, MBBI->getOperand(0), RTLIB::SMEABI_GET_CURRENT_VG);

  return Opc == TargetOpcode::COPY;
}

// Convenience function to determine whether I is part of the ZPR callee saves.
static bool isPartOfZPRCalleeSaves(MachineBasicBlock::iterator I) {
  switch (I->getOpcode()) {
  default:
    return false;
  case AArch64::LD1B_2Z_IMM:
  case AArch64::ST1B_2Z_IMM:
  case AArch64::STR_ZXI:
  case AArch64::LDR_ZXI:
  case AArch64::PTRUE_C_B:
    return I->getFlag(MachineInstr::FrameSetup) ||
           I->getFlag(MachineInstr::FrameDestroy);
  case AArch64::SEH_SaveZReg:
    return true;
  }
}

// Convenience function to determine whether I is part of the PPR callee saves.
static bool isPartOfPPRCalleeSaves(MachineBasicBlock::iterator I) {
  switch (I->getOpcode()) {
  default:
    return false;
  case AArch64::STR_PXI:
  case AArch64::LDR_PXI:
    return I->getFlag(MachineInstr::FrameSetup) ||
           I->getFlag(MachineInstr::FrameDestroy);
  case AArch64::SEH_SavePReg:
    return true;
  }
}

// Convenience function to determine whether I is part of the SVE callee saves.
static bool isPartOfSVECalleeSaves(MachineBasicBlock::iterator I) {
  return isPartOfZPRCalleeSaves(I) || isPartOfPPRCalleeSaves(I);
}

AArch64PrologueEpilogueCommon::AArch64PrologueEpilogueCommon(
    MachineFunction &MF, MachineBasicBlock &MBB,
    const AArch64FrameLowering &AFL)
    : MF(MF), MBB(MBB), MFI(MF.getFrameInfo()),
      Subtarget(MF.getSubtarget<AArch64Subtarget>()), AFL(AFL),
      RegInfo(*Subtarget.getRegisterInfo()) {
  TII = Subtarget.getInstrInfo();
  AFI = MF.getInfo<AArch64FunctionInfo>();

  HasFP = AFL.hasFP(MF);
  NeedsWinCFI = AFL.needsWinCFI(MF);

  if (AFL.hasSVECalleeSavesAboveFrameRecord(MF)) {
    if (AFI->hasStackHazardSlotIndex())
      reportFatalUsageError("SME hazard padding is not supported on Windows");
    SVELayout = SVEStackLayout::CalleeSavesAboveFrameRecord;
  } else if (AFI->hasSplitSVEObjects()) {
    SVELayout = SVEStackLayout::Split;
  }
}

MachineBasicBlock::iterator
AArch64PrologueEpilogueCommon::convertCalleeSaveRestoreToSPPrePostIncDec(
    MachineBasicBlock::iterator MBBI, const DebugLoc &DL, int CSStackSizeInc,
    bool EmitCFI, MachineInstr::MIFlag FrameFlag, int CFAOffset) const {
  unsigned NewOpc;

  // If the function contains streaming mode changes, we expect instructions
  // to calculate the value of VG before spilling. Move past these instructions
  // if necessary.
  if (AFL.requiresSaveVG(MF)) {
    auto &TLI = *Subtarget.getTargetLowering();
    while (isVGInstruction(MBBI, TLI))
      ++MBBI;
  }

  switch (MBBI->getOpcode()) {
  default:
    llvm_unreachable("Unexpected callee-save save/restore opcode!");
  case AArch64::STPXi:
    NewOpc = AArch64::STPXpre;
    break;
  case AArch64::STPDi:
    NewOpc = AArch64::STPDpre;
    break;
  case AArch64::STPQi:
    NewOpc = AArch64::STPQpre;
    break;
  case AArch64::STRXui:
    NewOpc = AArch64::STRXpre;
    break;
  case AArch64::STRDui:
    NewOpc = AArch64::STRDpre;
    break;
  case AArch64::STRQui:
    NewOpc = AArch64::STRQpre;
    break;
  case AArch64::LDPXi:
    NewOpc = AArch64::LDPXpost;
    break;
  case AArch64::LDPDi:
    NewOpc = AArch64::LDPDpost;
    break;
  case AArch64::LDPQi:
    NewOpc = AArch64::LDPQpost;
    break;
  case AArch64::LDRXui:
    NewOpc = AArch64::LDRXpost;
    break;
  case AArch64::LDRDui:
    NewOpc = AArch64::LDRDpost;
    break;
  case AArch64::LDRQui:
    NewOpc = AArch64::LDRQpost;
    break;
  }
  TypeSize Scale = TypeSize::getFixed(1), Width = TypeSize::getFixed(0);
  int64_t MinOffset, MaxOffset;
  bool Success = TII->getMemOpInfo(NewOpc, Scale, Width, MinOffset, MaxOffset);
  (void)Success;
  assert(Success && "unknown load/store opcode");

  // If the first store isn't right where we want SP then we can't fold the
  // update in so create a normal arithmetic instruction instead.
  //
  // On Windows, some register pairs involving LR can't be folded because
  // there isn't a corresponding unwind opcode.
  if (MBBI->getOperand(MBBI->getNumOperands() - 1).getImm() != 0 ||
      CSStackSizeInc < MinOffset * (int64_t)Scale.getFixedValue() ||
      CSStackSizeInc > MaxOffset * (int64_t)Scale.getFixedValue() ||
      (NeedsWinCFI &&
       (NewOpc == AArch64::LDPXpost || NewOpc == AArch64::STPXpre) &&
       RegInfo.getEncodingValue(MBBI->getOperand(0).getReg()) + 1 !=
           RegInfo.getEncodingValue(MBBI->getOperand(1).getReg()))) {
    // If we are destroying the frame, make sure we add the increment after the
    // last frame operation.
    if (FrameFlag == MachineInstr::FrameDestroy) {
      ++MBBI;
      // Also skip the SEH instruction, if needed
      if (NeedsWinCFI && AArch64InstrInfo::isSEHInstruction(*MBBI))
        ++MBBI;
    }
    emitFrameOffset(MBB, MBBI, DL, AArch64::SP, AArch64::SP,
                    StackOffset::getFixed(CSStackSizeInc), TII, FrameFlag,
                    false, NeedsWinCFI, &HasWinCFI, EmitCFI,
                    StackOffset::getFixed(CFAOffset));

    return std::prev(MBBI);
  }

  // Get rid of the SEH code associated with the old instruction.
  if (NeedsWinCFI) {
    auto SEH = std::next(MBBI);
    if (AArch64InstrInfo::isSEHInstruction(*SEH))
      SEH->eraseFromParent();
  }

  MachineInstrBuilder MIB = BuildMI(MBB, MBBI, DL, TII->get(NewOpc));
  MIB.addReg(AArch64::SP, RegState::Define);

  // Copy all operands other than the immediate offset.
  unsigned OpndIdx = 0;
  for (unsigned OpndEnd = MBBI->getNumOperands() - 1; OpndIdx < OpndEnd;
       ++OpndIdx)
    MIB.add(MBBI->getOperand(OpndIdx));

  assert(MBBI->getOperand(OpndIdx).getImm() == 0 &&
         "Unexpected immediate offset in first/last callee-save save/restore "
         "instruction!");
  assert(MBBI->getOperand(OpndIdx - 1).getReg() == AArch64::SP &&
         "Unexpected base register in callee-save save/restore instruction!");
  assert(CSStackSizeInc % Scale == 0);
  MIB.addImm(CSStackSizeInc / (int)Scale);

  MIB.setMIFlags(MBBI->getFlags());
  MIB.setMemRefs(MBBI->memoperands());

  // Generate a new SEH code that corresponds to the new instruction.
  if (NeedsWinCFI) {
    HasWinCFI = true;
    AFL.insertSEH(*MIB, *TII, FrameFlag);
  }

  if (EmitCFI)
    CFIInstBuilder(MBB, MBBI, FrameFlag)
        .buildDefCFAOffset(CFAOffset - CSStackSizeInc);

  return std::prev(MBB.erase(MBBI));
}

// Fix up the SEH opcode associated with the save/restore instruction.
static void fixupSEHOpcode(MachineBasicBlock::iterator MBBI,
                           unsigned LocalStackSize) {
  MachineOperand *ImmOpnd = nullptr;
  unsigned ImmIdx = MBBI->getNumOperands() - 1;
  switch (MBBI->getOpcode()) {
  default:
    llvm_unreachable("Fix the offset in the SEH instruction");
  case AArch64::SEH_SaveFPLR:
  case AArch64::SEH_SaveRegP:
  case AArch64::SEH_SaveReg:
  case AArch64::SEH_SaveFRegP:
  case AArch64::SEH_SaveFReg:
  case AArch64::SEH_SaveAnyRegI:
  case AArch64::SEH_SaveAnyRegIP:
  case AArch64::SEH_SaveAnyRegQP:
  case AArch64::SEH_SaveAnyRegQPX:
    ImmOpnd = &MBBI->getOperand(ImmIdx);
    break;
  }
  if (ImmOpnd)
    ImmOpnd->setImm(ImmOpnd->getImm() + LocalStackSize);
}

void AArch64PrologueEpilogueCommon::fixupCalleeSaveRestoreStackOffset(
    MachineInstr &MI, uint64_t LocalStackSize) const {
  if (AArch64InstrInfo::isSEHInstruction(MI))
    return;

  unsigned Opc = MI.getOpcode();
  unsigned Scale;
  switch (Opc) {
  case AArch64::STPXi:
  case AArch64::STRXui:
  case AArch64::STPDi:
  case AArch64::STRDui:
  case AArch64::LDPXi:
  case AArch64::LDRXui:
  case AArch64::LDPDi:
  case AArch64::LDRDui:
    Scale = 8;
    break;
  case AArch64::STPQi:
  case AArch64::STRQui:
  case AArch64::LDPQi:
  case AArch64::LDRQui:
    Scale = 16;
    break;
  default:
    llvm_unreachable("Unexpected callee-save save/restore opcode!");
  }

  unsigned OffsetIdx = MI.getNumExplicitOperands() - 1;
  assert(MI.getOperand(OffsetIdx - 1).getReg() == AArch64::SP &&
         "Unexpected base register in callee-save save/restore instruction!");
  // Last operand is immediate offset that needs fixing.
  MachineOperand &OffsetOpnd = MI.getOperand(OffsetIdx);
  // All generated opcodes have scaled offsets.
  assert(LocalStackSize % Scale == 0);
  OffsetOpnd.setImm(OffsetOpnd.getImm() + LocalStackSize / Scale);

  if (NeedsWinCFI) {
    HasWinCFI = true;
    auto MBBI = std::next(MachineBasicBlock::iterator(MI));
    assert(MBBI != MI.getParent()->end() && "Expecting a valid instruction");
    assert(AArch64InstrInfo::isSEHInstruction(*MBBI) &&
           "Expecting a SEH instruction");
    fixupSEHOpcode(MBBI, LocalStackSize);
  }
}

bool AArch64PrologueEpilogueCommon::shouldCombineCSRLocalStackBump(
    uint64_t StackBumpBytes) const {
  if (AFL.homogeneousPrologEpilog(MF))
    return false;

  if (AFI->getLocalStackSize() == 0)
    return false;

  // For WinCFI, if optimizing for size, prefer to not combine the stack bump
  // (to force a stp with predecrement) to match the packed unwind format,
  // provided that there actually are any callee saved registers to merge the
  // decrement with.
  //
  // Note that for certain paired saves, like "x19, lr", we can't actually
  // emit an predecrement stp, but packed unwind still expects a separate stack
  // adjustment.
  //
  // This is potentially marginally slower, but allows using the packed
  // unwind format for functions that both have a local area and callee saved
  // registers. Using the packed unwind format notably reduces the size of
  // the unwind info.
  if (AFL.needsWinCFI(MF) && AFI->getCalleeSavedStackSize() > 0 &&
      MF.getFunction().hasOptSize())
    return false;

  // 512 is the maximum immediate for stp/ldp that will be used for
  // callee-save save/restores
  if (StackBumpBytes >= 512 ||
      AFL.windowsRequiresStackProbe(MF, StackBumpBytes))
    return false;

  if (MFI.hasVarSizedObjects())
    return false;

  if (RegInfo.hasStackRealignment(MF))
    return false;

  // This isn't strictly necessary, but it simplifies things a bit since the
  // current RedZone handling code assumes the SP is adjusted by the
  // callee-save save/restore code.
  if (AFL.canUseRedZone(MF))
    return false;

  // When there is an SVE area on the stack, always allocate the
  // callee-saves and spills/locals separately.
  if (AFI->hasSVEStackSize())
    return false;

  return true;
}

SVEFrameSizes AArch64PrologueEpilogueCommon::getSVEStackFrameSizes() const {
  StackOffset PPRCalleeSavesSize =
      StackOffset::getScalable(AFI->getPPRCalleeSavedStackSize());
  StackOffset ZPRCalleeSavesSize =
      StackOffset::getScalable(AFI->getZPRCalleeSavedStackSize());
  StackOffset PPRLocalsSize = AFL.getPPRStackSize(MF) - PPRCalleeSavesSize;
  StackOffset ZPRLocalsSize = AFL.getZPRStackSize(MF) - ZPRCalleeSavesSize;
  if (SVELayout == SVEStackLayout::Split)
    return {{PPRCalleeSavesSize, PPRLocalsSize},
            {ZPRCalleeSavesSize, ZPRLocalsSize}};
  // For simplicity, attribute all locals to ZPRs when split SVE is disabled.
  return {{PPRCalleeSavesSize, StackOffset{}},
          {ZPRCalleeSavesSize, PPRLocalsSize + ZPRLocalsSize}};
}

SVEStackAllocations AArch64PrologueEpilogueCommon::getSVEStackAllocations(
    SVEFrameSizes const &SVE) {
  StackOffset AfterZPRs = SVE.ZPR.LocalsSize;
  StackOffset BeforePPRs = SVE.ZPR.CalleeSavesSize + SVE.PPR.CalleeSavesSize;
  StackOffset AfterPPRs = {};
  if (SVELayout == SVEStackLayout::Split) {
    BeforePPRs = SVE.PPR.CalleeSavesSize;
    // If there are no ZPR CSRs, place all local allocations after the ZPRs.
    if (SVE.ZPR.CalleeSavesSize)
      AfterPPRs += SVE.PPR.LocalsSize + SVE.ZPR.CalleeSavesSize;
    else
      AfterZPRs += SVE.PPR.LocalsSize; // Group allocation of locals.
  }
  return {BeforePPRs, AfterPPRs, AfterZPRs};
}

struct SVEPartitions {
  struct {
    MachineBasicBlock::iterator Begin, End;
  } PPR, ZPR;
};

static SVEPartitions partitionSVECS(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator MBBI,
                                    StackOffset PPRCalleeSavesSize,
                                    StackOffset ZPRCalleeSavesSize,
                                    bool IsEpilogue) {
  MachineBasicBlock::iterator PPRsI = MBBI;
  MachineBasicBlock::iterator End =
      IsEpilogue ? MBB.begin() : MBB.getFirstTerminator();
  auto AdjustI = [&](auto MBBI) { return IsEpilogue ? std::prev(MBBI) : MBBI; };
  // Process the SVE CS to find the starts/ends of the ZPR and PPR areas.
  if (PPRCalleeSavesSize) {
    PPRsI = AdjustI(PPRsI);
    assert(isPartOfPPRCalleeSaves(*PPRsI) && "Unexpected instruction");
    while (PPRsI != End && isPartOfPPRCalleeSaves(AdjustI(PPRsI)))
      IsEpilogue ? (--PPRsI) : (++PPRsI);
  }
  MachineBasicBlock::iterator ZPRsI = PPRsI;
  if (ZPRCalleeSavesSize) {
    ZPRsI = AdjustI(ZPRsI);
    assert(isPartOfZPRCalleeSaves(*ZPRsI) && "Unexpected instruction");
    while (ZPRsI != End && isPartOfZPRCalleeSaves(AdjustI(ZPRsI)))
      IsEpilogue ? (--ZPRsI) : (++ZPRsI);
  }
  if (IsEpilogue)
    return {{PPRsI, MBBI}, {ZPRsI, PPRsI}};
  return {{MBBI, PPRsI}, {PPRsI, ZPRsI}};
}

AArch64PrologueEmitter::AArch64PrologueEmitter(MachineFunction &MF,
                                               MachineBasicBlock &MBB,
                                               const AArch64FrameLowering &AFL)
    : AArch64PrologueEpilogueCommon(MF, MBB, AFL), F(MF.getFunction()) {
  EmitCFI = AFI->needsDwarfUnwindInfo(MF);
  EmitAsyncCFI = AFI->needsAsyncDwarfUnwindInfo(MF);
  IsFunclet = MBB.isEHFuncletEntry();
  HomPrologEpilog = AFL.homogeneousPrologEpilog(MF);

#ifndef NDEBUG
  collectBlockLiveins();
#endif
}

#ifndef NDEBUG
/// Collect live registers from the end of \p MI's parent up to (including) \p
/// MI in \p LiveRegs.
static void getLivePhysRegsUpTo(MachineInstr &MI, const TargetRegisterInfo &TRI,
                                LivePhysRegs &LiveRegs) {

  MachineBasicBlock &MBB = *MI.getParent();
  LiveRegs.addLiveOuts(MBB);
  for (const MachineInstr &MI :
       reverse(make_range(MI.getIterator(), MBB.instr_end())))
    LiveRegs.stepBackward(MI);
}

void AArch64PrologueEmitter::collectBlockLiveins() {
  // Collect live register from the end of MBB up to the start of the existing
  // frame setup instructions.
  PrologueEndI = MBB.begin();
  while (PrologueEndI != MBB.end() &&
         PrologueEndI->getFlag(MachineInstr::FrameSetup))
    ++PrologueEndI;

  if (PrologueEndI != MBB.end()) {
    getLivePhysRegsUpTo(*PrologueEndI, RegInfo, LiveRegs);
    // Ignore registers used for stack management for now.
    LiveRegs.removeReg(AArch64::SP);
    LiveRegs.removeReg(AArch64::X19);
    LiveRegs.removeReg(AArch64::FP);
    LiveRegs.removeReg(AArch64::LR);

    // X0 will be clobbered by a call to __arm_get_current_vg in the prologue.
    // This is necessary to spill VG if required where SVE is unavailable, but
    // X0 is preserved around this call.
    if (requiresGetVGCall())
      LiveRegs.removeReg(AArch64::X0);
  }
}

void AArch64PrologueEmitter::verifyPrologueClobbers() const {
  if (PrologueEndI == MBB.end())
    return;
  // Check if any of the newly instructions clobber any of the live registers.
  for (MachineInstr &MI :
       make_range(MBB.instr_begin(), PrologueEndI->getIterator())) {
    for (auto &Op : MI.operands())
      if (Op.isReg() && Op.isDef())
        assert(!LiveRegs.contains(Op.getReg()) &&
               "live register clobbered by inserted prologue instructions");
  }
}
#endif

void AArch64PrologueEmitter::determineLocalsStackSize(
    uint64_t StackSize, uint64_t PrologueSaveSize) {
  AFI->setLocalStackSize(StackSize - PrologueSaveSize);
  CombineSPBump = shouldCombineCSRLocalStackBump(StackSize);
}

// Return the maximum possible number of bytes for `Size` due to the
// architectural limit on the size of a SVE register.
static int64_t upperBound(StackOffset Size) {
  static const int64_t MAX_BYTES_PER_SCALABLE_BYTE = 16;
  return Size.getScalable() * MAX_BYTES_PER_SCALABLE_BYTE + Size.getFixed();
}

void AArch64PrologueEmitter::allocateStackSpace(
    MachineBasicBlock::iterator MBBI, int64_t RealignmentPadding,
    StackOffset AllocSize, bool EmitCFI, StackOffset InitialOffset,
    bool FollowupAllocs) {

  if (!AllocSize)
    return;

  DebugLoc DL;
  const int64_t MaxAlign = MFI.getMaxAlign().value();
  const uint64_t AndMask = ~(MaxAlign - 1);

  if (!Subtarget.getTargetLowering()->hasInlineStackProbe(MF)) {
    Register TargetReg = RealignmentPadding
                             ? AFL.findScratchNonCalleeSaveRegister(&MBB)
                             : AArch64::SP;
    // SUB Xd/SP, SP, AllocSize
    emitFrameOffset(MBB, MBBI, DL, TargetReg, AArch64::SP, -AllocSize, TII,
                    MachineInstr::FrameSetup, false, NeedsWinCFI, &HasWinCFI,
                    EmitCFI, InitialOffset);

    if (RealignmentPadding) {
      // AND SP, X9, 0b11111...0000
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::ANDXri), AArch64::SP)
          .addReg(TargetReg, RegState::Kill)
          .addImm(AArch64_AM::encodeLogicalImmediate(AndMask, 64))
          .setMIFlags(MachineInstr::FrameSetup);
      AFI->setStackRealigned(true);

      // No need for SEH instructions here; if we're realigning the stack,
      // we've set a frame pointer and already finished the SEH prologue.
      assert(!NeedsWinCFI);
    }
    return;
  }

  //
  // Stack probing allocation.
  //

  // Fixed length allocation. If we don't need to re-align the stack and don't
  // have SVE objects, we can use a more efficient sequence for stack probing.
  if (AllocSize.getScalable() == 0 && RealignmentPadding == 0) {
    Register ScratchReg = AFL.findScratchNonCalleeSaveRegister(&MBB);
    assert(ScratchReg != AArch64::NoRegister);
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::PROBED_STACKALLOC))
        .addDef(ScratchReg)
        .addImm(AllocSize.getFixed())
        .addImm(InitialOffset.getFixed())
        .addImm(InitialOffset.getScalable());
    // The fixed allocation may leave unprobed bytes at the top of the
    // stack. If we have subsequent allocation (e.g. if we have variable-sized
    // objects), we need to issue an extra probe, so these allocations start in
    // a known state.
    if (FollowupAllocs) {
      // LDR XZR, [SP]
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::LDRXui))
          .addDef(AArch64::XZR)
          .addReg(AArch64::SP)
          .addImm(0)
          .addMemOperand(MF.getMachineMemOperand(
              MachinePointerInfo::getUnknownStack(MF),
              MachineMemOperand::MOLoad | MachineMemOperand::MOVolatile, 8,
              Align(8)))
          .setMIFlags(MachineInstr::FrameSetup);
    }

    return;
  }

  // Variable length allocation.

  // If the (unknown) allocation size cannot exceed the probe size, decrement
  // the stack pointer right away.
  int64_t ProbeSize = AFI->getStackProbeSize();
  if (upperBound(AllocSize) + RealignmentPadding <= ProbeSize) {
    Register ScratchReg = RealignmentPadding
                              ? AFL.findScratchNonCalleeSaveRegister(&MBB)
                              : AArch64::SP;
    assert(ScratchReg != AArch64::NoRegister);
    // SUB Xd, SP, AllocSize
    emitFrameOffset(MBB, MBBI, DL, ScratchReg, AArch64::SP, -AllocSize, TII,
                    MachineInstr::FrameSetup, false, NeedsWinCFI, &HasWinCFI,
                    EmitCFI, InitialOffset);
    if (RealignmentPadding) {
      // AND SP, Xn, 0b11111...0000
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::ANDXri), AArch64::SP)
          .addReg(ScratchReg, RegState::Kill)
          .addImm(AArch64_AM::encodeLogicalImmediate(AndMask, 64))
          .setMIFlags(MachineInstr::FrameSetup);
      AFI->setStackRealigned(true);
    }
    if (FollowupAllocs || upperBound(AllocSize) + RealignmentPadding >
                              AArch64::StackProbeMaxUnprobedStack) {
      // LDR XZR, [SP]
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::LDRXui))
          .addDef(AArch64::XZR)
          .addReg(AArch64::SP)
          .addImm(0)
          .addMemOperand(MF.getMachineMemOperand(
              MachinePointerInfo::getUnknownStack(MF),
              MachineMemOperand::MOLoad | MachineMemOperand::MOVolatile, 8,
              Align(8)))
          .setMIFlags(MachineInstr::FrameSetup);
    }
    return;
  }

  // Emit a variable-length allocation probing loop.
  // TODO: As an optimisation, the loop can be "unrolled" into a few parts,
  // each of them guaranteed to adjust the stack by less than the probe size.
  Register TargetReg = AFL.findScratchNonCalleeSaveRegister(&MBB);
  assert(TargetReg != AArch64::NoRegister);
  // SUB Xd, SP, AllocSize
  emitFrameOffset(MBB, MBBI, DL, TargetReg, AArch64::SP, -AllocSize, TII,
                  MachineInstr::FrameSetup, false, NeedsWinCFI, &HasWinCFI,
                  EmitCFI, InitialOffset);
  if (RealignmentPadding) {
    // AND Xn, Xn, 0b11111...0000
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::ANDXri), TargetReg)
        .addReg(TargetReg, RegState::Kill)
        .addImm(AArch64_AM::encodeLogicalImmediate(AndMask, 64))
        .setMIFlags(MachineInstr::FrameSetup);
  }

  BuildMI(MBB, MBBI, DL, TII->get(AArch64::PROBED_STACKALLOC_VAR))
      .addReg(TargetReg);
  if (EmitCFI) {
    // Set the CFA register back to SP.
    CFIInstBuilder(MBB, MBBI, MachineInstr::FrameSetup)
        .buildDefCFARegister(AArch64::SP);
  }
  if (RealignmentPadding)
    AFI->setStackRealigned(true);
}

void AArch64PrologueEmitter::emitPrologue() {
  const MachineBasicBlock::iterator PrologueBeginI = MBB.begin();
  const MachineBasicBlock::iterator EndI = MBB.end();

  // At this point, we're going to decide whether or not the function uses a
  // redzone. In most cases, the function doesn't have a redzone so let's
  // assume that's false and set it to true in the case that there's a redzone.
  AFI->setHasRedZone(false);

  // Debug location must be unknown since the first debug location is used
  // to determine the end of the prologue.
  DebugLoc DL;

  // In some cases, particularly with CallingConv::SwiftTail, it is possible to
  // have a tail-call where the caller only needs to adjust the stack pointer in
  // the epilogue. In this case, we still need to emit a SEH prologue sequence.
  // See `seh-minimal-prologue-epilogue.ll` test cases.
  if (AFI->getArgumentStackToRestore())
    HasWinCFI |= NeedsWinCFI;

  if (AFI->shouldSignReturnAddress(MF)) {
    // If pac-ret+leaf is in effect, PAUTH_PROLOGUE pseudo instructions
    // are inserted by emitPacRetPlusLeafHardening().
    if (!AFL.shouldSignReturnAddressEverywhere(MF)) {
      BuildMI(MBB, PrologueBeginI, DL, TII->get(AArch64::PAUTH_PROLOGUE))
          .setMIFlag(MachineInstr::FrameSetup);
    }
    // AArch64PointerAuth pass will insert SEH_PACSignLR
    HasWinCFI |= NeedsWinCFI;
  }

  if (AFI->needsShadowCallStackPrologueEpilogue(MF)) {
    emitShadowCallStackPrologue(PrologueBeginI, DL);
    HasWinCFI |= NeedsWinCFI;
  }

  if (EmitCFI && AFI->isMTETagged())
    BuildMI(MBB, PrologueBeginI, DL, TII->get(AArch64::EMITMTETAGGED))
        .setMIFlag(MachineInstr::FrameSetup);

  // We signal the presence of a Swift extended frame to external tools by
  // storing FP with 0b0001 in bits 63:60. In normal userland operation a simple
  // ORR is sufficient, it is assumed a Swift kernel would initialize the TBI
  // bits so that is still true.
  if (HasFP && AFI->hasSwiftAsyncContext())
    emitSwiftAsyncContextFramePointer(PrologueBeginI, DL);

  // All calls are tail calls in GHC calling conv, and functions have no
  // prologue/epilogue.
  if (MF.getFunction().getCallingConv() == CallingConv::GHC)
    return;

  // Set tagged base pointer to the requested stack slot. Ideally it should
  // match SP value after prologue.
  if (std::optional<int> TBPI = AFI->getTaggedBasePointerIndex())
    AFI->setTaggedBasePointerOffset(-MFI.getObjectOffset(*TBPI));
  else
    AFI->setTaggedBasePointerOffset(MFI.getStackSize());

  // getStackSize() includes all the locals in its size calculation. We don't
  // include these locals when computing the stack size of a funclet, as they
  // are allocated in the parent's stack frame and accessed via the frame
  // pointer from the funclet.  We only save the callee saved registers in the
  // funclet, which are really the callee saved registers of the parent
  // function, including the funclet.
  int64_t NumBytes =
      IsFunclet ? AFL.getWinEHFuncletFrameSize(MF) : MFI.getStackSize();
  if (!AFI->hasStackFrame() && !AFL.windowsRequiresStackProbe(MF, NumBytes))
    return emitEmptyStackFramePrologue(NumBytes, PrologueBeginI, DL);

  bool IsWin64 = Subtarget.isCallingConvWin64(F.getCallingConv(), F.isVarArg());
  unsigned FixedObject = AFL.getFixedObjectSize(MF, AFI, IsWin64, IsFunclet);

  auto PrologueSaveSize = AFI->getCalleeSavedStackSize() + FixedObject;
  // All of the remaining stack allocations are for locals.
  determineLocalsStackSize(NumBytes, PrologueSaveSize);

  auto [PPR, ZPR] = getSVEStackFrameSizes();
  SVEStackAllocations SVEAllocs = getSVEStackAllocations({PPR, ZPR});

  MachineBasicBlock::iterator FirstGPRSaveI = PrologueBeginI;
  if (SVELayout == SVEStackLayout::CalleeSavesAboveFrameRecord) {
    assert(!SVEAllocs.AfterPPRs &&
           "unexpected SVE allocs after PPRs with CalleeSavesAboveFrameRecord");
    // If we're doing SVE saves first, we need to immediately allocate space
    // for fixed objects, then space for the SVE callee saves.
    //
    // Windows unwind requires that the scalable size is a multiple of 16;
    // that's handled when the callee-saved size is computed.
    auto SaveSize = SVEAllocs.BeforePPRs + StackOffset::getFixed(FixedObject);
    allocateStackSpace(PrologueBeginI, 0, SaveSize, false, StackOffset{},
                       /*FollowupAllocs=*/true);
    NumBytes -= FixedObject;

    // Now allocate space for the GPR callee saves.
    MachineBasicBlock::iterator MBBI = PrologueBeginI;
    while (MBBI != EndI && isPartOfSVECalleeSaves(MBBI))
      ++MBBI;
    FirstGPRSaveI = convertCalleeSaveRestoreToSPPrePostIncDec(
        MBBI, DL, -AFI->getCalleeSavedStackSize(), EmitAsyncCFI);
    NumBytes -= AFI->getCalleeSavedStackSize();
  } else if (CombineSPBump) {
    assert(!AFL.getSVEStackSize(MF) && "Cannot combine SP bump with SVE");
    emitFrameOffset(MBB, PrologueBeginI, DL, AArch64::SP, AArch64::SP,
                    StackOffset::getFixed(-NumBytes), TII,
                    MachineInstr::FrameSetup, false, NeedsWinCFI, &HasWinCFI,
                    EmitAsyncCFI);
    NumBytes = 0;
  } else if (HomPrologEpilog) {
    // Stack has been already adjusted.
    NumBytes -= PrologueSaveSize;
  } else if (PrologueSaveSize != 0) {
    FirstGPRSaveI = convertCalleeSaveRestoreToSPPrePostIncDec(
        PrologueBeginI, DL, -PrologueSaveSize, EmitAsyncCFI);
    NumBytes -= PrologueSaveSize;
  }
  assert(NumBytes >= 0 && "Negative stack allocation size!?");

  // Move past the saves of the callee-saved registers, fixing up the offsets
  // and pre-inc if we decided to combine the callee-save and local stack
  // pointer bump above.
  auto &TLI = *Subtarget.getTargetLowering();

  MachineBasicBlock::iterator AfterGPRSavesI = FirstGPRSaveI;
  while (AfterGPRSavesI != EndI &&
         AfterGPRSavesI->getFlag(MachineInstr::FrameSetup) &&
         !isPartOfSVECalleeSaves(AfterGPRSavesI)) {
    if (CombineSPBump &&
        // Only fix-up frame-setup load/store instructions.
        (!AFL.requiresSaveVG(MF) || !isVGInstruction(AfterGPRSavesI, TLI)))
      fixupCalleeSaveRestoreStackOffset(*AfterGPRSavesI,
                                        AFI->getLocalStackSize());
    ++AfterGPRSavesI;
  }

  // For funclets the FP belongs to the containing function. Only set up FP if
  // we actually need to.
  if (!IsFunclet && HasFP)
    emitFramePointerSetup(AfterGPRSavesI, DL, FixedObject);

  // Now emit the moves for whatever callee saved regs we have (including FP,
  // LR if those are saved). Frame instructions for SVE register are emitted
  // later, after the instruction which actually save SVE regs.
  if (EmitAsyncCFI)
    emitCalleeSavedGPRLocations(AfterGPRSavesI);

  // Alignment is required for the parent frame, not the funclet
  const bool NeedsRealignment =
      NumBytes && !IsFunclet && RegInfo.hasStackRealignment(MF);
  const int64_t RealignmentPadding =
      (NeedsRealignment && MFI.getMaxAlign() > Align(16))
          ? MFI.getMaxAlign().value() - 16
          : 0;

  if (AFL.windowsRequiresStackProbe(MF, NumBytes + RealignmentPadding))
    emitWindowsStackProbe(AfterGPRSavesI, DL, NumBytes, RealignmentPadding);

  StackOffset NonSVELocalsSize = StackOffset::getFixed(NumBytes);
  SVEAllocs.AfterZPRs += NonSVELocalsSize;

  StackOffset CFAOffset =
      StackOffset::getFixed(MFI.getStackSize()) - NonSVELocalsSize;
  MachineBasicBlock::iterator AfterSVESavesI = AfterGPRSavesI;
  // Allocate space for the callee saves and PPR locals (if any).
  if (SVELayout != SVEStackLayout::CalleeSavesAboveFrameRecord) {
    auto [PPRRange, ZPRRange] =
        partitionSVECS(MBB, AfterGPRSavesI, PPR.CalleeSavesSize,
                       ZPR.CalleeSavesSize, /*IsEpilogue=*/false);
    AfterSVESavesI = ZPRRange.End;
    if (EmitAsyncCFI)
      emitCalleeSavedSVELocations(AfterSVESavesI);

    allocateStackSpace(PPRRange.Begin, 0, SVEAllocs.BeforePPRs,
                       EmitAsyncCFI && !HasFP, CFAOffset,
                       MFI.hasVarSizedObjects() || SVEAllocs.AfterPPRs ||
                           SVEAllocs.AfterZPRs);
    CFAOffset += SVEAllocs.BeforePPRs;
    assert(PPRRange.End == ZPRRange.Begin &&
           "Expected ZPR callee saves after PPR locals");
    allocateStackSpace(PPRRange.End, 0, SVEAllocs.AfterPPRs,
                       EmitAsyncCFI && !HasFP, CFAOffset,
                       MFI.hasVarSizedObjects() || SVEAllocs.AfterZPRs);
    CFAOffset += SVEAllocs.AfterPPRs;
  } else {
    assert(SVELayout == SVEStackLayout::CalleeSavesAboveFrameRecord);
    // Note: With CalleeSavesAboveFrameRecord, the SVE CS (BeforePPRs) have
    // already been allocated. PPR locals (included in AfterPPRs) are not
    // supported (note: this is asserted above).
    CFAOffset += SVEAllocs.BeforePPRs;
  }

  // Allocate space for the rest of the frame including ZPR locals. Align the
  // stack as necessary.
  assert(!(AFL.canUseRedZone(MF) && NeedsRealignment) &&
         "Cannot use redzone with stack realignment");
  if (!AFL.canUseRedZone(MF)) {
    // FIXME: in the case of dynamic re-alignment, NumBytes doesn't have the
    // correct value here, as NumBytes also includes padding bytes, which
    // shouldn't be counted here.
    allocateStackSpace(AfterSVESavesI, RealignmentPadding, SVEAllocs.AfterZPRs,
                       EmitAsyncCFI && !HasFP, CFAOffset,
                       MFI.hasVarSizedObjects());
  }

  // If we need a base pointer, set it up here. It's whatever the value of the
  // stack pointer is at this point. Any variable size objects will be
  // allocated after this, so we can still use the base pointer to reference
  // locals.
  //
  // FIXME: Clarify FrameSetup flags here.
  // Note: Use emitFrameOffset() like above for FP if the FrameSetup flag is
  // needed.
  // For funclets the BP belongs to the containing function.
  if (!IsFunclet && RegInfo.hasBasePointer(MF)) {
    TII->copyPhysReg(MBB, AfterSVESavesI, DL, RegInfo.getBaseRegister(),
                     AArch64::SP, false);
    if (NeedsWinCFI) {
      HasWinCFI = true;
      BuildMI(MBB, AfterSVESavesI, DL, TII->get(AArch64::SEH_Nop))
          .setMIFlag(MachineInstr::FrameSetup);
    }
  }

  // The very last FrameSetup instruction indicates the end of prologue. Emit a
  // SEH opcode indicating the prologue end.
  if (NeedsWinCFI && HasWinCFI) {
    BuildMI(MBB, AfterSVESavesI, DL, TII->get(AArch64::SEH_PrologEnd))
        .setMIFlag(MachineInstr::FrameSetup);
  }

  // SEH funclets are passed the frame pointer in X1.  If the parent
  // function uses the base register, then the base register is used
  // directly, and is not retrieved from X1.
  if (IsFunclet && F.hasPersonalityFn()) {
    EHPersonality Per = classifyEHPersonality(F.getPersonalityFn());
    if (isAsynchronousEHPersonality(Per)) {
      BuildMI(MBB, AfterSVESavesI, DL, TII->get(TargetOpcode::COPY),
              AArch64::FP)
          .addReg(AArch64::X1)
          .setMIFlag(MachineInstr::FrameSetup);
      MBB.addLiveIn(AArch64::X1);
    }
  }

  if (EmitCFI && !EmitAsyncCFI) {
    if (HasFP) {
      emitDefineCFAWithFP(AfterSVESavesI, FixedObject);
    } else {
      StackOffset TotalSize =
          AFL.getSVEStackSize(MF) +
          StackOffset::getFixed((int64_t)MFI.getStackSize());
      CFIInstBuilder CFIBuilder(MBB, AfterSVESavesI, MachineInstr::FrameSetup);
      CFIBuilder.insertCFIInst(
          createDefCFA(RegInfo, /*FrameReg=*/AArch64::SP, /*Reg=*/AArch64::SP,
                       TotalSize, /*LastAdjustmentWasScalable=*/false));
    }
    emitCalleeSavedGPRLocations(AfterSVESavesI);
    emitCalleeSavedSVELocations(AfterSVESavesI);
  }
}

void AArch64PrologueEmitter::emitShadowCallStackPrologue(
    MachineBasicBlock::iterator MBBI, const DebugLoc &DL) const {
  // Shadow call stack prolog: str x30, [x18], #8
  BuildMI(MBB, MBBI, DL, TII->get(AArch64::STRXpost))
      .addReg(AArch64::X18, RegState::Define)
      .addReg(AArch64::LR)
      .addReg(AArch64::X18)
      .addImm(8)
      .setMIFlag(MachineInstr::FrameSetup);

  // This instruction also makes x18 live-in to the entry block.
  MBB.addLiveIn(AArch64::X18);

  if (NeedsWinCFI)
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::SEH_Nop))
        .setMIFlag(MachineInstr::FrameSetup);

  if (EmitCFI) {
    // Emit a CFI instruction that causes 8 to be subtracted from the value of
    // x18 when unwinding past this frame.
    static const char CFIInst[] = {
        dwarf::DW_CFA_val_expression,
        18, // register
        2,  // length
        static_cast<char>(unsigned(dwarf::DW_OP_breg18)),
        static_cast<char>(-8) & 0x7f, // addend (sleb128)
    };
    CFIInstBuilder(MBB, MBBI, MachineInstr::FrameSetup)
        .buildEscape(StringRef(CFIInst, sizeof(CFIInst)));
  }
}

void AArch64PrologueEmitter::emitSwiftAsyncContextFramePointer(
    MachineBasicBlock::iterator MBBI, const DebugLoc &DL) const {
  switch (MF.getTarget().Options.SwiftAsyncFramePointer) {
  case SwiftAsyncFramePointerMode::DeploymentBased:
    if (Subtarget.swiftAsyncContextIsDynamicallySet()) {
      // The special symbol below is absolute and has a *value* that can be
      // combined with the frame pointer to signal an extended frame.
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::LOADgot), AArch64::X16)
          .addExternalSymbol("swift_async_extendedFramePointerFlags",
                             AArch64II::MO_GOT);
      if (NeedsWinCFI) {
        BuildMI(MBB, MBBI, DL, TII->get(AArch64::SEH_Nop))
            .setMIFlags(MachineInstr::FrameSetup);
        HasWinCFI = true;
      }
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::ORRXrs), AArch64::FP)
          .addUse(AArch64::FP)
          .addUse(AArch64::X16)
          .addImm(Subtarget.isTargetILP32() ? 32 : 0);
      if (NeedsWinCFI) {
        BuildMI(MBB, MBBI, DL, TII->get(AArch64::SEH_Nop))
            .setMIFlags(MachineInstr::FrameSetup);
        HasWinCFI = true;
      }
      break;
    }
    [[fallthrough]];

  case SwiftAsyncFramePointerMode::Always:
    // ORR x29, x29, #0x1000_0000_0000_0000
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::ORRXri), AArch64::FP)
        .addUse(AArch64::FP)
        .addImm(0x1100)
        .setMIFlag(MachineInstr::FrameSetup);
    if (NeedsWinCFI) {
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::SEH_Nop))
          .setMIFlags(MachineInstr::FrameSetup);
      HasWinCFI = true;
    }
    break;

  case SwiftAsyncFramePointerMode::Never:
    break;
  }
}

void AArch64PrologueEmitter::emitEmptyStackFramePrologue(
    int64_t NumBytes, MachineBasicBlock::iterator MBBI,
    const DebugLoc &DL) const {
  assert(!HasFP && "unexpected function without stack frame but with FP");
  assert(!AFL.getSVEStackSize(MF) &&
         "unexpected function without stack frame but with SVE objects");
  // All of the stack allocation is for locals.
  AFI->setLocalStackSize(NumBytes);
  if (!NumBytes) {
    if (NeedsWinCFI && HasWinCFI) {
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::SEH_PrologEnd))
          .setMIFlag(MachineInstr::FrameSetup);
    }
    return;
  }
  // REDZONE: If the stack size is less than 128 bytes, we don't need
  // to actually allocate.
  if (AFL.canUseRedZone(MF)) {
    AFI->setHasRedZone(true);
    ++NumRedZoneFunctions;
  } else {
    emitFrameOffset(MBB, MBBI, DL, AArch64::SP, AArch64::SP,
                    StackOffset::getFixed(-NumBytes), TII,
                    MachineInstr::FrameSetup, false, NeedsWinCFI, &HasWinCFI);
    if (EmitCFI) {
      // Label used to tie together the PROLOG_LABEL and the MachineMoves.
      MCSymbol *FrameLabel = MF.getContext().createTempSymbol();
      // Encode the stack size of the leaf function.
      CFIInstBuilder(MBB, MBBI, MachineInstr::FrameSetup)
          .buildDefCFAOffset(NumBytes, FrameLabel);
    }
  }

  if (NeedsWinCFI) {
    HasWinCFI = true;
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::SEH_PrologEnd))
        .setMIFlag(MachineInstr::FrameSetup);
  }
}

void AArch64PrologueEmitter::emitFramePointerSetup(
    MachineBasicBlock::iterator MBBI, const DebugLoc &DL,
    unsigned FixedObject) {
  int64_t FPOffset = AFI->getCalleeSaveBaseToFrameRecordOffset();
  if (CombineSPBump)
    FPOffset += AFI->getLocalStackSize();

  if (AFI->hasSwiftAsyncContext()) {
    // Before we update the live FP we have to ensure there's a valid (or
    // null) asynchronous context in its slot just before FP in the frame
    // record, so store it now.
    const auto &Attrs = MF.getFunction().getAttributes();
    bool HaveInitialContext = Attrs.hasAttrSomewhere(Attribute::SwiftAsync);
    if (HaveInitialContext)
      MBB.addLiveIn(AArch64::X22);
    Register Reg = HaveInitialContext ? AArch64::X22 : AArch64::XZR;
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::StoreSwiftAsyncContext))
        .addUse(Reg)
        .addUse(AArch64::SP)
        .addImm(FPOffset - 8)
        .setMIFlags(MachineInstr::FrameSetup);
    if (NeedsWinCFI) {
      // WinCFI and arm64e, where StoreSwiftAsyncContext is expanded
      // to multiple instructions, should be mutually-exclusive.
      assert(Subtarget.getTargetTriple().getArchName() != "arm64e");
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::SEH_Nop))
          .setMIFlags(MachineInstr::FrameSetup);
      HasWinCFI = true;
    }
  }

  if (HomPrologEpilog) {
    auto Prolog = MBBI;
    --Prolog;
    assert(Prolog->getOpcode() == AArch64::HOM_Prolog);
    Prolog->addOperand(MachineOperand::CreateImm(FPOffset));
  } else {
    // Issue    sub fp, sp, FPOffset or
    //          mov fp,sp          when FPOffset is zero.
    // Note: All stores of callee-saved registers are marked as "FrameSetup".
    // This code marks the instruction(s) that set the FP also.
    emitFrameOffset(MBB, MBBI, DL, AArch64::FP, AArch64::SP,
                    StackOffset::getFixed(FPOffset), TII,
                    MachineInstr::FrameSetup, false, NeedsWinCFI, &HasWinCFI);
    if (NeedsWinCFI && HasWinCFI) {
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::SEH_PrologEnd))
          .setMIFlag(MachineInstr::FrameSetup);
      // After setting up the FP, the rest of the prolog doesn't need to be
      // included in the SEH unwind info.
      NeedsWinCFI = false;
    }
  }
  if (EmitAsyncCFI)
    emitDefineCFAWithFP(MBBI, FixedObject);
}

// Define the current CFA rule to use the provided FP.
void AArch64PrologueEmitter::emitDefineCFAWithFP(
    MachineBasicBlock::iterator MBBI, unsigned FixedObject) const {
  const int OffsetToFirstCalleeSaveFromFP =
      AFI->getCalleeSaveBaseToFrameRecordOffset() -
      AFI->getCalleeSavedStackSize();
  Register FramePtr = RegInfo.getFrameRegister(MF);
  CFIInstBuilder(MBB, MBBI, MachineInstr::FrameSetup)
      .buildDefCFA(FramePtr, FixedObject - OffsetToFirstCalleeSaveFromFP);
}

void AArch64PrologueEmitter::emitWindowsStackProbe(
    MachineBasicBlock::iterator MBBI, const DebugLoc &DL, int64_t &NumBytes,
    int64_t RealignmentPadding) const {
  if (AFI->getSVECalleeSavedStackSize())
    report_fatal_error("SVE callee saves not yet supported with stack probing");

  // Find an available register to spill the value of X15 to, if X15 is being
  // used already for nest.
  unsigned X15Scratch = AArch64::NoRegister;
  if (llvm::any_of(MBB.liveins(),
                   [this](const MachineBasicBlock::RegisterMaskPair &LiveIn) {
                     return RegInfo.isSuperOrSubRegisterEq(AArch64::X15,
                                                           LiveIn.PhysReg);
                   })) {
    X15Scratch = AFL.findScratchNonCalleeSaveRegister(&MBB, /*HasCall=*/true);
    assert(X15Scratch != AArch64::NoRegister &&
           (X15Scratch < AArch64::X15 || X15Scratch > AArch64::X17));
#ifndef NDEBUG
    LiveRegs.removeReg(AArch64::X15); // ignore X15 since we restore it
#endif
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::ORRXrr), X15Scratch)
        .addReg(AArch64::XZR)
        .addReg(AArch64::X15, RegState::Undef)
        .addReg(AArch64::X15, RegState::Implicit)
        .setMIFlag(MachineInstr::FrameSetup);
  }

  uint64_t NumWords = (NumBytes + RealignmentPadding) >> 4;
  if (NeedsWinCFI) {
    HasWinCFI = true;
    // alloc_l can hold at most 256MB, so assume that NumBytes doesn't
    // exceed this amount.  We need to move at most 2^24 - 1 into x15.
    // This is at most two instructions, MOVZ followed by MOVK.
    // TODO: Fix to use multiple stack alloc unwind codes for stacks
    // exceeding 256MB in size.
    if (NumBytes >= (1 << 28))
      report_fatal_error("Stack size cannot exceed 256MB for stack "
                         "unwinding purposes");

    uint32_t LowNumWords = NumWords & 0xFFFF;
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::MOVZXi), AArch64::X15)
        .addImm(LowNumWords)
        .addImm(AArch64_AM::getShifterImm(AArch64_AM::LSL, 0))
        .setMIFlag(MachineInstr::FrameSetup);
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::SEH_Nop))
        .setMIFlag(MachineInstr::FrameSetup);
    if ((NumWords & 0xFFFF0000) != 0) {
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::MOVKXi), AArch64::X15)
          .addReg(AArch64::X15)
          .addImm((NumWords & 0xFFFF0000) >> 16) // High half
          .addImm(AArch64_AM::getShifterImm(AArch64_AM::LSL, 16))
          .setMIFlag(MachineInstr::FrameSetup);
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::SEH_Nop))
          .setMIFlag(MachineInstr::FrameSetup);
    }
  } else {
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::MOVi64imm), AArch64::X15)
        .addImm(NumWords)
        .setMIFlags(MachineInstr::FrameSetup);
  }

  const AArch64TargetLowering *TLI = Subtarget.getTargetLowering();
  RTLIB::LibcallImpl ChkStkLibcall = TLI->getLibcallImpl(RTLIB::STACK_PROBE);
  if (ChkStkLibcall == RTLIB::Unsupported)
    reportFatalUsageError("no available implementation of __chkstk");

  const char *ChkStk = TLI->getLibcallImplName(ChkStkLibcall).data();
  switch (MF.getTarget().getCodeModel()) {
  case CodeModel::Tiny:
  case CodeModel::Small:
  case CodeModel::Medium:
  case CodeModel::Kernel:
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::BL))
        .addExternalSymbol(ChkStk)
        .addReg(AArch64::X15, RegState::Implicit)
        .addReg(AArch64::X16,
                RegState::Implicit | RegState::Define | RegState::Dead)
        .addReg(AArch64::X17,
                RegState::Implicit | RegState::Define | RegState::Dead)
        .addReg(AArch64::NZCV,
                RegState::Implicit | RegState::Define | RegState::Dead)
        .setMIFlags(MachineInstr::FrameSetup);
    if (NeedsWinCFI) {
      HasWinCFI = true;
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::SEH_Nop))
          .setMIFlag(MachineInstr::FrameSetup);
    }
    break;
  case CodeModel::Large:
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::MOVaddrEXT))
        .addReg(AArch64::X16, RegState::Define)
        .addExternalSymbol(ChkStk)
        .addExternalSymbol(ChkStk)
        .setMIFlags(MachineInstr::FrameSetup);
    if (NeedsWinCFI) {
      HasWinCFI = true;
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::SEH_Nop))
          .setMIFlag(MachineInstr::FrameSetup);
    }

    BuildMI(MBB, MBBI, DL, TII->get(getBLRCallOpcode(MF)))
        .addReg(AArch64::X16, RegState::Kill)
        .addReg(AArch64::X15, RegState::Implicit | RegState::Define)
        .addReg(AArch64::X16,
                RegState::Implicit | RegState::Define | RegState::Dead)
        .addReg(AArch64::X17,
                RegState::Implicit | RegState::Define | RegState::Dead)
        .addReg(AArch64::NZCV,
                RegState::Implicit | RegState::Define | RegState::Dead)
        .setMIFlags(MachineInstr::FrameSetup);
    if (NeedsWinCFI) {
      HasWinCFI = true;
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::SEH_Nop))
          .setMIFlag(MachineInstr::FrameSetup);
    }
    break;
  }

  BuildMI(MBB, MBBI, DL, TII->get(AArch64::SUBXrx64), AArch64::SP)
      .addReg(AArch64::SP, RegState::Kill)
      .addReg(AArch64::X15, RegState::Kill)
      .addImm(AArch64_AM::getArithExtendImm(AArch64_AM::UXTX, 4))
      .setMIFlags(MachineInstr::FrameSetup);
  if (NeedsWinCFI) {
    HasWinCFI = true;
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::SEH_StackAlloc))
        .addImm(NumBytes)
        .setMIFlag(MachineInstr::FrameSetup);
  }
  NumBytes = 0;

  if (RealignmentPadding > 0) {
    if (RealignmentPadding >= 4096) {
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::MOVi64imm))
          .addReg(AArch64::X16, RegState::Define)
          .addImm(RealignmentPadding)
          .setMIFlags(MachineInstr::FrameSetup);
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::ADDXrx64), AArch64::X15)
          .addReg(AArch64::SP)
          .addReg(AArch64::X16, RegState::Kill)
          .addImm(AArch64_AM::getArithExtendImm(AArch64_AM::UXTX, 0))
          .setMIFlag(MachineInstr::FrameSetup);
    } else {
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::ADDXri), AArch64::X15)
          .addReg(AArch64::SP)
          .addImm(RealignmentPadding)
          .addImm(0)
          .setMIFlag(MachineInstr::FrameSetup);
    }

    uint64_t AndMask = ~(MFI.getMaxAlign().value() - 1);
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::ANDXri), AArch64::SP)
        .addReg(AArch64::X15, RegState::Kill)
        .addImm(AArch64_AM::encodeLogicalImmediate(AndMask, 64));
    AFI->setStackRealigned(true);

    // No need for SEH instructions here; if we're realigning the stack,
    // we've set a frame pointer and already finished the SEH prologue.
    assert(!NeedsWinCFI);
  }
  if (X15Scratch != AArch64::NoRegister) {
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::ORRXrr), AArch64::X15)
        .addReg(AArch64::XZR)
        .addReg(X15Scratch, RegState::Undef)
        .addReg(X15Scratch, RegState::Implicit)
        .setMIFlag(MachineInstr::FrameSetup);
  }
}

void AArch64PrologueEmitter::emitCalleeSavedGPRLocations(
    MachineBasicBlock::iterator MBBI) const {
  const std::vector<CalleeSavedInfo> &CSI = MFI.getCalleeSavedInfo();
  if (CSI.empty())
    return;

  CFIInstBuilder CFIBuilder(MBB, MBBI, MachineInstr::FrameSetup);
  for (const auto &Info : CSI) {
    unsigned FrameIdx = Info.getFrameIdx();
    if (MFI.hasScalableStackID(FrameIdx))
      continue;

    assert(!Info.isSpilledToReg() && "Spilling to registers not implemented");
    int64_t Offset = MFI.getObjectOffset(FrameIdx) - AFL.getOffsetOfLocalArea();
    CFIBuilder.buildOffset(Info.getReg(), Offset);
  }
}

void AArch64PrologueEmitter::emitCalleeSavedSVELocations(
    MachineBasicBlock::iterator MBBI) const {
  // Add callee saved registers to move list.
  const std::vector<CalleeSavedInfo> &CSI = MFI.getCalleeSavedInfo();
  if (CSI.empty())
    return;

  CFIInstBuilder CFIBuilder(MBB, MBBI, MachineInstr::FrameSetup);

  std::optional<int64_t> IncomingVGOffsetFromDefCFA;
  if (AFL.requiresSaveVG(MF)) {
    auto IncomingVG = *find_if(
        reverse(CSI), [](auto &Info) { return Info.getReg() == AArch64::VG; });
    IncomingVGOffsetFromDefCFA = MFI.getObjectOffset(IncomingVG.getFrameIdx()) -
                                 AFL.getOffsetOfLocalArea();
  }

  StackOffset PPRStackSize = AFL.getPPRStackSize(MF);
  for (const auto &Info : CSI) {
    int FI = Info.getFrameIdx();
    if (!MFI.hasScalableStackID(FI))
      continue;

    // Not all unwinders may know about SVE registers, so assume the lowest
    // common denominator.
    assert(!Info.isSpilledToReg() && "Spilling to registers not implemented");
    MCRegister Reg = Info.getReg();
    if (!RegInfo.regNeedsCFI(Reg, Reg))
      continue;

    StackOffset Offset =
        StackOffset::getScalable(MFI.getObjectOffset(FI)) -
        StackOffset::getFixed(AFI->getCalleeSavedStackSize(MFI));

    // The scalable vectors are below (lower address) the scalable predicates
    // with split SVE objects, so we must subtract the size of the predicates.
    if (SVELayout == SVEStackLayout::Split &&
        MFI.getStackID(FI) == TargetStackID::ScalableVector)
      Offset -= PPRStackSize;

    CFIBuilder.insertCFIInst(
        createCFAOffset(RegInfo, Reg, Offset, IncomingVGOffsetFromDefCFA));
  }
}

static bool isFuncletReturnInstr(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  default:
    return false;
  case AArch64::CATCHRET:
  case AArch64::CLEANUPRET:
    return true;
  }
}

AArch64EpilogueEmitter::AArch64EpilogueEmitter(MachineFunction &MF,
                                               MachineBasicBlock &MBB,
                                               const AArch64FrameLowering &AFL)
    : AArch64PrologueEpilogueCommon(MF, MBB, AFL) {
  EmitCFI = AFI->needsAsyncDwarfUnwindInfo(MF);
  HomPrologEpilog = AFL.homogeneousPrologEpilog(MF, &MBB);
  SEHEpilogueStartI = MBB.end();
}

void AArch64EpilogueEmitter::moveSPBelowFP(MachineBasicBlock::iterator MBBI,
                                           StackOffset Offset) {
  // Other combinations could be supported, but are not currently needed.
  assert(Offset.getScalable() < 0 && Offset.getFixed() <= 0 &&
         "expected negative offset (with optional fixed portion)");
  Register Base = AArch64::FP;
  if (int64_t FixedOffset = Offset.getFixed()) {
    // If we have a negative fixed offset, we need to first subtract it in a
    // temporary register first (to avoid briefly deallocating the scalable
    // portion of the offset).
    Base = MF.getRegInfo().createVirtualRegister(&AArch64::GPR64RegClass);
    emitFrameOffset(MBB, MBBI, DL, Base, AArch64::FP,
                    StackOffset::getFixed(FixedOffset), TII,
                    MachineInstr::FrameDestroy);
  }
  emitFrameOffset(MBB, MBBI, DL, AArch64::SP, Base,
                  StackOffset::getScalable(Offset.getScalable()), TII,
                  MachineInstr::FrameDestroy);
}

void AArch64EpilogueEmitter::emitEpilogue() {
  MachineBasicBlock::iterator EpilogueEndI = MBB.getLastNonDebugInstr();
  if (MBB.end() != EpilogueEndI) {
    DL = EpilogueEndI->getDebugLoc();
    IsFunclet = isFuncletReturnInstr(*EpilogueEndI);
  }

  int64_t NumBytes =
      IsFunclet ? AFL.getWinEHFuncletFrameSize(MF) : MFI.getStackSize();

  // All calls are tail calls in GHC calling conv, and functions have no
  // prologue/epilogue.
  if (MF.getFunction().getCallingConv() == CallingConv::GHC)
    return;

  // How much of the stack used by incoming arguments this function is expected
  // to restore in this particular epilogue.
  int64_t ArgumentStackToRestore = AFL.getArgumentStackToRestore(MF, MBB);
  bool IsWin64 = Subtarget.isCallingConvWin64(MF.getFunction().getCallingConv(),
                                              MF.getFunction().isVarArg());
  unsigned FixedObject = AFL.getFixedObjectSize(MF, AFI, IsWin64, IsFunclet);

  int64_t AfterCSRPopSize = ArgumentStackToRestore;
  auto PrologueSaveSize = AFI->getCalleeSavedStackSize() + FixedObject;
  // We cannot rely on the local stack size set in emitPrologue if the function
  // has funclets, as funclets have different local stack size requirements, and
  // the current value set in emitPrologue may be that of the containing
  // function.
  if (MF.hasEHFunclets())
    AFI->setLocalStackSize(NumBytes - PrologueSaveSize);

  if (HomPrologEpilog) {
    assert(!NeedsWinCFI);
    auto FirstHomogenousEpilogI = MBB.getFirstTerminator();
    if (FirstHomogenousEpilogI != MBB.begin()) {
      auto HomogeneousEpilog = std::prev(FirstHomogenousEpilogI);
      if (HomogeneousEpilog->getOpcode() == AArch64::HOM_Epilog)
        FirstHomogenousEpilogI = HomogeneousEpilog;
    }

    // Adjust local stack
    emitFrameOffset(MBB, FirstHomogenousEpilogI, DL, AArch64::SP, AArch64::SP,
                    StackOffset::getFixed(AFI->getLocalStackSize()), TII,
                    MachineInstr::FrameDestroy, false, NeedsWinCFI, &HasWinCFI);

    // SP has been already adjusted while restoring callee save regs.
    // We've bailed-out the case with adjusting SP for arguments.
    assert(AfterCSRPopSize == 0);
    return;
  }

  bool CombineSPBump = shouldCombineCSRLocalStackBump(NumBytes);

  unsigned ProloguePopSize = PrologueSaveSize;
  if (SVELayout == SVEStackLayout::CalleeSavesAboveFrameRecord) {
    // With CalleeSavesAboveFrameRecord ProloguePopSize is the amount of stack
    // that needs to be popped until we reach the start of the SVE save area.
    // The "FixedObject" stack occurs after the SVE area and must be popped
    // later.
    ProloguePopSize -= FixedObject;
    AfterCSRPopSize += FixedObject;
  }

  // Assume we can't combine the last pop with the sp restore.
  if (!CombineSPBump && ProloguePopSize != 0) {
    MachineBasicBlock::iterator Pop = std::prev(MBB.getFirstTerminator());
    while (Pop->getOpcode() == TargetOpcode::CFI_INSTRUCTION ||
           AArch64InstrInfo::isSEHInstruction(*Pop) ||
           (SVELayout == SVEStackLayout::CalleeSavesAboveFrameRecord &&
            isPartOfSVECalleeSaves(Pop)))
      Pop = std::prev(Pop);
    // Converting the last ldp to a post-index ldp is valid only if the last
    // ldp's offset is 0.
    const MachineOperand &OffsetOp = Pop->getOperand(Pop->getNumOperands() - 1);
    // If the offset is 0 and the AfterCSR pop is not actually trying to
    // allocate more stack for arguments (in space that an untimely interrupt
    // may clobber), convert it to a post-index ldp.
    if (OffsetOp.getImm() == 0 && AfterCSRPopSize >= 0) {
      convertCalleeSaveRestoreToSPPrePostIncDec(
          Pop, DL, ProloguePopSize, EmitCFI, MachineInstr::FrameDestroy,
          ProloguePopSize);
    } else if (SVELayout == SVEStackLayout::CalleeSavesAboveFrameRecord) {
      MachineBasicBlock::iterator AfterLastPop = std::next(Pop);
      if (AArch64InstrInfo::isSEHInstruction(*AfterLastPop))
        ++AfterLastPop;
      // If not, and CalleeSavesAboveFrameRecord is enabled, deallocate
      // callee-save non-SVE registers to move the stack pointer to the start of
      // the SVE area.
      emitFrameOffset(MBB, AfterLastPop, DL, AArch64::SP, AArch64::SP,
                      StackOffset::getFixed(ProloguePopSize), TII,
                      MachineInstr::FrameDestroy, false, NeedsWinCFI,
                      &HasWinCFI);
    } else {
      // Otherwise, make sure to emit an add after the last ldp.
      // We're doing this by transferring the size to be restored from the
      // adjustment *before* the CSR pops to the adjustment *after* the CSR
      // pops.
      AfterCSRPopSize += ProloguePopSize;
    }
  }

  // Move past the restores of the callee-saved registers.
  // If we plan on combining the sp bump of the local stack size and the callee
  // save stack size, we might need to adjust the CSR save and restore offsets.
  MachineBasicBlock::iterator FirstGPRRestoreI = MBB.getFirstTerminator();
  MachineBasicBlock::iterator Begin = MBB.begin();
  while (FirstGPRRestoreI != Begin) {
    --FirstGPRRestoreI;
    if (!FirstGPRRestoreI->getFlag(MachineInstr::FrameDestroy) ||
        (SVELayout != SVEStackLayout::CalleeSavesAboveFrameRecord &&
         isPartOfSVECalleeSaves(FirstGPRRestoreI))) {
      ++FirstGPRRestoreI;
      break;
    } else if (CombineSPBump)
      fixupCalleeSaveRestoreStackOffset(*FirstGPRRestoreI,
                                        AFI->getLocalStackSize());
  }

  if (NeedsWinCFI) {
    // Note that there are cases where we insert SEH opcodes in the
    // epilogue when we had no SEH opcodes in the prologue. For
    // example, when there is no stack frame but there are stack
    // arguments. Insert the SEH_EpilogStart and remove it later if it
    // we didn't emit any SEH opcodes to avoid generating WinCFI for
    // functions that don't need it.
    BuildMI(MBB, FirstGPRRestoreI, DL, TII->get(AArch64::SEH_EpilogStart))
        .setMIFlag(MachineInstr::FrameDestroy);
    SEHEpilogueStartI = FirstGPRRestoreI;
    --SEHEpilogueStartI;
  }

  // Determine the ranges of SVE callee-saves. This is done before emitting any
  // code at the end of the epilogue (for Swift async), which can get in the way
  // of finding SVE callee-saves with CalleeSavesAboveFrameRecord.
  auto [PPR, ZPR] = getSVEStackFrameSizes();
  auto [PPRRange, ZPRRange] = partitionSVECS(
      MBB,
      SVELayout == SVEStackLayout::CalleeSavesAboveFrameRecord
          ? MBB.getFirstTerminator()
          : FirstGPRRestoreI,
      PPR.CalleeSavesSize, ZPR.CalleeSavesSize, /*IsEpilogue=*/true);

  if (HasFP && AFI->hasSwiftAsyncContext())
    emitSwiftAsyncContextFramePointer(EpilogueEndI, DL);

  // If there is a single SP update, insert it before the ret and we're done.
  if (CombineSPBump) {
    assert(!AFI->hasSVEStackSize() && "Cannot combine SP bump with SVE");

    // When we are about to restore the CSRs, the CFA register is SP again.
    if (EmitCFI && HasFP)
      CFIInstBuilder(MBB, FirstGPRRestoreI, MachineInstr::FrameDestroy)
          .buildDefCFA(AArch64::SP, NumBytes);

    emitFrameOffset(MBB, MBB.getFirstTerminator(), DL, AArch64::SP, AArch64::SP,
                    StackOffset::getFixed(NumBytes + AfterCSRPopSize), TII,
                    MachineInstr::FrameDestroy, false, NeedsWinCFI, &HasWinCFI,
                    EmitCFI, StackOffset::getFixed(NumBytes));
    return;
  }

  NumBytes -= PrologueSaveSize;
  assert(NumBytes >= 0 && "Negative stack allocation size!?");

  StackOffset SVECalleeSavesSize = ZPR.CalleeSavesSize + PPR.CalleeSavesSize;
  SVEStackAllocations SVEAllocs = getSVEStackAllocations({PPR, ZPR});

  // Deallocate the SVE area.
  if (SVELayout == SVEStackLayout::CalleeSavesAboveFrameRecord) {
    assert(!SVEAllocs.AfterPPRs &&
           "unexpected SVE allocs after PPRs with CalleeSavesAboveFrameRecord");
    // If the callee-save area is before FP, restoring the FP implicitly
    // deallocates non-callee-save SVE allocations. Otherwise, deallocate them
    // explicitly.
    if (!AFI->isStackRealigned() && !MFI.hasVarSizedObjects()) {
      emitFrameOffset(MBB, FirstGPRRestoreI, DL, AArch64::SP, AArch64::SP,
                      SVEAllocs.AfterZPRs, TII, MachineInstr::FrameDestroy,
                      false, NeedsWinCFI, &HasWinCFI);
    }

    // Deallocate callee-save SVE registers.
    emitFrameOffset(MBB, PPRRange.End, DL, AArch64::SP, AArch64::SP,
                    SVEAllocs.BeforePPRs, TII, MachineInstr::FrameDestroy,
                    false, NeedsWinCFI, &HasWinCFI);
  } else if (AFI->hasSVEStackSize()) {
    // If we have stack realignment or variable-sized objects we must use the FP
    // to restore SVE callee saves (as there is an unknown amount of
    // data/padding between the SP and SVE CS area).
    Register BaseForSVEDealloc =
        (AFI->isStackRealigned() || MFI.hasVarSizedObjects()) ? AArch64::FP
                                                              : AArch64::SP;
    if (SVECalleeSavesSize && BaseForSVEDealloc == AArch64::FP) {
      if (ZPR.CalleeSavesSize || SVELayout != SVEStackLayout::Split) {
        // The offset from the frame-pointer to the start of the ZPR saves.
        StackOffset FPOffsetZPR =
            -SVECalleeSavesSize - PPR.LocalsSize -
            StackOffset::getFixed(AFI->getCalleeSaveBaseToFrameRecordOffset());
        // Deallocate the stack space space by moving the SP to the start of the
        // ZPR/PPR callee-save area.
        moveSPBelowFP(ZPRRange.Begin, FPOffsetZPR);
      }
      // With split SVE, the predicates are stored in a separate area above the
      // ZPR saves, so we must adjust the stack to the start of the PPRs.
      if (PPR.CalleeSavesSize && SVELayout == SVEStackLayout::Split) {
        // The offset from the frame-pointer to the start of the PPR saves.
        StackOffset FPOffsetPPR = -PPR.CalleeSavesSize;
        // Move to the start of the PPR area.
        assert(!FPOffsetPPR.getFixed() && "expected only scalable offset");
        emitFrameOffset(MBB, ZPRRange.End, DL, AArch64::SP, AArch64::FP,
                        FPOffsetPPR, TII, MachineInstr::FrameDestroy);
      }
    } else if (BaseForSVEDealloc == AArch64::SP) {
      auto NonSVELocals = StackOffset::getFixed(NumBytes);
      auto CFAOffset = NonSVELocals + StackOffset::getFixed(PrologueSaveSize) +
                       SVEAllocs.totalSize();

      if (SVECalleeSavesSize || SVELayout == SVEStackLayout::Split) {
        // Deallocate non-SVE locals now. This is needed to reach the SVE callee
        // saves, but may also allow combining stack hazard bumps for split SVE.
        SVEAllocs.AfterZPRs += NonSVELocals;
        NumBytes -= NonSVELocals.getFixed();
      }
      // To deallocate the SVE stack adjust by the allocations in reverse.
      emitFrameOffset(MBB, ZPRRange.Begin, DL, AArch64::SP, AArch64::SP,
                      SVEAllocs.AfterZPRs, TII, MachineInstr::FrameDestroy,
                      false, NeedsWinCFI, &HasWinCFI, EmitCFI && !HasFP,
                      CFAOffset);
      CFAOffset -= SVEAllocs.AfterZPRs;
      assert(PPRRange.Begin == ZPRRange.End &&
             "Expected PPR restores after ZPR");
      emitFrameOffset(MBB, PPRRange.Begin, DL, AArch64::SP, AArch64::SP,
                      SVEAllocs.AfterPPRs, TII, MachineInstr::FrameDestroy,
                      false, NeedsWinCFI, &HasWinCFI, EmitCFI && !HasFP,
                      CFAOffset);
      CFAOffset -= SVEAllocs.AfterPPRs;
      emitFrameOffset(MBB, PPRRange.End, DL, AArch64::SP, AArch64::SP,
                      SVEAllocs.BeforePPRs, TII, MachineInstr::FrameDestroy,
                      false, NeedsWinCFI, &HasWinCFI, EmitCFI && !HasFP,
                      CFAOffset);
    }

    if (EmitCFI)
      emitCalleeSavedSVERestores(
          SVELayout == SVEStackLayout::Split ? ZPRRange.End : PPRRange.End);
  }

  if (!HasFP) {
    bool RedZone = AFL.canUseRedZone(MF);
    // If this was a redzone leaf function, we don't need to restore the
    // stack pointer (but we may need to pop stack args for fastcc).
    if (RedZone && AfterCSRPopSize == 0)
      return;

    // Pop the local variables off the stack. If there are no callee-saved
    // registers, it means we are actually positioned at the terminator and can
    // combine stack increment for the locals and the stack increment for
    // callee-popped arguments into (possibly) a single instruction and be done.
    bool NoCalleeSaveRestore = PrologueSaveSize == 0;
    int64_t StackRestoreBytes = RedZone ? 0 : NumBytes;
    if (NoCalleeSaveRestore)
      StackRestoreBytes += AfterCSRPopSize;

    emitFrameOffset(
        MBB, FirstGPRRestoreI, DL, AArch64::SP, AArch64::SP,
        StackOffset::getFixed(StackRestoreBytes), TII,
        MachineInstr::FrameDestroy, false, NeedsWinCFI, &HasWinCFI, EmitCFI,
        StackOffset::getFixed((RedZone ? 0 : NumBytes) + PrologueSaveSize));

    // If we were able to combine the local stack pop with the argument pop,
    // then we're done.
    if (NoCalleeSaveRestore || AfterCSRPopSize == 0)
      return;

    NumBytes = 0;
  }

  // Restore the original stack pointer.
  // FIXME: Rather than doing the math here, we should instead just use
  // non-post-indexed loads for the restores if we aren't actually going to
  // be able to save any instructions.
  if (!IsFunclet && (MFI.hasVarSizedObjects() || AFI->isStackRealigned())) {
    emitFrameOffset(
        MBB, FirstGPRRestoreI, DL, AArch64::SP, AArch64::FP,
        StackOffset::getFixed(-AFI->getCalleeSaveBaseToFrameRecordOffset()),
        TII, MachineInstr::FrameDestroy, false, NeedsWinCFI, &HasWinCFI);
  } else if (NumBytes)
    emitFrameOffset(MBB, FirstGPRRestoreI, DL, AArch64::SP, AArch64::SP,
                    StackOffset::getFixed(NumBytes), TII,
                    MachineInstr::FrameDestroy, false, NeedsWinCFI, &HasWinCFI);

  // When we are about to restore the CSRs, the CFA register is SP again.
  if (EmitCFI && HasFP)
    CFIInstBuilder(MBB, FirstGPRRestoreI, MachineInstr::FrameDestroy)
        .buildDefCFA(AArch64::SP, PrologueSaveSize);

  // This must be placed after the callee-save restore code because that code
  // assumes the SP is at the same location as it was after the callee-save save
  // code in the prologue.
  if (AfterCSRPopSize) {
    assert(AfterCSRPopSize > 0 && "attempting to reallocate arg stack that an "
                                  "interrupt may have clobbered");

    emitFrameOffset(
        MBB, MBB.getFirstTerminator(), DL, AArch64::SP, AArch64::SP,
        StackOffset::getFixed(AfterCSRPopSize), TII, MachineInstr::FrameDestroy,
        false, NeedsWinCFI, &HasWinCFI, EmitCFI,
        StackOffset::getFixed(AfterCSRPopSize - ArgumentStackToRestore));
  }
}

bool AArch64EpilogueEmitter::shouldCombineCSRLocalStackBump(
    uint64_t StackBumpBytes) const {
  if (!AArch64PrologueEpilogueCommon::shouldCombineCSRLocalStackBump(
          StackBumpBytes))
    return false;
  if (MBB.empty())
    return true;

  // Disable combined SP bump if the last instruction is an MTE tag store. It
  // is almost always better to merge SP adjustment into those instructions.
  MachineBasicBlock::iterator LastI = MBB.getFirstTerminator();
  MachineBasicBlock::iterator Begin = MBB.begin();
  while (LastI != Begin) {
    --LastI;
    if (LastI->isTransient())
      continue;
    if (!LastI->getFlag(MachineInstr::FrameDestroy))
      break;
  }
  switch (LastI->getOpcode()) {
  case AArch64::STGloop:
  case AArch64::STZGloop:
  case AArch64::STGi:
  case AArch64::STZGi:
  case AArch64::ST2Gi:
  case AArch64::STZ2Gi:
    return false;
  default:
    return true;
  }
  llvm_unreachable("unreachable");
}

void AArch64EpilogueEmitter::emitSwiftAsyncContextFramePointer(
    MachineBasicBlock::iterator MBBI, const DebugLoc &DL) const {
  switch (MF.getTarget().Options.SwiftAsyncFramePointer) {
  case SwiftAsyncFramePointerMode::DeploymentBased:
    // Avoid the reload as it is GOT relative, and instead fall back to the
    // hardcoded value below.  This allows a mismatch between the OS and
    // application without immediately terminating on the difference.
    [[fallthrough]];
  case SwiftAsyncFramePointerMode::Always:
    // We need to reset FP to its untagged state on return. Bit 60 is
    // currently used to show the presence of an extended frame.

    // BIC x29, x29, #0x1000_0000_0000_0000
    BuildMI(MBB, MBB.getFirstTerminator(), DL, TII->get(AArch64::ANDXri),
            AArch64::FP)
        .addUse(AArch64::FP)
        .addImm(0x10fe)
        .setMIFlag(MachineInstr::FrameDestroy);
    if (NeedsWinCFI) {
      BuildMI(MBB, MBBI, DL, TII->get(AArch64::SEH_Nop))
          .setMIFlags(MachineInstr::FrameDestroy);
      HasWinCFI = true;
    }
    break;

  case SwiftAsyncFramePointerMode::Never:
    break;
  }
}

void AArch64EpilogueEmitter::emitShadowCallStackEpilogue(
    MachineBasicBlock::iterator MBBI, const DebugLoc &DL) const {
  // Shadow call stack epilog: ldr x30, [x18, #-8]!
  BuildMI(MBB, MBBI, DL, TII->get(AArch64::LDRXpre))
      .addReg(AArch64::X18, RegState::Define)
      .addReg(AArch64::LR, RegState::Define)
      .addReg(AArch64::X18)
      .addImm(-8)
      .setMIFlag(MachineInstr::FrameDestroy);

  if (NeedsWinCFI)
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::SEH_Nop))
        .setMIFlag(MachineInstr::FrameDestroy);

  if (AFI->needsAsyncDwarfUnwindInfo(MF))
    CFIInstBuilder(MBB, MBBI, MachineInstr::FrameDestroy)
        .buildRestore(AArch64::X18);
}

void AArch64EpilogueEmitter::emitCalleeSavedRestores(
    MachineBasicBlock::iterator MBBI, bool SVE) const {
  const std::vector<CalleeSavedInfo> &CSI = MFI.getCalleeSavedInfo();
  if (CSI.empty())
    return;

  CFIInstBuilder CFIBuilder(MBB, MBBI, MachineInstr::FrameDestroy);

  for (const auto &Info : CSI) {
    if (SVE != MFI.hasScalableStackID(Info.getFrameIdx()))
      continue;

    MCRegister Reg = Info.getReg();
    if (SVE && !RegInfo.regNeedsCFI(Reg, Reg))
      continue;

    CFIBuilder.buildRestore(Info.getReg());
  }
}

void AArch64EpilogueEmitter::finalizeEpilogue() const {
  if (AFI->needsShadowCallStackPrologueEpilogue(MF)) {
    emitShadowCallStackEpilogue(MBB.getFirstTerminator(), DL);
    HasWinCFI |= NeedsWinCFI;
  }
  if (EmitCFI)
    emitCalleeSavedGPRRestores(MBB.getFirstTerminator());
  if (AFI->shouldSignReturnAddress(MF)) {
    // If pac-ret+leaf is in effect, PAUTH_EPILOGUE pseudo instructions
    // are inserted by emitPacRetPlusLeafHardening().
    if (!AFL.shouldSignReturnAddressEverywhere(MF)) {
      BuildMI(MBB, MBB.getFirstTerminator(), DL,
              TII->get(AArch64::PAUTH_EPILOGUE))
          .setMIFlag(MachineInstr::FrameDestroy);
    }
    // AArch64PointerAuth pass will insert SEH_PACSignLR
    HasWinCFI |= NeedsWinCFI;
  }
  if (HasWinCFI) {
    BuildMI(MBB, MBB.getFirstTerminator(), DL, TII->get(AArch64::SEH_EpilogEnd))
        .setMIFlag(MachineInstr::FrameDestroy);
    if (!MF.hasWinCFI())
      MF.setHasWinCFI(true);
  }
  if (NeedsWinCFI) {
    assert(SEHEpilogueStartI != MBB.end());
    if (!HasWinCFI)
      MBB.erase(SEHEpilogueStartI);
  }
}

} // namespace llvm
