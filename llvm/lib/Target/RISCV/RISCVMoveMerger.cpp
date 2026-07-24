//===-- RISCVMoveMerger.cpp - RISC-V move merge pass ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that performs move related peephole optimizations
// as Zcmp has specified. This pass should be run after register allocation.
//
// This pass also supports Xqccmp, which has identical instructions.
//
//===----------------------------------------------------------------------===//

#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"

using namespace llvm;

#define RISCV_MOVE_MERGE_NAME "RISC-V Zcmp move merging pass"

namespace {
struct RISCVMoveMerge : public MachineFunctionPass {
  static char ID;

  RISCVMoveMerge() : MachineFunctionPass(ID) {}

  const RISCVSubtarget *ST;
  const RISCVInstrInfo *TII;
  const TargetRegisterInfo *TRI;

  // Track which register units have been modified and used.
  LiveRegUnits ModifiedRegUnits, UsedRegUnits;

  bool isGPRPairCopyCandidate(const DestSourcePair &RegPair, bool EvenRegPair);

  bool isCandidateToMergeMVA01S(const DestSourcePair &RegPair);
  bool isCandidateToMergeMVSA01(const DestSourcePair &RegPair);

  bool isPLIPairCandidate(const MachineInstr &MI, bool EvenRegPair);

  // Merge the two instructions indicated into a single pair instruction.
  MachineBasicBlock::iterator
  mergeGPRPairInsns(MachineBasicBlock::iterator I,
                    MachineBasicBlock::iterator Paired, bool RegPairIsEven);
  MachineBasicBlock::iterator
  mergePairedInsns(MachineBasicBlock::iterator I,
                   MachineBasicBlock::iterator Paired, bool MoveFromSToA);
  MachineBasicBlock::iterator mergePLIPair(MachineBasicBlock::iterator I,
                                           MachineBasicBlock::iterator Paired,
                                           bool RegPairIsEven);

  MachineBasicBlock::iterator
  findMatchingGPRPairCopy(MachineBasicBlock::iterator &MBBI, bool EvenRegPair,
                          const DestSourcePair &RegPair);
  // Look for C.MV instruction that can be combined with
  // the given instruction into CM.MVA01S or CM.MVSA01. Return the matching
  // instruction if one exists.
  MachineBasicBlock::iterator
  findMatchingSACopy(MachineBasicBlock::iterator &MBBI, bool MoveFromSToA,
                     const DestSourcePair &RegPair);
  MachineBasicBlock::iterator findMatchingPLI(MachineBasicBlock::iterator &MBBI,
                                              bool EvenRegPair);
  bool mergeMovePairs(MachineBasicBlock &MBB);
  bool runOnMachineFunction(MachineFunction &Fn) override;

  StringRef getPassName() const override { return RISCV_MOVE_MERGE_NAME; }
};

char RISCVMoveMerge::ID = 0;

} // end of anonymous namespace

INITIALIZE_PASS(RISCVMoveMerge, "riscv-move-merge", RISCV_MOVE_MERGE_NAME,
                false, false)

static unsigned getCM_MVOpcode(const RISCVSubtarget &ST, bool MoveFromSToA) {
  if (ST.hasStdExtZcmp())
    return MoveFromSToA ? RISCV::CM_MVA01S : RISCV::CM_MVSA01;

  if (ST.hasVendorXqccmp())
    return MoveFromSToA ? RISCV::QC_CM_MVA01S : RISCV::QC_CM_MVSA01;

  llvm_unreachable("Unhandled subtarget with paired move.");
}

// Returns 0 if Opc has no paired form.
static unsigned getPairedPLIOpcode(unsigned Opc) {
  switch (Opc) {
  case RISCV::PLI_B:
    return RISCV::PLI_DB;
  case RISCV::PLI_H:
    return RISCV::PLI_DH;
  case RISCV::PLUI_H:
    return RISCV::PLUI_DH;
  default:
    return 0;
  }
}

bool RISCVMoveMerge::isGPRPairCopyCandidate(const DestSourcePair &RegPair,
                                            bool EvenRegPair) {
  Register Destination = RegPair.Destination->getReg();
  Register Source = RegPair.Source->getReg();

  if (Source == Destination)
    return false;

  if ((!ST->hasStdExtZdinx() && !ST->hasStdExtP()) || ST->is64Bit())
    return false;

  unsigned SubIdx = EvenRegPair ? RISCV::sub_gpr_even : RISCV::sub_gpr_odd;

  Register SrcPair =
      TRI->getMatchingSuperReg(Source, SubIdx, &RISCV::GPRPairRegClass);
  Register DestPair =
      TRI->getMatchingSuperReg(Destination, SubIdx, &RISCV::GPRPairRegClass);

  return SrcPair.isValid() && DestPair.isValid();
}

// Check if registers meet CM.MVA01S constraints.
bool RISCVMoveMerge::isCandidateToMergeMVA01S(const DestSourcePair &RegPair) {
  Register Destination = RegPair.Destination->getReg();
  Register Source = RegPair.Source->getReg();
  // If destination is not a0 or a1.
  if ((ST->hasStdExtZcmp() || ST->hasVendorXqccmp()) &&
      (Destination == RISCV::X10 || Destination == RISCV::X11) &&
      RISCV::SR07RegClass.contains(Source))
    return true;
  return false;
}

// Check if registers meet CM.MVSA01 constraints.
bool RISCVMoveMerge::isCandidateToMergeMVSA01(const DestSourcePair &RegPair) {
  Register Destination = RegPair.Destination->getReg();
  Register Source = RegPair.Source->getReg();
  // If Source is s0 - s7.
  if ((ST->hasStdExtZcmp() || ST->hasVendorXqccmp()) &&
      (Source == RISCV::X10 || Source == RISCV::X11) &&
      RISCV::SR07RegClass.contains(Destination))
    return true;
  return false;
}

// Check if MI is a single-reg pli/plui whose destination is a half of a
// GPRPair.
bool RISCVMoveMerge::isPLIPairCandidate(const MachineInstr &MI,
                                        bool EvenRegPair) {
  if (!ST->hasStdExtP() || ST->is64Bit())
    return false;
  if (!getPairedPLIOpcode(MI.getOpcode()))
    return false;
  unsigned SubIdx = EvenRegPair ? RISCV::sub_gpr_even : RISCV::sub_gpr_odd;
  return TRI
      ->getMatchingSuperReg(MI.getOperand(0).getReg(), SubIdx,
                            &RISCV::GPRPairRegClass)
      .isValid();
}

MachineBasicBlock::iterator
RISCVMoveMerge::mergeGPRPairInsns(MachineBasicBlock::iterator I,
                                  MachineBasicBlock::iterator Paired,
                                  bool RegPairIsEven) {
  MachineBasicBlock::iterator E = I->getParent()->end();
  MachineBasicBlock::iterator NextI = next_nodbg(I, E);
  DestSourcePair FirstPair = *TII->isCopyInstrImpl(*I);
  DestSourcePair SecondPair = *TII->isCopyInstrImpl(*Paired);

  if (NextI == Paired)
    NextI = next_nodbg(NextI, E);
  DebugLoc DL = I->getDebugLoc();

  // Make a copy of the second instruction to update the kill
  // flag.
  MachineOperand PairedSource = *SecondPair.Source;

  for (auto It = std::next(I); It != Paired && PairedSource.isKill(); ++It)
    if (It->readsRegister(PairedSource.getReg(), TRI))
      PairedSource.setIsKill(false);

  unsigned GPRPairIdx =
      RegPairIsEven ? RISCV::sub_gpr_even : RISCV::sub_gpr_odd;
  MCRegister SrcReg = TRI->getMatchingSuperReg(
      FirstPair.Source->getReg(), GPRPairIdx, &RISCV::GPRPairRegClass);
  MCRegister DestReg = TRI->getMatchingSuperReg(
      FirstPair.Destination->getReg(), GPRPairIdx, &RISCV::GPRPairRegClass);
  bool KillSrc = PairedSource.isKill() && FirstPair.Source->isKill();

  TII->copyPhysReg(*I->getParent(), I, DL, DestReg, SrcReg, KillSrc);

  I->eraseFromParent();
  Paired->eraseFromParent();
  return NextI;
}

MachineBasicBlock::iterator
RISCVMoveMerge::mergePairedInsns(MachineBasicBlock::iterator I,
                                 MachineBasicBlock::iterator Paired,
                                 bool MoveFromSToA) {
  const MachineOperand *Sreg1, *Sreg2;
  MachineBasicBlock::iterator E = I->getParent()->end();
  MachineBasicBlock::iterator NextI = next_nodbg(I, E);
  DestSourcePair FirstPair = *TII->isCopyInstrImpl(*I);
  DestSourcePair PairedRegs = *TII->isCopyInstrImpl(*Paired);

  if (NextI == Paired)
    NextI = next_nodbg(NextI, E);
  DebugLoc DL = I->getDebugLoc();

  // Make a copy so we can update the kill flag in the MoveFromSToA case. The
  // copied operand needs to be scoped outside the if since we make a pointer
  // to it.
  MachineOperand PairedSource = *PairedRegs.Source;

  // The order of S-reg depends on which instruction holds A0, instead of
  // the order of register pair.
  // e,g.
  //   mv a1, s1
  //   mv a0, s2    =>  cm.mva01s s2,s1
  //
  //   mv a0, s2
  //   mv a1, s1    =>  cm.mva01s s2,s1
  unsigned Opcode = getCM_MVOpcode(*ST, MoveFromSToA);
  if (MoveFromSToA) {
    // We are moving one of the copies earlier so its kill flag may become
    // invalid. Clear the copied kill flag if there are any reads of the
    // register between the new location and the old location.
    for (auto It = std::next(I); It != Paired && PairedSource.isKill(); ++It)
      if (It->readsRegister(PairedSource.getReg(), TRI))
        PairedSource.setIsKill(false);

    Sreg1 = FirstPair.Source;
    Sreg2 = &PairedSource;
    if (FirstPair.Destination->getReg() != RISCV::X10)
      std::swap(Sreg1, Sreg2);
  } else {
    Sreg1 = FirstPair.Destination;
    Sreg2 = PairedRegs.Destination;
    if (FirstPair.Source->getReg() != RISCV::X10)
      std::swap(Sreg1, Sreg2);
  }

  BuildMI(*I->getParent(), I, DL, TII->get(Opcode)).add(*Sreg1).add(*Sreg2);

  I->eraseFromParent();
  Paired->eraseFromParent();
  return NextI;
}

MachineBasicBlock::iterator
RISCVMoveMerge::mergePLIPair(MachineBasicBlock::iterator I,
                             MachineBasicBlock::iterator Paired,
                             bool RegPairIsEven) {
  MachineBasicBlock::iterator E = I->getParent()->end();
  MachineBasicBlock::iterator NextI = next_nodbg(I, E);

  if (NextI == Paired)
    NextI = next_nodbg(NextI, E);
  DebugLoc DL = I->getDebugLoc();

  unsigned Opcode = getPairedPLIOpcode(I->getOpcode());
  unsigned GPRPairIdx =
      RegPairIsEven ? RISCV::sub_gpr_even : RISCV::sub_gpr_odd;
  Register DestReg = TRI->getMatchingSuperReg(
      I->getOperand(0).getReg(), GPRPairIdx, &RISCV::GPRPairRegClass);

  BuildMI(*I->getParent(), I, DL, TII->get(Opcode), DestReg)
      .addImm(I->getOperand(1).getImm());

  I->eraseFromParent();
  Paired->eraseFromParent();
  return NextI;
}

MachineBasicBlock::iterator
RISCVMoveMerge::findMatchingGPRPairCopy(MachineBasicBlock::iterator &MBBI,
                                        bool EvenRegPair,
                                        const DestSourcePair &RegPair) {
  MachineBasicBlock::iterator E = MBBI->getParent()->end();
  ModifiedRegUnits.clear();
  UsedRegUnits.clear();
  unsigned RegPairIdx = EvenRegPair ? RISCV::sub_gpr_even : RISCV::sub_gpr_odd;
  unsigned SecondPairIdx =
      !EvenRegPair ? RISCV::sub_gpr_even : RISCV::sub_gpr_odd;

  // Get the expected source/destination registers of the matching lane.
  Register SrcGPRPair = TRI->getMatchingSuperReg(
      RegPair.Source->getReg(), RegPairIdx, &RISCV::GPRPairRegClass);
  Register DestGPRPair = TRI->getMatchingSuperReg(
      RegPair.Destination->getReg(), RegPairIdx, &RISCV::GPRPairRegClass);
  Register ExpectedSourceReg = TRI->getSubReg(SrcGPRPair, SecondPairIdx);
  Register ExpectedDestReg = TRI->getSubReg(DestGPRPair, SecondPairIdx);

  for (MachineBasicBlock::iterator I = next_nodbg(MBBI, E); I != E;
       I = next_nodbg(I, E)) {

    MachineInstr &MI = *I;

    if (auto SecondPair = TII->isCopyInstrImpl(MI)) {
      Register SourceReg = SecondPair->Source->getReg();
      Register DestReg = SecondPair->Destination->getReg();

      if (RegPair.Destination->getReg() == DestReg ||
          RegPair.Source->getReg() == SourceReg)
        return E;

      // Check if the second pair's registers match the other lane of the
      // GPRPairs.
      if (SourceReg == ExpectedSourceReg && DestReg == ExpectedDestReg)
        return I;
    }
    // Update modified / used register units.
    LiveRegUnits::accumulateUsedDefed(MI, ModifiedRegUnits, UsedRegUnits, TRI);
    // Once expected lane registers are clobbered/read in-between, we can stop
    // scanning since the pair cannot be legally merged anymore.
    if (!ModifiedRegUnits.available(ExpectedDestReg) ||
        !UsedRegUnits.available(ExpectedDestReg) ||
        !ModifiedRegUnits.available(ExpectedSourceReg))
      return E;
  }
  return E;
}

MachineBasicBlock::iterator
RISCVMoveMerge::findMatchingSACopy(MachineBasicBlock::iterator &MBBI,
                                   bool MoveFromSToA,
                                   const DestSourcePair &RegPair) {
  MachineBasicBlock::iterator E = MBBI->getParent()->end();

  // Track which register units have been modified and used between the first
  // insn and the second insn.
  ModifiedRegUnits.clear();
  UsedRegUnits.clear();

  for (MachineBasicBlock::iterator I = next_nodbg(MBBI, E); I != E;
       I = next_nodbg(I, E)) {

    MachineInstr &MI = *I;

    if (auto SecondPair = TII->isCopyInstrImpl(MI)) {
      Register SourceReg = SecondPair->Source->getReg();
      Register DestReg = SecondPair->Destination->getReg();

      bool IsCandidate = MoveFromSToA ? isCandidateToMergeMVA01S(*SecondPair)
                                      : isCandidateToMergeMVSA01(*SecondPair);
      if (IsCandidate) {
        // Second destination must be different.
        if (RegPair.Destination->getReg() == DestReg)
          return E;

        // For AtoS the source must also be different.
        if (!MoveFromSToA && RegPair.Source->getReg() == SourceReg)
          return E;

        // If paired destination register was modified or used, the source reg
        // was modified, there is no possibility of finding matching
        // instruction so exit early.
        if (!ModifiedRegUnits.available(DestReg) ||
            !UsedRegUnits.available(DestReg) ||
            !ModifiedRegUnits.available(SourceReg))
          return E;

        return I;
      }
    }
    // Update modified / used register units.
    LiveRegUnits::accumulateUsedDefed(MI, ModifiedRegUnits, UsedRegUnits, TRI);
  }
  return E;
}

// Look for a same-opcode pli/plui writing the other lane of the same GPRPair
// with the same immediate. Return the matching instruction if one exists.
MachineBasicBlock::iterator
RISCVMoveMerge::findMatchingPLI(MachineBasicBlock::iterator &MBBI,
                                bool EvenRegPair) {
  MachineBasicBlock::iterator E = MBBI->getParent()->end();
  ModifiedRegUnits.clear();
  UsedRegUnits.clear();
  unsigned Opc = MBBI->getOpcode();
  Register FirstDestReg = MBBI->getOperand(0).getReg();
  int64_t FirstImm = MBBI->getOperand(1).getImm();
  unsigned RegPairIdx = EvenRegPair ? RISCV::sub_gpr_even : RISCV::sub_gpr_odd;
  unsigned SecondPairIdx =
      !EvenRegPair ? RISCV::sub_gpr_even : RISCV::sub_gpr_odd;

  // Get the expected destination register of the matching lane.
  Register DestGPRPair = TRI->getMatchingSuperReg(FirstDestReg, RegPairIdx,
                                                  &RISCV::GPRPairRegClass);
  Register ExpectedDestReg = TRI->getSubReg(DestGPRPair, SecondPairIdx);

  for (MachineBasicBlock::iterator I = next_nodbg(MBBI, E); I != E;
       I = next_nodbg(I, E)) {

    MachineInstr &MI = *I;

    if (MI.getOpcode() == Opc) {
      Register DestReg = MI.getOperand(0).getReg();
      int64_t Imm = MI.getOperand(1).getImm();

      if (FirstDestReg == DestReg)
        return E;

      // Check if the second PLI matches the other lane and immediate.
      if (DestReg == ExpectedDestReg && Imm == FirstImm)
        return I;
    }
    // Update modified / used register units.
    LiveRegUnits::accumulateUsedDefed(MI, ModifiedRegUnits, UsedRegUnits, TRI);
    // Once the expected lane register is clobbered/read in-between, we can
    // stop scanning since the pair cannot be legally merged anymore.
    if (!ModifiedRegUnits.available(ExpectedDestReg) ||
        !UsedRegUnits.available(ExpectedDestReg))
      return E;
  }
  return E;
}

// Finds instructions, which could be represented as C.MV instructions and
// merged into CM.MVA01S or CM.MVSA01.
bool RISCVMoveMerge::mergeMovePairs(MachineBasicBlock &MBB) {
  bool Modified = false;

  for (MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
       MBBI != E;) {
    // Try merging a pair of single-reg PLI/PLUI into a paired form.
    bool IsPLIEven = isPLIPairCandidate(*MBBI, /*EvenRegPair=*/true);
    bool IsPLIOdd = isPLIPairCandidate(*MBBI, /*EvenRegPair=*/false);
    if (IsPLIEven != IsPLIOdd) {
      MachineBasicBlock::iterator Paired = findMatchingPLI(MBBI, IsPLIEven);
      if (Paired != E) {
        MBBI = mergePLIPair(MBBI, Paired, IsPLIEven);
        Modified = true;
        continue;
      }
    }

    // Check if the instruction can be compressed to C.MV instruction. If it
    // can, return Dest/Src register pair.
    auto RegPair = TII->isCopyInstrImpl(*MBBI);
    if (RegPair.has_value()) {
      bool MoveFromSToA = isCandidateToMergeMVA01S(*RegPair);
      bool MoveFromAToS = isCandidateToMergeMVSA01(*RegPair);
      bool IsEven = isGPRPairCopyCandidate(*RegPair, /*EvenRegPair=*/true);
      bool IsOdd = isGPRPairCopyCandidate(*RegPair, /*EvenRegPair=*/false);
      if (!MoveFromSToA && !MoveFromAToS && !IsEven && !IsOdd) {
        ++MBBI;
        continue;
      }

      MachineBasicBlock::iterator Paired = E;
      if (MoveFromSToA || MoveFromAToS) {
        Paired = findMatchingSACopy(MBBI, MoveFromSToA, *RegPair);
        if (Paired != E) {
          MBBI = mergePairedInsns(MBBI, Paired, MoveFromSToA);
          Modified = true;
          continue;
        }
      }
      if (IsEven != IsOdd) {
        Paired = findMatchingGPRPairCopy(MBBI, IsEven, *RegPair);
        if (Paired != E) {
          MBBI = mergeGPRPairInsns(MBBI, Paired, IsEven);
          Modified = true;
          continue;
        }
      }
    }
    ++MBBI;
  }
  return Modified;
}

bool RISCVMoveMerge::runOnMachineFunction(MachineFunction &Fn) {
  if (skipFunction(Fn.getFunction()))
    return false;

  ST = &Fn.getSubtarget<RISCVSubtarget>();
  bool HasGPRPairCopy =
      !ST->is64Bit() && (ST->hasStdExtZdinx() || ST->hasStdExtP());
  if (!ST->hasStdExtZcmp() && !ST->hasVendorXqccmp() && !HasGPRPairCopy)
    return false;

  TII = ST->getInstrInfo();
  TRI = ST->getRegisterInfo();
  // Resize the modified and used register unit trackers.  We do this once
  // per function and then clear the register units each time we optimize a
  // move.
  ModifiedRegUnits.init(*TRI);
  UsedRegUnits.init(*TRI);
  bool Modified = false;
  for (auto &MBB : Fn)
    Modified |= mergeMovePairs(MBB);
  return Modified;
}

/// createRISCVMoveMergePass - returns an instance of the
/// move merge pass.
FunctionPass *llvm::createRISCVMoveMergePass() { return new RISCVMoveMerge(); }
