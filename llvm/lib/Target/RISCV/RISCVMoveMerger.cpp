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

  bool isEvenRegisterCopy(const DestSourcePair &RegPair);
  bool isOddRegisterCopy(const DestSourcePair &RegPair);

  bool isCandidateToMergeMVA01S(const DestSourcePair &RegPair);
  bool isCandidateToMergeMVSA01(const DestSourcePair &RegPair);
  // Merge the two instructions indicated into a single pair instruction.
  MachineBasicBlock::iterator
  mergeGPRPairInsns(MachineBasicBlock::iterator I,
                    MachineBasicBlock::iterator Paired, bool RegPairIsEven);
  MachineBasicBlock::iterator
  mergePairedInsns(MachineBasicBlock::iterator I,
                   MachineBasicBlock::iterator Paired, bool MoveFromSToA);

  MachineBasicBlock::iterator
  findMatchingInstPair(MachineBasicBlock::iterator &MBBI, bool EvenRegPair,
                       const DestSourcePair &RegPair);
  // Look for C.MV instruction that can be combined with
  // the given instruction into CM.MVA01S or CM.MVSA01. Return the matching
  // instruction if one exists.
  MachineBasicBlock::iterator
  findMatchingInst(MachineBasicBlock::iterator &MBBI, bool MoveFromSToA,
                   const DestSourcePair &RegPair);
  bool mergeMoveSARegPair(MachineBasicBlock &MBB);
  bool runOnMachineFunction(MachineFunction &Fn) override;

  StringRef getPassName() const override { return RISCV_MOVE_MERGE_NAME; }
};

char RISCVMoveMerge::ID = 0;

} // end of anonymous namespace

INITIALIZE_PASS(RISCVMoveMerge, "riscv-move-merge", RISCV_MOVE_MERGE_NAME,
                false, false)

static unsigned getGPRPairCopyOpcode(const RISCVSubtarget &ST) {
  if (ST.hasStdExtZdinx())
    return RISCV::FSGNJ_D_IN32X;

  if (ST.hasStdExtP())
    return RISCV::PADD_DW;

  llvm_unreachable("Unhandled subtarget with paired move.");
}

static unsigned getCM_MVOpcode(const RISCVSubtarget &ST, bool MoveFromSToA) {
  if (ST.hasStdExtZcmp())
    return MoveFromSToA ? RISCV::CM_MVA01S : RISCV::CM_MVSA01;

  if (ST.hasVendorXqccmp())
    return MoveFromSToA ? RISCV::QC_CM_MVA01S : RISCV::QC_CM_MVSA01;

  llvm_unreachable("Unhandled subtarget with paired move.");
}

bool RISCVMoveMerge::isEvenRegisterCopy(const DestSourcePair &RegPair) {
  Register Destination = RegPair.Destination->getReg();
  Register Source = RegPair.Source->getReg();

  if (Source == Destination)
    return false;

  Register SrcPair = TRI->getMatchingSuperReg(Source, RISCV::sub_gpr_even,
                                              &RISCV::GPRPairRegClass);
  Register DestPair = TRI->getMatchingSuperReg(Destination, RISCV::sub_gpr_even,
                                               &RISCV::GPRPairRegClass);

  return SrcPair.isValid() && DestPair.isValid();
}

bool RISCVMoveMerge::isOddRegisterCopy(const DestSourcePair &RegPair) {
  Register Destination = RegPair.Destination->getReg();
  Register Source = RegPair.Source->getReg();

  if (Source == Destination)
    return false;

  Register SrcPair = TRI->getMatchingSuperReg(Source, RISCV::sub_gpr_odd,
                                              &RISCV::GPRPairRegClass);
  Register DestPair = TRI->getMatchingSuperReg(Destination, RISCV::sub_gpr_odd,
                                               &RISCV::GPRPairRegClass);

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

MachineBasicBlock::iterator
RISCVMoveMerge::mergeGPRPairInsns(MachineBasicBlock::iterator I,
                                  MachineBasicBlock::iterator Paired,
                                  bool RegPairIsEven) {
  MachineBasicBlock::iterator E = I->getParent()->end();
  MachineBasicBlock::iterator NextI = next_nodbg(I, E);
  DestSourcePair FirstPair = TII->isCopyInstrImpl(*I).value();
  DestSourcePair SecondPair = TII->isCopyInstrImpl(*Paired).value();

  if (NextI == Paired)
    NextI = next_nodbg(NextI, E);
  DebugLoc DL = I->getDebugLoc();

  // Make a copy of the second instruction to update the kill
  // flag.
  MachineOperand PairedSource = *SecondPair.Source;

  unsigned Opcode = getGPRPairCopyOpcode(*ST);
  for (auto It = std::next(I); It != Paired && PairedSource.isKill(); ++It)
    if (It->readsRegister(PairedSource.getReg(), TRI))
      PairedSource.setIsKill(false);

  Register SrcReg1, SrcReg2, DestReg;
  unsigned GPRPairIdx =
      RegPairIsEven ? RISCV::sub_gpr_even : RISCV::sub_gpr_odd;
  SrcReg1 = TRI->getMatchingSuperReg(FirstPair.Source->getReg(), GPRPairIdx,
                                     &RISCV::GPRPairRegClass);
  SrcReg2 = ST->hasStdExtZdinx() ? SrcReg1 : Register(RISCV::X0_Pair);
  DestReg = TRI->getMatchingSuperReg(FirstPair.Destination->getReg(),
                                     GPRPairIdx, &RISCV::GPRPairRegClass);

  BuildMI(*I->getParent(), I, DL, TII->get(Opcode), DestReg)
      .addReg(SrcReg1, getKillRegState(PairedSource.isKill() &&
                                       FirstPair.Source->isKill()))
      .addReg(SrcReg2, getKillRegState(PairedSource.isKill() &&
                                       FirstPair.Source->isKill()));

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
  DestSourcePair FirstPair = TII->isCopyInstrImpl(*I).value();
  DestSourcePair PairedRegs = TII->isCopyInstrImpl(*Paired).value();

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
RISCVMoveMerge::findMatchingInstPair(MachineBasicBlock::iterator &MBBI,
                                     bool EvenRegPair,
                                     const DestSourcePair &RegPair) {
  MachineBasicBlock::iterator E = MBBI->getParent()->end();
  ModifiedRegUnits.clear();
  UsedRegUnits.clear();

  for (MachineBasicBlock::iterator I = next_nodbg(MBBI, E); I != E;
       I = next_nodbg(I, E)) {

    MachineInstr &MI = *I;

    if (auto SecondPair = TII->isCopyInstrImpl(MI)) {
      Register SourceReg = SecondPair->Source->getReg();
      Register DestReg = SecondPair->Destination->getReg();

      if (RegPair.Destination->getReg() == DestReg ||
          RegPair.Source->getReg() == SourceReg)
        return E;

      unsigned RegPairIdx =
          EvenRegPair ? RISCV::sub_gpr_even : RISCV::sub_gpr_odd;
      unsigned SecondPairIdx =
          !EvenRegPair ? RISCV::sub_gpr_even : RISCV::sub_gpr_odd;

      // Get the register GPRPair.
      Register SrcGPRPair = TRI->getMatchingSuperReg(
          RegPair.Source->getReg(), RegPairIdx, &RISCV::GPRPairRegClass);

      Register DestGPRPair = TRI->getMatchingSuperReg(
          RegPair.Destination->getReg(), RegPairIdx, &RISCV::GPRPairRegClass);

      // Check if the second pair's registers match the other lane of the
      // GPRPairs.
      if (SourceReg != TRI->getSubReg(SrcGPRPair, SecondPairIdx) ||
          DestReg != TRI->getSubReg(DestGPRPair, SecondPairIdx))
        return E;

      if (!ModifiedRegUnits.available(DestReg) ||
          !UsedRegUnits.available(DestReg) ||
          !ModifiedRegUnits.available(SourceReg))
        return E;

      return I;
    }
    // Update modified / used register units.
    LiveRegUnits::accumulateUsedDefed(MI, ModifiedRegUnits, UsedRegUnits, TRI);
  }
  return E;
}

MachineBasicBlock::iterator
RISCVMoveMerge::findMatchingInst(MachineBasicBlock::iterator &MBBI,
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

// Finds instructions, which could be represented as C.MV instructions and
// merged into CM.MVA01S or CM.MVSA01.
bool RISCVMoveMerge::mergeMoveSARegPair(MachineBasicBlock &MBB) {
  bool Modified = false;

  for (MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
       MBBI != E;) {
    // Check if the instruction can be compressed to C.MV instruction. If it
    // can, return Dest/Src register pair.
    auto RegPair = TII->isCopyInstrImpl(*MBBI);
    if (RegPair.has_value()) {
      bool MoveFromSToA = isCandidateToMergeMVA01S(*RegPair);
      bool IsEven = isEvenRegisterCopy(*RegPair);
      bool IsOdd = isOddRegisterCopy(*RegPair);
      if (!MoveFromSToA && !isCandidateToMergeMVSA01(*RegPair) && !IsEven &&
          !IsOdd) {
        ++MBBI;
        continue;
      }

      MachineBasicBlock::iterator Paired = E;
      if (ST->hasStdExtZcmp() || ST->hasVendorXqccmp()) {
        Paired = findMatchingInst(MBBI, MoveFromSToA, RegPair.value());
        if (Paired != E) {
          MBBI = mergePairedInsns(MBBI, Paired, MoveFromSToA);
          Modified = true;
          continue;
        }
      }
      if (IsEven != IsOdd) {
        Paired = findMatchingInstPair(MBBI, IsEven, RegPair.value());
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
    Modified |= mergeMoveSARegPair(MBB);
  return Modified;
}

/// createRISCVMoveMergePass - returns an instance of the
/// move merge pass.
FunctionPass *llvm::createRISCVMoveMergePass() { return new RISCVMoveMerge(); }
