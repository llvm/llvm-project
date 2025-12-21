//===------------------- AMDGPUEarlyRegisterSpilling.cpp  -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUEarlyRegisterSpilling.h"
#include "AMDGPU.h"
#include "AMDGPUNextUseAnalysis.h"
#include "GCNSubtarget.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-early-register-spilling"

STATISTIC(NumOfERSSpills, "Number of ERS spills");

static cl::opt<bool> EarlyRegisterSpilling("early-register-spilling",
                                           cl::init(true), cl::Hidden);

// TODO: Remove this flag.
static cl::opt<unsigned>
    VGPRMaxNums("max-vgprs", cl::init(0), cl::Hidden,
                cl::desc("The maximum number of VGPRs per wave."));

// TODO: Preserve SlotIndexes analysis in getAnalysisUsage()
void AMDGPUEarlyRegisterSpilling::updateIndexes(MachineInstr *MI) {
  if (Indexes->hasIndex(*MI))
    Indexes->removeMachineInstrFromMaps(*MI);
  Indexes->insertMachineInstrInMaps(*MI);
}

// TODO: Preserve LiveIntervals analysis in getAnalysisUsage()
void AMDGPUEarlyRegisterSpilling::updateLiveness(Register Reg) {
  if (LIS->hasInterval(Reg))
    LIS->removeInterval(Reg);
  LIS->createAndComputeVirtRegInterval(Reg);
}

void AMDGPUEarlyRegisterSpilling::updateLiveness(MachineInstr *MI) {
  for (auto &MO : MI->operands()) {
    if (!MO.isReg())
      continue;
    auto Reg = MO.getReg();
    if (!Reg.isVirtual())
      continue;
    if (LIS->hasInterval(Reg))
      LIS->removeInterval(Reg);
    LIS->createAndComputeVirtRegInterval(Reg);
  }
}

// We need this because it does not make sense to spill a def which has a use in
// a phi at the beginning of a basic block and it is defined a bit later.
bool AMDGPUEarlyRegisterSpilling::hasPHIUseInSameBB(Register Reg,
                                                    MachineBasicBlock *CurMBB) {
  for (auto &UseMI : MRI->use_nodbg_instructions(Reg))
    if (UseMI.isPHI() && UseMI.getParent() == CurMBB)
      return true;
  return false;
}

MachineInstr *
AMDGPUEarlyRegisterSpilling::emitRestore(Register DefRegToSpill,
                                         MachineInstr *DefRegUseInstr, int FI) {
  const TargetRegisterClass *RC = TRI->getRegClassForReg(*MRI, DefRegToSpill);
  Register NewReg = MRI->createVirtualRegister(RC);
  RestoredRegs.insert(NewReg);
  MachineBasicBlock *DefRegUseInstrBB = DefRegUseInstr->getParent();
  MachineInstr *Restore = nullptr;
  assert(DefRegUseInstr->getOpcode() != AMDGPU::PHI &&
         "We cannot emit a restore instruction before a phi node");
  TII->loadRegFromStackSlot(*DefRegUseInstrBB, DefRegUseInstr->getIterator(),
                            NewReg, FI, RC, 0);
  Restore = DefRegUseInstr->getPrevNode();
  DefRegUseInstr->substituteRegister(DefRegToSpill, NewReg, 0, *TRI);
  LIS->InsertMachineInstrInMaps(*Restore);
  LLVM_DEBUG(dbgs() << "Emit restore before use: " << *DefRegUseInstr);
  LLVM_DEBUG(dbgs() << "Restore instruction = " << *Restore);
  LLVM_DEBUG(dbgs() << "Restore block = " << Restore->getParent()->getName()
                    << "\n");
  LLVM_DEBUG(dbgs() << "Register to replace spilled register = "
                    << printReg(NewReg, TRI) << "\n");
  return Restore;
}

MachineInstr *
AMDGPUEarlyRegisterSpilling::emitRestore(Register DefRegToSpill,
                                         MachineBasicBlock &InsertBB, int FI) {
  const TargetRegisterClass *RC = TRI->getRegClassForReg(*MRI, DefRegToSpill);
  Register NewReg = MRI->createVirtualRegister(RC);
  RestoredRegs.insert(NewReg);
  auto It = InsertBB.getFirstTerminator();
  if (It == InsertBB.end())
    It = InsertBB.instr_end();
  TII->loadRegFromStackSlot(*&InsertBB, It, NewReg, FI, RC, 0);
  MachineInstr *Restore = &*(std::prev(It));
  LIS->InsertMachineInstrInMaps(*Restore);
  LLVM_DEBUG(dbgs() << "Restore instruction = " << *Restore);
  LLVM_DEBUG(dbgs() << "Emit restore at the end of basic block: = "
                    << Restore->getParent()->getName() << "\n");
  LLVM_DEBUG(dbgs() << "Register to replace spilled register = "
                    << printReg(NewReg, TRI) << "\n");
  return Restore;
}

// TODO: Tune this check to improve spilling.
bool AMDGPUEarlyRegisterSpilling::isLegalToSpill(Register CandidateReg) {
  assert(MRI->hasOneDef(CandidateReg) &&
         "The Register does not have one definition");
  MachineInstr *CandidateMI = MRI->getOneDef(CandidateReg)->getParent();
  return !hasPHIUseInSameBB(CandidateReg, CandidateMI->getParent()) &&
         !MRI->use_nodbg_empty(CandidateReg) &&
         !TII->isVGPRSpill(CandidateMI->getOpcode()) &&
         !isSpilledReg(CandidateReg) && !isRestoredReg(CandidateReg) &&
         !CandidateReg.isPhysical() && !TRI->isAGPR(*MRI, CandidateReg) &&
         !CandidateMI->isTerminator() && TRI->isVGPR(*MRI, CandidateReg);
}

SmallVector<Register> AMDGPUEarlyRegisterSpilling::getRegistersToSpill(
    MachineInstr *CurMI, GCNDownwardRPTracker &RPTracker) {
  MachineBasicBlock *CurMBB = CurMI->getParent();
  bool IsCurMIInLoop = MLI->getLoopFor(CurMI->getParent());
  SmallVector<std::pair<Register, double>> RegCandidates;
  DenseMap<Register, unsigned> RegNumOfUses;
  MachineLoop *OutermostLoop = nullptr;
  double LoopDistance = 0;
  LLVM_DEBUG(dbgs() << "===========================================\n");
  if (IsCurMIInLoop) {
    OutermostLoop = MLI->getLoopFor(CurMI->getParent())->getOutermostLoop();
    auto [DistanceFromHeaderToExitingLatch, ExitingLatch] =
        NUA.getLoopDistanceAndExitingLatch(OutermostLoop->getHeader());
    LoopDistance = DistanceFromHeaderToExitingLatch * LoopWeight;
  }

  for (auto [CandidateReg, Mask] : RPTracker.getLiveRegs()) {

    if (!isLegalToSpill(CandidateReg))
      continue;

    MachineInstr *CandidateMI = MRI->getOneDef(CandidateReg)->getParent();
    MachineBasicBlock *CandidateMIMBB = CandidateMI->getParent();
    // Sanity checks that the spilled register is defined before the high
    // register pressure point
    if ((CurMI != CandidateMI) && DT->dominates(CurMI, CandidateMI))
      continue;

    if ((CurMBB != CandidateMIMBB) && isReachable(CurMBB, CandidateMIMBB))
      continue;

    MachineLoop *CandidateLoop = MLI->getLoopFor(CandidateMI->getParent());
    bool IsLoopCandidate =
        IsCurMIInLoop &&
        (!CandidateLoop || (CandidateLoop && OutermostLoop &&
                            ((CandidateLoop != OutermostLoop) ||
                             !OutermostLoop->contains(CandidateLoop))));

    if (IsCurMIInLoop && !IsLoopCandidate)
      continue;

    SmallVector<MachineInstr *> Uses;
    for (auto &UseMI : MRI->use_nodbg_instructions(CandidateReg)) {
      MachineBasicBlock *UseMBB = UseMI.getParent();
      if (isReachable(CurMBB, UseMBB) ||
          (CurMBB == UseMBB && DT->dominates(CurMI, &UseMI)))
        Uses.push_back(&UseMI);
    }

    if (Uses.empty())
      continue;

    RegNumOfUses[CandidateReg] = Uses.size();
    auto NextUseDist = NUA.getNextUseDistance(CandidateReg, CurMI, Uses);

    if (!IsCurMIInLoop) {
      // If CurMI is not in a loop, then we collect the registers that we
      // can spill based on their next-use-distance from CurMI in
      // 'RegCandidates'.
      RegCandidates.push_back(std::make_pair(CandidateReg, *NextUseDist));
      LLVM_DEBUG(dbgs() << "Candidate register to spill = "
                        << printReg(CandidateReg, TRI) << " with distance = "
                        << format("%.1f", *NextUseDist) << "\n");
    } else if (IsLoopCandidate && (NextUseDist > LoopDistance)) {
      // Collect only the live-through values.
      RegCandidates.push_back(std::make_pair(CandidateReg, *NextUseDist));
      LLVM_DEBUG(dbgs() << "Candidate register to spill = "
                        << printReg(CandidateReg, TRI) << " with distance = "
                        << format("%.1f", *NextUseDist) << "\n");
    }
  }

  LLVM_DEBUG(dbgs() << "==========================================\n");
  if (RegCandidates.empty())
    return {};

  // Return the registers with the longest next-use distance.
  llvm::sort(RegCandidates, [&](auto &Pair1, auto &Pair2) {
    double NUA1 = Pair1.second;
    double NUA2 = Pair2.second;
    if (NUA1 != NUA2)
      return NUA1 > NUA2;
    Register Reg1 = Pair1.first;
    Register Reg2 = Pair2.first;
    unsigned NumOfUses1 = RegNumOfUses[Reg1];
    unsigned NumOfUses2 = RegNumOfUses[Reg2];
    if (NumOfUses1 == NumOfUses2)
      return Reg1 < Reg2;
    return NumOfUses1 < NumOfUses2;
  });

  SmallVector<Register> RegistersToSpill;
  RegistersToSpill.reserve(RegCandidates.size());
  for (auto P : RegCandidates)
    RegistersToSpill.push_back(P.first);

  return RegistersToSpill;
}

// Helper function for finding the incoming blocks that are related to
// DefRegToSpill
static SmallVector<MachineBasicBlock *>
getPhiBlocksOfSpillReg(MachineInstr *UseMI, Register DefRegToSpill) {
  assert(UseMI->isPHI() && "The use is not phi instruction");
  SmallVector<MachineBasicBlock *> Blocks;
  auto Ops = UseMI->operands();
  for (auto It = std::next(Ops.begin()), ItE = Ops.end(); It != ItE;
       It = std::next(It, 2)) {
    auto &RegMO = *It;
    if (RegMO.isUndef())
      continue;
    auto &MBBMO = *std::next(It);
    assert(RegMO.isReg() && "Expected register operand of PHI");
    assert(MBBMO.isMBB() && "Expected MBB operand of PHI");
    if (RegMO.getReg() == DefRegToSpill)
      Blocks.push_back(MBBMO.getMBB());
  }
  return Blocks;
}

// TODO: check if the common dominator of restores is profitable
bool AMDGPUEarlyRegisterSpilling::shouldEmitRestoreInCommonDominator(
    MachineBasicBlock *SpillBlock, MachineBasicBlock *CurMBB,
    MachineBasicBlock *CommonDominatorToRestore) {
  if (SpillBlock == CommonDominatorToRestore)
    return false;
  if (CurMBB == CommonDominatorToRestore)
    return false;
  if (DT->dominates(CommonDominatorToRestore, SpillBlock))
    return false;
  if (isReachable(CommonDominatorToRestore, SpillBlock))
    return false;
  if (!DT->dominates(SpillBlock, CommonDominatorToRestore))
    return false;
  if (MLI->getLoopFor(CommonDominatorToRestore))
    return false;
  return true;
}

/// Helper data structure for grouping together uses where the head of the group
/// dominates all the other uses in the group.
class DomGroup {
  SmallVector<MachineInstr *> Uses;
  SmallVector<MachineBasicBlock *> UseBlocks;
  MachineBasicBlock *CommonDominator = nullptr;
  bool Deleted = false;

public:
  DomGroup(MachineInstr *MI, MachineBasicBlock *RestoreBlock) {
    Uses.push_back(MI);
    UseBlocks.push_back(RestoreBlock);
  }
  MachineInstr *getHead() const { return Uses.front(); }
  bool isDeleted() const { return Deleted; }
  void merge(DomGroup &Other) {
    for (auto *MI : Other.Uses)
      Uses.push_back(MI);

    for (auto *UseMBB : Other.UseBlocks)
      UseBlocks.push_back(UseMBB);

    Other.Deleted = true;
  }
  const auto &getUses() const { return Uses; }
  const auto &getUseBlocks() const { return UseBlocks; }
  size_t size() const { return Uses.size(); }
  void setCommonDominator(MachineBasicBlock *CD) { CommonDominator = CD; }
  MachineBasicBlock *getCommonDominator() const { return CommonDominator; }
  bool hasCommonDominator() const { return CommonDominator != nullptr; }
  MachineBasicBlock *getRestoreBlock() const { return UseBlocks.front(); }
};

void AMDGPUEarlyRegisterSpilling::emitRestoreInstrsForDominatedUses(
    Register DefRegToSpill, MachineInstr *SpillInstruction, MachineInstr *CurMI,
    SetVectorType &DominatedUses, SmallVector<MachineInstr *> &RestoreInstrs,
    SmallVector<MachineInstr *> &RestoreUses, int FI) {
  MachineBasicBlock *SpillBlock = SpillInstruction->getParent();
  MachineBasicBlock *CurMBB = CurMI->getParent();
  MachineLoop *SpillLoop = MLI->getLoopFor(SpillBlock);
  assert(!SpillLoop && "There should not be a spill loop.");

  std::vector<DomGroup> Groups;
  for (auto *Use : DominatedUses) {
    MachineLoop *UseLoop = MLI->getLoopFor(Use->getParent());
    if (UseLoop) {
      // If a use is in a loop then the restore instruction is emitted in the
      // outermost loop's preheader.
      MachineLoop *OutermostLoop = UseLoop->getOutermostLoop();
      MachineBasicBlock *OutermostLoopPreheader =
          OutermostLoop->getLoopPreheader();
      Groups.emplace_back(Use, OutermostLoopPreheader);
    } else if (Use->isPHI()) {
      // In case of phi nodes, the restore instructions are emitted at the
      // bottom of the incoming blocks.
      for (MachineBasicBlock *PhiOpMBB :
           getPhiBlocksOfSpillReg(Use, DefRegToSpill)) {
        Groups.emplace_back(Use, PhiOpMBB);
      }
    } else {
      // Emit restore before Use.
      Groups.emplace_back(Use, Use->getParent());
    }
  }

  // Our goal is to emit as few restores as possible by avoiding emitting
  // restore instructions if an earlier restore can be reused.
  //
  // Create groups of instructions where the group head dominates the rest in
  // the group. In addition, we check if we can find an eligible common
  // dominator where we can emit the restore instruction.
  //
  // In the following example, there are two groups. The first group consists of
  // the uses in BB3 and BB5 and the second group consists of the uses in BB4
  // and BB6. The head of the first group is the use in BB3 and the head of the
  // second group is the use in BB4.
  //
  //                    BB1
  //                      r1 = ...
  //                     |
  //                    BB2
  //                     spill r1 <-- high register pressure block
  //                   /    \
  //                BB3     BB4
  //      r2 = restore r1  r3 = restore r1
  //             ... = r2  ... = r3
  //                 |        |
  //                BB5      BB6
  //             ... = r2  ... = r3
  //
  // In the following example, we emit the restore instruction in the common
  // dominator of the two uses in BB4 and BB5.
  //                    BB1
  //                      r1 = ...
  //                     |
  //                    BB2
  //                     spill r1 <-- high register pressure block
  //                     |
  //                    BB3
  //               r2 = restore r1
  //                   /   \
  //                 BB4   BB5
  //            ... = r2   ... = r2
  //
  for (unsigned Idx1 = 0, E = Groups.size(); Idx1 != E; ++Idx1) {
    auto &G1 = Groups[Idx1];
    if (G1.isDeleted())
      continue;
    for (unsigned Idx2 = Idx1 + 1; Idx2 < E; ++Idx2) {
      auto &G2 = Groups[Idx2];
      if (G2.isDeleted())
        continue;

      MachineInstr *Head1 = G1.getHead();
      MachineInstr *Head2 = G2.getHead();
      MachineBasicBlock *RestoreBlock1 = G1.getRestoreBlock();
      MachineBasicBlock *RestoreBlock2 = G2.getRestoreBlock();
      SmallVector<MachineBasicBlock *> UseBlocks;
      for (auto *Block : G1.getUseBlocks())
        UseBlocks.push_back(Block);

      for (auto *Block : G2.getUseBlocks())
        UseBlocks.push_back(Block);

      MachineBasicBlock *CommonDom = DT->findNearestCommonDominator(
          make_range(UseBlocks.begin(), UseBlocks.end()));

      if ((RestoreBlock1 != RestoreBlock2) &&
          shouldEmitRestoreInCommonDominator(SpillBlock, CurMBB, CommonDom)) {
        // Set a common dominator if the two restore blocks are different.
        G1.merge(G2);
        G1.setCommonDominator(CommonDom);
      } else if (DT->dominates(Head1, Head2) && !G1.getCommonDominator() &&
                 !G2.getCommonDominator()) {
        // If there is no common dominator and one Head dominates the other,
        // then we can merge the two groups.
        G1.merge(G2);
      }
    }
  }

  // For each group emit one restore for the group header in the parent block of
  // the group header or the common dominator. The rest of the uses in the group
  // will reuse the value loaded by the restore of the header.
  for (auto &G1 : Groups) {
    if (G1.isDeleted())
      continue;
    MachineInstr *Head = G1.getHead();
    MachineBasicBlock *HeadMBB = G1.getRestoreBlock();
    MachineInstr *Restore = nullptr;
    if (G1.hasCommonDominator()) {
      MachineBasicBlock *CommonDominator = G1.getCommonDominator();
      MachineInstr *UseInCommonDominator = nullptr;
      for (auto *U : G1.getUses()) {
        if (U->getParent() == CommonDominator) {
          if (UseInCommonDominator) {
            if (DT->dominates(U, UseInCommonDominator))
              UseInCommonDominator = U;
          } else {
            UseInCommonDominator = U;
          }
        }
      }
      if (UseInCommonDominator) {
        Restore = emitRestore(DefRegToSpill, UseInCommonDominator, FI);
        Head = UseInCommonDominator;
        HeadMBB = CommonDominator;
      } else {
        Restore = emitRestore(DefRegToSpill, *CommonDominator, FI);
        Head->substituteRegister(DefRegToSpill, Restore->getOperand(0).getReg(),
                                 0, *TRI);
      }
    } else if (Head->isPHI()) {
      Restore = emitRestore(DefRegToSpill, *HeadMBB, FI);
      Head->substituteRegister(DefRegToSpill, Restore->getOperand(0).getReg(),
                               0, *TRI);
    } else if (MLI->getLoopFor(Head->getParent())) {
      Restore = emitRestore(DefRegToSpill, *HeadMBB, FI);
      Head->substituteRegister(DefRegToSpill, Restore->getOperand(0).getReg(),
                               0, *TRI);
    } else {
      Restore = emitRestore(DefRegToSpill, Head, FI);
    }
    RestoreInstrs.push_back(Restore);
    RestoreUses.push_back(Head);

    // Update the rest of the uses in the group to reuse the value restored by
    // the head of the group.
    for (auto *U : G1.getUses()) {
      assert(U != SpillInstruction);
      if (U == Head)
        continue;

      U->substituteRegister(DefRegToSpill, Restore->getOperand(0).getReg(), 0,
                            *TRI);
      RestoreUses.push_back(U);
      LLVM_DEBUG(dbgs() << "Updated use: " << *U);
      LLVM_DEBUG(dbgs() << "With register = "
                        << printReg(Restore->getOperand(0).getReg(), TRI)
                        << "\n");
    }
  }
}

SetVectorType AMDGPUEarlyRegisterSpilling::collectUsesThatNeedRestoreInstrs(
    Register DefRegToSpill, MachineInstr *SpillInstruction,
    const SetVectorType &UnreachableUses) {
  SetVectorType DominatedUses;
  MachineBasicBlock *SpillBlock = SpillInstruction->getParent();
  for (MachineInstr &U : MRI->use_nodbg_instructions(DefRegToSpill)) {
    if (&U == SpillInstruction)
      continue;

    if (UnreachableUses.contains(&U))
      continue;

    if (U.isPHI()) {
      for (auto *PhiOpMBB : getPhiBlocksOfSpillReg(&U, DefRegToSpill)) {
        if (DT->dominates(SpillBlock, PhiOpMBB)) {
          DominatedUses.insert(&U);
        }
      }
    } else if (DT->dominates(SpillInstruction, &U)) {
      DominatedUses.insert(&U);
    }
  }
  return DominatedUses;
}

void AMDGPUEarlyRegisterSpilling::emitRestores(
    Register DefRegToSpill, MachineInstr *CurMI, MachineInstr *SpillInstruction,
    const SetVectorType &UnreachableUses, const TargetRegisterClass *RC,
    int FI) {
  assert(MRI->hasOneDef(DefRegToSpill) &&
         "The Register does not have one definition");
  MachineInstr *InstrOfDefRegToSpill =
      MRI->getOneDef(DefRegToSpill)->getParent();

  // Collect the uses that are dominated by SpillInstruction
  SetVectorType DominatedUses = collectUsesThatNeedRestoreInstrs(
      DefRegToSpill, SpillInstruction, UnreachableUses);

  SmallVector<MachineInstr *> RestoreInstrs;
  SmallVector<MachineInstr *> RestoreUses;
  emitRestoreInstrsForDominatedUses(DefRegToSpill, SpillInstruction, CurMI,
                                    DominatedUses, RestoreInstrs, RestoreUses,
                                    FI);

  // Update the live interval analysis.
  updateIndexes(InstrOfDefRegToSpill);
  updateIndexes(SpillInstruction);
  updateLiveness(InstrOfDefRegToSpill);
  updateLiveness(SpillInstruction);

  if (InstrOfDefRegToSpill != CurMI) {
    updateIndexes(CurMI);
    updateLiveness(CurMI);
  }

  for (auto *Use : RestoreInstrs) {
    updateIndexes(Use);
    updateLiveness(Use);
  }

  for (auto *Use : RestoreUses) {
    updateIndexes(Use);
    updateLiveness(Use);
  }
}

// We have to collect the unreachable uses before we emit the spill instruction.
// This is due to the fact that some unreachable uses might become reachable if
// we spill in common dominator.
std::pair<SetVectorType, SetVectorType>
AMDGPUEarlyRegisterSpilling::collectNonDominatedReachableAndUnreachableUses(
    MachineBasicBlock *SpillBlock, Register DefRegToSpill,
    MachineInstr *CurMI) {
  // The reachable uses are the ones that can be reached by the SpillBlock.
  SetVectorType NonDominatedReachableUses;
  // The non-dominated uses are the uses that cannot be reached by the
  // SpillBlock.
  SetVectorType UnreachableUses;
  for (MachineInstr &U : MRI->use_nodbg_instructions(DefRegToSpill)) {
    if (U.isPHI()) {
      for (auto *PhiOpMBB : getPhiBlocksOfSpillReg(&U, DefRegToSpill)) {
        if (DT->dominates(SpillBlock, PhiOpMBB))
          continue;
        if (isReachable(SpillBlock, PhiOpMBB)) {
          if (!DT->dominates(SpillBlock, PhiOpMBB))
            NonDominatedReachableUses.insert(&U);
        } else if (PhiOpMBB != SpillBlock)
          UnreachableUses.insert(&U);
      }
    } else {
      MachineBasicBlock *UMBB = U.getParent();
      if (DT->dominates(CurMI, &U))
        continue;

      if (isReachable(SpillBlock, UMBB)) {
        if (!DT->dominates(SpillBlock, UMBB))
          NonDominatedReachableUses.insert(&U);
      } else
        UnreachableUses.insert(&U);
    }
  }
  return {NonDominatedReachableUses, UnreachableUses};
}

// Find the common dominator of the reachable uses and the block that we
// intend to spill(SpillBlock).
MachineBasicBlock *AMDGPUEarlyRegisterSpilling::findCommonDominatorToSpill(
    MachineBasicBlock *SpillBlock, Register DefRegToSpill,
    const SetVectorType &NonDominatedReachableUses) {
  SmallPtrSet<MachineBasicBlock *, 2> Blocks;
  for (auto *RU : NonDominatedReachableUses) {
    if (RU->isPHI()) {
      for (auto *PhiOpMBB : getPhiBlocksOfSpillReg(RU, DefRegToSpill))
        Blocks.insert(PhiOpMBB);
    } else
      Blocks.insert(RU->getParent());
  }

  Blocks.insert(SpillBlock);
  MachineBasicBlock *CommonDominatorToSpill =
      DT->findNearestCommonDominator(make_range(Blocks.begin(), Blocks.end()));

  return CommonDominatorToSpill;
}

std::pair<MachineBasicBlock *, MachineBasicBlock::iterator>
AMDGPUEarlyRegisterSpilling::getWhereToSpill(MachineInstr *CurMI,
                                             Register DefRegToSpill) {
  assert(MRI->hasOneDef(DefRegToSpill) &&
         "The Register does not have one definition");
  MachineInstr *InstrOfDefRegToSpill =
      MRI->getOneDef(DefRegToSpill)->getParent();
  MachineBasicBlock *DefRegMBB = InstrOfDefRegToSpill->getParent();
  MachineBasicBlock *CurMBB = CurMI->getParent();
  MachineLoop *DefInstrLoop = MLI->getLoopFor(DefRegMBB);
  MachineLoop *CurLoop = MLI->getLoopFor(CurMI->getParent());
  // We do not spill inside the loop nest because of the spill overhead. So,
  // we only need to know about the outermost loop.
  if (CurLoop)
    CurLoop = CurLoop->getOutermostLoop();

  MachineBasicBlock *SpillBlock = nullptr;
  MachineBasicBlock::iterator WhereToSpill;
  // case 1:
  // - the register we are about to spill (DefRegToSpill) is defined in loop
  // - the high register pressure (CurMI) is outside the loop
  // - we emit the spill instruction in one of the exit blocks of the loop
  // TODO: improve spilling in loops
  if ((DefInstrLoop && !CurLoop) ||
      (DefInstrLoop && CurLoop &&
       ((DefInstrLoop != CurLoop) || (!DefInstrLoop->contains(CurLoop) &&
                                      !CurLoop->contains(DefInstrLoop))))) {
    SmallVector<MachineBasicBlock *> ExitBlocks;
    MachineLoop *OutermostLoop = DefInstrLoop->getOutermostLoop();
    OutermostLoop->getUniqueExitBlocks(ExitBlocks);
    assert(ExitBlocks.size() == 1 && "There should be only one exit basic "
                                     "block after CFG structurization");
    MachineBasicBlock *ExitBB = ExitBlocks.back();
    if (!DT->dominates(ExitBB, CurMBB))
      return {};
    if (ExitBB == CurMBB) {
      WhereToSpill = CurMI->getIterator();
      SpillBlock = ExitBB;
    } else {
      WhereToSpill = ExitBB->getFirstTerminator();
      if (WhereToSpill == ExitBB->end())
        WhereToSpill = ExitBB->instr_end();
      SpillBlock = ExitBB;
    }
  }
  // case 2:
  // - the register we are about to spill is outside the loop
  // - the high register pressure instruction (CurMI) is inside the loop
  // - we emit the spill instruction in the loop preheader
  else if (!DefInstrLoop && CurLoop) {
    MachineBasicBlock *CurLoopPreheader = CurLoop->getLoopPreheader();
    assert(CurLoopPreheader && "There is not loop preheader");
    WhereToSpill = CurLoopPreheader->getFirstTerminator();
    if (WhereToSpill == CurLoopPreheader->end())
      WhereToSpill = CurLoopPreheader->back();
    SpillBlock = CurLoopPreheader;
  }
  // case 3:
  // - the high register pressure instruction is a PHI node
  // - we emit the spill instruction before the first non-PHI instruction
  else if (CurMI->isPHI()) {
    WhereToSpill = CurMBB->getFirstNonPHI();
    SpillBlock = CurMBB;
  }
  // case 4:
  // - the high register pressure instruction is also the instruction that
  //   defines the register we are about to spill
  // - we emit the spill instruction after the high reg pressure instr
  else if (CurMI == InstrOfDefRegToSpill) {
    WhereToSpill = std::next(CurMI->getIterator());
    SpillBlock = CurMBB;
  }
  // case 5:
  // - this is the general case. We spill just before the instruction where
  // we detect high register pressure.
  else {
    WhereToSpill = CurMI->getIterator();
    SpillBlock = CurMBB;
  }
  return {SpillBlock, WhereToSpill};
}

void AMDGPUEarlyRegisterSpilling::spill(MachineInstr *CurMI,
                                        GCNDownwardRPTracker &RPTracker,
                                        unsigned NumOfSpills) {
  // CurMI indicates the point of the code where there is high register
  // pressure.
  unsigned SpillCnt = 0;
  for (Register DefRegToSpill : getRegistersToSpill(CurMI, RPTracker)) {
    if (SpillCnt >= NumOfSpills)
      break;

    const TargetRegisterClass *RC = TRI->getRegClassForReg(*MRI, DefRegToSpill);
    unsigned Size = TRI->getSpillSize(*RC);
    Align Alignment = TRI->getSpillAlign(*RC);
    int FI = FrameInfo->CreateSpillStackObject(Size, Alignment);

    // First we find where we should emit the spill instruction.
    MachineInstr *SpillInstruction = nullptr;
    MachineBasicBlock *SpillBlock = nullptr;
    MachineBasicBlock::iterator WhereToSpill;
    std::tie(SpillBlock, WhereToSpill) = getWhereToSpill(CurMI, DefRegToSpill);
    if (SpillBlock == nullptr)
      continue;
    // The next step is to check if there are any uses which are reachable
    // from the SpillBlock. In this case, we have to emit the spill in the
    // common dominator of the SpillBlock and the blocks of the reachable
    // uses.
    SetVectorType NonDominatedReachableUses;
    SetVectorType UnreachableUses;
    std::tie(NonDominatedReachableUses, UnreachableUses) =
        collectNonDominatedReachableAndUnreachableUses(SpillBlock,
                                                       DefRegToSpill, CurMI);
    MachineBasicBlock *CommonDominatorToSpill = nullptr;
    if (!NonDominatedReachableUses.empty())
      CommonDominatorToSpill = findCommonDominatorToSpill(
          SpillBlock, DefRegToSpill, NonDominatedReachableUses);
    if (CommonDominatorToSpill && CommonDominatorToSpill != SpillBlock) {
      SpillBlock = CommonDominatorToSpill;
      WhereToSpill = SpillBlock->getFirstTerminator();
      if (WhereToSpill == SpillBlock->end())
        WhereToSpill = SpillBlock->instr_end();
    }
    // Emit the spill instruction.
    TII->storeRegToStackSlot(*SpillBlock, WhereToSpill, DefRegToSpill,
                             true, /* kill */
                             FI, RC, 0);
    SpillInstruction = &*(std::prev(WhereToSpill));
    // Maintain live intervals.
    LIS->InsertMachineInstrInMaps(*SpillInstruction);

    SpilledRegs.insert(DefRegToSpill);
    NumOfERSSpills++;
    SpillCnt++;

    assert(SpillInstruction && "There is not a spill instruction");
    LLVM_DEBUG(dbgs() << "High register pressure point = " << *CurMI);
    LLVM_DEBUG(dbgs() << "Register to spill = " << printReg(DefRegToSpill, TRI)
                      << "\n");
    LLVM_DEBUG(dbgs() << "Spill instruction = " << *SpillInstruction);
    LLVM_DEBUG(dbgs() << "Spill block = "
                      << SpillInstruction->getParent()->getName() << "\n");

    // Find the restore locations, emit the restore instructions and maintain
    // SSA when needed.
    emitRestores(DefRegToSpill, CurMI, SpillInstruction, UnreachableUses, RC,
                 FI);
  }
  // Reset the tracker because it has already read the next instruction which
  // we might have modified by emitting a spill or restore instruction.
  RPTracker.reset(*CurMI);
  RPTracker.advance();
}

GCNRegPressure
AMDGPUEarlyRegisterSpilling::getMaxPressure(const MachineFunction &MF) {
  GCNRegPressure MaxPressure;
  GCNUpwardRPTracker RPTracker(*LIS);
  for (auto &MBB : MF) {
    GCNRegPressure BBMaxPressure;

    if (!MBB.empty()) {
      RPTracker.reset(MBB.instr_back());
      for (auto &MI : reverse(MBB))
        RPTracker.recede(MI);

      BBMaxPressure = RPTracker.getMaxPressureAndReset();
    }
    MaxPressure = max(BBMaxPressure, MaxPressure);
  }
  return MaxPressure;
}

bool AMDGPUEarlyRegisterSpilling::runOnMachineFunction(MachineFunction &MF) {

  if (skipFunction(MF.getFunction()))
    return false;

  if (!EarlyRegisterSpilling)
    return false;

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  TRI = &TII->getRegisterInfo();
  MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  MRI = &MF.getRegInfo();
  MFI = MF.getInfo<SIMachineFunctionInfo>();
  FrameInfo = &MF.getFrameInfo();
  LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  Indexes = &getAnalysis<SlotIndexesWrapperPass>().getSI();
  DT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  NUA.run(MF, MLI);

  unsigned VgprNum = getMaxPressure(MF).getVGPRNum(ST.hasGFX90AInsts());
  unsigned Occupancy =
      ST.getOccupancyWithNumVGPRs(VgprNum, ST.getDynamicVGPRBlockSize());
  unsigned MaxVGPRs =
      VGPRMaxNums > 0
          ? VGPRMaxNums
          : std::min(ST.getMaxNumVGPRs(Occupancy, ST.getDynamicVGPRBlockSize()),
                     ST.getMaxNumVGPRs(MF));

  LLVM_DEBUG(dbgs() << "===========================================\n");
  LLVM_DEBUG(dbgs() << "Early Register Spilling\n");
  LLVM_DEBUG(dbgs() << "===========================================\n");
  LLVM_DEBUG(dbgs() << MF.getName() << "\n");
  LLVM_DEBUG(dbgs() << "MaxVGPRs = " << MaxVGPRs << "\n");

  GCNDownwardRPTracker RPTracker(*LIS);
  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);

  for (MachineBasicBlock *MBB : RPOT) {
    if (MBB->empty())
      continue;

    // Initialize the Register Pressure Tracker at the beginning of the
    // block.
    RPTracker.reset(*MBB->begin());
    RPTracker.advance();

    // Iterate over the instructions of MBB and check if the live registers
    // are more than the available registers.
    for (auto It = MBB->begin(), ItE = MBB->end(); It != ItE; ++It) {
      MachineInstr *MI = &*It;

      if (MI->isDebugInstr())
        continue;

      if (!TII->isVGPRSpill(MI->getOpcode()) && !MI->isBranch()) {

        const MachineInstr *LastTrackedMI = RPTracker.getLastTrackedMI();
        assert(MI == LastTrackedMI && "The tracker and the loop iteration "
                                      "should visit the same instruction.");
        unsigned VGPRLiveRegs = RPTracker.getPressure().getVGPRNum(false);
        // Spill if the live VGPR registers are more than the available
        // VGPRs.
        if (VGPRLiveRegs > MaxVGPRs) {
          unsigned NumOfSpills = VGPRLiveRegs - MaxVGPRs;
          spill(MI, RPTracker, NumOfSpills);
        }
      }

      // Move the tracker to the next instruction.
      // If we have reached the bottom of a basic block, then we have to
      // initialize the tracker at the beginning of the next basic block.
      if (MI == &MBB->back())
        continue;

      // Phi nodes might include registers that are defined later in the
      // code. Hence, we have to initialize the tracker again.
      if (MI->getOpcode() == AMDGPU::PHI) {
        RPTracker.reset(*MI->getNextNode());
      }
      RPTracker.advance();
    }
  }

  clearTables();
  return true;
}

char AMDGPUEarlyRegisterSpilling::ID = 0;

INITIALIZE_PASS_BEGIN(AMDGPUEarlyRegisterSpilling, DEBUG_TYPE,
                      "Early Register Spilling", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(SlotIndexesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_END(AMDGPUEarlyRegisterSpilling, DEBUG_TYPE,
                    "Early Register Spilling", false, false)

char &llvm::AMDGPUEarlyRegisterSpillingID = AMDGPUEarlyRegisterSpilling::ID;

FunctionPass *llvm::createAMDGPUEarlyRegisterSpillingPass() {
  return new AMDGPUEarlyRegisterSpilling();
}
