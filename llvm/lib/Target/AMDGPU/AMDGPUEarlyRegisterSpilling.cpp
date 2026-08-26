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
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/InitializePasses.h"
#include <algorithm>
#include <cmath>
#include <limits>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-early-register-spilling"

STATISTIC(NumOfERSSpills, "Number of ERS spills");

static cl::opt<bool> EarlyRegisterSpilling("early-register-spilling",
                                           cl::init(true), cl::Hidden);

// TODO: Remove this flag.
static cl::opt<unsigned>
    VGPRMaxNums("max-vgprs", cl::Hidden,
                cl::desc("The maximum number of VGPRs per wave."),
                cl::init(96));

/// Helper functions to update the live interval analysis which is used by
/// the Register Pressure Tracker.
static void updateIndexes(MachineInstr *MI, SlotIndexes *Indexes) {
  if (Indexes->hasIndex(*MI))
    Indexes->removeMachineInstrFromMaps(*MI);
  Indexes->insertMachineInstrInMaps(*MI);
}

static void updateLiveness(MachineInstr *MI, LiveIntervals *LIS) {
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

MachineInstr *
AMDGPUEarlyRegisterSpilling::emitRestore(Register CandidateReg,
                                         MachineInstr *DefRegUseInstr, int FI) {
  const TargetRegisterClass *RC = TRI->getRegClassForReg(*MRI, CandidateReg);
  Register NewReg = MRI->createVirtualRegister(RC);
  MachineBasicBlock *DefRegUseInstrBB = DefRegUseInstr->getParent();
  MachineInstr *Restore = nullptr;
  assert(DefRegUseInstr->getOpcode() != AMDGPU::PHI &&
         "We cannot emit a restore instruction before a phi node");
  TII->loadRegFromStackSlot(*DefRegUseInstrBB, DefRegUseInstr->getIterator(),
                            NewReg, FI, RC, 0);
  Restore = DefRegUseInstr->getPrevNode();
  DefRegUseInstr->substituteRegister(CandidateReg, NewReg, 0, *TRI);
  LIS->InsertMachineInstrInMaps(*Restore);
  MachineBasicBlock *RestoreBlock = Restore->getParent();
  LLVM_DEBUG(dbgs() << "Restore instruction = " << *Restore);
  LLVM_DEBUG(dbgs() << "Register to replace = " << printReg(NewReg, TRI)
                    << "\n");
  LLVM_DEBUG(dbgs() << "Restore block = " << "bb." << RestoreBlock->getNumber()
                    << "\n");
  if (MLI->getLoopFor(RestoreBlock)) {
    LLVM_DEBUG(dbgs() << "The restore block is in a loop\n");
  } else {
    LLVM_DEBUG(dbgs() << "The restore block is not in a loop\n");
  }

  return Restore;
}

MachineInstr *
AMDGPUEarlyRegisterSpilling::emitRestore(Register CandidateReg,
                                         MachineBasicBlock &InsertBB, int FI) {
  const TargetRegisterClass *RC = TRI->getRegClassForReg(*MRI, CandidateReg);
  Register NewReg = MRI->createVirtualRegister(RC);
  auto It = InsertBB.getFirstTerminator();
  if (It == InsertBB.end())
    It = InsertBB.instr_end();
  TII->loadRegFromStackSlot(*&InsertBB, It, NewReg, FI, RC, 0);
  MachineInstr *Restore = &*(std::prev(It));
  LIS->InsertMachineInstrInMaps(*Restore);
  LLVM_DEBUG(dbgs() << "Restore instruction = " << *Restore);
  LLVM_DEBUG(dbgs() << "Register to replace = " << printReg(NewReg, TRI)
                    << "\n");
  LLVM_DEBUG(dbgs() << "Emit restore at the end of a basic block.\n");
  LLVM_DEBUG(dbgs() << "Restore block = " << "bb." << InsertBB.getNumber()
                    << "\n");
  if (MLI->getLoopFor(&InsertBB)) {
    LLVM_DEBUG(dbgs() << "The restore block is in a loop\n");
  } else {
    LLVM_DEBUG(dbgs() << "The restore block is not in a loop\n");
  }

  return Restore;
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

// TODO: Tune this check to improve spilling.
bool AMDGPUEarlyRegisterSpilling::isLegalCandidate(Register CandidateReg) {
  assert(MRI->hasOneDef(CandidateReg) &&
         "The Register does not have one definition");
  MachineInstr *CandidateMI = MRI->getOneDef(CandidateReg)->getParent();
  return !hasPHIUseInSameBB(CandidateReg, CandidateMI->getParent()) &&
         !MRI->use_nodbg_empty(CandidateReg) && !isSpilledReg(CandidateReg) &&
         !isRestoredReg(CandidateReg) && !CandidateReg.isPhysical() &&
         !TRI->isAGPR(*MRI, CandidateReg) && !CandidateMI->isTerminator() &&
         TRI->isVGPR(*MRI, CandidateReg);
}

SmallVector<RegisterSpillCandidate>
AMDGPUEarlyRegisterSpilling::getCandidates(MachineInstr *CurMI,
                                           GCNDownwardRPTracker &RPTracker) {
  MachineBasicBlock *CurMBB = CurMI->getParent();
  MachineLoop *CurLoop = MLI->getLoopFor(CurMBB);
  SmallVector<RegisterSpillCandidate> RegCandidates;
  DenseMap<Register, unsigned> RegNumOfUses;
  MachineLoop *OutermostLoopOfCurLoop = nullptr;
  if (CurLoop)
    OutermostLoopOfCurLoop = CurLoop->getOutermostLoop();
  unsigned CandidateCnt = 0;
  for (auto [CandidateReg, Mask] : RPTracker.getLiveRegs()) {

    MachineInstr *CandidateMI = MRI->getOneDef(CandidateReg)->getParent();

    if (!isLegalCandidate(CandidateReg))
      continue;

    if (CandidateMI == CurMI)
      continue;

    MachineBasicBlock *CandidateMIMBB = CandidateMI->getParent();
    assert(!(CurMI != CandidateMI && DT->dominates(CurMI, CandidateMI) &&
             CurMBB != CandidateMIMBB &&
             NUA->isReachable(CurMBB, CandidateMIMBB)) &&
           "We expect the candidate to be defined before the high register "
           "pressure point");

    // In the following check, we reject candidates which are defined in the
    // same loop nest.
    MachineLoop *CandidateLoop = MLI->getLoopFor(CandidateMIMBB);
    bool AreIndependentLoops =
        CandidateLoop && OutermostLoopOfCurLoop &&
        (!CandidateLoop->contains(OutermostLoopOfCurLoop) &&
         !CandidateLoop->contains(CurLoop) &&
         !OutermostLoopOfCurLoop->contains(CandidateLoop));

    // If the high register pressure point is inside a loop, then we spill loop
    // live-ins and loop live-thoughs.
    // We spill values that are defined inside a loop in the exit block of the
    // loop when the high register pressure point is outside of the loop.
    if (CandidateLoop && OutermostLoopOfCurLoop && !AreIndependentLoops)
      continue;

    SmallVector<const MachineOperand *> UsesForNextUseDistCalculation;
    if (CurLoop) {
      MachineBasicBlock *PreHeader = OutermostLoopOfCurLoop->getLoopPreheader();
      // Find the loop uses of CandidateReg.
      MachineBasicBlock::iterator LastIt = PreHeader->getFirstTerminator();
      if (LastIt == PreHeader->end())
        LastIt = PreHeader->instr_back();
      MachineInstr *LastMIPreHeader = &*(LastIt);
      NUA->getReachableUses(CandidateReg, Mask, *LastMIPreHeader,
                            UsesForNextUseDistCalculation);

      // TODO: Split live-ranges for these cases.
      if (llvm::any_of(
              UsesForNextUseDistCalculation, [&](const MachineOperand *UseMO) {
                const MachineInstr *UseMI = UseMO->getParent();
                const MachineBasicBlock *UseMBB = UseMI->getParent();
                MachineLoop *UseLoop = MLI->getLoopFor(UseMBB);
                if (UseLoop && OutermostLoopOfCurLoop->contains(UseLoop) &&
                    (UseLoop->getLoopDepth() > 1))
                  return true;
                if (UseLoop && (UseLoop->getLoopDepth() > 1))
                  return true;
                return false;
              }))
        continue;

      if (UsesForNextUseDistCalculation.empty())
        continue;

      RegNumOfUses[CandidateReg] = UsesForNextUseDistCalculation.size();
      LLVM_DEBUG(CandidateCnt++);
      // All the live-ins are live through the backedge.
      auto NextUseDist = NUA->getShortestDistance(
          CandidateReg, *CandidateMI, UsesForNextUseDistCalculation);
      RegCandidates.push_back({CandidateReg, NextUseDist, Mask});
      LLVM_DEBUG({
        dbgs() << CandidateCnt
               << ": Candidate register = " << printReg(CandidateReg, TRI)
               << " with distance = " << NextUseDist.fmt() << "\n";
      });
    } else {
      NUA->getReachableUses(CandidateReg, Mask, *CurMI,
                            UsesForNextUseDistCalculation);
      if (UsesForNextUseDistCalculation.empty())
        continue;

      RegNumOfUses[CandidateReg] = UsesForNextUseDistCalculation.size();

      LLVM_DEBUG(CandidateCnt++);
      // Calculate the next-use distance for the spill candidates and add them
      // in 'RegCandidates'.
      auto NextUseDist = NUA->getShortestDistance(
          CandidateReg, *CurMI, UsesForNextUseDistCalculation);
      RegCandidates.push_back({CandidateReg, NextUseDist, Mask});
      LLVM_DEBUG({
        dbgs() << CandidateCnt
               << ": Candidate register = " << printReg(CandidateReg, TRI)
               << " with distance = " << NextUseDist.fmt() << "\n";
      });
    }
  }

  LLVM_DEBUG(dbgs() << "==========================================\n");
  if (RegCandidates.empty())
    return {};

  // Return the registers with the longest next-use distance.
  // TODO: Parametrize the next-use distance in order to take into consideration
  // the number of uses, the uses inside a loop etc.
  llvm::sort(RegCandidates, [&](const RegisterSpillCandidate &C1,
                                const RegisterSpillCandidate &C2) {
    if (C1.Dist != C2.Dist)
      return C1.Dist > C2.Dist;
    unsigned NumOfUses1 = RegNumOfUses[C1.Reg];
    unsigned NumOfUses2 = RegNumOfUses[C2.Reg];
    if (NumOfUses1 == NumOfUses2)
      return C1.Reg < C2.Reg;

    return NumOfUses1 < NumOfUses2;
  });

  SmallVector<RegisterSpillCandidate> FinalCandidates;
  FinalCandidates.reserve(RegCandidates.size());
  for (const auto &Candidate : RegCandidates)
    FinalCandidates.push_back(Candidate);

  return FinalCandidates;
}

// Helper function for finding the incoming blocks that are related to
// CandidateReg
static SmallVector<MachineBasicBlock *>
getPhiBlocksOfSpillReg(MachineInstr *UseMI, Register CandidateReg) {
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
    if (RegMO.getReg() == CandidateReg)
      Blocks.push_back(MBBMO.getMBB());
  }
  return Blocks;
}

void AMDGPUEarlyRegisterSpilling::spill(MachineInstr *CurMI,
                                        GCNDownwardRPTracker &RPTracker,
                                        unsigned NumOfSpills) {

  MachineBasicBlock *CurMBB = CurMI->getParent();
  MachineLoop *CurLoop = MLI->getLoopFor(CurMBB);
  MachineLoop *OutermostLoopOfCurLoop = nullptr;
  if (CurLoop)
    OutermostLoopOfCurLoop = CurLoop->getOutermostLoop();

  unsigned SpillCnt = 0;
  for (const auto &Candidate : getCandidates(CurMI, RPTracker)) {

    if (SpillCnt >= NumOfSpills)
      break;

    Register CandidateReg = Candidate.Reg;
    NextUseDistance NextUseDist = Candidate.Dist;
    LaneBitmask Mask = Candidate.Mask;
    MachineInstr *InstrOfCandidateReg =
        MRI->getOneDef(CandidateReg)->getParent();
    MachineBasicBlock *DefBlock = InstrOfCandidateReg->getParent();
    MachineLoop *DefLoop = MLI->getLoopFor(DefBlock);

    unsigned NumOfCoveredRegs = SIRegisterInfo::getNumCoveredRegs(Mask);
    unsigned NumOfSubregisters = TRI->getRegSizeInBits(CandidateReg, *MRI) / 32;

    SpillCnt += NumOfCoveredRegs;
    NumOfERSSpills++;
    SpilledRegs.insert(CandidateReg);

    // Spill at the defintion except from the case where the definition is
    // inside the loop. In this case, we spill in the exit block.
    MachineBasicBlock *SpillBlock = nullptr;
    MachineBasicBlock::iterator WhereToSpill;
    if (DefLoop) {
      SmallVector<MachineBasicBlock *> ExitBlocks;
      MachineLoop *OutermostLoopOfDefLoop = DefLoop->getOutermostLoop();
      OutermostLoopOfDefLoop->getUniqueExitBlocks(ExitBlocks);
      assert(ExitBlocks.size() == 1 && "There should be only one exit basic "
                                       "block after CFG structurization");
      MachineBasicBlock *ExitMBB = ExitBlocks.back();
      if (!DT->dominates(ExitMBB, CurMBB))
        continue;

      bool HasUsesDominatedByExitMBB = llvm::any_of(
          MRI->use_nodbg_instructions(CandidateReg), [&](MachineInstr &UseMI) {
            MachineBasicBlock *UseMBB = UseMI.getParent();
            if (DT->dominates(ExitMBB, UseMBB))
              return true;
            return false;
          });

      if (!HasUsesDominatedByExitMBB)
        continue;

      if (ExitMBB == CurMBB) {
        WhereToSpill = CurMI->getIterator();
        SpillBlock = ExitMBB;
      } else {
        WhereToSpill = ExitMBB->getFirstTerminator();
        if (WhereToSpill == ExitMBB->end())
          WhereToSpill = ExitMBB->instr_end();
        SpillBlock = ExitMBB;
      }
    } else {
      SpillBlock = DefBlock;
      if (InstrOfCandidateReg->isPHI()) {
        WhereToSpill = DefBlock->getFirstNonPHI();
        if (WhereToSpill == DefBlock->end())
          WhereToSpill = DefBlock->instr_end();
      } else {
        if (InstrOfCandidateReg == &(DefBlock->instr_back()))
          WhereToSpill = DefBlock->instr_end();
        else
          WhereToSpill = InstrOfCandidateReg->getNextNode()->getIterator();
      }
    }

    // Emit the spill instruction.
    const TargetRegisterClass *RC = TRI->getRegClassForReg(*MRI, CandidateReg);
    unsigned Size = TRI->getSpillSize(*RC);
    Align Alignment = TRI->getSpillAlign(*RC);
    int FI = FrameInfo->CreateSpillStackObject(Size, Alignment);
    TII->storeRegToStackSlot(*SpillBlock, WhereToSpill, CandidateReg,
                             true, /* kill */
                             FI, RC, 0);
    MachineInstr *SpillInstruction = &*(std::prev(WhereToSpill));
    LIS->InsertMachineInstrInMaps(*SpillInstruction);

    LLVM_DEBUG(dbgs() << "------------------------------------------------\n");
    LLVM_DEBUG(dbgs() << "The high register pressure point is " << *CurMI);
    LLVM_DEBUG(dbgs() << "The high register pressure block is bb."
                      << CurMBB->getNumber() << "\n");
    if (MLI->getLoopFor(CurMBB)) {
      LLVM_DEBUG(dbgs() << "The high register pressure point is in a loop\n");
    } else {
      LLVM_DEBUG(
          dbgs() << "The high register pressure point is not in a loop\n");
    }
    LLVM_DEBUG(dbgs() << "Candidate register = " << printReg(CandidateReg, TRI)
                      << "\n");
    LLVM_DEBUG(dbgs() << "Instruction of the register to spill = "
                      << *InstrOfCandidateReg << "\n");
    LLVM_DEBUG(dbgs() << "Spill instruction = " << *SpillInstruction);
    LLVM_DEBUG(dbgs() << "Spill block = " << "bb." << SpillBlock->getNumber()
                      << "\n");
    LLVM_DEBUG(dbgs() << "Live interval before spilling for spilled register "
                      << printReg(CandidateReg, TRI) << ": ";
               LIS->getInterval(CandidateReg).print(dbgs()); dbgs() << "\n");

    SmallVector<MachineInstr *> UsesToUpdate;
    for (MachineInstr &U : MRI->use_nodbg_instructions(CandidateReg)) {
      if (DefLoop && !DT->dominates(SpillInstruction, &U))
        continue;

      if (&U == SpillInstruction)
        continue;

      UsesToUpdate.push_back(&U);
    }

    SmallVector<MachineInstr *> RestoreInstrs;
    MachineInstr *Restore = nullptr;
    Register RestoreReg;

    for (MachineInstr *U : UsesToUpdate) {
      if (U->isPHI()) {
        LLVM_DEBUG(dbgs() << "The use is in a phi node: " << *U);
        SmallVector<MachineBasicBlock *> PhiBlocks =
            getPhiBlocksOfSpillReg(U, CandidateReg);
        for (auto *PhiOpMBB : PhiBlocks) {
          Restore = emitRestore(CandidateReg, *PhiOpMBB, FI);
          RestoreReg = Restore->getOperand(0).getReg();
          for (unsigned i = 1; i < U->getNumOperands(); i += 2) {
            if (U->getOperand(i).getReg() == CandidateReg &&
                U->getOperand(i + 1).getMBB() == PhiOpMBB) {
              U->getOperand(i).setReg(RestoreReg);
            }
          }
          // Keep the restore instructions and uses for updating the live
          // interval analysis.
          RestoreInstrs.push_back(Restore);
          // Save the restore registers in order not to spill them again.
          RestoredRegs.insert(RestoreReg);
        }

        MachineBasicBlock *UBB = U->getParent();
        LLVM_DEBUG(dbgs() << "Use block = " << "bb." << UBB->getNumber()
                          << "\n");
        if (MLI->getLoopFor(UBB)) {
          LLVM_DEBUG(dbgs() << "The use block is in a loop\n");
        } else {
          LLVM_DEBUG(dbgs() << "The use block is not in a loop\n");
        }

        LLVM_DEBUG(dbgs() << "Live interval for restored register "
                          << printReg(RestoreReg, TRI) << ": ";
                   LIS->getInterval(RestoreReg).print(dbgs()); dbgs() << "\n");

      } else {
        Restore = emitRestore(CandidateReg, U, FI);
        RestoreReg = Restore->getOperand(0).getReg();
        // Keep the restore instructions and uses for updating the live
        // interval analysis.
        RestoreInstrs.push_back(Restore);
        // Save the restore registers in order not to spill them again.
        RestoredRegs.insert(RestoreReg);

        MachineBasicBlock *UBB = U->getParent();
        LLVM_DEBUG(dbgs() << "Updated use: " << *U);
        LLVM_DEBUG(dbgs() << "Use block = " << "bb." << UBB->getNumber()
                          << "\n");
        if (MLI->getLoopFor(UBB)) {
          LLVM_DEBUG(dbgs() << "The use block is in a loop\n");
        } else {
          LLVM_DEBUG(dbgs() << "The use block is not in a loop\n");
        }

        LLVM_DEBUG(dbgs() << "Live interval for restored register "
                          << printReg(RestoreReg, TRI) << ": ";
                   LIS->getInterval(RestoreReg).print(dbgs()); dbgs() << "\n");
      }
    }

    // Update live interval analysis and next-use distance.
    SmallPtrSet<MachineBasicBlock *, 2> UpdatedBlockIds;
    updateIndexes(InstrOfCandidateReg, Indexes);
    updateIndexes(SpillInstruction, Indexes);
    updateLiveness(InstrOfCandidateReg, LIS);
    updateLiveness(SpillInstruction, LIS);
    NUA->updateInstrIds(InstrOfCandidateReg);
    UpdatedBlockIds.insert(DefBlock);
    if (DefBlock != SpillBlock) {
      NUA->updateInstrIds(SpillInstruction);
      UpdatedBlockIds.insert(SpillBlock);
    }

    if (InstrOfCandidateReg != CurMI) {
      updateIndexes(CurMI, Indexes);
      updateLiveness(CurMI, LIS);
      auto ItC = UpdatedBlockIds.find(CurMBB);
      if (ItC == UpdatedBlockIds.end()) {
        NUA->updateInstrIds(CurMI);
        UpdatedBlockIds.insert(CurMBB);
      }
    }

    for (auto *R : RestoreInstrs) {
      updateIndexes(R, Indexes);
      updateLiveness(R, LIS);
      MachineBasicBlock *RestoreBlock = R->getParent();
      auto ItR = UpdatedBlockIds.find(RestoreBlock);
      if (ItR == UpdatedBlockIds.end()) {
        NUA->updateInstrIds(R);
        UpdatedBlockIds.insert(RestoreBlock);
      }
    }

    for (auto *Use : UsesToUpdate) {
      updateIndexes(Use, Indexes);
      updateLiveness(Use, LIS);
      MachineBasicBlock *UseBlock = Use->getParent();
      auto ItU = UpdatedBlockIds.find(UseBlock);
      if (ItU == UpdatedBlockIds.end()) {
        NUA->updateInstrIds(Use);
        UpdatedBlockIds.insert(UseBlock);
      }
    }
  }

  // Reset the tracker because it has already read the next instruction which
  // we might have modified by emitting a spill or restore instruction.
  RPTracker.reset(*CurMI, CurMI->getParent()->end());
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

  Function &F = MF.getFunction();
  AttrBuilder builder(F.getContext());
  F.removeFnAttr("amdgpu-num-vgpr");
  // unsigned RAMaxVGPRs = VGPRMaxNums + 3;
  unsigned RAMaxVGPRs = VGPRMaxNums;
  builder.addAttribute("amdgpu-num-vgpr", std::to_string(RAMaxVGPRs));
  F.addFnAttrs(builder);

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
  NUA = &getAnalysis<AMDGPUNextUseAnalysisLegacyPass>().getNextUseAnalysis();
  NUA->setConfig(AMDGPUNextUseAnalysis::Config::Graphics());

  TgSplit =
      ST.hasTgSplitSupport() && AMDGPU::isTgSplitEnabled(MF.getFunction());

  unsigned VgprNum = getMaxPressure(MF).getVGPRNum(false);
  unsigned Occupancy =
      ST.getOccupancyWithNumVGPRs(VgprNum, ST.getDynamicVGPRBlockSize());
  MaxVGPRs =
      VGPRMaxNums > 0
          ? VGPRMaxNums
          : std::min(ST.getMaxNumVGPRs(Occupancy, ST.getDynamicVGPRBlockSize()),
                     ST.getMaxNumVGPRs(MF));
  MaxSGPRs =
      std::min(ST.getMaxNumSGPRs(Occupancy, true), ST.getMaxNumSGPRs(MF));

  LLVM_DEBUG(dbgs() << "===========================================\n");
  LLVM_DEBUG(dbgs() << "Early Register Spilling\n");
  LLVM_DEBUG(dbgs() << "===========================================\n");
  LLVM_DEBUG(dbgs() << MF.getName() << "\n");
  LLVM_DEBUG(dbgs() << "MaxVGPRs = " << MaxVGPRs << "\n");
  LLVM_DEBUG(dbgs() << "Live Ranges before ERS\n");
  LLVM_DEBUG(LIS->dump());

  GCNDownwardRPTracker RPTracker(*LIS);
  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  unsigned NumOfVGPRsForSGPR = 0;
  for (MachineBasicBlock *MBB : RPOT) {
    if (MBB->empty())
      continue;

    // Initialize the Register Pressure Tracker at the beginning of the
    // block.
    RPTracker.reset(*MBB->begin(), MBB->end());
    RPTracker.advance();

    // Iterate over the instructions of MBB and check if the live registers
    // are more than the available registers.
    unsigned ExcessiveSGPRs = 0;
    for (auto It = MBB->begin(), ItE = MBB->end(); It != ItE; ++It) {
      MachineInstr *MI = &*It;

      if (MI->isDebugInstr())
        continue;

      const MachineInstr *LastTrackedMI = RPTracker.getLastTrackedMI();
      assert(MI == LastTrackedMI && "The tracker and the loop iteration "
                                    "should visit the same instruction.");
      unsigned SGPRLiveRegs = RPTracker.getPressure().getSGPRNum();
      unsigned VGPRLiveRegs = RPTracker.getPressure().getArchVGPRNum();

      if (SGPRLiveRegs > MaxSGPRs) {
        // SGPRs are spilled into VGPRs. We estimate how many VGPRs are needed
        // for SGPR spilling. These are subtracted from the maximum number of
        // available VGPRs for register allocation.
        unsigned NewExcessiveSGPRs = SGPRLiveRegs - MaxSGPRs;
        if (ExcessiveSGPRs < NewExcessiveSGPRs) {
          ExcessiveSGPRs = NewExcessiveSGPRs;

          // Check whether we should group 32 or 64 SGPRs.
          auto NumOfSGPRsToGroup =
              MF.getSubtarget<GCNSubtarget>().isWave32() ? 32.0 : 64.0;
          NumOfVGPRsForSGPR = std::ceil(ExcessiveSGPRs / NumOfSGPRsToGroup);
        }
      }

      // Spill if the live VGPR registers are more than the available
      // VGPRs.
      if (VGPRLiveRegs > (MaxVGPRs - NumOfVGPRsForSGPR)) {
        LLVM_DEBUG(dbgs() << "===========================================\n");
        LLVM_DEBUG(dbgs() << "Current MI = " << *MI << "\n");
        unsigned NumOfSpills = VGPRLiveRegs - MaxVGPRs + NumOfVGPRsForSGPR;
        LLVM_DEBUG(dbgs() << "Number of spills = " << NumOfSpills << "\n");
        LLVM_DEBUG(dbgs() << "VGPRLiveRegs = " << VGPRLiveRegs << "\n");

        spill(MI, RPTracker, NumOfSpills);
      }

      // Move the tracker to the next instruction.
      // If we have reached the bottom of a basic block, then we have to
      // initialize the tracker at the beginning of the next basic block.
      if (MI == &MBB->back())
        continue;

      // Phi nodes might include registers that are defined later in the
      // code. Hence, we have to initialize the tracker again.
      if (MI->getOpcode() == AMDGPU::PHI) {
        RPTracker.reset(*MI->getNextNode(), MBB->end());
      }
      RPTracker.advance();
    }
  }

  LLVM_DEBUG(dbgs() << "===========================================\n");
  LLVM_DEBUG(dbgs() << "Live Ranges after ERS\n");
  LLVM_DEBUG(LIS->dump());
  LLVM_DEBUG(dbgs() << "===========================================\n");

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
INITIALIZE_PASS_DEPENDENCY(AMDGPUNextUseAnalysisLegacyPass)
INITIALIZE_PASS_END(AMDGPUEarlyRegisterSpilling, DEBUG_TYPE,
                    "Early Register Spilling", false, false)

char &llvm::AMDGPUEarlyRegisterSpillingID = AMDGPUEarlyRegisterSpilling::ID;

FunctionPass *llvm::createAMDGPUEarlyRegisterSpillingPass() {
  return new AMDGPUEarlyRegisterSpilling();
}
