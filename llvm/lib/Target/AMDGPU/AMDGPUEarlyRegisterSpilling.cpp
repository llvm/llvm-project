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

static cl::opt<bool>
    EmitRestoreInCommonDominator("emit-restore-in-common-dominator",
                                 cl::init(false), cl::Hidden);

static cl::opt<bool> DisableRestoreGrouping("disable-restore-grouping",
                                            cl::init(false), cl::Hidden);

static cl::opt<bool> EnableRestoreOptimization("enable-restore-optimization",
                                               cl::init(true), cl::Hidden);

static cl::opt<int>
    RestoreOptMinDistance("restore-optimization-min-distance",
                          cl::init(std::numeric_limits<int>::max()),
                          cl::Hidden);

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

class SpillOrRestoreCandidate {
public:
  enum class CodeGenPlan {
    EmitSpillRestore, // This is the common case where we emit spill/restore
                      // instructions.
    MoveRestoreInsideTheLoop, // Move the restore inside the loop.
    EmitNewRestoreBeforeUse, // Emit a restore before a use instead of using the
                             // restore of another use.
  };

protected:
  Register CandidateReg;
  LaneBitmask Mask;
  // We group the uses based on dominance. Uses that are dominated by one use
  // are all in the same group.
  std::vector<DomGroup> GroupsOfUses;
  int64_t RestoreCost = 0;
  NextUseDistance Dist;
  int64_t NormalizedRestoreCost = 0;
  int64_t NormalizedCost = 0;
  CodeGenPlan Plan;
  const SIRegisterInfo *TRI;
  MachineRegisterInfo *MRI;
  const SIInstrInfo *TII;
  MachineFrameInfo *FrameInfo;
  LiveIntervals *LIS;
  SlotIndexes *Indexes;
  MachineDominatorTree *DT;
  const MachineLoopInfo *MLI;

  /// Emit restore instruction where it is needed
  MachineInstr *emitRestore(Register SpillReg, MachineInstr *UseMI, int FI);
  /// Emit restore instruction at the end of a basic block.
  MachineInstr *emitRestore(Register SpillReg, MachineBasicBlock &InsertBB,
                            int FI);
  /// Helper function for generateSpillRestoreInstrs().
  void emitRestoresForHead(SmallVector<MachineInstr *> &RestoreInstrs,
                           SmallVector<MachineInstr *> &RestoreUses, int FI,
                           DenseMap<Register, DomGroup> &RestoreRegToDomGroup);

  unsigned loopWeight(unsigned LoopDepth) { return 1000 * LoopDepth; }

  SpillOrRestoreCandidate(Register CandidateReg, LaneBitmask Mask,
                          CodeGenPlan Plan, const SIRegisterInfo *TRI,
                          MachineRegisterInfo *MRI, const SIInstrInfo *TII,
                          MachineFrameInfo *FrameInfo, LiveIntervals *LIS,
                          SlotIndexes *Indexes, MachineDominatorTree *DT,
                          const MachineLoopInfo *MLI)
      : CandidateReg(CandidateReg), Mask(Mask),
        Dist(NextUseDistance::unreachable()), Plan(Plan), TRI(TRI), MRI(MRI),
        TII(TII), FrameInfo(FrameInfo), LIS(LIS), Indexes(Indexes), DT(DT),
        MLI(MLI) {}

public:
  virtual ~SpillOrRestoreCandidate() = default;
  Register getCandidateRegister() const { return CandidateReg; }
  LaneBitmask getMask() const { return Mask; }
  void addGroup(DomGroup DG) { GroupsOfUses.push_back(DG); }
  auto groups() { return make_range(GroupsOfUses.begin(), GroupsOfUses.end()); }
  int64_t getRestoreCost() const { return RestoreCost; }
  void setNextUseDistance(NextUseDistance NUD) { Dist = NUD; }
  NextUseDistance getNextUseDistance() const { return Dist; }
  void setNormalizedRestoreCost(int64_t NRC) { NormalizedRestoreCost = NRC; }
  int64_t getNormalizedRestoreCost() const { return NormalizedRestoreCost; }
  void setNormalizedCost(int64_t NC) { NormalizedCost = NC; }
  int64_t getNormalizedCost() const { return NormalizedCost; }
  void calculateRestoreCost();
  virtual void generateSpillRestoreInstrs(
      MachineInstr *CurMI,
      DenseMap<Register, DomGroup> &RestoreRegToDomGroup) = 0;
  CodeGenPlan getCodeGenPlan() const { return Plan; }
};

class SpillCandidate final : public SpillOrRestoreCandidate {
private:
  MachineBasicBlock *SpillBlock;
  MachineBasicBlock::iterator WhereToSpill;

public:
  SpillCandidate(Register CandidateReg, LaneBitmask Mask, CodeGenPlan Plan,
                 const SIRegisterInfo *TRI, MachineRegisterInfo *MRI,
                 const SIInstrInfo *TII, MachineFrameInfo *FrameInfo,
                 LiveIntervals *LIS, SlotIndexes *Indexes,
                 MachineDominatorTree *DT, const MachineLoopInfo *MLI,
                 MachineBasicBlock *SpillBlock,
                 MachineBasicBlock::iterator WhereToSpill)
      : SpillOrRestoreCandidate(CandidateReg, Mask, Plan, TRI, MRI, TII,
                                FrameInfo, LIS, Indexes, DT, MLI),
        SpillBlock(SpillBlock), WhereToSpill(WhereToSpill) {}

  MachineBasicBlock *getSpillBlock() const { return SpillBlock; }
  MachineBasicBlock::iterator getWhereToSpill() const { return WhereToSpill; }
  void generateSpillRestoreInstrs(
      MachineInstr *CurMI,
      DenseMap<Register, DomGroup> &RestoreRegToDomGroup) override;
};

class RestoreCandidate final : public SpillOrRestoreCandidate {
public:
  RestoreCandidate(Register CandidateReg, LaneBitmask Mask, CodeGenPlan Plan,
                   const SIRegisterInfo *TRI, MachineRegisterInfo *MRI,
                   const SIInstrInfo *TII, MachineFrameInfo *FrameInfo,
                   LiveIntervals *LIS, SlotIndexes *Indexes,
                   MachineDominatorTree *DT, const MachineLoopInfo *MLI)
      : SpillOrRestoreCandidate(CandidateReg, Mask, Plan, TRI, MRI, TII,
                                FrameInfo, LIS, Indexes, DT, MLI) {}

  void generateSpillRestoreInstrs(
      MachineInstr *CurMI,
      DenseMap<Register, DomGroup> &RestoreRegToDomGroup) override;
};

void SpillOrRestoreCandidate::calculateRestoreCost() {
  RestoreCost = 0;
  for (auto &G : GroupsOfUses) {
    MachineBasicBlock *RestoreBlock = G.getRestoreBlock();
    MachineLoop *RestoreLoop = MLI->getLoopFor(RestoreBlock);

    if (RestoreLoop) {
      RestoreCost += loopWeight(RestoreLoop->getLoopDepth());
    } else {
      RestoreCost += 1;
    }
  }
}

void SpillOrRestoreCandidate::emitRestoresForHead(
    SmallVector<MachineInstr *> &RestoreInstrs,
    SmallVector<MachineInstr *> &RestoreUses, int FI,
    DenseMap<Register, DomGroup> &RestoreRegToDomGroup) {

  // For each group emit one restore for the group header in the parent block
  // of the group header or the common dominator. The rest of the uses in the
  // group will reuse the value loaded by the restore of the header.
  for (auto &G1 : groups()) {
    if (G1.isDeleted())
      continue;
    MachineInstr *Head = G1.getHead();
    MachineBasicBlock *HeadMBB = G1.getRestoreBlock();
    MachineInstr *Restore = nullptr;
    if (G1.hasCommonDominator() && EmitRestoreInCommonDominator) {
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
        Restore = emitRestore(CandidateReg, UseInCommonDominator, FI);
        Head = UseInCommonDominator;
        HeadMBB = CommonDominator;
      } else {
        Restore = emitRestore(CandidateReg, *CommonDominator, FI);
        Head->substituteRegister(CandidateReg, Restore->getOperand(0).getReg(),
                                 0, *TRI);
      }
    } else if (Head->isPHI()) {
      LLVM_DEBUG(dbgs() << "Head is phi node: " << *Head);
      LLVM_DEBUG(dbgs() << "The group has " << G1.size() << " use(s). \n");
      Restore = emitRestore(CandidateReg, *HeadMBB, FI);
      for (unsigned i = 1; i < Head->getNumOperands(); i += 2) {
        if (Head->getOperand(i).getReg() == CandidateReg &&
            Head->getOperand(i + 1).getMBB() == HeadMBB) {
          Head->getOperand(i).setReg(Restore->getOperand(0).getReg());
        }
      }
    } else if (Head->getParent() != HeadMBB) {
      LLVM_DEBUG(dbgs() << "Restore in loop preheader.\n");
      LLVM_DEBUG(dbgs() << "The group has " << G1.size() << " use(s). \n");
      LLVM_DEBUG(dbgs() << "The head is " << *Head);
      Restore = emitRestore(CandidateReg, *HeadMBB, FI);
      Head->substituteRegister(CandidateReg, Restore->getOperand(0).getReg(), 0,
                               *TRI);
    } else {
      LLVM_DEBUG(dbgs() << "Common case.\n");
      LLVM_DEBUG(dbgs() << "The group has " << G1.size() << " use(s). \n");
      LLVM_DEBUG(dbgs() << "The head is " << *Head);
      Restore = emitRestore(CandidateReg, Head, FI);
    }
    RestoreInstrs.push_back(Restore);
    RestoreUses.push_back(Head);
    G1.setRestore(Restore);
    RestoreRegToDomGroup[Restore->getOperand(0).getReg()] = G1;

    // Update the rest of the uses in the group to reuse the value restored by
    // the head of the group.
    for (auto *U : G1.getUses()) {
      if (U == Head)
        continue;

      MachineBasicBlock *UBB = U->getParent();
      LLVM_DEBUG(dbgs() << "Updated use: " << *U);
      LLVM_DEBUG(dbgs() << "Use block = " << "bb." << UBB->getNumber() << "\n");
      if (MLI->getLoopFor(UBB)) {
        LLVM_DEBUG(dbgs() << "The use block is in a loop\n");
      } else {
        LLVM_DEBUG(dbgs() << "The use block is not in a loop\n");
      }

      if (U->isPHI()) {
        MachineBasicBlock *RestoreMBB = Restore->getParent();
        for (unsigned i = 1; i < U->getNumOperands(); i += 2) {
          if (U->getOperand(i).getReg() == CandidateReg &&
              U->getOperand(i + 1).getMBB() == G1.getRestoreBlockForPHI(U) &&
              U->getOperand(i + 1).getMBB() == RestoreMBB) {
            U->getOperand(i).setReg(Restore->getOperand(0).getReg());
          }
        }
      } else {
        U->substituteRegister(CandidateReg, Restore->getOperand(0).getReg(), 0,
                              *TRI);
      }
      RestoreUses.push_back(U);
    }
    LLVM_DEBUG(dbgs() << "Live interval for restored register "
                      << printReg(Restore->getOperand(0).getReg(), TRI) << ": ";
               LIS->getInterval(Restore->getOperand(0).getReg()).print(dbgs());
               dbgs() << "\n");
  }
}

void SpillCandidate::generateSpillRestoreInstrs(
    MachineInstr *CurMI, DenseMap<Register, DomGroup> &RestoreRegToDomGroup) {

  MachineInstr *InstrOfCandidateReg = MRI->getOneDef(CandidateReg)->getParent();
  MachineInstr *SpillInstruction = nullptr;
  const TargetRegisterClass *RC = TRI->getRegClassForReg(*MRI, CandidateReg);
  unsigned Size = TRI->getSpillSize(*RC);
  Align Alignment = TRI->getSpillAlign(*RC);
  int FI = FrameInfo->CreateSpillStackObject(Size, Alignment);
  LLVM_DEBUG(dbgs() << "------------------------------------------------\n");
  LLVM_DEBUG(dbgs() << "Plan: Emit spill/restore instructions.\n");
  LLVM_DEBUG(dbgs() << "Live interval before spilling for spilled register "
                    << printReg(CandidateReg, TRI) << ": ";
             LIS->getInterval(CandidateReg).print(dbgs()); dbgs() << "\n");

  // Emit the spill instruction.
  TII->storeRegToStackSlot(*SpillBlock, WhereToSpill, CandidateReg,
                           true, /* kill */
                           FI, RC, 0);
  SpillInstruction = &*(std::prev(WhereToSpill));
  LIS->InsertMachineInstrInMaps(*SpillInstruction);

  MachineBasicBlock *SpillBlock = SpillInstruction->getParent();
  MachineBasicBlock *CurMBB = CurMI->getParent();
  LLVM_DEBUG(dbgs() << "------------------------------------------------\n");
  LLVM_DEBUG(dbgs() << "The high register pressure point is " << *CurMI);
  LLVM_DEBUG(dbgs() << "The high register pressure block is bb."
                    << CurMBB->getNumber() << "\n");
  if (MLI->getLoopFor(CurMBB)) {
    LLVM_DEBUG(dbgs() << "The high register pressure point is in a loop\n");
  } else {
    LLVM_DEBUG(dbgs() << "The high register pressure point is not in a loop\n");
  }
  LLVM_DEBUG(dbgs() << "Candidate register = " << printReg(CandidateReg, TRI)
                    << "\n");
  LLVM_DEBUG(dbgs() << "Instruction of the register to spill = "
                    << *InstrOfCandidateReg << "\n");
  LLVM_DEBUG(dbgs() << "Spill instruction = " << *SpillInstruction);
  LLVM_DEBUG(dbgs() << "Spill block = " << "bb." << SpillBlock->getNumber()
                    << "\n");

  SmallVector<MachineInstr *> RestoreInstrs;
  SmallVector<MachineInstr *> RestoreUses;
  // Emit restore instructions for each group.
  emitRestoresForHead(RestoreInstrs, RestoreUses, FI, RestoreRegToDomGroup);

  // Update the live interval analysis.
  updateIndexes(InstrOfCandidateReg, Indexes);
  updateIndexes(SpillInstruction, Indexes);
  updateLiveness(InstrOfCandidateReg, LIS);
  updateLiveness(SpillInstruction, LIS);

  if (InstrOfCandidateReg != CurMI) {
    updateIndexes(CurMI, Indexes);
    updateLiveness(CurMI, LIS);
  }

  for (auto *Use : RestoreInstrs) {
    updateIndexes(Use, Indexes);
    updateLiveness(Use, LIS);
  }

  for (auto *Use : RestoreUses) {
    updateIndexes(Use, Indexes);
    updateLiveness(Use, LIS);
  }

  LLVM_DEBUG(dbgs() << "Live interval after spilling for spilled register "
                    << printReg(CandidateReg, TRI) << ": ";
             LIS->getInterval(CandidateReg).print(dbgs()); dbgs() << "\n");
}

void RestoreCandidate::generateSpillRestoreInstrs(
    MachineInstr *CurMI, DenseMap<Register, DomGroup> &RestoreRegToDomGroup) {

  MachineInstr *InstrOfCandidateReg = MRI->getOneDef(CandidateReg)->getParent();
  assert(TII->isVGPRSpill(InstrOfCandidateReg->getOpcode()) &&
         InstrOfCandidateReg->mayLoad() && "Expected restore instruction!");
  MachineBasicBlock *CurMBB = CurMI->getParent();

  if (getCodeGenPlan() == CodeGenPlan::MoveRestoreInsideTheLoop) {
    // Move the restore that is in loop preheader inside the loop before the
    // HEAD.
    assert(GroupsOfUses.size() == 1 &&
           "This candidate cannot have more than one DomGroups.");
    DomGroup &DG = *GroupsOfUses.begin();
    MachineInstr *Head = DG.getHead();
    MachineInstr *OrigRestore = DG.getRestore();
    assert(OrigRestore == InstrOfCandidateReg && "Wrong restore instruction.");
    OrigRestore->moveBefore(Head);
    updateIndexes(OrigRestore, Indexes);
    updateLiveness(OrigRestore, LIS);
    updateIndexes(Head, Indexes);
    updateLiveness(Head, LIS);
    updateIndexes(InstrOfCandidateReg, Indexes);
    updateLiveness(InstrOfCandidateReg, LIS);

    if (InstrOfCandidateReg != CurMI) {
      updateIndexes(CurMI, Indexes);
      updateLiveness(CurMI, LIS);
    }

    LLVM_DEBUG(dbgs() << "------------------------------------------------\n");
    LLVM_DEBUG(
        dbgs() << "Plan: Move restore instruction from loop preheader to "
                  "the head isnide the loop. \n");
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
    LLVM_DEBUG(dbgs() << "Original restore = " << *OrigRestore << "\n");
    LLVM_DEBUG(dbgs() << "Move restore before head : " << *Head << "\n");
    LLVM_DEBUG(
        dbgs() << "Live interval for restored register "
               << printReg(OrigRestore->getOperand(0).getReg(), TRI) << ": ";
        LIS->getInterval(OrigRestore->getOperand(0).getReg()).print(dbgs());
        dbgs() << "\n");

    // Update the restore block inside the DomGroup.
    DG.setRestoreBlock(OrigRestore->getParent());

    // Update RestoreRegToDomGroup map with the updated DomGroup.
    RestoreRegToDomGroup[OrigRestore->getOperand(0).getReg()] = DG;
  } else if (getCodeGenPlan() == CodeGenPlan::EmitNewRestoreBeforeUse) {
    LLVM_DEBUG(dbgs() << "------------------------------------------------\n");
    LLVM_DEBUG(
        dbgs()
        << "Plan: Emit new restore instructions wherever it is needed. \n");
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
    LLVM_DEBUG(dbgs() << "Original restore = " << *InstrOfCandidateReg << "\n");

    int FI = InstrOfCandidateReg->getOperand(1).getIndex();
    SmallVector<MachineInstr *> RestoreInstrs;
    SmallVector<MachineInstr *> RestoreUses;
    // Emit restore instructions for each group.
    emitRestoresForHead(RestoreInstrs, RestoreUses, FI, RestoreRegToDomGroup);

    // Update the live interval analysis.
    updateIndexes(InstrOfCandidateReg, Indexes);
    updateLiveness(InstrOfCandidateReg, LIS);

    if (InstrOfCandidateReg != CurMI) {
      updateIndexes(CurMI, Indexes);
      updateLiveness(CurMI, LIS);
    }

    for (auto *Use : RestoreInstrs) {
      updateIndexes(Use, Indexes);
      updateLiveness(Use, LIS);
    }

    for (auto *Use : RestoreUses) {
      updateIndexes(Use, Indexes);
      updateLiveness(Use, LIS);
    }
  }
}

MachineInstr *SpillOrRestoreCandidate::emitRestore(Register CandidateReg,
                                                   MachineInstr *DefRegUseInstr,
                                                   int FI) {
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

MachineInstr *SpillOrRestoreCandidate::emitRestore(Register CandidateReg,
                                                   MachineBasicBlock &InsertBB,
                                                   int FI) {
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
  // If EnableRestoreOptimization flag is true, then we should not block the
  // live-in registers (that are defined in restore instructions).
  bool IsRestore = !EnableRestoreOptimization && isRestoredReg(CandidateReg);

  return !hasPHIUseInSameBB(CandidateReg, CandidateMI->getParent()) &&
         !MRI->use_nodbg_empty(CandidateReg) && !isSpilledReg(CandidateReg) &&
         !CandidateReg.isPhysical() && !TRI->isAGPR(*MRI, CandidateReg) &&
         !CandidateMI->isTerminator() && TRI->isVGPR(*MRI, CandidateReg) &&
         !IsRestore;
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
    NUA->getReachableUses(CandidateReg, Mask, *CurMI,
                          UsesForNextUseDistCalculation);
    if (UsesForNextUseDistCalculation.empty())
      continue;

    RegNumOfUses[CandidateReg] = UsesForNextUseDistCalculation.size();

    LLVM_DEBUG(CandidateCnt++);
    // Calculate the next-use distance for the spill candidates and add them in
    // 'RegCandidates'.
    auto NextUseDist = NUA->getShortestDistance(CandidateReg, *CurMI,
                                                UsesForNextUseDistCalculation);

    if (EnableRestoreOptimization && RestoreOptMinDistance != 0 &&
        isRestoredReg(CandidateReg) && CandidateMI->getParent() == CurMBB) {
      NextUseDistance Limit(0);
      if (RestoreOptMinDistance < 0) {
        Limit = NextUseDistance(static_cast<unsigned>(-RestoreOptMinDistance));
      } else {
        const NextUseDistance MBBSpan = NUA->getSpan(*CurMBB);
        Limit = std::min<NextUseDistance>(
            static_cast<unsigned>(RestoreOptMinDistance), MBBSpan);
      }
      if (NextUseDist < Limit)
        continue;
    }

    RegCandidates.push_back({CandidateReg, NextUseDist, Mask});
    LLVM_DEBUG({
      dbgs() << CandidateCnt
             << ": Candidate register = " << printReg(CandidateReg, TRI)
             << " with distance = " << NextUseDist.fmt() << "\n";
    });
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

bool AMDGPUEarlyRegisterSpilling::shouldEmitRestoreInCommonDominator(
    MachineBasicBlock *SpillBlock, MachineBasicBlock *CurMBB,
    MachineBasicBlock *CommonDominatorToRestore) {

  if (!EmitRestoreInCommonDominator)
    return false;

  if (SpillBlock == CommonDominatorToRestore)
    return false;
  if (CurMBB == CommonDominatorToRestore)
    return false;
  if (DT->dominates(CommonDominatorToRestore, SpillBlock))
    return false;
  if (NUA->isReachable(CommonDominatorToRestore, SpillBlock))
    return false;
  if (!DT->dominates(SpillBlock, CommonDominatorToRestore))
    return false;
  if (MLI->getLoopFor(CommonDominatorToRestore))
    return false;
  return true;
}

static bool shouldGroupUses(MachineLoop *CurLoop, MachineLoop *Head1Loop,
                            MachineLoop *Head2Loop,
                            MachineBasicBlock *RestoreBlock1,
                            MachineBasicBlock *RestoreBlock2) {

  MachineLoop *OutermostLoopOfCurLoop = nullptr;
  if (CurLoop)
    OutermostLoopOfCurLoop = CurLoop->getOutermostLoop();

  // Do not group the restores if one of them is in a loop.
  if ((Head1Loop && !Head2Loop) || (!Head1Loop && Head2Loop))
    return false;

  // Do not group the restores if the loops are independent.
  if (!CurLoop && Head1Loop && Head2Loop && (RestoreBlock1 != RestoreBlock2) &&
      !Head1Loop->contains(Head2Loop) && !Head2Loop->contains(Head1Loop))
    return false;

  // Do not group the restores if all the loops are independent.
  if (CurLoop && Head1Loop && Head2Loop && (RestoreBlock1 != RestoreBlock2) &&
      !OutermostLoopOfCurLoop->contains(Head1Loop) &&
      !OutermostLoopOfCurLoop->contains(Head2Loop) &&
      !Head1Loop->contains(Head2Loop) && !Head2Loop->contains(Head1Loop))
    return false;

  // Do not group the restores if one use is in the loop nest of the CurLoop and
  // the other use is not in the loop nest of CurLoop.
  if (CurLoop && ((OutermostLoopOfCurLoop->contains(Head1Loop) &&
                   !OutermostLoopOfCurLoop->contains(Head2Loop)) ||
                  (!OutermostLoopOfCurLoop->contains(Head1Loop) &&
                   OutermostLoopOfCurLoop->contains(Head2Loop))))
    return false;

  // Do not group the restores if the current loop contains the loops of the
  // uses and the loops of the uses are independent.
  if (CurLoop && OutermostLoopOfCurLoop->contains(CurLoop) &&
      OutermostLoopOfCurLoop->contains(Head1Loop) &&
      OutermostLoopOfCurLoop->contains(Head2Loop) &&
      !Head1Loop->contains(Head2Loop) && !Head2Loop->contains(Head1Loop))
    return false;

  return true;
}

// PHI incoming edges require a restore at the end of that specific predecessor
// block. Do not merge groups that would share one restore across different
// predecessor blocks when either group carries a PHI incoming-edge placement.
static bool mustKeepSeparatePhiRestoreBlocks(const DomGroup &G1,
                                             const DomGroup &G2) {
  if (G1.getRestoreBlock() == G2.getRestoreBlock())
    return false;

  return G1.getWhereToRestore() ==
             DomGroup::RestorePlacement::IncomingBlockOfPhi ||
         G2.getWhereToRestore() ==
             DomGroup::RestorePlacement::IncomingBlockOfPhi;
}

void AMDGPUEarlyRegisterSpilling::groupUses(
    Register CandidateReg, MachineBasicBlock *SpillBlock, MachineInstr *CurMI,
    SetVectorType &DominatedUses, SmallVector<DomGroup> &GroupOfUses) {
  MachineBasicBlock *CurMBB = CurMI->getParent();
  MachineLoop *CurLoop = MLI->getLoopFor(CurMBB);
  MachineLoop *OutermostLoopOfCurLoop = nullptr;
  if (CurLoop)
    OutermostLoopOfCurLoop = CurLoop->getOutermostLoop();

  auto AreLoopsInSameLoopNest = [OutermostLoopOfCurLoop](MachineLoop *UseLoop,
                                                         MachineLoop *CurLoop) {
    return (UseLoop && CurLoop &&
            (UseLoop->contains(CurLoop) ||
             OutermostLoopOfCurLoop->contains(UseLoop)));
  };

  std::vector<DomGroup> Groups;
  for (auto *Use : DominatedUses) {
    MachineLoop *UseLoop = MLI->getLoopFor(Use->getParent());
    if (Use->isPHI()) {
      // In case of phi nodes, the restore instructions are emitted at the
      // bottom of the incoming blocks.
      for (MachineBasicBlock *PhiOpMBB :
           getPhiBlocksOfSpillReg(Use, CandidateReg)) {
        Groups.emplace_back(Use, PhiOpMBB,
                            DomGroup::RestorePlacement::IncomingBlockOfPhi);
      }
    } else if (UseLoop) {
      if (CurLoop && AreLoopsInSameLoopNest(UseLoop, CurLoop)) {
        // If the high register pressure point and the use are in the same
        // loop nest then the restore instruction is emitted before the use.
        Groups.emplace_back(Use, Use->getParent(),
                            DomGroup::RestorePlacement::BeforeHead);
      } else {
        // If the high register pressure point is outside of the loop nest of
        // the use, then the restore instruction is emitted in the outermost
        // loop's preheader.
        MachineLoop *OutermostLoop = UseLoop->getOutermostLoop();
        MachineBasicBlock *OutermostLoopPreheader =
            OutermostLoop->getLoopPreheader();
        Groups.emplace_back(Use, OutermostLoopPreheader,
                            DomGroup::RestorePlacement::LoopPreheader);
      }
    } else {
      // Emit restore before Use.
      Groups.emplace_back(Use, Use->getParent(),
                          DomGroup::RestorePlacement::BeforeHead);
    }
  }

  if (DisableRestoreGrouping) {
    for (auto &G1 : Groups) {
      if (G1.isDeleted())
        continue;

      GroupOfUses.push_back(G1);
    }
    return;
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
    // for (unsigned Idx2 = 0; Idx2 < E; ++Idx2) {
    for (unsigned Idx2 = 0; Idx2 != E; ++Idx2) {
      auto &G2 = Groups[Idx2];
      if (G1.getHead() == G2.getHead())
        continue;

      if (G2.isDeleted())
        continue;

      MachineInstr *Head1 = G1.getHead();
      MachineInstr *Head2 = G2.getHead();
      MachineBasicBlock *RestoreBlock1 = G1.getRestoreBlock();
      MachineBasicBlock *RestoreBlock2 = G2.getRestoreBlock();
      MachineLoop *Head1Loop = MLI->getLoopFor(Head1->getParent());
      MachineLoop *Head2Loop = MLI->getLoopFor(Head2->getParent());
      // Disable the grouping of the restore instructions for the following loop
      // scenarios.
      // TODO: Change this if it creates performance degradation.
      if (!shouldGroupUses(CurLoop, Head1Loop, Head2Loop, RestoreBlock1,
                           RestoreBlock2))
        continue;
      SmallVector<MachineBasicBlock *> UseBlocks;
      for (auto *Block : G1.getUseBlocks())
        UseBlocks.push_back(Block);

      for (auto *Block : G2.getUseBlocks())
        UseBlocks.push_back(Block);

      if (mustKeepSeparatePhiRestoreBlocks(G1, G2))
        continue;

      if (EmitRestoreInCommonDominator) {
        MachineBasicBlock *CommonDom = DT->findNearestCommonDominator(
            make_range(UseBlocks.begin(), UseBlocks.end()));

        if ((RestoreBlock1 != RestoreBlock2) &&
            shouldEmitRestoreInCommonDominator(SpillBlock, CurMBB, CommonDom)) {
          // Set a common dominator if the two restore blocks are different.
          G1.merge(G2);
          G1.setCommonDominator(CommonDom);
        } else if ((RestoreBlock1 == RestoreBlock2) &&
                   DT->dominates(Head1, Head2) && !G1.getCommonDominator() &&
                   !G2.getCommonDominator()) {
          // If there is no common dominator and one Head dominates the other,
          // then we can merge the two groups.
          G1.merge(G2);
        } else if ((RestoreBlock1 != RestoreBlock2) &&
                   DT->dominates(RestoreBlock1, RestoreBlock2) &&
                   !G1.getCommonDominator() && !G2.getCommonDominator()) {
          // If there is no common dominator and one Head dominates the other,
          // then we can merge the two groups.
          G1.merge(G2);
        }
      } else if ((RestoreBlock1 == RestoreBlock2) &&
                 DT->dominates(Head1, Head2)) {
        G1.merge(G2);
      } else if ((RestoreBlock1 != RestoreBlock2) &&
                 DT->dominates(RestoreBlock1, RestoreBlock2)) {
        G1.merge(G2);
      }
    }
  }

  for (auto &G1 : Groups) {
    if (G1.isDeleted())
      continue;

    GroupOfUses.push_back(G1);
  }
}

// We have to collect the unreachable uses before we emit the spill instruction.
// This is due to the fact that some unreachable uses might become reachable if
// we spill in common dominator.
void AMDGPUEarlyRegisterSpilling::classifyUses(
    MachineBasicBlock *SpillBlock, Register CandidateReg, MachineInstr *CurMI,
    SetVectorType &DominatedUses, SetVectorType &NonDominatedReachableUses,
    SetVectorType &UnreachableUses) {

  MachineBasicBlock *CurMBB = CurMI->getParent();

  std::set<MachineInstr *> Visited;
  for (MachineInstr &U : MRI->use_nodbg_instructions(CandidateReg)) {
    if (!Visited.insert(&U).second)
      continue;

    if (U.isPHI()) {
      SmallVector<MachineBasicBlock *> PhiBlocks =
          getPhiBlocksOfSpillReg(&U, CandidateReg);
      if (PhiBlocks.empty()) {
        // All incoming edges may be undef; treat as unreachable from
        // SpillBlock.
        UnreachableUses.insert(&U);
        continue;
      }
      int Inserts = 0;
      for (auto *PhiOpMBB : PhiBlocks) {
        MachineBasicBlock *UseMBB = U.getParent();
        // The uses which are before the high register pressure point are
        // unreachable.
        if (((CurMBB != UseMBB) && NUA->isReachable(UseMBB, CurMBB)) ||
            ((CurMBB == UseMBB) && DT->dominates(&U, CurMI))) {
          Inserts += UnreachableUses.insert(&U);
        } else if (DT->dominates(SpillBlock, PhiOpMBB)) {
          Inserts += DominatedUses.insert(&U);
        } else if (NUA->isReachable(SpillBlock, PhiOpMBB)) {
          Inserts += NonDominatedReachableUses.insert(&U);
        } else {
          Inserts += UnreachableUses.insert(&U);
        }
      }
      assert(Inserts == 1 &&
             "PHI has multiple uses with varying classifications");
    } else {
      MachineBasicBlock *UseMBB = U.getParent();
      // The uses which are before the high register pressure point are
      // unreachable.
      if (((CurMBB != UseMBB) && NUA->isReachable(UseMBB, CurMBB)) ||
          ((CurMBB == UseMBB) && DT->dominates(&U, CurMI))) {
        UnreachableUses.insert(&U);
      } else if (DT->dominates(CurMI, &U)) {
        DominatedUses.insert(&U);
      } else if (NUA->isReachable(SpillBlock, UseMBB)) {
        NonDominatedReachableUses.insert(&U);
      } else {
        UnreachableUses.insert(&U);
      }
    }
  }
  assert((Visited.size() ==
          (DominatedUses.size() + NonDominatedReachableUses.size() +
           UnreachableUses.size())) &&
         "Instruction not classified or has multiple classifications");
}

// Find the common dominator of the reachable uses and the block that we
// intend to spill(SpillBlock).
MachineBasicBlock *AMDGPUEarlyRegisterSpilling::findCommonDominatorToSpill(
    MachineBasicBlock *SpillBlock, Register CandidateReg,
    const SetVectorType &NonDominatedReachableUses) {
  SmallPtrSet<MachineBasicBlock *, 2> Blocks;
  for (auto *RU : NonDominatedReachableUses) {
    if (RU->isPHI()) {
      for (auto *PhiOpMBB : getPhiBlocksOfSpillReg(RU, CandidateReg))
        Blocks.insert(PhiOpMBB);
    } else
      Blocks.insert(RU->getParent());
  }

  Blocks.insert(SpillBlock);
  MachineBasicBlock *CommonDominatorToSpill =
      DT->findNearestCommonDominator(make_range(Blocks.begin(), Blocks.end()));
  MachineLoop *CommonDominatorLoop = MLI->getLoopFor(CommonDominatorToSpill);
  if (CommonDominatorLoop)
    return nullptr;

  return CommonDominatorToSpill;
}

std::pair<MachineBasicBlock *, MachineBasicBlock::iterator>
AMDGPUEarlyRegisterSpilling::getWhereToSpillIfDefintionInLoop(
    MachineInstr *CurMI, MachineBasicBlock *DefRegMBB) {

  MachineBasicBlock *CurMBB = CurMI->getParent();
  MachineLoop *DefInstrLoop = MLI->getLoopFor(DefRegMBB);
  MachineBasicBlock *SpillBlock = nullptr;
  MachineBasicBlock::iterator WhereToSpill;

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
  return {SpillBlock, WhereToSpill};
}

std::pair<MachineBasicBlock *, MachineBasicBlock::iterator>
AMDGPUEarlyRegisterSpilling::getWhereToSpill(MachineInstr *CurMI,
                                             Register CandidateReg) {
  assert(MRI->hasOneDef(CandidateReg) &&
         "The Register does not have one definition");
  MachineInstr *InstrOfCandidateReg = MRI->getOneDef(CandidateReg)->getParent();
  MachineBasicBlock *DefRegMBB = InstrOfCandidateReg->getParent();
  MachineBasicBlock *CurMBB = CurMI->getParent();
  MachineLoop *DefInstrLoop = MLI->getLoopFor(DefRegMBB);
  MachineLoop *CurLoop = MLI->getLoopFor(CurMI->getParent());
  MachineLoop *OutermostLoopOfCurLoop = nullptr;
  // We do not spill inside the loop nest because of the spill overhead. So,
  // we only need to know about the outermost loop.
  if (CurLoop)
    OutermostLoopOfCurLoop = CurLoop->getOutermostLoop();

  MachineBasicBlock *SpillBlock = nullptr;
  MachineBasicBlock::iterator WhereToSpill;
  // case 1:
  // - the register we are about to spill (CandidateReg) is defined in loop
  // - the high register pressure (CurMI) is outside the loop
  // - we emit the spill instruction in the exit block of the loop
  // TODO: improve spilling in loops
  if (DefInstrLoop && !CurLoop) {
    std::tie(SpillBlock, WhereToSpill) =
        getWhereToSpillIfDefintionInLoop(CurMI, DefRegMBB);
  }
  // case 2:
  // - the register we are about to spill is outside the loop
  // - the high register pressure instruction (CurMI) is inside the loop
  // - we emit the spill instruction in the loop preheader
  else if (!DefInstrLoop && CurLoop) {
    MachineBasicBlock *LoopPreheader =
        OutermostLoopOfCurLoop->getLoopPreheader();
    assert(LoopPreheader && "There is not loop preheader");
    WhereToSpill = LoopPreheader->getFirstTerminator();
    if (WhereToSpill == LoopPreheader->end())
      WhereToSpill = LoopPreheader->instr_end();
    SpillBlock = LoopPreheader;
  }
  // case 3:
  // - the register that we are about to spill and
  // - the high register pressure point are in different loops.
  // - we emit the spill instruction in the exit block of the loop or the loop
  // preheader
  else if ((DefInstrLoop && CurLoop &&
            (!DefInstrLoop->contains(CurLoop) &&
             !OutermostLoopOfCurLoop->contains(DefInstrLoop)))) {
    std::tie(SpillBlock, WhereToSpill) =
        getWhereToSpillIfDefintionInLoop(CurMI, DefRegMBB);
    if (!SpillBlock) {
      MachineBasicBlock *LoopPreheader =
          OutermostLoopOfCurLoop->getLoopPreheader();
      assert(LoopPreheader && "There is not loop preheader");
      WhereToSpill = LoopPreheader->getFirstTerminator();
      if (WhereToSpill == LoopPreheader->end())
        WhereToSpill = LoopPreheader->back();
      SpillBlock = LoopPreheader;
    }
  }
  // case 4:
  // - the high register pressure instruction is a PHI node
  // - we emit the spill instruction before the first non-PHI instruction
  else if (CurMI->isPHI()) {
    WhereToSpill = CurMBB->getFirstNonPHI();
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

/// Normalize the next-use distance and restore cost for all spill candidates
/// using log-scale normalization.
static void normalizeCosts(
    SmallVector<std::unique_ptr<SpillOrRestoreCandidate>> &AllCandidates,
    const SIRegisterInfo *TRI) {
  if (AllCandidates.empty())
    return;

  int64_t MinRestoreCost = AllCandidates[0]->getRestoreCost();
  int64_t MaxRestoreCost = AllCandidates[0]->getRestoreCost();
  NextUseDistance MinNextUseDist = AllCandidates[0]->getNextUseDistance();
  NextUseDistance MaxNextUseDist = AllCandidates[0]->getNextUseDistance();

  for (const auto &C : AllCandidates) {
    MinRestoreCost = std::min(MinRestoreCost, C->getRestoreCost());
    MaxRestoreCost = std::max(MaxRestoreCost, C->getRestoreCost());
    MinNextUseDist = llvm::min(MinNextUseDist, C->getNextUseDistance());
    MaxNextUseDist = llvm::max(MaxNextUseDist, C->getNextUseDistance());
  }

  LLVM_DEBUG(dbgs() << "------------------------------------------------\n");
  LLVM_DEBUG(dbgs() << "RestoreCost (min=" << MinRestoreCost
                    << ", max=" << MaxRestoreCost << ")\n");
  LLVM_DEBUG({
    dbgs() << "NextUseDist (min=";
    MinNextUseDist.print(dbgs());
    dbgs() << ", max=";
    MaxNextUseDist.print(dbgs());
    dbgs() << ")\n";
  });

  // Log-scale normalization.
  static constexpr int64_t Limit = 100;
  double LogMaxNextUseDist = MaxNextUseDist.logSpanFrom(MinNextUseDist);
  double LogMaxRestoreCost = std::log(MaxRestoreCost - MinRestoreCost + 1);

  for (auto &C : AllCandidates) {
    // Log-scale normalization for NextUseDistance.
    double LogNextUseDist = C->getNextUseDistance().logSpanFrom(MinNextUseDist);
    int64_t NormalizedNextUseDist =
        (LogMaxNextUseDist > 0)
            ? static_cast<int64_t>((LogNextUseDist * Limit) / LogMaxNextUseDist)
            : 0;

    // Log-scale normalization for RestoreCost.
    double LogRestoreCost = std::log(C->getRestoreCost() - MinRestoreCost + 1);
    int64_t NormalizedRestoreCost =
        (LogMaxRestoreCost > 0)
            ? static_cast<int64_t>((LogRestoreCost * Limit) / LogMaxRestoreCost)
            : 0;

    // // Outlier penalty for RestoreCost (higher cost is worse for spilling).
    // bool IsRestoreCostOutlier =
    //     C->getRestoreCost() > MeanRestoreCost + 2 * StdDevRestoreCost;
    // if (IsRestoreCostOutlier)
    //   NormalizedRestoreCost += OutlierBonus;

    C->setNormalizedRestoreCost(NormalizedRestoreCost);

    // Combined cost: prefer high next-use distance, low restore cost.
    int64_t NormalizedCost =
        NormalizedNextUseDist * 0.8 - NormalizedRestoreCost * 0.2;
    C->setNormalizedCost(NormalizedCost);

    LLVM_DEBUG(dbgs() << "Register " << printReg(C->getCandidateRegister(), TRI)
                      << " with NormalizedCost = " << NormalizedCost
                      << " , NormalizedRestoreCost = " << NormalizedRestoreCost
                      << " , NormalizedNextUseDist = " << NormalizedNextUseDist
                      << "\n");
  }
}

void AMDGPUEarlyRegisterSpilling::spill(MachineInstr *CurMI,
                                        GCNDownwardRPTracker &RPTracker,
                                        unsigned NumOfSpills) {
  // CurMI is the high register pressure point in the IR.

  // How we select which register to spill
  // -------------------------------------
  // The spill candidate registers are sorted by next-use-distance are placed in
  // the FinalCandidates vector. We get the top N (LiveRegsWindow) candidates
  // and sort them based on a secondary metric, the restore cost. We select the
  // best register using that metric and emit the spill and restore
  // instructions. This is the default CodeGenPlan which is named
  // EmitSpillRestore.
  //
  // There are two more code gen plans for optimizing restore placement.
  // If EnableRestoreOptimization flag is enabled, the candidate registers might
  // also be registers which are defined in restore instructions. When we emit
  // the restore instructions, we do not know if the uses will be in a high
  // register pressure area. We try to optimize the restore instructions e.g.
  // group together uses that dominate one another and emit only one restore
  // instruction for the head of the group or if a use is inside the loop, we
  // emit the restore in loop preheader. If we found out that the above restore
  // optimizations increase the pressure, then we do one of the following:
  // - CodeGenPlan::MoveRestoreInsideTheLoop : move the restore instruction from
  // the loop preheader to the use inside the loop.
  // - CodeGenPlan::EmitNewRestoreBeforeUse : This breaks the live range from
  // the Head of the group to the use(s) in the high register pressure area.
  //
  SmallVector<std::unique_ptr<SpillOrRestoreCandidate>> FinalCandidates;
  SmallVector<RegisterSpillCandidate> InitialVectorOfCandidates =
      getCandidates(CurMI, RPTracker);
  static constexpr unsigned LiveRegsWindow = 50;
  unsigned NumOfCandidates = std::min(
      (unsigned)InitialVectorOfCandidates.size(), NumOfSpills + LiveRegsWindow);
  ArrayRef<RegisterSpillCandidate> Candidates(
      InitialVectorOfCandidates.begin(),
      InitialVectorOfCandidates.begin() + NumOfCandidates);
  for (const auto &Candidate : Candidates) {
    Register CandidateReg = Candidate.Reg;
    NextUseDistance NextUseDist = Candidate.Dist;
    LaneBitmask Mask = Candidate.Mask;
    // TODO: Check if this is needed.
    unsigned NumOfCoveredRegs = SIRegisterInfo::getNumCoveredRegs(Mask);
    unsigned NumOfSubregisters = TRI->getRegSizeInBits(CandidateReg, *MRI) / 32;
    if (NumOfCoveredRegs != NumOfSubregisters)
      continue;

    MachineBasicBlock *SpillBlock = nullptr;
    MachineBasicBlock::iterator WhereToSpill;
    MachineInstr *InstrOfCandidateReg =
        MRI->getOneDef(CandidateReg)->getParent();
    // If the candidate register is defined in a restore instruction, we try
    // optimize the restore
    if (TII->isVGPRSpill(InstrOfCandidateReg->getOpcode()) &&
        InstrOfCandidateReg->mayLoad() && EnableRestoreOptimization) {
      auto It1 = RestoreRegToDomGroup.find(CandidateReg);
      if (It1 == RestoreRegToDomGroup.end())
        continue;

      DomGroup DG = It1->second;
      MachineInstr *Head = DG.getHead();
      MachineInstr *OrigRestore = DG.getRestore();
      MachineBasicBlock *CurMBB = CurMI->getParent();

      if (DG.getWhereToRestore() == DomGroup::RestorePlacement::LoopPreheader &&
          (CurMI == Head ||
           (DT->dominates(CurMI, Head) && !DT->dominates(Head, CurMI)))) {

        // Create Candidate information.
        auto Candidate = std::make_unique<RestoreCandidate>(
            OrigRestore->getOperand(0).getReg(), Mask,
            RestoreCandidate::CodeGenPlan::MoveRestoreInsideTheLoop, TRI, MRI,
            TII, FrameInfo, LIS, Indexes, DT, MLI);

        Candidate->addGroup(DG);

        // Calculate the restore cost.
        Candidate->calculateRestoreCost();
        Candidate->setNextUseDistance(NextUseDist);
        LLVM_DEBUG(dbgs() << "Restore cost for register = "
                          << printReg(CandidateReg, TRI) << " = "
                          << Candidate->getRestoreCost() << "\n");
        FinalCandidates.push_back(std::move(Candidate));
      } else {
        SetVectorType UsesDominatedByCurMI;
        for (MachineInstr *U : DG.getUses()) {
          MachineBasicBlock *UMBB = U->getParent();
          if (U == CurMI) {
            UsesDominatedByCurMI.insert(U);
          } else if (CurMBB == UMBB && DT->dominates(CurMI, U)) {
            UsesDominatedByCurMI.insert(U);
          } else if (CurMBB != UMBB && DT->dominates(CurMBB, UMBB)) {
            UsesDominatedByCurMI.insert(U);
          }
        }

        // Group the uses together.
        SmallVector<DomGroup> GroupOfUses;
        groupUses(CandidateReg, CurMBB, CurMI, UsesDominatedByCurMI,
                  GroupOfUses);

        auto Candidate = std::make_unique<RestoreCandidate>(
            OrigRestore->getOperand(0).getReg(), Mask,
            RestoreCandidate::CodeGenPlan::EmitNewRestoreBeforeUse, TRI, MRI,
            TII, FrameInfo, LIS, Indexes, DT, MLI);

        for (DomGroup &G : GroupOfUses)
          Candidate->addGroup(G);

        // Calculate the restore cost.
        Candidate->calculateRestoreCost();
        Candidate->setNextUseDistance(NextUseDist);
        LLVM_DEBUG(dbgs() << "Restore cost for register = "
                          << printReg(CandidateReg, TRI) << " = "
                          << Candidate->getRestoreCost() << "\n");
        FinalCandidates.push_back(std::move(Candidate));
      }
    } else {

      // To get the restore cost we first need to find where we should emit the
      // spill instruction.
      std::tie(SpillBlock, WhereToSpill) = getWhereToSpill(CurMI, CandidateReg);
      if (SpillBlock == nullptr)
        continue;

      // The next step is to check if there are any uses which are reachable
      // from the SpillBlock. In this case, we have to emit the spill in the
      // common dominator of the SpillBlock and the blocks of the reachable
      // uses.

      // The dominated uses are the ones that are dominated by the SpillBlock.
      SetVectorType DominatedUses;
      // The reachable uses are the ones that can be reached by the SpillBlock.
      SetVectorType NonDominatedReachableUses;
      // The unreachable uses are the ones that are not reachable by the
      // SpillBlock.
      SetVectorType UnreachableUses;
      classifyUses(SpillBlock, CandidateReg, CurMI, DominatedUses,
                   NonDominatedReachableUses, UnreachableUses);

      if (NonDominatedReachableUses.empty() && DominatedUses.empty() &&
          !UnreachableUses.empty()) {
        continue;
      }

      MachineBasicBlock *CommonDominatorToSpill = nullptr;
      if (!NonDominatedReachableUses.empty()) {
        CommonDominatorToSpill = findCommonDominatorToSpill(
            SpillBlock, CandidateReg, NonDominatedReachableUses);

        // If there are non-dominated reachable uses and we could not find a
        // common dominator, then we skip this regitster. The reason is that the
        // spill and the uses should be executing with either equal execution
        // masks or the spill should have a superset mask compared to the uses.
        if (!CommonDominatorToSpill)
          continue;

        for (auto *U : NonDominatedReachableUses)
          DominatedUses.insert(U);

        SpillBlock = CommonDominatorToSpill;
        WhereToSpill = SpillBlock->getFirstTerminator();
        if (WhereToSpill == SpillBlock->end())
          WhereToSpill = SpillBlock->instr_end();

        MachineInstr *InstrOfCandidateReg =
            MRI->getOneDef(CandidateReg)->getParent();
        MachineBasicBlock *DefBlock = InstrOfCandidateReg->getParent();
        if (DefBlock == SpillBlock && !InstrOfCandidateReg->isPHI() &&
            // Try to avoid to spill after an instruction that uses hardware
            // wait counters.
            (!TII->isVMEM(*InstrOfCandidateReg) &&
             !TII->isSMRD(*InstrOfCandidateReg) &&
             !TII->isDS(*InstrOfCandidateReg) &&
             !TII->isEXP(*InstrOfCandidateReg) &&
             (!TII->isFLAT(*InstrOfCandidateReg) ||
              (!TII->mayAccessVMEMThroughFlat(*InstrOfCandidateReg) &&
               !TII->mayAccessLDSThroughFlat(*InstrOfCandidateReg))))) {
          WhereToSpill = InstrOfCandidateReg->getNextNode()->getIterator();
          if (WhereToSpill == SpillBlock->end())
            WhereToSpill = SpillBlock->instr_end();
        }
        if (!SpillBlock->empty() && SpillBlock->front().isPHI()) {
          auto FirstNonPHI = SpillBlock->getFirstNonPHI();
          if (FirstNonPHI != SpillBlock->end() &&
              WhereToSpill != SpillBlock->end()) {
            for (auto It = SpillBlock->begin(); It != FirstNonPHI; ++It) {
              if (It == WhereToSpill) {
                WhereToSpill = FirstNonPHI;
                break;
              }
            }
          }
        }
      }

      // Find the restore locations.
      SmallVector<DomGroup> GroupOfUses;
      groupUses(CandidateReg, SpillBlock, CurMI, DominatedUses, GroupOfUses);

      // Create Candidate information.
      assert(!MLI->getLoopFor(SpillBlock) &&
             "There should not be a spill loop.");
      auto Candidate = std::make_unique<SpillCandidate>(
          CandidateReg, Mask, SpillCandidate::CodeGenPlan::EmitSpillRestore,
          TRI, MRI, TII, FrameInfo, LIS, Indexes, DT, MLI, SpillBlock,
          WhereToSpill);

      for (DomGroup &G : GroupOfUses)
        Candidate->addGroup(G);

      // Calculate the restore cost.
      Candidate->calculateRestoreCost();
      Candidate->setNextUseDistance(NextUseDist);
      LLVM_DEBUG(dbgs() << "Restore cost for register = "
                        << printReg(CandidateReg, TRI) << " = "
                        << Candidate->getRestoreCost() << "\n");
      FinalCandidates.push_back(std::move(Candidate));
    }
  }

  // Normalize restore costs and next-use distances.
  normalizeCosts(FinalCandidates, TRI);

  llvm::sort(FinalCandidates, [&](auto &Candidate1, auto &Candidate2) {
    int64_t Cost1 = Candidate1->getNormalizedCost();
    int64_t Cost2 = Candidate2->getNormalizedCost();
    if (Cost1 == Cost2)
      return Candidate1->getNormalizedRestoreCost() <
             Candidate2->getNormalizedRestoreCost();
    return Cost1 > Cost2;
  });

  unsigned SpillCnt = 0;
  for (auto &C : FinalCandidates) {
    if (SpillCnt >= NumOfSpills)
      break;

    Register CandidateReg = C->getCandidateRegister();
    unsigned NumOfCoveredRegs = SIRegisterInfo::getNumCoveredRegs(C->getMask());
    unsigned NumOfSubregisters =
        (TRI->getRegSizeInBits(CandidateReg, *MRI)) / 32;
    assert(NumOfCoveredRegs == NumOfSubregisters &&
           "The number of the sub-registers is different than number of "
           "covered registers.");

    SpillCnt += NumOfCoveredRegs;

    SpilledRegs.insert(CandidateReg);
    NumOfERSSpills++;

    C->generateSpillRestoreInstrs(CurMI, RestoreRegToDomGroup);
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
