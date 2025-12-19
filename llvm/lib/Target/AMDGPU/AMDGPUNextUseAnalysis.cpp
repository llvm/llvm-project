//===---------------------- AMDGPUNextUseAnalysis.cpp  --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUNextUseAnalysis.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "llvm/InitializePasses.h"
#include <limits>
#include <queue>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-next-use-analysis"

static cl::opt<bool> DumpNextUseDistance("dump-next-use-distance",
                                         cl::init(false), cl::Hidden);

bool AMDGPUNextUseAnalysis::isBackedge(MachineBasicBlock *From,
                                       MachineBasicBlock *To) const {
  if (!From->isSuccessor(To))
    return false;
  MachineLoop *Loop1 = MLI->getLoopFor(From);
  MachineLoop *Loop2 = MLI->getLoopFor(To);
  if (!Loop1 || !Loop2 || Loop1 != Loop2)
    return false;
  MachineBasicBlock *LoopHeader = Loop1->getHeader();
  if (To != LoopHeader)
    return false;
  SmallVector<MachineBasicBlock *, 2> Latches;
  Loop1->getLoopLatches(Latches);
  auto It = llvm::find(Latches, From);
  return It != Latches.end();
}

// Calculate the shortest distance between two blocks using Dijkstra algorithm.
std::pair<SmallVector<MachineBasicBlock *>, uint64_t>
AMDGPUNextUseAnalysis::getShortestPath(MachineBasicBlock *FromMBB,
                                       MachineBasicBlock *ToMBB) {
  assert(FromMBB != ToMBB && "The basic blocks should be different.\n");
  DenseSet<MachineBasicBlock *> Visited;
  struct Data {
    MachineBasicBlock *BestPred = nullptr;
    uint64_t ShortestDistance = std::numeric_limits<uint64_t>::max();
  };
  DenseMap<MachineBasicBlock *, Data> MBBData;

  auto Cmp = [&MBBData](MachineBasicBlock *MBB1, MachineBasicBlock *MBB2) {
    return MBBData[MBB1].ShortestDistance > MBBData[MBB2].ShortestDistance;
  };
  std::priority_queue<MachineBasicBlock *, std::vector<MachineBasicBlock *>,
                      decltype(Cmp)>
      Worklist(Cmp);

  Worklist.push(FromMBB);
  MBBData[FromMBB] = {nullptr, 0};

  while (!Worklist.empty()) {
    MachineBasicBlock *CurMBB = Worklist.top();
    Worklist.pop();

    if (!Visited.insert(CurMBB).second)
      continue;

    if (CurMBB == ToMBB) {
      // We found the destination node, build the path ToMBB->...->FromMBB
      SmallVector<MachineBasicBlock *> Path;
      MachineBasicBlock *PathMBB = ToMBB;
      while (PathMBB != nullptr) {
        Path.push_back(PathMBB);
        if (PathMBB == FromMBB)
          break;
        auto It = MBBData.find(PathMBB);
        PathMBB = It != MBBData.end() ? It->second.BestPred : nullptr;
      }
      assert(Path.back() == FromMBB && "Incomplete path!");
      auto *Pred = MBBData[CurMBB].BestPred;
      return {Path, MBBData[Pred].ShortestDistance -
                        MBBData[FromMBB].ShortestDistance};
    }

    auto Pair = MBBData.try_emplace(
        CurMBB, Data{nullptr, std::numeric_limits<uint64_t>::max()});
    uint64_t CurrMBBDist = Pair.first->second.ShortestDistance;

    for (MachineBasicBlock *Succ : CurMBB->successors()) {
      if (isBackedge(CurMBB, Succ))
        continue;

      auto GetEffectiveLoopDepth = [&](MachineBasicBlock *BB) -> unsigned {
        MachineLoop *LoopBB = MLI->getLoopFor(BB);
        unsigned LoopDepth = 0;
        for (MachineLoop *TmpLoop = LoopBB,
                         *End = LoopBB->getOutermostLoop()->getParentLoop();
             TmpLoop != End; TmpLoop = TmpLoop->getParentLoop()) {
          if (TmpLoop->contains(ToMBB))
            continue;
          LoopDepth++;
        }
        return LoopDepth;
      };

      auto GetLoopWeight = [&](MachineBasicBlock *BB) -> uint64_t {

        MachineLoop *LoopBB = MLI->getLoopFor(BB);
        MachineLoop *LoopTo = MLI->getLoopFor(ToMBB);
	if (!LoopBB && !LoopTo)
	  return 0;

        if (LoopBB && LoopTo &&
            (LoopTo->contains(LoopBB) && (LoopTo != LoopBB)))
          return LoopWeight *
                 (MLI->getLoopDepth(BB) - MLI->getLoopDepth(ToMBB));

	if ((LoopBB && LoopTo && LoopBB->contains(LoopTo)))
          return 1;

	if ((!LoopTo && LoopBB) ||
            (LoopBB && LoopTo && !LoopTo->contains(LoopBB)))
          return LoopWeight * GetEffectiveLoopDepth(BB);

        return 0;
      };

      auto GetWeightedSize = [&](MachineBasicBlock *BB) -> uint64_t {
        unsigned LoopWeight = GetLoopWeight(BB);
        if (LoopWeight!=0)
          return BB->size() * LoopWeight;
        return BB->size();
      };
      uint64_t NewSuccDist = CurrMBBDist + GetWeightedSize(Succ);

      auto &[SuccPred, SuccDist] = MBBData[Succ];
      if (NewSuccDist < SuccDist) {
        // We found a better path to Succ, update best predecessor and distance
        SuccPred = CurMBB;
        SuccDist = NewSuccDist;
      }

      Worklist.push(Succ);
    }
  }
  return {{}, std::numeric_limits<uint64_t>::max()};
}

void AMDGPUNextUseAnalysis::calculateShortestPaths(MachineFunction &MF) {
  for (MachineBasicBlock &MBB1 : MF) {
    for (MachineBasicBlock &MBB2 : MF) {
      if (&MBB1 == &MBB2)
        continue;
      ShortestPathTable[std::make_pair(&MBB1, &MBB2)] =
          getShortestPath(&MBB1, &MBB2);
    }
  }
}

uint64_t AMDGPUNextUseAnalysis::calculateShortestDistance(MachineInstr *CurMI,
                                                          MachineInstr *UseMI) {
  MachineBasicBlock *CurMBB = CurMI->getParent();
  MachineBasicBlock *UseMBB = UseMI->getParent();

  if (CurMBB == UseMBB)
    return getInstrId(UseMI) - getInstrId(CurMI);

  uint64_t CurMIDistanceToBBEnd =
      getInstrId(&*(std::prev(CurMBB->instr_end()))) - getInstrId(CurMI);
  uint64_t UseDistanceFromBBBegin =
      getInstrId(UseMI) - getInstrId(&*(UseMBB->instr_begin())) + 1;
  auto Dst = getShortestDistanceFromTable(CurMBB, UseMBB);
  assert(Dst != std::numeric_limits<uint64_t>::max());
  return CurMIDistanceToBBEnd + Dst + UseDistanceFromBBBegin;
}

std::pair<uint64_t, MachineBasicBlock *>
AMDGPUNextUseAnalysis::getShortestDistanceToExitingLatch(
    MachineBasicBlock *CurMBB, MachineLoop *CurLoop) const {
  SmallVector<MachineBasicBlock *, 2> Latches;
  CurLoop->getLoopLatches(Latches);
  uint64_t ShortestDistanceToLatch = std::numeric_limits<uint64_t>::max();
  MachineBasicBlock *ExitingLatch = nullptr;

  for (MachineBasicBlock *LMBB : Latches) {
    if (LMBB == CurMBB)
      return std::make_pair(0, CurMBB);

    uint64_t Dst = getShortestDistanceFromTable(CurMBB, LMBB);
    if (ShortestDistanceToLatch > Dst) {
      ShortestDistanceToLatch = Dst;
      ExitingLatch = LMBB;
    }
  }
  return std::make_pair(ShortestDistanceToLatch, ExitingLatch);
}

std::pair<uint64_t, MachineBasicBlock *>
AMDGPUNextUseAnalysis::getLoopDistanceAndExitingLatch(
    MachineBasicBlock *CurMBB) const {
  MachineLoop *CurLoop = MLI->getLoopFor(CurMBB);
  MachineBasicBlock *LoopHeader = CurLoop->getHeader();
  SmallVector<MachineBasicBlock *, 2> Latches;
  CurLoop->getLoopLatches(Latches);
  bool IsCurLoopLatch = llvm::any_of(
      Latches, [&](MachineBasicBlock *LMBB) { return CurMBB == LMBB; });
  MachineBasicBlock *ExitingLatch = nullptr;
  uint64_t DistanceToLatch = 0;
  uint64_t TotalDistance = 0;

  if (CurLoop->getNumBlocks() == 1)
    return std::make_pair(CurMBB->size(), CurMBB);

  if (CurMBB == LoopHeader) {
    std::tie(DistanceToLatch, ExitingLatch) =
        getShortestDistanceToExitingLatch(CurMBB, CurLoop);
    TotalDistance = LoopHeader->size() + DistanceToLatch + ExitingLatch->size();
    return std::make_pair(TotalDistance, ExitingLatch);
  }

  if (IsCurLoopLatch) {
    TotalDistance = LoopHeader->size() +
                    getShortestDistanceFromTable(LoopHeader, CurMBB) +
                    CurMBB->size();
    return std::make_pair(TotalDistance, CurMBB);
  }

  auto LoopHeaderToCurMBBDistance =
      getShortestDistanceFromTable(LoopHeader, CurMBB);

  std::tie(DistanceToLatch, ExitingLatch) =
      getShortestDistanceToExitingLatch(CurMBB, CurLoop);

  TotalDistance = LoopHeader->size() + LoopHeaderToCurMBBDistance +
                  CurMBB->size() + DistanceToLatch + ExitingLatch->size();
  return std::make_pair(TotalDistance, ExitingLatch);
}

// Calculates the overhead of a loop nest for three cases: 1. the use is outside
// of the current loop, but they share the same loop nest 2. the use is
// outside of the current loop nest and 3. the use is in a parent loop of the
// current loop nest.
std::pair<uint64_t, MachineBasicBlock *>
AMDGPUNextUseAnalysis::getNestedLoopDistanceAndExitingLatch(
    MachineBasicBlock *CurMBB, MachineBasicBlock *UseMBB,
    bool IsUseOutsideOfTheCurrentLoopNest, bool IsUseInParentLoop) {
  MachineLoop *CurLoop = MLI->getLoopFor(CurMBB);
  MachineLoop *UseLoop = MLI->getLoopFor(UseMBB);

  auto GetEffectiveLoopDepth = [&](MachineBasicBlock *BB) -> unsigned {
    MachineLoop *LoopBB = MLI->getLoopFor(BB);
    unsigned LoopDepth = 0;
    for (MachineLoop *TmpLoop = LoopBB,
                     *End = LoopBB->getOutermostLoop()->getParentLoop();
         TmpLoop != End; TmpLoop = TmpLoop->getParentLoop()) {
      if (TmpLoop->contains(UseLoop))
        continue;
      LoopDepth++;
    }
    return LoopDepth;
  };

  auto GetLoopDistance =
      [&](MachineLoop *ML) -> std::pair<uint64_t, MachineBasicBlock *> {
    uint64_t ShortestDistance = 0;
    uint64_t TmpDist = 0;
    MachineBasicBlock *ExitingLatch = nullptr;
    unsigned EffectiveLoopDepth = GetEffectiveLoopDepth(CurMBB);
    unsigned UseLoopDepth =
        !IsUseOutsideOfTheCurrentLoopNest ? MLI->getLoopDepth(UseMBB) : 0;
    if (ML->getNumBlocks() == 1) {
      ShortestDistance = ML->getHeader()->size() *
                         (MLI->getLoopDepth(ML->getHeader()) - UseLoopDepth) *
                         LoopWeight;
      return std::make_pair(ShortestDistance, ML->getLoopLatch());
    }
    std::tie(TmpDist, ExitingLatch) =
        getLoopDistanceAndExitingLatch(ML->getHeader());
    for (MachineBasicBlock *MBB :
         getShortestPathFromTable(ML->getHeader(), ExitingLatch)) {
      if (UseLoopDepth == 0 && EffectiveLoopDepth != 0)
        ShortestDistance +=
            MBB->size() * GetEffectiveLoopDepth(MBB) * LoopWeight;
      else
        ShortestDistance +=
            MBB->size() * (MLI->getLoopDepth(MBB) - UseLoopDepth) * LoopWeight;
    }
    return std::make_pair(ShortestDistance, ExitingLatch);
  };

  if (IsUseOutsideOfTheCurrentLoopNest) {
    MachineLoop *OutermostLoop = CurLoop->getOutermostLoop();
    if (OutermostLoop->contains(UseLoop)) {
      // The CurLoop and the UseLoop are independent and they are in the same
      // loop nest.
      if (MLI->getLoopDepth(CurMBB) <= MLI->getLoopDepth(UseMBB)) {
        return GetLoopDistance(CurLoop);
      } else {
        assert(CurLoop != OutermostLoop && "The loop cannot be the outermost.");
        MachineLoop *OuterLoopOfCurLoop = CurLoop;
        while (OutermostLoop != OuterLoopOfCurLoop &&
               MLI->getLoopDepth(OuterLoopOfCurLoop->getHeader()) !=
                   MLI->getLoopDepth(UseMBB)) {
          OuterLoopOfCurLoop = OuterLoopOfCurLoop->getParentLoop();
        }
        return GetLoopDistance(OuterLoopOfCurLoop);
      }
    } else {
      // We should take into consideration the whole loop nest in the
      // calculation of the distance because we will reach the use after
      // executing the whole loop nest.
      return GetLoopDistance(OutermostLoop);
    }
  } else if (IsUseInParentLoop) {
    MachineLoop *UseLoopSubLoop = nullptr;
    for (MachineLoop *ML : UseLoop->getSubLoopsVector()) {
      // All the sub-loops of the UseLoop will be executed before the use.
      // Hence, we should take this into consideration in distance calculation.
      if (ML->contains(CurLoop)) {
        UseLoopSubLoop = ML;
        break;
      }
    }
    return GetLoopDistance(UseLoopSubLoop);
  }
  llvm_unreachable("Failed to calculate the loop distance!");
}

uint64_t AMDGPUNextUseAnalysis::calculateCurLoopDistance(Register DefReg,
                                                         MachineInstr *CurMI,
                                                         MachineInstr *UseMI) {
  MachineBasicBlock *CurMBB = CurMI->getParent();
  MachineBasicBlock *UseMBB = UseMI->getParent();
  MachineLoop *CurLoop = MLI->getLoopFor(CurMBB);
  MachineLoop *UseLoop = MLI->getLoopFor(UseMBB);
  uint64_t LoopDistance = 0;
  MachineBasicBlock *ExitingLatch = nullptr;
  bool IsUseInParentLoop = CurLoop && UseLoop &&
                           (UseLoop->contains(CurLoop) && (UseLoop != CurLoop));

  bool IsUseOutsideOfTheCurrentLoopNest =
      (!UseLoop && CurLoop) ||
      (CurLoop && UseLoop && !UseLoop->contains(CurLoop) &&
       !CurLoop->contains(UseLoop));

  if (IsUseOutsideOfTheCurrentLoopNest) {
    if (CurLoop->getSubLoops().empty() && CurLoop->isOutermost()) {
      std::tie(LoopDistance, ExitingLatch) =
          getLoopDistanceAndExitingLatch(CurMBB);
      LoopDistance = LoopDistance * LoopWeight;
    } else {
      std::tie(LoopDistance, ExitingLatch) =
          getNestedLoopDistanceAndExitingLatch(CurMBB, UseMBB, true, false);
    }
  } else if (IsUseInParentLoop) {
    assert(MLI->getLoopDepth(UseMBB) < MLI->getLoopDepth(CurMBB) &&
           "The loop depth of the current instruction must be bigger than "
           "these.\n");
    if (isIncomingValFromBackedge(CurMI, UseMI, DefReg))
      return calculateBackedgeDistance(CurMI, UseMI);

    //  Get the loop distance of all the inner loops of UseLoop.
    std::tie(LoopDistance, ExitingLatch) =
        getNestedLoopDistanceAndExitingLatch(CurMBB, UseMBB, false, true);
  }

  uint64_t UseDistanceFromBBBegin = getInstrId(&*(UseMI->getIterator())) -
                                    getInstrId(&*(UseMBB->instr_begin())) + 1;
  return LoopDistance + getShortestDistanceFromTable(ExitingLatch, UseMBB) +
         UseDistanceFromBBBegin;
}

uint64_t AMDGPUNextUseAnalysis::calculateBackedgeDistance(MachineInstr *CurMI,
                                                          MachineInstr *UseMI) {
  MachineBasicBlock *CurMBB = CurMI->getParent();
  MachineBasicBlock *UseMBB = UseMI->getParent();
  MachineLoop *CurLoop = MLI->getLoopFor(CurMBB);
  MachineLoop *UseLoop = MLI->getLoopFor(UseMBB);
  assert(UseLoop && "There is no backedge.\n");
  uint64_t CurMIDistanceToBBEnd =
      getInstrId(&*(std::prev(CurMBB->instr_end()))) - getInstrId(CurMI);
  uint64_t UseDistanceFromBBBegin = getInstrId(&*(UseMI->getIterator())) -
                                    getInstrId(&*(UseMBB->instr_begin())) + 1;

  if (!CurLoop)
    return CurMIDistanceToBBEnd + getShortestDistanceFromTable(CurMBB, UseMBB) +
           UseDistanceFromBBBegin;

  if (CurLoop == UseLoop) {
    auto [DistanceToLatch, ExitingLatch] =
        getShortestDistanceToExitingLatch(CurMBB, CurLoop);
    if (ExitingLatch == CurMBB)
      return CurMIDistanceToBBEnd + UseDistanceFromBBBegin;
    return UseDistanceFromBBBegin + CurMIDistanceToBBEnd + DistanceToLatch +
           ExitingLatch->size();
  }

  if (!CurLoop->contains(UseLoop) && !UseLoop->contains(CurLoop)) {
    auto [LoopDistance, ExitingLatch] = getLoopDistanceAndExitingLatch(CurMBB);
    return LoopDistance + getShortestDistanceFromTable(ExitingLatch, UseMBB) +
           UseDistanceFromBBBegin;
  }

  if (!CurLoop->contains(UseLoop)) {
    auto [InnerLoopDistance, InnerLoopExitingLatch] =
        getNestedLoopDistanceAndExitingLatch(CurMBB, UseMBB, false, true);
    auto [DistanceToLatch, ExitingLatch] =
        getShortestDistanceToExitingLatch(InnerLoopExitingLatch, UseLoop);
    return InnerLoopDistance + DistanceToLatch + ExitingLatch->size() +
           UseDistanceFromBBBegin;
  }

  llvm_unreachable("The backedge distance has not been calculated!");
}

bool AMDGPUNextUseAnalysis::isIncomingValFromBackedge(MachineInstr *CurMI,
                                                      MachineInstr *UseMI,
                                                      Register DefReg) const {
  if (!UseMI->isPHI())
    return false;

  MachineLoop *CurLoop = MLI->getLoopFor(CurMI->getParent());
  MachineLoop *UseLoop = MLI->getLoopFor(UseMI->getParent());

  if (!UseLoop)
    return false;

  if (CurLoop && !UseLoop->contains(CurLoop))
    return false;

  if (UseMI->getParent() != UseLoop->getHeader())
    return false;

  SmallVector<MachineBasicBlock *, 2> Latches;
  UseLoop->getLoopLatches(Latches);

  bool IsNotIncomingValFromLatch = false;
  bool IsIncomingValFromLatch = false;
  auto Ops = UseMI->operands();
  for (auto It = std::next(Ops.begin()), ItE = Ops.end(); It != ItE;
       It = std::next(It, 2)) {
    auto &RegMO = *It;
    auto &MBBMO = *std::next(It);
    assert(RegMO.isReg() && "Expected register operand of PHI");
    assert(MBBMO.isMBB() && "Expected MBB operand of PHI");
    if (RegMO.getReg() == DefReg) {
      MachineBasicBlock *IncomingBB = MBBMO.getMBB();
      auto It = llvm::find(Latches, IncomingBB);
      if (It == Latches.end())
        IsNotIncomingValFromLatch = true;
      else
        IsIncomingValFromLatch = true;
    }
  }
  return IsIncomingValFromLatch && !IsNotIncomingValFromLatch;
}

void AMDGPUNextUseAnalysis::dumpShortestPaths() const {
  for (const auto &P : ShortestPathTable) {
    MachineBasicBlock *From = P.first.first;
    MachineBasicBlock *To = P.first.second;
    auto [ShortestPath, Dist] = P.second;
    errs() << "From: " << From->getName() << "-> To:" << To->getName() << " = "
           << Dist << "\n";
  }
}

void AMDGPUNextUseAnalysis::printAllDistances(MachineFunction &MF) {
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : *&MBB) {
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg())
          continue;

        Register Reg = MO.getReg();
        if (!MO.isReg())
          continue;

        if (MO.isUse())
          continue;

        if (Reg.isPhysical() || TRI->isAGPR(*MRI, Reg))
          continue;

        std::optional<uint64_t> NextUseDistance = getNextUseDistance(Reg);
        errs() << "Next-use distance of Register " << printReg(Reg, TRI)
               << " = ";
        if (NextUseDistance)
          errs() << *NextUseDistance;
        else
          errs() << "null";
        errs() << "\n";
      }
    }
  }
}

// TODO: Remove it. It is only used for testing.
std::optional<uint64_t>
AMDGPUNextUseAnalysis::getNextUseDistance(Register DefReg) {
  assert(!DefReg.isPhysical() && !TRI->isAGPR(*MRI, DefReg) &&
         "Next-use distance is calculated for SGPRs and VGPRs");
  uint64_t NextUseDistance = std::numeric_limits<uint64_t>::max();
  uint64_t CurrentNextUseDistance = std::numeric_limits<uint64_t>::max();
  MachineInstr *CurMI = &*MRI->def_instr_begin(DefReg);
  MachineBasicBlock *CurMBB = CurMI->getParent();
  MachineLoop *CurLoop = MLI->getLoopFor(CurMBB);
  for (auto &UseMI : MRI->use_nodbg_instructions(DefReg)) {
    MachineBasicBlock *UseMBB = UseMI.getParent();
    MachineLoop *UseLoop = MLI->getLoopFor(UseMBB);

    bool IsUseOutsideOfTheDefinitionLoop =
        (CurLoop && !UseLoop) ||
        (CurLoop && UseLoop &&
         ((!UseLoop->contains(CurLoop) && !CurLoop->contains(UseLoop)) ||
          (UseLoop->contains(CurLoop) && (UseLoop != CurLoop))));

    if (IsUseOutsideOfTheDefinitionLoop) {
      CurrentNextUseDistance = calculateCurLoopDistance(DefReg, CurMI, &UseMI);
    } else if (isIncomingValFromBackedge(CurMI, &UseMI, DefReg)) {
      CurrentNextUseDistance = calculateBackedgeDistance(CurMI, &UseMI);
    } else {
      CurrentNextUseDistance = calculateShortestDistance(CurMI, &UseMI);
    }

    if (CurrentNextUseDistance < NextUseDistance)
      NextUseDistance = CurrentNextUseDistance;
  }
  return NextUseDistance != std::numeric_limits<uint64_t>::max()
             ? std::optional<uint64_t>(NextUseDistance)
             : std::nullopt;
}

std::optional<uint64_t>
AMDGPUNextUseAnalysis::getNextUseDistance(Register DefReg, MachineInstr *CurMI,
                                          SmallVector<MachineInstr *> &Uses) {
  assert(!DefReg.isPhysical() && !TRI->isAGPR(*MRI, DefReg) &&
         "Next-use distance is calculated for SGPRs and VGPRs");
  uint64_t NextUseDistance = std::numeric_limits<uint64_t>::max();
  uint64_t CurrentNextUseDistance = std::numeric_limits<uint64_t>::max();
  MachineBasicBlock *CurMBB = CurMI->getParent();
  MachineLoop *CurLoop = MLI->getLoopFor(CurMBB);
  for (auto *UseMI : Uses) {
    MachineBasicBlock *UseMBB = UseMI->getParent();
    MachineLoop *UseLoop = MLI->getLoopFor(UseMBB);

    bool IsUseOutsideOfCurLoop =
        (CurLoop && !UseLoop) ||
        (CurLoop && UseLoop &&
         ((!UseLoop->contains(CurLoop) && !CurLoop->contains(UseLoop)) ||
          (UseLoop->contains(CurLoop) && (UseLoop != CurLoop))));

    if (IsUseOutsideOfCurLoop) {
      CurrentNextUseDistance = calculateCurLoopDistance(DefReg, CurMI, UseMI);
    } else if (isIncomingValFromBackedge(CurMI, UseMI, DefReg)) {
      CurrentNextUseDistance = calculateBackedgeDistance(CurMI, UseMI);
    } else {
      CurrentNextUseDistance = calculateShortestDistance(CurMI, UseMI);
    }

    if (CurrentNextUseDistance < NextUseDistance)
      NextUseDistance = CurrentNextUseDistance;
  }
  return NextUseDistance != std::numeric_limits<uint64_t>::max()
             ? std::optional<uint64_t>(NextUseDistance)
             : std::nullopt;
}

bool AMDGPUNextUseAnalysis::run(MachineFunction &MF,
                                const MachineLoopInfo *MLInfo) {

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  TRI = &TII->getRegisterInfo();
  MLI = MLInfo;
  MRI = &MF.getRegInfo();

  for (MachineBasicBlock &BB : MF) {
    unsigned Id = 0;
    for (MachineInstr &MI : BB) {
      InstrToId[&MI] = ++Id;
    }
  }

  calculateShortestPaths(MF);

  if (DumpNextUseDistance) {
    MF.print(errs());
    printAllDistances(MF);
  }

  return true;
}

bool AMDGPUNextUseAnalysisPass::runOnMachineFunction(MachineFunction &MF) {
  MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  AMDGPUNextUseAnalysis NUA;
  return NUA.run(MF, MLI);
}

char AMDGPUNextUseAnalysisPass::ID = 0;

INITIALIZE_PASS_BEGIN(AMDGPUNextUseAnalysisPass, DEBUG_TYPE,
                      "Next Use Analysis", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LiveVariablesWrapperPass)
INITIALIZE_PASS_END(AMDGPUNextUseAnalysisPass, DEBUG_TYPE, "Next Use Analysis",
                    false, false)

char &llvm::AMDGPUNextUseAnalysisID = AMDGPUNextUseAnalysisPass::ID;

FunctionPass *llvm::createAMDGPUNextUseAnalysisPass() {
  return new AMDGPUNextUseAnalysisPass();
}
