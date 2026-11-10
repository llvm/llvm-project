//===- SIWaitcntPadding.cpp - Counter padding hints for waitcnts ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Materialize exact-count padding hints before waitcnt insertion analyzes the
// machine function. The hint itself contributes no counter event and is
// removed once considered, whether or not its request is accepted.
//
//===----------------------------------------------------------------------===//

#include "SIWaitcntPadding.h"
#include "AMDGPUWaitcntUtils.h"
#include "GCNSubtarget.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineDomTreeUpdater.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/Support/Debug.h"
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "si-insert-waitcnts"

namespace {

struct PaddingEdge {
  MachineBasicBlock *Branch;
  MachineBasicBlock *Pred;
  MachineBasicBlock *Join;
};

[[maybe_unused]] void printIgnored(const MachineBasicBlock &Arm,
                                   StringRef Reason) {
  dbgs() << "Ignoring PAD_WAITCNT in " << printMBBReference(Arm) << " of "
         << Arm.getParent()->getName() << ": " << Reason << '\n';
}

std::optional<PaddingEdge> getCounterPaddingEdge(MachineBasicBlock &Arm,
                                                 MachineLoopInfo &MLI,
                                                 const SIInstrInfo &TII) {
  auto Reject = [&Arm](StringRef Reason) -> std::optional<PaddingEdge> {
    LLVM_DEBUG(printIgnored(Arm, Reason));
    return std::nullopt;
  };

  MachineBasicBlock *Branch = Arm.getSinglePredecessor();
  MachineBasicBlock *Join = Arm.getSingleSuccessor();
  if (!Branch || !Join)
    return Reject("arm lacks a single predecessor and successor");
  if (Branch->succ_size() != 2 || Join->pred_size() != 2)
    return Reject("branch or join does not have exactly two edges");
  if (Arm.isEHPad() || Join->isEHPad())
    return Reject("arm or join is an EH pad");

  MachineBasicBlock *Other = nullptr;
  for (MachineBasicBlock *Pred : Join->predecessors())
    if (Pred != &Arm)
      Other = Pred;
  if (!Other || (Other != Branch && (Other->getSinglePredecessor() != Branch ||
                                     Other->getSingleSuccessor() != Join)))
    return Reject("not a triangle or diamond");

  MachineLoop *Loop = MLI.getLoopFor(Join);
  if (MLI.getLoopFor(Branch) != Loop || MLI.getLoopFor(&Arm) != Loop ||
      MLI.getLoopFor(Other) != Loop)
    return Reject("branch, arms and join are not in the same loop");
  if (Loop && (Loop->getHeader() == Join || Loop->isLoopLatch(Join)))
    return Reject("join is a loop header or latch");

  MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
  SmallVector<MachineOperand, 4> Cond;
  if (TII.analyzeBranch(*Branch, TBB, FBB, Cond) || Cond.empty())
    return Reject("branch is not an analyzable conditional branch");
  for (MachineBasicBlock *MBB : {&Arm, Other}) {
    if (MBB == Branch)
      continue;
    for (const MachineInstr &MI : MBB->terminators())
      if (MI.getOpcode() != AMDGPU::S_BRANCH)
        return Reject("arm has a terminator other than S_BRANCH");
  }
  if (Other == Branch && !Branch->canSplitCriticalEdge(Join, &MLI))
    return Reject("branch-to-join edge cannot be split");
  return PaddingEdge{Branch, Other, Join};
}

} // end anonymous namespace

AMDGPU::CounterPaddingResult AMDGPU::expandCounterPaddingHints(
    MachineFunction &MF, MachineLoopInfo &MLI, MachineDominatorTree *MDT,
    MachinePostDominatorTree &PDT, const HardwareLimits &Limits) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIInstrInfo &TII = *ST.getInstrInfo();
  const SIRegisterInfo &TRI = TII.getRegisterInfo();

  struct Request {
    MachineBasicBlock *Arm;
    PaddingEdge Edge;
    unsigned Count;
    DebugLoc DL;
  };
  SmallVector<Request, 8> Requests;
  DenseSet<MachineBasicBlock *> SeenBranches, RepeatedBranches;
  CounterPaddingResult Result;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : make_early_inc_range(MBB.instrs())) {
      if (MI.getOpcode() != AMDGPU::PAD_WAITCNT)
        continue;
      int64_t Counter = MI.getOperand(0).getImm();
      int64_t Count = MI.getOperand(1).getImm();
      if (Counter != 0) {
        LLVM_DEBUG(printIgnored(MBB, "unsupported counter selector"));
      } else if (Count <= 0 || uint64_t(Count) > Limits.LoadcntMax - 1) {
        LLVM_DEBUG(printIgnored(MBB, "count is out of range"));
      } else if (auto Edge = getCounterPaddingEdge(MBB, MLI, TII)) {
        if (!SeenBranches.insert(Edge->Branch).second)
          RepeatedBranches.insert(Edge->Branch);
        Requests.push_back({&MBB, *Edge, unsigned(Count), MI.getDebugLoc()});
      }
      // An ignored hint grants no permission to weaken a wait.
      MI.eraseFromParent();
      Result.Changed = true;
    }
  }

  MachineDomTreeUpdater MDTU(MDT, &PDT,
                             MachineDomTreeUpdater::UpdateStrategy::Eager);
  for (const Request &R : Requests) {
    auto [Branch, Pred, Join] = R.Edge;
    if (RepeatedBranches.contains(Branch)) {
      LLVM_DEBUG(printIgnored(*R.Arm, "branch has multiple padding requests"));
      continue;
    }
    MachineBasicBlock *PadBlock = Pred;
    if (Pred->succ_size() != 1) {
      bool WasLayoutEdge = Pred->isLayoutSuccessor(Join);
      MachineBasicBlock::SplitCriticalEdgeAnalyses Analyses{};
      Analyses.MLI = &MLI;
      PadBlock = Pred->SplitCriticalEdge(Join, Analyses, &MDTU);
      if (!PadBlock) {
        LLVM_DEBUG(printIgnored(*R.Arm, "failed to split branch-to-join edge"));
        continue;
      }
      Result.ChangedCFG = true;
      PadBlock->setSectionID(Pred->getSectionID());
      if (!WasLayoutEdge) {
        // The splitter repairs branches for its initial layout. Move a taken
        // edge out of line and repair the predecessor's fallthrough again.
        MF.splice(MF.end(), PadBlock);
        Pred->updateTerminator(PadBlock);
      }
    }
    PadBlock->addLiveIn(TRI.getExec());
    PadBlock->sortUniqueLiveIns();
    auto InsertPt = PadBlock->getFirstTerminator();
    for (unsigned I = 0; I != R.Count; ++I)
      BuildMI(*PadBlock, InsertPt, R.DL, TII.get(AMDGPU::BUFFER_INV)).addImm(0);
    LLVM_DEBUG(dbgs() << "Padded " << printMBBReference(*PadBlock) << " with "
                      << R.Count << " BUFFER_INV for PAD_WAITCNT in "
                      << printMBBReference(*R.Arm) << " of " << MF.getName()
                      << '\n');
  }
  return Result;
}
