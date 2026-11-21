//===- SIWaitcntPadding.cpp - Counter padding hints for waitcnts ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower waitcnt padding hints to actual padding before waitcnt insertion
// analysis.
//
//===----------------------------------------------------------------------===//

#include "SIWaitcntPadding.h"
#include "AMDGPUWaitcntUtils.h"
#include "GCNSubtarget.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineDomTreeUpdater.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/Support/Debug.h"
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "si-insert-waitcnts"

namespace {

[[maybe_unused]] void printIgnored(const MachineBasicBlock &Arm,
                                   StringRef Reason) {
  dbgs() << "Ignoring PAD_WAITCNT in " << printMBBReference(Arm) << " of "
         << Arm.getParent()->getName() << ": " << Reason << '\n';
}

struct PaddingEdge {
  // When padding a triangle then PadBlockOrPred is the pred of the arm
  // requesting the padding. Otherwise it is the block to pad.
  MachineBasicBlock *PadBlockOrPred;
  MachineBasicBlock *Join;
};

std::optional<PaddingEdge> getCounterPaddingEdge(MachineBasicBlock &Arm,
                                                 MachineLoopInfo &MLI,
                                                 const SIInstrInfo &TII) {
  auto Reject = [&Arm](StringRef Reason) -> std::optional<PaddingEdge> {
    LLVM_DEBUG(printIgnored(Arm, Reason));
    return std::nullopt;
  };

  // We're looking for either a triangle:
  //
  //     Branch
  //      |   \
  //      |    \
  //      |     Arm
  //      |    /
  //      |   /
  //     Join
  //
  // or a diamond:
  //
  //      Branch
  //      /   \
  //     /     \
  // Other      Arm
  //     \     /
  //      \   /
  //       Join

  MachineBasicBlock *Branch = Arm.getSinglePredecessor();
  MachineBasicBlock *Join = Arm.getSingleSuccessor();
  if (!Branch || !Join)
    return Reject("arm lacks a single predecessor and successor");
  if (Branch->succ_size() != 2 || Join->pred_size() != 2)
    return Reject("branch or join does not have exactly two edges");
  if (Arm.isEHPad() || Join->isEHPad())
    return Reject("arm or join is an EH pad");

  auto Preds = Join->pred_begin();
  MachineBasicBlock *Other = *Preds != &Arm ? *Preds : *std::next(Preds);
  // Other is Branch from a triangle
  if (Other != Branch && (Other->getSinglePredecessor() != Branch ||
                          Other->getSingleSuccessor() != Join))
    return Reject("not a triangle or diamond");

  // If this is a triangle make sure we can insert a new block
  if (Other == Branch && !Branch->canSplitCriticalEdge(Join, &MLI))
    return Reject("branch-to-join edge cannot be split");
  return PaddingEdge{Other, Join};
}

} // end anonymous namespace

AMDGPU::CounterPaddingResult AMDGPU::expandCounterPaddingHints(
    MachineFunction &MF, MachineLoopInfo &MLI, MachineDominatorTree *MDT,
    MachinePostDominatorTree &PDT, const HardwareLimits &Limits) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIInstrInfo &TII = *ST.getInstrInfo();

  // In future this should encode the padding instruction, too, but for now
  // we only support `buffer_inv 0`.
  struct PaddingRequest {
    MachineBasicBlock *Join;
    unsigned Count;
    DebugLoc DL;
  };
  MapVector<MachineBasicBlock *, PaddingRequest> Requests;
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
      } else if (auto PE = getCounterPaddingEdge(MBB, MLI, TII)) {
        auto [It, Inserted] = Requests.try_emplace(
            PE->PadBlockOrPred,
            PaddingRequest{PE->Join, unsigned(Count), MI.getDebugLoc()});
        if (!Inserted)
          It->second.Count += Count;
      }
      // An ignored hint grants no permission to weaken a wait.
      MI.eraseFromParent();
      Result.Changed = true;
    }
  }

  MachineDomTreeUpdater MDTU(MDT, &PDT,
                             MachineDomTreeUpdater::UpdateStrategy::Eager);
  for (const auto &[PadBlockOrPred, R] : Requests) {
    MachineBasicBlock *PadBlock = PadBlockOrPred;
    // When the matched CFG is a triangle PadBlock will be the pred and have 2
    // succ.
    if (PadBlock->succ_size() != 1) {
      MachineBasicBlock::SplitCriticalEdgeAnalyses Analyses{};
      Analyses.MLI = &MLI;
      PadBlock = PadBlock->SplitCriticalEdge(R.Join, Analyses, &MDTU);
      if (!PadBlock) {
        LLVM_DEBUG(printIgnored(*PadBlockOrPred,
                                "failed to split branch-to-join edge"));
        continue;
      }
      Result.ChangedCFG = true;
    }
    auto InsertPt = PadBlock->getFirstTerminator();
    for (unsigned I = 0; I != R.Count; ++I)
      BuildMI(*PadBlock, InsertPt, R.DL, TII.get(AMDGPU::BUFFER_INV)).addImm(0);
    LLVM_DEBUG(dbgs() << "Padded " << printMBBReference(*PadBlock) << " with "
                      << R.Count << " BUFFER_INV for PAD_WAITCNT in "
                      << MF.getName() << '\n');
  }
  return Result;
}
