//===- EarlyReturnPass.cpp - Basic Block Code Layout optimization ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Desc HERE
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include <cassert>
#include <iterator>
#include <memory>

using namespace llvm;

#define DEBUG_TYPE "early-return"

STATISTIC(NumEarlyReturn, "Number of early return optimization done");
STATISTIC(NumDeadReturnBlocks, "Number of dead return blocks removed");

namespace {

#define MAX_OPTIMIZE_ATTEMPT 4

class EarlyReturnPass : public MachineFunctionPass {
  /// BasicBlockInfo - It stores the Offset and size (in bytes) for
  /// machine basic blocks
  struct BasicBlockInfo {
    /// Offset - Distance from the beginning of the function to the beginning
    /// of this basic block.
    unsigned Offset = 0;

    /// Size - Size of the basic block in bytes.  If the block contains
    /// inline assembly, this is a worst case estimate.
    ///   It does not account for any alignment padding whether from the
    /// beginning of the block, or from an aligned jump table at the end.
    unsigned Size = 0;

    BasicBlockInfo() = default;
  };

  SmallVector<BasicBlockInfo, 16> BlockInfo;

private:
  MachineFunction *MF = nullptr;
  const TargetInstrInfo *TII = nullptr;
  SmallVector<MachineBasicBlock *, 8> ReturnBlocks;

  /// Perform the early return for the given branch \p MI
  /// whose destination block is out of range.
  bool introduceEarlyReturn(MachineInstr &MI);

  /// Iterate the machine function, initializing the BlockInfo for all blocks
  /// within it.
  void initializeBasicBlockInfo();

  /// Creates and return the newly inserted block after \p AfterBB.
  /// It substitutes out of range \p BranchBB block branching coming
  /// out from parent \p MBB.
  MachineBasicBlock *createEarlyReturnMBB(MachineBasicBlock *MBB,
                                          MachineBasicBlock *BranchBB,
                                          MachineBasicBlock *AfterBB);

  /// Copies machine instruction from \p SrcBB to \p DestBB,
  /// along with the live-ins registers.
  void copyMachineInstrWithLiveness(const MachineBasicBlock &SrcBB,
                                    MachineBasicBlock *DestBB);

  /// Returns true if the distance between \p MI and
  /// \p DestBB can fit in MI's displacement field.
  bool isBlockInRange(const MachineInstr &MI,
                      const MachineBasicBlock &DestBB) const;

  /// Updates the BlockInfo, starting from \p Start block,
  /// to accommodate changes due to any newly inserted block.
  void adjustBlockOffsets(MachineBasicBlock &Start);

  /// Return the current offset of the specified machine
  /// instruction \p MI from the start of the function.
  unsigned getInstrOffset(const MachineInstr &MI) const;

public:
  static char ID;

  EarlyReturnPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &mf) override;
};

} // end anonymous namespace

char EarlyReturnPass::ID = 0;

char &llvm::EarlyReturnPassID = EarlyReturnPass::ID;

INITIALIZE_PASS(EarlyReturnPass, DEBUG_TYPE, "Branch Early Return Block", false,
                false)

bool EarlyReturnPass::introduceEarlyReturn(MachineInstr &MI) {
  DebugLoc DL = MI.getDebugLoc();
  MachineBasicBlock *MBB = MI.getParent();
  MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
  MachineBasicBlock *NewTBB = nullptr, *NewFBB = nullptr;
  SmallVector<MachineOperand, 4> Cond;

  bool UnAnalyzableBranch = TII->analyzeBranch(*MBB, TBB, FBB, Cond);
  bool NeedEarlyReturnForFBB =
      FBB && FBB->isReturnBlock() && !isBlockInRange(MI, *FBB);

  // TODO : Currently, the situation like multiple conditional branch
  // not handled.
  if (UnAnalyzableBranch) {
    LLVM_DEBUG(dbgs() << "Branch is unanylazable in "
                      << printMBBReference(*MBB));
    return false;
  }

  // If Cond is non-empty, along with FBB as nullptr, it implies
  // fall-through is happening via conditional branch. So, NewFBB would be
  // that very block.
  //
  // Hence, NewFBB could be either be fall-through or valid FBB block.
  if (!FBB && !Cond.empty()) {
    NewFBB = &(*std::next(MachineFunction::iterator(MBB)));
  } else {
    NewFBB = FBB;
  }

  NewTBB = createEarlyReturnMBB(MBB, TBB, MBB);
  if (NeedEarlyReturnForFBB) {
    // If needed NewFBB would hold newly inserted block now.
    NewFBB = createEarlyReturnMBB(MBB, FBB, NewTBB);
  }

  // Removing old branch, followed by inserting new branch to newly created
  // blocks. if FBB is null, then fall-through would work fine.
  unsigned &BBSize = BlockInfo[MBB->getNumber()].Size;
  int RemovedSize = 0;
  int NewBrSize = 0;

  TII->removeBranch(*MBB, &RemovedSize);
  if (TBB && !FBB && Cond.empty()) {
    // Do Nothing, fallthorugh would take care.
  } else if (TBB && !FBB && !Cond.empty()) {
    if (!TII->reverseBranchCondition(Cond)) {
      TII->insertBranch(*MBB, NewFBB, nullptr, Cond, DL, &NewBrSize);
    } else {
      TII->insertBranch(*MBB, NewTBB, NewFBB, Cond, DL, &NewBrSize);
    }
  } else {
    assert(TBB && FBB && !Cond.empty());
    if (!TII->reverseBranchCondition(Cond)) {
      TII->insertBranch(*MBB, NewFBB, nullptr, Cond, DL, &NewBrSize);
    } else {
      TII->insertBranch(*MBB, NewTBB, NewFBB, Cond, DL, &NewBrSize);
    }
  }

  BBSize -= RemovedSize;
  BBSize += NewBrSize;

  // update the block offsets to account for newly created blocks.
  adjustBlockOffsets(*MBB);

  return true;
}

void EarlyReturnPass::initializeBasicBlockInfo() {
  BlockInfo.clear();
  BlockInfo.resize(MF->getNumBlockIDs());
  ReturnBlocks.clear();

  // First thing, compute the size of all basic blocks, and see if the function
  // has any inline assembly in it, which would be worst-case scenario.
  for (MachineBasicBlock &MBB : *MF) {
    unsigned &MBBSize = BlockInfo[MBB.getNumber()].Size;
    MBBSize = 0;

    for (const MachineInstr &MI : MBB)
      MBBSize += TII->getInstSizeInBytes(MI);
  }

  // Compute block offsets for all blocks in MF.
  adjustBlockOffsets(*(MF->begin()));
}

MachineBasicBlock *
EarlyReturnPass::createEarlyReturnMBB(MachineBasicBlock *MBB,
                                      MachineBasicBlock *BranchBB,
                                      MachineBasicBlock *AfterBB) {
  // Create new block and insert it after AfterBB.
  MachineBasicBlock *NewBranchBB =
      MF->CreateMachineBasicBlock(MBB->getBasicBlock());
  MF->insert(++AfterBB->getIterator(), NewBranchBB);

  assert(MBB->isSuccessor(BranchBB));
  MBB->replaceSuccessor(BranchBB, NewBranchBB);
  assert(NewBranchBB->succ_empty());

  // Copies MI into new block and add its entry into BlockInfo.
  copyMachineInstrWithLiveness(*BranchBB, NewBranchBB);
  BlockInfo.insert(BlockInfo.begin() + NewBranchBB->getNumber(),
                   BasicBlockInfo());
  BlockInfo[NewBranchBB->getNumber()].Size =
      BlockInfo[BranchBB->getNumber()].Size;

  LLVM_DEBUG(
      dbgs()
      << "Copies Machine instructions : Old return block -> New return block\n"
      << printMBBReference(*BranchBB) << " from "
      << printMBBReference(*NewBranchBB) << " for " << printMBBReference(*MBB)
      << " comes after " << printMBBReference(*AfterBB) << '\n');

  return NewBranchBB;
}

void EarlyReturnPass::copyMachineInstrWithLiveness(
    const MachineBasicBlock &SrcBB, MachineBasicBlock *DestBB) {
  for (const MachineInstr &I : SrcBB) {
    MachineInstr *MI = MF->CloneMachineInstr(&I);

    // Make a copy of the call site info.
    if (I.isCandidateForCallSiteEntry())
      MF->copyCallSiteInfo(&I, MI);

    DestBB->insert(DestBB->end(), MI);
  }

  // Add live-ins from SrcBB to DestBB.
  for (const MachineBasicBlock::RegisterMaskPair &LiveIn : SrcBB.liveins())
    DestBB->addLiveIn(LiveIn);
  DestBB->sortUniqueLiveIns();
}

bool EarlyReturnPass::isBlockInRange(const MachineInstr &MI,
                                     const MachineBasicBlock &DestBB) const {
  int64_t BrOffset = getInstrOffset(MI);
  int64_t DestOffset = BlockInfo[DestBB.getNumber()].Offset;
  int64_t distance = DestOffset - BrOffset;

  if (TII->isBranchOffsetInRange(MI.getOpcode(), distance))
    return true;

  LLVM_DEBUG(dbgs() << "Out of range branch to destination "
                    << printMBBReference(DestBB) << " from "
                    << printMBBReference(*MI.getParent()) << " to "
                    << DestOffset << " offset " << DestOffset - BrOffset << '\t'
                    << MI);

  return false;
}

void EarlyReturnPass::adjustBlockOffsets(MachineBasicBlock &Start) {
  MachineFunction *MF = Start.getParent();

  // Compute the offset immediately following this block. \p MBB is the
  // block after PrevMBB.
  auto postOffset = [&](const BasicBlockInfo &PrevMBBInfo,
                        const MachineBasicBlock &MBB) -> unsigned {
    const unsigned PO = PrevMBBInfo.Offset + PrevMBBInfo.Size;
    const Align Alignment = MBB.getAlignment();
    const Align ParentAlign = MF->getAlignment();
    if (Alignment <= ParentAlign)
      return alignTo(PO, Alignment);

    // The alignment of this MBB is larger than the function's alignment, so we
    // can't tell whether or not it will insert nops. Assume that it will.
    return alignTo(PO, Alignment) + Alignment.value() - ParentAlign.value();
  };

  unsigned PrevNum = Start.getNumber();
  for (auto &MBB :
       make_range(std::next(MachineFunction::iterator(Start)), MF->end())) {
    unsigned Num = MBB.getNumber();
    // Get the offset and known bits at the end of the layout predecessor.
    // Includes the alignment of the current MBB block.
    BlockInfo[Num].Offset = postOffset(BlockInfo[PrevNum], MBB);
    PrevNum = Num;
  }
}

unsigned EarlyReturnPass::getInstrOffset(const MachineInstr &MI) const {
  const MachineBasicBlock *MBB = MI.getParent();

  // The offset is composed of two things: the sum of the sizes of all MBB's
  // before this instruction's block, and the offset from the start of the block
  // it is in.
  unsigned Offset = BlockInfo[MBB->getNumber()].Offset;

  // Sum up the instructions before MI in MBB.
  for (MachineBasicBlock::const_iterator I = MBB->begin(); &*I != &MI; ++I) {
    assert(I != MBB->end() && "Didn't find MI in its own basic block?");
    Offset += TII->getInstSizeInBytes(*I);
  }

  return Offset;
}

bool EarlyReturnPass::runOnMachineFunction(MachineFunction &mf) {
  MF = &mf;
  bool MadeChange = false;

  LLVM_DEBUG(dbgs() << "***** Branch Early Return Started*****\n");

  const TargetSubtargetInfo &ST = MF->getSubtarget();
  TII = ST.getInstrInfo();

  // Renumber all of the machine basic blocks in the function, guaranteeing that
  // the numbers agree with the position of the block in the function.
  MF->RenumberBlocks();

  // Initialize the basicBlock information by scanning the MF at start.
  initializeBasicBlockInfo();

  // Each MBB would require a minimum number of reoptimization
  // attempt to reach most possible optimized state.
  // (implicit assumption : branch is analyzable)
  // <OR> -> Out of Range
  //
  // Case -1 : TBB && !FBB && Cond.empty() ->
  // Max Attempt to solve = 1 {as it eliminates branch in MBB after once.}
  // Ex: MBB : b TBB
  //     .......
  // <OR>TBB :
  // ==> MBB :
  //     NewTBB :
  //
  // Case -2 : TBB && !FBB && !Cond.empty() ->
  // Max Attempt to solve = 2 {as it loops back to intial state in worst
  // case scenario, after third attempt}
  // Ex: MBB : be TBB
  //     FBB :
  // <OR>TBB :
  // ==> MBB    : bne FBB
  //     NewTBB :
  // <OR>FBB    :
  // ==> MBB    : be NewTBB
  //     NewFBB :
  // <OR>NewTBB :
  //
  // Case -3 : TBB && FBB && Cond.empty() ->
  // Max Attempt to solve = 4 {as it loops back to previous state, from
  // which triggering loop re-eval.}
  // Ex: MBB : be TBB
  //         : b  FBB
  //     .......
  // <OR>TBB :
  //     FBB :
  // ==> MBB    : bne FBB
  //     NewTBB :
  //     .......
  // <OR>FBB    :
  // ==> MBB    : be NewTBB
  //     NewFBB :
  // <OR>NewTBB :
  //     .......
  //
  // This last state is as same as initial state of case-2, implying after
  // 2 more attempts, it would saturate.

  // Main Logic performing early return block insertion for given machine
  // function.
  for (MachineBasicBlock &MBB : *MF) {
    if (MBB.isReturnBlock()) {
      ReturnBlocks.push_back(&MBB);
      continue;
    }

    unsigned NumAttempt = 0;
    while (NumAttempt < MAX_OPTIMIZE_ATTEMPT) {
      MachineBasicBlock::iterator Curr = MBB.getFirstTerminator();
      if (Curr == MBB.end())
        break;

      MachineInstr &MI = *Curr;
      if (!MI.isConditionalBranch() && !MI.isUnconditionalBranch())
        break;

      MachineBasicBlock *DestBB = TII->getBranchDestBlock(MI);
      if (DestBB && DestBB->isReturnBlock() && !isBlockInRange(MI, *DestBB)) {
        if (introduceEarlyReturn(MI)) {
          MadeChange = true;
          NumEarlyReturn++;
        } else {
          // If unable to introduce early return (due to unanylazable branch),
          // no benefit of trying it again for MBB.
          break;
        }
      } else {
        // If no out of range Return block found, no need to attempt anymore.
        break;
      }

      NumAttempt++;
    }

    if (NumAttempt == MAX_OPTIMIZE_ATTEMPT) {
      LLVM_DEBUG(dbgs() << "Reached the most optimized possible state for "
                        << printMBBReference(MBB) << '\n');
    }
  }

  // Now, check for dead return block, only if any changes were made.
  if (MadeChange)
    for (MachineBasicBlock *RBB : ReturnBlocks) {
      if (RBB->pred_empty() && !RBB->isMachineBlockAddressTaken()) {
        LLVM_DEBUG(dbgs() << "\nRemoving this block: "
                          << printMBBReference(*RBB));

        assert(RBB->succ_empty() && "Dead block is not a return block");
        // Update call site info.
        for (const MachineInstr &MI : *RBB)
          if (MI.shouldUpdateCallSiteInfo())
            MF->eraseCallSiteInfo(&MI);

        // Remove the block.
        MF->erase(RBB);
        ++NumDeadReturnBlocks;
      }
    }

  BlockInfo.clear();
  ReturnBlocks.clear();

  LLVM_DEBUG(dbgs() << "***** Branch Early Return Ended*****\n");

  return MadeChange;
}