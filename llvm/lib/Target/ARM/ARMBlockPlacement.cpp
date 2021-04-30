//===-- ARMBlockPlacement.cpp - ARM block placement pass ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass re-arranges machine basic blocks to suit target requirements.
// Currently it only moves blocks to fix backwards WLS branches.
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "ARMBaseInstrInfo.h"
#include "ARMBasicBlockInfo.h"
#include "ARMSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"

using namespace llvm;

#define DEBUG_TYPE "arm-block-placement"
#define DEBUG_PREFIX "ARM Block Placement: "

namespace llvm {
class ARMBlockPlacement : public MachineFunctionPass {
private:
  const ARMBaseInstrInfo *TII;
  std::unique_ptr<ARMBasicBlockUtils> BBUtils = nullptr;
  MachineLoopInfo *MLI = nullptr;

public:
  static char ID;
  ARMBlockPlacement() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;
  void moveBasicBlock(MachineBasicBlock *BB, MachineBasicBlock *After);
  bool blockIsBefore(MachineBasicBlock *BB, MachineBasicBlock *Other);
  bool fixBackwardsWLS(MachineLoop *ML);
  bool processPostOrderLoops(MachineLoop *ML);

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineLoopInfo>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // namespace llvm

FunctionPass *llvm::createARMBlockPlacementPass() {
  return new ARMBlockPlacement();
}

char ARMBlockPlacement::ID = 0;

INITIALIZE_PASS(ARMBlockPlacement, DEBUG_TYPE, "ARM block placement", false,
                false)

static MachineInstr *findWLSInBlock(MachineBasicBlock *MBB) {
  for (auto &Terminator : MBB->terminators()) {
    if (Terminator.getOpcode() == ARM::t2WhileLoopStartLR)
      return &Terminator;
  }
  return nullptr;
}

/// Find t2WhileLoopStartLR in the loop predecessor BB or otherwise in its only
/// predecessor. If found, returns (BB, WLS Instr) pair, otherwise a null pair.
static MachineInstr *findWLS(MachineLoop *ML) {
  MachineBasicBlock *Predecessor = ML->getLoopPredecessor();
  if (!Predecessor)
    return nullptr;
  MachineInstr *WlsInstr = findWLSInBlock(Predecessor);
  if (WlsInstr)
    return WlsInstr;
  if (Predecessor->pred_size() == 1)
    return findWLSInBlock(*Predecessor->pred_begin());
  return nullptr;
}

/// Checks if loop has a backwards branching WLS, and if possible, fixes it.
/// This requires checking the preheader (or it's predecessor) for a WLS and if
/// its target is before it.
/// If moving the target block wouldn't produce another backwards WLS or a new
/// forwards LE branch, then move the target block after the preheader (or it's
/// predecessor).
bool ARMBlockPlacement::fixBackwardsWLS(MachineLoop *ML) {
  MachineInstr *WlsInstr = findWLS(ML);
  if (!WlsInstr)
    return false;

  MachineBasicBlock *Predecessor = WlsInstr->getParent();
  MachineBasicBlock *LoopExit = WlsInstr->getOperand(2).getMBB();
  // We don't want to move the function's entry block.
  if (!LoopExit->getPrevNode())
    return false;
  if (blockIsBefore(Predecessor, LoopExit))
    return false;
  LLVM_DEBUG(dbgs() << DEBUG_PREFIX << "Found a backwards WLS from "
                    << Predecessor->getFullName() << " to "
                    << LoopExit->getFullName() << "\n");

  // Make sure that moving the target block doesn't cause any of its WLSs
  // that were previously not backwards to become backwards
  bool CanMove = true;
  MachineInstr *WlsInLoopExit = findWLSInBlock(LoopExit);
  if (WlsInLoopExit) {
    // An example loop structure where the LoopExit can't be moved, since
    // bb1's WLS will become backwards once it's moved after bb3
    // bb1:          - LoopExit
    //      WLS bb2
    // bb2:          - LoopExit2
    //      ...
    // bb3:          - Predecessor
    //      WLS bb1
    // bb4:          - Header
    MachineBasicBlock *LoopExit2 = WlsInLoopExit->getOperand(2).getMBB();
    // If the WLS from LoopExit to LoopExit2 is already backwards then
    // moving LoopExit won't affect it, so it can be moved. If LoopExit2 is
    // after the Predecessor then moving will keep it as a forward branch, so it
    // can be moved. If LoopExit2 is between the Predecessor and LoopExit then
    // moving LoopExit will make it a backwards branch, so it can't be moved
    // since we'd fix one and introduce one backwards branch.
    // TODO: Analyse the blocks to make a decision if it would be worth
    // moving LoopExit even if LoopExit2 is between the Predecessor and
    // LoopExit.
    if (!blockIsBefore(LoopExit2, LoopExit) &&
        (LoopExit2 == Predecessor || blockIsBefore(LoopExit2, Predecessor))) {
      LLVM_DEBUG(dbgs() << DEBUG_PREFIX
                        << "Can't move the target block as it would "
                           "introduce a new backwards WLS branch\n");
      CanMove = false;
    }
  }

  if (CanMove) {
    // Make sure no LEs become forwards.
    // An example loop structure where the LoopExit can't be moved, since
    // bb2's LE will become forwards once bb1 is moved after bb3.
    // bb1:           - LoopExit
    // bb2:
    //      LE  bb1  - Terminator
    // bb3:          - Predecessor
    //      WLS bb1
    // bb4:          - Header
    for (auto It = LoopExit->getIterator(); It != Predecessor->getIterator();
         It++) {
      MachineBasicBlock *MBB = &*It;
      for (auto &Terminator : MBB->terminators()) {
        if (Terminator.getOpcode() != ARM::t2LoopEnd &&
            Terminator.getOpcode() != ARM::t2LoopEndDec)
          continue;
        MachineBasicBlock *LETarget = Terminator.getOperand(2).getMBB();
        // The LE will become forwards branching if it branches to LoopExit
        // which isn't allowed by the architecture, so we should avoid
        // introducing these.
        // TODO: Analyse the blocks to make a decision if it would be worth
        // moving LoopExit even if we'd introduce a forwards LE
        if (LETarget == LoopExit) {
          LLVM_DEBUG(dbgs() << DEBUG_PREFIX
                            << "Can't move the target block as it would "
                               "introduce a new forwards LE branch\n");
          CanMove = false;
          break;
        }
      }
    }
  }

  if (CanMove)
    moveBasicBlock(LoopExit, Predecessor);

  return CanMove;
}

/// Updates ordering (of WLS BB and their loopExits) in inner loops first
/// Returns true if any change was made in any of the loops
bool ARMBlockPlacement::processPostOrderLoops(MachineLoop *ML) {
  bool Changed = false;
  for (auto *InnerML : *ML)
    Changed |= processPostOrderLoops(InnerML);
  return Changed | fixBackwardsWLS(ML);
}

bool ARMBlockPlacement::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;
  const ARMSubtarget &ST = static_cast<const ARMSubtarget &>(MF.getSubtarget());
  if (!ST.hasLOB())
    return false;
  LLVM_DEBUG(dbgs() << DEBUG_PREFIX << "Running on " << MF.getName() << "\n");
  MLI = &getAnalysis<MachineLoopInfo>();
  TII = static_cast<const ARMBaseInstrInfo *>(ST.getInstrInfo());
  BBUtils = std::unique_ptr<ARMBasicBlockUtils>(new ARMBasicBlockUtils(MF));
  MF.RenumberBlocks();
  BBUtils->computeAllBlockSizes();
  BBUtils->adjustBBOffsetsAfter(&MF.front());
  bool Changed = false;

  // Find loops with a backwards branching WLS and fix if possible.
  for (auto *ML : *MLI)
    Changed |= processPostOrderLoops(ML);

  return Changed;
}

bool ARMBlockPlacement::blockIsBefore(MachineBasicBlock *BB,
                                      MachineBasicBlock *Other) {
  return BBUtils->getOffsetOf(Other) > BBUtils->getOffsetOf(BB);
}

/// Moves a given MBB to be positioned after another MBB while maintaining
/// existing control flow
void ARMBlockPlacement::moveBasicBlock(MachineBasicBlock *BB,
                                       MachineBasicBlock *After) {
  LLVM_DEBUG(dbgs() << DEBUG_PREFIX << "Moving " << BB->getName() << " after "
                    << After->getName() << "\n");
  MachineBasicBlock *BBPrevious = BB->getPrevNode();
  assert(BBPrevious && "Cannot move the function entry basic block");
  MachineBasicBlock *AfterNext = After->getNextNode();
  MachineBasicBlock *BBNext = BB->getNextNode();

  BB->moveAfter(After);

  // Since only the blocks are to be moved around (but the control flow must
  // not change), if there were any fall-throughs (to/from adjacent blocks),
  // replace with unconditional branch to the fall through block.
  auto FixFallthrough = [&](MachineBasicBlock *From, MachineBasicBlock *To) {
    LLVM_DEBUG(dbgs() << DEBUG_PREFIX << "Checking for fallthrough from "
                      << From->getName() << " to " << To->getName() << "\n");
    assert(From->isSuccessor(To) &&
           "'To' is expected to be a successor of 'From'");
    MachineInstr &Terminator = *(--From->terminators().end());
    if (!Terminator.isUnconditionalBranch()) {
      // The BB doesn't have an unconditional branch so it relied on
      // fall-through. Fix by adding an unconditional branch to the moved BB.
      unsigned BrOpc =
          BBUtils->isBBInRange(&Terminator, To, 254) ? ARM::tB : ARM::t2B;
      MachineInstrBuilder MIB =
          BuildMI(From, Terminator.getDebugLoc(), TII->get(BrOpc));
      MIB.addMBB(To);
      MIB.addImm(ARMCC::CondCodes::AL);
      MIB.addReg(ARM::NoRegister);
      LLVM_DEBUG(dbgs() << DEBUG_PREFIX << "Adding unconditional branch from "
                        << From->getName() << " to " << To->getName() << ": "
                        << *MIB.getInstr());
    }
  };

  // Fix fall-through to the moved BB from the one that used to be before it.
  if (BBPrevious->isSuccessor(BB))
    FixFallthrough(BBPrevious, BB);
  // Fix fall through from the destination BB to the one that used to follow.
  if (AfterNext && After->isSuccessor(AfterNext))
    FixFallthrough(After, AfterNext);
  // Fix fall through from the moved BB to the one that used to follow.
  if (BBNext && BB->isSuccessor(BBNext))
    FixFallthrough(BB, BBNext);

  BBUtils->adjustBBOffsetsAfter(After);
}
