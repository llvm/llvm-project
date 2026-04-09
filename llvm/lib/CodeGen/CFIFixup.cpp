//===------ CFIFixup.cpp - Insert CFI remember/restore instructions -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

// This pass inserts the necessary  instructions to adjust for the inconsistency
// of the call-frame information caused by final machine basic block layout.
// The pass relies in constraints LLVM imposes on the placement of
// save/restore points (cf. ShrinkWrap) and has certain preconditions about
// placement of CFI instructions:
// * For any two CFI instructions of the function prologue one dominates
//   and is post-dominated by the other.
// * The function possibly contains multiple epilogue blocks, where each
//   epilogue block is complete and self-contained, i.e. CSR restore
//   instructions (and the corresponding CFI instructions)
//   are not split across two or more blocks.
// * CFI instructions are not contained in any loops.

// Thus, during execution, at the beginning and at the end of each basic block,
// following the prologue, the function can be in one of two states:
//  - "has a call frame", if the function has executed the prologue, and
//    has not executed any epilogue
//  - "does not have a call frame", if the function has not executed the
//    prologue, or has executed an epilogue
// which can be computed by a single RPO traversal.

// The location of the prologue is determined by finding the first block in the
// reverse traversal which contains CFI instructions.

// In order to accommodate backends which do not generate unwind info in
// epilogues we compute an additional property "strong no call frame on entry",
// which is set for the entry point of the function and for every block
// reachable from the entry along a path that does not execute the prologue. If
// this property holds, it takes precedence over the "has a call frame"
// property.

// From the point of view of the unwind tables, the "has/does not have call
// frame" state at beginning of each block is determined by the state at the end
// of the previous block, in layout order. Where these states differ, we insert
// compensating CFI instructions, which come in two flavours:

//   - CFI instructions, which reset the unwind table state to the initial one.
//     This is done by a target specific hook and is expected to be trivial
//     to implement, for example it could be:
//       .cfi_def_cfa <sp>, 0
//       .cfi_same_value <rN>
//       .cfi_same_value <rN-1>
//       ...
//     where <rN> are the callee-saved registers.
//   - CFI instructions, which reset the unwind table state to the one
//     created by the function prologue. These are
//       .cfi_restore_state
//       .cfi_remember_state
//     In this case we also insert a `.cfi_remember_state` after the last CFI
//     instruction in the function prologue.
//
// Known limitations:
//  * the pass cannot handle an epilogue preceding the prologue in the basic
//    block layout
//  * the pass does not handle functions where SP is used as a frame pointer and
//    SP adjustments up and down are done in different basic blocks (TODO)
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/CFIFixup.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/Target/TargetMachine.h"

#include <iterator>

using namespace llvm;

#define DEBUG_TYPE "cfi-fixup"

char CFIFixup::ID = 0;

INITIALIZE_PASS(CFIFixup, "cfi-fixup",
                "Insert CFI remember/restore state instructions", false, false)
FunctionPass *llvm::createCFIFixup() { return new CFIFixup(); }

static bool isPrologueCFIInstruction(const MachineInstr &MI) {
  return MI.getOpcode() == TargetOpcode::CFI_INSTRUCTION &&
         MI.getFlag(MachineInstr::FrameSetup);
}

static bool containsEpilogue(const MachineBasicBlock &MBB) {
  return llvm::any_of(llvm::reverse(MBB), [](const auto &MI) {
    return MI.getOpcode() == TargetOpcode::CFI_INSTRUCTION &&
           MI.getFlag(MachineInstr::FrameDestroy);
  });
}

static MachineBasicBlock *
findPrologueEnd(MachineFunction &MF, MachineBasicBlock::iterator &PrologueEnd) {
  // Even though we should theoretically traverse the blocks in post-order, we
  // can't encode correctly cases where prologue blocks are not laid out in
  // topological order. Then, assuming topological order, we can just traverse
  // the function in reverse.
  for (MachineBasicBlock &MBB : reverse(MF)) {
    for (MachineInstr &MI : reverse(MBB.instrs())) {
      if (!isPrologueCFIInstruction(MI))
        continue;
      PrologueEnd = std::next(MI.getIterator());
      return &MBB;
    }
  }
  return nullptr;
}

// Represents a basic block's relationship to the call frame. This metadata
// reflects what the state *should* be, which may differ from the actual state
// after final machine basic block layout.
struct BlockFlags {
  bool Reachable : 1;
  bool StrongNoFrameOnEntry : 1;
  bool HasFrameOnEntry : 1;
  bool HasFrameOnExit : 1;
  BlockFlags()
      : Reachable(false), StrongNoFrameOnEntry(false), HasFrameOnEntry(false),
        HasFrameOnExit(false) {}
};

// Most functions will have <= 32 basic blocks.
using BlockFlagsVector = SmallVector<BlockFlags, 32>;

// Computes the frame information for each block in the function. Frame info
// for a block is inferred from its predecessors.
static BlockFlagsVector
computeBlockInfo(const MachineFunction &MF,
                 const MachineBasicBlock *PrologueBlock) {
  BlockFlagsVector BlockInfo(MF.getNumBlockIDs());
  BlockInfo[0].Reachable = true;
  BlockInfo[0].StrongNoFrameOnEntry = true;

  // Compute the presence/absence of frame at each basic block.
  ReversePostOrderTraversal<const MachineBasicBlock *> RPOT(&*MF.begin());
  for (const MachineBasicBlock *MBB : RPOT) {
    BlockFlags &Info = BlockInfo[MBB->getNumber()];

    // Set to true if the current block contains the prologue or the epilogue,
    // respectively.
    bool HasPrologue = MBB == PrologueBlock;
    bool HasEpilogue = false;

    if (Info.HasFrameOnEntry || HasPrologue)
      HasEpilogue = containsEpilogue(*MBB);

    // If the function has a call frame at the entry of the current block or the
    // current block contains the prologue, then the function has a call frame
    // at the exit of the block, unless the block contains the epilogue.
    Info.HasFrameOnExit = (Info.HasFrameOnEntry || HasPrologue) && !HasEpilogue;

    // Set the successors' state on entry.
    for (MachineBasicBlock *Succ : MBB->successors()) {
      BlockFlags &SuccInfo = BlockInfo[Succ->getNumber()];
      SuccInfo.Reachable = true;
      SuccInfo.StrongNoFrameOnEntry |=
          Info.StrongNoFrameOnEntry && !HasPrologue;
      SuccInfo.HasFrameOnEntry = Info.HasFrameOnExit;
    }
  }

  return BlockInfo;
}

// Represents the point within a basic block where we can insert an instruction.
// Note that we need the MachineBasicBlock* as well as the iterator since the
// iterator can point to the end of the block. Instructions are inserted
// *before* the iterator.
struct InsertionPoint {
  MachineBasicBlock *MBB = nullptr;
  MachineBasicBlock::iterator Iterator;
};

// Inserts a `.cfi_remember_state` instruction before PrologueEnd and a
// `.cfi_restore_state` instruction before DstInsertPt. Returns an iterator
// to the first instruction after the inserted `.cfi_restore_state` instruction.
static InsertionPoint
insertRememberRestorePair(const InsertionPoint &RememberInsertPt,
                          const InsertionPoint &RestoreInsertPt) {
  MachineFunction &MF = *RememberInsertPt.MBB->getParent();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();

  // Insert the `.cfi_remember_state` instruction.
  unsigned CFIIndex =
      MF.addFrameInst(MCCFIInstruction::createRememberState(nullptr));
  BuildMI(*RememberInsertPt.MBB, RememberInsertPt.Iterator, DebugLoc(),
          TII.get(TargetOpcode::CFI_INSTRUCTION))
      .addCFIIndex(CFIIndex);

  // Insert the `.cfi_restore_state` instruction.
  CFIIndex = MF.addFrameInst(MCCFIInstruction::createRestoreState(nullptr));

  return {RestoreInsertPt.MBB,
          std::next(BuildMI(*RestoreInsertPt.MBB, RestoreInsertPt.Iterator,
                            DebugLoc(), TII.get(TargetOpcode::CFI_INSTRUCTION))
                        .addCFIIndex(CFIIndex)
                        ->getIterator())};
}

// Copies all CFI instructions before PrologueEnd and inserts them before
// DstInsertPt. Returns the iterator to the first instruction after the
// inserted instructions.
static InsertionPoint cloneCfiPrologue(const InsertionPoint &PrologueEnd,
                                       const InsertionPoint &DstInsertPt) {
  MachineFunction &MF = *DstInsertPt.MBB->getParent();

  auto cloneCfiInstructions = [&](MachineBasicBlock::iterator Begin,
                                  MachineBasicBlock::iterator End) {
    auto ToClone = map_range(
        make_filter_range(make_range(Begin, End), isPrologueCFIInstruction),
        [&](const MachineInstr &MI) { return MF.CloneMachineInstr(&MI); });
    DstInsertPt.MBB->insert(DstInsertPt.Iterator, ToClone.begin(),
                            ToClone.end());
  };

  // Clone all CFI instructions from previous blocks.
  for (auto &MBB : make_range(MF.begin(), PrologueEnd.MBB->getIterator()))
    cloneCfiInstructions(MBB.begin(), MBB.end());
  // Clone all CFI instructions from the final prologue block.
  cloneCfiInstructions(PrologueEnd.MBB->begin(), PrologueEnd.Iterator);
  return DstInsertPt;
}

// Fixes up the CFI instructions in a basic block to be consistent with the
// intended frame state, adding or removing CFI instructions as necessary.
// Returns true if a change was made and false otherwise.
static bool
fixupBlock(MachineBasicBlock &CurrBB, const BlockFlagsVector &BlockInfo,
           SmallDenseMap<MBBSectionID, InsertionPoint> &InsertionPts,
           const InsertionPoint &Prologue) {
  const MachineFunction &MF = *CurrBB.getParent();
  const TargetFrameLowering &TFL = *MF.getSubtarget().getFrameLowering();
  const BlockFlags &Info = BlockInfo[CurrBB.getNumber()];

  if (!Info.Reachable)
    return false;

  // If we don't need to perform full CFI fix up, we only need to fix up the
  // first basic block in the section.
  if (!TFL.enableFullCFIFixup(MF) && !CurrBB.isBeginSection())
    return false;

  // If the previous block and the current block are in the same section,
  // the frame info will propagate from the previous block to the current one.
  const BlockFlags &PrevInfo =
      BlockInfo[std::prev(CurrBB.getIterator())->getNumber()];
  bool HasFrame = PrevInfo.HasFrameOnExit && !CurrBB.isBeginSection();
  bool NeedsFrame = Info.HasFrameOnEntry && !Info.StrongNoFrameOnEntry;

#ifndef NDEBUG
  if (!Info.StrongNoFrameOnEntry) {
    for (auto *Pred : CurrBB.predecessors()) {
      const BlockFlags &PredInfo = BlockInfo[Pred->getNumber()];
      assert((!PredInfo.Reachable ||
              Info.HasFrameOnEntry == PredInfo.HasFrameOnExit) &&
             "Inconsistent call frame state");
    }
  }
#endif

  if (HasFrame == NeedsFrame)
    return false;

  if (!NeedsFrame) {
    // Reset to the state upon function entry.
    TFL.resetCFIToInitialState(CurrBB);
    return true;
  }

  // Reset to the "after prologue" state.
  InsertionPoint &InsertPt = InsertionPts[CurrBB.getSectionID()];
  if (InsertPt.MBB == nullptr) {
    // CurBB is the first block in its section, so there is no "after
    // prologue" state. Clone the CFI instructions from the prologue block
    // to create it.
    InsertPt = cloneCfiPrologue(Prologue, {&CurrBB, CurrBB.begin()});
  } else {
    // There's an earlier block known to have a stack frame. Insert a
    // `.cfi_remember_state` instruction into that block and a
    // `.cfi_restore_state` instruction at the beginning of the current
    // block.
    InsertPt = insertRememberRestorePair(InsertPt, {&CurrBB, CurrBB.begin()});
  }
  return true;
}

bool CFIFixup::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getSubtarget().getFrameLowering()->enableCFIFixup(MF))
    return false;

  if (MF.getNumBlockIDs() < 2)
    return false;

  // Find the prologue and the point where we can issue the first
  // `.cfi_remember_state`.
  MachineBasicBlock::iterator PrologueEnd;
  MachineBasicBlock *PrologueBlock = findPrologueEnd(MF, PrologueEnd);
  if (PrologueBlock == nullptr)
    return false;

  BlockFlagsVector BlockInfo = computeBlockInfo(MF, PrologueBlock);

  // Walk the blocks of the function in "physical" order.
  // Every block inherits the frame state (as recorded in the unwind tables)
  // of the previous block. If the intended frame state is different, insert
  // compensating CFI instructions.
  bool Change = false;
  // `InsertPt[sectionID]` always points to the point in a preceding block where
  // we have to insert a `.cfi_remember_state`, in the case that the current
  // block needs a `.cfi_restore_state`.
  SmallDenseMap<MBBSectionID, InsertionPoint> InsertionPts;
  InsertionPts[PrologueBlock->getSectionID()] = {PrologueBlock, PrologueEnd};

  assert(PrologueEnd != PrologueBlock->begin() &&
         "Inconsistent notion of \"prologue block\"");

  // No point starting before the prologue block.
  // TODO: the unwind tables will still be incorrect if an epilogue physically
  // preceeds the prologue.
  for (MachineBasicBlock &MBB :
       make_range(std::next(PrologueBlock->getIterator()), MF.end())) {
    Change |=
        fixupBlock(MBB, BlockInfo, InsertionPts, {PrologueBlock, PrologueEnd});
  }

  return Change;
}
