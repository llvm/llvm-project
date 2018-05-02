//===-- WebAssemblyCFGSort.cpp - CFG Sorting ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a CFG sorting pass.
///
/// This pass reorders the blocks in a function to put them into topological
/// order, ignoring loop backedges, and without any loop being interrupted
/// by a block not dominated by the loop header, with special care to keep the
/// order as similar as possible to the original order.
///
////===----------------------------------------------------------------------===//

#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssembly.h"
#include "WebAssemblySubtarget.h"
#include "WebAssemblyUtilities.h"
#include "llvm/ADT/PriorityQueue.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-cfg-sort"

namespace {
class WebAssemblyCFGSort final : public MachineFunctionPass {
  StringRef getPassName() const override { return "WebAssembly CFG Sort"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineDominatorTree>();
    AU.addPreserved<MachineDominatorTree>();
    AU.addRequired<MachineLoopInfo>();
    AU.addPreserved<MachineLoopInfo>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

public:
  static char ID; // Pass identification, replacement for typeid
  WebAssemblyCFGSort() : MachineFunctionPass(ID) {}
};
} // end anonymous namespace

char WebAssemblyCFGSort::ID = 0;
INITIALIZE_PASS(WebAssemblyCFGSort, DEBUG_TYPE,
                "Reorders blocks in topological order", false, false)

FunctionPass *llvm::createWebAssemblyCFGSort() {
  return new WebAssemblyCFGSort();
}

static void MaybeUpdateTerminator(MachineBasicBlock *MBB) {
#ifndef NDEBUG
  bool AnyBarrier = false;
#endif
  bool AllAnalyzable = true;
  for (const MachineInstr &Term : MBB->terminators()) {
#ifndef NDEBUG
    AnyBarrier |= Term.isBarrier();
#endif
    AllAnalyzable &= Term.isBranch() && !Term.isIndirectBranch();
  }
  assert((AnyBarrier || AllAnalyzable) &&
         "AnalyzeBranch needs to analyze any block with a fallthrough");
  if (AllAnalyzable)
    MBB->updateTerminator();
}

namespace {
/// Sort blocks by their number.
struct CompareBlockNumbers {
  bool operator()(const MachineBasicBlock *A,
                  const MachineBasicBlock *B) const {
    return A->getNumber() > B->getNumber();
  }
};
/// Sort blocks by their number in the opposite order..
struct CompareBlockNumbersBackwards {
  bool operator()(const MachineBasicBlock *A,
                  const MachineBasicBlock *B) const {
    return A->getNumber() < B->getNumber();
  }
};
/// Bookkeeping for a loop to help ensure that we don't mix blocks not dominated
/// by the loop header among the loop's blocks.
struct Entry {
  const MachineLoop *Loop;
  unsigned NumBlocksLeft;

  /// List of blocks not dominated by Loop's header that are deferred until
  /// after all of Loop's blocks have been seen.
  std::vector<MachineBasicBlock *> Deferred;

  explicit Entry(const MachineLoop *L)
      : Loop(L), NumBlocksLeft(L->getNumBlocks()) {}
};
} // end anonymous namespace

/// Sort the blocks, taking special care to make sure that loops are not
/// interrupted by blocks not dominated by their header.
/// TODO: There are many opportunities for improving the heuristics here.
/// Explore them.
static void SortBlocks(MachineFunction &MF, const MachineLoopInfo &MLI,
                       const MachineDominatorTree &MDT) {
  // Prepare for a topological sort: Record the number of predecessors each
  // block has, ignoring loop backedges.
  MF.RenumberBlocks();
  SmallVector<unsigned, 16> NumPredsLeft(MF.getNumBlockIDs(), 0);
  for (MachineBasicBlock &MBB : MF) {
    unsigned N = MBB.pred_size();
    if (MachineLoop *L = MLI.getLoopFor(&MBB))
      if (L->getHeader() == &MBB)
        for (const MachineBasicBlock *Pred : MBB.predecessors())
          if (L->contains(Pred))
            --N;
    NumPredsLeft[MBB.getNumber()] = N;
  }

  // Topological sort the CFG, with additional constraints:
  //  - Between a loop header and the last block in the loop, there can be
  //    no blocks not dominated by the loop header.
  //  - It's desirable to preserve the original block order when possible.
  // We use two ready lists; Preferred and Ready. Preferred has recently
  // processed successors, to help preserve block sequences from the original
  // order. Ready has the remaining ready blocks.
  PriorityQueue<MachineBasicBlock *, std::vector<MachineBasicBlock *>,
                CompareBlockNumbers>
      Preferred;
  PriorityQueue<MachineBasicBlock *, std::vector<MachineBasicBlock *>,
                CompareBlockNumbersBackwards>
      Ready;
  SmallVector<Entry, 4> Loops;
  for (MachineBasicBlock *MBB = &MF.front();;) {
    const MachineLoop *L = MLI.getLoopFor(MBB);
    if (L) {
      // If MBB is a loop header, add it to the active loop list. We can't put
      // any blocks that it doesn't dominate until we see the end of the loop.
      if (L->getHeader() == MBB)
        Loops.push_back(Entry(L));
      // For each active loop the block is in, decrement the count. If MBB is
      // the last block in an active loop, take it off the list and pick up any
      // blocks deferred because the header didn't dominate them.
      for (Entry &E : Loops)
        if (E.Loop->contains(MBB) && --E.NumBlocksLeft == 0)
          for (auto DeferredBlock : E.Deferred)
            Ready.push(DeferredBlock);
      while (!Loops.empty() && Loops.back().NumBlocksLeft == 0)
        Loops.pop_back();
    }
    // The main topological sort logic.
    for (MachineBasicBlock *Succ : MBB->successors()) {
      // Ignore backedges.
      if (MachineLoop *SuccL = MLI.getLoopFor(Succ))
        if (SuccL->getHeader() == Succ && SuccL->contains(MBB))
          continue;
      // Decrement the predecessor count. If it's now zero, it's ready.
      if (--NumPredsLeft[Succ->getNumber()] == 0)
        Preferred.push(Succ);
    }
    // Determine the block to follow MBB. First try to find a preferred block,
    // to preserve the original block order when possible.
    MachineBasicBlock *Next = nullptr;
    while (!Preferred.empty()) {
      Next = Preferred.top();
      Preferred.pop();
      // If X isn't dominated by the top active loop header, defer it until that
      // loop is done.
      if (!Loops.empty() &&
          !MDT.dominates(Loops.back().Loop->getHeader(), Next)) {
        Loops.back().Deferred.push_back(Next);
        Next = nullptr;
        continue;
      }
      // If Next was originally ordered before MBB, and it isn't because it was
      // loop-rotated above the header, it's not preferred.
      if (Next->getNumber() < MBB->getNumber() &&
          (!L || !L->contains(Next) ||
           L->getHeader()->getNumber() < Next->getNumber())) {
        Ready.push(Next);
        Next = nullptr;
        continue;
      }
      break;
    }
    // If we didn't find a suitable block in the Preferred list, check the
    // general Ready list.
    if (!Next) {
      // If there are no more blocks to process, we're done.
      if (Ready.empty()) {
        MaybeUpdateTerminator(MBB);
        break;
      }
      for (;;) {
        Next = Ready.top();
        Ready.pop();
        // If Next isn't dominated by the top active loop header, defer it until
        // that loop is done.
        if (!Loops.empty() &&
            !MDT.dominates(Loops.back().Loop->getHeader(), Next)) {
          Loops.back().Deferred.push_back(Next);
          continue;
        }
        break;
      }
    }
    // Move the next block into place and iterate.
    Next->moveAfter(MBB);
    MaybeUpdateTerminator(MBB);
    MBB = Next;
  }
  assert(Loops.empty() && "Active loop list not finished");
  MF.RenumberBlocks();

#ifndef NDEBUG
  SmallSetVector<MachineLoop *, 8> OnStack;

  // Insert a sentinel representing the degenerate loop that starts at the
  // function entry block and includes the entire function as a "loop" that
  // executes once.
  OnStack.insert(nullptr);

  for (auto &MBB : MF) {
    assert(MBB.getNumber() >= 0 && "Renumbered blocks should be non-negative.");

    MachineLoop *Loop = MLI.getLoopFor(&MBB);
    if (Loop && &MBB == Loop->getHeader()) {
      // Loop header. The loop predecessor should be sorted above, and the other
      // predecessors should be backedges below.
      for (auto Pred : MBB.predecessors())
        assert(
            (Pred->getNumber() < MBB.getNumber() || Loop->contains(Pred)) &&
            "Loop header predecessors must be loop predecessors or backedges");
      assert(OnStack.insert(Loop) && "Loops should be declared at most once.");
    } else {
      // Not a loop header. All predecessors should be sorted above.
      for (auto Pred : MBB.predecessors())
        assert(Pred->getNumber() < MBB.getNumber() &&
               "Non-loop-header predecessors should be topologically sorted");
      assert(OnStack.count(MLI.getLoopFor(&MBB)) &&
             "Blocks must be nested in their loops");
    }
    while (OnStack.size() > 1 && &MBB == LoopBottom(OnStack.back()))
      OnStack.pop_back();
  }
  assert(OnStack.pop_back_val() == nullptr &&
         "The function entry block shouldn't actually be a loop header");
  assert(OnStack.empty() &&
         "Control flow stack pushes and pops should be balanced.");
#endif
}

bool WebAssemblyCFGSort::runOnMachineFunction(MachineFunction &MF) {
  DEBUG(dbgs() << "********** CFG Sorting **********\n"
                  "********** Function: "
               << MF.getName() << '\n');

  const auto &MLI = getAnalysis<MachineLoopInfo>();
  auto &MDT = getAnalysis<MachineDominatorTree>();
  // Liveness is not tracked for VALUE_STACK physreg.
  MF.getRegInfo().invalidateLiveness();

  // Sort the blocks, with contiguous loops.
  SortBlocks(MF, MLI, MDT);

  return true;
}
