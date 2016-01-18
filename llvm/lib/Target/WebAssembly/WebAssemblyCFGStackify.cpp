//===-- WebAssemblyCFGStackify.cpp - CFG Stackification -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements a CFG stacking pass.
///
/// This pass reorders the blocks in a function to put them into a reverse
/// post-order [0], with special care to keep the order as similar as possible
/// to the original order, and to keep loops contiguous even in the case of
/// split backedges.
///
/// Then, it inserts BLOCK and LOOP markers to mark the start of scopes, since
/// scope boundaries serve as the labels for WebAssembly's control transfers.
///
/// This is sufficient to convert arbitrary CFGs into a form that works on
/// WebAssembly, provided that all loops are single-entry.
///
/// [0] https://en.wikipedia.org/wiki/Depth-first_search#Vertex_orderings
///
//===----------------------------------------------------------------------===//

#include "WebAssembly.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssemblySubtarget.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-cfg-stackify"

namespace {
class WebAssemblyCFGStackify final : public MachineFunctionPass {
  const char *getPassName() const override {
    return "WebAssembly CFG Stackify";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineLoopInfo>();
    AU.addPreserved<MachineLoopInfo>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

public:
  static char ID; // Pass identification, replacement for typeid
  WebAssemblyCFGStackify() : MachineFunctionPass(ID) {}
};
} // end anonymous namespace

char WebAssemblyCFGStackify::ID = 0;
FunctionPass *llvm::createWebAssemblyCFGStackify() {
  return new WebAssemblyCFGStackify();
}

static void EliminateMultipleEntryLoops(MachineFunction &MF,
                                        const MachineLoopInfo &MLI) {
  SmallPtrSet<MachineBasicBlock *, 8> InSet;
  for (scc_iterator<MachineFunction *> I = scc_begin(&MF), E = scc_end(&MF);
       I != E; ++I) {
    const std::vector<MachineBasicBlock *> &CurrentSCC = *I;

    // Skip trivial SCCs.
    if (CurrentSCC.size() == 1)
      continue;

    InSet.insert(CurrentSCC.begin(), CurrentSCC.end());
    MachineBasicBlock *Header = nullptr;
    for (MachineBasicBlock *MBB : CurrentSCC) {
      for (MachineBasicBlock *Pred : MBB->predecessors()) {
        if (InSet.count(Pred))
          continue;
        if (!Header) {
          Header = MBB;
          break;
        }
        // TODO: Implement multiple-entry loops.
        report_fatal_error("multiple-entry loops are not supported yet");
      }
    }
    assert(MLI.isLoopHeader(Header));

    InSet.clear();
  }
}

namespace {
/// Post-order traversal stack entry.
struct POStackEntry {
  MachineBasicBlock *MBB;
  SmallVector<MachineBasicBlock *, 0> Succs;

  POStackEntry(MachineBasicBlock *MBB, MachineFunction &MF,
               const MachineLoopInfo &MLI);
};
} // end anonymous namespace

POStackEntry::POStackEntry(MachineBasicBlock *MBB, MachineFunction &MF,
                           const MachineLoopInfo &MLI)
    : MBB(MBB), Succs(MBB->successors()) {
  // RPO is not a unique form, since at every basic block with multiple
  // successors, the DFS has to pick which order to visit the successors in.
  // Sort them strategically (see below).
  MachineLoop *Loop = MLI.getLoopFor(MBB);
  MachineFunction::iterator Next = next(MachineFunction::iterator(MBB));
  MachineBasicBlock *LayoutSucc = Next == MF.end() ? nullptr : &*Next;
  std::stable_sort(
      Succs.begin(), Succs.end(),
      [=, &MLI](const MachineBasicBlock *A, const MachineBasicBlock *B) {
        if (A == B)
          return false;

        // Keep loops contiguous by preferring the block that's in the same
        // loop.
        MachineLoop *LoopA = MLI.getLoopFor(A);
        MachineLoop *LoopB = MLI.getLoopFor(B);
        if (LoopA == Loop && LoopB != Loop)
          return true;
        if (LoopA != Loop && LoopB == Loop)
          return false;

        // Minimize perturbation by preferring the block which is the immediate
        // layout successor.
        if (A == LayoutSucc)
          return true;
        if (B == LayoutSucc)
          return false;

        // TODO: More sophisticated orderings may be profitable here.

        return false;
      });
}

/// Sort the blocks in RPO, taking special care to make sure that loops are
/// contiguous even in the case of split backedges.
static void SortBlocks(MachineFunction &MF, const MachineLoopInfo &MLI) {
  // Note that we do our own RPO rather than using
  // "llvm/ADT/PostOrderIterator.h" because we want control over the order that
  // successors are visited in (see above). Also, we can sort the blocks in the
  // MachineFunction as we go.
  SmallPtrSet<MachineBasicBlock *, 16> Visited;
  SmallVector<POStackEntry, 16> Stack;

  MachineBasicBlock *Entry = MF.begin();
  Visited.insert(Entry);
  Stack.push_back(POStackEntry(Entry, MF, MLI));

  for (;;) {
    POStackEntry &Entry = Stack.back();
    SmallVectorImpl<MachineBasicBlock *> &Succs = Entry.Succs;
    if (!Succs.empty()) {
      MachineBasicBlock *Succ = Succs.pop_back_val();
      if (Visited.insert(Succ).second)
        Stack.push_back(POStackEntry(Succ, MF, MLI));
      continue;
    }

    // Put the block in its position in the MachineFunction.
    MachineBasicBlock &MBB = *Entry.MBB;
    MBB.moveBefore(MF.begin());

    // Branch instructions may utilize a fallthrough, so update them if a
    // fallthrough has been added or removed.
    if (!MBB.empty() && MBB.back().isTerminator() && !MBB.back().isBranch() &&
        !MBB.back().isBarrier())
      report_fatal_error(
          "Non-branch terminator with fallthrough cannot yet be rewritten");
    if (MBB.empty() || !MBB.back().isTerminator() || MBB.back().isBranch())
      MBB.updateTerminator();

    Stack.pop_back();
    if (Stack.empty())
      break;
  }

  // Now that we've sorted the blocks in RPO, renumber them.
  MF.RenumberBlocks();

#ifndef NDEBUG
  for (auto &MBB : MF)
    if (MachineLoop *Loop = MLI.getLoopFor(&MBB)) {
      // Assert that loops are contiguous.
      assert(Loop->getHeader() == Loop->getTopBlock());
      assert((Loop->getHeader() == &MBB ||
              Loop->contains(
                  MLI.getLoopFor(prev(MachineFunction::iterator(&MBB))))) &&
             "Loop isn't contiguous");
    } else {
      // Assert that non-loops have no backedge predecessors.
      for (auto Pred : MBB.predecessors())
        assert(Pred->getNumber() < MBB.getNumber() &&
               "CFG still has multiple-entry loops");
    }
#endif
}

/// Insert BLOCK markers at appropriate places.
static void PlaceBlockMarkers(MachineBasicBlock &MBB, MachineBasicBlock &Succ,
                              MachineFunction &MF, const MachineLoopInfo &MLI,
                              const WebAssemblyInstrInfo &TII) {
  // Backward branches are loop backedges, and we place the LOOP markers
  // separately. So only consider forward branches here.
  if (Succ.getNumber() <= MBB.getNumber())
    return;

  // Place the BLOCK for a forward branch. For simplicity, we just insert
  // blocks immediately inside loop boundaries.
  MachineLoop *Loop = MLI.getLoopFor(&Succ);
  MachineBasicBlock &Header = *(Loop ? Loop->getHeader() : &MF.front());
  MachineBasicBlock::iterator InsertPos = Header.begin(), End = Header.end();
  if (InsertPos != End) {
    if (InsertPos->getOpcode() == WebAssembly::LOOP)
      ++InsertPos;
    int SuccNumber = Succ.getNumber();
    // Position the BLOCK in nesting order.
    for (; InsertPos != End && InsertPos->getOpcode() == WebAssembly::BLOCK;
         ++InsertPos) {
      int N = InsertPos->getOperand(0).getMBB()->getNumber();
      if (N < SuccNumber)
        break;
      // If there's already a BLOCK for Succ, we don't need another.
      if (N == SuccNumber)
        return;
    }
  }

  BuildMI(Header, InsertPos, DebugLoc(), TII.get(WebAssembly::BLOCK))
      .addMBB(&Succ);
}

/// Insert LOOP and BLOCK markers at appropriate places.
static void PlaceMarkers(MachineFunction &MF, const MachineLoopInfo &MLI,
                         const WebAssemblyInstrInfo &TII) {
  for (auto &MBB : MF) {
    // Place the LOOP for loops.
    if (MachineLoop *Loop = MLI.getLoopFor(&MBB))
      if (Loop->getHeader() == &MBB) {
        // The operand of a LOOP is the first block after the loop. If the loop
        // is the bottom of the function, insert a dummy block at the end.
        MachineBasicBlock *Bottom = Loop->getBottomBlock();
        auto Iter = next(MachineFunction::iterator(Bottom));
        if (Iter == MF.end()) {
          MF.push_back(MF.CreateMachineBasicBlock());
          Iter = next(MachineFunction::iterator(Bottom));
        }
        BuildMI(MBB, MBB.begin(), DebugLoc(), TII.get(WebAssembly::LOOP))
            .addMBB(Iter);
      }

    // Check for forward branches and switches that need BLOCKS placed.
    for (auto &Term : MBB.terminators())
      for (auto &MO : Term.operands())
        if (MO.isMBB())
          PlaceBlockMarkers(MBB, *MO.getMBB(), MF, MLI, TII);
  }
}

bool WebAssemblyCFGStackify::runOnMachineFunction(MachineFunction &MF) {
  DEBUG(dbgs() << "********** CFG Stackifying **********\n"
                  "********** Function: "
               << MF.getName() << '\n');

  const auto &MLI = getAnalysis<MachineLoopInfo>();
  const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();

  // RPO sorting needs all loops to be single-entry.
  EliminateMultipleEntryLoops(MF, MLI);

  // Sort the blocks in RPO, with contiguous loops.
  SortBlocks(MF, MLI);

  // Place the BLOCK and LOOP markers to indicate the beginnings of scopes.
  PlaceMarkers(MF, MLI, TII);

  return true;
}
