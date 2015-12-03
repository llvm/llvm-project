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
#include "llvm/CodeGen/MachineDominators.h"
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
    AU.addRequired<MachineDominatorTree>();
    AU.addPreserved<MachineDominatorTree>();
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

static bool LoopContains(const MachineLoop *Loop,
                         const MachineBasicBlock *MBB) {
  return Loop ? Loop->contains(MBB) : true;
}

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
        bool LoopContainsA = LoopContains(Loop, A);
        bool LoopContainsB = LoopContains(Loop, B);
        if (LoopContainsA && !LoopContainsB)
          return true;
        if (!LoopContainsA && LoopContainsB)
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

  MachineBasicBlock *EntryBlock = &*MF.begin();
  Visited.insert(EntryBlock);
  Stack.push_back(POStackEntry(EntryBlock, MF, MLI));

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
    MBB.moveBefore(&*MF.begin());

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
      // Assert that all containing loops are contiguous.
      for (MachineLoop *L = Loop; L; L = L->getParentLoop()) {
        if (&MBB == L->getHeader()) {
          assert(&MBB == L->getTopBlock());
        } else {
          assert(&MBB != L->getTopBlock());
          assert(L->contains(
                     MLI.getLoopFor(&*prev(MachineFunction::iterator(&MBB)))) &&
                 "Loop isn't contiguous");
        }
      }
    } else {
      // Assert that non-loops have no backedge predecessors.
      for (auto Pred : MBB.predecessors())
        assert(Pred->getNumber() < MBB.getNumber() &&
               "CFG still has multiple-entry loops");
    }
#endif
}

static unsigned GetLoopDepth(const MachineLoop *Loop) {
  return Loop ? Loop->getLoopDepth() : 0;
}

/// Insert a BLOCK marker for branches to MBB (if needed).
static void PlaceBlockMarkers(MachineBasicBlock &MBB,
                              const WebAssemblyInstrInfo &TII,
                              MachineDominatorTree &MDT,
                              const MachineLoopInfo &MLI) {
  // Place the BLOCK for forward non-fallthrough branches. Put it at the nearest
  // common dominator of all preceding predecesors so that we minimize the time
  // that it's on the stack, which reduces overall stack height.
  MachineBasicBlock *Header = nullptr;
  bool IsBranchedTo = false;
  int MBBNumber = MBB.getNumber();
  for (MachineBasicBlock *Pred : MBB.predecessors())
    if (Pred->getNumber() < MBBNumber) {
      Header = Header ? MDT.findNearestCommonDominator(Header, Pred) : Pred;
      if (!Pred->isLayoutSuccessor(&MBB) ||
          !(Pred->empty() || !Pred->back().isBarrier()))
        IsBranchedTo = true;
    }
  if (!Header)
    return;
  if (!IsBranchedTo)
    return;

  MachineBasicBlock::iterator InsertPos;
  MachineLoop *HeaderLoop = MLI.getLoopFor(Header);
  unsigned MBBLoopDepth = GetLoopDepth(MLI.getLoopFor(&MBB));
  unsigned HeaderLoopDepth = GetLoopDepth(HeaderLoop);
  if (HeaderLoopDepth > MBBLoopDepth) {
    // The nearest common dominating point is more deeply nested. Insert the
    // BLOCK just above the LOOP.
    for (unsigned i = 0; i < HeaderLoopDepth - 1 - MBBLoopDepth; ++i)
      HeaderLoop = HeaderLoop->getParentLoop();
    Header = HeaderLoop->getHeader();
    InsertPos = Header->begin();
    // Don't insert a BLOCK if we can reuse a loop exit label though.
    if (InsertPos != Header->end() &&
        InsertPos->getOpcode() == WebAssembly::LOOP &&
        InsertPos->getOperand(0).getMBB() == &MBB)
      return;
  } else {
    // Insert the BLOCK as late in the block as we can, but before any existing
    // BLOCKs.
    InsertPos = Header->getFirstTerminator();
    while (InsertPos != Header->begin() &&
           std::prev(InsertPos)->getOpcode() == WebAssembly::BLOCK)
      --InsertPos;
  }

  BuildMI(*Header, InsertPos, DebugLoc(), TII.get(WebAssembly::BLOCK))
      .addMBB(&MBB);
}

/// Insert LOOP and BLOCK markers at appropriate places.
static void PlaceMarkers(MachineFunction &MF, const MachineLoopInfo &MLI,
                         const WebAssemblyInstrInfo &TII,
                         MachineDominatorTree &MDT) {
  for (auto &MBB : MF) {
    // Place the LOOP for MBB if MBB is the header of a loop.
    if (MachineLoop *Loop = MLI.getLoopFor(&MBB))
      if (Loop->getHeader() == &MBB) {
        // The operand of a LOOP is the first block after the loop. If the loop
        // is the bottom of the function, insert a dummy block at the end.
        MachineBasicBlock *Bottom = Loop->getBottomBlock();
        auto Iter = next(MachineFunction::iterator(Bottom));
        if (Iter == MF.end()) {
          MachineBasicBlock *Label = MF.CreateMachineBasicBlock();
          // Give it a fake predecessor so that AsmPrinter prints its label.
          Label->addSuccessor(Label);
          MF.push_back(Label);
          Iter = next(MachineFunction::iterator(Bottom));
        }
        BuildMI(MBB, MBB.begin(), DebugLoc(), TII.get(WebAssembly::LOOP))
            .addMBB(&*Iter);

        // Emit a special no-op telling the asm printer that we need a label
        // to close the loop scope, even though the destination is only
        // reachable by fallthrough.
        if (!Bottom->back().isBarrier())
          BuildMI(*Bottom, Bottom->end(), DebugLoc(),
                  TII.get(WebAssembly::LOOP_END));
      }

    // Place the BLOCK for MBB if MBB is branched to from above.
    PlaceBlockMarkers(MBB, TII, MDT, MLI);
  }
}

#ifndef NDEBUG
static bool
IsOnStack(const SmallVectorImpl<std::pair<MachineBasicBlock *, bool>> &Stack,
          const MachineBasicBlock *MBB) {
  for (const auto &Pair : Stack)
    if (Pair.first == MBB)
      return true;
  return false;
}
#endif

bool WebAssemblyCFGStackify::runOnMachineFunction(MachineFunction &MF) {
  DEBUG(dbgs() << "********** CFG Stackifying **********\n"
                  "********** Function: "
               << MF.getName() << '\n');

  const auto &MLI = getAnalysis<MachineLoopInfo>();
  auto &MDT = getAnalysis<MachineDominatorTree>();
  const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();

  // RPO sorting needs all loops to be single-entry.
  EliminateMultipleEntryLoops(MF, MLI);

  // Sort the blocks in RPO, with contiguous loops.
  SortBlocks(MF, MLI);

  // Place the BLOCK and LOOP markers to indicate the beginnings of scopes.
  PlaceMarkers(MF, MLI, TII, MDT);

#ifndef NDEBUG
  // Verify that block and loop beginnings and endings are in LIFO order, and
  // that all references to blocks are to blocks on the stack at the point of
  // the reference.
  SmallVector<std::pair<MachineBasicBlock *, bool>, 0> Stack;
  for (auto &MBB : MF) {
    while (!Stack.empty() && Stack.back().first == &MBB)
      if (Stack.back().second) {
        assert(Stack.size() >= 2);
        Stack.pop_back();
        Stack.pop_back();
      } else {
        assert(Stack.size() >= 1);
        Stack.pop_back();
      }
    for (auto &MI : MBB)
      switch (MI.getOpcode()) {
      case WebAssembly::LOOP:
        Stack.push_back(std::make_pair(&MBB, false));
        Stack.push_back(std::make_pair(MI.getOperand(0).getMBB(), true));
        break;
      case WebAssembly::BLOCK:
        Stack.push_back(std::make_pair(MI.getOperand(0).getMBB(), false));
        break;
      default:
        for (const MachineOperand &MO : MI.explicit_operands())
          if (MO.isMBB())
            assert(IsOnStack(Stack, MO.getMBB()));
        break;
      }
  }
  assert(Stack.empty());
#endif

  return true;
}
