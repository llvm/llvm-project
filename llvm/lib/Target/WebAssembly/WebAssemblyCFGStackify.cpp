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
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblySubtarget.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
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

/// Return the "bottom" block of a loop. This differs from
/// MachineLoop::getBottomBlock in that it works even if the loop is
/// discontiguous.
static MachineBasicBlock *LoopBottom(const MachineLoop *Loop) {
  MachineBasicBlock *Bottom = Loop->getHeader();
  for (MachineBasicBlock *MBB : Loop->blocks())
    if (MBB->getNumber() > Bottom->getNumber())
      Bottom = MBB;
  return Bottom;
}

/// Sort the blocks in RPO, taking special care to make sure that loops are
/// contiguous even in the case of split backedges.
///
/// TODO: Determine whether RPO is actually worthwhile, or whether we should
/// move to just a stable-topological-sort-based approach that would preserve
/// more of the original order.
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

/// Test whether Pred has any terminators explicitly branching to MBB, as
/// opposed to falling through. Note that it's possible (eg. in unoptimized
/// code) for a branch instruction to both branch to a block and fallthrough
/// to it, so we check the actual branch operands to see if there are any
/// explicit mentions.
static bool ExplicitlyBranchesTo(MachineBasicBlock *Pred,
                                 MachineBasicBlock *MBB) {
  for (MachineInstr &MI : Pred->terminators())
    for (MachineOperand &MO : MI.explicit_operands())
      if (MO.isMBB() && MO.getMBB() == MBB)
        return true;
  return false;
}

/// Test whether MI is a child of some other node in an expression tree.
static bool IsChild(const MachineInstr *MI,
                    const WebAssemblyFunctionInfo &MFI) {
  if (MI->getNumOperands() == 0)
    return false;
  const MachineOperand &MO = MI->getOperand(0);
  if (!MO.isReg() || MO.isImplicit() || !MO.isDef())
    return false;
  unsigned Reg = MO.getReg();
  return TargetRegisterInfo::isVirtualRegister(Reg) &&
         MFI.isVRegStackified(Reg);
}

/// Insert a BLOCK marker for branches to MBB (if needed).
static void PlaceBlockMarker(MachineBasicBlock &MBB, MachineFunction &MF,
                             SmallVectorImpl<MachineBasicBlock *> &ScopeTops,
                             const WebAssemblyInstrInfo &TII,
                             const MachineLoopInfo &MLI,
                             MachineDominatorTree &MDT,
                             WebAssemblyFunctionInfo &MFI) {
  // First compute the nearest common dominator of all forward non-fallthrough
  // predecessors so that we minimize the time that the BLOCK is on the stack,
  // which reduces overall stack height.
  MachineBasicBlock *Header = nullptr;
  bool IsBranchedTo = false;
  int MBBNumber = MBB.getNumber();
  for (MachineBasicBlock *Pred : MBB.predecessors())
    if (Pred->getNumber() < MBBNumber) {
      Header = Header ? MDT.findNearestCommonDominator(Header, Pred) : Pred;
      if (ExplicitlyBranchesTo(Pred, &MBB))
        IsBranchedTo = true;
    }
  if (!Header)
    return;
  if (!IsBranchedTo)
    return;

  assert(&MBB != &MF.front() && "Header blocks shouldn't have predecessors");
  MachineBasicBlock *LayoutPred = &*prev(MachineFunction::iterator(&MBB));

  // If the nearest common dominator is inside a more deeply nested context,
  // walk out to the nearest scope which isn't more deeply nested.
  for (MachineFunction::iterator I(LayoutPred), E(Header); I != E; --I) {
    if (MachineBasicBlock *ScopeTop = ScopeTops[I->getNumber()]) {
      if (ScopeTop->getNumber() > Header->getNumber()) {
        // Skip over an intervening scope.
        I = next(MachineFunction::iterator(ScopeTop));
      } else {
        // We found a scope level at an appropriate depth.
        Header = ScopeTop;
        break;
      }
    }
  }

  // If there's a loop which ends just before MBB which contains Header, we can
  // reuse its label instead of inserting a new BLOCK.
  for (MachineLoop *Loop = MLI.getLoopFor(LayoutPred);
       Loop && Loop->contains(LayoutPred); Loop = Loop->getParentLoop())
    if (Loop && LoopBottom(Loop) == LayoutPred && Loop->contains(Header))
      return;

  // Decide where in Header to put the BLOCK.
  MachineBasicBlock::iterator InsertPos;
  MachineLoop *HeaderLoop = MLI.getLoopFor(Header);
  if (HeaderLoop && MBB.getNumber() > LoopBottom(HeaderLoop)->getNumber()) {
    // Header is the header of a loop that does not lexically contain MBB, so
    // the BLOCK needs to be above the LOOP.
    InsertPos = Header->begin();
  } else {
    // Otherwise, insert the BLOCK as late in Header as we can, but before the
    // beginning of the local expression tree and any nested BLOCKs.
    InsertPos = Header->getFirstTerminator();
    while (InsertPos != Header->begin() &&
           IsChild(prev(InsertPos), MFI) &&
           prev(InsertPos)->getOpcode() != WebAssembly::LOOP &&
           prev(InsertPos)->getOpcode() != WebAssembly::END_BLOCK &&
           prev(InsertPos)->getOpcode() != WebAssembly::END_LOOP)
      --InsertPos;
  }

  // Add the BLOCK.
  BuildMI(*Header, InsertPos, DebugLoc(), TII.get(WebAssembly::BLOCK));

  // Mark the end of the block.
  InsertPos = MBB.begin();
  while (InsertPos != MBB.end() &&
         InsertPos->getOpcode() == WebAssembly::END_LOOP)
    ++InsertPos;
  BuildMI(MBB, InsertPos, DebugLoc(), TII.get(WebAssembly::END_BLOCK));

  // Track the farthest-spanning scope that ends at this point.
  int Number = MBB.getNumber();
  if (!ScopeTops[Number] ||
      ScopeTops[Number]->getNumber() > Header->getNumber())
    ScopeTops[Number] = Header;
}

/// Insert a LOOP marker for a loop starting at MBB (if it's a loop header).
static void PlaceLoopMarker(
    MachineBasicBlock &MBB, MachineFunction &MF,
    SmallVectorImpl<MachineBasicBlock *> &ScopeTops,
    DenseMap<const MachineInstr *, const MachineBasicBlock *> &LoopTops,
    const WebAssemblyInstrInfo &TII, const MachineLoopInfo &MLI) {
  MachineLoop *Loop = MLI.getLoopFor(&MBB);
  if (!Loop || Loop->getHeader() != &MBB)
    return;

  // The operand of a LOOP is the first block after the loop. If the loop is the
  // bottom of the function, insert a dummy block at the end.
  MachineBasicBlock *Bottom = LoopBottom(Loop);
  auto Iter = next(MachineFunction::iterator(Bottom));
  if (Iter == MF.end()) {
    MachineBasicBlock *Label = MF.CreateMachineBasicBlock();
    // Give it a fake predecessor so that AsmPrinter prints its label.
    Label->addSuccessor(Label);
    MF.push_back(Label);
    Iter = next(MachineFunction::iterator(Bottom));
  }
  MachineBasicBlock *AfterLoop = &*Iter;

  // Mark the beginning of the loop (after the end of any existing loop that
  // ends here).
  auto InsertPos = MBB.begin();
  while (InsertPos != MBB.end() &&
         InsertPos->getOpcode() == WebAssembly::END_LOOP)
    ++InsertPos;
  BuildMI(MBB, InsertPos, DebugLoc(), TII.get(WebAssembly::LOOP));

  // Mark the end of the loop.
  MachineInstr *End = BuildMI(*AfterLoop, AfterLoop->begin(), DebugLoc(),
                              TII.get(WebAssembly::END_LOOP));
  LoopTops[End] = &MBB;

  assert((!ScopeTops[AfterLoop->getNumber()] ||
          ScopeTops[AfterLoop->getNumber()]->getNumber() < MBB.getNumber()) &&
         "With RPO we should visit the outer-most loop for a block first.");
  if (!ScopeTops[AfterLoop->getNumber()])
    ScopeTops[AfterLoop->getNumber()] = &MBB;
}

static unsigned
GetDepth(const SmallVectorImpl<const MachineBasicBlock *> &Stack,
         const MachineBasicBlock *MBB) {
  unsigned Depth = 0;
  for (auto X : reverse(Stack)) {
    if (X == MBB)
      break;
    ++Depth;
  }
  assert(Depth < Stack.size() && "Branch destination should be in scope");
  return Depth;
}

/// Insert LOOP and BLOCK markers at appropriate places.
static void PlaceMarkers(MachineFunction &MF, const MachineLoopInfo &MLI,
                         const WebAssemblyInstrInfo &TII,
                         MachineDominatorTree &MDT,
                         WebAssemblyFunctionInfo &MFI) {
  // For each block whose label represents the end of a scope, record the block
  // which holds the beginning of the scope. This will allow us to quickly skip
  // over scoped regions when walking blocks. We allocate one more than the
  // number of blocks in the function to accommodate for the possible fake block
  // we may insert at the end.
  SmallVector<MachineBasicBlock *, 8> ScopeTops(MF.getNumBlockIDs() + 1);

  // For eacn LOOP_END, the corresponding LOOP.
  DenseMap<const MachineInstr *, const MachineBasicBlock *> LoopTops;

  for (auto &MBB : MF) {
    // Place the LOOP for MBB if MBB is the header of a loop.
    PlaceLoopMarker(MBB, MF, ScopeTops, LoopTops, TII, MLI);

    // Place the BLOCK for MBB if MBB is branched to from above.
    PlaceBlockMarker(MBB, MF, ScopeTops, TII, MLI, MDT, MFI);
  }

  // Now rewrite references to basic blocks to be depth immediates.
  SmallVector<const MachineBasicBlock *, 8> Stack;
  for (auto &MBB : reverse(MF)) {
    for (auto &MI : reverse(MBB)) {
      switch (MI.getOpcode()) {
      case WebAssembly::BLOCK:
        assert(ScopeTops[Stack.back()->getNumber()] == &MBB &&
               "Block should be balanced");
        Stack.pop_back();
        break;
      case WebAssembly::LOOP:
        assert(Stack.back() == &MBB && "Loop top should be balanced");
        Stack.pop_back();
        Stack.pop_back();
        break;
      case WebAssembly::END_BLOCK:
        Stack.push_back(&MBB);
        break;
      case WebAssembly::END_LOOP:
        Stack.push_back(&MBB);
        Stack.push_back(LoopTops[&MI]);
        break;
      default:
        if (MI.isTerminator()) {
          // Rewrite MBB operands to be depth immediates.
          SmallVector<MachineOperand, 4> Ops(MI.operands());
          while (MI.getNumOperands() > 0)
            MI.RemoveOperand(MI.getNumOperands() - 1);
          for (auto MO : Ops) {
            if (MO.isMBB())
              MO = MachineOperand::CreateImm(GetDepth(Stack, MO.getMBB()));
            MI.addOperand(MF, MO);
          }
        }
        break;
      }
    }
  }
  assert(Stack.empty() && "Control flow should be balanced");
}

bool WebAssemblyCFGStackify::runOnMachineFunction(MachineFunction &MF) {
  DEBUG(dbgs() << "********** CFG Stackifying **********\n"
                  "********** Function: "
               << MF.getName() << '\n');

  const auto &MLI = getAnalysis<MachineLoopInfo>();
  auto &MDT = getAnalysis<MachineDominatorTree>();
  // Liveness is not tracked for EXPR_STACK physreg.
  const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();
  WebAssemblyFunctionInfo &MFI = *MF.getInfo<WebAssemblyFunctionInfo>();
  MF.getRegInfo().invalidateLiveness();

  // RPO sorting needs all loops to be single-entry.
  EliminateMultipleEntryLoops(MF, MLI);

  // Sort the blocks in RPO, with contiguous loops.
  SortBlocks(MF, MLI);

  // Place the BLOCK and LOOP markers to indicate the beginnings of scopes.
  PlaceMarkers(MF, MLI, TII, MDT, MFI);

  return true;
}
