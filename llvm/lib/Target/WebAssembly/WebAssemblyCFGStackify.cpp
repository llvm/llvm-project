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
/// This pass reorders the blocks in a function to put them into topological
/// order, ignoring loop backedges, and without any loop being interrupted
/// by a block not dominated by the loop header, with special care to keep the
/// order as similar as possible to the original order.
///
/// Then, it inserts BLOCK and LOOP markers to mark the start of scopes, since
/// scope boundaries serve as the labels for WebAssembly's control transfers.
///
/// This is sufficient to convert arbitrary CFGs into a form that works on
/// WebAssembly, provided that all loops are single-entry.
///
//===----------------------------------------------------------------------===//

#include "WebAssembly.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblySubtarget.h"
#include "llvm/ADT/PriorityQueue.h"
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
}

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
  // processed sucessors, to help preserve block sequences from the original
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
static bool IsChild(const MachineInstr &MI,
                    const WebAssemblyFunctionInfo &MFI) {
  if (MI.getNumOperands() == 0)
    return false;
  const MachineOperand &MO = MI.getOperand(0);
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
    // the BLOCK needs to be above the LOOP, after any END constructs.
    InsertPos = Header->begin();
    while (InsertPos->getOpcode() != WebAssembly::LOOP)
      ++InsertPos;
  } else {
    // Otherwise, insert the BLOCK as late in Header as we can, but before the
    // beginning of the local expression tree and any nested BLOCKs.
    InsertPos = Header->getFirstTerminator();
    while (InsertPos != Header->begin() && IsChild(*prev(InsertPos), MFI) &&
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
         "With block sorting the outermost loop for a block should be first.");
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

  // Sort the blocks, with contiguous loops.
  SortBlocks(MF, MLI, MDT);

  // Place the BLOCK and LOOP markers to indicate the beginnings of scopes.
  PlaceMarkers(MF, MLI, TII, MDT, MFI);

  return true;
}
