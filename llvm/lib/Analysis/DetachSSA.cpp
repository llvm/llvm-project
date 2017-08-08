//===-- DetachSSA.cpp - Detach SSA Builder---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------===//
//
// This file implements the DetachSSA class.
//
//===----------------------------------------------------------------===//
#include "llvm/Analysis/DetachSSA.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Analysis/IteratedDominanceFrontier.h"
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormattedStream.h"

#define DEBUG_TYPE "detachssa"
using namespace llvm;
INITIALIZE_PASS_BEGIN(DetachSSAWrapperPass, "detachssa", "Detach SSA", false,
                      true)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(DetachSSAWrapperPass, "detachssa", "Detach SSA", false,
                    true)

INITIALIZE_PASS_BEGIN(DetachSSAPrinterLegacyPass, "print-detachssa",
                      "Detach SSA Printer", false, false)
INITIALIZE_PASS_DEPENDENCY(DetachSSAWrapperPass)
INITIALIZE_PASS_END(DetachSSAPrinterLegacyPass, "print-detachssa",
                    "Detach SSA Printer", false, false)

static cl::opt<bool>
    VerifyDetachSSA("verify-detachssa", cl::init(false), cl::Hidden,
                    cl::desc("Verify DetachSSA in legacy printer pass."));

namespace llvm {
/// \brief An assembly annotator class to print Detach SSA information in
/// comments.
class DetachSSAAnnotatedWriter : public AssemblyAnnotationWriter {
  friend class DetachSSA;
  const DetachSSA *DSSA;

public:
  DetachSSAAnnotatedWriter(const DetachSSA *D) : DSSA(D) {}

  virtual void emitBasicBlockStartAnnot(const BasicBlock *BB,
                                        formatted_raw_ostream &OS) {
    if (DetachAccess *DA = DSSA->getDetachAccess(BB))
      OS << "; " << *DA << "\n";
  }

  virtual void emitInstructionAnnot(const Instruction *I,
                                    formatted_raw_ostream &OS) {
    if (DetachAccess *DA = DSSA->getDetachAccess(I))
      OS << "; " << *DA << "\n";
  }
};

struct RenamePassData {
  DomTreeNode *DTN;
  DomTreeNode::const_iterator ChildIt;
  DetachAccess *IncomingVal;

  RenamePassData(DomTreeNode *D, DomTreeNode::const_iterator It,
                 DetachAccess *M)
      : DTN(D), ChildIt(It), IncomingVal(M) {}
  void swap(RenamePassData &RHS) {
    std::swap(DTN, RHS.DTN);
    std::swap(ChildIt, RHS.ChildIt);
    std::swap(IncomingVal, RHS.IncomingVal);
  }
};
} // anonymous namespace

namespace llvm {

void DetachSSA::renameSuccessorPhis(BasicBlock *BB, DetachAccess *IncomingVal,
                                    bool RenameAllUses) {
  // Pass through values to our successors
  for (const BasicBlock *S : successors(BB)) {
    auto It = PerBlockAccesses.find(S);
    // Rename the phi nodes in our successor block
    if (It == PerBlockAccesses.end() || !isa<DetachPhi>(It->second->front()))
      continue;
    AccessList *Accesses = It->second.get();
    auto *Phi = cast<DetachPhi>(&Accesses->front());
    if (RenameAllUses) {
      int PhiIndex = Phi->getBasicBlockIndex(BB);
      assert(PhiIndex != -1 && "Incomplete phi during partial rename");
      Phi->setIncomingValue(PhiIndex, IncomingVal);
    } else
      Phi->addIncoming(IncomingVal, BB);
  }
}

/// \brief Rename a single basic block into DetachSSA form.
/// Uses the standard SSA renaming algorithm.
/// \returns The new incoming value.
DetachAccess *DetachSSA::renameBlock(BasicBlock *BB, DetachAccess *IncomingVal,
                                     bool RenameAllUses) {
  auto It = PerBlockAccesses.find(BB);
  // Skip most processing if the list is empty.
  if (It != PerBlockAccesses.end()) {
    AccessList *Accesses = It->second.get();
    for (DetachAccess &L : *Accesses) {
      if (DetachUseOrDef *DUD = dyn_cast<DetachUseOrDef>(&L)) {
        if (DUD->getDefiningAccess() == nullptr || RenameAllUses)
          DUD->setDefiningAccess(IncomingVal);
        if (isa<DetachDef>(&L))
          IncomingVal = &L;
      } else {
        IncomingVal = &L;
      }
    }
  }
  return IncomingVal;
}

/// \brief This is the standard SSA renaming algorithm.
///
/// We walk the dominator tree in preorder, renaming accesses, and then filling
/// in phi nodes in our successors.
void DetachSSA::renamePass(DomTreeNode *Root, DetachAccess *IncomingVal,
                           SmallPtrSetImpl<BasicBlock *> &Visited,
                           bool SkipVisited, bool RenameAllUses) {
  SmallVector<RenamePassData, 32> WorkStack;
  // Skip everything if we already renamed this block and we are skipping.
  // Note: You can't sink this into the if, because we need it to occur
  // regardless of whether we skip blocks or not.
  bool AlreadyVisited = !Visited.insert(Root->getBlock()).second;
  if (SkipVisited && AlreadyVisited)
    return;

  IncomingVal = renameBlock(Root->getBlock(), IncomingVal, RenameAllUses);
  renameSuccessorPhis(Root->getBlock(), IncomingVal, RenameAllUses);
  WorkStack.push_back({Root, Root->begin(), IncomingVal});

  while (!WorkStack.empty()) {
    DomTreeNode *Node = WorkStack.back().DTN;
    DomTreeNode::const_iterator ChildIt = WorkStack.back().ChildIt;
    IncomingVal = WorkStack.back().IncomingVal;

    if (ChildIt == Node->end()) {
      WorkStack.pop_back();
    } else {
      DomTreeNode *Child = *ChildIt;
      ++WorkStack.back().ChildIt;
      BasicBlock *BB = Child->getBlock();
      // Note: You can't sink this into the if, because we need it to occur
      // regardless of whether we skip blocks or not.
      AlreadyVisited = !Visited.insert(BB).second;
      if (SkipVisited && AlreadyVisited) {
        // We already visited this during our renaming, which can happen when
        // being asked to rename multiple blocks. Figure out the incoming val,
        // which is the last def.
        // Incoming value can only change if there is a block def, and in that
        // case, it's the last block def in the list.
        if (auto *BlockDefs = getWritableBlockDefs(BB))
          IncomingVal = &*BlockDefs->rbegin();
      } else
        IncomingVal = renameBlock(BB, IncomingVal, RenameAllUses);
      renameSuccessorPhis(BB, IncomingVal, RenameAllUses);
      WorkStack.push_back({Child, Child->begin(), IncomingVal});
    }
  }
}

/// \brief This handles unreachable block accesses by deleting phi nodes in
/// unreachable blocks, and marking all other unreachable DetachAccess's as
/// being uses of the live on entry definition.
void DetachSSA::markUnreachableAsLiveOnEntry(BasicBlock *BB) {
  assert(!DT->isReachableFromEntry(BB) &&
         "Reachable block found while handling unreachable blocks");

  // Make sure phi nodes in our reachable successors end up with a
  // LiveOnEntryDef for our incoming edge, even though our block is forward
  // unreachable.  We could just disconnect these blocks from the CFG fully,
  // but we do not right now.
  for (const BasicBlock *S : successors(BB)) {
    if (!DT->isReachableFromEntry(S))
      continue;
    auto It = PerBlockAccesses.find(S);
    // Rename the phi nodes in our successor block
    if (It == PerBlockAccesses.end() || !isa<DetachPhi>(It->second->front()))
      continue;
    AccessList *Accesses = It->second.get();
    auto *Phi = cast<DetachPhi>(&Accesses->front());
    Phi->addIncoming(LiveOnEntryDef.get(), BB);
  }

  auto It = PerBlockAccesses.find(BB);
  if (It == PerBlockAccesses.end())
    return;

  auto &Accesses = It->second;
  for (auto AI = Accesses->begin(), AE = Accesses->end(); AI != AE;) {
    auto Next = std::next(AI);
    // If we have a phi, just remove it. We are going to replace all
    // users with live on entry.
    if (auto *UseOrDef = dyn_cast<DetachUseOrDef>(AI))
      UseOrDef->setDefiningAccess(LiveOnEntryDef.get());
    else
      Accesses->erase(AI);
    AI = Next;
  }
}

DetachSSA::DetachSSA(Function &Func, DominatorTree *DT)
    : DT(DT), F(Func),
      NextID(INVALID_DETACHACCESS_ID) {
  buildDetachSSA();
}

DetachSSA::~DetachSSA() {
  // Drop all our references
  for (const auto &Pair : PerBlockAccesses)
    for (DetachAccess &DA : *Pair.second)
      DA.dropAllReferences();
}

DetachSSA::AccessList *DetachSSA::getOrCreateAccessList(const BasicBlock *BB) {
  auto Res = PerBlockAccesses.insert(std::make_pair(BB, nullptr));

  if (Res.second)
    Res.first->second = make_unique<AccessList>();
  return Res.first->second.get();
}
DetachSSA::DefsList *DetachSSA::getOrCreateDefsList(const BasicBlock *BB) {
  auto Res = PerBlockDefs.insert(std::make_pair(BB, nullptr));

  if (Res.second)
    Res.first->second = make_unique<DefsList>();
  return Res.first->second.get();
}

// /// This class is a batch walker of all DetachUse's in the program, and points
// /// their defining access at the thing that actually clobbers them.  Because it
// /// is a batch walker that touches everything, it does not operate like the
// /// other walkers.  This walker is basically performing a top-down SSA renaming
// /// pass, where the version stack is used as the cache.  This enables it to be
// /// significantly more time and detach efficient than using the regular walker,
// /// which is walking bottom-up.
// class DetachSSA::OptimizeUses {
// public:
//   OptimizeUses(DetachSSA *DSSA, DetachSSAWalker *Walker, AliasAnalysis *AA,
//                DominatorTree *DT)
//       : DSSA(DSSA), Walker(Walker), AA(AA), DT(DT) {
//     Walker = DSSA->getWalker();
//   }

//   void optimizeUses();

// private:
//   /// This represents where a given detachlocation is in the stack.
//   struct MemlocStackInfo {
//     // This essentially is keeping track of versions of the stack. Whenever
//     // the stack changes due to pushes or pops, these versions increase.
//     unsigned long StackEpoch;
//     unsigned long PopEpoch;
//     // This is the lower bound of places on the stack to check. It is equal to
//     // the place the last stack walk ended.
//     // Note: Correctness depends on this being initialized to 0, which densemap
//     // does
//     unsigned long LowerBound;
//     const BasicBlock *LowerBoundBlock;
//     // This is where the last walk for this detach location ended.
//     unsigned long LastKill;
//     bool LastKillValid;
//   };
//   void optimizeUsesInBlock(const BasicBlock *, unsigned long &, unsigned long &,
//                            SmallVectorImpl<DetachAccess *> &,
//                            DenseMap<DetachLocOrCall, MemlocStackInfo> &);
//   DetachSSA *DSSA;
//   DetachSSAWalker *Walker;
//   AliasAnalysis *AA;
//   DominatorTree *DT;
// };

// /// Optimize the uses in a given block This is basically the SSA renaming
// /// algorithm, with one caveat: We are able to use a single stack for all
// /// DetachUses.  This is because the set of *possible* reaching DetachDefs is
// /// the same for every DetachUse.  The *actual* clobbering DetachDef is just
// /// going to be some position in that stack of possible ones.
// ///
// /// We track the stack positions that each DetachLocation needs
// /// to check, and last ended at.  This is because we only want to check the
// /// things that changed since last time.  The same DetachLocation should
// /// get clobbered by the same store (getModRefInfo does not use invariantness or
// /// things like this, and if they start, we can modify DetachLocOrCall to
// /// include relevant data)
// void DetachSSA::OptimizeUses::optimizeUsesInBlock(
//     const BasicBlock *BB, unsigned long &StackEpoch, unsigned long &PopEpoch,
//     SmallVectorImpl<DetachAccess *> &VersionStack,
//     DenseMap<DetachLocOrCall, MemlocStackInfo> &LocStackInfo) {

//   /// If no accesses, nothing to do.
//   DetachSSA::AccessList *Accesses = DSSA->getWritableBlockAccesses(BB);
//   if (Accesses == nullptr)
//     return;

//   // Pop everything that doesn't dominate the current block off the stack,
//   // increment the PopEpoch to account for this.
//   while (true) {
//     assert(
//         !VersionStack.empty() &&
//         "Version stack should have liveOnEntry sentinel dominating everything");
//     BasicBlock *BackBlock = VersionStack.back()->getBlock();
//     if (DT->dominates(BackBlock, BB))
//       break;
//     while (VersionStack.back()->getBlock() == BackBlock)
//       VersionStack.pop_back();
//     ++PopEpoch;
//   }

//   for (DetachAccess &DA : *Accesses) {
//     auto *MU = dyn_cast<DetachUse>(&DA);
//     if (!MU) {
//       VersionStack.push_back(&DA);
//       ++StackEpoch;
//       continue;
//     }

//     if (isUseTriviallyOptimizableToLiveOnEntry(*AA, MU->getDetachInst())) {
//       MU->setDefiningAccess(DSSA->getLiveOnEntryDef(), true);
//       continue;
//     }

//     DetachLocOrCall UseMLOC(MU);
//     auto &LocInfo = LocStackInfo[UseMLOC];
//     // If the pop epoch changed, it means we've removed stuff from top of
//     // stack due to changing blocks. We may have to reset the lower bound or
//     // last kill info.
//     if (LocInfo.PopEpoch != PopEpoch) {
//       LocInfo.PopEpoch = PopEpoch;
//       LocInfo.StackEpoch = StackEpoch;
//       // If the lower bound was in something that no longer dominates us, we
//       // have to reset it.
//       // We can't simply track stack size, because the stack may have had
//       // pushes/pops in the meantime.
//       // XXX: This is non-optimal, but only is slower cases with heavily
//       // branching dominator trees.  To get the optimal number of queries would
//       // be to make lowerbound and lastkill a per-loc stack, and pop it until
//       // the top of that stack dominates us.  This does not seem worth it ATM.
//       // A much cheaper optimization would be to always explore the deepest
//       // branch of the dominator tree first. This will guarantee this resets on
//       // the smallest set of blocks.
//       if (LocInfo.LowerBoundBlock && LocInfo.LowerBoundBlock != BB &&
//           !DT->dominates(LocInfo.LowerBoundBlock, BB)) {
//         // Reset the lower bound of things to check.
//         // TODO: Some day we should be able to reset to last kill, rather than
//         // 0.
//         LocInfo.LowerBound = 0;
//         LocInfo.LowerBoundBlock = VersionStack[0]->getBlock();
//         LocInfo.LastKillValid = false;
//       }
//     } else if (LocInfo.StackEpoch != StackEpoch) {
//       // If all that has changed is the StackEpoch, we only have to check the
//       // new things on the stack, because we've checked everything before.  In
//       // this case, the lower bound of things to check remains the same.
//       LocInfo.PopEpoch = PopEpoch;
//       LocInfo.StackEpoch = StackEpoch;
//     }
//     if (!LocInfo.LastKillValid) {
//       LocInfo.LastKill = VersionStack.size() - 1;
//       LocInfo.LastKillValid = true;
//     }

//     // At this point, we should have corrected last kill and LowerBound to be
//     // in bounds.
//     assert(LocInfo.LowerBound < VersionStack.size() &&
//            "Lower bound out of range");
//     assert(LocInfo.LastKill < VersionStack.size() &&
//            "Last kill info out of range");
//     // In any case, the new upper bound is the top of the stack.
//     unsigned long UpperBound = VersionStack.size() - 1;

//     if (UpperBound - LocInfo.LowerBound > MaxCheckLimit) {
//       DEBUG(dbgs() << "DetachSSA skipping optimization of " << *MU << " ("
//                    << *(MU->getDetachInst()) << ")"
//                    << " because there are " << UpperBound - LocInfo.LowerBound
//                    << " stores to disambiguate\n");
//       // Because we did not walk, LastKill is no longer valid, as this may
//       // have been a kill.
//       LocInfo.LastKillValid = false;
//       continue;
//     }
//     bool FoundClobberResult = false;
//     while (UpperBound > LocInfo.LowerBound) {
//       if (isa<DetachPhi>(VersionStack[UpperBound])) {
//         // For phis, use the walker, see where we ended up, go there
//         Instruction *UseInst = MU->getDetachInst();
//         DetachAccess *Result = Walker->getClobberingDetachAccess(UseInst);
//         // We are guaranteed to find it or something is wrong
//         while (VersionStack[UpperBound] != Result) {
//           assert(UpperBound != 0);
//           --UpperBound;
//         }
//         FoundClobberResult = true;
//         break;
//       }

//       DetachDef *MD = cast<DetachDef>(VersionStack[UpperBound]);
//       // If the lifetime of the pointer ends at this instruction, it's live on
//       // entry.
//       if (!UseMLOC.IsCall && lifetimeEndsAt(MD, UseMLOC.getLoc(), *AA)) {
//         // Reset UpperBound to liveOnEntryDef's place in the stack
//         UpperBound = 0;
//         FoundClobberResult = true;
//         break;
//       }
//       if (instructionClobbersQuery(MD, MU, UseMLOC, *AA)) {
//         FoundClobberResult = true;
//         break;
//       }
//       --UpperBound;
//     }
//     // At the end of this loop, UpperBound is either a clobber, or lower bound
//     // PHI walking may cause it to be < LowerBound, and in fact, < LastKill.
//     if (FoundClobberResult || UpperBound < LocInfo.LastKill) {
//       MU->setDefiningAccess(VersionStack[UpperBound], true);
//       // We were last killed now by where we got to
//       LocInfo.LastKill = UpperBound;
//     } else {
//       // Otherwise, we checked all the new ones, and now we know we can get to
//       // LastKill.
//       MU->setDefiningAccess(VersionStack[LocInfo.LastKill], true);
//     }
//     LocInfo.LowerBound = VersionStack.size() - 1;
//     LocInfo.LowerBoundBlock = BB;
//   }
// }

// /// Optimize uses to point to their actual clobbering definitions.
// void DetachSSA::OptimizeUses::optimizeUses() {
//   SmallVector<DetachAccess *, 16> VersionStack;
//   DenseMap<DetachLocOrCall, MemlocStackInfo> LocStackInfo;
//   VersionStack.push_back(DSSA->getLiveOnEntryDef());

//   unsigned long StackEpoch = 1;
//   unsigned long PopEpoch = 1;
//   // We perform a non-recursive top-down dominator tree walk.
//   for (const auto *DomNode : depth_first(DT->getRootNode()))
//     optimizeUsesInBlock(DomNode->getBlock(), StackEpoch, PopEpoch, VersionStack,
//                         LocStackInfo);
// }

void DetachSSA::placePHINodes(
    const SmallPtrSetImpl<BasicBlock *> &DefiningBlocks,
    const DenseMap<const BasicBlock *, unsigned int> &BBNumbers) {
  // Determine where our DetachPhi's should go
  ForwardIDFCalculator IDFs(*DT);
  IDFs.setDefiningBlocks(DefiningBlocks);
  SmallVector<BasicBlock *, 32> IDFBlocks;
  IDFs.calculate(IDFBlocks);

  std::sort(IDFBlocks.begin(), IDFBlocks.end(),
            [&BBNumbers](const BasicBlock *A, const BasicBlock *B) {
              return BBNumbers.lookup(A) < BBNumbers.lookup(B);
            });

  // Now place DetachPhi nodes.
  for (auto &BB : IDFBlocks)
    createDetachPhi(BB);
}

void DetachSSA::buildDetachSSA() {
  BasicBlock &StartingPoint = F.getEntryBlock();
  LiveOnEntryDef = make_unique<DetachDef>(F.getContext(), nullptr, nullptr,
                                          &StartingPoint, NextID++);
  DenseMap<const BasicBlock *, unsigned int> BBNumbers;
  unsigned NextBBNum = 0;

  // We maintain lists of detach accesses per block, trading memory for time. We
  // could just look up the detach access for every possible instruction in the
  // stream.
  SmallPtrSet<BasicBlock *, 32> DefiningBlocks;
  // Go through each block, figure out where defs occur, and chain together all
  // the accesses.
  for (BasicBlock &B : F) {
    BBNumbers[&B] = NextBBNum++;
    bool InsertIntoDef = false;
    AccessList *Accesses = nullptr;
    DefsList *Defs = nullptr;
    if (isa<SyncInst>(B.getTerminator()) ||
        isa<DetachInst>(B.getTerminator())) {
      DetachUseOrDef *DUD = new DetachDef(B.getContext(), nullptr,
                                          B.getTerminator(), &B,
                                          NextID++);
      ValueToDetachAccess[B.getTerminator()] = DUD;

      if (!Accesses)
        Accesses = getOrCreateAccessList(&B);
      Accesses->push_back(DUD);
      InsertIntoDef = true;
      if (!Defs)
        Defs = getOrCreateDefsList(&B);
      Defs->push_back(*DUD);
    }
    if (InsertIntoDef)
      DefiningBlocks.insert(&B);
  }
  placePHINodes(DefiningBlocks, BBNumbers);

  // Now do regular SSA renaming on the DetachDef/DetachUse. Visited will get
  // filled in with all blocks.
  SmallPtrSet<BasicBlock *, 16> Visited;
  renamePass(DT->getRootNode(), LiveOnEntryDef.get(), Visited);

  // CachingWalker *Walker = getWalkerImpl();

  // // We're doing a batch of updates; don't drop useful caches between them.
  // Walker->setAutoResetWalker(false);
  // OptimizeUses(this, Walker, AA, DT).optimizeUses();
  // Walker->setAutoResetWalker(true);
  // Walker->resetClobberWalker();

  // Mark the uses in unreachable blocks as live on entry, so that they go
  // somewhere.
  for (auto &BB : F)
    if (!Visited.count(&BB))
      markUnreachableAsLiveOnEntry(&BB);
}

// This is a helper function used by the creation routines. It places NewAccess
// into the access and defs lists for a given basic block, at the given
// insertion point.
void DetachSSA::insertIntoListsForBlock(DetachAccess *NewAccess,
                                        const BasicBlock *BB,
                                        InsertionPlace Point) {
  auto *Accesses = getOrCreateAccessList(BB);
  if (Point == Beginning) {
    // If it's a phi node, it goes first, otherwise, it goes after any phi
    // nodes.
    if (isa<DetachPhi>(NewAccess)) {
      Accesses->push_front(NewAccess);
      auto *Defs = getOrCreateDefsList(BB);
      Defs->push_front(*NewAccess);
    } else {
      auto AI = find_if_not(
          *Accesses, [](const DetachAccess &DA) { return isa<DetachPhi>(DA); });
      Accesses->insert(AI, NewAccess);
      if (!isa<DetachUse>(NewAccess)) {
        auto *Defs = getOrCreateDefsList(BB);
        auto DI = find_if_not(
            *Defs, [](const DetachAccess &DA) { return isa<DetachPhi>(DA); });
        Defs->insert(DI, *NewAccess);
      }
    }
  } else {
    Accesses->push_back(NewAccess);
    if (!isa<DetachUse>(NewAccess)) {
      auto *Defs = getOrCreateDefsList(BB);
      Defs->push_back(*NewAccess);
    }
  }
  BlockNumberingValid.erase(BB);
}

void DetachSSA::insertIntoListsBefore(DetachAccess *What, const BasicBlock *BB,
                                      AccessList::iterator InsertPt) {
  auto *Accesses = getWritableBlockAccesses(BB);
  bool WasEnd = InsertPt == Accesses->end();
  Accesses->insert(AccessList::iterator(InsertPt), What);
  if (!isa<DetachUse>(What)) {
    auto *Defs = getOrCreateDefsList(BB);
    // If we got asked to insert at the end, we have an easy job, just shove it
    // at the end. If we got asked to insert before an existing def, we also get
    // an terator. If we got asked to insert before a use, we have to hunt for
    // the next def.
    if (WasEnd) {
      Defs->push_back(*What);
    } else if (isa<DetachDef>(InsertPt)) {
      Defs->insert(InsertPt->getDefsIterator(), *What);
    } else {
      while (InsertPt != Accesses->end() && !isa<DetachDef>(InsertPt))
        ++InsertPt;
      // Either we found a def, or we are inserting at the end
      if (InsertPt == Accesses->end())
        Defs->push_back(*What);
      else
        Defs->insert(InsertPt->getDefsIterator(), *What);
    }
  }
  BlockNumberingValid.erase(BB);
}

// Move What before Where in the IR.  The end result is that What will belong to
// the right lists and have the right Block set, but will not otherwise be
// correct. It will not have the right defining access, and if it is a def,
// things below it will not properly be updated.
void DetachSSA::moveTo(DetachUseOrDef *What, BasicBlock *BB,
                       AccessList::iterator Where) {
  // Keep it in the lookup tables, remove from the lists
  removeFromLists(What, false);
  What->setBlock(BB);
  insertIntoListsBefore(What, BB, Where);
}

void DetachSSA::moveTo(DetachUseOrDef *What, BasicBlock *BB,
                       InsertionPlace Point) {
  removeFromLists(What, false);
  What->setBlock(BB);
  insertIntoListsForBlock(What, BB, Point);
}

DetachPhi *DetachSSA::createDetachPhi(BasicBlock *BB) {
  assert(!getDetachAccess(BB) && "DetachPhi already exists for this BB");
  DetachPhi *Phi = new DetachPhi(BB->getContext(), BB, NextID++);
  // Phi's always are placed at the front of the block.
  insertIntoListsForBlock(Phi, BB, Beginning);
  ValueToDetachAccess[BB] = Phi;
  return Phi;
}

// DetachUseOrDef *DetachSSA::createDefinedAccess(Instruction *I,
//                                                DetachAccess *Definition) {
//   assert(!isa<PHINode>(I) && "Cannot create a defined access for a PHI");
//   DetachUseOrDef *NewAccess = createNewAccess(I);
//   assert(
//       NewAccess != nullptr &&
//       "Tried to create a detach access for a non-detach touching instruction");
//   NewAccess->setDefiningAccess(Definition);
//   return NewAccess;
// }

// /// \brief Helper function to create new detach accesses
// DetachUseOrDef *DetachSSA::createNewAccess(Instruction *I) {
//   bool Def = isa<DetachInst>(I);
//   bool Use = isa<SyncInst>(I);

//   if (!Def && !Use)
//     return nullptr;

//   DetachUseOrDef *DUD;
//   if (Def)
//     DUD = new DetachDef(I->getContext, nullptr, I,
//                         cast<DetachInst>(I)->getContinue(), NextID++);
//   else if (Use)
//     DUD = new DetachUse(I->getContext, nullptr, I, I->getParent());
//   ValueToDetachAccess[I] = DUD;
//   return DUD;
// }

/// \brief Returns true if \p Replacer dominates \p Replacee .
bool DetachSSA::dominatesUse(const DetachAccess *Replacer,
                             const DetachAccess *Replacee) const {
  if (isa<DetachUseOrDef>(Replacee))
    return DT->dominates(Replacer->getBlock(), Replacee->getBlock());
  const auto *DP = cast<DetachPhi>(Replacee);
  // For a phi node, the use occurs in the predecessor block of the phi node.
  // Since we may occur multiple times in the phi node, we have to check each
  // operand to ensure Replacer dominates each operand where Replacee occurs.
  for (const Use &Arg : DP->operands()) {
    if (Arg.get() != Replacee &&
        !DT->dominates(Replacer->getBlock(), DP->getIncomingBlock(Arg)))
      return false;
  }
  return true;
}

/// \brief Properly remove \p DA from all of DetachSSA's lookup tables.
void DetachSSA::removeFromLookups(DetachAccess *DA) {
  assert(DA->use_empty() &&
         "Trying to remove detach access that still has uses");
  BlockNumbering.erase(DA);
  if (DetachUseOrDef *MUD = dyn_cast<DetachUseOrDef>(DA))
    MUD->setDefiningAccess(nullptr);
  // // Invalidate our walker's cache if necessary
  // if (!isa<DetachUse>(DA))
  //   Walker->invalidateInfo(DA);
  // The call below to erase will destroy DA, so we can't change the order we
  // are doing things here
  Value *DAInst;
  if (DetachUseOrDef *DUD = dyn_cast<DetachUseOrDef>(DA)) {
    DAInst = DUD->getDAInst();
  } else {
    DAInst = DA->getBlock();
  }
  auto VDA = ValueToDetachAccess.find(DAInst);
  if (VDA->second == DA)
    ValueToDetachAccess.erase(VDA);
}

/// \brief Properly remove \p DA from all of DetachSSA's lists.
///
/// Because of the way the intrusive list and use lists work, it is important to
/// do removal in the right order.
/// ShouldDelete defaults to true, and will cause the detach access to also be
/// deleted, not just removed.
void DetachSSA::removeFromLists(DetachAccess *DA, bool ShouldDelete) {
  // The access list owns the reference, so we erase it from the non-owning list
  // first.
  if (!isa<DetachUse>(DA)) {
    auto DefsIt = PerBlockDefs.find(DA->getBlock());
    std::unique_ptr<DefsList> &Defs = DefsIt->second;
    Defs->remove(*DA);
    if (Defs->empty())
      PerBlockDefs.erase(DefsIt);
  }

  // The erase call here will delete it. If we don't want it deleted, we call
  // remove instead.
  auto AccessIt = PerBlockAccesses.find(DA->getBlock());
  std::unique_ptr<AccessList> &Accesses = AccessIt->second;
  if (ShouldDelete)
    Accesses->erase(DA);
  else
    Accesses->remove(DA);

  if (Accesses->empty())
    PerBlockAccesses.erase(AccessIt);
}

void DetachSSA::print(raw_ostream &OS) const {
  DetachSSAAnnotatedWriter Writer(this);
  F.print(OS, &Writer);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void DetachSSA::dump() const { print(dbgs()); }
#endif

void DetachSSA::verifyDetachSSA() const {
  verifyDefUses(F);
  verifyDomination(F);
  verifyOrdering(F);
  // Walker->verify(this);
}

/// \brief Verify that the order and existence of DetachAccesses matches the
/// order and existence of detach affecting instructions.
void DetachSSA::verifyOrdering(Function &F) const {
  // Walk all the blocks, comparing what the lookups think and what the access
  // lists think, as well as the order in the blocks vs the order in the access
  // lists.
  SmallVector<DetachAccess *, 32> ActualAccesses;
  SmallVector<DetachAccess *, 32> ActualDefs;
  for (BasicBlock &B : F) {
    const AccessList *AL = getBlockAccesses(&B);
    const auto *DL = getBlockDefs(&B);
    DetachAccess *Phi = getDetachAccess(&B);
    if (Phi) {
      ActualAccesses.push_back(Phi);
      ActualDefs.push_back(Phi);
    }

    for (Instruction &I : B) {
      DetachAccess *DA = getDetachAccess(&I);
      assert((!DA || (AL && (isa<DetachUse>(DA) || DL))) &&
             "We have detach affecting instructions "
             "in this block but they are not in the "
             "access list or defs list");
      if (DA) {
        ActualAccesses.push_back(DA);
        if (isa<DetachDef>(DA))
          ActualDefs.push_back(DA);
      }
    }
    // Either we hit the assert, really have no accesses, or we have both
    // accesses and an access list.
    // Same with defs.
    if (!AL && !DL)
      continue;
    assert(AL->size() == ActualAccesses.size() &&
           "We don't have the same number of accesses in the block as on the "
           "access list");
    assert((DL || ActualDefs.size() == 0) &&
           "Either we should have a defs list, or we should have no defs");
    assert((!DL || DL->size() == ActualDefs.size()) &&
           "We don't have the same number of defs in the block as on the "
           "def list");
    auto ALI = AL->begin();
    auto AAI = ActualAccesses.begin();
    while (ALI != AL->end() && AAI != ActualAccesses.end()) {
      assert(&*ALI == *AAI && "Not the same accesses in the same order");
      ++ALI;
      ++AAI;
    }
    ActualAccesses.clear();
    if (DL) {
      auto DLI = DL->begin();
      auto ADI = ActualDefs.begin();
      while (DLI != DL->end() && ADI != ActualDefs.end()) {
        assert(&*DLI == *ADI && "Not the same defs in the same order");
        ++DLI;
        ++ADI;
      }
    }
    ActualDefs.clear();
  }
}

/// \brief Verify the domination properties of DetachSSA by checking that each
/// definition dominates all of its uses.
void DetachSSA::verifyDomination(Function &F) const {
#ifndef NDEBUG
  for (BasicBlock &B : F) {
    // Phi nodes are attached to basic blocks
    if (DetachPhi *DP = getDetachAccess(&B))
      for (const Use &U : DP->uses())
        assert(dominates(DP, U) && "Detach PHI does not dominate it's uses");

    for (Instruction &I : B) {
      DetachAccess *MD = dyn_cast_or_null<DetachDef>(getDetachAccess(&I));
      if (!MD)
        continue;

      for (const Use &U : MD->uses())
        assert(dominates(MD, U) && "Detach Def does not dominate it's uses");
    }
  }
#endif
}

/// \brief Verify the def-use lists in DetachSSA, by verifying that \p Use
/// appears in the use list of \p Def.

void DetachSSA::verifyUseInDefs(DetachAccess *Def, DetachAccess *Use) const {
#ifndef NDEBUG
  if (!Def)
    assert(isLiveOnEntryDef(Use) &&
           "Null def but use not point to live on entry def");
  else
    assert(is_contained(Def->users(), Use) &&
           "Did not find use in def's use list");
#endif
}

/// \brief Verify the immediate use information, by walking all the detach
/// accesses and verifying that, for each use, it appears in the
/// appropriate def's use list
void DetachSSA::verifyDefUses(Function &F) const {
  for (BasicBlock &B : F) {
    // Phi nodes are attached to basic blocks
    if (DetachPhi *Phi = getDetachAccess(&B)) {
      assert(Phi->getNumOperands() == static_cast<unsigned>(std::distance(
                                          pred_begin(&B), pred_end(&B))) &&
             "Incomplete DetachPhi Node");
      for (unsigned I = 0, E = Phi->getNumIncomingValues(); I != E; ++I)
        verifyUseInDefs(Phi->getIncomingValue(I), Phi);
    }

    for (Instruction &I : B) {
      if (DetachUseOrDef *DA = getDetachAccess(&I)) {
        verifyUseInDefs(DA->getDefiningAccess(), DA);
      }
    }
  }
}

DetachUseOrDef *DetachSSA::getDetachAccess(const Instruction *I) const {
  return cast_or_null<DetachUseOrDef>(ValueToDetachAccess.lookup(I));
}

DetachPhi *DetachSSA::getDetachAccess(const BasicBlock *BB) const {
  return cast_or_null<DetachPhi>(ValueToDetachAccess.lookup(cast<Value>(BB)));
}

/// Perform a local numbering on blocks so that instruction ordering can be
/// determined in constant time.
/// TODO: We currently just number in order.  If we numbered by N, we could
/// allow at least N-1 sequences of insertBefore or insertAfter (and at least
/// log2(N) sequences of mixed before and after) without needing to invalidate
/// the numbering.
void DetachSSA::renumberBlock(const BasicBlock *B) const {
  // The pre-increment ensures the numbers really start at 1.
  unsigned long CurrentNumber = 0;
  const AccessList *AL = getBlockAccesses(B);
  assert(AL != nullptr && "Asking to renumber an empty block");
  for (const auto &I : *AL)
    BlockNumbering[&I] = ++CurrentNumber;
  BlockNumberingValid.insert(B);
}

/// \brief Determine, for two detach accesses in the same block,
/// whether \p Dominator dominates \p Dominatee.
/// \returns True if \p Dominator dominates \p Dominatee.
bool DetachSSA::locallyDominates(const DetachAccess *Dominator,
                                 const DetachAccess *Dominatee) const {

  const BasicBlock *DominatorBlock = Dominator->getBlock();

  assert((DominatorBlock == Dominatee->getBlock()) &&
         "Asking for local domination when accesses are in different blocks!");
  // A node dominates itself.
  if (Dominatee == Dominator)
    return true;

  // When Dominatee is defined on function entry, it is not dominated by another
  // detach access.
  if (isLiveOnEntryDef(Dominatee))
    return false;

  // When Dominator is defined on function entry, it dominates the other detach
  // access.
  if (isLiveOnEntryDef(Dominator))
    return true;

  if (!BlockNumberingValid.count(DominatorBlock))
    renumberBlock(DominatorBlock);

  unsigned long DominatorNum = BlockNumbering.lookup(Dominator);
  // All numbers start with 1
  assert(DominatorNum != 0 && "Block was not numbered properly");
  unsigned long DominateeNum = BlockNumbering.lookup(Dominatee);
  assert(DominateeNum != 0 && "Block was not numbered properly");
  return DominatorNum < DominateeNum;
}

bool DetachSSA::dominates(const DetachAccess *Dominator,
                          const DetachAccess *Dominatee) const {
  if (Dominator == Dominatee)
    return true;

  if (isLiveOnEntryDef(Dominatee))
    return false;

  if (Dominator->getBlock() != Dominatee->getBlock())
    return DT->dominates(Dominator->getBlock(), Dominatee->getBlock());
  return locallyDominates(Dominator, Dominatee);
}

bool DetachSSA::dominates(const DetachAccess *Dominator,
                          const Use &Dominatee) const {
  if (DetachPhi *DP = dyn_cast<DetachPhi>(Dominatee.getUser())) {
    BasicBlock *UseBB = DP->getIncomingBlock(Dominatee);
    // The def must dominate the incoming block of the phi.
    if (UseBB != Dominator->getBlock())
      return DT->dominates(Dominator->getBlock(), UseBB);
    // If the UseBB and the DefBB are the same, compare locally.
    return locallyDominates(Dominator, cast<DetachAccess>(Dominatee));
  }
  // If it's not a PHI node use, the normal dominates can already handle it.
  return dominates(Dominator, cast<DetachAccess>(Dominatee.getUser()));
}

void DetachAccess::print(raw_ostream &OS) const {
  switch (getValueID()) {
  case DetachPhiVal: return static_cast<const DetachPhi *>(this)->print(OS);
  case DetachDefVal: return static_cast<const DetachDef *>(this)->print(OS);
  case DetachUseVal: return static_cast<const DetachUse *>(this)->print(OS);
  }
  llvm_unreachable("invalid value id");
}

void DetachDef::print(raw_ostream &OS) const {
  DetachAccess *UO = getDefiningAccess();

  OS << getID() << " = DetachDef(";
  if (UO && UO->getID())
    OS << UO->getID();
  OS << ')';
}

void DetachPhi::print(raw_ostream &OS) const {
  bool First = true;
  OS << getID() << " = DetachPhi(";
  for (const auto &Op : operands()) {
    BasicBlock *BB = getIncomingBlock(Op);
    DetachAccess *DA = cast<DetachAccess>(Op);
    if (!First)
      OS << ',';
    else
      First = false;

    OS << '{';
    if (BB->hasName())
      OS << BB->getName();
    else
      BB->printAsOperand(OS, false);
    OS << ',';
    if (unsigned ID = DA->getID())
      OS << ID;
    OS << '}';
  }
  OS << ')';
}

void DetachUse::print(raw_ostream &OS) const {
  DetachAccess *UO = getDefiningAccess();
  OS << "DetachUse(";
  if (UO && UO->getID())
    OS << UO->getID();
  OS << ')';
}

void DetachAccess::dump() const {
// Cannot completely remove virtual function even in release mode.
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  print(dbgs());
  dbgs() << "\n";
#endif
}

char DetachSSAPrinterLegacyPass::ID = 0;

DetachSSAPrinterLegacyPass::DetachSSAPrinterLegacyPass() : FunctionPass(ID) {
  initializeDetachSSAPrinterLegacyPassPass(*PassRegistry::getPassRegistry());
}

void DetachSSAPrinterLegacyPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<DetachSSAWrapperPass>();
  AU.addPreserved<DetachSSAWrapperPass>();
}

bool DetachSSAPrinterLegacyPass::runOnFunction(Function &F) {
  auto &DSSA = getAnalysis<DetachSSAWrapperPass>().getDSSA();
  DSSA.print(dbgs());
  if (VerifyDetachSSA)
    DSSA.verifyDetachSSA();
  return false;
}

AnalysisKey DetachSSAAnalysis::Key;

DetachSSAAnalysis::Result DetachSSAAnalysis::run(Function &F,
                                                 FunctionAnalysisManager &AM) {
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  return DetachSSAAnalysis::Result(make_unique<DetachSSA>(F, &DT));
}

PreservedAnalyses DetachSSAPrinterPass::run(Function &F,
                                            FunctionAnalysisManager &AM) {
  OS << "DetachSSA for function: " << F.getName() << "\n";
  AM.getResult<DetachSSAAnalysis>(F).getDSSA().print(OS);

  return PreservedAnalyses::all();
}

PreservedAnalyses DetachSSAVerifierPass::run(Function &F,
                                             FunctionAnalysisManager &AM) {
  AM.getResult<DetachSSAAnalysis>(F).getDSSA().verifyDetachSSA();

  return PreservedAnalyses::all();
}

char DetachSSAWrapperPass::ID = 0;

DetachSSAWrapperPass::DetachSSAWrapperPass() : FunctionPass(ID) {
  initializeDetachSSAWrapperPassPass(*PassRegistry::getPassRegistry());
}

void DetachSSAWrapperPass::releaseMemory() { DSSA.reset(); }

void DetachSSAWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequiredTransitive<DominatorTreeWrapperPass>();
}

bool DetachSSAWrapperPass::runOnFunction(Function &F) {
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  DSSA.reset(new DetachSSA(F, &DT));
  return false;
}

void DetachSSAWrapperPass::verifyAnalysis() const { DSSA->verifyDetachSSA(); }

void DetachSSAWrapperPass::print(raw_ostream &OS, const Module *M) const {
  DSSA->print(OS);
}
} // namespace llvm

void DetachPhi::deleteMe(DerivedUser *Self) {
  delete static_cast<DetachPhi *>(Self);
}

void DetachDef::deleteMe(DerivedUser *Self) {
  delete static_cast<DetachDef *>(Self);
}

void DetachUse::deleteMe(DerivedUser *Self) {
  delete static_cast<DetachUse *>(Self);
}
