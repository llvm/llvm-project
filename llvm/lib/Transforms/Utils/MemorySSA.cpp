//===-- MemorySSA.cpp - Memory SSA Builder---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------===//
//
// This file implements the MemorySSA class.
//
//===----------------------------------------------------------------===//
#include "llvm/Transforms/Utils/MemorySSA.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/IteratedDominanceFrontier.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/PHITransAddr.h"
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Transforms/Scalar.h"
#include <algorithm>

#define DEBUG_TYPE "memoryssa"
using namespace llvm;
STATISTIC(NumClobberCacheLookups, "Number of Memory SSA version cache lookups");
STATISTIC(NumClobberCacheHits, "Number of Memory SSA version cache hits");
STATISTIC(NumClobberCacheInserts, "Number of MemorySSA version cache inserts");

INITIALIZE_PASS_BEGIN(MemorySSAWrapperPass, "memoryssa", "Memory SSA", false,
                      true)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_END(MemorySSAWrapperPass, "memoryssa", "Memory SSA", false,
                    true)

namespace llvm {
/// \brief An assembly annotator class to print Memory SSA information in
/// comments.
class MemorySSAAnnotatedWriter : public AssemblyAnnotationWriter {
  friend class MemorySSA;
  const MemorySSA *MSSA;

public:
  MemorySSAAnnotatedWriter(const MemorySSA *M) : MSSA(M) {}

  virtual void emitBasicBlockStartAnnot(const BasicBlock *BB,
                                        formatted_raw_ostream &OS) {
    if (MemoryAccess *MA = MSSA->getMemoryAccess(BB))
      OS << "; " << *MA << "\n";
  }

  virtual void emitInstructionAnnot(const Instruction *I,
                                    formatted_raw_ostream &OS) {
    if (MemoryAccess *MA = MSSA->getMemoryAccess(I))
      OS << "; " << *MA << "\n";
  }
};

/// \brief A MemorySSAWalker that does AA walks and caching of lookups to
/// disambiguate accesses.
///
/// FIXME: The current implementation of this can take quadratic space in rare
/// cases. This can be fixed, but it is something to note until it is fixed.
///
/// In order to trigger this behavior, you need to store to N distinct locations
/// (that AA can prove don't alias), perform M stores to other memory
/// locations that AA can prove don't alias any of the initial N locations, and
/// then load from all of the N locations. In this case, we insert M cache
/// entries for each of the N loads.
///
/// For example:
/// define i32 @foo() {
///   %a = alloca i32, align 4
///   %b = alloca i32, align 4
///   store i32 0, i32* %a, align 4
///   store i32 0, i32* %b, align 4
///
///   ; Insert M stores to other memory that doesn't alias %a or %b here
///
///   %c = load i32, i32* %a, align 4 ; Caches M entries in
///                                   ; CachedUpwardsClobberingAccess for the
///                                   ; MemoryLocation %a
///   %d = load i32, i32* %b, align 4 ; Caches M entries in
///                                   ; CachedUpwardsClobberingAccess for the
///                                   ; MemoryLocation %b
///
///   ; For completeness' sake, loading %a or %b again would not cache *another*
///   ; M entries.
///   %r = add i32 %c, %d
///   ret i32 %r
/// }
class MemorySSA::CachingWalker final : public MemorySSAWalker {
public:
  CachingWalker(MemorySSA *, AliasAnalysis *, DominatorTree *);
  ~CachingWalker() override;

  MemoryAccess *getClobberingMemoryAccess(const Instruction *) override;
  MemoryAccess *getClobberingMemoryAccess(MemoryAccess *,
                                          MemoryLocation &) override;
  void invalidateInfo(MemoryAccess *) override;

protected:
  struct UpwardsMemoryQuery;
  MemoryAccess *doCacheLookup(const MemoryAccess *, const UpwardsMemoryQuery &,
                              const MemoryLocation &);

  void doCacheInsert(const MemoryAccess *, MemoryAccess *,
                     const UpwardsMemoryQuery &, const MemoryLocation &);

  void doCacheRemove(const MemoryAccess *, const UpwardsMemoryQuery &,
                     const MemoryLocation &);

private:
  MemoryAccessPair UpwardsDFSWalk(MemoryAccess *, const MemoryLocation &,
                                  UpwardsMemoryQuery &, bool);
  MemoryAccess *getClobberingMemoryAccess(MemoryAccess *, UpwardsMemoryQuery &);
  bool instructionClobbersQuery(const MemoryDef *, UpwardsMemoryQuery &,
                                const MemoryLocation &Loc) const;
  void verifyRemoved(MemoryAccess *);
  SmallDenseMap<ConstMemoryAccessPair, MemoryAccess *>
      CachedUpwardsClobberingAccess;
  DenseMap<const MemoryAccess *, MemoryAccess *> CachedUpwardsClobberingCall;
  AliasAnalysis *AA;
  DominatorTree *DT;
};
}

namespace {
struct RenamePassData {
  DomTreeNode *DTN;
  DomTreeNode::const_iterator ChildIt;
  MemoryAccess *IncomingVal;

  RenamePassData(DomTreeNode *D, DomTreeNode::const_iterator It,
                 MemoryAccess *M)
      : DTN(D), ChildIt(It), IncomingVal(M) {}
  void swap(RenamePassData &RHS) {
    std::swap(DTN, RHS.DTN);
    std::swap(ChildIt, RHS.ChildIt);
    std::swap(IncomingVal, RHS.IncomingVal);
  }
};
}

namespace llvm {
/// \brief Rename a single basic block into MemorySSA form.
/// Uses the standard SSA renaming algorithm.
/// \returns The new incoming value.
MemoryAccess *MemorySSA::renameBlock(BasicBlock *BB,
                                     MemoryAccess *IncomingVal) {
  auto It = PerBlockAccesses.find(BB);
  // Skip most processing if the list is empty.
  if (It != PerBlockAccesses.end()) {
    AccessList *Accesses = It->second.get();
    for (MemoryAccess &L : *Accesses) {
      switch (L.getValueID()) {
      case Value::MemoryUseVal:
        cast<MemoryUse>(&L)->setDefiningAccess(IncomingVal);
        break;
      case Value::MemoryDefVal:
        // We can't legally optimize defs, because we only allow single
        // memory phis/uses on operations, and if we optimize these, we can
        // end up with multiple reaching defs. Uses do not have this
        // problem, since they do not produce a value
        cast<MemoryDef>(&L)->setDefiningAccess(IncomingVal);
        IncomingVal = &L;
        break;
      case Value::MemoryPhiVal:
        IncomingVal = &L;
        break;
      }
    }
  }

  // Pass through values to our successors
  for (const BasicBlock *S : successors(BB)) {
    auto It = PerBlockAccesses.find(S);
    // Rename the phi nodes in our successor block
    if (It == PerBlockAccesses.end() || !isa<MemoryPhi>(It->second->front()))
      continue;
    AccessList *Accesses = It->second.get();
    auto *Phi = cast<MemoryPhi>(&Accesses->front());
    assert(std::find(succ_begin(BB), succ_end(BB), S) != succ_end(BB) &&
           "Must be at least one edge from Succ to BB!");
    Phi->addIncoming(IncomingVal, BB);
  }

  return IncomingVal;
}

/// \brief This is the standard SSA renaming algorithm.
///
/// We walk the dominator tree in preorder, renaming accesses, and then filling
/// in phi nodes in our successors.
void MemorySSA::renamePass(DomTreeNode *Root, MemoryAccess *IncomingVal,
                           SmallPtrSet<BasicBlock *, 16> &Visited) {
  SmallVector<RenamePassData, 32> WorkStack;
  IncomingVal = renameBlock(Root->getBlock(), IncomingVal);
  WorkStack.push_back({Root, Root->begin(), IncomingVal});
  Visited.insert(Root->getBlock());

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
      Visited.insert(BB);
      IncomingVal = renameBlock(BB, IncomingVal);
      WorkStack.push_back({Child, Child->begin(), IncomingVal});
    }
  }
}

/// \brief Compute dominator levels, used by the phi insertion algorithm above.
void MemorySSA::computeDomLevels(DenseMap<DomTreeNode *, unsigned> &DomLevels) {
  for (auto DFI = df_begin(DT->getRootNode()), DFE = df_end(DT->getRootNode());
       DFI != DFE; ++DFI)
    DomLevels[*DFI] = DFI.getPathLength() - 1;
}

/// \brief This handles unreachable block acccesses by deleting phi nodes in
/// unreachable blocks, and marking all other unreachable MemoryAccess's as
/// being uses of the live on entry definition.
void MemorySSA::markUnreachableAsLiveOnEntry(BasicBlock *BB) {
  assert(!DT->isReachableFromEntry(BB) &&
         "Reachable block found while handling unreachable blocks");

  auto It = PerBlockAccesses.find(BB);
  if (It == PerBlockAccesses.end())
    return;

  auto &Accesses = It->second;
  for (auto AI = Accesses->begin(), AE = Accesses->end(); AI != AE;) {
    auto Next = std::next(AI);
    // If we have a phi, just remove it. We are going to replace all
    // users with live on entry.
    if (auto *UseOrDef = dyn_cast<MemoryUseOrDef>(AI))
      UseOrDef->setDefiningAccess(LiveOnEntryDef.get());
    else
      Accesses->erase(AI);
    AI = Next;
  }
}

MemorySSA::MemorySSA(Function &Func, AliasAnalysis *AA, DominatorTree *DT)
    : AA(AA), DT(DT), F(Func), LiveOnEntryDef(nullptr), Walker(nullptr),
      NextID(0) {
  buildMemorySSA();
}

MemorySSA::MemorySSA(MemorySSA &&MSSA)
    : AA(MSSA.AA), DT(MSSA.DT), F(MSSA.F),
      ValueToMemoryAccess(std::move(MSSA.ValueToMemoryAccess)),
      PerBlockAccesses(std::move(MSSA.PerBlockAccesses)),
      LiveOnEntryDef(std::move(MSSA.LiveOnEntryDef)),
      Walker(std::move(MSSA.Walker)), NextID(MSSA.NextID) {
  // Update the Walker MSSA pointer so it doesn't point to the moved-from MSSA
  // object any more.
  Walker->MSSA = this;
}

MemorySSA::~MemorySSA() {
  // Drop all our references
  for (const auto &Pair : PerBlockAccesses)
    for (MemoryAccess &MA : *Pair.second)
      MA.dropAllReferences();
}

MemorySSA::AccessList *MemorySSA::getOrCreateAccessList(const BasicBlock *BB) {
  auto Res = PerBlockAccesses.insert(std::make_pair(BB, nullptr));

  if (Res.second)
    Res.first->second = make_unique<AccessList>();
  return Res.first->second.get();
}

void MemorySSA::buildMemorySSA() {
  // We create an access to represent "live on entry", for things like
  // arguments or users of globals, where the memory they use is defined before
  // the beginning of the function. We do not actually insert it into the IR.
  // We do not define a live on exit for the immediate uses, and thus our
  // semantics do *not* imply that something with no immediate uses can simply
  // be removed.
  BasicBlock &StartingPoint = F.getEntryBlock();
  LiveOnEntryDef = make_unique<MemoryDef>(F.getContext(), nullptr, nullptr,
                                          &StartingPoint, NextID++);

  // We maintain lists of memory accesses per-block, trading memory for time. We
  // could just look up the memory access for every possible instruction in the
  // stream.
  SmallPtrSet<BasicBlock *, 32> DefiningBlocks;
  SmallPtrSet<BasicBlock *, 32> DefUseBlocks;
  // Go through each block, figure out where defs occur, and chain together all
  // the accesses.
  for (BasicBlock &B : F) {
    bool InsertIntoDef = false;
    AccessList *Accesses = nullptr;
    for (Instruction &I : B) {
      MemoryUseOrDef *MUD = createNewAccess(&I);
      if (!MUD)
        continue;
      InsertIntoDef |= isa<MemoryDef>(MUD);

      if (!Accesses)
        Accesses = getOrCreateAccessList(&B);
      Accesses->push_back(MUD);
    }
    if (InsertIntoDef)
      DefiningBlocks.insert(&B);
    if (Accesses)
      DefUseBlocks.insert(&B);
  }

  // Compute live-in.
  // Live in is normally defined as "all the blocks on the path from each def to
  // each of it's uses".
  // MemoryDef's are implicit uses of previous state, so they are also uses.
  // This means we don't really have def-only instructions.  The only
  // MemoryDef's that are not really uses are those that are of the LiveOnEntry
  // variable (because LiveOnEntry can reach anywhere, and every def is a
  // must-kill of LiveOnEntry).
  // In theory, you could precisely compute live-in by using alias-analysis to
  // disambiguate defs and uses to see which really pair up with which.
  // In practice, this would be really expensive and difficult. So we simply
  // assume all defs are also uses that need to be kept live.
  // Because of this, the end result of this live-in computation will be "the
  // entire set of basic blocks that reach any use".

  SmallPtrSet<BasicBlock *, 32> LiveInBlocks;
  SmallVector<BasicBlock *, 64> LiveInBlockWorklist(DefUseBlocks.begin(),
                                                    DefUseBlocks.end());
  // Now that we have a set of blocks where a value is live-in, recursively add
  // predecessors until we find the full region the value is live.
  while (!LiveInBlockWorklist.empty()) {
    BasicBlock *BB = LiveInBlockWorklist.pop_back_val();

    // The block really is live in here, insert it into the set.  If already in
    // the set, then it has already been processed.
    if (!LiveInBlocks.insert(BB).second)
      continue;

    // Since the value is live into BB, it is either defined in a predecessor or
    // live into it to.
    LiveInBlockWorklist.append(pred_begin(BB), pred_end(BB));
  }

  // Determine where our MemoryPhi's should go
  ForwardIDFCalculator IDFs(*DT);
  IDFs.setDefiningBlocks(DefiningBlocks);
  IDFs.setLiveInBlocks(LiveInBlocks);
  SmallVector<BasicBlock *, 32> IDFBlocks;
  IDFs.calculate(IDFBlocks);

  // Now place MemoryPhi nodes.
  for (auto &BB : IDFBlocks) {
    // Insert phi node
    AccessList *Accesses = getOrCreateAccessList(BB);
    MemoryPhi *Phi = new MemoryPhi(BB->getContext(), BB, NextID++);
    ValueToMemoryAccess.insert(std::make_pair(BB, Phi));
    // Phi's always are placed at the front of the block.
    Accesses->push_front(Phi);
  }

  // Now do regular SSA renaming on the MemoryDef/MemoryUse. Visited will get
  // filled in with all blocks.
  SmallPtrSet<BasicBlock *, 16> Visited;
  renamePass(DT->getRootNode(), LiveOnEntryDef.get(), Visited);

  MemorySSAWalker *Walker = getWalker();

  // Now optimize the MemoryUse's defining access to point to the nearest
  // dominating clobbering def.
  // This ensures that MemoryUse's that are killed by the same store are
  // immediate users of that store, one of the invariants we guarantee.
  for (auto DomNode : depth_first(DT)) {
    BasicBlock *BB = DomNode->getBlock();
    auto AI = PerBlockAccesses.find(BB);
    if (AI == PerBlockAccesses.end())
      continue;
    AccessList *Accesses = AI->second.get();
    for (auto &MA : *Accesses) {
      if (auto *MU = dyn_cast<MemoryUse>(&MA)) {
        Instruction *Inst = MU->getMemoryInst();
        MU->setDefiningAccess(Walker->getClobberingMemoryAccess(Inst));
      }
    }
  }

  // Mark the uses in unreachable blocks as live on entry, so that they go
  // somewhere.
  for (auto &BB : F)
    if (!Visited.count(&BB))
      markUnreachableAsLiveOnEntry(&BB);
}

MemorySSAWalker *MemorySSA::getWalker() {
  if (Walker)
    return Walker.get();

  Walker = make_unique<CachingWalker>(this, AA, DT);
  return Walker.get();
}

MemoryPhi *MemorySSA::createMemoryPhi(BasicBlock *BB) {
  assert(!getMemoryAccess(BB) && "MemoryPhi already exists for this BB");
  AccessList *Accesses = getOrCreateAccessList(BB);
  MemoryPhi *Phi = new MemoryPhi(BB->getContext(), BB, NextID++);
  ValueToMemoryAccess.insert(std::make_pair(BB, Phi));
  // Phi's always are placed at the front of the block.
  Accesses->push_front(Phi);
  return Phi;
}

MemoryUseOrDef *MemorySSA::createDefinedAccess(Instruction *I,
                                               MemoryAccess *Definition) {
  assert(!isa<PHINode>(I) && "Cannot create a defined access for a PHI");
  MemoryUseOrDef *NewAccess = createNewAccess(I);
  assert(
      NewAccess != nullptr &&
      "Tried to create a memory access for a non-memory touching instruction");
  NewAccess->setDefiningAccess(Definition);
  return NewAccess;
}

MemoryAccess *MemorySSA::createMemoryAccessInBB(Instruction *I,
                                                MemoryAccess *Definition,
                                                const BasicBlock *BB,
                                                InsertionPlace Point) {
  MemoryUseOrDef *NewAccess = createDefinedAccess(I, Definition);
  auto *Accesses = getOrCreateAccessList(BB);
  if (Point == Beginning) {
    // It goes after any phi nodes
    auto AI = std::find_if(
        Accesses->begin(), Accesses->end(),
        [](const MemoryAccess &MA) { return !isa<MemoryPhi>(MA); });

    Accesses->insert(AI, NewAccess);
  } else {
    Accesses->push_back(NewAccess);
  }

  return NewAccess;
}
MemoryAccess *MemorySSA::createMemoryAccessBefore(Instruction *I,
                                                  MemoryAccess *Definition,
                                                  MemoryAccess *InsertPt) {
  assert(I->getParent() == InsertPt->getBlock() &&
         "New and old access must be in the same block");
  MemoryUseOrDef *NewAccess = createDefinedAccess(I, Definition);
  auto *Accesses = getOrCreateAccessList(InsertPt->getBlock());
  Accesses->insert(AccessList::iterator(InsertPt), NewAccess);
  return NewAccess;
}

MemoryAccess *MemorySSA::createMemoryAccessAfter(Instruction *I,
                                                 MemoryAccess *Definition,
                                                 MemoryAccess *InsertPt) {
  assert(I->getParent() == InsertPt->getBlock() &&
         "New and old access must be in the same block");
  MemoryUseOrDef *NewAccess = createDefinedAccess(I, Definition);
  auto *Accesses = getOrCreateAccessList(InsertPt->getBlock());
  Accesses->insertAfter(AccessList::iterator(InsertPt), NewAccess);
  return NewAccess;
}

/// \brief Helper function to create new memory accesses
MemoryUseOrDef *MemorySSA::createNewAccess(Instruction *I) {
  // The assume intrinsic has a control dependency which we model by claiming
  // that it writes arbitrarily. Ignore that fake memory dependency here.
  // FIXME: Replace this special casing with a more accurate modelling of
  // assume's control dependency.
  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I))
    if (II->getIntrinsicID() == Intrinsic::assume)
      return nullptr;

  // Find out what affect this instruction has on memory.
  ModRefInfo ModRef = AA->getModRefInfo(I);
  bool Def = bool(ModRef & MRI_Mod);
  bool Use = bool(ModRef & MRI_Ref);

  // It's possible for an instruction to not modify memory at all. During
  // construction, we ignore them.
  if (!Def && !Use)
    return nullptr;

  assert((Def || Use) &&
         "Trying to create a memory access with a non-memory instruction");

  MemoryUseOrDef *MUD;
  if (Def)
    MUD = new MemoryDef(I->getContext(), nullptr, I, I->getParent(), NextID++);
  else
    MUD = new MemoryUse(I->getContext(), nullptr, I, I->getParent());
  ValueToMemoryAccess.insert(std::make_pair(I, MUD));
  return MUD;
}

MemoryAccess *MemorySSA::findDominatingDef(BasicBlock *UseBlock,
                                           enum InsertionPlace Where) {
  // Handle the initial case
  if (Where == Beginning)
    // The only thing that could define us at the beginning is a phi node
    if (MemoryPhi *Phi = getMemoryAccess(UseBlock))
      return Phi;

  DomTreeNode *CurrNode = DT->getNode(UseBlock);
  // Need to be defined by our dominator
  if (Where == Beginning)
    CurrNode = CurrNode->getIDom();
  Where = End;
  while (CurrNode) {
    auto It = PerBlockAccesses.find(CurrNode->getBlock());
    if (It != PerBlockAccesses.end()) {
      auto &Accesses = It->second;
      for (MemoryAccess &RA : reverse(*Accesses)) {
        if (isa<MemoryDef>(RA) || isa<MemoryPhi>(RA))
          return &RA;
      }
    }
    CurrNode = CurrNode->getIDom();
  }
  return LiveOnEntryDef.get();
}

/// \brief Returns true if \p Replacer dominates \p Replacee .
bool MemorySSA::dominatesUse(const MemoryAccess *Replacer,
                             const MemoryAccess *Replacee) const {
  if (isa<MemoryUseOrDef>(Replacee))
    return DT->dominates(Replacer->getBlock(), Replacee->getBlock());
  const auto *MP = cast<MemoryPhi>(Replacee);
  // For a phi node, the use occurs in the predecessor block of the phi node.
  // Since we may occur multiple times in the phi node, we have to check each
  // operand to ensure Replacer dominates each operand where Replacee occurs.
  for (const Use &Arg : MP->operands()) {
    if (Arg.get() != Replacee &&
        !DT->dominates(Replacer->getBlock(), MP->getIncomingBlock(Arg)))
      return false;
  }
  return true;
}

/// \brief If all arguments of a MemoryPHI are defined by the same incoming
/// argument, return that argument.
static MemoryAccess *onlySingleValue(MemoryPhi *MP) {
  MemoryAccess *MA = nullptr;

  for (auto &Arg : MP->operands()) {
    if (!MA)
      MA = cast<MemoryAccess>(Arg);
    else if (MA != Arg)
      return nullptr;
  }
  return MA;
}

/// \brief Properly remove \p MA from all of MemorySSA's lookup tables.
///
/// Because of the way the intrusive list and use lists work, it is important to
/// do removal in the right order.
void MemorySSA::removeFromLookups(MemoryAccess *MA) {
  assert(MA->use_empty() &&
         "Trying to remove memory access that still has uses");
  if (MemoryUseOrDef *MUD = dyn_cast<MemoryUseOrDef>(MA))
    MUD->setDefiningAccess(nullptr);
  // Invalidate our walker's cache if necessary
  if (!isa<MemoryUse>(MA))
    Walker->invalidateInfo(MA);
  // The call below to erase will destroy MA, so we can't change the order we
  // are doing things here
  Value *MemoryInst;
  if (MemoryUseOrDef *MUD = dyn_cast<MemoryUseOrDef>(MA)) {
    MemoryInst = MUD->getMemoryInst();
  } else {
    MemoryInst = MA->getBlock();
  }
  ValueToMemoryAccess.erase(MemoryInst);

  auto AccessIt = PerBlockAccesses.find(MA->getBlock());
  std::unique_ptr<AccessList> &Accesses = AccessIt->second;
  Accesses->erase(MA);
  if (Accesses->empty())
    PerBlockAccesses.erase(AccessIt);
}

void MemorySSA::removeMemoryAccess(MemoryAccess *MA) {
  assert(!isLiveOnEntryDef(MA) && "Trying to remove the live on entry def");
  // We can only delete phi nodes if they have no uses, or we can replace all
  // uses with a single definition.
  MemoryAccess *NewDefTarget = nullptr;
  if (MemoryPhi *MP = dyn_cast<MemoryPhi>(MA)) {
    // Note that it is sufficient to know that all edges of the phi node have
    // the same argument.  If they do, by the definition of dominance frontiers
    // (which we used to place this phi), that argument must dominate this phi,
    // and thus, must dominate the phi's uses, and so we will not hit the assert
    // below.
    NewDefTarget = onlySingleValue(MP);
    assert((NewDefTarget || MP->use_empty()) &&
           "We can't delete this memory phi");
  } else {
    NewDefTarget = cast<MemoryUseOrDef>(MA)->getDefiningAccess();
  }

  // Re-point the uses at our defining access
  if (!MA->use_empty())
    MA->replaceAllUsesWith(NewDefTarget);

  // The call below to erase will destroy MA, so we can't change the order we
  // are doing things here
  removeFromLookups(MA);
}

void MemorySSA::print(raw_ostream &OS) const {
  MemorySSAAnnotatedWriter Writer(this);
  F.print(OS, &Writer);
}

void MemorySSA::dump() const {
  MemorySSAAnnotatedWriter Writer(this);
  F.print(dbgs(), &Writer);
}

void MemorySSA::verifyMemorySSA() const {
  verifyDefUses(F);
  verifyDomination(F);
  verifyOrdering(F);
}

/// \brief Verify that the order and existence of MemoryAccesses matches the
/// order and existence of memory affecting instructions.
void MemorySSA::verifyOrdering(Function &F) const {
  // Walk all the blocks, comparing what the lookups think and what the access
  // lists think, as well as the order in the blocks vs the order in the access
  // lists.
  SmallVector<MemoryAccess *, 32> ActualAccesses;
  for (BasicBlock &B : F) {
    const AccessList *AL = getBlockAccesses(&B);
    MemoryAccess *Phi = getMemoryAccess(&B);
    if (Phi)
      ActualAccesses.push_back(Phi);
    for (Instruction &I : B) {
      MemoryAccess *MA = getMemoryAccess(&I);
      assert((!MA || AL) && "We have memory affecting instructions "
                            "in this block but they are not in the "
                            "access list");
      if (MA)
        ActualAccesses.push_back(MA);
    }
    // Either we hit the assert, really have no accesses, or we have both
    // accesses and an access list
    if (!AL)
      continue;
    assert(AL->size() == ActualAccesses.size() &&
           "We don't have the same number of accesses in the block as on the "
           "access list");
    auto ALI = AL->begin();
    auto AAI = ActualAccesses.begin();
    while (ALI != AL->end() && AAI != ActualAccesses.end()) {
      assert(&*ALI == *AAI && "Not the same accesses in the same order");
      ++ALI;
      ++AAI;
    }
    ActualAccesses.clear();
  }
}

/// \brief Verify the domination properties of MemorySSA by checking that each
/// definition dominates all of its uses.
void MemorySSA::verifyDomination(Function &F) const {
  for (BasicBlock &B : F) {
    // Phi nodes are attached to basic blocks
    if (MemoryPhi *MP = getMemoryAccess(&B)) {
      for (User *U : MP->users()) {
        BasicBlock *UseBlock;
        // Phi operands are used on edges, we simulate the right domination by
        // acting as if the use occurred at the end of the predecessor block.
        if (MemoryPhi *P = dyn_cast<MemoryPhi>(U)) {
          for (const auto &Arg : P->operands()) {
            if (Arg == MP) {
              UseBlock = P->getIncomingBlock(Arg);
              break;
            }
          }
        } else {
          UseBlock = cast<MemoryAccess>(U)->getBlock();
        }
        (void)UseBlock;
        assert(DT->dominates(MP->getBlock(), UseBlock) &&
               "Memory PHI does not dominate it's uses");
      }
    }

    for (Instruction &I : B) {
      MemoryAccess *MD = dyn_cast_or_null<MemoryDef>(getMemoryAccess(&I));
      if (!MD)
        continue;

      for (User *U : MD->users()) {
        BasicBlock *UseBlock;
        (void)UseBlock;
        // Things are allowed to flow to phi nodes over their predecessor edge.
        if (auto *P = dyn_cast<MemoryPhi>(U)) {
          for (const auto &Arg : P->operands()) {
            if (Arg == MD) {
              UseBlock = P->getIncomingBlock(Arg);
              break;
            }
          }
        } else {
          UseBlock = cast<MemoryAccess>(U)->getBlock();
        }
        assert(DT->dominates(MD->getBlock(), UseBlock) &&
               "Memory Def does not dominate it's uses");
      }
    }
  }
}

/// \brief Verify the def-use lists in MemorySSA, by verifying that \p Use
/// appears in the use list of \p Def.
///
/// llvm_unreachable is used instead of asserts because this may be called in
/// a build without asserts. In that case, we don't want this to turn into a
/// nop.
void MemorySSA::verifyUseInDefs(MemoryAccess *Def, MemoryAccess *Use) const {
  // The live on entry use may cause us to get a NULL def here
  if (!Def) {
    if (!isLiveOnEntryDef(Use))
      llvm_unreachable("Null def but use not point to live on entry def");
  } else if (std::find(Def->user_begin(), Def->user_end(), Use) ==
             Def->user_end()) {
    llvm_unreachable("Did not find use in def's use list");
  }
}

/// \brief Verify the immediate use information, by walking all the memory
/// accesses and verifying that, for each use, it appears in the
/// appropriate def's use list
void MemorySSA::verifyDefUses(Function &F) const {
  for (BasicBlock &B : F) {
    // Phi nodes are attached to basic blocks
    if (MemoryPhi *Phi = getMemoryAccess(&B)) {
      assert(Phi->getNumOperands() == static_cast<unsigned>(std::distance(
                                          pred_begin(&B), pred_end(&B))) &&
             "Incomplete MemoryPhi Node");
      for (unsigned I = 0, E = Phi->getNumIncomingValues(); I != E; ++I)
        verifyUseInDefs(Phi->getIncomingValue(I), Phi);
    }

    for (Instruction &I : B) {
      if (MemoryAccess *MA = getMemoryAccess(&I)) {
        assert(isa<MemoryUseOrDef>(MA) &&
               "Found a phi node not attached to a bb");
        verifyUseInDefs(cast<MemoryUseOrDef>(MA)->getDefiningAccess(), MA);
      }
    }
  }
}

MemoryAccess *MemorySSA::getMemoryAccess(const Value *I) const {
  return ValueToMemoryAccess.lookup(I);
}

MemoryPhi *MemorySSA::getMemoryAccess(const BasicBlock *BB) const {
  return cast_or_null<MemoryPhi>(getMemoryAccess((const Value *)BB));
}

/// \brief Determine, for two memory accesses in the same block,
/// whether \p Dominator dominates \p Dominatee.
/// \returns True if \p Dominator dominates \p Dominatee.
bool MemorySSA::locallyDominates(const MemoryAccess *Dominator,
                                 const MemoryAccess *Dominatee) const {

  assert((Dominator->getBlock() == Dominatee->getBlock()) &&
         "Asking for local domination when accesses are in different blocks!");

  // A node dominates itself.
  if (Dominatee == Dominator)
    return true;

  // When Dominatee is defined on function entry, it is not dominated by another
  // memory access.
  if (isLiveOnEntryDef(Dominatee))
    return false;

  // When Dominator is defined on function entry, it dominates the other memory
  // access.
  if (isLiveOnEntryDef(Dominator))
    return true;

  // Get the access list for the block
  const AccessList *AccessList = getBlockAccesses(Dominator->getBlock());
  AccessList::const_reverse_iterator It(Dominator->getIterator());

  // If we hit the beginning of the access list before we hit dominatee, we must
  // dominate it
  return std::none_of(It, AccessList->rend(),
                      [&](const MemoryAccess &MA) { return &MA == Dominatee; });
}

const static char LiveOnEntryStr[] = "liveOnEntry";

void MemoryDef::print(raw_ostream &OS) const {
  MemoryAccess *UO = getDefiningAccess();

  OS << getID() << " = MemoryDef(";
  if (UO && UO->getID())
    OS << UO->getID();
  else
    OS << LiveOnEntryStr;
  OS << ')';
}

void MemoryPhi::print(raw_ostream &OS) const {
  bool First = true;
  OS << getID() << " = MemoryPhi(";
  for (const auto &Op : operands()) {
    BasicBlock *BB = getIncomingBlock(Op);
    MemoryAccess *MA = cast<MemoryAccess>(Op);
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
    if (unsigned ID = MA->getID())
      OS << ID;
    else
      OS << LiveOnEntryStr;
    OS << '}';
  }
  OS << ')';
}

MemoryAccess::~MemoryAccess() {}

void MemoryUse::print(raw_ostream &OS) const {
  MemoryAccess *UO = getDefiningAccess();
  OS << "MemoryUse(";
  if (UO && UO->getID())
    OS << UO->getID();
  else
    OS << LiveOnEntryStr;
  OS << ')';
}

void MemoryAccess::dump() const {
  print(dbgs());
  dbgs() << "\n";
}

char MemorySSAAnalysis::PassID;

MemorySSA MemorySSAAnalysis::run(Function &F, AnalysisManager<Function> &AM) {
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &AA = AM.getResult<AAManager>(F);
  return MemorySSA(F, &AA, &DT);
}

PreservedAnalyses MemorySSAPrinterPass::run(Function &F,
                                            FunctionAnalysisManager &AM) {
  OS << "MemorySSA for function: " << F.getName() << "\n";
  AM.getResult<MemorySSAAnalysis>(F).print(OS);

  return PreservedAnalyses::all();
}

PreservedAnalyses MemorySSAVerifierPass::run(Function &F,
                                             FunctionAnalysisManager &AM) {
  AM.getResult<MemorySSAAnalysis>(F).verifyMemorySSA();

  return PreservedAnalyses::all();
}

char MemorySSAWrapperPass::ID = 0;

MemorySSAWrapperPass::MemorySSAWrapperPass() : FunctionPass(ID) {
  initializeMemorySSAWrapperPassPass(*PassRegistry::getPassRegistry());
}

void MemorySSAWrapperPass::releaseMemory() { MSSA.reset(); }

void MemorySSAWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequiredTransitive<DominatorTreeWrapperPass>();
  AU.addRequiredTransitive<AAResultsWrapperPass>();
}

bool MemorySSAWrapperPass::runOnFunction(Function &F) {
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
  MSSA.reset(new MemorySSA(F, &AA, &DT));
  return false;
}

void MemorySSAWrapperPass::verifyAnalysis() const { MSSA->verifyMemorySSA(); }

void MemorySSAWrapperPass::print(raw_ostream &OS, const Module *M) const {
  MSSA->print(OS);
}

MemorySSAWalker::MemorySSAWalker(MemorySSA *M) : MSSA(M) {}

MemorySSA::CachingWalker::CachingWalker(MemorySSA *M, AliasAnalysis *A,
                                        DominatorTree *D)
    : MemorySSAWalker(M), AA(A), DT(D) {}

MemorySSA::CachingWalker::~CachingWalker() {}

struct MemorySSA::CachingWalker::UpwardsMemoryQuery {
  // True if we saw a phi whose predecessor was a backedge
  bool SawBackedgePhi;
  // True if our original query started off as a call
  bool IsCall;
  // The pointer location we started the query with. This will be empty if
  // IsCall is true.
  MemoryLocation StartingLoc;
  // This is the instruction we were querying about.
  const Instruction *Inst;
  // Set of visited Instructions for this query.
  DenseSet<MemoryAccessPair> Visited;
  // Vector of visited call accesses for this query. This is separated out
  // because you can always cache and lookup the result of call queries (IE when
  // IsCall == true) for every call in the chain. The calls have no AA location
  // associated with them with them, and thus, no context dependence.
  SmallVector<const MemoryAccess *, 32> VisitedCalls;
  // The MemoryAccess we actually got called with, used to test local domination
  const MemoryAccess *OriginalAccess;

  UpwardsMemoryQuery()
      : SawBackedgePhi(false), IsCall(false), Inst(nullptr),
        OriginalAccess(nullptr) {}

  UpwardsMemoryQuery(const Instruction *Inst, const MemoryAccess *Access)
      : SawBackedgePhi(false), IsCall(ImmutableCallSite(Inst)), Inst(Inst),
        OriginalAccess(Access) {}
};

void MemorySSA::CachingWalker::invalidateInfo(MemoryAccess *MA) {

  // TODO: We can do much better cache invalidation with differently stored
  // caches.  For now, for MemoryUses, we simply remove them
  // from the cache, and kill the entire call/non-call cache for everything
  // else.  The problem is for phis or defs, currently we'd need to follow use
  // chains down and invalidate anything below us in the chain that currently
  // terminates at this access.

  // See if this is a MemoryUse, if so, just remove the cached info. MemoryUse
  // is by definition never a barrier, so nothing in the cache could point to
  // this use. In that case, we only need invalidate the info for the use
  // itself.

  if (MemoryUse *MU = dyn_cast<MemoryUse>(MA)) {
    UpwardsMemoryQuery Q;
    Instruction *I = MU->getMemoryInst();
    Q.IsCall = bool(ImmutableCallSite(I));
    Q.Inst = I;
    if (!Q.IsCall)
      Q.StartingLoc = MemoryLocation::get(I);
    doCacheRemove(MA, Q, Q.StartingLoc);
  } else {
    // If it is not a use, the best we can do right now is destroy the cache.
    CachedUpwardsClobberingCall.clear();
    CachedUpwardsClobberingAccess.clear();
  }

#ifdef EXPENSIVE_CHECKS
  // Run this only when expensive checks are enabled.
  verifyRemoved(MA);
#endif
}

void MemorySSA::CachingWalker::doCacheRemove(const MemoryAccess *M,
                                             const UpwardsMemoryQuery &Q,
                                             const MemoryLocation &Loc) {
  if (Q.IsCall)
    CachedUpwardsClobberingCall.erase(M);
  else
    CachedUpwardsClobberingAccess.erase({M, Loc});
}

void MemorySSA::CachingWalker::doCacheInsert(const MemoryAccess *M,
                                             MemoryAccess *Result,
                                             const UpwardsMemoryQuery &Q,
                                             const MemoryLocation &Loc) {
  // This is fine for Phis, since there are times where we can't optimize them.
  // Making a def its own clobber is never correct, though.
  assert((Result != M || isa<MemoryPhi>(M)) &&
         "Something can't clobber itself!");
  ++NumClobberCacheInserts;
  if (Q.IsCall)
    CachedUpwardsClobberingCall[M] = Result;
  else
    CachedUpwardsClobberingAccess[{M, Loc}] = Result;
}

MemoryAccess *
MemorySSA::CachingWalker::doCacheLookup(const MemoryAccess *M,
                                        const UpwardsMemoryQuery &Q,
                                        const MemoryLocation &Loc) {
  ++NumClobberCacheLookups;
  MemoryAccess *Result;

  if (Q.IsCall)
    Result = CachedUpwardsClobberingCall.lookup(M);
  else
    Result = CachedUpwardsClobberingAccess.lookup({M, Loc});

  if (Result)
    ++NumClobberCacheHits;
  return Result;
}

bool MemorySSA::CachingWalker::instructionClobbersQuery(
    const MemoryDef *MD, UpwardsMemoryQuery &Q,
    const MemoryLocation &Loc) const {
  Instruction *DefMemoryInst = MD->getMemoryInst();
  assert(DefMemoryInst && "Defining instruction not actually an instruction");

  if (!Q.IsCall)
    return AA->getModRefInfo(DefMemoryInst, Loc) & MRI_Mod;

  // If this is a call, mark it for caching
  if (ImmutableCallSite(DefMemoryInst))
    Q.VisitedCalls.push_back(MD);
  ModRefInfo I = AA->getModRefInfo(DefMemoryInst, ImmutableCallSite(Q.Inst));
  return I != MRI_NoModRef;
}

MemoryAccessPair MemorySSA::CachingWalker::UpwardsDFSWalk(
    MemoryAccess *StartingAccess, const MemoryLocation &Loc,
    UpwardsMemoryQuery &Q, bool FollowingBackedge) {
  MemoryAccess *ModifyingAccess = nullptr;

  auto DFI = df_begin(StartingAccess);
  for (auto DFE = df_end(StartingAccess); DFI != DFE;) {
    MemoryAccess *CurrAccess = *DFI;
    if (MSSA->isLiveOnEntryDef(CurrAccess))
      return {CurrAccess, Loc};
    // If this is a MemoryDef, check whether it clobbers our current query. This
    // needs to be done before consulting the cache, because the cache reports
    // the clobber for CurrAccess. If CurrAccess is a clobber for this query,
    // and we ask the cache for information first, then we might skip this
    // clobber, which is bad.
    if (auto *MD = dyn_cast<MemoryDef>(CurrAccess)) {
      // If we hit the top, stop following this path.
      // While we can do lookups, we can't sanely do inserts here unless we were
      // to track everything we saw along the way, since we don't know where we
      // will stop.
      if (instructionClobbersQuery(MD, Q, Loc)) {
        ModifyingAccess = CurrAccess;
        break;
      }
    }
    if (auto CacheResult = doCacheLookup(CurrAccess, Q, Loc))
      return {CacheResult, Loc};

    // We need to know whether it is a phi so we can track backedges.
    // Otherwise, walk all upward defs.
    if (!isa<MemoryPhi>(CurrAccess)) {
      ++DFI;
      continue;
    }

#ifndef NDEBUG
    // The loop below visits the phi's children for us. Because phis are the
    // only things with multiple edges, skipping the children should always lead
    // us to the end of the loop.
    //
    // Use a copy of DFI because skipChildren would kill our search stack, which
    // would make caching anything on the way back impossible.
    auto DFICopy = DFI;
    assert(DFICopy.skipChildren() == DFE &&
           "Skipping phi's children doesn't end the DFS?");
#endif

    const MemoryAccessPair PHIPair(CurrAccess, Loc);

    // Don't try to optimize this phi again if we've already tried to do so.
    if (!Q.Visited.insert(PHIPair).second) {
      ModifyingAccess = CurrAccess;
      break;
    }

    std::size_t InitialVisitedCallSize = Q.VisitedCalls.size();

    // Recurse on PHI nodes, since we need to change locations.
    // TODO: Allow graphtraits on pairs, which would turn this whole function
    // into a normal single depth first walk.
    MemoryAccess *FirstDef = nullptr;
    for (auto MPI = upward_defs_begin(PHIPair), MPE = upward_defs_end();
         MPI != MPE; ++MPI) {
      bool Backedge =
          !FollowingBackedge &&
          DT->dominates(CurrAccess->getBlock(), MPI.getPhiArgBlock());

      MemoryAccessPair CurrentPair =
          UpwardsDFSWalk(MPI->first, MPI->second, Q, Backedge);
      // All the phi arguments should reach the same point if we can bypass
      // this phi. The alternative is that they hit this phi node, which
      // means we can skip this argument.
      if (FirstDef && CurrentPair.first != PHIPair.first &&
          CurrentPair.first != FirstDef) {
        ModifyingAccess = CurrAccess;
        break;
      }

      if (!FirstDef)
        FirstDef = CurrentPair.first;
    }

    // If we exited the loop early, go with the result it gave us.
    if (!ModifyingAccess) {
      assert(FirstDef && "Found a Phi with no upward defs?");
      ModifyingAccess = FirstDef;
    } else {
      // If we can't optimize this Phi, then we can't safely cache any of the
      // calls we visited when trying to optimize it. Wipe them out now.
      Q.VisitedCalls.resize(InitialVisitedCallSize);
    }
    break;
  }

  if (!ModifyingAccess)
    return {MSSA->getLiveOnEntryDef(), Q.StartingLoc};

  const BasicBlock *OriginalBlock = StartingAccess->getBlock();
  assert(DFI.getPathLength() > 0 && "We dropped our path?");
  unsigned N = DFI.getPathLength();
  // If we found a clobbering def, the last element in the path will be our
  // clobber, so we don't want to cache that to itself. OTOH, if we optimized a
  // phi, we can add the last thing in the path to the cache, since that won't
  // be the result.
  if (DFI.getPath(N - 1) == ModifyingAccess)
    --N;
  for (; N > 1; --N) {
    MemoryAccess *CacheAccess = DFI.getPath(N - 1);
    BasicBlock *CurrBlock = CacheAccess->getBlock();
    if (!FollowingBackedge)
      doCacheInsert(CacheAccess, ModifyingAccess, Q, Loc);
    if (DT->dominates(CurrBlock, OriginalBlock) &&
        (CurrBlock != OriginalBlock || !FollowingBackedge ||
         MSSA->locallyDominates(CacheAccess, StartingAccess)))
      break;
  }

  // Cache everything else on the way back. The caller should cache
  // StartingAccess for us.
  for (; N > 1; --N) {
    MemoryAccess *CacheAccess = DFI.getPath(N - 1);
    doCacheInsert(CacheAccess, ModifyingAccess, Q, Loc);
  }
  assert(Q.Visited.size() < 1000 && "Visited too much");

  return {ModifyingAccess, Loc};
}

/// \brief Walk the use-def chains starting at \p MA and find
/// the MemoryAccess that actually clobbers Loc.
///
/// \returns our clobbering memory access
MemoryAccess *MemorySSA::CachingWalker::getClobberingMemoryAccess(
    MemoryAccess *StartingAccess, UpwardsMemoryQuery &Q) {
  return UpwardsDFSWalk(StartingAccess, Q.StartingLoc, Q, false).first;
}

MemoryAccess *MemorySSA::CachingWalker::getClobberingMemoryAccess(
    MemoryAccess *StartingAccess, MemoryLocation &Loc) {
  if (isa<MemoryPhi>(StartingAccess))
    return StartingAccess;

  auto *StartingUseOrDef = cast<MemoryUseOrDef>(StartingAccess);
  if (MSSA->isLiveOnEntryDef(StartingUseOrDef))
    return StartingUseOrDef;

  Instruction *I = StartingUseOrDef->getMemoryInst();

  // Conservatively, fences are always clobbers, so don't perform the walk if we
  // hit a fence.
  if (isa<FenceInst>(I))
    return StartingUseOrDef;

  UpwardsMemoryQuery Q;
  Q.OriginalAccess = StartingUseOrDef;
  Q.StartingLoc = Loc;
  Q.Inst = StartingUseOrDef->getMemoryInst();
  Q.IsCall = false;

  if (auto CacheResult = doCacheLookup(StartingUseOrDef, Q, Q.StartingLoc))
    return CacheResult;

  // Unlike the other function, do not walk to the def of a def, because we are
  // handed something we already believe is the clobbering access.
  MemoryAccess *DefiningAccess = isa<MemoryUse>(StartingUseOrDef)
                                     ? StartingUseOrDef->getDefiningAccess()
                                     : StartingUseOrDef;

  MemoryAccess *Clobber = getClobberingMemoryAccess(DefiningAccess, Q);
  // Only cache this if it wouldn't make Clobber point to itself.
  if (Clobber != StartingAccess)
    doCacheInsert(Q.OriginalAccess, Clobber, Q, Q.StartingLoc);
  DEBUG(dbgs() << "Starting Memory SSA clobber for " << *I << " is ");
  DEBUG(dbgs() << *StartingUseOrDef << "\n");
  DEBUG(dbgs() << "Final Memory SSA clobber for " << *I << " is ");
  DEBUG(dbgs() << *Clobber << "\n");
  return Clobber;
}

MemoryAccess *
MemorySSA::CachingWalker::getClobberingMemoryAccess(const Instruction *I) {
  // There should be no way to lookup an instruction and get a phi as the
  // access, since we only map BB's to PHI's. So, this must be a use or def.
  auto *StartingAccess = cast<MemoryUseOrDef>(MSSA->getMemoryAccess(I));

  // We can't sanely do anything with a FenceInst, they conservatively
  // clobber all memory, and have no locations to get pointers from to
  // try to disambiguate
  if (isa<FenceInst>(I))
    return StartingAccess;

  UpwardsMemoryQuery Q;
  Q.OriginalAccess = StartingAccess;
  Q.IsCall = bool(ImmutableCallSite(I));
  if (!Q.IsCall)
    Q.StartingLoc = MemoryLocation::get(I);
  Q.Inst = I;
  if (auto CacheResult = doCacheLookup(StartingAccess, Q, Q.StartingLoc))
    return CacheResult;

  // Start with the thing we already think clobbers this location
  MemoryAccess *DefiningAccess = StartingAccess->getDefiningAccess();

  // At this point, DefiningAccess may be the live on entry def.
  // If it is, we will not get a better result.
  if (MSSA->isLiveOnEntryDef(DefiningAccess))
    return DefiningAccess;

  MemoryAccess *Result = getClobberingMemoryAccess(DefiningAccess, Q);
  // DFS won't cache a result for DefiningAccess. So, if DefiningAccess isn't
  // our clobber, be sure that it gets a cache entry, too.
  if (Result != DefiningAccess)
    doCacheInsert(DefiningAccess, Result, Q, Q.StartingLoc);
  doCacheInsert(Q.OriginalAccess, Result, Q, Q.StartingLoc);
  // TODO: When this implementation is more mature, we may want to figure out
  // what this additional caching buys us. It's most likely A Good Thing.
  if (Q.IsCall)
    for (const MemoryAccess *MA : Q.VisitedCalls)
      if (MA != Result)
        doCacheInsert(MA, Result, Q, Q.StartingLoc);

  DEBUG(dbgs() << "Starting Memory SSA clobber for " << *I << " is ");
  DEBUG(dbgs() << *DefiningAccess << "\n");
  DEBUG(dbgs() << "Final Memory SSA clobber for " << *I << " is ");
  DEBUG(dbgs() << *Result << "\n");

  return Result;
}

// Verify that MA doesn't exist in any of the caches.
void MemorySSA::CachingWalker::verifyRemoved(MemoryAccess *MA) {
#ifndef NDEBUG
  for (auto &P : CachedUpwardsClobberingAccess)
    assert(P.first.first != MA && P.second != MA &&
           "Found removed MemoryAccess in cache.");
  for (auto &P : CachedUpwardsClobberingCall)
    assert(P.first != MA && P.second != MA &&
           "Found removed MemoryAccess in cache.");
#endif // !NDEBUG
}

MemoryAccess *
DoNothingMemorySSAWalker::getClobberingMemoryAccess(const Instruction *I) {
  MemoryAccess *MA = MSSA->getMemoryAccess(I);
  if (auto *Use = dyn_cast<MemoryUseOrDef>(MA))
    return Use->getDefiningAccess();
  return MA;
}

MemoryAccess *DoNothingMemorySSAWalker::getClobberingMemoryAccess(
    MemoryAccess *StartingAccess, MemoryLocation &) {
  if (auto *Use = dyn_cast<MemoryUseOrDef>(StartingAccess))
    return Use->getDefiningAccess();
  return StartingAccess;
}
}
