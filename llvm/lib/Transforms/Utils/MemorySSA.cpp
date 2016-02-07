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
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/MemorySSA.h"
#include <algorithm>

#define DEBUG_TYPE "memoryssa"
using namespace llvm;
STATISTIC(NumClobberCacheLookups, "Number of Memory SSA version cache lookups");
STATISTIC(NumClobberCacheHits, "Number of Memory SSA version cache hits");
STATISTIC(NumClobberCacheInserts, "Number of MemorySSA version cache inserts");
INITIALIZE_PASS_WITH_OPTIONS_BEGIN(MemorySSAPrinterPass, "print-memoryssa",
                                   "Memory SSA", true, true)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(GlobalsAAWrapperPass)
INITIALIZE_PASS_END(MemorySSAPrinterPass, "print-memoryssa", "Memory SSA", true,
                    true)
INITIALIZE_PASS(MemorySSALazy, "memoryssalazy", "Memory SSA", true, true)

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
    AccessListType *Accesses = It->second.get();
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
    AccessListType *Accesses = It->second.get();
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

MemorySSA::MemorySSA(Function &Func)
    : AA(nullptr), DT(nullptr), F(Func), LiveOnEntryDef(nullptr),
      Walker(nullptr), NextID(0) {}

MemorySSA::~MemorySSA() {
  // Drop all our references
  for (const auto &Pair : PerBlockAccesses)
    for (MemoryAccess &MA : *Pair.second)
      MA.dropAllReferences();
}

MemorySSA::AccessListType *MemorySSA::getOrCreateAccessList(BasicBlock *BB) {
  auto Res = PerBlockAccesses.insert(std::make_pair(BB, nullptr));

  if (Res.second)
    Res.first->second = make_unique<AccessListType>();
  return Res.first->second.get();
}

MemorySSAWalker *MemorySSA::buildMemorySSA(AliasAnalysis *AA,
                                           DominatorTree *DT) {
  if (Walker)
    return Walker;

  assert(!this->AA && !this->DT &&
         "MemorySSA without a walker already has AA or DT?");

  auto *Result = new CachingMemorySSAWalker(this, AA, DT);
  this->AA = AA;
  this->DT = DT;

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
    bool InsertIntoDefUse = false;
    bool InsertIntoDef = false;
    AccessListType *Accesses = nullptr;
    for (Instruction &I : B) {
      MemoryAccess *MA = createNewAccess(&I, true);
      if (!MA)
        continue;
      if (isa<MemoryDef>(MA))
        InsertIntoDef = true;
      else if (isa<MemoryUse>(MA))
        InsertIntoDefUse = true;

      if (!Accesses)
        Accesses = getOrCreateAccessList(&B);
      Accesses->push_back(MA);
    }
    if (InsertIntoDef)
      DefiningBlocks.insert(&B);
    if (InsertIntoDefUse)
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
  IDFCalculator IDFs(*DT);
  IDFs.setDefiningBlocks(DefiningBlocks);
  IDFs.setLiveInBlocks(LiveInBlocks);
  SmallVector<BasicBlock *, 32> IDFBlocks;
  IDFs.calculate(IDFBlocks);

  // Now place MemoryPhi nodes.
  for (auto &BB : IDFBlocks) {
    // Insert phi node
    AccessListType *Accesses = getOrCreateAccessList(BB);
    MemoryPhi *Phi = new MemoryPhi(F.getContext(), BB, NextID++);
    InstructionToMemoryAccess.insert(std::make_pair(BB, Phi));
    // Phi's always are placed at the front of the block.
    Accesses->push_front(Phi);
  }

  // Now do regular SSA renaming on the MemoryDef/MemoryUse. Visited will get
  // filled in with all blocks.
  SmallPtrSet<BasicBlock *, 16> Visited;
  renamePass(DT->getRootNode(), LiveOnEntryDef.get(), Visited);

  // Now optimize the MemoryUse's defining access to point to the nearest
  // dominating clobbering def.
  // This ensures that MemoryUse's that are killed by the same store are
  // immediate users of that store, one of the invariants we guarantee.
  for (auto DomNode : depth_first(DT)) {
    BasicBlock *BB = DomNode->getBlock();
    auto AI = PerBlockAccesses.find(BB);
    if (AI == PerBlockAccesses.end())
      continue;
    AccessListType *Accesses = AI->second.get();
    for (auto &MA : *Accesses) {
      if (auto *MU = dyn_cast<MemoryUse>(&MA)) {
        Instruction *Inst = MU->getMemoryInst();
        MU->setDefiningAccess(Result->getClobberingMemoryAccess(Inst));
      }
    }
  }

  // Mark the uses in unreachable blocks as live on entry, so that they go
  // somewhere.
  for (auto &BB : F)
    if (!Visited.count(&BB))
      markUnreachableAsLiveOnEntry(&BB);

  Walker = Result;
  return Walker;
}

/// \brief Helper function to create new memory accesses
MemoryAccess *MemorySSA::createNewAccess(Instruction *I, bool IgnoreNonMemory) {
  // Find out what affect this instruction has on memory.
  ModRefInfo ModRef = AA->getModRefInfo(I);
  bool Def = bool(ModRef & MRI_Mod);
  bool Use = bool(ModRef & MRI_Ref);

  // It's possible for an instruction to not modify memory at all. During
  // construction, we ignore them.
  if (IgnoreNonMemory && !Def && !Use)
    return nullptr;

  assert((Def || Use) &&
         "Trying to create a memory access with a non-memory instruction");

  MemoryUseOrDef *MA;
  if (Def)
    MA = new MemoryDef(I->getContext(), nullptr, I, I->getParent(),
                       NextID++);
  else
    MA =
        new MemoryUse(I->getContext(), nullptr, I, I->getParent());
  InstructionToMemoryAccess.insert(std::make_pair(I, MA));
  return MA;
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
      for (auto RAI = Accesses->rbegin(), RAE = Accesses->rend(); RAI != RAE;
           ++RAI) {
        if (isa<MemoryDef>(*RAI) || isa<MemoryPhi>(*RAI))
          return &*RAI;
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

void MemorySSA::print(raw_ostream &OS) const {
  MemorySSAAnnotatedWriter Writer(this);
  F.print(OS, &Writer);
}

void MemorySSA::dump() const {
  MemorySSAAnnotatedWriter Writer(this);
  F.print(dbgs(), &Writer);
}

/// \brief Verify the domination properties of MemorySSA by checking that each
/// definition dominates all of its uses.
void MemorySSA::verifyDomination(Function &F) {
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

      for (const auto &U : MD->users()) {
        BasicBlock *UseBlock;
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
void MemorySSA::verifyUseInDefs(MemoryAccess *Def, MemoryAccess *Use) {
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
void MemorySSA::verifyDefUses(Function &F) {
  for (BasicBlock &B : F) {
    // Phi nodes are attached to basic blocks
    if (MemoryPhi *Phi = getMemoryAccess(&B))
      for (unsigned I = 0, E = Phi->getNumIncomingValues(); I != E; ++I)
        verifyUseInDefs(Phi->getIncomingValue(I), Phi);

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
  return InstructionToMemoryAccess.lookup(I);
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
  // Get the access list for the block
  const AccessListType *AccessList = getBlockAccesses(Dominator->getBlock());
  AccessListType::const_reverse_iterator It(Dominator->getIterator());

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

char MemorySSAPrinterPass::ID = 0;

MemorySSAPrinterPass::MemorySSAPrinterPass() : FunctionPass(ID) {
  initializeMemorySSAPrinterPassPass(*PassRegistry::getPassRegistry());
}

void MemorySSAPrinterPass::releaseMemory() {
  // Subtlety: Be sure to delete the walker before MSSA, because the walker's
  // dtor may try to access MemorySSA.
  Walker.reset();
  MSSA.reset();
}

void MemorySSAPrinterPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<AAResultsWrapperPass>();
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addPreserved<DominatorTreeWrapperPass>();
  AU.addPreserved<GlobalsAAWrapperPass>();
}

bool MemorySSAPrinterPass::doInitialization(Module &M) {
  VerifyMemorySSA = M.getContext()
                        .getOption<bool, MemorySSAPrinterPass,
                                   &MemorySSAPrinterPass::VerifyMemorySSA>();
  return false;
}

void MemorySSAPrinterPass::registerOptions() {
  OptionRegistry::registerOption<bool, MemorySSAPrinterPass,
                                 &MemorySSAPrinterPass::VerifyMemorySSA>(
      "verify-memoryssa", "Run the Memory SSA verifier", false);
}

void MemorySSAPrinterPass::print(raw_ostream &OS, const Module *M) const {
  MSSA->print(OS);
}

bool MemorySSAPrinterPass::runOnFunction(Function &F) {
  this->F = &F;
  MSSA.reset(new MemorySSA(F));
  AliasAnalysis *AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
  DominatorTree *DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  Walker.reset(MSSA->buildMemorySSA(AA, DT));

  if (VerifyMemorySSA) {
    MSSA->verifyDefUses(F);
    MSSA->verifyDomination(F);
  }

  return false;
}

char MemorySSALazy::ID = 0;

MemorySSALazy::MemorySSALazy() : FunctionPass(ID) {
  initializeMemorySSALazyPass(*PassRegistry::getPassRegistry());
}

void MemorySSALazy::releaseMemory() { MSSA.reset(); }

bool MemorySSALazy::runOnFunction(Function &F) {
  MSSA.reset(new MemorySSA(F));
  return false;
}

MemorySSAWalker::MemorySSAWalker(MemorySSA *M) : MSSA(M) {}

CachingMemorySSAWalker::CachingMemorySSAWalker(MemorySSA *M, AliasAnalysis *A,
                                               DominatorTree *D)
    : MemorySSAWalker(M), AA(A), DT(D) {}

CachingMemorySSAWalker::~CachingMemorySSAWalker() {}

struct CachingMemorySSAWalker::UpwardsMemoryQuery {
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
  // Set of visited call accesses for this query. This is separated out because
  // you can always cache and lookup the result of call queries (IE when IsCall
  // == true) for every call in the chain. The calls have no AA location
  // associated with them with them, and thus, no context dependence.
  SmallPtrSet<const MemoryAccess *, 32> VisitedCalls;
  // The MemoryAccess we actually got called with, used to test local domination
  const MemoryAccess *OriginalAccess;
  // The Datalayout for the module we started in
  const DataLayout *DL;

  UpwardsMemoryQuery()
      : SawBackedgePhi(false), IsCall(false), Inst(nullptr),
        OriginalAccess(nullptr), DL(nullptr) {}
};

void CachingMemorySSAWalker::doCacheRemove(const MemoryAccess *M,
                                           const UpwardsMemoryQuery &Q,
                                           const MemoryLocation &Loc) {
  if (Q.IsCall)
    CachedUpwardsClobberingCall.erase(M);
  else
    CachedUpwardsClobberingAccess.erase({M, Loc});
}

void CachingMemorySSAWalker::doCacheInsert(const MemoryAccess *M,
                                           MemoryAccess *Result,
                                           const UpwardsMemoryQuery &Q,
                                           const MemoryLocation &Loc) {
  ++NumClobberCacheInserts;
  if (Q.IsCall)
    CachedUpwardsClobberingCall[M] = Result;
  else
    CachedUpwardsClobberingAccess[{M, Loc}] = Result;
}

MemoryAccess *CachingMemorySSAWalker::doCacheLookup(const MemoryAccess *M,
                                                    const UpwardsMemoryQuery &Q,
                                                    const MemoryLocation &Loc) {
  ++NumClobberCacheLookups;
  MemoryAccess *Result = nullptr;

  if (Q.IsCall)
    Result = CachedUpwardsClobberingCall.lookup(M);
  else
    Result = CachedUpwardsClobberingAccess.lookup({M, Loc});

  if (Result)
    ++NumClobberCacheHits;
  return Result;
}

bool CachingMemorySSAWalker::instructionClobbersQuery(
    const MemoryDef *MD, UpwardsMemoryQuery &Q,
    const MemoryLocation &Loc) const {
  Instruction *DefMemoryInst = MD->getMemoryInst();
  assert(DefMemoryInst && "Defining instruction not actually an instruction");

  if (!Q.IsCall)
    return AA->getModRefInfo(DefMemoryInst, Loc) & MRI_Mod;

  // If this is a call, mark it for caching
  if (ImmutableCallSite(DefMemoryInst))
    Q.VisitedCalls.insert(MD);
  ModRefInfo I = AA->getModRefInfo(DefMemoryInst, ImmutableCallSite(Q.Inst));
  return I != MRI_NoModRef;
}

MemoryAccessPair CachingMemorySSAWalker::UpwardsDFSWalk(
    MemoryAccess *StartingAccess, const MemoryLocation &Loc,
    UpwardsMemoryQuery &Q, bool FollowingBackedge) {
  MemoryAccess *ModifyingAccess = nullptr;

  auto DFI = df_begin(StartingAccess);
  for (auto DFE = df_end(StartingAccess); DFI != DFE;) {
    MemoryAccess *CurrAccess = *DFI;
    if (MSSA->isLiveOnEntryDef(CurrAccess))
      return {CurrAccess, Loc};
    if (auto CacheResult = doCacheLookup(CurrAccess, Q, Loc))
      return {CacheResult, Loc};
    // If this is a MemoryDef, check whether it clobbers our current query.
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

    // We need to know whether it is a phi so we can track backedges.
    // Otherwise, walk all upward defs.
    if (!isa<MemoryPhi>(CurrAccess)) {
      ++DFI;
      continue;
    }

    // Recurse on PHI nodes, since we need to change locations.
    // TODO: Allow graphtraits on pairs, which would turn this whole function
    // into a normal single depth first walk.
    MemoryAccess *FirstDef = nullptr;
    DFI = DFI.skipChildren();
    const MemoryAccessPair PHIPair(CurrAccess, Loc);
    bool VisitedOnlyOne = true;
    for (auto MPI = upward_defs_begin(PHIPair), MPE = upward_defs_end();
         MPI != MPE; ++MPI) {
      // Don't follow this path again if we've followed it once
      if (!Q.Visited.insert(*MPI).second)
        continue;

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
      else
        VisitedOnlyOne = false;
    }

    // The above loop determines if all arguments of the phi node reach the
    // same place. However we skip arguments that are cyclically dependent
    // only on the value of this phi node. This means in some cases, we may
    // only visit one argument of the phi node, and the above loop will
    // happily say that all the arguments are the same. However, in that case,
    // we still can't walk past the phi node, because that argument still
    // kills the access unless we hit the top of the function when walking
    // that argument.
    if (VisitedOnlyOne && FirstDef && !MSSA->isLiveOnEntryDef(FirstDef))
      ModifyingAccess = CurrAccess;
  }

  if (!ModifyingAccess)
    return {MSSA->getLiveOnEntryDef(), Q.StartingLoc};

  const BasicBlock *OriginalBlock = Q.OriginalAccess->getBlock();
  unsigned N = DFI.getPathLength();
  MemoryAccess *FinalAccess = ModifyingAccess;
  for (; N != 0; --N) {
    ModifyingAccess = DFI.getPath(N - 1);
    BasicBlock *CurrBlock = ModifyingAccess->getBlock();
    if (!FollowingBackedge)
      doCacheInsert(ModifyingAccess, FinalAccess, Q, Loc);
    if (DT->dominates(CurrBlock, OriginalBlock) &&
        (CurrBlock != OriginalBlock || !FollowingBackedge ||
         MSSA->locallyDominates(ModifyingAccess, Q.OriginalAccess)))
      break;
  }

  // Cache everything else on the way back. The caller should cache
  // Q.OriginalAccess for us.
  for (; N != 0; --N) {
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
MemoryAccess *
CachingMemorySSAWalker::getClobberingMemoryAccess(MemoryAccess *StartingAccess,
                                                  UpwardsMemoryQuery &Q) {
  return UpwardsDFSWalk(StartingAccess, Q.StartingLoc, Q, false).first;
}

MemoryAccess *
CachingMemorySSAWalker::getClobberingMemoryAccess(MemoryAccess *StartingAccess,
                                                  MemoryLocation &Loc) {
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
  Q.DL = &Q.Inst->getModule()->getDataLayout();

  if (auto CacheResult = doCacheLookup(StartingUseOrDef, Q, Q.StartingLoc))
    return CacheResult;

  // Unlike the other function, do not walk to the def of a def, because we are
  // handed something we already believe is the clobbering access.
  MemoryAccess *DefiningAccess = isa<MemoryUse>(StartingUseOrDef)
                                     ? StartingUseOrDef->getDefiningAccess()
                                     : StartingUseOrDef;

  MemoryAccess *Clobber = getClobberingMemoryAccess(DefiningAccess, Q);
  doCacheInsert(Q.OriginalAccess, Clobber, Q, Q.StartingLoc);
  DEBUG(dbgs() << "Starting Memory SSA clobber for " << *I << " is ");
  DEBUG(dbgs() << *StartingUseOrDef << "\n");
  DEBUG(dbgs() << "Final Memory SSA clobber for " << *I << " is ");
  DEBUG(dbgs() << *Clobber << "\n");
  return Clobber;
}

MemoryAccess *
CachingMemorySSAWalker::getClobberingMemoryAccess(const Instruction *I) {
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
  Q.DL = &Q.Inst->getModule()->getDataLayout();
  if (auto CacheResult = doCacheLookup(StartingAccess, Q, Q.StartingLoc))
    return CacheResult;

  // Start with the thing we already think clobbers this location
  MemoryAccess *DefiningAccess = StartingAccess->getDefiningAccess();

  // At this point, DefiningAccess may be the live on entry def.
  // If it is, we will not get a better result.
  if (MSSA->isLiveOnEntryDef(DefiningAccess))
    return DefiningAccess;

  MemoryAccess *Result = getClobberingMemoryAccess(DefiningAccess, Q);
  doCacheInsert(Q.OriginalAccess, Result, Q, Q.StartingLoc);
  // TODO: When this implementation is more mature, we may want to figure out
  // what this additional caching buys us. It's most likely A Good Thing.
  if (Q.IsCall)
    for (const MemoryAccess *MA : Q.VisitedCalls)
      doCacheInsert(MA, Result, Q, Q.StartingLoc);

  DEBUG(dbgs() << "Starting Memory SSA clobber for " << *I << " is ");
  DEBUG(dbgs() << *DefiningAccess << "\n");
  DEBUG(dbgs() << "Final Memory SSA clobber for " << *I << " is ");
  DEBUG(dbgs() << *Result << "\n");

  return Result;
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
