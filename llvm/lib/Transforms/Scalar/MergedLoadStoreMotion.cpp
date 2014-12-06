//===- MergedLoadStoreMotion.cpp - merge and hoist/sink load/stores -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//! \file
//! \brief This pass performs merges of loads and stores on both sides of a
//  diamond (hammock). It hoists the loads and sinks the stores.
//
// The algorithm iteratively hoists two loads to the same address out of a
// diamond (hammock) and merges them into a single load in the header. Similar
// it sinks and merges two stores to the tail block (footer). The algorithm
// iterates over the instructions of one side of the diamond and attempts to
// find a matching load/store on the other side. It hoists / sinks when it
// thinks it safe to do so.  This optimization helps with eg. hiding load
// latencies, triggering if-conversion, and reducing static code size.
//
//===----------------------------------------------------------------------===//
//
//
// Example:
// Diamond shaped code before merge:
//
//            header:
//                     br %cond, label %if.then, label %if.else
//                        +                    +
//                       +                      +
//                      +                        +
//            if.then:                         if.else:
//               %lt = load %addr_l               %le = load %addr_l
//               <use %lt>                        <use %le>
//               <...>                            <...>
//               store %st, %addr_s               store %se, %addr_s
//               br label %if.end                 br label %if.end
//                     +                         +
//                      +                       +
//                       +                     +
//            if.end ("footer"):
//                     <...>
//
// Diamond shaped code after merge:
//
//            header:
//                     %l = load %addr_l
//                     br %cond, label %if.then, label %if.else
//                        +                    +
//                       +                      +
//                      +                        +
//            if.then:                         if.else:
//               <use %l>                         <use %l>
//               <...>                            <...>
//               br label %if.end                 br label %if.end
//                      +                        +
//                       +                      +
//                        +                    +
//            if.end ("footer"):
//                     %s.sink = phi [%st, if.then], [%se, if.else]
//                     <...>
//                     store %s.sink, %addr_s
//                     <...>
//
//
//===----------------------- TODO -----------------------------------------===//
//
// 1) Generalize to regions other than diamonds
// 2) Be more aggressive merging memory operations
// Note that both changes require register pressure control
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
#include <vector>
using namespace llvm;

#define DEBUG_TYPE "mldst-motion"

//===----------------------------------------------------------------------===//
//                         MergedLoadStoreMotion Pass
//===----------------------------------------------------------------------===//
static cl::opt<bool>
EnableMLSM("mlsm", cl::desc("Enable motion of merged load and store"),
           cl::init(true));

namespace {
class MergedLoadStoreMotion : public FunctionPass {
  AliasAnalysis *AA;
  MemoryDependenceAnalysis *MD;

public:
  static char ID; // Pass identification, replacement for typeid
  explicit MergedLoadStoreMotion(void)
      : FunctionPass(ID), MD(nullptr), MagicCompileTimeControl(250) {
    initializeMergedLoadStoreMotionPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;

private:
  // This transformation requires dominator postdominator info
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetLibraryInfo>();
    AU.addRequired<MemoryDependenceAnalysis>();
    AU.addRequired<AliasAnalysis>();
    AU.addPreserved<AliasAnalysis>();
  }

  // Helper routines

  ///
  /// \brief Remove instruction from parent and update memory dependence
  /// analysis.
  ///
  void removeInstruction(Instruction *Inst);
  BasicBlock *getDiamondTail(BasicBlock *BB);
  bool isDiamondHead(BasicBlock *BB);
  // Routines for hoisting loads
  bool isLoadHoistBarrier(Instruction *Inst);
  LoadInst *canHoistFromBlock(BasicBlock *BB, LoadInst *LI);
  void hoistInstruction(BasicBlock *BB, Instruction *HoistCand,
                        Instruction *ElseInst);
  bool isSafeToHoist(Instruction *I) const;
  bool hoistLoad(BasicBlock *BB, LoadInst *HoistCand, LoadInst *ElseInst);
  bool mergeLoads(BasicBlock *BB);
  // Routines for sinking stores
  StoreInst *canSinkFromBlock(BasicBlock *BB, StoreInst *SI);
  PHINode *getPHIOperand(BasicBlock *BB, StoreInst *S0, StoreInst *S1);
  bool isStoreSinkBarrier(Instruction *Inst);
  bool sinkStore(BasicBlock *BB, StoreInst *SinkCand, StoreInst *ElseInst);
  bool mergeStores(BasicBlock *BB);
  // The mergeLoad/Store algorithms could have Size0 * Size1 complexity,
  // where Size0 and Size1 are the #instructions on the two sides of
  // the diamond. The constant chosen here is arbitrary. Compiler Time
  // Control is enforced by the check Size0 * Size1 < MagicCompileTimeControl.
  const int MagicCompileTimeControl;
};

char MergedLoadStoreMotion::ID = 0;
}

///
/// \brief createMergedLoadStoreMotionPass - The public interface to this file.
///
FunctionPass *llvm::createMergedLoadStoreMotionPass() {
  return new MergedLoadStoreMotion();
}

INITIALIZE_PASS_BEGIN(MergedLoadStoreMotion, "mldst-motion",
                      "MergedLoadStoreMotion", false, false)
INITIALIZE_PASS_DEPENDENCY(MemoryDependenceAnalysis)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfo)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_END(MergedLoadStoreMotion, "mldst-motion",
                    "MergedLoadStoreMotion", false, false)

///
/// \brief Remove instruction from parent and update memory dependence analysis.
///
void MergedLoadStoreMotion::removeInstruction(Instruction *Inst) {
  // Notify the memory dependence analysis.
  if (MD) {
    MD->removeInstruction(Inst);
    if (LoadInst *LI = dyn_cast<LoadInst>(Inst))
      MD->invalidateCachedPointerInfo(LI->getPointerOperand());
    if (Inst->getType()->getScalarType()->isPointerTy()) {
      MD->invalidateCachedPointerInfo(Inst);
    }
  }
  Inst->eraseFromParent();
}

///
/// \brief Return tail block of a diamond.
///
BasicBlock *MergedLoadStoreMotion::getDiamondTail(BasicBlock *BB) {
  assert(isDiamondHead(BB) && "Basic block is not head of a diamond");
  BranchInst *BI = (BranchInst *)(BB->getTerminator());
  BasicBlock *Succ0 = BI->getSuccessor(0);
  BasicBlock *Tail = Succ0->getTerminator()->getSuccessor(0);
  return Tail;
}

///
/// \brief True when BB is the head of a diamond (hammock)
///
bool MergedLoadStoreMotion::isDiamondHead(BasicBlock *BB) {
  if (!BB)
    return false;
  if (!isa<BranchInst>(BB->getTerminator()))
    return false;
  if (BB->getTerminator()->getNumSuccessors() != 2)
    return false;

  BranchInst *BI = (BranchInst *)(BB->getTerminator());
  BasicBlock *Succ0 = BI->getSuccessor(0);
  BasicBlock *Succ1 = BI->getSuccessor(1);

  if (!Succ0->getSinglePredecessor() ||
      Succ0->getTerminator()->getNumSuccessors() != 1)
    return false;
  if (!Succ1->getSinglePredecessor() ||
      Succ1->getTerminator()->getNumSuccessors() != 1)
    return false;

  BasicBlock *Tail = Succ0->getTerminator()->getSuccessor(0);
  // Ignore triangles.
  if (Succ1->getTerminator()->getSuccessor(0) != Tail)
    return false;
  return true;
}

///
/// \brief True when instruction is a hoist barrier for a load
///
/// Whenever an instruction could possibly modify the value
/// being loaded or protect against the load from happening
/// it is considered a hoist barrier.
///
bool MergedLoadStoreMotion::isLoadHoistBarrier(Instruction *Inst) {
  // FIXME: A call with no side effects should not be a barrier.
  // Aren't all such calls covered by mayHaveSideEffects() below?
  // Then this check can be removed.
  if (isa<CallInst>(Inst))
    return true;
  if (isa<TerminatorInst>(Inst))
    return true;
  // FIXME: Conservatively let a store instruction block the load.
  // Use alias analysis instead.
  if (isa<StoreInst>(Inst))
    return true;
  // Note: mayHaveSideEffects covers all instructions that could
  // trigger a change to state. Eg. in-flight stores have to be executed
  // before ordered loads or fences, calls could invoke functions that store
  // data to memory etc.
  if (Inst->mayHaveSideEffects()) {
    return true;
  }
  DEBUG(dbgs() << "No Hoist Barrier\n");
  return false;
}

///
/// \brief Decide if a load can be hoisted
///
/// When there is a load in \p BB to the same address as \p LI
/// and it can be hoisted from \p BB, return that load.
/// Otherwise return Null.
///
LoadInst *MergedLoadStoreMotion::canHoistFromBlock(BasicBlock *BB,
                                                   LoadInst *LI) {
  LoadInst *I = nullptr;
  assert(isa<LoadInst>(LI));
  if (LI->isUsedOutsideOfBlock(LI->getParent()))
    return nullptr;

  for (BasicBlock::iterator BBI = BB->begin(), BBE = BB->end(); BBI != BBE;
       ++BBI) {
    Instruction *Inst = BBI;

    // Only merge and hoist loads when their result in used only in BB
    if (isLoadHoistBarrier(Inst))
      break;
    if (!isa<LoadInst>(Inst))
      continue;
    if (Inst->isUsedOutsideOfBlock(Inst->getParent()))
      continue;

    AliasAnalysis::Location LocLI = AA->getLocation(LI);
    AliasAnalysis::Location LocInst = AA->getLocation((LoadInst *)Inst);
    if (AA->isMustAlias(LocLI, LocInst) && LI->getType() == Inst->getType()) {
      I = (LoadInst *)Inst;
      break;
    }
  }
  return I;
}

///
/// \brief Merge two equivalent instructions \p HoistCand and \p ElseInst into
/// \p BB
///
/// BB is the head of a diamond
///
void MergedLoadStoreMotion::hoistInstruction(BasicBlock *BB,
                                             Instruction *HoistCand,
                                             Instruction *ElseInst) {
  DEBUG(dbgs() << " Hoist Instruction into BB \n"; BB->dump();
        dbgs() << "Instruction Left\n"; HoistCand->dump(); dbgs() << "\n";
        dbgs() << "Instruction Right\n"; ElseInst->dump(); dbgs() << "\n");
  // Hoist the instruction.
  assert(HoistCand->getParent() != BB);

  // Intersect optional metadata.
  HoistCand->intersectOptionalDataWith(ElseInst);
  HoistCand->dropUnknownMetadata();

  // Prepend point for instruction insert
  Instruction *HoistPt = BB->getTerminator();

  // Merged instruction
  Instruction *HoistedInst = HoistCand->clone();

  // Notify AA of the new value.
  if (isa<LoadInst>(HoistCand))
    AA->copyValue(HoistCand, HoistedInst);

  // Hoist instruction.
  HoistedInst->insertBefore(HoistPt);

  HoistCand->replaceAllUsesWith(HoistedInst);
  removeInstruction(HoistCand);
  // Replace the else block instruction.
  ElseInst->replaceAllUsesWith(HoistedInst);
  removeInstruction(ElseInst);
}

///
/// \brief Return true if no operand of \p I is defined in I's parent block
///
bool MergedLoadStoreMotion::isSafeToHoist(Instruction *I) const {
  BasicBlock *Parent = I->getParent();
  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
    Instruction *Instr = dyn_cast<Instruction>(I->getOperand(i));
    if (Instr && Instr->getParent() == Parent)
      return false;
  }
  return true;
}

///
/// \brief Merge two equivalent loads and GEPs and hoist into diamond head
///
bool MergedLoadStoreMotion::hoistLoad(BasicBlock *BB, LoadInst *L0,
                                      LoadInst *L1) {
  // Only one definition?
  Instruction *A0 = dyn_cast<Instruction>(L0->getPointerOperand());
  Instruction *A1 = dyn_cast<Instruction>(L1->getPointerOperand());
  if (A0 && A1 && A0->isIdenticalTo(A1) && isSafeToHoist(A0) &&
      A0->hasOneUse() && (A0->getParent() == L0->getParent()) &&
      A1->hasOneUse() && (A1->getParent() == L1->getParent()) &&
      isa<GetElementPtrInst>(A0)) {
    DEBUG(dbgs() << "Hoist Instruction into BB \n"; BB->dump();
          dbgs() << "Instruction Left\n"; L0->dump(); dbgs() << "\n";
          dbgs() << "Instruction Right\n"; L1->dump(); dbgs() << "\n");
    hoistInstruction(BB, A0, A1);
    hoistInstruction(BB, L0, L1);
    return true;
  } else
    return false;
}

///
/// \brief Try to hoist two loads to same address into diamond header
///
/// Starting from a diamond head block, iterate over the instructions in one
/// successor block and try to match a load in the second successor.
///
bool MergedLoadStoreMotion::mergeLoads(BasicBlock *BB) {
  bool MergedLoads = false;
  assert(isDiamondHead(BB));
  BranchInst *BI = dyn_cast<BranchInst>(BB->getTerminator());
  BasicBlock *Succ0 = BI->getSuccessor(0);
  BasicBlock *Succ1 = BI->getSuccessor(1);
  // #Instructions in Succ1 for Compile Time Control
  int Size1 = Succ1->size();
  int NLoads = 0;
  for (BasicBlock::iterator BBI = Succ0->begin(), BBE = Succ0->end();
       BBI != BBE;) {

    Instruction *I = BBI;
    ++BBI;
    if (isLoadHoistBarrier(I))
      break;

    // Only move non-simple (atomic, volatile) loads.
    if (!isa<LoadInst>(I))
      continue;

    LoadInst *L0 = (LoadInst *)I;
    if (!L0->isSimple())
      continue;

    ++NLoads;
    if (NLoads * Size1 >= MagicCompileTimeControl)
      break;
    if (LoadInst *L1 = canHoistFromBlock(Succ1, L0)) {
      bool Res = hoistLoad(BB, L0, L1);
      MergedLoads |= Res;
      // Don't attempt to hoist above loads that had not been hoisted.
      if (!Res)
        break;
    }
  }
  return MergedLoads;
}

///
/// \brief True when instruction is sink barrier for a store
/// 
bool MergedLoadStoreMotion::isStoreSinkBarrier(Instruction *Inst) {
  // FIXME: Conservatively let a load instruction block the store.
  // Use alias analysis instead.
  if (isa<LoadInst>(Inst))
    return true;
  if (isa<CallInst>(Inst))
    return true;
  if (isa<TerminatorInst>(Inst) && !isa<BranchInst>(Inst))
    return true;
  // Note: mayHaveSideEffects covers all instructions that could
  // trigger a change to state. Eg. in-flight stores have to be executed
  // before ordered loads or fences, calls could invoke functions that store
  // data to memory etc.
  if (!isa<StoreInst>(Inst) && Inst->mayHaveSideEffects()) {
    return true;
  }
  DEBUG(dbgs() << "No Sink Barrier\n");
  return false;
}

///
/// \brief Check if \p BB contains a store to the same address as \p SI
///
/// \return The store in \p  when it is safe to sink. Otherwise return Null.
///
StoreInst *MergedLoadStoreMotion::canSinkFromBlock(BasicBlock *BB,
                                                   StoreInst *SI) {
  StoreInst *I = 0;
  DEBUG(dbgs() << "can Sink? : "; SI->dump(); dbgs() << "\n");
  for (BasicBlock::reverse_iterator RBI = BB->rbegin(), RBE = BB->rend();
       RBI != RBE; ++RBI) {
    Instruction *Inst = &*RBI;

    // Only move loads if they are used in the block.
    if (isStoreSinkBarrier(Inst))
      break;
    if (isa<StoreInst>(Inst)) {
      AliasAnalysis::Location LocSI = AA->getLocation(SI);
      AliasAnalysis::Location LocInst = AA->getLocation((StoreInst *)Inst);
      if (AA->isMustAlias(LocSI, LocInst)) {
        I = (StoreInst *)Inst;
        break;
      }
    }
  }
  return I;
}

///
/// \brief Create a PHI node in BB for the operands of S0 and S1
///
PHINode *MergedLoadStoreMotion::getPHIOperand(BasicBlock *BB, StoreInst *S0,
                                              StoreInst *S1) {
  // Create a phi if the values mismatch.
  PHINode *NewPN = 0;
  Value *Opd1 = S0->getValueOperand();
  Value *Opd2 = S1->getValueOperand();
  if (Opd1 != Opd2) {
    NewPN = PHINode::Create(Opd1->getType(), 2, Opd2->getName() + ".sink",
                            BB->begin());
    NewPN->addIncoming(Opd1, S0->getParent());
    NewPN->addIncoming(Opd2, S1->getParent());
    if (NewPN->getType()->getScalarType()->isPointerTy()) {
      // Notify AA of the new value.
      AA->copyValue(Opd1, NewPN);
      AA->copyValue(Opd2, NewPN);
      // AA needs to be informed when a PHI-use of the pointer value is added
      for (unsigned I = 0, E = NewPN->getNumIncomingValues(); I != E; ++I) {
        unsigned J = PHINode::getOperandNumForIncomingValue(I);
        AA->addEscapingUse(NewPN->getOperandUse(J));
      }
      if (MD)
        MD->invalidateCachedPointerInfo(NewPN);
    }
  }
  return NewPN;
}

///
/// \brief Merge two stores to same address and sink into \p BB
///
/// Also sinks GEP instruction computing the store address
///
bool MergedLoadStoreMotion::sinkStore(BasicBlock *BB, StoreInst *S0,
                                      StoreInst *S1) {
  // Only one definition?
  Instruction *A0 = dyn_cast<Instruction>(S0->getPointerOperand());
  Instruction *A1 = dyn_cast<Instruction>(S1->getPointerOperand());
  if (A0 && A1 && A0->isIdenticalTo(A1) && A0->hasOneUse() &&
      (A0->getParent() == S0->getParent()) && A1->hasOneUse() &&
      (A1->getParent() == S1->getParent()) && isa<GetElementPtrInst>(A0)) {
    DEBUG(dbgs() << "Sink Instruction into BB \n"; BB->dump();
          dbgs() << "Instruction Left\n"; S0->dump(); dbgs() << "\n";
          dbgs() << "Instruction Right\n"; S1->dump(); dbgs() << "\n");
    // Hoist the instruction.
    BasicBlock::iterator InsertPt = BB->getFirstInsertionPt();
    // Intersect optional metadata.
    S0->intersectOptionalDataWith(S1);
    S0->dropUnknownMetadata();

    // Create the new store to be inserted at the join point.
    StoreInst *SNew = (StoreInst *)(S0->clone());
    Instruction *ANew = A0->clone();
    AA->copyValue(S0, SNew);
    SNew->insertBefore(InsertPt);
    ANew->insertBefore(SNew);

    assert(S0->getParent() == A0->getParent());
    assert(S1->getParent() == A1->getParent());

    PHINode *NewPN = getPHIOperand(BB, S0, S1);
    // New PHI operand? Use it.
    if (NewPN)
      SNew->setOperand(0, NewPN);
    removeInstruction(S0);
    removeInstruction(S1);
    A0->replaceAllUsesWith(ANew);
    removeInstruction(A0);
    A1->replaceAllUsesWith(ANew);
    removeInstruction(A1);
    return true;
  }
  return false;
}

///
/// \brief True when two stores are equivalent and can sink into the footer
///
/// Starting from a diamond tail block, iterate over the instructions in one
/// predecessor block and try to match a store in the second predecessor.
///
bool MergedLoadStoreMotion::mergeStores(BasicBlock *T) {

  bool MergedStores = false;
  assert(T && "Footer of a diamond cannot be empty");

  pred_iterator PI = pred_begin(T), E = pred_end(T);
  assert(PI != E);
  BasicBlock *Pred0 = *PI;
  ++PI;
  BasicBlock *Pred1 = *PI;
  ++PI;
  // tail block  of a diamond/hammock?
  if (Pred0 == Pred1)
    return false; // No.
  if (PI != E)
    return false; // No. More than 2 predecessors.

  // #Instructions in Succ1 for Compile Time Control
  int Size1 = Pred1->size();
  int NStores = 0;

  for (BasicBlock::reverse_iterator RBI = Pred0->rbegin(), RBE = Pred0->rend();
       RBI != RBE;) {

    Instruction *I = &*RBI;
    ++RBI;
    if (isStoreSinkBarrier(I))
      break;
    // Sink move non-simple (atomic, volatile) stores
    if (!isa<StoreInst>(I))
      continue;
    StoreInst *S0 = (StoreInst *)I;
    if (!S0->isSimple())
      continue;

    ++NStores;
    if (NStores * Size1 >= MagicCompileTimeControl)
      break;
    if (StoreInst *S1 = canSinkFromBlock(Pred1, S0)) {
      bool Res = sinkStore(T, S0, S1);
      MergedStores |= Res;
      // Don't attempt to sink below stores that had to stick around
      // But after removal of a store and some of its feeding
      // instruction search again from the beginning since the iterator
      // is likely stale at this point.
      if (!Res)
        break;
      else {
        RBI = Pred0->rbegin();
        RBE = Pred0->rend();
        DEBUG(dbgs() << "Search again\n"; Instruction *I = &*RBI; I->dump());
      }
    }
  }
  return MergedStores;
}
///
/// \brief Run the transformation for each function
///
bool MergedLoadStoreMotion::runOnFunction(Function &F) {
  MD = &getAnalysis<MemoryDependenceAnalysis>();
  AA = &getAnalysis<AliasAnalysis>();

  bool Changed = false;
  if (!EnableMLSM)
    return false;
  DEBUG(dbgs() << "Instruction Merger\n");

  // Merge unconditional branches, allowing PRE to catch more
  // optimization opportunities.
  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE;) {
    BasicBlock *BB = FI++;

    // Hoist equivalent loads and sink stores
    // outside diamonds when possible
    // Run outside core GVN
    if (isDiamondHead(BB)) {
      Changed |= mergeLoads(BB);
      Changed |= mergeStores(getDiamondTail(BB));
    }
  }
  return Changed;
}
