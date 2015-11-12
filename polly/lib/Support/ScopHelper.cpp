//===- ScopHelper.cpp - Some Helper Functions for Scop.  ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Small functions that help with Scop and LLVM-IR.
//
//===----------------------------------------------------------------------===//

#include "polly/Support/ScopHelper.h"
#include "polly/Options.h"
#include "polly/ScopInfo.h"
#include "polly/Support/SCEVValidator.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;
using namespace polly;

#define DEBUG_TYPE "polly-scop-helper"

Value *polly::getPointerOperand(Instruction &Inst) {
  if (LoadInst *load = dyn_cast<LoadInst>(&Inst))
    return load->getPointerOperand();
  else if (StoreInst *store = dyn_cast<StoreInst>(&Inst))
    return store->getPointerOperand();
  else if (GetElementPtrInst *gep = dyn_cast<GetElementPtrInst>(&Inst))
    return gep->getPointerOperand();

  return 0;
}

bool polly::hasInvokeEdge(const PHINode *PN) {
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i < e; ++i)
    if (InvokeInst *II = dyn_cast<InvokeInst>(PN->getIncomingValue(i)))
      if (II->getParent() == PN->getIncomingBlock(i))
        return true;

  return false;
}

// Ensures that there is just one predecessor to the entry node from outside the
// region.
// The identity of the region entry node is preserved.
static void simplifyRegionEntry(Region *R, DominatorTree *DT, LoopInfo *LI,
                                RegionInfo *RI) {
  BasicBlock *EnteringBB = R->getEnteringBlock();
  BasicBlock *Entry = R->getEntry();

  // Before (one of):
  //
  //                       \    /            //
  //                      EnteringBB         //
  //                        |    \------>    //
  //   \   /                |                //
  //   Entry <--\         Entry <--\         //
  //   /   \    /         /   \    /         //
  //        ....               ....          //

  // Create single entry edge if the region has multiple entry edges.
  if (!EnteringBB) {
    SmallVector<BasicBlock *, 4> Preds;
    for (BasicBlock *P : predecessors(Entry))
      if (!R->contains(P))
        Preds.push_back(P);

    BasicBlock *NewEntering =
        SplitBlockPredecessors(Entry, Preds, ".region_entering", DT, LI);

    if (RI) {
      // The exit block of predecessing regions must be changed to NewEntering
      for (BasicBlock *ExitPred : predecessors(NewEntering)) {
        Region *RegionOfPred = RI->getRegionFor(ExitPred);
        if (RegionOfPred->getExit() != Entry)
          continue;

        while (!RegionOfPred->isTopLevelRegion() &&
               RegionOfPred->getExit() == Entry) {
          RegionOfPred->replaceExit(NewEntering);
          RegionOfPred = RegionOfPred->getParent();
        }
      }

      // Make all ancestors use EnteringBB as entry; there might be edges to it
      Region *AncestorR = R->getParent();
      RI->setRegionFor(NewEntering, AncestorR);
      while (!AncestorR->isTopLevelRegion() && AncestorR->getEntry() == Entry) {
        AncestorR->replaceEntry(NewEntering);
        AncestorR = AncestorR->getParent();
      }
    }

    EnteringBB = NewEntering;
  }
  assert(R->getEnteringBlock() == EnteringBB);

  // After:
  //
  //    \    /       //
  //  EnteringBB     //
  //      |          //
  //      |          //
  //    Entry <--\   //
  //    /   \    /   //
  //         ....    //
}

// Ensure that the region has a single block that branches to the exit node.
static void simplifyRegionExit(Region *R, DominatorTree *DT, LoopInfo *LI,
                               RegionInfo *RI) {
  BasicBlock *ExitBB = R->getExit();
  BasicBlock *ExitingBB = R->getExitingBlock();

  // Before:
  //
  //   (Region)   ______/  //
  //      \  |   /         //
  //       ExitBB          //
  //       /    \          //

  if (!ExitingBB) {
    SmallVector<BasicBlock *, 4> Preds;
    for (BasicBlock *P : predecessors(ExitBB))
      if (R->contains(P))
        Preds.push_back(P);

    //  Preds[0] Preds[1]      otherBB //
    //         \  |  ________/         //
    //          \ | /                  //
    //           BB                    //
    ExitingBB =
        SplitBlockPredecessors(ExitBB, Preds, ".region_exiting", DT, LI);
    // Preds[0] Preds[1]      otherBB  //
    //        \  /           /         //
    // BB.region_exiting    /          //
    //                  \  /           //
    //                   BB            //

    if (RI)
      RI->setRegionFor(ExitingBB, R);

    // Change the exit of nested regions, but not the region itself,
    R->replaceExitRecursive(ExitingBB);
    R->replaceExit(ExitBB);
  }
  assert(ExitingBB == R->getExitingBlock());

  // After:
  //
  //     \   /                //
  //    ExitingBB     _____/  //
  //          \      /        //
  //           ExitBB         //
  //           /    \         //
}

void polly::simplifyRegion(Region *R, DominatorTree *DT, LoopInfo *LI,
                           RegionInfo *RI) {
  assert(R && !R->isTopLevelRegion());
  assert(!RI || RI == R->getRegionInfo());
  assert((!RI || DT) &&
         "RegionInfo requires DominatorTree to be updated as well");

  simplifyRegionEntry(R, DT, LI, RI);
  simplifyRegionExit(R, DT, LI, RI);
  assert(R->isSimple());
}

// Split the block into two successive blocks.
//
// Like llvm::SplitBlock, but also preserves RegionInfo
static BasicBlock *splitBlock(BasicBlock *Old, Instruction *SplitPt,
                              DominatorTree *DT, llvm::LoopInfo *LI,
                              RegionInfo *RI) {
  assert(Old && SplitPt);

  // Before:
  //
  //  \   /  //
  //   Old   //
  //  /   \  //

  BasicBlock *NewBlock = llvm::SplitBlock(Old, SplitPt, DT, LI);

  if (RI) {
    Region *R = RI->getRegionFor(Old);
    RI->setRegionFor(NewBlock, R);
  }

  // After:
  //
  //   \   /    //
  //    Old     //
  //     |      //
  //  NewBlock  //
  //   /   \    //

  return NewBlock;
}

void polly::splitEntryBlockForAlloca(BasicBlock *EntryBlock, Pass *P) {
  // Find first non-alloca instruction. Every basic block has a non-alloc
  // instruction, as every well formed basic block has a terminator.
  BasicBlock::iterator I = EntryBlock->begin();
  while (isa<AllocaInst>(I))
    ++I;

  auto *DTWP = P->getAnalysisIfAvailable<DominatorTreeWrapperPass>();
  auto *DT = DTWP ? &DTWP->getDomTree() : nullptr;
  auto *LIWP = P->getAnalysisIfAvailable<LoopInfoWrapperPass>();
  auto *LI = LIWP ? &LIWP->getLoopInfo() : nullptr;
  RegionInfoPass *RIP = P->getAnalysisIfAvailable<RegionInfoPass>();
  RegionInfo *RI = RIP ? &RIP->getRegionInfo() : nullptr;

  // splitBlock updates DT, LI and RI.
  splitBlock(EntryBlock, &*I, DT, LI, RI);
}

/// The SCEVExpander will __not__ generate any code for an existing SDiv/SRem
/// instruction but just use it, if it is referenced as a SCEVUnknown. We want
/// however to generate new code if the instruction is in the analyzed region
/// and we generate code outside/in front of that region. Hence, we generate the
/// code for the SDiv/SRem operands in front of the analyzed region and then
/// create a new SDiv/SRem operation there too.
struct ScopExpander : SCEVVisitor<ScopExpander, const SCEV *> {
  friend struct SCEVVisitor<ScopExpander, const SCEV *>;

  explicit ScopExpander(const Region &R, ScalarEvolution &SE,
                        const DataLayout &DL, const char *Name, ValueMapT *VMap)
      : Expander(SCEVExpander(SE, DL, Name)), SE(SE), Name(Name), R(R),
        VMap(VMap) {}

  Value *expandCodeFor(const SCEV *E, Type *Ty, Instruction *I) {
    // If we generate code in the region we will immediately fall back to the
    // SCEVExpander, otherwise we will stop at all unknowns in the SCEV and if
    // needed replace them by copies computed in the entering block.
    if (!R.contains(I))
      E = visit(E);
    return Expander.expandCodeFor(E, Ty, I);
  }

private:
  SCEVExpander Expander;
  ScalarEvolution &SE;
  const char *Name;
  const Region &R;
  ValueMapT *VMap;

  const SCEV *visitUnknown(const SCEVUnknown *E) {

    // If a value mapping was given try if the underlying value is remapped.
    if (VMap)
      if (Value *NewVal = VMap->lookup(E->getValue()))
        if (NewVal != E->getValue())
          return visit(SE.getSCEV(NewVal));

    Instruction *Inst = dyn_cast<Instruction>(E->getValue());
    if (!Inst || (Inst->getOpcode() != Instruction::SRem &&
                  Inst->getOpcode() != Instruction::SDiv))
      return E;

    if (!R.contains(Inst))
      return E;

    Instruction *StartIP = R.getEnteringBlock()->getTerminator();

    const SCEV *LHSScev = visit(SE.getSCEV(Inst->getOperand(0)));
    const SCEV *RHSScev = visit(SE.getSCEV(Inst->getOperand(1)));

    Value *LHS = Expander.expandCodeFor(LHSScev, E->getType(), StartIP);
    Value *RHS = Expander.expandCodeFor(RHSScev, E->getType(), StartIP);

    Inst = BinaryOperator::Create((Instruction::BinaryOps)Inst->getOpcode(),
                                  LHS, RHS, Inst->getName() + Name, StartIP);
    return SE.getSCEV(Inst);
  }

  /// The following functions will just traverse the SCEV and rebuild it with
  /// the new operands returned by the traversal.
  ///
  ///{
  const SCEV *visitConstant(const SCEVConstant *E) { return E; }
  const SCEV *visitTruncateExpr(const SCEVTruncateExpr *E) {
    return SE.getTruncateExpr(visit(E->getOperand()), E->getType());
  }
  const SCEV *visitZeroExtendExpr(const SCEVZeroExtendExpr *E) {
    return SE.getZeroExtendExpr(visit(E->getOperand()), E->getType());
  }
  const SCEV *visitSignExtendExpr(const SCEVSignExtendExpr *E) {
    return SE.getSignExtendExpr(visit(E->getOperand()), E->getType());
  }
  const SCEV *visitUDivExpr(const SCEVUDivExpr *E) {
    return SE.getUDivExpr(visit(E->getLHS()), visit(E->getRHS()));
  }
  const SCEV *visitAddExpr(const SCEVAddExpr *E) {
    SmallVector<const SCEV *, 4> NewOps;
    for (const SCEV *Op : E->operands())
      NewOps.push_back(visit(Op));
    return SE.getAddExpr(NewOps);
  }
  const SCEV *visitMulExpr(const SCEVMulExpr *E) {
    SmallVector<const SCEV *, 4> NewOps;
    for (const SCEV *Op : E->operands())
      NewOps.push_back(visit(Op));
    return SE.getMulExpr(NewOps);
  }
  const SCEV *visitUMaxExpr(const SCEVUMaxExpr *E) {
    SmallVector<const SCEV *, 4> NewOps;
    for (const SCEV *Op : E->operands())
      NewOps.push_back(visit(Op));
    return SE.getUMaxExpr(NewOps);
  }
  const SCEV *visitSMaxExpr(const SCEVSMaxExpr *E) {
    SmallVector<const SCEV *, 4> NewOps;
    for (const SCEV *Op : E->operands())
      NewOps.push_back(visit(Op));
    return SE.getSMaxExpr(NewOps);
  }
  const SCEV *visitAddRecExpr(const SCEVAddRecExpr *E) {
    SmallVector<const SCEV *, 4> NewOps;
    for (const SCEV *Op : E->operands())
      NewOps.push_back(visit(Op));
    return SE.getAddRecExpr(NewOps, E->getLoop(), E->getNoWrapFlags());
  }
  ///}
};

Value *polly::expandCodeFor(Scop &S, ScalarEvolution &SE, const DataLayout &DL,
                            const char *Name, const SCEV *E, Type *Ty,
                            Instruction *IP, ValueMapT *VMap) {
  ScopExpander Expander(S.getRegion(), SE, DL, Name, VMap);
  return Expander.expandCodeFor(E, Ty, IP);
}

bool polly::isErrorBlock(BasicBlock &BB, const Region &R, LoopInfo &LI,
                         const DominatorTree &DT) {

  if (isa<UnreachableInst>(BB.getTerminator()))
    return true;

  if (LI.isLoopHeader(&BB))
    return false;

  // Basic blocks that are always executed are not considered error blocks,
  // as their execution can not be a rare event.
  bool DominatesAllPredecessors = true;
  for (auto Pred : predecessors(R.getExit()))
    if (R.contains(Pred) && !DT.dominates(&BB, Pred))
      DominatesAllPredecessors = false;

  if (DominatesAllPredecessors)
    return false;

  // FIXME: This is a simple heuristic to determine if the load is executed
  //        in a conditional. However, we actually would need the control
  //        condition, i.e., the post dominance frontier. Alternatively we
  //        could walk up the dominance tree until we find a block that is
  //        not post dominated by the load and check if it is a conditional
  //        or a loop header.
  auto *DTNode = DT.getNode(&BB);
  auto *IDomBB = DTNode->getIDom()->getBlock();
  if (LI.isLoopHeader(IDomBB))
    return false;

  for (Instruction &Inst : BB)
    if (CallInst *CI = dyn_cast<CallInst>(&Inst)) {
      if (isIgnoredIntrinsic(CI))
        return false;

      if (!CI->doesNotAccessMemory())
        return true;
      if (CI->doesNotReturn())
        return true;
    }

  return false;
}

Value *polly::getConditionFromTerminator(TerminatorInst *TI) {
  if (BranchInst *BR = dyn_cast<BranchInst>(TI)) {
    if (BR->isUnconditional())
      return ConstantInt::getTrue(Type::getInt1Ty(TI->getContext()));

    return BR->getCondition();
  }

  if (SwitchInst *SI = dyn_cast<SwitchInst>(TI))
    return SI->getCondition();

  return nullptr;
}

bool polly::isHoistableLoad(LoadInst *LInst, Region &R, LoopInfo &LI,
                            ScalarEvolution &SE) {
  Loop *L = LI.getLoopFor(LInst->getParent());
  const SCEV *PtrSCEV = SE.getSCEVAtScope(LInst->getPointerOperand(), L);
  while (L && R.contains(L)) {
    if (!SE.isLoopInvariant(PtrSCEV, L))
      return false;
    L = L->getParentLoop();
  }

  return true;
}

bool polly::isIgnoredIntrinsic(const Value *V) {
  if (auto *IT = dyn_cast<IntrinsicInst>(V)) {
    switch (IT->getIntrinsicID()) {
    // Lifetime markers are supported/ignored.
    case llvm::Intrinsic::lifetime_start:
    case llvm::Intrinsic::lifetime_end:
    // Invariant markers are supported/ignored.
    case llvm::Intrinsic::invariant_start:
    case llvm::Intrinsic::invariant_end:
    // Some misc annotations are supported/ignored.
    case llvm::Intrinsic::var_annotation:
    case llvm::Intrinsic::ptr_annotation:
    case llvm::Intrinsic::annotation:
    case llvm::Intrinsic::donothing:
    case llvm::Intrinsic::assume:
    case llvm::Intrinsic::expect:
    // Some debug info intrisics are supported/ignored.
    case llvm::Intrinsic::dbg_value:
    case llvm::Intrinsic::dbg_declare:
      return true;
    default:
      break;
    }
  }
  return false;
}

bool polly::canSynthesize(const Value *V, const llvm::LoopInfo *LI,
                          ScalarEvolution *SE, const Region *R) {
  if (!V || !SE->isSCEVable(V->getType()))
    return false;

  if (const SCEV *Scev = SE->getSCEV(const_cast<Value *>(V)))
    if (!isa<SCEVCouldNotCompute>(Scev))
      if (!hasScalarDepsInsideRegion(Scev, R))
        return true;

  return false;
}
