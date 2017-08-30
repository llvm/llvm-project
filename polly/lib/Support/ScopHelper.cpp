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

static cl::opt<bool> PollyAllowErrorBlocks(
    "polly-allow-error-blocks",
    cl::desc("Allow to speculate on the execution of 'error blocks'."),
    cl::Hidden, cl::init(true), cl::ZeroOrMore, cl::cat(PollyCategory));

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

void polly::splitEntryBlockForAlloca(BasicBlock *EntryBlock, DominatorTree *DT,
                                     LoopInfo *LI, RegionInfo *RI) {
  // Find first non-alloca instruction. Every basic block has a non-alloca
  // instruction, as every well formed basic block has a terminator.
  BasicBlock::iterator I = EntryBlock->begin();
  while (isa<AllocaInst>(I))
    ++I;

  // splitBlock updates DT, LI and RI.
  splitBlock(EntryBlock, &*I, DT, LI, RI);
}

void polly::splitEntryBlockForAlloca(BasicBlock *EntryBlock, Pass *P) {
  auto *DTWP = P->getAnalysisIfAvailable<DominatorTreeWrapperPass>();
  auto *DT = DTWP ? &DTWP->getDomTree() : nullptr;
  auto *LIWP = P->getAnalysisIfAvailable<LoopInfoWrapperPass>();
  auto *LI = LIWP ? &LIWP->getLoopInfo() : nullptr;
  RegionInfoPass *RIP = P->getAnalysisIfAvailable<RegionInfoPass>();
  RegionInfo *RI = RIP ? &RIP->getRegionInfo() : nullptr;

  // splitBlock updates DT, LI and RI.
  polly::splitEntryBlockForAlloca(EntryBlock, DT, LI, RI);
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
                        const DataLayout &DL, const char *Name, ValueMapT *VMap,
                        BasicBlock *RTCBB)
      : Expander(SCEVExpander(SE, DL, Name)), SE(SE), Name(Name), R(R),
        VMap(VMap), RTCBB(RTCBB) {}

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
  BasicBlock *RTCBB;

  const SCEV *visitGenericInst(const SCEVUnknown *E, Instruction *Inst,
                               Instruction *IP) {
    if (!Inst || !R.contains(Inst))
      return E;

    assert(!Inst->mayThrow() && !Inst->mayReadOrWriteMemory() &&
           !isa<PHINode>(Inst));

    auto *InstClone = Inst->clone();
    for (auto &Op : Inst->operands()) {
      assert(SE.isSCEVable(Op->getType()));
      auto *OpSCEV = SE.getSCEV(Op);
      auto *OpClone = expandCodeFor(OpSCEV, Op->getType(), IP);
      InstClone->replaceUsesOfWith(Op, OpClone);
    }

    InstClone->setName(Name + Inst->getName());
    InstClone->insertBefore(IP);
    return SE.getSCEV(InstClone);
  }

  const SCEV *visitUnknown(const SCEVUnknown *E) {

    // If a value mapping was given try if the underlying value is remapped.
    Value *NewVal = VMap ? VMap->lookup(E->getValue()) : nullptr;
    if (NewVal) {
      auto *NewE = SE.getSCEV(NewVal);

      // While the mapped value might be different the SCEV representation might
      // not be. To this end we will check before we go into recursion here.
      if (E != NewE)
        return visit(NewE);
    }

    Instruction *Inst = dyn_cast<Instruction>(E->getValue());
    Instruction *IP;
    if (Inst && !R.contains(Inst))
      IP = Inst;
    else if (Inst && RTCBB->getParent() == Inst->getFunction())
      IP = RTCBB->getTerminator();
    else
      IP = RTCBB->getParent()->getEntryBlock().getTerminator();

    if (!Inst || (Inst->getOpcode() != Instruction::SRem &&
                  Inst->getOpcode() != Instruction::SDiv))
      return visitGenericInst(E, Inst, IP);

    const SCEV *LHSScev = SE.getSCEV(Inst->getOperand(0));
    const SCEV *RHSScev = SE.getSCEV(Inst->getOperand(1));

    if (!SE.isKnownNonZero(RHSScev))
      RHSScev = SE.getUMaxExpr(RHSScev, SE.getConstant(E->getType(), 1));

    Value *LHS = expandCodeFor(LHSScev, E->getType(), IP);
    Value *RHS = expandCodeFor(RHSScev, E->getType(), IP);

    Inst = BinaryOperator::Create((Instruction::BinaryOps)Inst->getOpcode(),
                                  LHS, RHS, Inst->getName() + Name, IP);
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
    auto *RHSScev = visit(E->getRHS());
    if (!SE.isKnownNonZero(RHSScev))
      RHSScev = SE.getUMaxExpr(RHSScev, SE.getConstant(E->getType(), 1));
    return SE.getUDivExpr(visit(E->getLHS()), RHSScev);
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
                            Instruction *IP, ValueMapT *VMap,
                            BasicBlock *RTCBB) {
  ScopExpander Expander(S.getRegion(), SE, DL, Name, VMap, RTCBB);
  return Expander.expandCodeFor(E, Ty, IP);
}

bool polly::isErrorBlock(BasicBlock &BB, const Region &R, LoopInfo &LI,
                         const DominatorTree &DT) {
  if (!PollyAllowErrorBlocks)
    return false;

  if (isa<UnreachableInst>(BB.getTerminator()))
    return true;

  if (LI.isLoopHeader(&BB))
    return false;

  // Basic blocks that are always executed are not considered error blocks,
  // as their execution can not be a rare event.
  bool DominatesAllPredecessors = true;
  if (R.isTopLevelRegion()) {
    for (BasicBlock &I : *R.getEntry()->getParent())
      if (isa<ReturnInst>(I.getTerminator()) && !DT.dominates(&BB, &I))
        DominatesAllPredecessors = false;
  } else {
    for (auto Pred : predecessors(R.getExit()))
      if (R.contains(Pred) && !DT.dominates(&BB, Pred))
        DominatesAllPredecessors = false;
  }

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
        continue;

      // memset, memcpy and memmove are modeled intrinsics.
      if (isa<MemSetInst>(CI) || isa<MemTransferInst>(CI))
        continue;

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
                            ScalarEvolution &SE, const DominatorTree &DT) {
  Loop *L = LI.getLoopFor(LInst->getParent());
  auto *Ptr = LInst->getPointerOperand();
  const SCEV *PtrSCEV = SE.getSCEVAtScope(Ptr, L);
  while (L && R.contains(L)) {
    if (!SE.isLoopInvariant(PtrSCEV, L))
      return false;
    L = L->getParentLoop();
  }

  for (auto *User : Ptr->users()) {
    auto *UserI = dyn_cast<Instruction>(User);
    if (!UserI || !R.contains(UserI))
      continue;
    if (!UserI->mayWriteToMemory())
      continue;

    auto &BB = *UserI->getParent();
    if (DT.dominates(&BB, LInst->getParent()))
      return false;

    bool DominatesAllPredecessors = true;
    for (auto Pred : predecessors(R.getExit()))
      if (R.contains(Pred) && !DT.dominates(&BB, Pred))
        DominatesAllPredecessors = false;

    if (!DominatesAllPredecessors)
      continue;

    return false;
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
    // Some debug info intrinsics are supported/ignored.
    case llvm::Intrinsic::dbg_value:
    case llvm::Intrinsic::dbg_declare:
      return true;
    default:
      break;
    }
  }
  return false;
}

bool polly::canSynthesize(const Value *V, const Scop &S, ScalarEvolution *SE,
                          Loop *Scope) {
  if (!V || !SE->isSCEVable(V->getType()))
    return false;

  const InvariantLoadsSetTy &ILS = S.getRequiredInvariantLoads();
  if (const SCEV *Scev = SE->getSCEVAtScope(const_cast<Value *>(V), Scope))
    if (!isa<SCEVCouldNotCompute>(Scev))
      if (!hasScalarDepsInsideRegion(Scev, &S.getRegion(), Scope, false, ILS))
        return true;

  return false;
}

llvm::BasicBlock *polly::getUseBlock(const llvm::Use &U) {
  Instruction *UI = dyn_cast<Instruction>(U.getUser());
  if (!UI)
    return nullptr;

  if (PHINode *PHI = dyn_cast<PHINode>(UI))
    return PHI->getIncomingBlock(U);

  return UI->getParent();
}

std::tuple<std::vector<const SCEV *>, std::vector<int>>
polly::getIndexExpressionsFromGEP(GetElementPtrInst *GEP, ScalarEvolution &SE) {
  std::vector<const SCEV *> Subscripts;
  std::vector<int> Sizes;

  Type *Ty = GEP->getPointerOperandType();

  bool DroppedFirstDim = false;

  for (unsigned i = 1; i < GEP->getNumOperands(); i++) {

    const SCEV *Expr = SE.getSCEV(GEP->getOperand(i));

    if (i == 1) {
      if (auto *PtrTy = dyn_cast<PointerType>(Ty)) {
        Ty = PtrTy->getElementType();
      } else if (auto *ArrayTy = dyn_cast<ArrayType>(Ty)) {
        Ty = ArrayTy->getElementType();
      } else {
        Subscripts.clear();
        Sizes.clear();
        break;
      }
      if (auto *Const = dyn_cast<SCEVConstant>(Expr))
        if (Const->getValue()->isZero()) {
          DroppedFirstDim = true;
          continue;
        }
      Subscripts.push_back(Expr);
      continue;
    }

    auto *ArrayTy = dyn_cast<ArrayType>(Ty);
    if (!ArrayTy) {
      Subscripts.clear();
      Sizes.clear();
      break;
    }

    Subscripts.push_back(Expr);
    if (!(DroppedFirstDim && i == 2))
      Sizes.push_back(ArrayTy->getNumElements());

    Ty = ArrayTy->getElementType();
  }

  return std::make_tuple(Subscripts, Sizes);
}

llvm::Loop *polly::getFirstNonBoxedLoopFor(llvm::Loop *L, llvm::LoopInfo &LI,
                                           const BoxedLoopsSetTy &BoxedLoops) {
  while (BoxedLoops.count(L))
    L = L->getParentLoop();
  return L;
}

llvm::Loop *polly::getFirstNonBoxedLoopFor(llvm::BasicBlock *BB,
                                           llvm::LoopInfo &LI,
                                           const BoxedLoopsSetTy &BoxedLoops) {
  Loop *L = LI.getLoopFor(BB);
  return getFirstNonBoxedLoopFor(L, LI, BoxedLoops);
}
