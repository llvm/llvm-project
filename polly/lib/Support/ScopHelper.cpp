//===- ScopHelper.cpp - Some Helper Functions for Scop.  ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

using namespace llvm;
using namespace polly;

#define DEBUG_TYPE "polly-scop-helper"

static cl::opt<bool> PollyAllowErrorBlocks(
    "polly-allow-error-blocks",
    cl::desc("Allow to speculate on the execution of 'error blocks'."),
    cl::Hidden, cl::init(true), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::list<std::string> DebugFunctions(
    "polly-debug-func",
    cl::desc("Allow calls to the specified functions in SCoPs even if their "
             "side-effects are unknown. This can be used to do debug output in "
             "Polly-transformed code."),
    cl::Hidden, cl::ZeroOrMore, cl::CommaSeparated, cl::cat(PollyCategory));

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

void polly::recordAssumption(polly::RecordedAssumptionsTy *RecordedAssumptions,
                             polly::AssumptionKind Kind, isl::set Set,
                             DebugLoc Loc, polly::AssumptionSign Sign,
                             BasicBlock *BB) {
  assert((Set.is_params() || BB) &&
         "Assumptions without a basic block must be parameter sets");
  if (RecordedAssumptions)
    RecordedAssumptions->push_back({Kind, Sign, Set, Loc, BB});
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
      : Expander(SE, DL, Name, /*PreserveLCSSA=*/false), SE(SE), Name(Name),
        R(R), VMap(VMap), RTCBB(RTCBB) {}

  Value *expandCodeFor(const SCEV *E, Type *Ty, Instruction *I) {
    // If we generate code in the region we will immediately fall back to the
    // SCEVExpander, otherwise we will stop at all unknowns in the SCEV and if
    // needed replace them by copies computed in the entering block.
    if (!R.contains(I))
      E = visit(E);
    return Expander.expandCodeFor(E, Ty, I);
  }

  const SCEV *visit(const SCEV *E) {
    // Cache the expansion results for intermediate SCEV expressions. A SCEV
    // expression can refer to an operand multiple times (e.g. "x*x), so
    // a naive visitor takes exponential time.
    if (SCEVCache.count(E))
      return SCEVCache[E];
    const SCEV *Result = SCEVVisitor::visit(E);
    SCEVCache[E] = Result;
    return Result;
  }

private:
  SCEVExpander Expander;
  ScalarEvolution &SE;
  const char *Name;
  const Region &R;
  ValueMapT *VMap;
  BasicBlock *RTCBB;
  DenseMap<const SCEV *, const SCEV *> SCEVCache;

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
  const SCEV *visitUMinExpr(const SCEVUMinExpr *E) {
    SmallVector<const SCEV *, 4> NewOps;
    for (const SCEV *Op : E->operands())
      NewOps.push_back(visit(Op));
    return SE.getUMinExpr(NewOps);
  }
  const SCEV *visitSMinExpr(const SCEVSMinExpr *E) {
    SmallVector<const SCEV *, 4> NewOps;
    for (const SCEV *Op : E->operands())
      NewOps.push_back(visit(Op));
    return SE.getSMinExpr(NewOps);
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

  for (Instruction &Inst : BB)
    if (CallInst *CI = dyn_cast<CallInst>(&Inst)) {
      if (isDebugCall(CI))
        continue;

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

Value *polly::getConditionFromTerminator(Instruction *TI) {
  if (BranchInst *BR = dyn_cast<BranchInst>(TI)) {
    if (BR->isUnconditional())
      return ConstantInt::getTrue(Type::getInt1Ty(TI->getContext()));

    return BR->getCondition();
  }

  if (SwitchInst *SI = dyn_cast<SwitchInst>(TI))
    return SI->getCondition();

  return nullptr;
}

Loop *polly::getLoopSurroundingScop(Scop &S, LoopInfo &LI) {
  // Start with the smallest loop containing the entry and expand that
  // loop until it contains all blocks in the region. If there is a loop
  // containing all blocks in the region check if it is itself contained
  // and if so take the parent loop as it will be the smallest containing
  // the region but not contained by it.
  Loop *L = LI.getLoopFor(S.getEntry());
  while (L) {
    bool AllContained = true;
    for (auto *BB : S.blocks())
      AllContained &= L->contains(BB);
    if (AllContained)
      break;
    L = L->getParentLoop();
  }

  return L ? (S.contains(L) ? L->getParentLoop() : L) : nullptr;
}

unsigned polly::getNumBlocksInLoop(Loop *L) {
  unsigned NumBlocks = L->getNumBlocks();
  SmallVector<BasicBlock *, 4> ExitBlocks;
  L->getExitBlocks(ExitBlocks);

  for (auto ExitBlock : ExitBlocks) {
    if (isa<UnreachableInst>(ExitBlock->getTerminator()))
      NumBlocks++;
  }
  return NumBlocks;
}

unsigned polly::getNumBlocksInRegionNode(RegionNode *RN) {
  if (!RN->isSubRegion())
    return 1;

  Region *R = RN->getNodeAs<Region>();
  return std::distance(R->block_begin(), R->block_end());
}

Loop *polly::getRegionNodeLoop(RegionNode *RN, LoopInfo &LI) {
  if (!RN->isSubRegion()) {
    BasicBlock *BB = RN->getNodeAs<BasicBlock>();
    Loop *L = LI.getLoopFor(BB);

    // Unreachable statements are not considered to belong to a LLVM loop, as
    // they are not part of an actual loop in the control flow graph.
    // Nevertheless, we handle certain unreachable statements that are common
    // when modeling run-time bounds checks as being part of the loop to be
    // able to model them and to later eliminate the run-time bounds checks.
    //
    // Specifically, for basic blocks that terminate in an unreachable and
    // where the immediate predecessor is part of a loop, we assume these
    // basic blocks belong to the loop the predecessor belongs to. This
    // allows us to model the following code.
    //
    // for (i = 0; i < N; i++) {
    //   if (i > 1024)
    //     abort();            <- this abort might be translated to an
    //                            unreachable
    //
    //   A[i] = ...
    // }
    if (!L && isa<UnreachableInst>(BB->getTerminator()) && BB->getPrevNode())
      L = LI.getLoopFor(BB->getPrevNode());
    return L;
  }

  Region *NonAffineSubRegion = RN->getNodeAs<Region>();
  Loop *L = LI.getLoopFor(NonAffineSubRegion->getEntry());
  while (L && NonAffineSubRegion->contains(L))
    L = L->getParentLoop();
  return L;
}

static bool hasVariantIndex(GetElementPtrInst *Gep, Loop *L, Region &R,
                            ScalarEvolution &SE) {
  for (const Use &Val : llvm::drop_begin(Gep->operands(), 1)) {
    const SCEV *PtrSCEV = SE.getSCEVAtScope(Val, L);
    Loop *OuterLoop = R.outermostLoopInRegion(L);
    if (!SE.isLoopInvariant(PtrSCEV, OuterLoop))
      return true;
  }
  return false;
}

bool polly::isHoistableLoad(LoadInst *LInst, Region &R, LoopInfo &LI,
                            ScalarEvolution &SE, const DominatorTree &DT,
                            const InvariantLoadsSetTy &KnownInvariantLoads) {
  Loop *L = LI.getLoopFor(LInst->getParent());
  auto *Ptr = LInst->getPointerOperand();

  // A LoadInst is hoistable if the address it is loading from is also
  // invariant; in this case: another invariant load (whether that address
  // is also not written to has to be checked separately)
  // TODO: This only checks for a LoadInst->GetElementPtrInst->LoadInst
  // pattern generated by the Chapel frontend, but generally this applies
  // for any chain of instruction that does not also depend on any
  // induction variable
  if (auto *GepInst = dyn_cast<GetElementPtrInst>(Ptr)) {
    if (!hasVariantIndex(GepInst, L, R, SE)) {
      if (auto *DecidingLoad =
              dyn_cast<LoadInst>(GepInst->getPointerOperand())) {
        if (KnownInvariantLoads.count(DecidingLoad))
          return true;
      }
    }
  }

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
    if (R.isTopLevelRegion()) {
      for (BasicBlock &I : *R.getEntry()->getParent())
        if (isa<ReturnInst>(I.getTerminator()) && !DT.dominates(&BB, &I))
          DominatesAllPredecessors = false;
    } else {
      for (auto Pred : predecessors(R.getExit()))
        if (R.contains(Pred) && !DT.dominates(&BB, Pred))
          DominatesAllPredecessors = false;
    }

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

bool polly::isDebugCall(Instruction *Inst) {
  auto *CI = dyn_cast<CallInst>(Inst);
  if (!CI)
    return false;

  Function *CF = CI->getCalledFunction();
  if (!CF)
    return false;

  return std::find(DebugFunctions.begin(), DebugFunctions.end(),
                   CF->getName()) != DebugFunctions.end();
}

static bool hasDebugCall(BasicBlock *BB) {
  for (Instruction &Inst : *BB) {
    if (isDebugCall(&Inst))
      return true;
  }
  return false;
}

bool polly::hasDebugCall(ScopStmt *Stmt) {
  // Quick skip if no debug functions have been defined.
  if (DebugFunctions.empty())
    return false;

  if (!Stmt)
    return false;

  for (Instruction *Inst : Stmt->getInstructions())
    if (isDebugCall(Inst))
      return true;

  if (Stmt->isRegionStmt()) {
    for (BasicBlock *RBB : Stmt->getRegion()->blocks())
      if (RBB != Stmt->getEntryBlock() && ::hasDebugCall(RBB))
        return true;
  }

  return false;
}
