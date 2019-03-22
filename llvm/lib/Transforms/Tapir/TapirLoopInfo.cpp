//===- TapirLoopInfo.h - Utility functions for Tapir loops -----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements utility functions for handling Tapir loops.
//
// Many of these routines are adapted from
// Transforms/Vectorize/LoopVectorize.cpp.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/Transforms/Tapir/LoweringUtils.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

#define DEBUG_TYPE "tapir"

/// Update information on this Tapir loop based on its metadata.
void TapirLoopInfo::readTapirLoopMetadata(OptimizationRemarkEmitter &ORE) {
  TapirLoopHints Hints(getLoop());

  // Get a grainsize for this Tapir loop from the metadata, if the metadata
  // gives a grainsize.
  Grainsize = Hints.getGrainsize();
}

static Type *convertPointerToIntegerType(const DataLayout &DL, Type *Ty) {
  if (Ty->isPointerTy())
    return DL.getIntPtrType(Ty);

  // It is possible that char's or short's overflow when we ask for the loop's
  // trip count, work around this by changing the type size.
  if (Ty->getScalarSizeInBits() < 32)
    return Type::getInt32Ty(Ty->getContext());

  return Ty;
}

static Type *getWiderType(const DataLayout &DL, Type *Ty0, Type *Ty1) {
  Ty0 = convertPointerToIntegerType(DL, Ty0);
  Ty1 = convertPointerToIntegerType(DL, Ty1);
  if (Ty0->getScalarSizeInBits() > Ty1->getScalarSizeInBits())
    return Ty0;
  return Ty1;
}

/// Adds \p Phi, with induction descriptor ID, to the inductions list.  This can
/// set \p Phi as the main induction of the loop if \p Phi is a better choice
/// for the main induction than the existing one.
void TapirLoopInfo::addInductionPhi(PHINode *Phi,
                                    const InductionDescriptor &ID) {
  Inductions[Phi] = ID;

  Type *PhiTy = Phi->getType();
  const DataLayout &DL = Phi->getModule()->getDataLayout();

  // Get the widest type.
  if (!PhiTy->isFloatingPointTy()) {
    if (!WidestIndTy)
      WidestIndTy = convertPointerToIntegerType(DL, PhiTy);
    else
      WidestIndTy = getWiderType(DL, PhiTy, WidestIndTy);
  }

  // Int inductions are special because we only allow one IV.
  if (ID.getKind() == InductionDescriptor::IK_IntInduction &&
      ID.getConstIntStepValue() && ID.getConstIntStepValue()->isOne() &&
      isa<Constant>(ID.getStartValue()) &&
      cast<Constant>(ID.getStartValue())->isNullValue()) {

    // Use the phi node with the widest type as induction. Use the last
    // one if there are multiple (no good reason for doing this other
    // than it is expedient). We've checked that it begins at zero and
    // steps by one, so this is a canonical induction variable.
    if (!PrimaryInduction || PhiTy == WidestIndTy)
      PrimaryInduction = Phi;
  }

  // // Both the PHI node itself, and the "post-increment" value feeding
  // // back into the PHI node may have external users.
  // // We can allow those uses, except if the SCEVs we have for them rely
  // // on predicates that only hold within the loop, since allowing the exit
  // // currently means re-using this SCEV outside the loop.
  // if (PSE.getUnionPredicate().isAlwaysTrue()) {
  //   AllowedExit.insert(Phi);
  //   AllowedExit.insert(Phi->getIncomingValueForBlock(TheLoop->getLoopLatch()));
  // }

  LLVM_DEBUG(dbgs() << "TapirLoop: Found an induction variable.\n");
}

/// Create an analysis remark that explains why vectorization failed
///
/// \p RemarkName is the identifier for the remark.  If \p I is passed it is an
/// instruction that prevents vectorization.  Otherwise \p TheLoop is used for
/// the location of the remark.  \return the remark object that can be streamed
/// to.
static OptimizationRemarkAnalysis
createMissedAnalysis(StringRef RemarkName, const Loop *TheLoop,
                     Instruction *I = nullptr) {
  const Value *CodeRegion = TheLoop->getHeader();
  DebugLoc DL = TheLoop->getStartLoc();

  if (I) {
    CodeRegion = I->getParent();
    // If there is no debug location attached to the instruction, revert back to
    // using the loop's.
    if (I->getDebugLoc())
      DL = I->getDebugLoc();
  }

  OptimizationRemarkAnalysis R(DEBUG_TYPE, RemarkName, DL, CodeRegion);
  R << "Tapir loop not transformed: ";
  return R;
}

/// Gather all induction variables in this loop that need special handling
/// during outlining.
bool TapirLoopInfo::collectIVs(PredicatedScalarEvolution &PSE,
                               OptimizationRemarkEmitter &ORE) {
  Loop *L = getLoop();
  for (Instruction &I : *TheLoop->getHeader()) {
    if (auto *Phi = dyn_cast<PHINode>(&I)) {
      Type *PhiTy = Phi->getType();
      // Check that this PHI type is allowed.
      if (!PhiTy->isIntegerTy() && !PhiTy->isFloatingPointTy() &&
          !PhiTy->isPointerTy()) {
        ORE.emit(createMissedAnalysis("CFGNotUnderstood", L, Phi)
                 << "loop control flow is not understood by loop spawning");
        LLVM_DEBUG(dbgs() << "TapirLoop: Found an non-int non-pointer PHI.\n");
        return false;
      }

      // We only allow if-converted PHIs with exactly two incoming values.
      if (Phi->getNumIncomingValues() != 2) {
        ORE.emit(createMissedAnalysis("CFGNotUnderstood", L, Phi)
                 << "loop control flow is not understood by loop spawning");
        LLVM_DEBUG(dbgs() << "TapirLoop: Found an invalid PHI.\n");
        return false;
      }

      InductionDescriptor ID;
      if (InductionDescriptor::isInductionPHI(Phi, L, PSE, ID)) {
        LLVM_DEBUG(dbgs() << "\tFound induction PHI " << *Phi << "\n");
        addInductionPhi(Phi, ID);
        // if (ID.hasUnsafeAlgebra() && !HasFunNoNaNAttr)
        //   Requirements->addUnsafeAlgebraInst(ID.getUnsafeAlgebraInst());
        continue;
      }

      // As a last resort, coerce the PHI to a AddRec expression and re-try
      // classifying it a an induction PHI.
      if (InductionDescriptor::isInductionPHI(Phi, L, PSE, ID, true)) {
        LLVM_DEBUG(dbgs() << "\tCoerced induction PHI " << *Phi << "\n");
        addInductionPhi(Phi, ID);
        continue;
      }

      LLVM_DEBUG(dbgs() << "\tPassed PHI " << *Phi << "\n");
    } // end of PHI handling
  }

  if (!PrimaryInduction) {
    LLVM_DEBUG(dbgs()
               << "TapirLoop: Did not find one integer induction var.\n");
    if (Inductions.empty()) {
      ORE.emit(createMissedAnalysis("NoInductionVariable", L)
               << "loop induction variable could not be identified");
      return false;
    }
  }

  // Now we know the widest induction type, check if our found induction is the
  // same size.
  //
  // TODO: Check if this code is dead due to IndVarSimplify.
  if (PrimaryInduction && WidestIndTy != PrimaryInduction->getType())
    PrimaryInduction = nullptr;

  return true;
}

/// Replace all induction variables in this loop that are not primary with
/// stronger forms.
void TapirLoopInfo::replaceNonPrimaryIVs(PredicatedScalarEvolution &PSE) {
  BasicBlock *Header = getLoop()->getHeader();
  IRBuilder<> B(&*Header->getFirstInsertionPt());
  const DataLayout &DL = Header->getModule()->getDataLayout();
  SmallVector<std::pair<PHINode *, InductionDescriptor>, 4> InductionsToRemove;

  // Replace all non-primary inductions with strengthened forms.
  for (auto &InductionEntry : Inductions) {
    PHINode *OrigPhi = InductionEntry.first;
    InductionDescriptor II = InductionEntry.second;
    if (OrigPhi == PrimaryInduction) continue;
    LLVM_DEBUG(dbgs() << "Replacing Phi " << *OrigPhi << "\n");
    // If Induction is not canonical, replace it with some computation based on
    // PrimaryInduction.
    Type *StepType = II.getStep()->getType();
    Instruction::CastOps CastOp =
      CastInst::getCastOpcode(PrimaryInduction, true, StepType, true);
    Value *CRD = B.CreateCast(CastOp, PrimaryInduction, StepType, "cast.crd");
    Value *PhiRepl = II.transform(B, CRD, PSE.getSE(), DL);
    PhiRepl->setName(OrigPhi->getName() + ".tl.repl");
    OrigPhi->replaceAllUsesWith(PhiRepl);
    InductionsToRemove.push_back(InductionEntry);
  }

  // Remove all inductions that were replaced from Inductions.
  for (auto &InductionEntry : InductionsToRemove) {
    PHINode *OrigPhi = InductionEntry.first;
    OrigPhi->eraseFromParent();
    Inductions.erase(OrigPhi);
  }
}

void TapirLoopInfo::getLoopCondition() {
  // Get the loop condition.
  BranchInst *BI =
    dyn_cast<BranchInst>(getLoop()->getLoopLatch()->getTerminator());
  assert(BI && "Loop latch not terminated by a branch.");
  Condition = dyn_cast<ICmpInst>(BI->getCondition());
  LLVM_DEBUG(dbgs() << "\tLoop condition " << *Condition << "\n");
  assert(Condition && "Condition is not an integer comparison.");
  assert(Condition->isEquality() && "Condition is not an equality comparison.");

  if (Condition->getOperand(0) == PrimaryInduction ||
      Condition->getOperand(1) == PrimaryInduction)
    // The condition examines the primary induction before the increment.
    // Hence, the end iteration is included in the loop bounds.
    InclusiveRange = true;
}

static Value *getEscapeValue(Instruction *UI, const InductionDescriptor &II,
                             Value *TripCount, PredicatedScalarEvolution &PSE,
                             bool PostInc) {
  const DataLayout &DL = UI->getModule()->getDataLayout();
  IRBuilder<> B(&*UI->getParent()->getFirstInsertionPt());
  Value *EffTripCount = TripCount;
  if (!PostInc)
    EffTripCount = B.CreateSub(
        TripCount, ConstantInt::get(TripCount->getType(), 1));

  Value *Count = !II.getStep()->getType()->isIntegerTy()
    ? B.CreateCast(Instruction::SIToFP, EffTripCount,
                   II.getStep()->getType())
    : B.CreateSExtOrTrunc(EffTripCount, II.getStep()->getType());
  if (PostInc)
    Count->setName("cast.count");
  else
    Count->setName("cast.cmo");

  Value *Escape = II.transform(B, Count, PSE.getSE(), DL);
  Escape->setName(UI->getName() + ".escape");
  return Escape;
}

/// Fix up external users of the induction variable.  We assume we are in LCSSA
/// form, with all external PHIs that use the IV having one input value, coming
/// from the remainder loop.  We need those PHIs to also have a correct value
/// for the IV when arriving directly from the middle block.
void TapirLoopInfo::fixupIVUsers(PHINode *OrigPhi, const InductionDescriptor &II,
                                 PredicatedScalarEvolution &PSE) {
  // There are two kinds of external IV usages - those that use the value
  // computed in the last iteration (the PHI) and those that use the penultimate
  // value (the value that feeds into the phi from the loop latch).
  // We allow both, but they, obviously, have different values.
  assert(getExitBlock() && "Expected a single exit block");
  assert(getTripCount() && "Expected valid trip count");
  Loop *L = getLoop();
  Task *T = getTask();
  Value *TripCount = getTripCount();

  DenseMap<Value *, Value *> MissingVals;

  // An external user of the last iteration's value should see the value that
  // the remainder loop uses to initialize its own IV.
  Value *PostInc = OrigPhi->getIncomingValueForBlock(L->getLoopLatch());
  for (User *U : PostInc->users()) {
    Instruction *UI = cast<Instruction>(U);
    if (!L->contains(UI) && !T->encloses(UI->getParent())) {
      assert(isa<PHINode>(UI) && "Expected LCSSA form");
      MissingVals[UI] = getEscapeValue(UI, II, TripCount, PSE, true);
    }
  }

  // An external user of the penultimate value needs to see TripCount - Step.
  // The simplest way to get this is to recompute it from the constituent SCEVs,
  // that is Start + (Step * (TripCount - 1)).
  for (User *U : OrigPhi->users()) {
    Instruction *UI = cast<Instruction>(U);
    if (!L->contains(UI) && !T->encloses(UI->getParent())) {
      assert(isa<PHINode>(UI) && "Expected LCSSA form");
      MissingVals[UI] = getEscapeValue(UI, II, TripCount, PSE, false);
    }
  }

  for (auto &I : MissingVals) {
    LLVM_DEBUG(dbgs() << "Replacing external IV use:" << *I.first << " with "
               << *I.second << "\n");
    PHINode *PHI = cast<PHINode>(I.first);
    PHI->replaceAllUsesWith(I.second);
    PHI->eraseFromParent();
  }
}

const SCEV *TapirLoopInfo::getBackedgeTakenCount(
    PredicatedScalarEvolution &PSE) const {
  Loop *L = getLoop();
  ScalarEvolution *SE = PSE.getSE();
  const SCEV *BackedgeTakenCount = PSE.getBackedgeTakenCount();
  if (BackedgeTakenCount == SE->getCouldNotCompute())
    BackedgeTakenCount = SE->getExitCount(L, L->getLoopLatch());

  if (BackedgeTakenCount == SE->getCouldNotCompute())
    return BackedgeTakenCount;

  Type *IdxTy = getWidestInductionType();

  // The exit count might have the type of i64 while the phi is i32. This can
  // happen if we have an induction variable that is sign extended before the
  // compare. The only way that we get a backedge taken count is that the
  // induction variable was signed and as such will not overflow. In such a case
  // truncation is legal.
  if (BackedgeTakenCount->getType()->getPrimitiveSizeInBits() >
      IdxTy->getPrimitiveSizeInBits())
    BackedgeTakenCount = SE->getTruncateOrNoop(BackedgeTakenCount, IdxTy);
  BackedgeTakenCount = SE->getNoopOrZeroExtend(BackedgeTakenCount, IdxTy);

  return BackedgeTakenCount;
}

const SCEV *TapirLoopInfo::getExitCount(const SCEV *BackedgeTakenCount,
                                        PredicatedScalarEvolution &PSE) const {
  ScalarEvolution *SE = PSE.getSE();
  const SCEV *ExitCount;
  if (InclusiveRange)
    ExitCount = BackedgeTakenCount;
  else
    // Get the total trip count from the count by adding 1.
    ExitCount = SE->getAddExpr(
        BackedgeTakenCount, SE->getOne(BackedgeTakenCount->getType()));
  return ExitCount;
}

/// Returns (and creates if needed) the original loop trip count.
Value *TapirLoopInfo::getOrCreateTripCount(PredicatedScalarEvolution &PSE) {
  if (TripCount)
    return TripCount;
  Loop *L = getLoop();

  // Get the existing SSA value being used for the end condition of the loop.
  if (!Condition)
    getLoopCondition();

  Value *ConditionEnd = Condition->getOperand(0);
  {
    Value *PrimaryIVInc =
      PrimaryInduction->getIncomingValueForBlock(Condition->getParent());
    if (ConditionEnd == (InclusiveRange ? PrimaryInduction : PrimaryIVInc))
      ConditionEnd = Condition->getOperand(1);
  }

  IRBuilder<> Builder(L->getLoopPreheader()->getTerminator());
  ScalarEvolution *SE = PSE.getSE();

  // Find the loop boundaries.
  const SCEV *BackedgeTakenCount = getBackedgeTakenCount(PSE);

  if (BackedgeTakenCount == SE->getCouldNotCompute())
    return nullptr;
  // assert(BackedgeTakenCount != SE->getCouldNotCompute() &&
  //        "Invalid loop count");

  const SCEV *ExitCount = getExitCount(BackedgeTakenCount, PSE);

  Type *IdxTy = getWidestInductionType();

  const DataLayout &DL = L->getHeader()->getModule()->getDataLayout();

  if (ExitCount == SE->getSCEV(ConditionEnd)) {
    TripCount = ConditionEnd;
    return TripCount;
  }

  // Expand the trip count and place the new instructions in the preheader.
  // Notice that the pre-header does not change, only the loop body.
  SCEVExpander Exp(*SE, DL, "induction");

  // Count holds the overall loop count (N).
  TripCount = Exp.expandCodeFor(ExitCount, ExitCount->getType(),
                                L->getLoopPreheader()->getTerminator());

  if (TripCount->getType()->isPointerTy())
    TripCount =
        CastInst::CreatePointerCast(TripCount, IdxTy, "exitcount.ptrcnt.to.int",
                                    L->getLoopPreheader()->getTerminator());

  // Try to use the existing ConditionEnd for the trip count.
  if (TripCount != ConditionEnd)
    if (PSE.areAddRecsEqualWithPreds(PSE.getAsAddRec(TripCount),
                                     PSE.getAsAddRec(ConditionEnd)))
      TripCount = ConditionEnd;

  return TripCount;
}

/// Top-level call to prepare a Tapir loop for outlining.
bool TapirLoopInfo::prepareForOutlining(
    DominatorTree &DT, LoopInfo &LI, TaskInfo &TI,
    PredicatedScalarEvolution &PSE, AssumptionCache &AC,
    OptimizationRemarkEmitter &ORE, const TargetTransformInfo &TTI) {
  Loop *L = getLoop();
  LLVM_DEBUG(dbgs() << "LS processing loop " << *L << "\n");

  // Collect the IVs in this loop.
  collectIVs(PSE, ORE);

  // If no primary induction was found, just bail.
  if (!PrimaryInduction)
    return false;

  LLVM_DEBUG(dbgs() << "\tPrimary induction " << *PrimaryInduction << "\n");

  // Replace any non-primary IV's.
  replaceNonPrimaryIVs(PSE);

  // Compute the trip count for this loop.
  //
  // We need the trip count for two reasons.
  //
  // 1) In the call to the helper that will replace this loop, we need to pass
  // the total number of loop iterations.
  //
  // 2) In the helper itself, the strip-mined loop must iterate to the
  // end-iteration argument, not the total number of iterations.
  Value *TripCount = getOrCreateTripCount(PSE);
  if (!TripCount)
    return false;

  LLVM_DEBUG(dbgs() << "\tTrip count " << *TripCount << "\n");

  // FIXME: This test is probably too simple.
  assert(((Condition->getOperand(0) == TripCount) ||
          (Condition->getOperand(1) == TripCount)) &&
         "Condition does not use trip count.");

  // Fixup all external uses of the IVs.
  for (auto &InductionEntry : Inductions)
    fixupIVUsers(InductionEntry.first, InductionEntry.second, PSE);

  return true;
}
