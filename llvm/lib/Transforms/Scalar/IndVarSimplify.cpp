//===- IndVarSimplify.cpp - Induction Variable Elimination ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This transformation analyzes and transforms the induction variables (and
// computations derived from them) into simpler forms suitable for subsequent
// analysis and transformation.
//
// This transformation makes the following changes to each loop with an
// identifiable induction variable:
//   1. All loops are transformed to have a SINGLE canonical induction variable
//      which starts at zero and steps by one.
//   2. The canonical induction variable is guaranteed to be the first PHI node
//      in the loop header block.
//   3. The canonical induction variable is guaranteed to be in a wide enough
//      type so that IV expressions need not be (directly) zero-extended or
//      sign-extended.
//   4. Any pointer arithmetic recurrences are raised to use array subscripts.
//
// If the trip count of a loop is computable, this pass also makes the following
// changes:
//   1. The exit condition for the loop is canonicalized to compare the
//      induction value against the exit value.  This turns loops like:
//        'for (i = 7; i*i < 1000; ++i)' into 'for (i = 0; i != 25; ++i)'
//   2. Any use outside of the loop of an expression derived from the indvar
//      is changed to compute the derived value outside of the loop, eliminating
//      the dependence on the exit value of the induction variable.  If the only
//      purpose of the loop is to compute the exit value of some derived
//      expression, this transformation will make the loop dead.
//
// This transformation should be followed by strength reduction after all of the
// desired loop transformations have been performed.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "indvars"
#include "llvm/Transforms/Scalar.h"
#include "llvm/BasicBlock.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/LLVMContext.h"
#include "llvm/Type.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/IVUsers.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
using namespace llvm;

STATISTIC(NumRemoved , "Number of aux indvars removed");
STATISTIC(NumInserted, "Number of canonical indvars added");
STATISTIC(NumReplaced, "Number of exit values replaced");
STATISTIC(NumLFTR    , "Number of loop exit tests replaced");

namespace {
  class VISIBILITY_HIDDEN IndVarSimplify : public LoopPass {
    IVUsers         *IU;
    LoopInfo        *LI;
    ScalarEvolution *SE;
    DominatorTree   *DT;
    bool Changed;
  public:

    static char ID; // Pass identification, replacement for typeid
    IndVarSimplify() : LoopPass(&ID) {}

    virtual bool runOnLoop(Loop *L, LPPassManager &LPM);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<DominatorTree>();
      AU.addRequired<LoopInfo>();
      AU.addRequired<ScalarEvolution>();
      AU.addRequiredID(LoopSimplifyID);
      AU.addRequiredID(LCSSAID);
      AU.addRequired<IVUsers>();
      AU.addPreserved<ScalarEvolution>();
      AU.addPreservedID(LoopSimplifyID);
      AU.addPreservedID(LCSSAID);
      AU.addPreserved<IVUsers>();
      AU.setPreservesCFG();
    }

  private:

    void RewriteNonIntegerIVs(Loop *L);

    ICmpInst *LinearFunctionTestReplace(Loop *L, const SCEV *BackedgeTakenCount,
                                   Value *IndVar,
                                   BasicBlock *ExitingBlock,
                                   BranchInst *BI,
                                   SCEVExpander &Rewriter);
    void RewriteLoopExitValues(Loop *L, const SCEV *BackedgeTakenCount,
                               SCEVExpander &Rewriter);

    void RewriteIVExpressions(Loop *L, const Type *LargestType,
                              SCEVExpander &Rewriter);

    void SinkUnusedInvariants(Loop *L);

    void HandleFloatingPointIV(Loop *L, PHINode *PH);
  };
}

char IndVarSimplify::ID = 0;
static RegisterPass<IndVarSimplify>
X("indvars", "Canonicalize Induction Variables");

Pass *llvm::createIndVarSimplifyPass() {
  return new IndVarSimplify();
}

/// LinearFunctionTestReplace - This method rewrites the exit condition of the
/// loop to be a canonical != comparison against the incremented loop induction
/// variable.  This pass is able to rewrite the exit tests of any loop where the
/// SCEV analysis can determine a loop-invariant trip count of the loop, which
/// is actually a much broader range than just linear tests.
ICmpInst *IndVarSimplify::LinearFunctionTestReplace(Loop *L,
                                   const SCEV *BackedgeTakenCount,
                                   Value *IndVar,
                                   BasicBlock *ExitingBlock,
                                   BranchInst *BI,
                                   SCEVExpander &Rewriter) {
  // If the exiting block is not the same as the backedge block, we must compare
  // against the preincremented value, otherwise we prefer to compare against
  // the post-incremented value.
  Value *CmpIndVar;
  const SCEV *RHS = BackedgeTakenCount;
  if (ExitingBlock == L->getLoopLatch()) {
    // Add one to the "backedge-taken" count to get the trip count.
    // If this addition may overflow, we have to be more pessimistic and
    // cast the induction variable before doing the add.
    const SCEV *Zero = SE->getIntegerSCEV(0, BackedgeTakenCount->getType());
    const SCEV *N =
      SE->getAddExpr(BackedgeTakenCount,
                     SE->getIntegerSCEV(1, BackedgeTakenCount->getType()));
    if ((isa<SCEVConstant>(N) && !N->isZero()) ||
        SE->isLoopGuardedByCond(L, ICmpInst::ICMP_NE, N, Zero)) {
      // No overflow. Cast the sum.
      RHS = SE->getTruncateOrZeroExtend(N, IndVar->getType());
    } else {
      // Potential overflow. Cast before doing the add.
      RHS = SE->getTruncateOrZeroExtend(BackedgeTakenCount,
                                        IndVar->getType());
      RHS = SE->getAddExpr(RHS,
                           SE->getIntegerSCEV(1, IndVar->getType()));
    }

    // The BackedgeTaken expression contains the number of times that the
    // backedge branches to the loop header.  This is one less than the
    // number of times the loop executes, so use the incremented indvar.
    CmpIndVar = L->getCanonicalInductionVariableIncrement();
  } else {
    // We have to use the preincremented value...
    RHS = SE->getTruncateOrZeroExtend(BackedgeTakenCount,
                                      IndVar->getType());
    CmpIndVar = IndVar;
  }

  // Expand the code for the iteration count.
  assert(RHS->isLoopInvariant(L) &&
         "Computed iteration count is not loop invariant!");
  Value *ExitCnt = Rewriter.expandCodeFor(RHS, IndVar->getType(), BI);

  // Insert a new icmp_ne or icmp_eq instruction before the branch.
  ICmpInst::Predicate Opcode;
  if (L->contains(BI->getSuccessor(0)))
    Opcode = ICmpInst::ICMP_NE;
  else
    Opcode = ICmpInst::ICMP_EQ;

  DOUT << "INDVARS: Rewriting loop exit condition to:\n"
       << "      LHS:" << *CmpIndVar << '\n'
       << "       op:\t"
       << (Opcode == ICmpInst::ICMP_NE ? "!=" : "==") << "\n"
       << "      RHS:\t" << *RHS << "\n";

  ICmpInst *Cond = new ICmpInst(BI, Opcode, CmpIndVar, ExitCnt, "exitcond");

  Instruction *OrigCond = cast<Instruction>(BI->getCondition());
  // It's tempting to use replaceAllUsesWith here to fully replace the old
  // comparison, but that's not immediately safe, since users of the old
  // comparison may not be dominated by the new comparison. Instead, just
  // update the branch to use the new comparison; in the common case this
  // will make old comparison dead.
  BI->setCondition(Cond);
  RecursivelyDeleteTriviallyDeadInstructions(OrigCond);

  ++NumLFTR;
  Changed = true;
  return Cond;
}

/// RewriteLoopExitValues - Check to see if this loop has a computable
/// loop-invariant execution count.  If so, this means that we can compute the
/// final value of any expressions that are recurrent in the loop, and
/// substitute the exit values from the loop into any instructions outside of
/// the loop that use the final values of the current expressions.
///
/// This is mostly redundant with the regular IndVarSimplify activities that
/// happen later, except that it's more powerful in some cases, because it's
/// able to brute-force evaluate arbitrary instructions as long as they have
/// constant operands at the beginning of the loop.
void IndVarSimplify::RewriteLoopExitValues(Loop *L,
                                           const SCEV *BackedgeTakenCount,
                                           SCEVExpander &Rewriter) {
  // Verify the input to the pass in already in LCSSA form.
  assert(L->isLCSSAForm());

  SmallVector<BasicBlock*, 8> ExitBlocks;
  L->getUniqueExitBlocks(ExitBlocks);

  // Find all values that are computed inside the loop, but used outside of it.
  // Because of LCSSA, these values will only occur in LCSSA PHI Nodes.  Scan
  // the exit blocks of the loop to find them.
  for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i) {
    BasicBlock *ExitBB = ExitBlocks[i];

    // If there are no PHI nodes in this exit block, then no values defined
    // inside the loop are used on this path, skip it.
    PHINode *PN = dyn_cast<PHINode>(ExitBB->begin());
    if (!PN) continue;

    unsigned NumPreds = PN->getNumIncomingValues();

    // Iterate over all of the PHI nodes.
    BasicBlock::iterator BBI = ExitBB->begin();
    while ((PN = dyn_cast<PHINode>(BBI++))) {
      if (PN->use_empty())
        continue; // dead use, don't replace it
      // Iterate over all of the values in all the PHI nodes.
      for (unsigned i = 0; i != NumPreds; ++i) {
        // If the value being merged in is not integer or is not defined
        // in the loop, skip it.
        Value *InVal = PN->getIncomingValue(i);
        if (!isa<Instruction>(InVal) ||
            // SCEV only supports integer expressions for now.
            (!isa<IntegerType>(InVal->getType()) &&
             !isa<PointerType>(InVal->getType())))
          continue;

        // If this pred is for a subloop, not L itself, skip it.
        if (LI->getLoopFor(PN->getIncomingBlock(i)) != L)
          continue; // The Block is in a subloop, skip it.

        // Check that InVal is defined in the loop.
        Instruction *Inst = cast<Instruction>(InVal);
        if (!L->contains(Inst->getParent()))
          continue;

        // Okay, this instruction has a user outside of the current loop
        // and varies predictably *inside* the loop.  Evaluate the value it
        // contains when the loop exits, if possible.
        const SCEV *ExitValue = SE->getSCEVAtScope(Inst, L->getParentLoop());
        if (!ExitValue->isLoopInvariant(L))
          continue;

        Changed = true;
        ++NumReplaced;

        Value *ExitVal = Rewriter.expandCodeFor(ExitValue, PN->getType(), Inst);

        DOUT << "INDVARS: RLEV: AfterLoopVal = " << *ExitVal << '\n'
             << "  LoopVal = " << *Inst << "\n";

        PN->setIncomingValue(i, ExitVal);

        // If this instruction is dead now, delete it.
        RecursivelyDeleteTriviallyDeadInstructions(Inst);

        if (NumPreds == 1) {
          // Completely replace a single-pred PHI. This is safe, because the
          // NewVal won't be variant in the loop, so we don't need an LCSSA phi
          // node anymore.
          PN->replaceAllUsesWith(ExitVal);
          RecursivelyDeleteTriviallyDeadInstructions(PN);
        }
      }
      if (NumPreds != 1) {
        // Clone the PHI and delete the original one. This lets IVUsers and
        // any other maps purge the original user from their records.
        PHINode *NewPN = PN->clone(PN->getContext());
        NewPN->takeName(PN);
        NewPN->insertBefore(PN);
        PN->replaceAllUsesWith(NewPN);
        PN->eraseFromParent();
      }
    }
  }
}

void IndVarSimplify::RewriteNonIntegerIVs(Loop *L) {
  // First step.  Check to see if there are any floating-point recurrences.
  // If there are, change them into integer recurrences, permitting analysis by
  // the SCEV routines.
  //
  BasicBlock *Header    = L->getHeader();

  SmallVector<WeakVH, 8> PHIs;
  for (BasicBlock::iterator I = Header->begin();
       PHINode *PN = dyn_cast<PHINode>(I); ++I)
    PHIs.push_back(PN);

  for (unsigned i = 0, e = PHIs.size(); i != e; ++i)
    if (PHINode *PN = dyn_cast_or_null<PHINode>(PHIs[i]))
      HandleFloatingPointIV(L, PN);

  // If the loop previously had floating-point IV, ScalarEvolution
  // may not have been able to compute a trip count. Now that we've done some
  // re-writing, the trip count may be computable.
  if (Changed)
    SE->forgetLoopBackedgeTakenCount(L);
}

bool IndVarSimplify::runOnLoop(Loop *L, LPPassManager &LPM) {
  IU = &getAnalysis<IVUsers>();
  LI = &getAnalysis<LoopInfo>();
  SE = &getAnalysis<ScalarEvolution>();
  DT = &getAnalysis<DominatorTree>();
  Changed = false;

  // If there are any floating-point recurrences, attempt to
  // transform them to use integer recurrences.
  RewriteNonIntegerIVs(L);

  BasicBlock *ExitingBlock = L->getExitingBlock(); // may be null
  const SCEV *BackedgeTakenCount = SE->getBackedgeTakenCount(L);

  // Create a rewriter object which we'll use to transform the code with.
  SCEVExpander Rewriter(*SE);

  // Check to see if this loop has a computable loop-invariant execution count.
  // If so, this means that we can compute the final value of any expressions
  // that are recurrent in the loop, and substitute the exit values from the
  // loop into any instructions outside of the loop that use the final values of
  // the current expressions.
  //
  if (!isa<SCEVCouldNotCompute>(BackedgeTakenCount))
    RewriteLoopExitValues(L, BackedgeTakenCount, Rewriter);

  // Compute the type of the largest recurrence expression, and decide whether
  // a canonical induction variable should be inserted.
  const Type *LargestType = 0;
  bool NeedCannIV = false;
  if (!isa<SCEVCouldNotCompute>(BackedgeTakenCount)) {
    LargestType = BackedgeTakenCount->getType();
    LargestType = SE->getEffectiveSCEVType(LargestType);
    // If we have a known trip count and a single exit block, we'll be
    // rewriting the loop exit test condition below, which requires a
    // canonical induction variable.
    if (ExitingBlock)
      NeedCannIV = true;
  }
  for (unsigned i = 0, e = IU->StrideOrder.size(); i != e; ++i) {
    const SCEV *Stride = IU->StrideOrder[i];
    const Type *Ty = SE->getEffectiveSCEVType(Stride->getType());
    if (!LargestType ||
        SE->getTypeSizeInBits(Ty) >
          SE->getTypeSizeInBits(LargestType))
      LargestType = Ty;

    std::map<const SCEV *, IVUsersOfOneStride *>::iterator SI =
      IU->IVUsesByStride.find(IU->StrideOrder[i]);
    assert(SI != IU->IVUsesByStride.end() && "Stride doesn't exist!");

    if (!SI->second->Users.empty())
      NeedCannIV = true;
  }

  // Now that we know the largest of of the induction variable expressions
  // in this loop, insert a canonical induction variable of the largest size.
  Value *IndVar = 0;
  if (NeedCannIV) {
    // Check to see if the loop already has a canonical-looking induction
    // variable. If one is present and it's wider than the planned canonical
    // induction variable, temporarily remove it, so that the Rewriter
    // doesn't attempt to reuse it.
    PHINode *OldCannIV = L->getCanonicalInductionVariable();
    if (OldCannIV) {
      if (SE->getTypeSizeInBits(OldCannIV->getType()) >
          SE->getTypeSizeInBits(LargestType))
        OldCannIV->removeFromParent();
      else
        OldCannIV = 0;
    }

    IndVar = Rewriter.getOrInsertCanonicalInductionVariable(L, LargestType);

    ++NumInserted;
    Changed = true;
    DOUT << "INDVARS: New CanIV: " << *IndVar << '\n';

    // Now that the official induction variable is established, reinsert
    // the old canonical-looking variable after it so that the IR remains
    // consistent. It will be deleted as part of the dead-PHI deletion at
    // the end of the pass.
    if (OldCannIV)
      OldCannIV->insertAfter(cast<Instruction>(IndVar));
  }

  // If we have a trip count expression, rewrite the loop's exit condition
  // using it.  We can currently only handle loops with a single exit.
  ICmpInst *NewICmp = 0;
  if (!isa<SCEVCouldNotCompute>(BackedgeTakenCount) && ExitingBlock) {
    assert(NeedCannIV &&
           "LinearFunctionTestReplace requires a canonical induction variable");
    // Can't rewrite non-branch yet.
    if (BranchInst *BI = dyn_cast<BranchInst>(ExitingBlock->getTerminator()))
      NewICmp = LinearFunctionTestReplace(L, BackedgeTakenCount, IndVar,
                                          ExitingBlock, BI, Rewriter);
  }

  // Rewrite IV-derived expressions. Clears the rewriter cache.
  RewriteIVExpressions(L, LargestType, Rewriter);

  // The Rewriter may not be used from this point on.

  // Loop-invariant instructions in the preheader that aren't used in the
  // loop may be sunk below the loop to reduce register pressure.
  SinkUnusedInvariants(L);

  // For completeness, inform IVUsers of the IV use in the newly-created
  // loop exit test instruction.
  if (NewICmp)
    IU->AddUsersIfInteresting(cast<Instruction>(NewICmp->getOperand(0)));

  // Clean up dead instructions.
  DeleteDeadPHIs(L->getHeader());
  // Check a post-condition.
  assert(L->isLCSSAForm() && "Indvars did not leave the loop in lcssa form!");
  return Changed;
}

void IndVarSimplify::RewriteIVExpressions(Loop *L, const Type *LargestType,
                                          SCEVExpander &Rewriter) {
  SmallVector<WeakVH, 16> DeadInsts;

  // Rewrite all induction variable expressions in terms of the canonical
  // induction variable.
  //
  // If there were induction variables of other sizes or offsets, manually
  // add the offsets to the primary induction variable and cast, avoiding
  // the need for the code evaluation methods to insert induction variables
  // of different sizes.
  for (unsigned i = 0, e = IU->StrideOrder.size(); i != e; ++i) {
    const SCEV *Stride = IU->StrideOrder[i];

    std::map<const SCEV *, IVUsersOfOneStride *>::iterator SI =
      IU->IVUsesByStride.find(IU->StrideOrder[i]);
    assert(SI != IU->IVUsesByStride.end() && "Stride doesn't exist!");
    ilist<IVStrideUse> &List = SI->second->Users;
    for (ilist<IVStrideUse>::iterator UI = List.begin(),
         E = List.end(); UI != E; ++UI) {
      Value *Op = UI->getOperandValToReplace();
      const Type *UseTy = Op->getType();
      Instruction *User = UI->getUser();

      // Compute the final addrec to expand into code.
      const SCEV *AR = IU->getReplacementExpr(*UI);

      // FIXME: It is an extremely bad idea to indvar substitute anything more
      // complex than affine induction variables.  Doing so will put expensive
      // polynomial evaluations inside of the loop, and the str reduction pass
      // currently can only reduce affine polynomials.  For now just disable
      // indvar subst on anything more complex than an affine addrec, unless
      // it can be expanded to a trivial value.
      if (!AR->isLoopInvariant(L) && !Stride->isLoopInvariant(L))
        continue;

      // Determine the insertion point for this user. By default, insert
      // immediately before the user. The SCEVExpander class will automatically
      // hoist loop invariants out of the loop. For PHI nodes, there may be
      // multiple uses, so compute the nearest common dominator for the
      // incoming blocks.
      Instruction *InsertPt = User;
      if (PHINode *PHI = dyn_cast<PHINode>(InsertPt))
        for (unsigned i = 0, e = PHI->getNumIncomingValues(); i != e; ++i)
          if (PHI->getIncomingValue(i) == Op) {
            if (InsertPt == User)
              InsertPt = PHI->getIncomingBlock(i)->getTerminator();
            else
              InsertPt =
                DT->findNearestCommonDominator(InsertPt->getParent(),
                                               PHI->getIncomingBlock(i))
                      ->getTerminator();
          }

      // Now expand it into actual Instructions and patch it into place.
      Value *NewVal = Rewriter.expandCodeFor(AR, UseTy, InsertPt);

      // Patch the new value into place.
      if (Op->hasName())
        NewVal->takeName(Op);
      User->replaceUsesOfWith(Op, NewVal);
      UI->setOperandValToReplace(NewVal);
      DOUT << "INDVARS: Rewrote IV '" << *AR << "' " << *Op << '\n'
           << "   into = " << *NewVal << "\n";
      ++NumRemoved;
      Changed = true;

      // The old value may be dead now.
      DeadInsts.push_back(Op);
    }
  }

  // Clear the rewriter cache, because values that are in the rewriter's cache
  // can be deleted in the loop below, causing the AssertingVH in the cache to
  // trigger.
  Rewriter.clear();
  // Now that we're done iterating through lists, clean up any instructions
  // which are now dead.
  while (!DeadInsts.empty()) {
    Instruction *Inst = dyn_cast_or_null<Instruction>(DeadInsts.pop_back_val());
    if (Inst)
      RecursivelyDeleteTriviallyDeadInstructions(Inst);
  }
}

/// If there's a single exit block, sink any loop-invariant values that
/// were defined in the preheader but not used inside the loop into the
/// exit block to reduce register pressure in the loop.
void IndVarSimplify::SinkUnusedInvariants(Loop *L) {
  BasicBlock *ExitBlock = L->getExitBlock();
  if (!ExitBlock) return;

  Instruction *InsertPt = ExitBlock->getFirstNonPHI();
  BasicBlock *Preheader = L->getLoopPreheader();
  BasicBlock::iterator I = Preheader->getTerminator();
  while (I != Preheader->begin()) {
    --I;
    // New instructions were inserted at the end of the preheader.
    if (isa<PHINode>(I))
      break;
    // Don't move instructions which might have side effects, since the side
    // effects need to complete before instructions inside the loop.  Also
    // don't move instructions which might read memory, since the loop may
    // modify memory. Note that it's okay if the instruction might have
    // undefined behavior: LoopSimplify guarantees that the preheader
    // dominates the exit block.
    if (I->mayHaveSideEffects() || I->mayReadFromMemory())
      continue;
    // Don't sink static AllocaInsts out of the entry block, which would
    // turn them into dynamic allocas!
    if (AllocaInst *AI = dyn_cast<AllocaInst>(I))
      if (AI->isStaticAlloca())
        continue;
    // Determine if there is a use in or before the loop (direct or
    // otherwise).
    bool UsedInLoop = false;
    for (Value::use_iterator UI = I->use_begin(), UE = I->use_end();
         UI != UE; ++UI) {
      BasicBlock *UseBB = cast<Instruction>(UI)->getParent();
      if (PHINode *P = dyn_cast<PHINode>(UI)) {
        unsigned i =
          PHINode::getIncomingValueNumForOperand(UI.getOperandNo());
        UseBB = P->getIncomingBlock(i);
      }
      if (UseBB == Preheader || L->contains(UseBB)) {
        UsedInLoop = true;
        break;
      }
    }
    // If there is, the def must remain in the preheader.
    if (UsedInLoop)
      continue;
    // Otherwise, sink it to the exit block.
    Instruction *ToMove = I;
    bool Done = false;
    if (I != Preheader->begin())
      --I;
    else
      Done = true;
    ToMove->moveBefore(InsertPt);
    if (Done)
      break;
    InsertPt = ToMove;
  }
}

/// Return true if it is OK to use SIToFPInst for an inducation variable
/// with given inital and exit values.
static bool useSIToFPInst(ConstantFP &InitV, ConstantFP &ExitV,
                          uint64_t intIV, uint64_t intEV) {

  if (InitV.getValueAPF().isNegative() || ExitV.getValueAPF().isNegative())
    return true;

  // If the iteration range can be handled by SIToFPInst then use it.
  APInt Max = APInt::getSignedMaxValue(32);
  if (Max.getZExtValue() > static_cast<uint64_t>(abs64(intEV - intIV)))
    return true;

  return false;
}

/// convertToInt - Convert APF to an integer, if possible.
static bool convertToInt(const APFloat &APF, uint64_t *intVal) {

  bool isExact = false;
  if (&APF.getSemantics() == &APFloat::PPCDoubleDouble)
    return false;
  if (APF.convertToInteger(intVal, 32, APF.isNegative(),
                           APFloat::rmTowardZero, &isExact)
      != APFloat::opOK)
    return false;
  if (!isExact)
    return false;
  return true;

}

/// HandleFloatingPointIV - If the loop has floating induction variable
/// then insert corresponding integer induction variable if possible.
/// For example,
/// for(double i = 0; i < 10000; ++i)
///   bar(i)
/// is converted into
/// for(int i = 0; i < 10000; ++i)
///   bar((double)i);
///
void IndVarSimplify::HandleFloatingPointIV(Loop *L, PHINode *PH) {

  unsigned IncomingEdge = L->contains(PH->getIncomingBlock(0));
  unsigned BackEdge     = IncomingEdge^1;

  // Check incoming value.
  ConstantFP *InitValue = dyn_cast<ConstantFP>(PH->getIncomingValue(IncomingEdge));
  if (!InitValue) return;
  uint64_t newInitValue =
              Type::getInt32Ty(PH->getContext())->getPrimitiveSizeInBits();
  if (!convertToInt(InitValue->getValueAPF(), &newInitValue))
    return;

  // Check IV increment. Reject this PH if increement operation is not
  // an add or increment value can not be represented by an integer.
  BinaryOperator *Incr =
    dyn_cast<BinaryOperator>(PH->getIncomingValue(BackEdge));
  if (!Incr) return;
  if (Incr->getOpcode() != Instruction::FAdd) return;
  ConstantFP *IncrValue = NULL;
  unsigned IncrVIndex = 1;
  if (Incr->getOperand(1) == PH)
    IncrVIndex = 0;
  IncrValue = dyn_cast<ConstantFP>(Incr->getOperand(IncrVIndex));
  if (!IncrValue) return;
  uint64_t newIncrValue =
              Type::getInt32Ty(PH->getContext())->getPrimitiveSizeInBits();
  if (!convertToInt(IncrValue->getValueAPF(), &newIncrValue))
    return;

  // Check Incr uses. One user is PH and the other users is exit condition used
  // by the conditional terminator.
  Value::use_iterator IncrUse = Incr->use_begin();
  Instruction *U1 = cast<Instruction>(IncrUse++);
  if (IncrUse == Incr->use_end()) return;
  Instruction *U2 = cast<Instruction>(IncrUse++);
  if (IncrUse != Incr->use_end()) return;

  // Find exit condition.
  FCmpInst *EC = dyn_cast<FCmpInst>(U1);
  if (!EC)
    EC = dyn_cast<FCmpInst>(U2);
  if (!EC) return;

  if (BranchInst *BI = dyn_cast<BranchInst>(EC->getParent()->getTerminator())) {
    if (!BI->isConditional()) return;
    if (BI->getCondition() != EC) return;
  }

  // Find exit value. If exit value can not be represented as an interger then
  // do not handle this floating point PH.
  ConstantFP *EV = NULL;
  unsigned EVIndex = 1;
  if (EC->getOperand(1) == Incr)
    EVIndex = 0;
  EV = dyn_cast<ConstantFP>(EC->getOperand(EVIndex));
  if (!EV) return;
  uint64_t intEV = Type::getInt32Ty(PH->getContext())->getPrimitiveSizeInBits();
  if (!convertToInt(EV->getValueAPF(), &intEV))
    return;

  // Find new predicate for integer comparison.
  CmpInst::Predicate NewPred = CmpInst::BAD_ICMP_PREDICATE;
  switch (EC->getPredicate()) {
  case CmpInst::FCMP_OEQ:
  case CmpInst::FCMP_UEQ:
    NewPred = CmpInst::ICMP_EQ;
    break;
  case CmpInst::FCMP_OGT:
  case CmpInst::FCMP_UGT:
    NewPred = CmpInst::ICMP_UGT;
    break;
  case CmpInst::FCMP_OGE:
  case CmpInst::FCMP_UGE:
    NewPred = CmpInst::ICMP_UGE;
    break;
  case CmpInst::FCMP_OLT:
  case CmpInst::FCMP_ULT:
    NewPred = CmpInst::ICMP_ULT;
    break;
  case CmpInst::FCMP_OLE:
  case CmpInst::FCMP_ULE:
    NewPred = CmpInst::ICMP_ULE;
    break;
  default:
    break;
  }
  if (NewPred == CmpInst::BAD_ICMP_PREDICATE) return;

  // Insert new integer induction variable.
  PHINode *NewPHI = PHINode::Create(Type::getInt32Ty(PH->getContext()),
                                    PH->getName()+".int", PH);
  NewPHI->addIncoming(ConstantInt::get(Type::getInt32Ty(PH->getContext()),
                                       newInitValue),
                      PH->getIncomingBlock(IncomingEdge));

  Value *NewAdd = BinaryOperator::CreateAdd(NewPHI,
                           ConstantInt::get(Type::getInt32Ty(PH->getContext()),
                                                             newIncrValue),
                                            Incr->getName()+".int", Incr);
  NewPHI->addIncoming(NewAdd, PH->getIncomingBlock(BackEdge));

  // The back edge is edge 1 of newPHI, whatever it may have been in the
  // original PHI.
  ConstantInt *NewEV = ConstantInt::get(Type::getInt32Ty(PH->getContext()),
                                        intEV);
  Value *LHS = (EVIndex == 1 ? NewPHI->getIncomingValue(1) : NewEV);
  Value *RHS = (EVIndex == 1 ? NewEV : NewPHI->getIncomingValue(1));
  ICmpInst *NewEC = new ICmpInst(EC->getParent()->getTerminator(),
                                 NewPred, LHS, RHS, EC->getName());

  // In the following deltions, PH may become dead and may be deleted.
  // Use a WeakVH to observe whether this happens.
  WeakVH WeakPH = PH;

  // Delete old, floating point, exit comparision instruction.
  NewEC->takeName(EC);
  EC->replaceAllUsesWith(NewEC);
  RecursivelyDeleteTriviallyDeadInstructions(EC);

  // Delete old, floating point, increment instruction.
  Incr->replaceAllUsesWith(UndefValue::get(Incr->getType()));
  RecursivelyDeleteTriviallyDeadInstructions(Incr);

  // Replace floating induction variable, if it isn't already deleted.
  // Give SIToFPInst preference over UIToFPInst because it is faster on
  // platforms that are widely used.
  if (WeakPH && !PH->use_empty()) {
    if (useSIToFPInst(*InitValue, *EV, newInitValue, intEV)) {
      SIToFPInst *Conv = new SIToFPInst(NewPHI, PH->getType(), "indvar.conv",
                                        PH->getParent()->getFirstNonPHI());
      PH->replaceAllUsesWith(Conv);
    } else {
      UIToFPInst *Conv = new UIToFPInst(NewPHI, PH->getType(), "indvar.conv",
                                        PH->getParent()->getFirstNonPHI());
      PH->replaceAllUsesWith(Conv);
    }
    RecursivelyDeleteTriviallyDeadInstructions(PH);
  }

  // Add a new IVUsers entry for the newly-created integer PHI.
  IU->AddUsersIfInteresting(NewPHI);
}
