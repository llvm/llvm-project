//===- ScalarEvolutionNormalization.cpp - See below -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for working with "normalized" expressions.
// See the comments at the top of ScalarEvolutionNormalization.h for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ScalarEvolutionNormalization.h"
using namespace llvm;

/// IVUseShouldUsePostIncValue - We have discovered a "User" of an IV expression
/// and now we need to decide whether the user should use the preinc or post-inc
/// value.  If this user should use the post-inc version of the IV, return true.
///
/// Choosing wrong here can break dominance properties (if we choose to use the
/// post-inc value when we cannot) or it can end up adding extra live-ranges to
/// the loop, resulting in reg-reg copies (if we use the pre-inc value when we
/// should use the post-inc value).
static bool IVUseShouldUsePostIncValue(Instruction *User, Value *Operand,
                                       const Loop *L, DominatorTree *DT) {
  // If the user is in the loop, use the preinc value.
  if (L->contains(User)) return false;

  BasicBlock *LatchBlock = L->getLoopLatch();
  if (!LatchBlock)
    return false;

  // Ok, the user is outside of the loop.  If it is dominated by the latch
  // block, use the post-inc value.
  if (DT->dominates(LatchBlock, User->getParent()))
    return true;

  // There is one case we have to be careful of: PHI nodes.  These little guys
  // can live in blocks that are not dominated by the latch block, but (since
  // their uses occur in the predecessor block, not the block the PHI lives in)
  // should still use the post-inc value.  Check for this case now.
  PHINode *PN = dyn_cast<PHINode>(User);
  if (!PN || !Operand) return false; // not a phi, not dominated by latch block.

  // Look at all of the uses of Operand by the PHI node.  If any use corresponds
  // to a block that is not dominated by the latch block, give up and use the
  // preincremented value.
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
    if (PN->getIncomingValue(i) == Operand &&
        !DT->dominates(LatchBlock, PN->getIncomingBlock(i)))
      return false;

  // Okay, all uses of Operand by PN are in predecessor blocks that really are
  // dominated by the latch block.  Use the post-incremented value.
  return true;
}

namespace {

/// Hold the state used during post-inc expression transformation, including a
/// map of transformed expressions.
class PostIncTransform {
  TransformKind Kind;
  PostIncLoopSet &Loops;
  ScalarEvolution &SE;
  DominatorTree &DT;

  DenseMap<const SCEV*, const SCEV*> Transformed;

public:
  PostIncTransform(TransformKind kind, PostIncLoopSet &loops,
                   ScalarEvolution &se, DominatorTree &dt):
    Kind(kind), Loops(loops), SE(se), DT(dt) {}

  const SCEV *TransformSubExpr(const SCEV *S, Instruction *User,
                               Value *OperandValToReplace);

protected:
  const SCEV *TransformImpl(const SCEV *S, Instruction *User,
                            Value *OperandValToReplace);
};

} // namespace

/// Implement post-inc transformation for all valid expression types.
const SCEV *PostIncTransform::
TransformImpl(const SCEV *S, Instruction *User, Value *OperandValToReplace) {

  if (const SCEVCastExpr *X = dyn_cast<SCEVCastExpr>(S)) {
    const SCEV *O = X->getOperand();
    const SCEV *N = TransformSubExpr(O, User, OperandValToReplace);
    if (O != N)
      switch (S->getSCEVType()) {
      case scZeroExtend: return SE.getZeroExtendExpr(N, S->getType());
      case scSignExtend: return SE.getSignExtendExpr(N, S->getType());
      case scTruncate: return SE.getTruncateExpr(N, S->getType());
      default: llvm_unreachable("Unexpected SCEVCastExpr kind!");
      }
    return S;
  }

  if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(S)) {
    // An addrec. This is the interesting part.
    SmallVector<const SCEV *, 8> Operands;
    const Loop *L = AR->getLoop();
    // The addrec conceptually uses its operands at loop entry.
    Instruction *LUser = L->getHeader()->begin();
    // Transform each operand.
    for (SCEVNAryExpr::op_iterator I = AR->op_begin(), E = AR->op_end();
         I != E; ++I) {
      Operands.push_back(TransformSubExpr(*I, LUser, 0));
    }
    // Conservatively use AnyWrap until/unless we need FlagNW.
    const SCEV *Result = SE.getAddRecExpr(Operands, L, SCEV::FlagAnyWrap);
    switch (Kind) {
    case NormalizeAutodetect:
      if (IVUseShouldUsePostIncValue(User, OperandValToReplace, L, &DT)) {
        const SCEV *TransformedStep =
          TransformSubExpr(AR->getStepRecurrence(SE),
                           User, OperandValToReplace);
        Result = SE.getMinusSCEV(Result, TransformedStep);
        Loops.insert(L);
      }
#if 0
      // This assert is conceptually correct, but ScalarEvolution currently
      // sometimes fails to canonicalize two equal SCEVs to exactly the same
      // form. It's possibly a pessimization when this happens, but it isn't a
      // correctness problem, so disable this assert for now.
      assert(S == TransformSubExpr(Result, User, OperandValToReplace) &&
             "SCEV normalization is not invertible!");
#endif
      break;
    case Normalize:
      if (Loops.count(L)) {
        const SCEV *TransformedStep =
          TransformSubExpr(AR->getStepRecurrence(SE),
                           User, OperandValToReplace);
        Result = SE.getMinusSCEV(Result, TransformedStep);
      }
#if 0
      // See the comment on the assert above.
      assert(S == TransformSubExpr(Result, User, OperandValToReplace) &&
             "SCEV normalization is not invertible!");
#endif
      break;
    case Denormalize:
      if (Loops.count(L))
        Result = cast<SCEVAddRecExpr>(Result)->getPostIncExpr(SE);
      break;
    }
    return Result;
  }

  if (const SCEVNAryExpr *X = dyn_cast<SCEVNAryExpr>(S)) {
    SmallVector<const SCEV *, 8> Operands;
    bool Changed = false;
    // Transform each operand.
    for (SCEVNAryExpr::op_iterator I = X->op_begin(), E = X->op_end();
         I != E; ++I) {
      const SCEV *O = *I;
      const SCEV *N = TransformSubExpr(O, User, OperandValToReplace);
      Changed |= N != O;
      Operands.push_back(N);
    }
    // If any operand actually changed, return a transformed result.
    if (Changed)
      switch (S->getSCEVType()) {
      case scAddExpr: return SE.getAddExpr(Operands);
      case scMulExpr: return SE.getMulExpr(Operands);
      case scSMaxExpr: return SE.getSMaxExpr(Operands);
      case scUMaxExpr: return SE.getUMaxExpr(Operands);
      default: llvm_unreachable("Unexpected SCEVNAryExpr kind!");
      }
    return S;
  }

  if (const SCEVUDivExpr *X = dyn_cast<SCEVUDivExpr>(S)) {
    const SCEV *LO = X->getLHS();
    const SCEV *RO = X->getRHS();
    const SCEV *LN = TransformSubExpr(LO, User, OperandValToReplace);
    const SCEV *RN = TransformSubExpr(RO, User, OperandValToReplace);
    if (LO != LN || RO != RN)
      return SE.getUDivExpr(LN, RN);
    return S;
  }

  llvm_unreachable("Unexpected SCEV kind!");
}

/// Manage recursive transformation across an expression DAG. Revisiting
/// expressions would lead to exponential recursion.
const SCEV *PostIncTransform::
TransformSubExpr(const SCEV *S, Instruction *User, Value *OperandValToReplace) {

  if (isa<SCEVConstant>(S) || isa<SCEVUnknown>(S))
    return S;

  const SCEV *Result = Transformed.lookup(S);
  if (Result)
    return Result;

  Result = TransformImpl(S, User, OperandValToReplace);
  Transformed[S] = Result;
  return Result;
}

/// Top level driver for transforming an expression DAG into its requested
/// post-inc form (either "Normalized" or "Denormalized".
const SCEV *llvm::TransformForPostIncUse(TransformKind Kind,
                                         const SCEV *S,
                                         Instruction *User,
                                         Value *OperandValToReplace,
                                         PostIncLoopSet &Loops,
                                         ScalarEvolution &SE,
                                         DominatorTree &DT) {
  PostIncTransform Transform(Kind, Loops, SE, DT);
  return Transform.TransformSubExpr(S, User, OperandValToReplace);
}
