//===- ScalarEvolutionNormalization.cpp - See below -----------------------===//
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

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ScalarEvolutionNormalization.h"
using namespace llvm;

/// TransformKind - Different types of transformations that
/// TransformForPostIncUse can do.
enum TransformKind {
  /// Normalize - Normalize according to the given loops.
  Normalize,
  /// Denormalize - Perform the inverse transform on the expression with the
  /// given loop set.
  Denormalize
};

namespace {
struct NormalizeDenormalizeRewriter
    : public SCEVRewriteVisitor<NormalizeDenormalizeRewriter> {
  const TransformKind Kind;

  // NB! Pred is a function_ref.  Storing it here is okay only because
  // we're careful about the lifetime of NormalizeDenormalizeRewriter.
  const NormalizePredTy Pred;

  NormalizeDenormalizeRewriter(TransformKind Kind, NormalizePredTy Pred,
                               ScalarEvolution &SE)
      : SCEVRewriteVisitor<NormalizeDenormalizeRewriter>(SE), Kind(Kind),
        Pred(Pred) {}
  const SCEV *visitAddRecExpr(const SCEVAddRecExpr *Expr);
};
} // namespace

const SCEV *
NormalizeDenormalizeRewriter::visitAddRecExpr(const SCEVAddRecExpr *AR) {
  SmallVector<const SCEV *, 8> Operands;

  transform(AR->operands(), std::back_inserter(Operands),
            [&](const SCEV *Op) { return visit(Op); });

  // Conservatively use AnyWrap until/unless we need FlagNW.
  const SCEV *Result =
      SE.getAddRecExpr(Operands, AR->getLoop(), SCEV::FlagAnyWrap);
  switch (Kind) {
  case Normalize:
    // We want to normalize step expression, because otherwise we might not be
    // able to denormalize to the original expression.
    //
    // Here is an example what will happen if we don't normalize step:
    //  ORIGINAL ISE:
    //    {(100 /u {1,+,1}<%bb16>),+,(100 /u {1,+,1}<%bb16>)}<%bb25>
    //  NORMALIZED ISE:
    //    {((-1 * (100 /u {1,+,1}<%bb16>)) + (100 /u {0,+,1}<%bb16>)),+,
    //     (100 /u {0,+,1}<%bb16>)}<%bb25>
    //  DENORMALIZED BACK ISE:
    //    {((2 * (100 /u {1,+,1}<%bb16>)) + (-1 * (100 /u {2,+,1}<%bb16>))),+,
    //     (100 /u {1,+,1}<%bb16>)}<%bb25>
    //  Note that the initial value changes after normalization +
    //  denormalization, which isn't correct.
    if (Pred(AR)) {
      const SCEV *TransformedStep = visit(AR->getStepRecurrence(SE));
      Result = SE.getMinusSCEV(Result, TransformedStep);
    }
    break;
  case Denormalize:
    // Here we want to normalize step expressions for the same reasons, as
    // stated above.
    if (Pred(AR)) {
      const SCEV *TransformedStep = visit(AR->getStepRecurrence(SE));
      Result = SE.getAddExpr(Result, TransformedStep);
    }
    break;
  }
  return Result;
}

const SCEV *llvm::normalizeForPostIncUse(const SCEV *S,
                                         const PostIncLoopSet &Loops,
                                         ScalarEvolution &SE) {
  auto Pred = [&](const SCEVAddRecExpr *AR) {
    return Loops.count(AR->getLoop());
  };
  return NormalizeDenormalizeRewriter(Normalize, Pred, SE).visit(S);
}

const SCEV *llvm::normalizeForPostIncUseIf(const SCEV *S, NormalizePredTy Pred,
                                           ScalarEvolution &SE) {
  return NormalizeDenormalizeRewriter(Normalize, Pred, SE).visit(S);
}

const SCEV *llvm::denormalizeForPostIncUse(const SCEV *S,
                                           const PostIncLoopSet &Loops,
                                           ScalarEvolution &SE) {
  auto Pred = [&](const SCEVAddRecExpr *AR) {
    return Loops.count(AR->getLoop());
  };
  return NormalizeDenormalizeRewriter(Denormalize, Pred, SE).visit(S);
}
