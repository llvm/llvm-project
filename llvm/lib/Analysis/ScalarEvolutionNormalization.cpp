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

typedef DenseMap<const SCEV *, const SCEV *> NormalizedCacheTy;

static const SCEV *transformSubExpr(const TransformKind Kind,
                                    NormalizePredTy Pred, ScalarEvolution &SE,
                                    NormalizedCacheTy &Cache, const SCEV *S);

/// Implement post-inc transformation for all valid expression types.
static const SCEV *transformImpl(const TransformKind Kind, NormalizePredTy Pred,
                                 ScalarEvolution &SE, NormalizedCacheTy &Cache,
                                 const SCEV *S) {
  if (const SCEVCastExpr *X = dyn_cast<SCEVCastExpr>(S)) {
    const SCEV *O = X->getOperand();
    const SCEV *N = transformSubExpr(Kind, Pred, SE, Cache, O);
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

    transform(AR->operands(), std::back_inserter(Operands),
              [&](const SCEV *Op) {
                return transformSubExpr(Kind, Pred, SE, Cache, Op);
              });

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
        const SCEV *TransformedStep =
            transformSubExpr(Kind, Pred, SE, Cache, AR->getStepRecurrence(SE));
        Result = SE.getMinusSCEV(Result, TransformedStep);
      }
      break;
    case Denormalize:
      // Here we want to normalize step expressions for the same reasons, as
      // stated above.
      if (Pred(AR)) {
        const SCEV *TransformedStep =
            transformSubExpr(Kind, Pred, SE, Cache, AR->getStepRecurrence(SE));
        Result = SE.getAddExpr(Result, TransformedStep);
      }
      break;
    }
    return Result;
  }

  if (const SCEVNAryExpr *X = dyn_cast<SCEVNAryExpr>(S)) {
    SmallVector<const SCEV *, 8> Operands;
    bool Changed = false;
    // Transform each operand.
    for (auto *O : X->operands()) {
      const SCEV *N = transformSubExpr(Kind, Pred, SE, Cache, O);
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
    const SCEV *LN = transformSubExpr(Kind, Pred, SE, Cache, LO);
    const SCEV *RN = transformSubExpr(Kind, Pred, SE, Cache, RO);
    if (LO != LN || RO != RN)
      return SE.getUDivExpr(LN, RN);
    return S;
  }

  llvm_unreachable("Unexpected SCEV kind!");
}

/// Manage recursive transformation across an expression DAG. Revisiting
/// expressions would lead to exponential recursion.
static const SCEV *transformSubExpr(const TransformKind Kind,
                                    NormalizePredTy Pred, ScalarEvolution &SE,
                                    NormalizedCacheTy &Cache, const SCEV *S) {
  if (isa<SCEVConstant>(S) || isa<SCEVUnknown>(S))
    return S;

  const SCEV *Result = Cache.lookup(S);
  if (Result)
    return Result;

  Result = transformImpl(Kind, Pred, SE, Cache, S);
  Cache[S] = Result;
  return Result;
}

const SCEV *llvm::normalizeForPostIncUse(const SCEV *S,
                                         const PostIncLoopSet &Loops,
                                         ScalarEvolution &SE) {
  auto Pred = [&](const SCEVAddRecExpr *AR) {
    return Loops.count(AR->getLoop());
  };
  NormalizedCacheTy Cache;
  return transformSubExpr(Normalize, Pred, SE, Cache, S);
}

const SCEV *llvm::normalizeForPostIncUseIf(const SCEV *S, NormalizePredTy Pred,
                                           ScalarEvolution &SE) {
  NormalizedCacheTy Cache;
  return transformSubExpr(Normalize, Pred, SE, Cache, S);
}

const SCEV *llvm::denormalizeForPostIncUse(const SCEV *S,
                                           const PostIncLoopSet &Loops,
                                           ScalarEvolution &SE) {
  auto Pred = [&](const SCEVAddRecExpr *AR) {
    return Loops.count(AR->getLoop());
  };
  NormalizedCacheTy Cache;
  return transformSubExpr(Denormalize, Pred, SE, Cache, S);
}
