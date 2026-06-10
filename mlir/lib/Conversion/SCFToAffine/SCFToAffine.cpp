//===- SCFToAffine.cpp - SCF to Affine conversion -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to raise scf ops to affine ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SCFToAffine/SCFToAffine.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"

namespace mlir {
#define GEN_PASS_DEF_RAISESCFTOAFFINEPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "raise-scf-to-affine"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// SCFToAffinePass
//===----------------------------------------------------------------------===//

struct SCFToAffinePass
    : public impl::RaiseSCFToAffinePassBase<SCFToAffinePass> {
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// ForOpRewrite
//===----------------------------------------------------------------------===//

/// Raise an `scf.for` to an equivalent `affine.for` if lb, ub and step satisfy
/// certain constraints making this possible.
struct ForOpRewrite : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override;

private:
  /// Definitively decide whether we are going to raise or not.
  ///
  /// An `scf.for` can trivially be raised if lb, ub are dimensions and step is
  /// a constant. With some more work one can raise under relaxed constraints as
  /// expressed by this function.
  bool canRaiseToAffine(scf::ForOp op) const;

  /// Cast lb, ub, step and the induction variable of an integer-typed `op` to
  /// `index`, in place. The bound and step casts are placed at the top level of
  /// the affine scope so they are valid affine symbols; the induction variable
  /// is cast back to its original type at the start of the body so the body is
  /// left unchanged. Assumes `canRaiseToAffine(op) == true`.
  void castBoundsToIndex(scf::ForOp op, PatternRewriter &rewriter) const;

  /// Returns an equivalent `affine.for` skeleton and the *old* induction
  /// variable for use by the body that is inlined later. The affine loop body
  /// is left empty except for an operation computing the old induction variable
  /// from the new one *iff* it differs from the new one.
  ///
  /// Assumes `canRaiseToAffine(op) == true` and index casts were performed (if
  /// necessary).
  ///
  /// There are two cases:
  ///
  /// 1. step is constant
  /// 2. step is dynamic (not constant)
  ///
  /// In case (1) and if lb, ub are (valid) dimensions `scf.for` is trivially
  /// raised (leaving lb, ub, iv as is). If lb is an `affine.max` we "inline" it
  /// into the loop's lower bound map. Similarly if ub is an `affine.min`.
  ///
  /// In case (2) we *normalize* the loop to run from 0 with step 1: the new
  /// upper bound is `ceil((ub - lb) / step)` and the original induction
  /// variable is recovered in the body as `lb + step * new_iv`. Here we require
  /// lb to be a dimension; ub may still be an `affine.min`, which is rescaled
  /// accordingly.
  std::pair<affine::AffineForOp, Value>
  createAffineFor(scf::ForOp op, PatternRewriter &rewriter) const;

  std::pair<affine::AffineForOp, Value>
  caseConstantStep(scf::ForOp op, int64_t step,
                   PatternRewriter &rewriter) const;

  std::pair<affine::AffineForOp, Value>
  caseDynamicStep(scf::ForOp op, PatternRewriter &rewriter) const;
};

bool indexBoundsRaisable(scf::ForOp op) {
  auto lb = op.getLowerBound();
  auto ub = op.getUpperBound();
  IntegerAttr constAttr;

  // The asymmetry between lb and ub comes from the fact that the step
  // normalization (for non-constant (dynamic) steps) does not work with
  // multiple *lower* bounds (max).
  bool lbOK = affine::isValidDim(lb) ||
              (isa_and_present<affine::AffineMaxOp>(lb.getDefiningOp()) &&
               matchPattern(op.getStep(), m_Constant(&constAttr)));
  bool ubOK = affine::isValidDim(ub) ||
              isa_and_present<affine::AffineMinOp>(ub.getDefiningOp());
  bool stepOK = affine::isValidSymbol(op.getStep());

  return lbOK && ubOK && stepOK;
}

/// Decide whether an integer-typed loop can be raised by first casting its
/// bounds (lb, ub, step) to `index`. Requires the cast to be lossless under
/// affine's *signed* `index` interpretation, and every bound to be available at
/// the top level of the affine scope (so the inserted casts are valid symbols).
bool intBoundsRaisable(scf::ForOp op, IntegerType intType) {
  uint64_t indexWidth = DataLayout::closest(op)
                            .getTypeSizeInBits(IndexType::get(op.getContext()))
                            .getFixedValue();
  // Lossless under signed index: sign-extend needs width <= indexWidth;
  // zero-extend (unsigned) needs a spare sign bit, i.e. width < indexWidth.
  uint64_t need = intType.getWidth() + (op.getUnsignedCmp() ? 1 : 0);
  if (need > indexWidth)
    return false;

  Region *scope = affine::getAffineScope(op);
  if (!scope)
    return false;

  // Being top-level implies the value is a symbol once it is casted to index.
  return affine::isTopLevelValue(op.getLowerBound(), scope) &&
         affine::isTopLevelValue(op.getUpperBound(), scope) &&
         affine::isTopLevelValue(op.getStep(), scope);
}

bool ForOpRewrite::canRaiseToAffine(scf::ForOp op) const {
  Type type = op.getInductionVar().getType();
  if (isa<IndexType>(type))
    return indexBoundsRaisable(op);
  if (auto intType = dyn_cast<IntegerType>(type))
    return intBoundsRaisable(op, intType);
  return false;
}

LogicalResult ForOpRewrite::matchAndRewrite(scf::ForOp op,
                                            PatternRewriter &rewriter) const {
  if (!canRaiseToAffine(op)) {
    LDBG() << "[affine] Cannot raise scf op: " << op << "\n";
    return failure();
  }

  if (!isa<IndexType>(op.getInductionVar().getType()))
    castBoundsToIndex(op, rewriter);

  auto [affineFor, oldIV] = createAffineFor(op, rewriter);
  Block *affineBody = affineFor.getBody();

  if (affineBody->mightHaveTerminator()) {
    // No unregistered ops in the body, so this is definitive.
    Operation *terminator = affineBody->getTerminator();
    assert(isa<affine::AffineYieldOp>(terminator) &&
           "expected affine.yield if there *might* be terminator");
    rewriter.eraseOp(terminator);
  }

  SmallVector<Value> argValues;
  argValues.push_back(oldIV);
  llvm::append_range(argValues, affineFor.getRegionIterArgs());
  rewriter.inlineBlockBefore(op.getBody(), affineBody, affineBody->end(),
                             argValues);

  auto scfYieldOp = cast<scf::YieldOp>(affineBody->getTerminator());
  rewriter.setInsertionPointToEnd(affineBody);
  rewriter.replaceOpWithNewOp<affine::AffineYieldOp>(scfYieldOp,
                                                     scfYieldOp->getOperands());

  rewriter.replaceOp(op, affineFor);
  return success();
}

std::pair<affine::AffineForOp, Value>
ForOpRewrite::createAffineFor(scf::ForOp op, PatternRewriter &rewriter) const {
  IntegerAttr constAttr;
  if (matchPattern(op.getStep(), m_Constant(&constAttr))) {
    int64_t step = constAttr.getInt();
    assert(step > 0 && "scf.for has positive step");
    return caseConstantStep(op, step, rewriter);
  }
  return caseDynamicStep(op, rewriter);
}

std::pair<affine::AffineForOp, Value>
ForOpRewrite::caseConstantStep(scf::ForOp op, int64_t step,
                               PatternRewriter &rewriter) const {
  auto lb = op.getLowerBound();
  auto ub = op.getUpperBound();

  auto lbOperands = ValueRange(lb);
  auto ubOperands = ValueRange(ub);

  auto lbMap = AffineMap::getMultiDimIdentityMap(1, rewriter.getContext());
  auto ubMap = AffineMap::getMultiDimIdentityMap(1, rewriter.getContext());

  if (auto ubMinOp = ub.getDefiningOp<affine::AffineMinOp>()) {
    ubOperands = ubMinOp->getOperands();
    ubMap = ubMinOp.getAffineMap();
  }

  if (auto lbMaxOp = lb.getDefiningOp<affine::AffineMaxOp>()) {
    lbOperands = lbMaxOp->getOperands();
    lbMap = lbMaxOp.getAffineMap();
  }

  auto affineFor =
      affine::AffineForOp::create(rewriter, op.getLoc(), lbOperands, lbMap,
                                  ubOperands, ubMap, step, op.getInits());

  return std::make_pair(affineFor, affineFor.getInductionVar());
}

std::pair<affine::AffineForOp, Value>
ForOpRewrite::caseDynamicStep(scf::ForOp op, PatternRewriter &rewriter) const {
  Value lb = op.getLowerBound();
  Value ub = op.getUpperBound();
  Value step = op.getStep();

  assert(affine::isValidDim(lb) &&
         "dynamic-step lower bound must be a valid affine dim");

  AffineExpr d0 = rewriter.getAffineDimExpr(0);
  AffineExpr d1 = rewriter.getAffineDimExpr(1);
  AffineExpr s0 = rewriter.getAffineSymbolExpr(0);
  AffineMap zeroMap = rewriter.getConstantAffineMap(0);

  llvm::SmallVector<Value, 3> ubOperands = {lb, ub, step};

  // ub is transformed with (x - lb + step - 1) floorDiv step where x ranges
  // over all ub_i. lb is transformed to zero.

  AffineMap ubMap = AffineMap::get(2, 1, (d1 - d0 + s0 - 1).floorDiv(s0));

  if (auto ubMinOp = ub.getDefiningOp<affine::AffineMinOp>()) {
    AffineMap origUbMap = ubMinOp.getAffineMap();
    unsigned ubDims = origUbMap.getNumDims();
    unsigned ubSyms = origUbMap.getNumSymbols();

    AffineExpr lbDim = rewriter.getAffineDimExpr(ubDims);
    AffineExpr stepSym = rewriter.getAffineSymbolExpr(ubSyms);

    SmallVector<AffineExpr> ubExprs;
    ubExprs.reserve(origUbMap.getNumResults());
    for (AffineExpr ubI : origUbMap.getResults()) {
      ubExprs.push_back((ubI - lbDim + stepSym - 1).floorDiv(stepSym));
    }

    // Combined space: dims = [ub dims, lb]
    //                 syms = [ub syms, step]
    ubMap =
        AffineMap::get(ubDims + 1, ubSyms + 1, ubExprs, rewriter.getContext());

    // Operand order consistent with "combined space" above:
    ValueRange ubOps = ubMinOp->getOperands();
    SmallVector<Value> combined;
    combined.append(ubOps.begin(), ubOps.begin() + ubDims); // ub dims
    combined.push_back(lb);                                 // lb (single dim)
    combined.append(ubOps.begin() + ubDims, ubOps.end());   // ub syms
    combined.push_back(op.getStep());                       // step (single sym)
    ubOperands = std::move(combined);
  }

  auto affineFor = affine::AffineForOp::create(
      rewriter, op.getLoc(), {}, zeroMap, ubOperands, ubMap, 1, op.getInits());

  // old_iv = old_lb + new_iv * step
  AffineMap ivMap = AffineMap::get(2, 1, d0 + d1 * s0);

  llvm::SmallVector<Value, 3> ivOperands = {lb, affineFor.getInductionVar(),
                                            step};

  rewriter.setInsertionPointToStart(affineFor.getBody());
  auto oldIV =
      affine::AffineApplyOp::create(rewriter, op.getLoc(), ivMap, ivOperands);

  return std::make_pair(affineFor, oldIV);
}

void ForOpRewrite::castBoundsToIndex(scf::ForOp loop,
                                     PatternRewriter &rewriter) const {
  OpBuilder::InsertionGuard guard(rewriter);

  Value lb = loop.getLowerBound();
  Value ub = loop.getUpperBound();
  Value step = loop.getStep();
  Type originalType = step.getType();

  assert(lb.getType() == originalType && ub.getType() == originalType &&
         "expected lb, ub, and step to have the same type");

  auto createIndexCast = [&](Type out, Value in) -> Value {
    Location loc = loop.getLoc();
    if (loop.getUnsignedCmp()) {
      return arith::IndexCastUIOp::create(rewriter, loc, out, in);
    }
    return arith::IndexCastOp::create(rewriter, loc, out, in);
  };

  // We place the bound casts at the top level of the affine scope so that they
  // are identified as valid affine symbols.

  Region *scope = affine::getAffineScope(loop);
  Operation *anchor = loop;
  while (anchor->getParentRegion() != scope)
    anchor = anchor->getParentOp();
  rewriter.setInsertionPoint(anchor);

  Value newLb = createIndexCast(rewriter.getIndexType(), lb);
  Value newUb = createIndexCast(rewriter.getIndexType(), ub);
  Value newStep = createIndexCast(rewriter.getIndexType(), step);

  rewriter.modifyOpInPlace(loop, [&] {
    loop.setLowerBound(newLb);
    loop.setUpperBound(newUb);
    loop.setStep(newStep);

    Value originalIV = loop.getInductionVar();
    Value iv = loop.getBody()->insertArgument(
        (unsigned)0, rewriter.getIndexType(), loop.getLoc());

    rewriter.setInsertionPointToStart(loop.getBody());
    Value castIV = createIndexCast(originalType, iv);
    rewriter.replaceAllUsesWith(originalIV, castIV);

    // Original induction var is now at index 1.
    loop.getBody()->eraseArgument(1);
  });
}

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

void SCFToAffinePass::runOnOperation() {
  MLIRContext &ctx = getContext();
  RewritePatternSet patterns(&ctx);
  populateSCFToAffineConversionPatterns(patterns);

  (void)applyPatternsGreedily(getOperation(), std::move(patterns));
}

} // namespace

//===----------------------------------------------------------------------===//
// API
//===----------------------------------------------------------------------===//

void mlir::populateSCFToAffineConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<ForOpRewrite>(patterns.getContext());
}
