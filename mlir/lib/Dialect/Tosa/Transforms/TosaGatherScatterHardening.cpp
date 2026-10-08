//===- TosaGatherScatterHardening.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that clamps gather and scatter indices to the
// statically known bounds of their indexed tensors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace mlir {
namespace tosa {
#define GEN_PASS_DEF_TOSAGATHERSCATTERHARDENINGPASS
#include "mlir/Dialect/Tosa/Transforms/Passes.h.inc"
} // namespace tosa
} // namespace mlir

using namespace mlir;
using namespace mlir::tosa;

namespace {

/// Returns the effective upper bound when the indexed dimension is static.
static FailureOr<int64_t> getIndexUpperBound(Operation *op) {
  Value values = op->getOperand(0);
  auto valuesType = dyn_cast<RankedTensorType>(values.getType());
  if (!valuesType || valuesType.isDynamicDim(1)) {
    op->emitOpError("requires a statically known indexed dimension for "
                    "gather/scatter hardening");
    return failure();
  }

  auto indicesType = cast<ShapedType>(op->getOperand(1).getType());
  auto elementType = cast<IntegerType>(indicesType.getElementType());

  // The upper bound must be representable in the index element type.
  int64_t maxRepresentable =
      llvm::APInt::getSignedMaxValue(elementType.getWidth()).getSExtValue();
  return std::min(valuesType.getDimSize(1) - 1, maxRepresentable);
}

/// Returns whether the indices already have sufficiently restrictive bounds.
template <typename OuterOp, typename InnerOp>
static bool isAlreadyHardened(Value indices, int64_t requiredUpperBound) {
  static_assert(
      (std::is_same_v<OuterOp, tosa::MinimumOp> &&
       std::is_same_v<InnerOp, tosa::MaximumOp>) ||
          (std::is_same_v<OuterOp, tosa::MaximumOp> &&
           std::is_same_v<InnerOp, tosa::MinimumOp>),
      "expected a tosa::MinimumOp/tosa::MaximumOp pair in either order");

  auto outerOp = indices.getDefiningOp<OuterOp>();
  if (!outerOp)
    return false;

  // Either operand can be the bound, including when both are constants. A
  // constant match alone is not enough: try the other operand if it is unsafe.
  for (unsigned boundOperand = 0; boundOperand < 2; ++boundOperand) {
    llvm::APInt outerBound;
    if (!matchPattern(outerOp->getOperand(boundOperand),
                      m_ConstantInt(&outerBound)))
      continue;

    llvm::APInt requiredUpper(outerBound.getBitWidth(),
                              static_cast<uint64_t>(requiredUpperBound));
    // The outer bound must itself be in range, since it can override the inner
    // bound, e.g. minimum(maximum(x, 0), -1) would produce -1. This guarantees
    // the other operand is meeting the outer bound check (e.g. smaller or equal
    // to the required upper bound if outer op is a minimum). The inner
    // operation only needs to enforce the opposite bound.
    if (outerBound.isNegative() || outerBound.sgt(requiredUpper))
      continue;

    auto matchesInnerBound = [&](Value value) {
      llvm::APInt innerBound;
      if (!matchPattern(value, m_ConstantInt(&innerBound)))
        return false;
      return isa<tosa::MinimumOp>(outerOp) ? !innerBound.isNegative()
                                           : innerBound.sle(requiredUpper);
    };

    // Check whether other operand of the outer op is also a constant and is
    // meeting the inner bound check. No inner op involved and the result is
    // therefore completely within bound thanks to the earlier check.
    Value innerResult = outerOp->getOperand(1 - boundOperand);
    if (matchesInnerBound(innerResult))
      return true;
    // The other outer op operand is not a constant so check that the inner op
    // enforces inner bound check.
    if (auto innerOp = innerResult.getDefiningOp<InnerOp>())
      if (llvm::any_of(innerOp->getOperands(), matchesInnerBound))
        return true;
  }
  return false;
}

/// Creates a rank-two splat constant suitable for index broadcasting.
static Value createIndexBoundConstant(OpBuilder &builder, Location loc,
                                      IntegerType elementType, int64_t value) {
  auto type = RankedTensorType::get({1, 1}, elementType);
  auto valueAttr =
      IntegerAttr::get(elementType, llvm::APInt(elementType.getWidth(),
                                                static_cast<uint64_t>(value)));
  auto values = DenseElementsAttr::get(type, valueAttr);
  return tosa::ConstOp::create(builder, loc, type, values).getResult();
}

/// Independently hardens one gather or scatter operation's indices.
template <typename OpTy>
struct HardenIndexUsePattern final : OpRewritePattern<OpTy> {
  HardenIndexUsePattern(MLIRContext *context, bool &hardeningFailed)
      : OpRewritePattern<OpTy>(context), hardeningFailed(hardeningFailed) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    FailureOr<int64_t> upperBound = getIndexUpperBound(op.getOperation());
    if (failed(upperBound)) {
      hardeningFailed = true;
      return rewriter.notifyMatchFailure(
          op, "indexed dimension does not have a static upper bound");
    }

    Value indices = op->getOperand(1);
    if (isAlreadyHardened<tosa::MinimumOp, tosa::MaximumOp>(indices,
                                                            *upperBound) ||
        isAlreadyHardened<tosa::MaximumOp, tosa::MinimumOp>(indices,
                                                            *upperBound))
      return rewriter.notifyMatchFailure(op, "indices are already hardened");

    auto indicesType = cast<ShapedType>(indices.getType());
    auto elementType = cast<IntegerType>(indicesType.getElementType());
    Value lowerBound =
        createIndexBoundConstant(rewriter, op.getLoc(), elementType, 0);
    Value upperBoundValue = createIndexBoundConstant(rewriter, op.getLoc(),
                                                     elementType, *upperBound);
    Value nonNegativeIndices =
        tosa::MaximumOp::create(rewriter, op.getLoc(), indices.getType(),
                                indices, lowerBound)
            .getResult();
    Value clampedIndices =
        tosa::MinimumOp::create(rewriter, op.getLoc(), indices.getType(),
                                nonNegativeIndices, upperBoundValue)
            .getResult();

    rewriter.modifyOpInPlace(
        op, [&] { op->setOperand(/*indices=*/1, clampedIndices); });
    return success();
  }

private:
  bool &hardeningFailed;
};

struct TosaGatherScatterHardeningPass
    : public tosa::impl::TosaGatherScatterHardeningPassBase<
          TosaGatherScatterHardeningPass> {
  using Base::Base;

  void runOnOperation() override {
    bool hardeningFailed = false;
    RewritePatternSet patterns(&getContext());
    patterns.add<HardenIndexUsePattern<tosa::GatherOp>,
                 HardenIndexUsePattern<tosa::ScatterOp>>(&getContext(),
                                                         hardeningFailed);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))) ||
        hardeningFailed)
      signalPassFailure();
  }
};

} // namespace
