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

#include <algorithm>
#include <cstdint>

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
static bool isAlreadyHardened(Value indices, int64_t requiredUpperBound) {
  auto minimumOp = indices.getDefiningOp<tosa::MinimumOp>();
  if (!minimumOp)
    return false;

  Value maximumResult = minimumOp.getInput1();
  llvm::APInt upperBound;
  if (!matchPattern(minimumOp.getInput2(), m_ConstantInt(&upperBound))) {
    maximumResult = minimumOp.getInput2();
    if (!matchPattern(minimumOp.getInput1(), m_ConstantInt(&upperBound)))
      return false;
  }

  auto maximumOp = maximumResult.getDefiningOp<tosa::MaximumOp>();
  if (!maximumOp)
    return false;

  llvm::APInt lowerBound;
  if (!matchPattern(maximumOp.getInput2(), m_ConstantInt(&lowerBound)) &&
      !matchPattern(maximumOp.getInput1(), m_ConstantInt(&lowerBound)))
    return false;

  unsigned bitWidth = upperBound.getBitWidth();
  llvm::APInt requiredUpper(bitWidth,
                            static_cast<uint64_t>(requiredUpperBound));
  return lowerBound.getBitWidth() == bitWidth && !lowerBound.isNegative() &&
         !upperBound.isNegative() && upperBound.sle(requiredUpper);
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
    if (isAlreadyHardened(indices, *upperBound))
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
