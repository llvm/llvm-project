//===- TosaReduceReshapes.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace tosa {
#define GEN_PASS_DEF_TOSAREDUCERESHAPES
#include "mlir/Dialect/Tosa/Transforms/Passes.h.inc"
} // namespace tosa
} // namespace mlir

using namespace mlir;
using namespace mlir::tosa;

namespace {
/// Computes a higher-rank operand shape that preserves broadcasting.
///
/// For example, expanding an operator result from [6, 4] to [2, 3, 4] gives
/// the reassociation {{0, 1}, {2}}. An operand with shape [6, 4] expands to
/// [2, 3, 4] because both of its dimensions match the source shape. A broadcast
/// operand with shape [1, 4] instead expands to [1, 1, 4]: the source dimension
/// of size 6 is split into two unit dimensions so that the operand continues to
/// broadcast over the corresponding target dimensions of sizes 2 and 3.
///
/// An operand dimension must therefore either match its source dimension or be
/// one. Any other dimension is incompatible with the source result shape.
static FailureOr<SmallVector<int64_t>> getExpandedBroadcastShape(
    RankedTensorType operandType, RankedTensorType sourceType,
    RankedTensorType targetType, ArrayRef<ReassociationIndices> reassociation) {
  SmallVector<int64_t> expandedShape;
  // A scalar has no source dimensions, so the loop below would leave its
  // operand scalar. Use the target shape to match the replacement op's rank.
  if (sourceType.getRank() == 0) {
    return SmallVector<int64_t>(targetType.getShape());
  }

  for (auto [sourceIndex, targetIndices] : llvm::enumerate(reassociation)) {
    if (operandType.getDimSize(sourceIndex) ==
        sourceType.getDimSize(sourceIndex)) {
      for (int64_t targetIndex : targetIndices) {
        expandedShape.push_back(targetType.getDimSize(targetIndex));
      }
    } else if (operandType.getDimSize(sourceIndex) == 1) {
      expandedShape.append(targetIndices.size(), 1);
    } else {
      return failure();
    }
  }
  return expandedShape;
}

static bool isSupportedProducer(Operation *op) {
  bool preservesRank = op->hasTrait<OpTrait::SameOperandsAndResultRank>() ||
                       op->hasTrait<OpTrait::SameOperandsAndResultShape>();

  bool isSupportedCast = false;
  if (auto castOp = dyn_cast<tosa::CastOp>(op)) {
    // Only ordinary scalar casts are independent of the tensor shape.
    auto inputType = cast<ShapedType>(castOp.getInput().getType());
    auto outputType = cast<ShapedType>(castOp.getOutput().getType());
    isSupportedCast = inputType.getElementType().isIntOrIndexOrFloat() &&
                      outputType.getElementType().isIntOrIndexOrFloat();
  }

  return preservesRank &&
         (isSupportedCast ||
          op->hasTrait<OpTrait::tosa::TosaElementwiseOperator>());
}

/// Whether reshaping `operand` avoids leaving an additional reshape. An
/// existing reshape must have one use so that it can be removed after the
/// inserted reshape folds. Match the constant cases handled by ReshapeOp::fold.
static bool canFoldOperandReshape(Value operand, RankedTensorType operandType) {
  if (isa_and_nonnull<tosa::ReshapeOp>(operand.getDefiningOp())) {
    return operand.hasOneUse();
  }

  auto constOp = operand.getDefiningOp<tosa::ConstOp>();
  if (!constOp || !operandType.getElementType().isIntOrIndexOrFloat()) {
    return false;
  }

  ElementsAttr values = constOp.getValues();
  if (isa<DenseResourceElementsAttr>(values)) {
    return true;
  }
  auto denseValues = dyn_cast<DenseElementsAttr>(values);
  return denseValues && (denseValues.isSplat() || operand.hasOneUse());
}

struct PropagateExpandingReshape : public OpRewritePattern<tosa::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ReshapeOp reshape,
                                PatternRewriter &rewriter) const override {
    if (!reshape.getInput1().hasOneUse()) {
      return failure();
    }

    Operation *op = reshape.getInput1().getDefiningOp();
    if (!op || !isSupportedProducer(op) || op->getNumResults() != 1) {
      return failure();
    }

    auto sourceType = dyn_cast<RankedTensorType>(reshape.getInput1().getType());
    auto targetType = dyn_cast<RankedTensorType>(reshape.getOutput().getType());
    if (!sourceType || !targetType || !sourceType.hasStaticShape() ||
        !targetType.hasStaticShape() ||
        targetType.getRank() <= sourceType.getRank()) {
      return failure();
    }

    auto reassociation =
        getReassociationIndicesForReshape(sourceType, targetType);
    if (!reassociation) {
      return failure();
    }

    unsigned nonFoldableOperands = 0;
    SmallVector<RankedTensorType> expandedTypes;
    for (Value operand : op->getOperands()) {
      auto operandType = dyn_cast<RankedTensorType>(operand.getType());
      if (!operandType || !operandType.hasStaticShape()) {
        return failure();
      }
      if (!canFoldOperandReshape(operand, operandType) &&
          ++nonFoldableOperands > 1) {
        return failure();
      }
      FailureOr<SmallVector<int64_t>> expandedShape = getExpandedBroadcastShape(
          operandType, sourceType, targetType, *reassociation);
      if (failed(expandedShape)) {
        return failure();
      }
      expandedTypes.push_back(operandType.clone(*expandedShape));
    }

    SmallVector<Value> expandedOperands;
    for (auto [operand, expandedType] :
         llvm::zip_equal(op->getOperands(), expandedTypes)) {
      Value shape =
          getTosaConstShape(rewriter, op->getLoc(), expandedType.getShape());
      expandedOperands.push_back(tosa::ReshapeOp::create(
          rewriter, op->getLoc(), expandedType, operand, shape));
    }

    Value expandedResult =
        rewriter
            .create(op->getLoc(), op->getName().getIdentifier(),
                    expandedOperands, targetType, op->getAttrs())
            ->getResult(0);
    rewriter.replaceOp(reshape, expandedResult);
    rewriter.eraseOp(op);
    return success();
  }
};

struct TosaReduceReshapes final
    : public tosa::impl::TosaReduceReshapesBase<TosaReduceReshapes> {
  using Base::Base;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<PropagateExpandingReshape>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace
