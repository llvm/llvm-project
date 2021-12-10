//===- ConvertConst.cpp - Quantizes constant ops --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Quant/Passes.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/Quant/QuantizeUtils.h"
#include "mlir/Dialect/Quant/UniformSupport.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::quant;

namespace {
struct ConvertConstPass : public QuantConvertConstBase<ConvertConstPass> {
  void runOnFunction() override;
};

struct QuantizedConstRewrite : public OpRewritePattern<QuantizeCastOp> {
  using OpRewritePattern<QuantizeCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(QuantizeCastOp qbarrier,
                                PatternRewriter &rewriter) const override;
};

} // namespace

/// Matches a [constant] -> [qbarrier] where the qbarrier results type is
/// quantized and the operand type is quantizable.

LogicalResult
QuantizedConstRewrite::matchAndRewrite(QuantizeCastOp qbarrier,
                                       PatternRewriter &rewriter) const {
  Attribute value;

  // Is the operand a constant?
  if (!matchPattern(qbarrier.arg(), m_Constant(&value))) {
    return failure();
  }

  // Does the qbarrier convert to a quantized type. This will not be true
  // if a quantized type has not yet been chosen or if the cast to an equivalent
  // storage type is not supported.
  Type qbarrierResultType = qbarrier.getResult().getType();
  QuantizedType quantizedElementType =
      QuantizedType::getQuantizedElementType(qbarrierResultType);
  if (!quantizedElementType) {
    return failure();
  }
  if (!QuantizedType::castToStorageType(qbarrierResultType)) {
    return failure();
  }

  // Is the operand type compatible with the expressed type of the quantized
  // type? This will not be true if the qbarrier is superfluous (converts
  // from and to a quantized type).
  if (!quantizedElementType.isCompatibleExpressedType(
          qbarrier.arg().getType())) {
    return failure();
  }

  // Is the constant value a type expressed in a way that we support?
  if (!value.isa<FloatAttr, DenseElementsAttr, SparseElementsAttr>()) {
    return failure();
  }

  Type newConstValueType;
  auto newConstValue =
      quantizeAttr(value, quantizedElementType, newConstValueType);
  if (!newConstValue) {
    return failure();
  }

  // When creating the new const op, use a fused location that combines the
  // original const and the qbarrier that led to the quantization.
  auto fusedLoc = rewriter.getFusedLoc(
      {qbarrier.arg().getDefiningOp()->getLoc(), qbarrier.getLoc()});
  auto newConstOp = rewriter.create<arith::ConstantOp>(
      fusedLoc, newConstValueType, newConstValue);
  rewriter.replaceOpWithNewOp<StorageCastOp>(qbarrier, qbarrier.getType(),
                                             newConstOp);
  return success();
}

void ConvertConstPass::runOnFunction() {
  RewritePatternSet patterns(&getContext());
  auto func = getFunction();
  auto *context = &getContext();
  patterns.add<QuantizedConstRewrite>(context);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

std::unique_ptr<OperationPass<FuncOp>> mlir::quant::createConvertConstPass() {
  return std::make_unique<ConvertConstPass>();
}
