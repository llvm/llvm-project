//===- MathToEmitC.cpp - Math to EmitC Pass Implementation ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MathToEmitC/MathToEmitC.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
template <typename OpType>
class LowerToEmitCCallOpaque : public mlir::OpRewritePattern<OpType> {
  std::string calleeStr;

public:
  LowerToEmitCCallOpaque(MLIRContext *context, std::string calleeStr)
      : OpRewritePattern<OpType>(context), calleeStr(std::move(calleeStr)) {}

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override;
};

template <typename OpType>
LogicalResult LowerToEmitCCallOpaque<OpType>::matchAndRewrite(
    OpType op, PatternRewriter &rewriter) const {
  auto actualOp = mlir::cast<OpType>(op);
  if (!llvm::all_of(
          actualOp->getOperands(),
          [](Value operand) { return isa<FloatType>(operand.getType()); }) ||
      !llvm::all_of(actualOp->getResultTypes(),
                    [](mlir::Type type) { return isa<FloatType>(type); })) {
    op.emitError("non-float types are not supported");
    return mlir::failure();
  }
  mlir::StringAttr callee = rewriter.getStringAttr(calleeStr);
  rewriter.replaceOpWithNewOp<mlir::emitc::CallOpaqueOp>(
      actualOp, actualOp.getType(), callee, actualOp->getOperands());
  return mlir::success();
}

} // namespace

// Populates patterns to replace `math` operations with `emitc.call_opaque`,
// using function names consistent with those in <math.h>.
void mlir::populateConvertMathToEmitCPatterns(RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  patterns.insert<LowerToEmitCCallOpaque<math::FloorOp>>(context, "floor");
  patterns.insert<LowerToEmitCCallOpaque<math::RoundEvenOp>>(context, "rint");
  patterns.insert<LowerToEmitCCallOpaque<math::ExpOp>>(context, "exp");
  patterns.insert<LowerToEmitCCallOpaque<math::CosOp>>(context, "cos");
  patterns.insert<LowerToEmitCCallOpaque<math::SinOp>>(context, "sin");
  patterns.insert<LowerToEmitCCallOpaque<math::AcosOp>>(context, "acos");
  patterns.insert<LowerToEmitCCallOpaque<math::AsinOp>>(context, "asin");
  patterns.insert<LowerToEmitCCallOpaque<math::Atan2Op>>(context, "atan2");
  patterns.insert<LowerToEmitCCallOpaque<math::CeilOp>>(context, "ceil");
  patterns.insert<LowerToEmitCCallOpaque<math::AbsFOp>>(context, "fabs");
  patterns.insert<LowerToEmitCCallOpaque<math::PowFOp>>(context, "pow");
}
