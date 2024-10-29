//===- MathToEmitC.cpp - Math to EmitC  Patterns ----------------*- C++ -*-===//
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
class LowerToEmitCCallOpaque : public OpRewritePattern<OpType> {
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
  if (!llvm::all_of(op->getOperandTypes(), llvm::IsaPred<Float32Type, Float64Type>)||
      !llvm::all_of(op->getResultTypes(),llvm::IsaPred<Float32Type, Float64Type>))
    return rewriter.notifyMatchFailure(op.getLoc(), "expected all operands and results to be of type f32 or f64");
  rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
      op, op.getType(), calleeStr, op->getOperands());
  return success();
}

} // namespace

// Populates patterns to replace `math` operations with `emitc.call_opaque`,
// using function names consistent with those in <math.h>.
void mlir::populateConvertMathToEmitCPatterns(RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  patterns.insert<LowerToEmitCCallOpaque<math::FloorOp>>(context, "floorf");
  patterns.insert<LowerToEmitCCallOpaque<math::RoundOp>>(context, "roundf");
  patterns.insert<LowerToEmitCCallOpaque<math::ExpOp>>(context, "expf");
  patterns.insert<LowerToEmitCCallOpaque<math::CosOp>>(context, "cosf");
  patterns.insert<LowerToEmitCCallOpaque<math::SinOp>>(context, "sinf");
  patterns.insert<LowerToEmitCCallOpaque<math::AcosOp>>(context, "acosf");
  patterns.insert<LowerToEmitCCallOpaque<math::AsinOp>>(context, "asinf");
  patterns.insert<LowerToEmitCCallOpaque<math::Atan2Op>>(context, "atan2f");
  patterns.insert<LowerToEmitCCallOpaque<math::CeilOp>>(context, "ceilf");
  patterns.insert<LowerToEmitCCallOpaque<math::AbsFOp>>(context, "fabsf");
  patterns.insert<LowerToEmitCCallOpaque<math::PowFOp>>(context, "powf");
}
