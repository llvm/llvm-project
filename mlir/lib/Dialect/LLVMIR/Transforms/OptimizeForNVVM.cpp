//===- OptimizeForNVVM.cpp - Optimize LLVM IR for NVVM ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/Transforms/OptimizeForNVVM.h"
#include "PassDetail.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
// Replaces fdiv on fp16 with fp32 multiplication with reciprocal plus one
// (conditional) Newton iteration.
//
// This as accurate as promoting the division to fp32 in the NVPTX backend, but
// faster because it performs less Newton iterations, avoids the slow path
// for e.g. denormals, and allows reuse of the reciprocal for multiple divisions
// by the same divisor.
struct ExpandDivF16 : public OpRewritePattern<LLVM::FDivOp> {
  using OpRewritePattern<LLVM::FDivOp>::OpRewritePattern;

private:
  LogicalResult matchAndRewrite(LLVM::FDivOp op,
                                PatternRewriter &rewriter) const override;
};

struct NVVMOptimizeForTarget
    : public NVVMOptimizeForTargetBase<NVVMOptimizeForTarget> {
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<NVVM::NVVMDialect>();
  }
};
} // namespace

LogicalResult ExpandDivF16::matchAndRewrite(LLVM::FDivOp op,
                                            PatternRewriter &rewriter) const {
  if (!op.getType().isF16())
    return rewriter.notifyMatchFailure(op, "not f16");
  Location loc = op.getLoc();

  Type f32Type = rewriter.getF32Type();
  Type i32Type = rewriter.getI32Type();

  // Extend lhs and rhs to fp32.
  Value lhs = rewriter.create<LLVM::FPExtOp>(loc, f32Type, op.getLhs());
  Value rhs = rewriter.create<LLVM::FPExtOp>(loc, f32Type, op.getRhs());

  // float rcp = rcp.approx.ftz.f32(rhs), approx = lhs * rcp.
  Value rcp = rewriter.create<NVVM::RcpApproxFtzF32Op>(loc, f32Type, rhs);
  Value approx = rewriter.create<LLVM::FMulOp>(loc, lhs, rcp);

  // Refine the approximation with one Newton iteration:
  // float refined = approx + (lhs - approx * rhs) * rcp;
  Value err = rewriter.create<LLVM::FMAOp>(
      loc, approx, rewriter.create<LLVM::FNegOp>(loc, rhs), lhs);
  Value refined = rewriter.create<LLVM::FMAOp>(loc, err, rcp, approx);

  // Use refined value if approx is normal (exponent neither all 0 or all 1).
  Value mask = rewriter.create<LLVM::ConstantOp>(
      loc, i32Type, rewriter.getUI32IntegerAttr(0x7f800000));
  Value cast = rewriter.create<LLVM::BitcastOp>(loc, i32Type, approx);
  Value exp = rewriter.create<LLVM::AndOp>(loc, i32Type, cast, mask);
  Value zero = rewriter.create<LLVM::ConstantOp>(
      loc, i32Type, rewriter.getUI32IntegerAttr(0));
  Value pred = rewriter.create<LLVM::OrOp>(
      loc,
      rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, exp, zero),
      rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, exp, mask));
  Value result =
      rewriter.create<LLVM::SelectOp>(loc, f32Type, pred, approx, refined);

  // Replace with trucation back to fp16.
  rewriter.replaceOpWithNewOp<LLVM::FPTruncOp>(op, op.getType(), result);

  return success();
}

void NVVMOptimizeForTarget::runOnOperation() {
  MLIRContext *ctx = getOperation()->getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<ExpandDivF16>(ctx);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<Pass> NVVM::createOptimizeForTargetPass() {
  return std::make_unique<NVVMOptimizeForTarget>();
}
