//===- OptimizeForNVVM.cpp - Optimize LLVM IR for NVVM ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/Transforms/OptimizeForNVVM.h"

#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace LLVM {
#define GEN_PASS_DEF_NVVMOPTIMIZEFORTARGETPASS
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h.inc"
} // namespace LLVM
} // namespace mlir

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
    : public LLVM::impl::NVVMOptimizeForTargetPassBase<NVVMOptimizeForTarget> {
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
  Value lhs = LLVM::FPExtOp::create(rewriter, loc, f32Type, op.getLhs());
  Value rhs = LLVM::FPExtOp::create(rewriter, loc, f32Type, op.getRhs());

  // float rcp = rcp.approx.ftz.f32(rhs), approx = lhs * rcp.
  Value rcp = NVVM::RcpApproxFtzF32Op::create(rewriter, loc, f32Type, rhs);
  Value approx = LLVM::FMulOp::create(rewriter, loc, lhs, rcp);

  // Refine the approximation with one Newton iteration:
  // float refined = approx + (lhs - approx * rhs) * rcp;
  Value err = LLVM::FMAOp::create(
      rewriter, loc, approx, LLVM::FNegOp::create(rewriter, loc, rhs), lhs);
  Value refined = LLVM::FMAOp::create(rewriter, loc, err, rcp, approx);

  // Use refined value if approx is normal (exponent neither all 0 or all 1).
  Value mask = LLVM::ConstantOp::create(
      rewriter, loc, i32Type, rewriter.getUI32IntegerAttr(0x7f800000));
  Value cast = LLVM::BitcastOp::create(rewriter, loc, i32Type, approx);
  Value exp = LLVM::AndOp::create(rewriter, loc, i32Type, cast, mask);
  Value zero = LLVM::ConstantOp::create(rewriter, loc, i32Type,
                                        rewriter.getUI32IntegerAttr(0));
  Value pred = LLVM::OrOp::create(
      rewriter, loc,
      LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::eq, exp, zero),
      LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::eq, exp, mask));
  Value result =
      LLVM::SelectOp::create(rewriter, loc, f32Type, pred, approx, refined);

  // Replace with trucation back to fp16.
  rewriter.replaceOpWithNewOp<LLVM::FPTruncOp>(op, op.getType(), result);

  return success();
}

void NVVMOptimizeForTarget::runOnOperation() {
  MLIRContext *ctx = getOperation()->getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<ExpandDivF16>(ctx);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}
