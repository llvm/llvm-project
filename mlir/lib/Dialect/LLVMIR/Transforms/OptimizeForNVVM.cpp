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
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace NVVM {
#define GEN_PASS_DEF_NVVMOPTIMIZEFORTARGET
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h.inc"
} // namespace NVVM
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

// Replaces sitofp or uitofp on src types no wider than the dst type mantissa
// with a faster combination of bit ops and add/sub.
template <typename OpTy> // OpTy should be LLVM::SIToFPOp or LLVM::UIToFPOp.
struct ExpandIToFP : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

private:
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override;
};

struct NVVMOptimizeForTarget
    : public NVVM::impl::NVVMOptimizeForTargetBase<NVVMOptimizeForTarget> {
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

template <typename OpTy>
LogicalResult
ExpandIToFP<OpTy>::matchAndRewrite(OpTy op, PatternRewriter &rewriter) const {
  Type srcType = op.getOperand().getType();
  auto intType = dyn_cast<IntegerType>(getElementTypeOrSelf(srcType));
  if (!intType)
    return rewriter.notifyMatchFailure(op, "src type is not integer");
  Type dstType = op.getType();
  auto floatType = dyn_cast<FloatType>(getElementTypeOrSelf(dstType));
  if (!floatType)
    return rewriter.notifyMatchFailure(op, "dst type is not float");

  // Mantissa width includes the integer bit, e.g. 24 for fp32.
  auto mantissaWidth = floatType.getFPMantissaWidth();
  if (mantissaWidth < 2)
    return rewriter.notifyMatchFailure(op, "mantissa is less than 2 bits");
  auto intWidth = intType.getWidth();
  if (intWidth > mantissaWidth)
    return rewriter.notifyMatchFailure(op, "src is wider than dst mantissa");

  Type extType = IntegerType::get(rewriter.getContext(), floatType.getWidth(),
                                  intType.getSignedness());
  if (ShapedType shapedType = dyn_cast<ShapedType>(srcType))
    extType = shapedType.clone(extType);
  auto getAttr = [&](APInt value) -> TypedAttr {
    if (ShapedType shapedType = dyn_cast<ShapedType>(extType))
      return DenseElementsAttr::get(shapedType, value);
    return IntegerAttr::get(extType, value);
  };
  ImplicitLocOpBuilder builder(op.getLoc(), rewriter);

  if (intWidth == mantissaWidth) {
    if (std::is_same_v<OpTy, LLVM::UIToFPOp>) {
      return rewriter.notifyMatchFailure(
          op, "unsigned src is as wide as dst mantissa");
    }
    // Create a float bit-pattern with zero biased-exponent and zero mantissa.
    APFloat::integerPart intPart = 1ull << (mantissaWidth - 1);
    APFloat floatBits(floatType.getFloatSemantics(), intPart);
    if (floatBits.bitcastToAPInt()[mantissaWidth - 1])
      return rewriter.notifyMatchFailure(op, "bias exponent lsb bit is set");
    TypedAttr intAttr = getAttr(floatBits.bitcastToAPInt());

    // Combine zero-extended src and float bit-pattern. The msb of src becomes
    // the lsb of the exponent.
    Value zext = builder.create<LLVM::ZExtOp>(extType, op.getOperand());
    Value intConst = builder.create<LLVM::ConstantOp>(intAttr);
    Value pattern = builder.create<LLVM::OrOp>(zext, intConst);

    // Mask the exponent-lsb and the mantissa to get two separate values.
    auto mask = APInt::getBitsSetFrom(floatType.getWidth(), mantissaWidth - 1);
    Value exponentMask = builder.create<LLVM::ConstantOp>(getAttr(mask));
    Value mantissaMask = builder.create<LLVM::ConstantOp>(getAttr(mask - 1));
    Value exponentAnd = builder.create<LLVM::AndOp>(pattern, exponentMask);
    Value mantissaAnd = builder.create<LLVM::AndOp>(pattern, mantissaMask);

    // Bitcast these values to float and subtract or add them.
    Value exponentCast = builder.create<LLVM::BitcastOp>(dstType, exponentAnd);
    Value mantissaCast = builder.create<LLVM::BitcastOp>(dstType, mantissaAnd);
    rewriter.replaceOpWithNewOp<LLVM::FSubOp>(op, mantissaCast, exponentCast);
    return success();
  }

  // Create a float with zero biased-exponent and msb-set mantissa.
  APFloat::integerPart intPart = 3ull << (mantissaWidth - 2);
  APFloat floatBits(floatType.getFloatSemantics(), intPart);
  TypedAttr intAttr = getAttr(floatBits.bitcastToAPInt());
  TypedAttr floatAttr = FloatAttr::get(floatType, floatBits);
  if (ShapedType shapedType = dyn_cast<ShapedType>(dstType))
    floatAttr = DenseElementsAttr::get(shapedType, floatAttr);

  // Add extended src and bit-pattern of float, then subtract float.
  using ExtOp = std::conditional_t<std::is_same_v<OpTy, LLVM::SIToFPOp>,
                                   LLVM::SExtOp, LLVM::ZExtOp>;
  Value ext = builder.create<ExtOp>(extType, op.getOperand());
  Value intConst = builder.create<LLVM::ConstantOp>(intAttr);
  Value add = builder.create<LLVM::AddOp>(ext, intConst);
  Value bitcast = builder.create<LLVM::BitcastOp>(dstType, add);
  Value floatConst = builder.create<LLVM::ConstantOp>(floatAttr);
  rewriter.replaceOpWithNewOp<LLVM::FSubOp>(op, bitcast, floatConst);
  return success();
}

void NVVMOptimizeForTarget::runOnOperation() {
  MLIRContext *ctx = getOperation()->getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<ExpandDivF16, ExpandIToFP<LLVM::SIToFPOp>,
               ExpandIToFP<LLVM::UIToFPOp>>(ctx);

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<Pass> NVVM::createOptimizeForTargetPass() {
  return std::make_unique<NVVMOptimizeForTarget>();
}
