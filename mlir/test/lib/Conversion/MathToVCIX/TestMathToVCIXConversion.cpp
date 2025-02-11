//===- TestMathToVCIXConversion.cpp - Test conversion to VCIX ops ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/VCIXDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace {

/// Return number of extracts required to make input VectorType \vt legal and
/// also return thatlegal vector type.
/// For fixed vectors nothing special is needed. Scalable vectors are legalizes
/// according to LLVM's encoding:
/// https://lists.llvm.org/pipermail/llvm-dev/2020-October/145850.html
static std::pair<unsigned, VectorType> legalizeVectorType(const Type &type) {
  VectorType vt = cast<VectorType>(type);
  // To simplify test pass, avoid multi-dimensional vectors.
  if (!vt || vt.getRank() != 1)
    return {0, nullptr};

  if (!vt.isScalable())
    return {1, vt};

  Type eltTy = vt.getElementType();
  unsigned sew = 0;
  if (eltTy.isF32())
    sew = 32;
  else if (eltTy.isF64())
    sew = 64;
  else if (auto intTy = dyn_cast<IntegerType>(eltTy))
    sew = intTy.getWidth();
  else
    return {0, nullptr};

  unsigned eltCount = vt.getShape()[0];
  const unsigned lmul = eltCount * sew / 64;

  unsigned n = lmul > 8 ? llvm::Log2_32(lmul) - 2 : 1;
  return {n, VectorType::get({eltCount >> (n - 1)}, eltTy, {true})};
}

/// Replace math.cos(v) operation with vcix.v.iv(v).
struct MathCosToVCIX final : OpRewritePattern<math::CosOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::CosOp op,
                                PatternRewriter &rewriter) const override {
    const Type opType = op.getOperand().getType();
    auto [n, legalType] = legalizeVectorType(opType);
    if (!legalType)
      return rewriter.notifyMatchFailure(op, "cannot legalize type for RVV");
    Location loc = op.getLoc();
    Value vec = op.getOperand();
    Attribute immAttr = rewriter.getI32IntegerAttr(0);
    Attribute opcodeAttr = rewriter.getI64IntegerAttr(0);
    Value rvl = nullptr;
    if (legalType.isScalable())
      // Use arbitrary runtime vector length when vector type is scalable.
      // Proper conversion pass should take it from the IR.
      rvl = rewriter.create<arith::ConstantOp>(loc,
                                               rewriter.getI64IntegerAttr(9));
    Value res;
    if (n == 1) {
      res = rewriter.create<vcix::BinaryImmOp>(loc, legalType, opcodeAttr, vec,
                                               immAttr, rvl);
    } else {
      const unsigned eltCount = legalType.getShape()[0];
      Type eltTy = legalType.getElementType();
      Value zero = rewriter.create<arith::ConstantOp>(
          loc, eltTy, rewriter.getZeroAttr(eltTy));
      res = rewriter.create<vector::BroadcastOp>(loc, opType, zero /*dummy*/);
      for (unsigned i = 0; i < n; ++i) {
        Value extracted = rewriter.create<vector::ScalableExtractOp>(
            loc, legalType, vec, i * eltCount);
        Value v = rewriter.create<vcix::BinaryImmOp>(loc, legalType, opcodeAttr,
                                                     extracted, immAttr, rvl);
        res = rewriter.create<vector::ScalableInsertOp>(loc, v, res,
                                                        i * eltCount);
      }
    }
    rewriter.replaceOp(op, res);
    return success();
  }
};

// Replace math.sin(v) operation with vcix.v.sv(v, v).
struct MathSinToVCIX final : OpRewritePattern<math::SinOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::SinOp op,
                                PatternRewriter &rewriter) const override {
    const Type opType = op.getOperand().getType();
    auto [n, legalType] = legalizeVectorType(opType);
    if (!legalType)
      return rewriter.notifyMatchFailure(op, "cannot legalize type for RVV");
    Location loc = op.getLoc();
    Value vec = op.getOperand();
    Attribute opcodeAttr = rewriter.getI64IntegerAttr(0);
    Value rvl = nullptr;
    if (legalType.isScalable())
      // Use arbitrary runtime vector length when vector type is scalable.
      // Proper conversion pass should take it from the IR.
      rvl = rewriter.create<arith::ConstantOp>(loc,
                                               rewriter.getI64IntegerAttr(9));
    Value res;
    if (n == 1) {
      res = rewriter.create<vcix::BinaryOp>(loc, legalType, opcodeAttr, vec,
                                            vec, rvl);
    } else {
      const unsigned eltCount = legalType.getShape()[0];
      Type eltTy = legalType.getElementType();
      Value zero = rewriter.create<arith::ConstantOp>(
          loc, eltTy, rewriter.getZeroAttr(eltTy));
      res = rewriter.create<vector::BroadcastOp>(loc, opType, zero /*dummy*/);
      for (unsigned i = 0; i < n; ++i) {
        Value extracted = rewriter.create<vector::ScalableExtractOp>(
            loc, legalType, vec, i * eltCount);
        Value v = rewriter.create<vcix::BinaryOp>(loc, legalType, opcodeAttr,
                                                  extracted, extracted, rvl);
        res = rewriter.create<vector::ScalableInsertOp>(loc, v, res,
                                                        i * eltCount);
      }
    }
    rewriter.replaceOp(op, res);
    return success();
  }
};

// Replace math.tan(v) operation with vcix.v.sv(v, 0.0f).
struct MathTanToVCIX final : OpRewritePattern<math::TanOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::TanOp op,
                                PatternRewriter &rewriter) const override {
    const Type opType = op.getOperand().getType();
    auto [n, legalType] = legalizeVectorType(opType);
    Type eltTy = legalType.getElementType();
    if (!legalType)
      return rewriter.notifyMatchFailure(op, "cannot legalize type for RVV");
    Location loc = op.getLoc();
    Value vec = op.getOperand();
    Attribute opcodeAttr = rewriter.getI64IntegerAttr(0);
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, eltTy, rewriter.getZeroAttr(eltTy));
    Value rvl = nullptr;
    if (legalType.isScalable())
      // Use arbitrary runtime vector length when vector type is scalable.
      // Proper conversion pass should take it from the IR.
      rvl = rewriter.create<arith::ConstantOp>(loc,
                                               rewriter.getI64IntegerAttr(9));
    Value res;
    if (n == 1) {
      res = rewriter.create<vcix::BinaryOp>(loc, legalType, opcodeAttr, vec,
                                            zero, rvl);
    } else {
      const unsigned eltCount = legalType.getShape()[0];
      res = rewriter.create<vector::BroadcastOp>(loc, opType, zero /*dummy*/);
      for (unsigned i = 0; i < n; ++i) {
        Value extracted = rewriter.create<vector::ScalableExtractOp>(
            loc, legalType, vec, i * eltCount);
        Value v = rewriter.create<vcix::BinaryOp>(loc, legalType, opcodeAttr,
                                                  extracted, zero, rvl);
        res = rewriter.create<vector::ScalableInsertOp>(loc, v, res,
                                                        i * eltCount);
      }
    }
    rewriter.replaceOp(op, res);
    return success();
  }
};

// Replace math.log(v) operation with vcix.v.sv(v, 0).
struct MathLogToVCIX final : OpRewritePattern<math::LogOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::LogOp op,
                                PatternRewriter &rewriter) const override {
    const Type opType = op.getOperand().getType();
    auto [n, legalType] = legalizeVectorType(opType);
    if (!legalType)
      return rewriter.notifyMatchFailure(op, "cannot legalize type for RVV");
    Location loc = op.getLoc();
    Value vec = op.getOperand();
    Attribute opcodeAttr = rewriter.getI64IntegerAttr(0);
    Value rvl = nullptr;
    Value zeroInt = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
    if (legalType.isScalable())
      // Use arbitrary runtime vector length when vector type is scalable.
      // Proper conversion pass should take it from the IR.
      rvl = rewriter.create<arith::ConstantOp>(loc,
                                               rewriter.getI64IntegerAttr(9));
    Value res;
    if (n == 1) {
      res = rewriter.create<vcix::BinaryOp>(loc, legalType, opcodeAttr, vec,
                                            zeroInt, rvl);
    } else {
      const unsigned eltCount = legalType.getShape()[0];
      Type eltTy = legalType.getElementType();
      Value zero = rewriter.create<arith::ConstantOp>(
          loc, eltTy, rewriter.getZeroAttr(eltTy));
      res = rewriter.create<vector::BroadcastOp>(loc, opType, zero /*dummy*/);
      for (unsigned i = 0; i < n; ++i) {
        Value extracted = rewriter.create<vector::ScalableExtractOp>(
            loc, legalType, vec, i * eltCount);
        Value v = rewriter.create<vcix::BinaryOp>(loc, legalType, opcodeAttr,
                                                  extracted, zeroInt, rvl);
        res = rewriter.create<vector::ScalableInsertOp>(loc, v, res,
                                                        i * eltCount);
      }
    }
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct TestMathToVCIX
    : PassWrapper<TestMathToVCIX, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestMathToVCIX)

  StringRef getArgument() const final { return "test-math-to-vcix"; }

  StringRef getDescription() const final {
    return "Test lowering patterns that converts some vector operations to "
           "VCIX. Since DLA can implement VCIX instructions in completely "
           "different way, conversions of that test pass only lives here.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, math::MathDialect,
                    vcix::VCIXDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<MathCosToVCIX, MathSinToVCIX, MathTanToVCIX, MathLogToVCIX>(
        ctx);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

namespace test {
void registerTestMathToVCIXPass() { PassRegistration<TestMathToVCIX>(); }
} // namespace test
} // namespace mlir
