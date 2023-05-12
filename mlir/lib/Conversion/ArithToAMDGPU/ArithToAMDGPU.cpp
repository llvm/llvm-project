//===- ArithToAMDGPU.cpp - Arith to AMDGPU dialect conversion ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithToAMDGPU/ArithToAMDGPU.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_ARITHTOAMDGPUCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct ArithToAMDGPUConversionPass final
    : impl::ArithToAMDGPUConversionPassBase<ArithToAMDGPUConversionPass> {
  using impl::ArithToAMDGPUConversionPassBase<
      ArithToAMDGPUConversionPass>::ArithToAMDGPUConversionPassBase;

  void runOnOperation() override;
};

struct ExtfOnFloat8RewritePattern final
    : public OpRewritePattern<arith::ExtFOp> {
  using OpRewritePattern<arith::ExtFOp>::OpRewritePattern;

  LogicalResult match(arith::ExtFOp op) const override;
  void rewrite(arith::ExtFOp op, PatternRewriter &rewriter) const override;
};

struct TruncfToFloat8RewritePattern final
    : public OpRewritePattern<arith::TruncFOp> {
  using OpRewritePattern<arith::TruncFOp>::OpRewritePattern;

  LogicalResult match(arith::TruncFOp op) const override;
  void rewrite(arith::TruncFOp op, PatternRewriter &rewriter) const override;
};
} // end namespace

static Value castF32To(Type elementType, Value f32, Location loc,
                       PatternRewriter &rewriter) {
  if (elementType.isF32())
    return f32;
  if (elementType.getIntOrFloatBitWidth() < 32)
    return rewriter.create<arith::TruncFOp>(loc, elementType, f32);
  if (elementType.getIntOrFloatBitWidth() > 32)
    return rewriter.create<arith::ExtFOp>(loc, elementType, f32);
  llvm_unreachable("The only 32-bit float type is f32");
}

LogicalResult ExtfOnFloat8RewritePattern::match(arith::ExtFOp op) const {
  Type inType = op.getIn().getType();
  if (auto inVecType = inType.dyn_cast<VectorType>()) {
    if (inVecType.isScalable())
      return failure();
    if (inVecType.getShape().size() > 1)
      // Multi-dimensional vectors are currently unsupported.
      return failure();
    inType = inVecType.getElementType();
  }
  return success(inType.isFloat8E5M2FNUZ() || inType.isFloat8E4M3FNUZ());
}

void ExtfOnFloat8RewritePattern::rewrite(arith::ExtFOp op,
                                         PatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Value in = op.getIn();
  Type outElemType = getElementTypeOrSelf(op.getOut().getType());
  if (!in.getType().isa<VectorType>()) {
    Value asFloat = rewriter.create<amdgpu::ExtPackedFp8Op>(
        loc, rewriter.getF32Type(), in, 0);
    Value result = castF32To(outElemType, asFloat, loc, rewriter);
    return rewriter.replaceOp(op, result);
  }
  VectorType inType = in.getType().cast<VectorType>();
  int64_t numElements = inType.getNumElements();
  Value zero = rewriter.createOrFold<arith::ConstantOp>(
      loc, outElemType, rewriter.getFloatAttr(outElemType, 0.0));
  Value result =
      rewriter.createOrFold<vector::SplatOp>(loc, op.getOut().getType(), zero);
  if (inType.getShape().empty()) {
    Value scalarIn = rewriter.create<vector::ExtractElementOp>(loc, in);
    // Recurse to send the 0-D vector case to the 1-D vector case
    Value scalarExt =
        rewriter.create<arith::ExtFOp>(loc, outElemType, scalarIn);
    result = rewriter.create<vector::InsertElementOp>(loc, scalarExt, zero);
    return rewriter.replaceOp(op, result);
  }
  for (int64_t i = 0; i < numElements; i += 4) {
    int64_t elemsThisOp = std::min(numElements, i + 4) - i;
    Value inSlice = rewriter.create<vector::ExtractStridedSliceOp>(
        loc, in, i, elemsThisOp, 1);
    for (int64_t j = 0; j < elemsThisOp; ++j) {
      Value asFloat = rewriter.create<amdgpu::ExtPackedFp8Op>(
          loc, rewriter.getF32Type(), inSlice, j);
      Value asType = castF32To(outElemType, asFloat, loc, rewriter);
      result = rewriter.create<vector::InsertElementOp>(
          loc, asType, result,
          rewriter.createOrFold<arith::ConstantIndexOp>(loc, i + j));
    }
  }
  rewriter.replaceOp(op, result);
}

static Value castToF32(Value value, Location loc, PatternRewriter &rewriter) {
  Type type = value.getType();
  if (type.isF32())
    return value;
  if (type.getIntOrFloatBitWidth() < 32)
    return rewriter.create<arith::ExtFOp>(loc, rewriter.getF32Type(), value);
  if (type.getIntOrFloatBitWidth() > 32)
    return rewriter.create<arith::TruncFOp>(loc, rewriter.getF32Type(), value);
  llvm_unreachable("The only 32-bit float type is f32");
}

LogicalResult TruncfToFloat8RewritePattern::match(arith::TruncFOp op) const {
  Type outType = op.getOut().getType();
  if (auto outVecType = outType.dyn_cast<VectorType>()) {
    if (outVecType.isScalable())
      return failure();
    if (outVecType.getShape().size() > 1)
      // Multi-dimensional vectors are currently unsupported.
      return failure();
    outType = outVecType.getElementType();
  }
  return success(outType.isFloat8E5M2FNUZ() || outType.isFloat8E4M3FNUZ());
}

void TruncfToFloat8RewritePattern::rewrite(arith::TruncFOp op,
                                           PatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Value in = op.getIn();
  Type outElemType = getElementTypeOrSelf(op.getOut().getType());
  VectorType truncResType = VectorType::get(4, outElemType);
  if (!in.getType().isa<VectorType>()) {
    Value asFloat = castToF32(in, loc, rewriter);
    Value asF8s = rewriter.create<amdgpu::PackedTrunc2xFp8Op>(
        loc, truncResType, asFloat, /*sourceB=*/nullptr, 0,
        /*existing=*/nullptr);
    Value result = rewriter.create<vector::ExtractElementOp>(
        loc, asF8s, rewriter.createOrFold<arith::ConstantIndexOp>(loc, 0));
    return rewriter.replaceOp(op, result);
  }
  VectorType outType = op.getOut().getType().cast<VectorType>();
  int64_t numElements = outType.getNumElements();
  Value zero = rewriter.createOrFold<arith::ConstantOp>(
      loc, outElemType, rewriter.getFloatAttr(outElemType, 0.0));
  Value result = rewriter.createOrFold<vector::SplatOp>(loc, outType, zero);
  if (outType.getShape().empty()) {
    Value scalarIn = rewriter.create<vector::ExtractElementOp>(loc, in);
    // Recurse to send the 0-D vector case to the 1-D vector case
    Value scalarTrunc =
        rewriter.create<arith::TruncFOp>(loc, outElemType, scalarIn);
    result = rewriter.create<vector::InsertElementOp>(loc, scalarTrunc, zero);
    return rewriter.replaceOp(op, result);
  }

  for (int64_t i = 0; i < numElements; i += 4) {
    int64_t elemsThisOp = std::min(numElements, i + 4) - i;
    Value thisResult = nullptr;
    for (int64_t j = 0; j < elemsThisOp; j += 2) {
      Value elemA = rewriter.create<vector::ExtractElementOp>(
          loc, in, rewriter.create<arith::ConstantIndexOp>(loc, i + j));
      Value asFloatA = castToF32(elemA, loc, rewriter);
      Value asFloatB = nullptr;
      if (j + 1 < elemsThisOp) {
        Value elemB = rewriter.create<vector::ExtractElementOp>(
            loc, in,
            rewriter.createOrFold<arith::ConstantIndexOp>(loc, i + j + 1));
        asFloatB = castToF32(elemB, loc, rewriter);
      }
      thisResult = rewriter.create<amdgpu::PackedTrunc2xFp8Op>(
          loc, truncResType, asFloatA, asFloatB, j / 2, thisResult);
    }
    if (elemsThisOp < 4)
      thisResult = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, thisResult, 0, elemsThisOp, 1);
    result = rewriter.create<vector::InsertStridedSliceOp>(loc, thisResult,
                                                           result, i, 1);
  }
  rewriter.replaceOp(op, result);
}

void mlir::arith::populateArithToAMDGPUConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ExtfOnFloat8RewritePattern, TruncfToFloat8RewritePattern>(
      patterns.getContext());
}

void ArithToAMDGPUConversionPass::runOnOperation() {
  Operation *op = getOperation();
  RewritePatternSet patterns(op->getContext());
  arith::populateArithToAMDGPUConversionPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
    return signalPassFailure();
}
