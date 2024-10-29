//===- VectorToXeGPU.cpp - Convert vector to XeGPU dialect ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of vector operations to XeGPU dialect ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/VectorToXeGPU/VectorToXeGPU.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>
#include <optional>

namespace mlir {
#define GEN_PASS_DEF_CONVERTVECTORTOXEGPU
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

// Return true if value represents a zero constant.
static bool isZeroConstant(Value val) {
  auto constant = val.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;

  return TypeSwitch<Attribute, bool>(constant.getValue())
      .Case<FloatAttr>(
          [](auto floatAttr) { return floatAttr.getValue().isZero(); })
      .Case<IntegerAttr>(
          [](auto intAttr) { return intAttr.getValue().isZero(); })
      .Default([](auto) { return false; });
}

static LogicalResult storeLoadPreconditions(PatternRewriter &rewriter,
                                            Operation *op, VectorType vecTy) {
  // Validate only vector as the basic vector store and load ops guarantee
  // XeGPU-compatible memref source.
  unsigned vecRank = vecTy.getRank();
  if (!(vecRank == 1 || vecRank == 2))
    return rewriter.notifyMatchFailure(op, "Expects 1D or 2D vector");

  return success();
}

static LogicalResult transferPreconditions(PatternRewriter &rewriter,
                                           VectorTransferOpInterface xferOp) {
  if (xferOp.getMask())
    return rewriter.notifyMatchFailure(xferOp,
                                       "Masked transfer is not supported");

  auto srcTy = dyn_cast<MemRefType>(xferOp.getShapedType());
  if (!srcTy)
    return rewriter.notifyMatchFailure(xferOp, "Expects memref source");

  // Perform common data transfer checks.
  VectorType vecTy = xferOp.getVectorType();
  if (failed(storeLoadPreconditions(rewriter, xferOp, vecTy)))
    return failure();

  // Validate further transfer op semantics.
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(getStridesAndOffset(srcTy, strides, offset)) ||
      strides.back() != 1)
    return rewriter.notifyMatchFailure(
        xferOp, "Buffer must be contiguous in the innermost dimension");

  unsigned vecRank = vecTy.getRank();
  AffineMap map = xferOp.getPermutationMap();
  if (!map.isProjectedPermutation(/*allowZeroInResults=*/false))
    return rewriter.notifyMatchFailure(xferOp, "Unsupported permutation map");
  unsigned numInputDims = map.getNumInputs();
  for (AffineExpr expr : map.getResults().take_back(vecRank)) {
    auto dim = dyn_cast<AffineDimExpr>(expr);
    if (dim.getPosition() < (numInputDims - vecRank))
      return rewriter.notifyMatchFailure(
          xferOp, "Only the innermost dimensions can be accessed");
  }

  return success();
}

static xegpu::CreateNdDescOp
createNdDescriptor(PatternRewriter &rewriter, Location loc,
                   xegpu::TensorDescType descType, TypedValue<MemRefType> src,
                   Operation::operand_range offsets) {
  MemRefType srcTy = src.getType();
  auto [strides, offset] = getStridesAndOffset(srcTy);

  xegpu::CreateNdDescOp ndDesc;
  if (srcTy.hasStaticShape()) {
    ndDesc = rewriter.create<xegpu::CreateNdDescOp>(loc, descType, src,
                                                    getAsOpFoldResult(offsets));
  } else {
    // In case of any dynamic shapes, source's shape and strides have to be
    // explicitly provided.
    SmallVector<Value> sourceDims;
    unsigned srcRank = srcTy.getRank();
    for (unsigned i = 0; i < srcRank; ++i)
      sourceDims.push_back(rewriter.create<memref::DimOp>(loc, src, i));

    SmallVector<int64_t> constOffsets;
    SmallVector<Value> dynOffsets;
    for (Value offset : offsets) {
      std::optional<int64_t> staticVal = getConstantIntValue(offset);
      if (!staticVal)
        dynOffsets.push_back(offset);
      constOffsets.push_back(staticVal.value_or(ShapedType::kDynamic));
    }

    SmallVector<Value> dynShapes;
    for (auto [idx, shape] : llvm::enumerate(srcTy.getShape())) {
      if (shape == ShapedType::kDynamic)
        dynShapes.push_back(sourceDims[idx]);
    }

    // Compute strides in reverse order.
    SmallVector<Value> dynStrides;
    Value accStride = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    // Last stride is guaranteed to be static and unit.
    for (int i = static_cast<int>(strides.size()) - 2; i >= 0; --i) {
      accStride =
          rewriter.create<arith::MulIOp>(loc, accStride, sourceDims[i + 1]);
      if (strides[i] == ShapedType::kDynamic)
        dynStrides.push_back(accStride);
    }
    std::reverse(dynStrides.begin(), dynStrides.end());

    ndDesc = rewriter.create<xegpu::CreateNdDescOp>(
        loc, descType, src, dynOffsets, dynShapes, dynStrides,
        DenseI64ArrayAttr::get(rewriter.getContext(), constOffsets),
        DenseI64ArrayAttr::get(rewriter.getContext(), srcTy.getShape()),
        DenseI64ArrayAttr::get(rewriter.getContext(), strides));
  }

  return ndDesc;
}

struct TransferReadLowering : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {
    Location loc = readOp.getLoc();

    if (failed(transferPreconditions(rewriter, readOp)))
      return failure();

    bool isOutOfBounds = readOp.hasOutOfBoundsDim();
    if (isOutOfBounds && !isZeroConstant(readOp.getPadding()))
      return rewriter.notifyMatchFailure(
          readOp, "Unsupported non-zero padded out-of-bounds read");

    AffineMap readMap = readOp.getPermutationMap();
    bool isTransposeLoad = !readMap.isMinorIdentity();

    VectorType vecTy = readOp.getVectorType();
    Type elementType = vecTy.getElementType();
    unsigned minTransposeBitWidth = 32;
    if (isTransposeLoad &&
        elementType.getIntOrFloatBitWidth() < minTransposeBitWidth)
      return rewriter.notifyMatchFailure(
          readOp, "Unsupported data type for tranposition");

    // If load is transposed, get the base shape for the tensor descriptor.
    SmallVector<int64_t> descShape{vecTy.getShape()};
    if (isTransposeLoad)
      std::reverse(descShape.begin(), descShape.end());
    auto descType = xegpu::TensorDescType::get(
        descShape, elementType, /*array_length=*/1,
        /*boundary_check=*/isOutOfBounds, xegpu::MemorySpace::Global);

    xegpu::CreateNdDescOp ndDesc =
        createNdDescriptor(rewriter, loc, descType,
                           dyn_cast<TypedValue<MemRefType>>(readOp.getSource()),
                           readOp.getIndices());

    DenseI64ArrayAttr transposeAttr =
        !isTransposeLoad ? nullptr
                         : DenseI64ArrayAttr::get(rewriter.getContext(),
                                                  ArrayRef<int64_t>{1, 0});
    // By default, no specific caching policy is assigned.
    xegpu::CachePolicyAttr hint = nullptr;
    auto loadOp = rewriter.create<xegpu::LoadNdOp>(
        loc, vecTy, ndDesc, /*packed=*/nullptr, transposeAttr,
        /*l1_hint=*/hint,
        /*l2_hint=*/hint, /*l3_hint=*/hint);
    rewriter.replaceOp(readOp, loadOp);

    return success();
  }
};

struct TransferWriteLowering
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern<vector::TransferWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                PatternRewriter &rewriter) const override {
    Location loc = writeOp.getLoc();

    if (failed(transferPreconditions(rewriter, writeOp)))
      return failure();

    AffineMap map = writeOp.getPermutationMap();
    if (!map.isMinorIdentity())
      return rewriter.notifyMatchFailure(writeOp, "Expects identity map");

    VectorType vecTy = writeOp.getVectorType();
    auto descType = xegpu::TensorDescType::get(
        vecTy.getShape(), vecTy.getElementType(),
        /*array_length=*/1, /*boundary_check=*/writeOp.hasOutOfBoundsDim(),
        xegpu::MemorySpace::Global);
    xegpu::CreateNdDescOp ndDesc = createNdDescriptor(
        rewriter, loc, descType,
        dyn_cast<TypedValue<MemRefType>>(writeOp.getSource()),
        writeOp.getIndices());

    // By default, no specific caching policy is assigned.
    xegpu::CachePolicyAttr hint = nullptr;
    auto storeOp =
        rewriter.create<xegpu::StoreNdOp>(loc, writeOp.getVector(), ndDesc,
                                          /*l1_hint=*/hint,
                                          /*l2_hint=*/hint, /*l3_hint=*/hint);
    rewriter.replaceOp(writeOp, storeOp);

    return success();
  }
};

struct LoadLowering : public OpRewritePattern<vector::LoadOp> {
  using OpRewritePattern<vector::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    Location loc = loadOp.getLoc();

    VectorType vecTy = loadOp.getResult().getType();
    if (failed(storeLoadPreconditions(rewriter, loadOp, vecTy)))
      return failure();

    auto descType = xegpu::TensorDescType::get(
        vecTy.getShape(), vecTy.getElementType(), /*array_length=*/1,
        /*boundary_check=*/true, xegpu::MemorySpace::Global);
    xegpu::CreateNdDescOp ndDesc = createNdDescriptor(
        rewriter, loc, descType, loadOp.getBase(), loadOp.getIndices());

    // By default, no specific caching policy is assigned.
    xegpu::CachePolicyAttr hint = nullptr;
    auto loadNdOp = rewriter.create<xegpu::LoadNdOp>(
        loc, vecTy, ndDesc, /*packed=*/nullptr, /*transpose=*/nullptr,
        /*l1_hint=*/hint,
        /*l2_hint=*/hint, /*l3_hint=*/hint);
    rewriter.replaceOp(loadOp, loadNdOp);

    return success();
  }
};

struct StoreLowering : public OpRewritePattern<vector::StoreOp> {
  using OpRewritePattern<vector::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    Location loc = storeOp.getLoc();

    TypedValue<VectorType> vector = storeOp.getValueToStore();
    VectorType vecTy = vector.getType();
    if (failed(storeLoadPreconditions(rewriter, storeOp, vecTy)))
      return failure();

    auto descType =
        xegpu::TensorDescType::get(vecTy.getShape(), vecTy.getElementType(),
                                   /*array_length=*/1, /*boundary_check=*/true,
                                   xegpu::MemorySpace::Global);
    xegpu::CreateNdDescOp ndDesc = createNdDescriptor(
        rewriter, loc, descType, storeOp.getBase(), storeOp.getIndices());

    // By default, no specific caching policy is assigned.
    xegpu::CachePolicyAttr hint = nullptr;
    auto storeNdOp =
        rewriter.create<xegpu::StoreNdOp>(loc, vector, ndDesc,
                                          /*l1_hint=*/hint,
                                          /*l2_hint=*/hint, /*l3_hint=*/hint);
    rewriter.replaceOp(storeOp, storeNdOp);

    return success();
  }
};

struct ConvertVectorToXeGPUPass
    : public impl::ConvertVectorToXeGPUBase<ConvertVectorToXeGPUPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateVectorToXeGPUConversionPatterns(patterns);
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

void mlir::populateVectorToXeGPUConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<TransferReadLowering, TransferWriteLowering, LoadLowering,
               StoreLowering>(patterns.getContext());
}

std::unique_ptr<Pass> mlir::createConvertVectorToXeGPUPass() {
  return std::make_unique<ConvertVectorToXeGPUPass>();
}
