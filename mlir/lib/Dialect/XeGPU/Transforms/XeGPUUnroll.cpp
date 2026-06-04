//===- XeGPUUnroll.cpp - patterns to do unrolling ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains patterns for unrolling XeGPU operations. It follows a
// similar concept and design as vector unroll patterns, serving as a complement
// to them.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/Dialect/XeGPU/Transforms/XeGPULayoutImpl.h"
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/DebugLog.h"

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPUUNROLL
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

#define DEBUG_TYPE "xegpu-unroll"

using namespace mlir;

namespace {

template <typename SourceOp>
struct UnrollPattern : public OpRewritePattern<SourceOp> {
  UnrollPattern(MLIRContext *context, const xegpu::UnrollOptions &options,
                PatternBenefit benefit = 1)
      : OpRewritePattern<SourceOp>(context, benefit), options(options) {}

protected:
  /// Return the target shape for the given `op`. Return std::nullopt if the
  /// op shouldn't be or cannot be unrolled.
  std::optional<SmallVector<int64_t>> getTargetShape(Operation *op) const {
    LDBG() << "Get unroll shape for: " << *op;

    if (options.filterConstraint && failed(options.filterConstraint(op))) {
      LDBG() << "--no filter constraint -> BAIL";
      return std::nullopt;
    }

    assert(options.nativeShape &&
           "expects the native shape for native shape call back function.");
    auto nativeShape = options.nativeShape(op);
    return nativeShape;
  }

  SmallVector<Type> getUnrolledTypes(ShapedType type,
                                     ArrayRef<int64_t> tileShape,
                                     bool returnSingleType = false) const {
    return options.getUnrolledTypes(type, tileShape, returnSingleType);
  }

  /// Emulate the the unpack behavior using insert_strided_slice for VectorType
  /// values and unrealized_conversion_cast for TensorDescType values.
  Value unpack(ValueRange srcs, Type destTy, ArrayRef<int64_t> blockSize,
               Location loc, PatternRewriter &rewriter) const {
    if (auto vecTy = dyn_cast<VectorType>(destTy)) {
      auto shape = vecTy.getShape();
      return xegpu::createVectorWithShapeFromValues(rewriter, loc, srcs, shape);
    }

    if (isa<xegpu::TensorDescType>(destTy)) {
      auto attr = NamedAttribute(rewriter.getStringAttr(unpackAttrName),
                                 rewriter.getUnitAttr());
      auto blkAttr = NamedAttribute(rewriter.getStringAttr(blockAttrName),
                                    rewriter.getDenseI64ArrayAttr(blockSize));
      auto castOp = UnrealizedConversionCastOp::create(
          rewriter, loc, destTy, srcs,
          ArrayRef<NamedAttribute>({attr, blkAttr}));
      return castOp.getResult(0);
    }

    llvm_unreachable("Unexpected destTy.");
    return Value();
  }

  /// Emulate the the pack behavior using extract_strided_slice for VectorType
  /// values and unrealized_conversion_cast for TensorDescType values.
  SmallVector<Value> pack(Value src, TypeRange destTypes,
                          ArrayRef<int64_t> blockSize, Location loc,
                          PatternRewriter &rewriter) const {
    if (auto vecTy = dyn_cast<VectorType>(src.getType())) {
      return xegpu::extractVectorsWithShapeFromValue(rewriter, loc, src,
                                                     blockSize);
    }

    if (isa<xegpu::TensorDescType>(src.getType())) {
      auto attr = NamedAttribute(rewriter.getStringAttr(packAttrName),
                                 rewriter.getUnitAttr());
      auto blkAttr = NamedAttribute(rewriter.getStringAttr(blockAttrName),
                                    rewriter.getDenseI64ArrayAttr(blockSize));
      auto castOp = UnrealizedConversionCastOp::create(
          rewriter, loc, destTypes, src,
          ArrayRef<NamedAttribute>({attr, blkAttr}));
      return castOp.getResults();
    }

    llvm_unreachable("Unexpected src type.");
    return SmallVector<Value>();
  }

  /// Helper to pack operands for DPAS-like operations with early return if
  /// no unrolling is needed.
  SmallVector<Value> packOperandForDpas(Value operand,
                                        ArrayRef<int64_t> blockSize,
                                        Location loc,
                                        PatternRewriter &rewriter) const {
    auto vecType = cast<VectorType>(operand.getType());
    std::optional<SmallVector<int64_t>> grids =
        computeShapeRatio(vecType.getShape(), blockSize);
    assert(grids && "Expecting grids to be computed.");
    auto numNewOps = computeProduct(*grids);
    if (numNewOps == 1)
      return SmallVector<Value>({operand});
    VectorType newVecTy =
        vecType.cloneWith(blockSize, vecType.getElementType());
    SmallVector<Type> convertedTypes(numNewOps, newVecTy);
    return pack(operand, convertedTypes, blockSize, loc, rewriter);
  }

private:
  const char *const packAttrName = "__xegpu_blocking_pack__";
  const char *const unpackAttrName = "__xegpu_blocking_unpack__";
  const char *const blockAttrName = "__xegpu_blocking_tile_shape__";

  xegpu::UnrollOptions options;
};

// Generic helper function for unrolling operations with offsets.
//
// Iterates over tile offsets within the tensor descriptor shape and calls
// the provided createOp function for each computed offset. This is used by
// operations like LoadNd, StoreNd, CreateNdDesc, and PrefetchNd when they
// have explicit offsets that need to be adjusted for each unrolled tile.
SmallVector<Value> computeUnrolledOffsets(
    SmallVector<OpFoldResult> mixedOffsets, xegpu::TensorDescType tdescTy,
    ArrayRef<int64_t> targetShape,
    const std::function<Value(SmallVector<OpFoldResult>)> &createOp,
    Location loc, PatternRewriter &rewriter) {
  int64_t rank = tdescTy.getRank();
  ArrayRef<int64_t> shape = tdescTy.getShape();

  auto addi = [&](OpFoldResult a, int64_t b) -> Value {
    std::optional<int64_t> maybeInt = getConstantIntValue(a);
    if (maybeInt) {
      return arith::ConstantIndexOp::create(rewriter, loc, *maybeInt + b);
    } else {
      auto aV = llvm::cast<Value>(a);
      auto bV = arith::ConstantIndexOp::create(rewriter, loc, b);
      return rewriter.createOrFold<arith::AddIOp>(loc, aV, bV);
    }
  };

  SmallVector<OpFoldResult> oldOffsets = llvm::to_vector(
      llvm::drop_begin(mixedOffsets, mixedOffsets.size() - rank));
  auto validIdxes =
      llvm::seq<int64_t>(mixedOffsets.size() - rank, mixedOffsets.size());

  SmallVector<Value> newOps;
  for (SmallVector<int64_t> offsets :
       StaticTileOffsetRange(shape, targetShape)) {

    for (auto [idx, oldOff, offset] :
         llvm::zip(validIdxes, oldOffsets, offsets))
      mixedOffsets[idx] = addi(oldOff, offset);

    auto newOp = createOp(mixedOffsets);
    newOps.push_back(newOp);
  }
  return newOps;
}

struct UnrollCreateNdOp : public UnrollPattern<xegpu::CreateNdDescOp> {
  using UnrollPattern<xegpu::CreateNdDescOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::CreateNdDescOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    xegpu::TensorDescType tdescTy = op.getType();

    std::optional<SmallVector<int64_t>> targetShape = getTargetShape(op);
    if (!targetShape)
      return failure();

    SmallVector<Value> newOps;

    auto newTdescTy = getUnrolledTypes(tdescTy, *targetShape)[0];
    auto newOp =
        xegpu::CreateNdDescOp::create(rewriter, loc, newTdescTy, op.getSource(),
                                      op.getMixedSizes(), op.getMixedStrides());
    newOps.push_back(newOp);
    Value castOp = unpack(newOps, tdescTy, *targetShape, loc, rewriter);
    rewriter.replaceOp(op, castOp);

    return success();
  }
};

struct UnrollPrefetchNdOp : public UnrollPattern<xegpu::PrefetchNdOp> {
  using UnrollPattern<xegpu::PrefetchNdOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::PrefetchNdOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    xegpu::TensorDescType tdescTy = op.getTensorDescType();

    std::optional<SmallVector<int64_t>> targetShape = getTargetShape(op);
    if (!targetShape)
      return failure();

    xegpu::DistributeLayoutAttr layout = op.getLayoutAttr();
    if (layout)
      layout = layout.dropInstData();

    SmallVector<Type> convertedTdescTypes =
        getUnrolledTypes(tdescTy, *targetShape, /*returnSingleType*/ true);

    SmallVector<Value> convertedTdesc = pack(
        op.getTensorDesc(), convertedTdescTypes, *targetShape, loc, rewriter);

    auto createPrefetch = [&](SmallVector<OpFoldResult> offsets) -> Value {
      xegpu::PrefetchNdOp::create(rewriter, loc, convertedTdesc[0], offsets,
                                  op.getL1HintAttr(), op.getL2HintAttr(),
                                  op.getL3HintAttr(), layout);
      // return dummy Value to satisfy function's signature
      return nullptr;
    };

    computeUnrolledOffsets(op.getMixedOffsets(), tdescTy, *targetShape,
                           createPrefetch, loc, rewriter);

    rewriter.eraseOp(op);
    return success();
  }
};

struct UnrollLoadNdOp : public UnrollPattern<xegpu::LoadNdOp> {
  using UnrollPattern<xegpu::LoadNdOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::LoadNdOp op,
                                PatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    VectorType valueTy = op.getType();
    xegpu::TensorDescType tdescTy = op.getTensorDescType();

    std::optional<SmallVector<int64_t>> targetShape = getTargetShape(op);
    if (!targetShape)
      return failure();

    xegpu::DistributeLayoutAttr layout = op.getLayoutAttr();
    if (layout)
      layout = layout.dropInstData();

    Type elemTy = tdescTy.getElementType();
    VectorType newValueTy = valueTy.cloneWith(*targetShape, elemTy);

    SmallVector<Type> convertedTdescTypes =
        getUnrolledTypes(tdescTy, *targetShape, /*returnSingleType*/ true);

    SmallVector<Value> convertedTdescs = pack(
        op.getTensorDesc(), convertedTdescTypes, *targetShape, loc, rewriter);
    SmallVector<Value> newOps;

    auto createLoad = [&](SmallVector<OpFoldResult> offsets) {
      return xegpu::LoadNdOp::create(
          rewriter, loc, newValueTy, convertedTdescs[0], offsets,
          op.getPackedAttr(), op.getTransposeAttr(), op.getL1HintAttr(),
          op.getL2HintAttr(), op.getL3HintAttr(), layout);
    };
    newOps = computeUnrolledOffsets(op.getMixedOffsets(), tdescTy, *targetShape,
                                    createLoad, loc, rewriter);

    Value castOp = unpack(newOps, op.getType(), *targetShape, loc, rewriter);

    rewriter.replaceOp(op, castOp);
    return success();
  }
};

struct UnrollStoreNdOp : public UnrollPattern<xegpu::StoreNdOp> {
  using UnrollPattern<xegpu::StoreNdOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::StoreNdOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    VectorType valueTy = op.getValueType();
    xegpu::TensorDescType tdescTy = op.getTensorDescType();

    std::optional<SmallVector<int64_t>> targetShape = getTargetShape(op);
    if (!targetShape)
      return failure();

    xegpu::DistributeLayoutAttr layout = op.getLayoutAttr();
    if (layout)
      layout = layout.dropInstData();

    SmallVector<Type> convertedValTypes =
        getUnrolledTypes(valueTy, *targetShape);
    SmallVector<Type> convertedTdescTypes =
        getUnrolledTypes(tdescTy, *targetShape, /*returnSingleType*/ true);

    SmallVector<Value> convertedTdescs = pack(
        op.getTensorDesc(), convertedTdescTypes, *targetShape, loc, rewriter);

    SmallVector<Value> convertedValues =
        pack(op.getValue(), convertedValTypes, *targetShape, loc, rewriter);

    size_t valueIndex = 0;
    auto createStore = [&](SmallVector<OpFoldResult> offsets) {
      xegpu::StoreNdOp::create(rewriter, loc, convertedValues[valueIndex++],
                               convertedTdescs[0], offsets, op.getL1HintAttr(),
                               op.getL2HintAttr(), op.getL3HintAttr(), layout);
      // return dummy Value to satisfy function's signature
      return nullptr;
    };

    computeUnrolledOffsets(op.getMixedOffsets(), tdescTy, *targetShape,
                           createStore, loc, rewriter);

    rewriter.eraseOp(op);
    return success();
  }
};

struct UnrollDpasOp : public UnrollPattern<xegpu::DpasOp> {
  using UnrollPattern<xegpu::DpasOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::DpasOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    std::optional<SmallVector<int64_t>> targetShape = getTargetShape(op);
    if (!targetShape || targetShape->size() != 3)
      return failure();
    auto M = (*targetShape)[0];
    auto K = (*targetShape)[1];
    auto N = (*targetShape)[2];

    int64_t aBlockSize[2] = {M, K};
    int64_t bBlockSize[2] = {K, N};
    int64_t cBlockSize[2] = {M, N};

    auto a = op.getLhs();
    auto b = op.getRhs();
    auto c = op.getAcc();

    SmallVector<Value> aVals = packOperandForDpas(a, aBlockSize, loc, rewriter);
    SmallVector<Value> bVals = packOperandForDpas(b, bBlockSize, loc, rewriter);
    SmallVector<Value> cVals;
    if (c)
      cVals = packOperandForDpas(c, cBlockSize, loc, rewriter);

    auto ranges = c ? SmallVector<ValueRange>({aVals, bVals, cVals})
                    : SmallVector<ValueRange>({aVals, bVals});
    if (llvm::any_of(ranges, [](auto &v) { return v.size() == 0; }) ||
        llvm::all_of(ranges, [](auto &v) { return v.size() == 1; }))
      return failure();

    VectorType resultTy = op.getResult().getType();
    auto vecTy = VectorType::get(cBlockSize, resultTy.getElementType());

    auto aShape = a.getType().getShape();
    auto bShape = b.getType().getShape();
    int64_t mIters = aShape[0] / M;
    int64_t kIters = aShape[1] / K;
    int64_t nIters = bShape[1] / N;

    SmallVector<Value> newOps;
    for (int64_t i = 0; i < mIters; ++i) {
      for (int64_t j = 0; j < nIters; ++j) {
        Value tmpC;
        if (c)
          tmpC = cVals[i * nIters + j];

        for (int64_t k = 0; k < kIters; ++k) {
          Value aVec = aVals[i * kIters + k];
          Value bVec = bVals[k * nIters + j];
          SmallVector<Value> operands({aVec, bVec});
          if (tmpC)
            operands.push_back(tmpC);

          tmpC =
              xegpu::DpasOp::create(rewriter, loc, vecTy, operands,
                                    xegpu::dropInstDataOnAttrs(op->getAttrs()));
        }
        newOps.push_back(tmpC);
      }
    }
    Value castOp = unpack(newOps, resultTy, cBlockSize, loc, rewriter);
    rewriter.replaceOp(op, castOp);
    return success();
  }
};

struct UnrollDpasMxOp : public UnrollPattern<xegpu::DpasMxOp> {
  using UnrollPattern<xegpu::DpasMxOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::DpasMxOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    std::optional<SmallVector<int64_t>> targetShape = getTargetShape(op);
    if (!targetShape || targetShape->size() != 4)
      return failure();
    auto M = (*targetShape)[0];
    auto K = (*targetShape)[1];
    auto N = (*targetShape)[2];
    auto S = (*targetShape)[3];

    int64_t aBlockSize[2] = {M, K};
    int64_t bBlockSize[2] = {K, N};
    int64_t cBlockSize[2] = {M, N};
    int64_t aScaleBlockSize[2] = {M, S};
    int64_t bScaleBlockSize[2] = {S, N};

    auto a = op.getA();
    auto b = op.getB();
    auto c = op.getAcc();
    auto ascale = dyn_cast<TypedValue<VectorType>>(op.getScaleA());
    auto bscale = dyn_cast<TypedValue<VectorType>>(op.getScaleB());

    SmallVector<Value> aVals = packOperandForDpas(a, aBlockSize, loc, rewriter);
    SmallVector<Value> bVals = packOperandForDpas(b, bBlockSize, loc, rewriter);
    SmallVector<Value> cVals;
    if (c)
      cVals = packOperandForDpas(c, cBlockSize, loc, rewriter);
    SmallVector<Value> aScaleVals;
    if (ascale)
      aScaleVals = packOperandForDpas(ascale, aScaleBlockSize, loc, rewriter);
    SmallVector<Value> bScaleVals;
    if (bscale)
      bScaleVals = packOperandForDpas(bscale, bScaleBlockSize, loc, rewriter);

    VectorType resultTy = op.getResult().getType();
    auto vecTy = VectorType::get(cBlockSize, resultTy.getElementType());

    auto aShape = a.getType().getShape();
    auto bShape = b.getType().getShape();
    int64_t mIters = aShape[0] / M;
    int64_t kIters = aShape[1] / K;
    int64_t nIters = bShape[1] / N;

    SmallVector<Value> newOps;
    xegpu::DpasMxOp newDpasMxOp;
    for (int64_t i = 0; i < mIters; ++i) {
      for (int64_t j = 0; j < nIters; ++j) {
        Value tmpC;
        if (c)
          tmpC = cVals[i * nIters + j];

        for (int64_t k = 0; k < kIters; ++k) {
          Value aVec = aVals[i * kIters + k];
          Value bVec = bVals[k * nIters + j];
          SmallVector<Value> operands({aVec, bVec});
          if (tmpC)
            operands.push_back(tmpC);
          if (ascale)
            operands.push_back(aScaleVals[i * kIters + k]);
          if (bscale)
            operands.push_back(bScaleVals[k * nIters + j]);

          newDpasMxOp = xegpu::DpasMxOp::create(
              rewriter, loc, vecTy, operands,
              xegpu::dropInstDataOnAttrs(op->getAttrs()));
          tmpC = newDpasMxOp.getResult();
        }
        newOps.push_back(newDpasMxOp);
      }
    }
    Value castOp = unpack(newOps, resultTy, cBlockSize, loc, rewriter);
    rewriter.replaceOp(op, castOp);
    return success();
  }
};

/// This pattern handles the unrolling of LoadGatherOp with offsets (gathered
/// load).
/// It unrolls the offsets and mask operands accordingly, and creates multiple
/// LoadGatherOp with the unrolled operands.
struct UnrollLoadGatherOp : public UnrollPattern<xegpu::LoadGatherOp> {
  using UnrollPattern<xegpu::LoadGatherOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::LoadGatherOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    VectorType valueTy = llvm::dyn_cast<VectorType>(op.getType());
    Value offsets = op.getOffsets();
    Value mask = op.getMask();

    std::optional<SmallVector<int64_t>> targetShape = getTargetShape(op);
    if (!targetShape)
      return failure();

    SmallVector<int64_t> targetMaskShape(*targetShape);
    int64_t chunkSize = 1;
    if (auto chunkSizeAttr = op->getAttr("chunk_size")) {
      if (auto intAttr = llvm::dyn_cast<IntegerAttr>(chunkSizeAttr))
        chunkSize = intAttr.getInt();
    }

    // Unroll mask and offsets with correct shape
    VectorType maskTy = llvm::dyn_cast<VectorType>(mask.getType());
    VectorType offsetsTy = llvm::dyn_cast<VectorType>(offsets.getType());
    Type elemTy = valueTy.getElementType();
    VectorType newValueTy = VectorType::get(*targetShape, elemTy);

    SmallVector<Type> convertedMaskTypes;
    SmallVector<Value> convertedMasks;
    SmallVector<Type> convertedOffsetTypes;
    SmallVector<Value> convertedOffsets;

    if (chunkSize > 1) {
      // For chunked loads, mask and offsets have one less dimension
      targetMaskShape.pop_back();
      int64_t blockedChunkSize = targetShape->back();
      int64_t numNewChunks = chunkSize / blockedChunkSize;
      chunkSize = blockedChunkSize;

      convertedMaskTypes = getUnrolledTypes(maskTy, targetMaskShape);
      convertedOffsetTypes = getUnrolledTypes(offsetsTy, targetMaskShape);

      SmallVector<Value> convertedMasksBase =
          pack(mask, convertedMaskTypes, targetMaskShape, loc, rewriter);
      SmallVector<Value> convertedOffsetsBase =
          pack(offsets, convertedOffsetTypes, targetMaskShape, loc, rewriter);

      for (auto maskVal : convertedMasksBase)
        convertedMasks.append(numNewChunks, maskVal);

      for (auto [baseOffset, offsetType] :
           llvm::zip(convertedOffsetsBase, convertedOffsetTypes)) {
        for (int64_t i = 0; i < numNewChunks; ++i) {
          Value inc = arith::ConstantIndexOp::create(rewriter, loc,
                                                     i * blockedChunkSize);
          Value incVec =
              vector::BroadcastOp::create(rewriter, loc, offsetType, inc);
          Value offsetVal =
              arith::AddIOp::create(rewriter, loc, baseOffset, incVec);
          convertedOffsets.push_back(offsetVal);
        }
      }
    } else {
      convertedMaskTypes = getUnrolledTypes(maskTy, targetMaskShape);
      convertedMasks =
          pack(mask, convertedMaskTypes, targetMaskShape, loc, rewriter);

      convertedOffsetTypes = getUnrolledTypes(offsetsTy, *targetShape);
      convertedOffsets =
          pack(offsets, convertedOffsetTypes, *targetShape, loc, rewriter);
    }

    auto layout = op.getLayoutAttr();
    if (layout)
      layout = layout.dropInstData();

    SmallVector<Value> newOps;
    for (auto [o, m] : llvm::zip(convertedOffsets, convertedMasks)) {
      auto newOp = xegpu::LoadGatherOp::create(
          rewriter, loc, newValueTy, op.getSource(), o, m,
          rewriter.getI64IntegerAttr(chunkSize), op.getL1HintAttr(),
          op.getL2HintAttr(), op.getL3HintAttr(), layout);
      newOps.push_back(newOp);
    }

    Value castOp = unpack(newOps, op.getType(), *targetShape, loc, rewriter);
    rewriter.replaceOp(op, castOp);
    return success();
  }
};

/// This pattern handles the unrolling of StoreScatterOp with offsets (scattered
/// store).
/// It unrolls the offsets and mask operands accordingly, and creates multiple
/// StoreScatterOp with the unrolled operands.
struct UnrollStoreScatterOp : public UnrollPattern<xegpu::StoreScatterOp> {
  using UnrollPattern<xegpu::StoreScatterOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::StoreScatterOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    VectorType valueTy = llvm::dyn_cast<VectorType>(op.getValue().getType());
    Value offsets = op.getOffsets();
    Value mask = op.getMask();

    std::optional<SmallVector<int64_t>> targetShape = getTargetShape(op);
    if (!targetShape)
      return failure();

    int64_t chunkSize = 1;
    if (auto chunkSizeAttr = op->getAttr("chunk_size")) {
      if (auto intAttr = llvm::dyn_cast<IntegerAttr>(chunkSizeAttr))
        chunkSize = intAttr.getInt();
    }

    SmallVector<int64_t> targetMaskShape(*targetShape);
    VectorType maskTy = llvm::dyn_cast<VectorType>(mask.getType());
    VectorType offsetsTy = llvm::dyn_cast<VectorType>(offsets.getType());

    SmallVector<Type> convertedMaskTypes;
    SmallVector<Value> convertedMasks;
    SmallVector<Type> convertedOffsetTypes;
    SmallVector<Value> convertedOffsets;

    if (chunkSize > 1) {
      targetMaskShape.pop_back();
      int64_t blockedChunkSize = targetShape->back();
      int64_t numNewChunks = chunkSize / blockedChunkSize;
      chunkSize = blockedChunkSize;

      convertedMaskTypes = getUnrolledTypes(maskTy, targetMaskShape);
      convertedOffsetTypes = getUnrolledTypes(offsetsTy, targetMaskShape);

      SmallVector<Value> convertedMasksBase =
          pack(mask, convertedMaskTypes, targetMaskShape, loc, rewriter);
      SmallVector<Value> convertedOffsetsBase =
          pack(offsets, convertedOffsetTypes, targetMaskShape, loc, rewriter);

      for (auto maskVal : convertedMasksBase)
        convertedMasks.append(numNewChunks, maskVal);

      for (auto [baseOffset, offsetType] :
           llvm::zip(convertedOffsetsBase, convertedOffsetTypes)) {
        for (int64_t i = 0; i < numNewChunks; ++i) {
          Value inc = arith::ConstantIndexOp::create(rewriter, loc,
                                                     i * blockedChunkSize);
          Value incVec =
              vector::BroadcastOp::create(rewriter, loc, offsetType, inc);
          Value offsetVal =
              arith::AddIOp::create(rewriter, loc, baseOffset, incVec);
          convertedOffsets.push_back(offsetVal);
        }
      }
    } else {
      convertedMaskTypes = getUnrolledTypes(maskTy, targetMaskShape);
      convertedMasks =
          pack(mask, convertedMaskTypes, targetMaskShape, loc, rewriter);

      convertedOffsetTypes = getUnrolledTypes(offsetsTy, *targetShape);
      convertedOffsets =
          pack(offsets, convertedOffsetTypes, *targetShape, loc, rewriter);
    }

    SmallVector<Type> convertedValTypes =
        getUnrolledTypes(valueTy, *targetShape);
    SmallVector<Value> convertedValues =
        pack(op.getValue(), convertedValTypes, *targetShape, loc, rewriter);

    auto layout = op.getLayoutAttr();
    if (layout)
      layout = layout.dropInstData();

    for (auto [v, o, m] :
         llvm::zip(convertedValues, convertedOffsets, convertedMasks)) {
      xegpu::StoreScatterOp::create(rewriter, loc, v, op.getDest(), o, m,
                                    rewriter.getI64IntegerAttr(chunkSize),
                                    op.getL1HintAttr(), op.getL2HintAttr(),
                                    op.getL3HintAttr(), layout);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct UnrollLoadMatrixOp : public UnrollPattern<xegpu::LoadMatrixOp> {
  using UnrollPattern<xegpu::LoadMatrixOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::LoadMatrixOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    VectorType valueTy = llvm::dyn_cast<VectorType>(op.getType());
    assert(valueTy && "the value type must be vector type!");

    std::optional<SmallVector<int64_t>> targetShape = getTargetShape(op);
    if (!targetShape || targetShape->size() != (size_t)valueTy.getRank())
      return failure();

    Type elemTy = valueTy.getElementType();
    ArrayRef<int64_t> shape = valueTy.getShape();
    xegpu::DistributeLayoutAttr layout = op.getLayoutAttr();

    VectorType newValueTy = valueTy.cloneWith(*targetShape, elemTy);

    SmallVector<OpFoldResult> mixedOffsets = op.getMixedOffsets();
    SmallVector<SmallVector<OpFoldResult>> offsetsList;
    for (SmallVector<int64_t> offsets :
         StaticTileOffsetRange(shape, *targetShape)) {
      auto adds = xegpu::addElementwise(
          rewriter, loc, mixedOffsets,
          getAsIndexOpFoldResult(op.getContext(), offsets));
      offsetsList.push_back(adds);
    }

    SmallVector<Value> newOps;
    if (layout)
      layout = layout.dropInstData();
    for (SmallVector<OpFoldResult> offsets : offsetsList) {
      auto newOp = xegpu::LoadMatrixOp::create(
          rewriter, op.getLoc(), newValueTy, op.getMemDesc(), offsets, layout);
      newOps.push_back(newOp);
    }
    Value castOp = unpack(newOps, op.getType(), *targetShape, loc, rewriter);
    rewriter.replaceOp(op, castOp);
    return success();
  }
};

struct UnrollStoreMatrixOp : public UnrollPattern<xegpu::StoreMatrixOp> {
  using UnrollPattern<xegpu::StoreMatrixOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::StoreMatrixOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<SmallVector<int64_t>> targetShape = getTargetShape(op);
    if (!targetShape)
      return failure();

    Location loc = op.getLoc();
    VectorType valueTy = llvm::dyn_cast<VectorType>(op.getData().getType());
    assert(valueTy && "the value type must be vector type!");
    ArrayRef<int64_t> shape = valueTy.getShape();
    xegpu::DistributeLayoutAttr layout = op.getLayoutAttr();
    if (layout)
      layout = layout.dropInstData();

    SmallVector<Type> convertedValTypes =
        getUnrolledTypes(valueTy, *targetShape);
    SmallVector<Value> convertedValues =
        pack(op.getData(), convertedValTypes, *targetShape, loc, rewriter);

    SmallVector<OpFoldResult> mixedOffsets = op.getMixedOffsets();
    SmallVector<SmallVector<OpFoldResult>> offsetsList;
    for (SmallVector<int64_t> offsets :
         StaticTileOffsetRange(shape, *targetShape)) {
      auto adds = xegpu::addElementwise(
          rewriter, loc, mixedOffsets,
          getAsIndexOpFoldResult(op.getContext(), offsets));
      offsetsList.push_back(adds);
    }

    for (auto [v, offsets] : llvm::zip_equal(convertedValues, offsetsList))
      xegpu::StoreMatrixOp::create(rewriter, loc, v, op.getMemDesc(), offsets,
                                   layout);

    rewriter.eraseOp(op);
    return success();
  }
};

/// UnrollConvertLayoutOp pattern for unrolling xegpu::ConvertLayoutOp
/// operations. It first check whether the convert layout op has valid layouts
/// after inst_data stripped. If it does, it will unroll the vector into
/// multiple smaller vectors according to the target shape, and create multiple
/// ConvertLayoutOp with the unrolled vectors and the stripped layouts.
struct UnrollConvertLayoutOp : public UnrollPattern<xegpu::ConvertLayoutOp> {
  using UnrollPattern<xegpu::ConvertLayoutOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::ConvertLayoutOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type valType = op.getType();

    xegpu::DistributeLayoutAttr inputLayout = op.getInputLayoutAttr();
    xegpu::DistributeLayoutAttr targetLayout = op.getTargetLayoutAttr();
    if (!inputLayout || !targetLayout)
      return rewriter.notifyMatchFailure(op, "missing layout attributes.");

    if (valType.isIntOrFloat()) {
      rewriter.replaceOp(op, op.getSource());
      assert(!inputLayout.dropInstData() && !targetLayout.dropInstData() &&
             "unexpected layout attributes for scalar type");
      return success();
    }

    if (inputLayout.getEffectiveInstDataAsInt().empty() ||
        targetLayout.getEffectiveInstDataAsInt().empty())
      return rewriter.notifyMatchFailure(op, "Not a target ConvertLayoutOp.");

    inputLayout = inputLayout.dropInstData();
    targetLayout = targetLayout.dropInstData();

    VectorType valueTy = llvm::dyn_cast<VectorType>(op.getType());
    assert(valueTy && "the value type must be vector type!");

    std::optional<SmallVector<int64_t>> targetShape = getTargetShape(op);
    if (!targetShape || targetShape->size() != (size_t)valueTy.getRank())
      return failure();

    Value newSource = op.getSource();
    SmallVector<Value> newOps;
    if (inputLayout && targetLayout) {
      SmallVector<Type> convertedValTypes =
          getUnrolledTypes(valueTy, *targetShape);
      SmallVector<Value> convertedValues =
          pack(op.getOperand(), convertedValTypes, *targetShape, loc, rewriter);
      for (auto [v, t] : llvm::zip(convertedValues, convertedValTypes)) {
        auto newOp = xegpu::ConvertLayoutOp::create(rewriter, loc, t, v,
                                                    inputLayout, targetLayout);
        newOps.push_back(newOp);
      }
      newSource = unpack(newOps, op.getType(), *targetShape, loc, rewriter);
    }

    rewriter.replaceOp(op, newSource);
    return success();
  }
};

/// Unrolls vector.multi_reduction by sequentially reducing tiles with
/// elementwise arith operations first, then a single multi_reduction
/// per non-reduced tile position. This avoids generating long chains of
/// multi_reduction ops (as the upstream pattern does) and is more efficient.
///
/// Example:
/// vector.multi_reduction <32x64xf16> to <32xf16> (tile_shape=32, 32)
/// -- Upstream pattern generates:
/// %tmp1 = vector.multi_reduction %tile0, %zero_acc <32x32xf16> to <32xf16>
/// %res = vector.multi_reduction %tmp1, %tile1 <32x32xf16> to <32xf16>
/// -- This pattern generates:
/// %tmp1 = arith.reduction %tile0, %tile1 <32x32xf16> -> <32x32xf16> //
/// elementwise %res = vector.multi_reduction %tmp1, %zero_acc <32x32xf16> to
/// <32xf16>
struct UnrollMultiReductionOp
    : public UnrollPattern<vector::MultiDimReductionOp> {
  UnrollMultiReductionOp(MLIRContext *context,
                         const xegpu::UnrollOptions &options,
                         PatternBenefit benefit = 2)
      : UnrollPattern<vector::MultiDimReductionOp>(context, options, benefit) {}

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp reductionOp,
                                PatternRewriter &rewriter) const override {
    VectorType srcTy = reductionOp.getSourceVectorType();
    ArrayRef<int64_t> srcShape = srcTy.getShape();
    int64_t srcRank = srcTy.getRank();

    Location loc = reductionOp.getLoc();
    Value source = reductionOp.getSource();
    Value acc = reductionOp.getAcc();
    vector::CombiningKind kind = reductionOp.getKind();

    // Result must be a vector (not scalar).
    auto resultType = dyn_cast<VectorType>(reductionOp.getDestType());
    if (!resultType)
      return failure();

    std::optional<SmallVector<int64_t>> targetShapeOpt =
        getTargetShape(reductionOp);
    if (!targetShapeOpt ||
        static_cast<int64_t>(targetShapeOpt->size()) != srcRank)
      return failure();

    SmallVector<int64_t> targetShape = *targetShapeOpt;

    // Check divisibility for all dimensions.
    for (int64_t i = 0; i < srcRank; ++i) {
      if (srcShape[i] % targetShape[i] != 0)
        return failure();
    }

    SmallVector<bool> reductionMask = reductionOp.getReductionMask();
    // Identify reduced and kept dimensions from the reduction mask.
    SmallVector<int64_t> reducedDims, keptDims;
    for (int64_t i = 0; i < srcRank; ++i) {
      if (reductionMask[i])
        reducedDims.push_back(i);
      else
        keptDims.push_back(i);
    }

    // Compute the number of tiles along each reduced dimension and their
    // product
    SmallVector<int64_t> numReducedTilesPerDim;
    for (int64_t d : reducedDims)
      numReducedTilesPerDim.push_back(srcShape[d] / targetShape[d]);

    // Build kept shapes for iterating over non-reduced dimensions.
    SmallVector<int64_t> keptShape, keptTileShape;
    for (int64_t d : keptDims) {
      keptShape.push_back(srcShape[d]);
      keptTileShape.push_back(targetShape[d]);
    }

    // Initialize the result vector for assembly.
    Value result = arith::ConstantOp::create(rewriter, loc, resultType,
                                             rewriter.getZeroAttr(resultType));

    // Iterate over all tile positions in the kept dimensions.
    // Ex: [off0, off1, _ _ off4]
    // blanks are offsets for the reduced dims, they will be
    // generated in the inner loop below
    for (SmallVector<int64_t> keptOffsets :
         StaticTileOffsetRange(keptShape, keptTileShape)) {

      // Reconstruct full-rank base offsets with 0 for reduced dims.
      // Ex: [off0, off1, 0, 0, off4]
      SmallVector<int64_t> baseOffsets(srcRank, 0);
      for (auto [idx, dim] : llvm::enumerate(keptDims))
        baseOffsets[dim] = keptOffsets[idx];

      // Generate the full tile indices for the reduced dimensions.
      // Ex: if reduceDimShapes = [32, 64] and
      // reducedDimTargetShapes = [16, 16], then reducedTileCoords:
      // [(0, 0), (0, 1), (0, 2), (0, 3),
      //  (1, 0), (1, 1), (1, 2), (1, 3)]
      auto reducedTileCoords = StaticTileOffsetRange(
          numReducedTilesPerDim, SmallVector<int64_t>(reducedDims.size(), 1));

      // Step 1: Fill "blanks" in the offsets for the reduced dimensions
      // using 'reducedTileCoords' and extract according tiles.
      // Ex: tiles = [source[off0, off1, off2_red, off3_red, off4], ...]
      SmallVector<Value> tiles;
      for (SmallVector<int64_t> reducedTileIdx : reducedTileCoords) {
        SmallVector<int64_t> offsets(baseOffsets);
        for (auto [idx, dim] : llvm::enumerate(reducedDims))
          offsets[dim] = reducedTileIdx[idx] * targetShape[dim];
        SmallVector<int64_t> strides(srcRank, 1);
        Value tile = vector::ExtractStridedSliceOp::create(
            rewriter, loc, source, offsets, targetShape, strides);
        tiles.push_back(tile);
      }

      // Step 2: Sequentially reduce tiles using elementwise arith operations.
      Value reduced = tiles[0];
      for (size_t i = 1; i < tiles.size(); ++i)
        reduced =
            vector::makeArithReduction(rewriter, loc, kind, reduced, tiles[i]);

      // Step 3: Perform a single multi_reduction with the accumulator slice.
      SmallVector<int64_t> accStrides(keptTileShape.size(), 1);
      Value accSlice = vector::ExtractStridedSliceOp::create(
          rewriter, loc, acc, keptOffsets, keptTileShape, accStrides);

      auto newReduction = vector::MultiDimReductionOp::create(
          rewriter, loc, reduced, accSlice, reductionMask, kind);

      // Step 4: Insert the reduced result into the output vector.
      SmallVector<int64_t> dstStrides(keptTileShape.size(), 1);
      result = vector::InsertStridedSliceOp::create(
          rewriter, loc, newReduction, result, keptOffsets, dstStrides);
    }

    rewriter.replaceOp(reductionOp, result);
    return success();
  }
};

} // namespace

void mlir::xegpu::populateXeGPUUnrollPatterns(
    RewritePatternSet &patterns, const xegpu::UnrollOptions &options) {
  patterns
      .add<UnrollCreateNdOp, UnrollPrefetchNdOp, UnrollLoadNdOp,
           UnrollStoreNdOp, UnrollDpasOp, UnrollDpasMxOp, UnrollLoadMatrixOp,
           UnrollStoreMatrixOp, UnrollLoadGatherOp, UnrollStoreScatterOp,
           UnrollConvertLayoutOp, UnrollMultiReductionOp>(patterns.getContext(),
                                                          options);
}
