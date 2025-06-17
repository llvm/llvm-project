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

#include "mlir/Dialect/XeGPU/Transforms/Passes.h"

#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include <numeric>

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPUUNROLL
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

#define DEBUG_TYPE "xegpu-unroll"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

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
    LDBG("");
    LDBG("Get unroll shape for: " << *op);

    if (options.filterConstraint && failed(options.filterConstraint(op))) {
      LDBG("--no filter constraint -> BAIL");
      return std::nullopt;
    }

    assert(options.nativeShape &&
           "expects the native shape for native shape call back function.");
    auto nativeShape = options.nativeShape(op);
    return nativeShape;
  }

  SmallVector<Type> getUnrolledTypes(ShapedType type,
                                     ArrayRef<int64_t> tileShape) const {
    return options.getUnrolledTypes(type, tileShape);
  }

  /// Emulate the the unpack behavior using insert_strided_slice for VectorType
  /// values and unrealized_conversion_cast for TensorDescType values.
  Value unpack(ValueRange srcs, Type destTy, ArrayRef<int64_t> blockSize,
               Location loc, PatternRewriter &rewriter) const {
    if (auto vecTy = dyn_cast<VectorType>(destTy)) {
      assert(vecTy.getRank() == static_cast<int64_t>(blockSize.size()) &&
             "Expecting blockSize size to match the rank of destTy.");
      auto shape = vecTy.getShape();
      return xegpu::createVectorWithShapeFromValues(rewriter, loc, srcs, shape);
    }

    if (isa<xegpu::TensorDescType>(destTy)) {
      auto attr = NamedAttribute(rewriter.getStringAttr(unpackAttrName),
                                 rewriter.getUnitAttr());
      auto blkAttr = NamedAttribute(rewriter.getStringAttr(blockAttrName),
                                    rewriter.getDenseI64ArrayAttr(blockSize));
      auto castOp = rewriter.create<UnrealizedConversionCastOp>(
          loc, destTy, srcs, ArrayRef<NamedAttribute>({attr, blkAttr}));
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
      assert(vecTy.getRank() == static_cast<int64_t>(blockSize.size()) &&
             "Expecting blockSize size to match the rank of src.");
      return xegpu::extractVectorsWithShapeFromValue(rewriter, loc, src,
                                                     blockSize);
    }

    if (isa<xegpu::TensorDescType>(src.getType())) {
      auto attr = NamedAttribute(rewriter.getStringAttr(packAttrName),
                                 rewriter.getUnitAttr());
      auto blkAttr = NamedAttribute(rewriter.getStringAttr(blockAttrName),
                                    rewriter.getDenseI64ArrayAttr(blockSize));
      auto castOp = rewriter.create<UnrealizedConversionCastOp>(
          loc, destTypes, src, ArrayRef<NamedAttribute>({attr, blkAttr}));
      return castOp.getResults();
    }

    llvm_unreachable("Unexpected src type.");
    return SmallVector<Value>();
  }

private:
  const char *const packAttrName = "__xegpu_blocking_pack__";
  const char *const unpackAttrName = "__xegpu_blocking_unpack__";
  const char *const blockAttrName = "__xegpu_blocking_tile_shape__";

  xegpu::UnrollOptions options;
};

struct UnrollCreateNdOp : public UnrollPattern<xegpu::CreateNdDescOp> {
  using UnrollPattern<xegpu::CreateNdDescOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::CreateNdDescOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    xegpu::TensorDescType tdescTy = op.getType();
    int64_t rank = tdescTy.getRank();
    ArrayRef<int64_t> shape = tdescTy.getShape();

    std::optional<SmallVector<int64_t>> targetShape = getTargetShape(op);
    if (!targetShape)
      return failure();

    auto newTdescTy = getUnrolledTypes(tdescTy, *targetShape)[0];

    auto addi = [&](OpFoldResult a, int64_t b) -> Value {
      std::optional<int64_t> maybeInt = getConstantIntValue(a);
      if (maybeInt) {
        return rewriter.create<arith::ConstantIndexOp>(loc, *maybeInt + b);
      } else {
        auto aV = llvm::cast<Value>(a);
        auto bV = rewriter.create<arith::ConstantIndexOp>(loc, b);
        return rewriter.createOrFold<arith::AddIOp>(loc, aV, bV);
      }
    };

    SmallVector<OpFoldResult> mixedOffsets = op.getMixedOffsets();

    // For n-D memrefs where n > rank, we need to handle the last `rank`
    // dimensions only, and keep the first `n-rank` dimensions as is.
    SmallVector<OpFoldResult> oldOffsets = llvm::to_vector(
        llvm::drop_begin(mixedOffsets, mixedOffsets.size() - rank));
    auto validIdxes =
        llvm::seq<int64_t>(mixedOffsets.size() - rank, mixedOffsets.size());

    SmallVector<Value> newOps;
    for (SmallVector<int64_t> offsets :
         StaticTileOffsetRange(shape, *targetShape)) {

      for (auto [idx, oldOff, offset] :
           llvm::zip(validIdxes, oldOffsets, offsets))
        mixedOffsets[idx] = addi(oldOff, offset);

      auto newOp = rewriter.create<xegpu::CreateNdDescOp>(
          loc, newTdescTy, op.getSource(), mixedOffsets, op.getMixedSizes(),
          op.getMixedStrides());
      newOps.push_back(newOp);
    }
    Value castOp = unpack(newOps, tdescTy, *targetShape, loc, rewriter);
    rewriter.replaceOp(op, castOp);

    return success();
  }
};

struct UnrollUpdateNdOffsetOp : public UnrollPattern<xegpu::UpdateNdOffsetOp> {
  using UnrollPattern<xegpu::UpdateNdOffsetOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::UpdateNdOffsetOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    xegpu::TensorDescType tdescTy = op.getTensorDescType();

    std::optional<SmallVector<int64_t>> targetShape = getTargetShape(op);
    if (!targetShape)
      return failure();

    SmallVector<Type> convertedTdescTypes =
        getUnrolledTypes(tdescTy, *targetShape);
    SmallVector<Value> convertedTdesc = pack(
        op.getTensorDesc(), convertedTdescTypes, *targetShape, loc, rewriter);

    SmallVector<Value> newOps;
    for (auto t : convertedTdesc) {
      auto newOp = rewriter.create<xegpu::UpdateNdOffsetOp>(
          loc, t.getType(), t, op.getOffsets(), op.getConstOffsets());
      newOps.push_back(newOp);
    }
    Value castOp = unpack(newOps, op.getType(), *targetShape, loc, rewriter);
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

    SmallVector<Type> convertedTdescTypes =
        getUnrolledTypes(tdescTy, *targetShape);
    SmallVector<Value> convertedTdesc = pack(
        op.getTensorDesc(), convertedTdescTypes, *targetShape, loc, rewriter);

    for (auto t : convertedTdesc)
      rewriter.create<xegpu::PrefetchNdOp>(loc, TypeRange(), t, op->getAttrs());

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

    Type elemTy = tdescTy.getElementType();
    VectorType newValueTy = valueTy.cloneWith(*targetShape, elemTy);

    SmallVector<Type> convertedTdescTypes =
        getUnrolledTypes(tdescTy, *targetShape);
    SmallVector<Value> convertedTdescs = pack(
        op.getTensorDesc(), convertedTdescTypes, *targetShape, loc, rewriter);

    SmallVector<Value> newOps;
    for (auto t : convertedTdescs) {
      auto newOp =
          rewriter.create<xegpu::LoadNdOp>(loc, newValueTy, t, op->getAttrs());
      newOps.push_back(newOp);
    }

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

    SmallVector<Type> convertedValTypes =
        getUnrolledTypes(valueTy, *targetShape);
    SmallVector<Type> convertedTdescTypes =
        getUnrolledTypes(tdescTy, *targetShape);

    SmallVector<Value> convertedValues =
        pack(op.getValue(), convertedValTypes, *targetShape, loc, rewriter);
    SmallVector<Value> convertedTdescs = pack(
        op.getTensorDesc(), convertedTdescTypes, *targetShape, loc, rewriter);

    for (auto [v, t] : llvm::zip(convertedValues, convertedTdescs))
      rewriter.create<xegpu::StoreNdOp>(loc, v, t, op.getL1HintAttr(),
                                        op.getL2HintAttr(), op.getL3HintAttr());

    rewriter.eraseOp(op);
    return success();
  }
};

struct UnrollDpasOp : public UnrollPattern<xegpu::DpasOp> {
  using UnrollPattern<xegpu::DpasOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::DpasOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // expecting every operands is a 2D Vector
    if (llvm::any_of(op->getOperandTypes(), [&](Type type) {
          auto vecTy = dyn_cast<VectorType>(type);
          return !vecTy || vecTy.getRank() != 2;
        }))
      return failure();

    // A vector of 3 elements should be returned, representing M, K, N
    // respectively.
    std::optional<SmallVector<int64_t>> targetShape = getTargetShape(op);
    if (!targetShape || targetShape->size() != 3)
      return failure();
    auto M = (*targetShape)[0];
    auto K = (*targetShape)[1];
    auto N = (*targetShape)[2];

    int64_t aBlockSize[2] = {M, K};
    int64_t bBlockSize[2] = {K, N};
    int64_t cBlockSize[2] = {M, N};

    auto packWrapper = [&](TypedValue<VectorType> val,
                           ArrayRef<int64_t> blockSize) {
      VectorType type = val.getType();
      std::optional<SmallVector<int64_t>> grids =
          computeShapeRatio(type.getShape(), blockSize);
      assert(grids && "Expecting grids to be computed.");
      auto numNewOps = computeProduct(*grids);
      if (numNewOps == 1)
        return SmallVector<Value>({val});
      VectorType newVecTy = type.cloneWith(blockSize, type.getElementType());
      SmallVector<Type> convertedTypes(numNewOps, newVecTy);
      SmallVector<Value> values =
          pack(val, convertedTypes, blockSize, loc, rewriter);
      return values;
    };

    auto a = op.getLhs();
    auto b = op.getRhs();
    auto c = op.getAcc();

    auto aShape = a.getType().getShape();
    auto bShape = b.getType().getShape();

    SmallVector<Value> aVals, bVals, cVals;
    aVals = packWrapper(a, aBlockSize);
    bVals = packWrapper(b, bBlockSize);

    if (c)
      cVals = packWrapper(c, cBlockSize);

    // Skip the operation if every operand has an invalid blocking size (empty)
    // or if the original shape matches the blocking size (size == 1).
    auto ranges = c ? SmallVector<ValueRange>({aVals, bVals, cVals})
                    : SmallVector<ValueRange>({aVals, bVals});
    if (llvm::any_of(ranges, [](auto &v) { return v.size() == 0; }) ||
        llvm::all_of(ranges, [](auto &v) { return v.size() == 1; }))
      return failure();

    VectorType resultTy = op.getResult().getType();
    auto vecTy = VectorType::get(cBlockSize, resultTy.getElementType());

    int64_t mIters = aShape[0] / M;
    int64_t kIters = aShape[1] / K;
    int64_t nIters = bShape[1] / N;

    SmallVector<Value> newOps;
    for (int64_t i = 0; i < mIters; ++i) {
      for (int64_t j = 0; j < nIters; ++j) {
        Value tmpC;
        if (c)
          tmpC = cVals[i * nIters + j]; // init with acc

        for (int64_t k = 0; k < kIters; ++k) {
          Value aVec = aVals[i * kIters + k];
          Value bVec = bVals[k * nIters + j];
          SmallVector<Value> operands({aVec, bVec});
          if (tmpC)
            operands.push_back(tmpC);

          tmpC = rewriter.create<xegpu::DpasOp>(loc, vecTy, operands,
                                                op->getAttrs());
        }
        newOps.push_back(tmpC);
      }
    }
    Value castOp = unpack(newOps, resultTy, cBlockSize, loc, rewriter);
    rewriter.replaceOp(op, castOp);
    return success();
  }
};

struct UnrollCreateDescOp : public UnrollPattern<xegpu::CreateDescOp> {
  using UnrollPattern<xegpu::CreateDescOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::CreateDescOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    xegpu::TensorDescType tdescTy = op.getType();
    TypedValue<::mlir::VectorType> indiceVec = op.getOffsets();
    VectorType indiceVecTy = indiceVec.getType();

    if (!tdescTy.isScattered())
      return failure();

    std::optional<SmallVector<int64_t>> targetShape = getTargetShape(op);
    if (!targetShape)
      return failure();

    SmallVector<int64_t> targetIndiceShape(*targetShape);
    int64_t originalChunkSize = tdescTy.getChunkSize();
    // IndiceVec is 1 dim lower than tdescTy when chunkSize is larger than 1.
    if (originalChunkSize > 1)
      targetIndiceShape.pop_back();

    auto newTdescTy = getUnrolledTypes(tdescTy, *targetShape)[0];
    SmallVector<Type> convertedIndiceTypes =
        getUnrolledTypes(indiceVecTy, targetIndiceShape);
    SmallVector<Value> convertedIndiceVec =
        pack(indiceVec, convertedIndiceTypes, targetIndiceShape, loc, rewriter);

    SmallVector<Value> newOps;

    // More indices is need when chunkSize > 1. Since a big load from one
    // address could be break into multiple small loads.
    if (originalChunkSize > 1) {
      int64_t blockedChunkSize = targetShape->back();
      int64_t numNewChunks = originalChunkSize / blockedChunkSize;

      for (auto [indice, indiceType] :
           llvm::zip(convertedIndiceVec, convertedIndiceTypes)) {
        for (int64_t i = 0; i < numNewChunks; ++i) {
          // Compute the offset
          Value inc = rewriter.create<arith::ConstantIndexOp>(
              loc, i * blockedChunkSize);
          Value incVec = rewriter.create<vector::SplatOp>(loc, indiceType, inc);
          Value offsetIndice =
              rewriter.create<arith::AddIOp>(loc, indice, incVec);

          auto newOp = rewriter.create<xegpu::CreateDescOp>(
              loc, newTdescTy, op.getSource(), offsetIndice);

          newOps.push_back(newOp);
        }
      }
    } else {
      for (auto indice : convertedIndiceVec) {
        auto newOp = rewriter.create<xegpu::CreateDescOp>(
            loc, newTdescTy, op.getSource(), indice);
        newOps.push_back(newOp);
      }
    }

    Value castOp = unpack(newOps, tdescTy, *targetShape, loc, rewriter);
    rewriter.replaceOp(op, castOp);

    return success();
  }
};

struct UnrollLoadGatherOp : public UnrollPattern<xegpu::LoadGatherOp> {
  using UnrollPattern<xegpu::LoadGatherOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::LoadGatherOp op,
                                PatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    VectorType valueTy = llvm::dyn_cast<VectorType>(op.getValue().getType());
    xegpu::TensorDescType tdescTy = op.getTensorDescType();

    if (!tdescTy.isScattered())
      return failure();

    std::optional<SmallVector<int64_t>> targetShape = getTargetShape(op);
    if (!targetShape)
      return failure();

    SmallVector<int64_t> targetMaskShape(*targetShape);
    int64_t originalChunkSize = tdescTy.getChunkSize();

    VectorType maskTy = llvm::dyn_cast<VectorType>(op.getMask().getType());

    Type elemTy = tdescTy.getElementType();
    VectorType newValueTy = valueTy.cloneWith(*targetShape, elemTy);

    SmallVector<Type> convertedTdescTypes =
        getUnrolledTypes(tdescTy, *targetShape);
    SmallVector<Value> convertedTdescs = pack(
        op.getTensorDesc(), convertedTdescTypes, *targetShape, loc, rewriter);

    SmallVector<Type> convertedMaskTypes;
    SmallVector<Value> convertedMasks;

    if (originalChunkSize > 1) {
      targetMaskShape.pop_back();
      convertedMaskTypes = getUnrolledTypes(maskTy, targetMaskShape);
      SmallVector<Value> convertedMasks1D = pack(
          op.getMask(), convertedMaskTypes, targetMaskShape, loc, rewriter);
      int64_t blockedChunkSize = targetShape->back();
      int64_t numNewChunks = originalChunkSize / blockedChunkSize;

      for (auto mask : convertedMasks1D) {
        for (int64_t i = 0; i < numNewChunks; ++i)
          convertedMasks.push_back(mask);
      }
      // This is to handle the transpose effect when chunkSize > 1.
      std::swap((*targetShape)[0], (*targetShape)[1]);
      newValueTy = valueTy.cloneWith(*targetShape, elemTy);
    } else {
      convertedMaskTypes = getUnrolledTypes(maskTy, targetMaskShape);
      convertedMasks = pack(op.getMask(), convertedMaskTypes, targetMaskShape,
                            loc, rewriter);
    }

    SmallVector<Value> newOps;
    for (auto [t, m] : llvm::zip(convertedTdescs, convertedMasks)) {
      auto newOp = rewriter.create<xegpu::LoadGatherOp>(
          loc, newValueTy, t, m, op.getTransposeAttr(), op.getL1HintAttr(),
          op.getL2HintAttr(), op.getL3HintAttr());
      newOps.push_back(newOp);
    }

    Value castOp = unpack(newOps, op.getType(), *targetShape, loc, rewriter);
    rewriter.replaceOp(op, castOp);
    return success();
  }
};

struct UnrollPrefetchOp : public UnrollPattern<xegpu::PrefetchOp> {
  using UnrollPattern<xegpu::PrefetchOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::PrefetchOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    xegpu::TensorDescType tdescTy = op.getTensorDescType();

    if (!tdescTy.isScattered())
      return failure();

    std::optional<SmallVector<int64_t>> targetShape = getTargetShape(op);
    if (!targetShape)
      return failure();

    SmallVector<Type> convertedTdescTypes =
        getUnrolledTypes(tdescTy, *targetShape);
    SmallVector<Value> convertedTdesc = pack(
        op.getTensorDesc(), convertedTdescTypes, *targetShape, loc, rewriter);

    for (auto t : convertedTdesc)
      rewriter.create<xegpu::PrefetchOp>(loc, TypeRange(), t, op->getAttrs());

    rewriter.eraseOp(op);
    return success();
  }
};

struct UnrollStoreScatterOp : public UnrollPattern<xegpu::StoreScatterOp> {
  using UnrollPattern<xegpu::StoreScatterOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::StoreScatterOp op,
                                PatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    VectorType valueTy = llvm::dyn_cast<VectorType>(op.getValue().getType());
    xegpu::TensorDescType tdescTy = op.getTensorDescType();

    if (!tdescTy.isScattered())
      return failure();

    std::optional<SmallVector<int64_t>> targetShape = getTargetShape(op);
    if (!targetShape)
      return failure();

    SmallVector<int64_t> targetIndiceShape(*targetShape);
    int64_t originalChunkSize = tdescTy.getChunkSize();

    VectorType maskTy = llvm::dyn_cast<VectorType>(op.getMask().getType());

    SmallVector<Type> convertedTdescTypes =
        getUnrolledTypes(tdescTy, *targetShape);
    SmallVector<Value> convertedTdescs = pack(
        op.getTensorDesc(), convertedTdescTypes, *targetShape, loc, rewriter);

    SmallVector<Type> convertedMaskTypes;
    SmallVector<Value> convertedMasks;

    if (originalChunkSize > 1) {
      int64_t blockedChunkSize = targetShape->back();
      int64_t numNewChunks = originalChunkSize / blockedChunkSize;
      convertedMaskTypes = getUnrolledTypes(maskTy, (*targetShape)[0]);
      SmallVector<Value> convertedMasks1D = pack(
          op.getMask(), convertedMaskTypes, (*targetShape)[0], loc, rewriter);

      for (auto mask : convertedMasks1D) {
        for (int64_t i = 0; i < numNewChunks; ++i) {
          convertedMasks.push_back(mask);
        }
      }
      // This is to handle the transpose effect when chunkSize > 1.
      std::swap((*targetShape)[0], (*targetShape)[1]);

    } else {
      convertedMaskTypes = getUnrolledTypes(maskTy, *targetShape);
      convertedMasks =
          pack(op.getMask(), convertedMaskTypes, *targetShape, loc, rewriter);
    }

    SmallVector<Type> convertedValTypes =
        getUnrolledTypes(valueTy, *targetShape);
    SmallVector<Value> convertedValues =
        pack(op.getValue(), convertedValTypes, *targetShape, loc, rewriter);

    for (size_t i = 0; i < convertedValues.size(); ++i) {
      Value v = convertedValues[i];
      Value t = convertedTdescs[i];
      Value m = op.getMask() ? convertedMasks[i] : nullptr;
      rewriter.create<xegpu::StoreScatterOp>(
          loc, v, t, m, op.getTransposeAttr(), op.getL1HintAttr(),
          op.getL2HintAttr(), op.getL3HintAttr());
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct UnrollUpdateOffsetOp : public UnrollPattern<xegpu::UpdateOffsetOp> {
  using UnrollPattern<xegpu::UpdateOffsetOp>::UnrollPattern;
  LogicalResult matchAndRewrite(xegpu::UpdateOffsetOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    xegpu::TensorDescType tdescTy = op.getTensorDescType();

    if (tdescTy.getRank() > 2)
      return failure();

    if (!tdescTy.isScattered())
      return failure();

    std::optional<SmallVector<int64_t>> targetShape = getTargetShape(op);
    if (!targetShape)
      return failure();

    SmallVector<Type> convertedTdescTypes =
        getUnrolledTypes(tdescTy, *targetShape);
    SmallVector<Value> convertedTdesc = pack(
        op.getTensorDesc(), convertedTdescTypes, *targetShape, loc, rewriter);

    TypedValue<::mlir::VectorType> offsetVec = op.getOffsets();
    VectorType offsetVecTy = offsetVec.getType();
    SmallVector<Type> convertedOffsetTypes;
    SmallVector<Value> convertedOffsetVec;
    SmallVector<Value> newOps;
    int64_t originalChunkSize = tdescTy.getChunkSize();
    if (originalChunkSize > 1) {
      SmallVector<int64_t> shape1D(targetShape->begin(),
                                   targetShape->end() - 1);
      convertedOffsetTypes = getUnrolledTypes(offsetVecTy, shape1D);
      SmallVector<Value> convertedOffsetVec1D =
          pack(offsetVec, convertedOffsetTypes, shape1D, loc, rewriter);

      int64_t blockedChunkSize = targetShape->back();
      int64_t numNewChunks = originalChunkSize / blockedChunkSize;

      for (auto offset : convertedOffsetVec1D) {
        for (int64_t i = 0; i < numNewChunks; ++i) {
          convertedOffsetVec.push_back(offset);
        }
      }

    } else {
      convertedOffsetTypes = getUnrolledTypes(offsetVecTy, *targetShape);
      convertedOffsetVec =
          pack(offsetVec, convertedOffsetTypes, *targetShape, loc, rewriter);
    }

    for (auto [t, o] : llvm::zip(convertedTdesc, convertedOffsetVec)) {
      auto newOp =
          rewriter.create<xegpu::UpdateOffsetOp>(loc, t.getType(), t, o);
      newOps.push_back(newOp);
    }
    Value castOp = unpack(newOps, op.getType(), *targetShape, loc, rewriter);
    rewriter.replaceOp(op, castOp);
    return success();
  }
};

} // namespace

void mlir::xegpu::populateXeGPUUnrollPatterns(
    RewritePatternSet &patterns, const xegpu::UnrollOptions &options) {
  patterns.add<UnrollCreateNdOp, UnrollUpdateNdOffsetOp, UnrollPrefetchNdOp,
               UnrollLoadNdOp, UnrollStoreNdOp, UnrollDpasOp,
               UnrollCreateDescOp, UnrollLoadGatherOp, UnrollStoreScatterOp,
               UnrollPrefetchOp, UnrollUpdateOffsetOp>(patterns.getContext(),
                                                       options);
}
