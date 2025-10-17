//===- VectorUnrollDistribute.cpp - patterns to do vector unrolling -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to do vector unrolling and vector distribution.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/InterleavedRange.h"
#include <optional>

#define DEBUG_TYPE "vector-unroll"

using namespace mlir;
using namespace mlir::vector;

/// Compute the indices of the slice `index` for a transfer op.
static SmallVector<Value> sliceTransferIndices(ArrayRef<int64_t> elementOffsets,
                                               ArrayRef<Value> indices,
                                               AffineMap permutationMap,
                                               Location loc,
                                               OpBuilder &builder) {
  MLIRContext *ctx = builder.getContext();
  auto isBroadcast = [](AffineExpr expr) {
    if (auto constExpr = dyn_cast<AffineConstantExpr>(expr))
      return constExpr.getValue() == 0;
    return false;
  };
  // Compute 'sliceIndices' by adding 'sliceOffsets[i]' to 'indices[i]'.
  SmallVector<Value> slicedIndices(indices);
  for (const auto &dim : llvm::enumerate(permutationMap.getResults())) {
    if (isBroadcast(dim.value()))
      continue;
    unsigned pos = cast<AffineDimExpr>(dim.value()).getPosition();
    auto expr = getAffineDimExpr(0, builder.getContext()) +
                getAffineConstantExpr(elementOffsets[dim.index()], ctx);
    auto map = AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0, expr);
    slicedIndices[pos] =
        affine::AffineApplyOp::create(builder, loc, map, indices[pos]);
  }
  return slicedIndices;
}

// Compute the new indices by adding `offsets` to `originalIndices`.
// If m < n (m = offsets.size(), n = originalIndices.size()),
// then only the trailing m values in `originalIndices` are updated.
static SmallVector<Value> sliceLoadStoreIndices(PatternRewriter &rewriter,
                                                Location loc,
                                                OperandRange originalIndices,
                                                ArrayRef<int64_t> offsets) {
  assert(offsets.size() <= originalIndices.size() &&
         "Offsets should not exceed the number of original indices");
  SmallVector<Value> indices(originalIndices);

  auto start = indices.size() - offsets.size();
  for (auto [i, offset] : llvm::enumerate(offsets)) {
    if (offset != 0) {
      indices[start + i] = arith::AddIOp::create(
          rewriter, loc, originalIndices[start + i],
          arith::ConstantIndexOp::create(rewriter, loc, offset));
    }
  }
  return indices;
}

// Clones `op` into a new operations that takes `operands` and returns
// `resultTypes`.
static Operation *cloneOpWithOperandsAndTypes(OpBuilder &builder, Location loc,
                                              Operation *op,
                                              ArrayRef<Value> operands,
                                              ArrayRef<Type> resultTypes) {
  return builder.create(loc, op->getName().getIdentifier(), operands,
                        resultTypes, op->getAttrs());
}

/// Return the target shape for unrolling for the given `op`. Return
/// std::nullopt if the op shouldn't be or cannot be unrolled.
static std::optional<SmallVector<int64_t>>
getTargetShape(const vector::UnrollVectorOptions &options, Operation *op) {
  LDBG() << "Get unroll shape for op " << op->getName().getStringRef();
  if (options.filterConstraint && failed(options.filterConstraint(op))) {
    LDBG() << "--no filter constraint -> BAIL";
    return std::nullopt;
  }
  assert(options.nativeShape &&
         "vector unrolling expects the native shape or native"
         "shape call back function to be set");
  auto unrollableVectorOp = dyn_cast<VectorUnrollOpInterface>(op);
  if (!unrollableVectorOp) {
    LDBG() << "--not an unrollable op -> BAIL";
    return std::nullopt;
  }
  auto maybeUnrollShape = unrollableVectorOp.getShapeForUnroll();
  if (!maybeUnrollShape) {
    LDBG() << "--could not get shape of op " << *op << " -> BAIL";
    return std::nullopt;
  }
  LDBG() << "--vector op shape: " << llvm::interleaved(*maybeUnrollShape);

  std::optional<SmallVector<int64_t>> targetShape = options.nativeShape(op);
  if (!targetShape) {
    LDBG() << "--no unrolling target shape defined " << *op << "-> SKIP";
    return std::nullopt;
  }
  LDBG() << "--target shape: " << llvm::interleaved(*targetShape);

  auto maybeShapeRatio = computeShapeRatio(*maybeUnrollShape, *targetShape);
  if (!maybeShapeRatio) {
    LDBG() << "--could not compute integral shape ratio -> BAIL";
    return std::nullopt;
  }
  if (llvm::all_of(*maybeShapeRatio, [](int64_t v) { return v == 1; })) {
    LDBG() << "--no unrolling needed -> SKIP";
    return std::nullopt;
  }
  LDBG() << "--found an integral shape ratio to unroll to -> SUCCESS";
  return targetShape;
}

static SmallVector<int64_t>
getUnrollOrder(unsigned numLoops, Operation *op,
               const vector::UnrollVectorOptions &options) {
  SmallVector<int64_t> loopOrder =
      llvm::to_vector(llvm::seq<int64_t>(0, static_cast<int64_t>(numLoops)));
  if (options.traversalOrderCallback != nullptr) {
    std::optional<SmallVector<int64_t>> order =
        options.traversalOrderCallback(op);
    if (order) {
      loopOrder = std::move(*order);
    }
  }
  return loopOrder;
}

namespace {

struct UnrollTransferReadPattern
    : public OpRewritePattern<vector::TransferReadOp> {
  UnrollTransferReadPattern(MLIRContext *context,
                            const vector::UnrollVectorOptions &options,
                            PatternBenefit benefit = 1)
      : OpRewritePattern<vector::TransferReadOp>(context, benefit),
        options(options) {}

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {
    // TODO: support 0-d corner case.
    if (readOp.getTransferRank() == 0)
      return failure();
    if (readOp.getMask())
      return failure();
    auto targetShape = getTargetShape(options, readOp);
    if (!targetShape)
      return failure();
    auto sourceVectorType = readOp.getVectorType();
    SmallVector<int64_t> strides(targetShape->size(), 1);
    Location loc = readOp.getLoc();
    ArrayRef<int64_t> originalSize = sourceVectorType.getShape();

    // Prepare the result vector;
    Value result =
        arith::ConstantOp::create(rewriter, loc, sourceVectorType,
                                  rewriter.getZeroAttr(sourceVectorType));
    auto targetType =
        VectorType::get(*targetShape, sourceVectorType.getElementType());
    SmallVector<Value> originalIndices(readOp.getIndices().begin(),
                                       readOp.getIndices().end());
    SmallVector<int64_t> loopOrder =
        getUnrollOrder(originalSize.size(), readOp, options);
    for (SmallVector<int64_t> elementOffsets :
         StaticTileOffsetRange(originalSize, *targetShape, loopOrder)) {
      SmallVector<Value> indices =
          sliceTransferIndices(elementOffsets, originalIndices,
                               readOp.getPermutationMap(), loc, rewriter);
      auto slicedRead = vector::TransferReadOp::create(
          rewriter, loc, targetType, readOp.getBase(), indices,
          readOp.getPermutationMapAttr(), readOp.getPadding(), readOp.getMask(),
          readOp.getInBoundsAttr());

      result = rewriter.createOrFold<vector::InsertStridedSliceOp>(
          loc, slicedRead, result, elementOffsets, strides);
    }
    rewriter.replaceOp(readOp, result);
    return success();
  }

private:
  vector::UnrollVectorOptions options;
};

struct UnrollTransferWritePattern
    : public OpRewritePattern<vector::TransferWriteOp> {
  UnrollTransferWritePattern(MLIRContext *context,
                             const vector::UnrollVectorOptions &options,
                             PatternBenefit benefit = 1)
      : OpRewritePattern<vector::TransferWriteOp>(context, benefit),
        options(options) {}

  LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                PatternRewriter &rewriter) const override {
    // TODO: support 0-d corner case.
    if (writeOp.getTransferRank() == 0)
      return failure();

    if (writeOp.getMask())
      return failure();
    auto targetShape = getTargetShape(options, writeOp);
    if (!targetShape)
      return failure();
    auto sourceVectorType = writeOp.getVectorType();
    SmallVector<int64_t> strides(targetShape->size(), 1);
    Location loc = writeOp.getLoc();
    ArrayRef<int64_t> originalSize = sourceVectorType.getShape();
    // Bail-out if rank(source) != rank(target). The main limitation here is the
    // fact that `ExtractStridedSlice` requires the rank for the input and
    // output to match. If needed, we can relax this later.
    if (originalSize.size() != targetShape->size())
      return rewriter.notifyMatchFailure(
          writeOp,
          "expected source input vector rank to match target shape rank");

    SmallVector<Value> originalIndices(writeOp.getIndices().begin(),
                                       writeOp.getIndices().end());
    SmallVector<int64_t> loopOrder =
        getUnrollOrder(originalSize.size(), writeOp, options);
    Value resultTensor;
    for (SmallVector<int64_t> elementOffsets :
         StaticTileOffsetRange(originalSize, *targetShape, loopOrder)) {
      Value slicedVector = rewriter.createOrFold<vector::ExtractStridedSliceOp>(
          loc, writeOp.getVector(), elementOffsets, *targetShape, strides);
      SmallVector<Value> indices =
          sliceTransferIndices(elementOffsets, originalIndices,
                               writeOp.getPermutationMap(), loc, rewriter);
      Operation *slicedWrite = vector::TransferWriteOp::create(
          rewriter, loc, slicedVector,
          resultTensor ? resultTensor : writeOp.getBase(), indices,
          writeOp.getPermutationMapAttr(), writeOp.getInBoundsAttr());
      // For the tensor case update the destination for the next transfer write.
      if (!slicedWrite->getResults().empty())
        resultTensor = slicedWrite->getResult(0);
    }
    if (resultTensor)
      rewriter.replaceOp(writeOp, resultTensor);
    else
      rewriter.eraseOp(writeOp);
    return success();
  }

private:
  vector::UnrollVectorOptions options;
};

struct OffsetMapInfo {
  static SmallVector<int64_t> getEmptyKey() { return {int64_t(-1)}; }

  static SmallVector<int64_t> getTombstoneKey() { return {int64_t(-2)}; }

  static unsigned getHashValue(const SmallVector<int64_t> &v) {
    return static_cast<unsigned>(llvm::hash_combine_range(v));
  }

  static bool isEqual(const SmallVector<int64_t> &lhs,
                      const SmallVector<int64_t> &rhs) {
    return lhs == rhs;
  }
};

struct UnrollContractionPattern
    : public OpRewritePattern<vector::ContractionOp> {
  UnrollContractionPattern(MLIRContext *context,
                           const vector::UnrollVectorOptions &options,
                           PatternBenefit benefit = 1)
      : OpRewritePattern<vector::ContractionOp>(context, benefit),
        options(options) {}

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {
    auto targetShape = getTargetShape(options, contractOp);
    if (!targetShape)
      return failure();
    auto dstVecType = cast<VectorType>(contractOp.getResultType());
    SmallVector<int64_t> originalSize = *contractOp.getShapeForUnroll();

    Location loc = contractOp.getLoc();
    unsigned accIndex = vector::ContractionOp::getAccOperandIndex();
    AffineMap dstAffineMap = contractOp.getIndexingMapsArray()[accIndex];
    llvm::MapVector<
        SmallVector<int64_t>, Value,
        llvm::DenseMap<SmallVector<int64_t>, unsigned, OffsetMapInfo>>
        accCache;

    SmallVector<int64_t> loopOrder = getUnrollOrder(
        contractOp.getIteratorTypes().size(), contractOp, options);

    for (SmallVector<int64_t> offsets :
         StaticTileOffsetRange(originalSize, *targetShape, loopOrder)) {
      SmallVector<Value> slicesOperands(contractOp.getNumOperands());

      // Helper to compute the new shape of each operand and extract the slice.
      auto extractOperand = [&](unsigned index, Value operand,
                                AffineMap permutationMap,
                                ArrayRef<int64_t> operandOffets) {
        SmallVector<int64_t> operandShape = applyPermutationMap(
            permutationMap, ArrayRef<int64_t>(*targetShape));
        SmallVector<int64_t> operandStrides(operandOffets.size(), 1);
        slicesOperands[index] =
            rewriter.createOrFold<vector::ExtractStridedSliceOp>(
                loc, operand, operandOffets, operandShape, operandStrides);
      };

      // Extract the new lhs operand.
      AffineMap lhsPermutationMap = contractOp.getIndexingMapsArray()[0];
      SmallVector<int64_t> lhsOffets =
          applyPermutationMap(lhsPermutationMap, ArrayRef<int64_t>(offsets));
      extractOperand(0, contractOp.getLhs(), lhsPermutationMap, lhsOffets);

      // Extract the new rhs operand.
      AffineMap rhsPermutationMap = contractOp.getIndexingMapsArray()[1];
      SmallVector<int64_t> rhsOffets =
          applyPermutationMap(rhsPermutationMap, ArrayRef<int64_t>(offsets));
      extractOperand(1, contractOp.getRhs(), rhsPermutationMap, rhsOffets);

      AffineMap accPermutationMap = contractOp.getIndexingMapsArray()[2];
      SmallVector<int64_t> accOffets =
          applyPermutationMap(accPermutationMap, ArrayRef<int64_t>(offsets));
      // If a version of the accumulator has already been computed, use it
      // otherwise extract the first version from the original operand.
      auto *accIt = accCache.find(accOffets);
      if (accIt != accCache.end())
        slicesOperands[2] = accIt->second;
      else
        extractOperand(2, contractOp.getAcc(), accPermutationMap, accOffets);

      SmallVector<int64_t> dstShape =
          applyPermutationMap(dstAffineMap, ArrayRef<int64_t>(*targetShape));
      auto targetType = VectorType::get(dstShape, dstVecType.getElementType());
      Operation *newOp = cloneOpWithOperandsAndTypes(
          rewriter, loc, contractOp, slicesOperands, targetType);

      SmallVector<int64_t> dstOffets =
          applyPermutationMap(dstAffineMap, ArrayRef<int64_t>(offsets));
      // Save the accumulated value untill all the loops are unrolled since
      // reduction loop keep updating the accumulator.
      accCache[dstOffets] = newOp->getResult(0);
    }
    // Assemble back the accumulator into a single vector.
    Value result = arith::ConstantOp::create(rewriter, loc, dstVecType,
                                             rewriter.getZeroAttr(dstVecType));
    for (const auto &it : accCache) {
      SmallVector<int64_t> dstStrides(it.first.size(), 1);
      result = rewriter.createOrFold<vector::InsertStridedSliceOp>(
          loc, it.second, result, it.first, dstStrides);
    }
    rewriter.replaceOp(contractOp, result);
    return success();
  }

private:
  vector::UnrollVectorOptions options;
};

struct UnrollMultiReductionPattern
    : public OpRewritePattern<vector::MultiDimReductionOp> {
  UnrollMultiReductionPattern(MLIRContext *context,
                              const vector::UnrollVectorOptions &options,
                              PatternBenefit benefit = 1)
      : OpRewritePattern<vector::MultiDimReductionOp>(context, benefit),
        options(options) {}

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp reductionOp,
                                PatternRewriter &rewriter) const override {
    auto resultType = reductionOp->getResult(0).getType();
    if (resultType.isIntOrFloat()) {
      return rewriter.notifyMatchFailure(reductionOp,
                                         "Unrolling scalars is not supported");
    }
    std::optional<SmallVector<int64_t>> targetShape =
        getTargetShape(options, reductionOp);
    if (!targetShape)
      return failure();
    SmallVector<int64_t> originalSize = *reductionOp.getShapeForUnroll();
    llvm::MapVector<
        SmallVector<int64_t>, Value,
        llvm::DenseMap<SmallVector<int64_t>, unsigned, OffsetMapInfo>>
        accCache;
    Location loc = reductionOp.getLoc();

    // Stride of the ratios, this gives us the offsets of sliceCount in a basis
    // of multiples of the targetShape.
    for (SmallVector<int64_t> offsets :
         StaticTileOffsetRange(originalSize, *targetShape)) {
      SmallVector<Value> operands;
      SmallVector<int64_t> operandStrides(offsets.size(), 1);
      Value slicedOperand =
          rewriter.createOrFold<vector::ExtractStridedSliceOp>(
              loc, reductionOp.getSource(), offsets, *targetShape,
              operandStrides);
      operands.push_back(slicedOperand);
      SmallVector<int64_t> dstShape;
      SmallVector<int64_t> destOffset;
      for (size_t i : llvm::seq(size_t(0), targetShape->size())) {
        if (!reductionOp.isReducedDim(i)) {
          destOffset.push_back(offsets[i]);
          dstShape.push_back((*targetShape)[i]);
        }
      }
      Value acc;
      SmallVector<int64_t> accStrides(destOffset.size(), 1);
      // If a version of the accumulator has already been computed, use it
      // otherwise extract the first version from the original operand.
      auto *accIt = accCache.find(destOffset);
      if (accIt != accCache.end())
        acc = accIt->second;
      else
        acc = rewriter.createOrFold<vector::ExtractStridedSliceOp>(
            loc, reductionOp.getAcc(), destOffset, dstShape, accStrides);
      operands.push_back(acc);
      auto targetType = VectorType::get(
          dstShape, reductionOp.getSourceVectorType().getElementType());
      Operation *newOp = cloneOpWithOperandsAndTypes(rewriter, loc, reductionOp,
                                                     operands, targetType);
      Value result = newOp->getResult(0);
      accCache[destOffset] = result;
    }
    // Assemble back the accumulator into a single vector.
    Value result = arith::ConstantOp::create(
        rewriter, loc, reductionOp.getDestType(),
        rewriter.getZeroAttr(reductionOp.getDestType()));
    for (const auto &it : accCache) {
      SmallVector<int64_t> dstStrides(it.first.size(), 1);
      result = rewriter.createOrFold<vector::InsertStridedSliceOp>(
          loc, it.second, result, it.first, dstStrides);
    }
    rewriter.replaceOp(reductionOp, result);
    return success();
  }

private:
  vector::UnrollVectorOptions options;
};

struct UnrollElementwisePattern : public RewritePattern {
  UnrollElementwisePattern(MLIRContext *context,
                           const vector::UnrollVectorOptions &options,
                           PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context),
        options(options) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!OpTrait::hasElementwiseMappableTraits(op) || op->getNumResults() != 1)
      return failure();
    auto targetShape = getTargetShape(options, op);
    if (!targetShape)
      return failure();
    int64_t targetShapeRank = targetShape->size();
    auto dstVecType = cast<VectorType>(op->getResult(0).getType());
    SmallVector<int64_t> originalSize =
        *cast<VectorUnrollOpInterface>(op).getShapeForUnroll();
    int64_t originalShapeRank = originalSize.size();

    Location loc = op->getLoc();

    // Handle rank mismatch by adding leading unit dimensions to targetShape
    SmallVector<int64_t> adjustedTargetShape(originalShapeRank);
    int64_t rankDiff = originalShapeRank - targetShapeRank;
    std::fill(adjustedTargetShape.begin(),
              adjustedTargetShape.begin() + rankDiff, 1);
    std::copy(targetShape->begin(), targetShape->end(),
              adjustedTargetShape.begin() + rankDiff);

    int64_t adjustedTargetShapeRank = adjustedTargetShape.size();
    // Prepare the result vector.
    Value result = arith::ConstantOp::create(rewriter, loc, dstVecType,
                                             rewriter.getZeroAttr(dstVecType));
    SmallVector<int64_t> strides(adjustedTargetShapeRank, 1);
    VectorType unrolledVecType =
        VectorType::get(*targetShape, dstVecType.getElementType());

    // Create the unrolled computation.
    for (SmallVector<int64_t> offsets :
         StaticTileOffsetRange(originalSize, adjustedTargetShape)) {
      SmallVector<Value> extractOperands;
      for (OpOperand &operand : op->getOpOperands()) {
        auto vecType = dyn_cast<VectorType>(operand.get().getType());
        if (!vecType) {
          extractOperands.push_back(operand.get());
          continue;
        }
        Value extracted = rewriter.createOrFold<vector::ExtractStridedSliceOp>(
            loc, operand.get(), offsets, adjustedTargetShape, strides);

        // Reshape to remove leading unit dims if needed
        if (adjustedTargetShapeRank > targetShapeRank) {
          extracted = rewriter.createOrFold<vector::ShapeCastOp>(
              loc, VectorType::get(*targetShape, vecType.getElementType()),
              extracted);
        }
        extractOperands.push_back(extracted);
      }

      Operation *newOp = cloneOpWithOperandsAndTypes(
          rewriter, loc, op, extractOperands, unrolledVecType);

      Value computeResult = newOp->getResult(0);

      // Use strides sized to targetShape for proper insertion
      SmallVector<int64_t> insertStrides =
          (adjustedTargetShapeRank > targetShapeRank)
              ? SmallVector<int64_t>(targetShapeRank, 1)
              : strides;

      result = rewriter.createOrFold<vector::InsertStridedSliceOp>(
          loc, computeResult, result, offsets, insertStrides);
    }
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  vector::UnrollVectorOptions options;
};

struct UnrollReductionPattern : public OpRewritePattern<vector::ReductionOp> {
  UnrollReductionPattern(MLIRContext *context,
                         const vector::UnrollVectorOptions &options,
                         PatternBenefit benefit = 1)
      : OpRewritePattern<vector::ReductionOp>(context, benefit),
        options(options) {}

  LogicalResult matchAndRewrite(vector::ReductionOp reductionOp,
                                PatternRewriter &rewriter) const override {
    std::optional<SmallVector<int64_t>> targetShape =
        getTargetShape(options, reductionOp);
    if (!targetShape)
      return failure();
    SmallVector<int64_t> originalSize = *reductionOp.getShapeForUnroll();

    // Create unrolled vector reduction.
    Location loc = reductionOp.getLoc();
    Value accumulator = nullptr;
    for (SmallVector<int64_t> offsets :
         StaticTileOffsetRange(originalSize, *targetShape)) {
      SmallVector<int64_t> strides(offsets.size(), 1);
      Value slicedOperand =
          rewriter.createOrFold<vector::ExtractStridedSliceOp>(
              loc, reductionOp.getVector(), offsets, *targetShape, strides);
      Operation *newOp = cloneOpWithOperandsAndTypes(
          rewriter, loc, reductionOp, slicedOperand, reductionOp.getType());
      Value result = newOp->getResult(0);

      if (!accumulator) {
        // This is the first reduction.
        accumulator = result;
      } else {
        // On subsequent reduction, combine with the accumulator.
        accumulator = makeArithReduction(rewriter, loc, reductionOp.getKind(),
                                         accumulator, result);
      }
    }

    rewriter.replaceOp(reductionOp, accumulator);
    return success();
  }

private:
  const vector::UnrollVectorOptions options;
};

struct UnrollTransposePattern : public OpRewritePattern<vector::TransposeOp> {
  UnrollTransposePattern(MLIRContext *context,
                         const vector::UnrollVectorOptions &options,
                         PatternBenefit benefit = 1)
      : OpRewritePattern<vector::TransposeOp>(context, benefit),
        options(options) {}

  LogicalResult matchAndRewrite(vector::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    if (transposeOp.getResultVectorType().getRank() == 0)
      return failure();
    auto targetShape = getTargetShape(options, transposeOp);
    if (!targetShape)
      return failure();
    auto originalVectorType = transposeOp.getResultVectorType();
    SmallVector<int64_t> strides(targetShape->size(), 1);
    Location loc = transposeOp.getLoc();
    ArrayRef<int64_t> originalSize = originalVectorType.getShape();

    // Prepare the result vector;
    Value result =
        arith::ConstantOp::create(rewriter, loc, originalVectorType,
                                  rewriter.getZeroAttr(originalVectorType));
    ArrayRef<int64_t> permutation = transposeOp.getPermutation();

    // Unroll the computation.
    for (SmallVector<int64_t> elementOffsets :
         StaticTileOffsetRange(originalSize, *targetShape)) {
      SmallVector<int64_t> permutedOffsets(elementOffsets.size());
      SmallVector<int64_t> permutedShape(elementOffsets.size());
      // Compute the source offsets and shape.
      for (auto indices : llvm::enumerate(permutation)) {
        permutedOffsets[indices.value()] = elementOffsets[indices.index()];
        permutedShape[indices.value()] = (*targetShape)[indices.index()];
      }
      Value slicedOperand =
          rewriter.createOrFold<vector::ExtractStridedSliceOp>(
              loc, transposeOp.getVector(), permutedOffsets, permutedShape,
              strides);
      Value transposedSlice = rewriter.createOrFold<vector::TransposeOp>(
          loc, slicedOperand, permutation);
      result = rewriter.createOrFold<vector::InsertStridedSliceOp>(
          loc, transposedSlice, result, elementOffsets, strides);
    }
    rewriter.replaceOp(transposeOp, result);
    return success();
  }

private:
  vector::UnrollVectorOptions options;
};

struct UnrollGatherPattern : public OpRewritePattern<vector::GatherOp> {
  UnrollGatherPattern(MLIRContext *context,
                      const vector::UnrollVectorOptions &options,
                      PatternBenefit benefit = 1)
      : OpRewritePattern<vector::GatherOp>(context, benefit), options(options) {
  }

  LogicalResult matchAndRewrite(vector::GatherOp gatherOp,
                                PatternRewriter &rewriter) const override {
    VectorType sourceVectorType = gatherOp.getVectorType();
    if (sourceVectorType.getRank() == 0)
      return failure();
    auto targetShape = getTargetShape(options, gatherOp);
    if (!targetShape)
      return failure();
    SmallVector<int64_t> strides(targetShape->size(), 1);
    Location loc = gatherOp.getLoc();
    ArrayRef<int64_t> originalSize = gatherOp.getVectorType().getShape();

    // Prepare the result vector;
    Value result =
        arith::ConstantOp::create(rewriter, loc, sourceVectorType,
                                  rewriter.getZeroAttr(sourceVectorType));
    auto targetType =
        VectorType::get(*targetShape, sourceVectorType.getElementType());

    SmallVector<int64_t> loopOrder =
        getUnrollOrder(originalSize.size(), gatherOp, options);
    for (SmallVector<int64_t> elementOffsets :
         StaticTileOffsetRange(originalSize, *targetShape, loopOrder)) {
      // To get the unrolled gather, extract the same slice based on the
      // decomposed shape from each of the index, mask, and pass-through
      // vectors.
      Value indexSubVec = rewriter.createOrFold<vector::ExtractStridedSliceOp>(
          loc, gatherOp.getIndices(), elementOffsets, *targetShape, strides);
      Value maskSubVec = rewriter.createOrFold<vector::ExtractStridedSliceOp>(
          loc, gatherOp.getMask(), elementOffsets, *targetShape, strides);
      Value passThruSubVec =
          rewriter.createOrFold<vector::ExtractStridedSliceOp>(
              loc, gatherOp.getPassThru(), elementOffsets, *targetShape,
              strides);
      auto slicedGather = vector::GatherOp::create(
          rewriter, loc, targetType, gatherOp.getBase(), gatherOp.getOffsets(),
          indexSubVec, maskSubVec, passThruSubVec);

      result = rewriter.createOrFold<vector::InsertStridedSliceOp>(
          loc, slicedGather, result, elementOffsets, strides);
    }
    rewriter.replaceOp(gatherOp, result);
    return success();
  }

private:
  vector::UnrollVectorOptions options;
};

struct UnrollLoadPattern : public OpRewritePattern<vector::LoadOp> {
  UnrollLoadPattern(MLIRContext *context,
                    const vector::UnrollVectorOptions &options,
                    PatternBenefit benefit = 1)
      : OpRewritePattern<vector::LoadOp>(context, benefit), options(options) {}

  LogicalResult matchAndRewrite(vector::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    VectorType vecType = loadOp.getVectorType();

    auto targetShape = getTargetShape(options, loadOp);
    if (!targetShape)
      return failure();

    Location loc = loadOp.getLoc();
    ArrayRef<int64_t> originalShape = vecType.getShape();
    SmallVector<int64_t> strides(targetShape->size(), 1);

    Value result = arith::ConstantOp::create(rewriter, loc, vecType,
                                             rewriter.getZeroAttr(vecType));

    SmallVector<int64_t> loopOrder =
        getUnrollOrder(originalShape.size(), loadOp, options);

    auto targetVecType =
        VectorType::get(*targetShape, vecType.getElementType());

    for (SmallVector<int64_t> offsets :
         StaticTileOffsetRange(originalShape, *targetShape, loopOrder)) {
      SmallVector<Value> indices =
          sliceLoadStoreIndices(rewriter, loc, loadOp.getIndices(), offsets);
      Value slicedLoad = vector::LoadOp::create(rewriter, loc, targetVecType,
                                                loadOp.getBase(), indices);
      result = rewriter.createOrFold<vector::InsertStridedSliceOp>(
          loc, slicedLoad, result, offsets, strides);
    }
    rewriter.replaceOp(loadOp, result);
    return success();
  }

private:
  vector::UnrollVectorOptions options;
};

struct UnrollStorePattern : public OpRewritePattern<vector::StoreOp> {
  UnrollStorePattern(MLIRContext *context,
                     const vector::UnrollVectorOptions &options,
                     PatternBenefit benefit = 1)
      : OpRewritePattern<vector::StoreOp>(context, benefit), options(options) {}

  LogicalResult matchAndRewrite(vector::StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    VectorType vecType = storeOp.getVectorType();

    auto targetShape = getTargetShape(options, storeOp);
    if (!targetShape)
      return failure();

    Location loc = storeOp.getLoc();
    ArrayRef<int64_t> originalShape = vecType.getShape();
    SmallVector<int64_t> strides(targetShape->size(), 1);

    Value base = storeOp.getBase();
    Value vector = storeOp.getValueToStore();

    SmallVector<int64_t> loopOrder =
        getUnrollOrder(originalShape.size(), storeOp, options);

    for (SmallVector<int64_t> offsets :
         StaticTileOffsetRange(originalShape, *targetShape, loopOrder)) {
      SmallVector<Value> indices =
          sliceLoadStoreIndices(rewriter, loc, storeOp.getIndices(), offsets);
      Value slice = rewriter.createOrFold<vector::ExtractStridedSliceOp>(
          loc, vector, offsets, *targetShape, strides);
      vector::StoreOp::create(rewriter, loc, slice, base, indices);
    }
    rewriter.eraseOp(storeOp);
    return success();
  }

private:
  vector::UnrollVectorOptions options;
};

struct UnrollBroadcastPattern : public OpRewritePattern<vector::BroadcastOp> {
  UnrollBroadcastPattern(MLIRContext *context,
                         const vector::UnrollVectorOptions &options,
                         PatternBenefit benefit = 1)
      : OpRewritePattern<vector::BroadcastOp>(context, benefit),
        options(options) {}

  LogicalResult matchAndRewrite(vector::BroadcastOp broadcastOp,
                                PatternRewriter &rewriter) const override {
    auto targetShape = getTargetShape(options, broadcastOp);
    if (!targetShape)
      return failure();

    Location loc = broadcastOp.getLoc();
    VectorType srcType = dyn_cast<VectorType>(broadcastOp.getSourceType());
    VectorType resType = broadcastOp.getResultVectorType();
    VectorType targetType =
        resType.cloneWith(*targetShape, resType.getElementType());
    Value result = arith::ConstantOp::create(rewriter, loc, resType,
                                             rewriter.getZeroAttr(resType));

    SmallVector<int64_t> originalShape = *broadcastOp.getShapeForUnroll();
    SmallVector<int64_t> strides(originalShape.size(), 1);

    for (SmallVector<int64_t> offsets :
         StaticTileOffsetRange(originalShape, *targetShape)) {
      Value newSrc;
      if (!srcType) {
        // Scalar to vector broadcast.
        newSrc = broadcastOp.getSource();
      } else {
        // Vector to vector broadcast.
        int64_t rank = srcType.getRank();
        SmallVector<int64_t> srcOffsets(offsets.end() - rank, offsets.end());
        SmallVector<int64_t> srcShape(targetShape->end() - rank,
                                      targetShape->end());
        SmallVector<int64_t> srcStrides(strides.end() - rank, strides.end());
        // adjust the offset and shape for src if the corresponding dim is 1.
        for (int64_t i = 0; i < rank; ++i) {
          if (srcType.getDimSize(i) == 1) {
            srcOffsets[i] = 0;
            srcShape[i] = 1;
          }
        }
        newSrc = rewriter.createOrFold<vector::ExtractStridedSliceOp>(
            loc, broadcastOp.getSource(), srcOffsets, srcShape, srcStrides);
      }

      Operation *newOp = cloneOpWithOperandsAndTypes(rewriter, loc, broadcastOp,
                                                     newSrc, targetType);

      result = rewriter.createOrFold<vector::InsertStridedSliceOp>(
          loc, newOp->getResult(0), result, offsets, strides);
    }

    rewriter.replaceOp(broadcastOp, result);
    return success();
  }

private:
  vector::UnrollVectorOptions options;
};

/// Unrolls 2 or more dimensional `vector.to_elements` ops by unrolling the
/// outermost dimension of the operand. For example:
///
/// ```
/// %0:4 = vector.to_elements %v : vector<2x2xf32>
///
/// ==>
///
/// %v0 = vector.extract %v[0] : vector<2x2xf32> from vector<2x2x2xf32>
/// %v1 = vector.extract %v[1] : vector<2x2xf32> from vector<2x2x2xf32>
/// %0:4 = vector.to_elements %v0 : vector<2x2xf32>
/// %1:4 = vector.to_elements %v1 : vector<2x2xf32>
/// ```
///
/// When this pattern is applied until a fixed-point is reached,
/// this will produce a sequence of 1-d from_elements
/// ops.
struct UnrollToElements final : public OpRewritePattern<vector::ToElementsOp> {
  UnrollToElements(MLIRContext *context,
                   const vector::UnrollVectorOptions &options,
                   PatternBenefit benefit = 1)
      : OpRewritePattern<vector::ToElementsOp>(context, benefit),
        options(options) {}

  LogicalResult matchAndRewrite(vector::ToElementsOp op,
                                PatternRewriter &rewriter) const override {

    TypedValue<VectorType> source = op.getSource();
    FailureOr<SmallVector<Value>> result =
        vector::unrollVectorValue(source, rewriter);
    if (failed(result)) {
      return failure();
    }
    SmallVector<Value> vectors = *result;

    SmallVector<Value> results;
    for (Value vector : vectors) {
      auto subElements =
          vector::ToElementsOp::create(rewriter, op.getLoc(), vector);
      llvm::append_range(results, subElements.getResults());
    }
    rewriter.replaceOp(op, results);
    return success();
  }

private:
  vector::UnrollVectorOptions options;
};

/// This pattern unrolls `vector.step` operations according to the provided
/// target unroll shape. It decomposes a large step vector into smaller step
/// vectors (segments) and assembles the result by inserting each computed
/// segment into the appropriate offset of the original vector.
///
/// The pattern does not support scalable vectors and will fail to match them.
///
/// For each segment, it adds the base step vector and the segment's offset,
/// then inserts the result into the output vector at the corresponding
/// position.
///
/// Example:
///   Given a step operation:
///     %0 = vector.step : vector<8xindex>
///
///   and a target unroll shape of <4>, the pattern produces:
///
///     %base = vector.step : vector<4xindex>
///     %zero = arith.constant dense<0> : vector<8xindex>
///     %result0 = vector.insert_strided_slice %base, %zero
///       {offsets = [0], strides = [1]} : vector<4xindex> into vector<8xindex>
///     %offset = arith.constant dense<4> : vector<4xindex>
///     %segment1 = arith.addi %base, %offset : vector<4xindex>
///     %result1 = vector.insert_strided_slice %segment1, %result0
///       {offsets = [4], strides = [1]} : vector<4xindex> into vector<8xindex>
///
struct UnrollStepPattern : public OpRewritePattern<vector::StepOp> {
  UnrollStepPattern(MLIRContext *context,
                    const vector::UnrollVectorOptions &options,
                    PatternBenefit benefit = 1)
      : OpRewritePattern<vector::StepOp>(context, benefit), options(options) {}

  LogicalResult matchAndRewrite(vector::StepOp stepOp,
                                PatternRewriter &rewriter) const override {
    std::optional<SmallVector<int64_t>> targetShape =
        getTargetShape(options, stepOp);
    if (!targetShape)
      return failure();

    VectorType vecType = stepOp.getType();
    if (vecType.isScalable()) {
      // Scalable vectors are not supported by this pattern.
      return failure();
    }
    int64_t originalSize = vecType.getShape()[0];
    Location loc = stepOp.getLoc();
    SmallVector<int64_t> strides(1, 1);

    Value result = arith::ConstantOp::create(rewriter, loc, vecType,
                                             rewriter.getZeroAttr(vecType));

    auto targetVecType =
        VectorType::get(*targetShape, vecType.getElementType());
    Value baseStep = vector::StepOp::create(rewriter, loc, targetVecType);
    for (const SmallVector<int64_t> &offsets :
         StaticTileOffsetRange({originalSize}, *targetShape)) {
      Value bcastOffset = arith::ConstantOp::create(
          rewriter, loc, targetVecType,
          DenseElementsAttr::get(
              targetVecType,
              IntegerAttr::get(targetVecType.getElementType(), offsets[0])));
      Value tileStep =
          arith::AddIOp::create(rewriter, loc, baseStep, bcastOffset);

      result = rewriter.createOrFold<vector::InsertStridedSliceOp>(
          loc, tileStep, result, offsets, strides);
    }
    rewriter.replaceOp(stepOp, result);
    return success();
  }

private:
  vector::UnrollVectorOptions options;
};

/// Unrolls 2 or more dimensional `vector.from_elements` ops by unrolling the
/// outermost dimension. For example:
/// ```
/// %v = vector.from_elements %e0, %e1, %e2, %e3, %e4, %e5 : vector<2x3xf32>
///
/// ==>
///
/// %0   = ub.poison : vector<2x3xf32>
/// %v0  = vector.from_elements %e0, %e1, %e2 : vector<3xf32>
/// %1   = vector.insert %v0, %0 [0] : vector<3xf32> into vector<2x3xf32>
/// %v1  = vector.from_elements %e3, %e4, %e5 : vector<3xf32>
/// %v   = vector.insert %v1, %1 [1] : vector<3xf32> into vector<2x3xf32>
/// ```
///
/// When this pattern is applied until a fixed-point is reached,
/// this will produce a sequence of 1-d from_elements
/// ops.
struct UnrollFromElements : OpRewritePattern<vector::FromElementsOp> {
  UnrollFromElements(MLIRContext *context,
                     const vector::UnrollVectorOptions &options,
                     PatternBenefit benefit = 1)
      : OpRewritePattern<vector::FromElementsOp>(context, benefit),
        options(options) {}

  LogicalResult matchAndRewrite(vector::FromElementsOp op,
                                PatternRewriter &rewriter) const override {
    ValueRange allElements = op.getElements();

    auto unrollFromElementsFn = [&](PatternRewriter &rewriter, Location loc,
                                    VectorType subTy, int64_t index) {
      size_t subTyNumElements = subTy.getNumElements();
      assert((index + 1) * subTyNumElements <= allElements.size() &&
             "out of bounds");
      ValueRange subElements =
          allElements.slice(index * subTyNumElements, subTyNumElements);
      return vector::FromElementsOp::create(rewriter, loc, subTy, subElements);
    };

    return unrollVectorOp(op, rewriter, unrollFromElementsFn);
  }

private:
  vector::UnrollVectorOptions options;
};

} // namespace

void mlir::vector::populateVectorUnrollPatterns(
    RewritePatternSet &patterns, const UnrollVectorOptions &options,
    PatternBenefit benefit) {
  patterns.add<UnrollTransferReadPattern, UnrollTransferWritePattern,
               UnrollContractionPattern, UnrollElementwisePattern,
               UnrollReductionPattern, UnrollMultiReductionPattern,
               UnrollTransposePattern, UnrollGatherPattern, UnrollLoadPattern,
               UnrollStorePattern, UnrollBroadcastPattern, UnrollFromElements,
               UnrollToElements, UnrollStepPattern>(patterns.getContext(),
                                                    options, benefit);
}

void mlir::vector::populateVectorToElementsUnrollPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<UnrollToElements>(patterns.getContext(), UnrollVectorOptions(),
                                 benefit);
}

void mlir::vector::populateVectorFromElementsUnrollPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<UnrollFromElements>(patterns.getContext(), UnrollVectorOptions(),
                                   benefit);
}
