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
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Support/MathExtras.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include <numeric>
#include <optional>

#define DEBUG_TYPE "vector-unroll"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::vector;

/// Compute the indices of the slice `index` for a tranfer op.
static SmallVector<Value> sliceTransferIndices(ArrayRef<int64_t> elementOffsets,
                                               ArrayRef<Value> indices,
                                               AffineMap permutationMap,
                                               Location loc,
                                               OpBuilder &builder) {
  MLIRContext *ctx = builder.getContext();
  auto isBroadcast = [](AffineExpr expr) {
    if (auto constExpr = expr.dyn_cast<AffineConstantExpr>())
      return constExpr.getValue() == 0;
    return false;
  };
  // Compute 'sliceIndices' by adding 'sliceOffsets[i]' to 'indices[i]'.
  SmallVector<Value> slicedIndices(indices.begin(), indices.end());
  for (const auto &dim : llvm::enumerate(permutationMap.getResults())) {
    if (isBroadcast(dim.value()))
      continue;
    unsigned pos = dim.value().cast<AffineDimExpr>().getPosition();
    auto expr = getAffineDimExpr(0, builder.getContext()) +
                getAffineConstantExpr(elementOffsets[dim.index()], ctx);
    auto map = AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0, expr);
    slicedIndices[pos] =
        builder.create<affine::AffineApplyOp>(loc, map, indices[pos]);
  }
  return slicedIndices;
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
  LDBG("");
  LDBG("Get unroll shape for op " << op->getName().getStringRef());
  if (options.filterConstraint && failed(options.filterConstraint(op))) {
    LDBG("--no filter constraint -> BAIL");
    return std::nullopt;
  }
  assert(options.nativeShape &&
         "vector unrolling expects the native shape or native"
         "shape call back function to be set");
  auto unrollableVectorOp = dyn_cast<VectorUnrollOpInterface>(op);
  if (!unrollableVectorOp) {
    LDBG("--not an unrollable op -> BAIL");
    return std::nullopt;
  }
  auto maybeUnrollShape = unrollableVectorOp.getShapeForUnroll();
  if (!maybeUnrollShape) {
    LDBG("--could not get shape of op " << *op << " -> BAIL");
    return std::nullopt;
  }
  LLVM_DEBUG(
      llvm::interleaveComma(*maybeUnrollShape, DBGS() << "--vector op shape: ");
      llvm::dbgs() << "\n";);

  std::optional<SmallVector<int64_t>> targetShape = options.nativeShape(op);
  if (!targetShape) {
    LDBG("--no unrolling target shape defined " << *op << "-> SKIP");
    return std::nullopt;
  }
  LLVM_DEBUG(llvm::interleaveComma(*targetShape, DBGS() << "--target shape: ");
             llvm::dbgs() << "\n";);

  auto maybeShapeRatio = computeShapeRatio(*maybeUnrollShape, *targetShape);
  if (!maybeShapeRatio) {
    LDBG("--could not compute integral shape ratio -> BAIL");
    return std::nullopt;
  }
  if (llvm::all_of(*maybeShapeRatio, [](int64_t v) { return v == 1; })) {
    LDBG("--no unrolling needed -> SKIP");
    return std::nullopt;
  }
  LDBG("--found an integral shape ratio to unroll to -> SUCCESS");
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
    ArrayRef<int64_t> originalSize = readOp.getVectorType().getShape();

    // Prepare the result vector;
    Value result = rewriter.create<arith::ConstantOp>(
        loc, sourceVectorType, rewriter.getZeroAttr(sourceVectorType));
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
      auto slicedRead = rewriter.create<vector::TransferReadOp>(
          loc, targetType, readOp.getSource(), indices,
          readOp.getPermutationMapAttr(), readOp.getPadding(), readOp.getMask(),
          readOp.getInBoundsAttr());

      result = rewriter.create<vector::InsertStridedSliceOp>(
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
    SmallVector<Value> originalIndices(writeOp.getIndices().begin(),
                                       writeOp.getIndices().end());
    SmallVector<int64_t> loopOrder =
        getUnrollOrder(originalSize.size(), writeOp, options);
    Value resultTensor;
    for (SmallVector<int64_t> elementOffsets :
         StaticTileOffsetRange(originalSize, *targetShape, loopOrder)) {
      Value slicedVector = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, writeOp.getVector(), elementOffsets, *targetShape, strides);
      SmallVector<Value> indices =
          sliceTransferIndices(elementOffsets, originalIndices,
                               writeOp.getPermutationMap(), loc, rewriter);
      Operation *slicedWrite = rewriter.create<vector::TransferWriteOp>(
          loc, slicedVector, resultTensor ? resultTensor : writeOp.getSource(),
          indices, writeOp.getPermutationMapAttr(), writeOp.getInBoundsAttr());
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
    return static_cast<unsigned>(llvm::hash_combine_range(v.begin(), v.end()));
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
        slicesOperands[index] = rewriter.create<vector::ExtractStridedSliceOp>(
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
      auto accIt = accCache.find(accOffets);
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
    Value result = rewriter.create<arith::ConstantOp>(
        loc, dstVecType, rewriter.getZeroAttr(dstVecType));
    for (const auto &it : accCache) {
      SmallVector<int64_t> dstStrides(it.first.size(), 1);
      result = rewriter.create<vector::InsertStridedSliceOp>(
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
      Value slicedOperand = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, reductionOp.getSource(), offsets, *targetShape, operandStrides);
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
      auto accIt = accCache.find(destOffset);
      if (accIt != accCache.end())
        acc = accIt->second;
      else
        acc = rewriter.create<vector::ExtractStridedSliceOp>(
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
    Value result = rewriter.create<arith::ConstantOp>(
        loc, reductionOp.getDestType(),
        rewriter.getZeroAttr(reductionOp.getDestType()));
    for (const auto &it : accCache) {
      SmallVector<int64_t> dstStrides(it.first.size(), 1);
      result = rewriter.create<vector::InsertStridedSliceOp>(
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
    auto dstVecType = cast<VectorType>(op->getResult(0).getType());
    SmallVector<int64_t> originalSize =
        *cast<VectorUnrollOpInterface>(op).getShapeForUnroll();
    Location loc = op->getLoc();
    // Prepare the result vector.
    Value result = rewriter.create<arith::ConstantOp>(
        loc, dstVecType, rewriter.getZeroAttr(dstVecType));
    SmallVector<int64_t> strides(targetShape->size(), 1);
    VectorType newVecType =
        VectorType::get(*targetShape, dstVecType.getElementType());

    // Create the unrolled computation.
    for (SmallVector<int64_t> offsets :
         StaticTileOffsetRange(originalSize, *targetShape)) {
      SmallVector<Value> extractOperands;
      for (OpOperand &operand : op->getOpOperands()) {
        auto vecType = dyn_cast<VectorType>(operand.get().getType());
        if (!vecType) {
          extractOperands.push_back(operand.get());
          continue;
        }
        extractOperands.push_back(
            rewriter.create<vector::ExtractStridedSliceOp>(
                loc, operand.get(), offsets, *targetShape, strides));
      }
      Operation *newOp = cloneOpWithOperandsAndTypes(
          rewriter, loc, op, extractOperands, newVecType);
      result = rewriter.create<vector::InsertStridedSliceOp>(
          loc, newOp->getResult(0), result, offsets, strides);
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
      Value slicedOperand = rewriter.create<vector::ExtractStridedSliceOp>(
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
    Value result = rewriter.create<arith::ConstantOp>(
        loc, originalVectorType, rewriter.getZeroAttr(originalVectorType));
    SmallVector<int64_t> permutation;
    transposeOp.getTransp(permutation);

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
      Value slicedOperand = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, transposeOp.getVector(), permutedOffsets, permutedShape,
          strides);
      Value transposedSlice =
          rewriter.create<vector::TransposeOp>(loc, slicedOperand, permutation);
      result = rewriter.create<vector::InsertStridedSliceOp>(
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
    Value result = rewriter.create<arith::ConstantOp>(
        loc, sourceVectorType, rewriter.getZeroAttr(sourceVectorType));
    auto targetType =
        VectorType::get(*targetShape, sourceVectorType.getElementType());

    SmallVector<int64_t> loopOrder =
        getUnrollOrder(originalSize.size(), gatherOp, options);
    for (SmallVector<int64_t> elementOffsets :
         StaticTileOffsetRange(originalSize, *targetShape, loopOrder)) {
      // To get the unrolled gather, extract the same slice based on the
      // decomposed shape from each of the index, mask, and pass-through
      // vectors.
      Value indexSubVec = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, gatherOp.getIndexVec(), elementOffsets, *targetShape, strides);
      Value maskSubVec = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, gatherOp.getMask(), elementOffsets, *targetShape, strides);
      Value passThruSubVec = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, gatherOp.getPassThru(), elementOffsets, *targetShape, strides);
      auto slicedGather = rewriter.create<vector::GatherOp>(
          loc, targetType, gatherOp.getBase(), gatherOp.getIndices(),
          indexSubVec, maskSubVec, passThruSubVec);

      result = rewriter.create<vector::InsertStridedSliceOp>(
          loc, slicedGather, result, elementOffsets, strides);
    }
    rewriter.replaceOp(gatherOp, result);
    return success();
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
               UnrollTransposePattern, UnrollGatherPattern>(
      patterns.getContext(), options, benefit);
}
