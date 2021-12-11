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
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "vector-unrolling"

using namespace mlir;
using namespace mlir::vector;

/// During unrolling from `originalShape` to `targetShape` return the offset for
/// the slice `index`.
static SmallVector<int64_t, 4> getVectorOffset(ArrayRef<int64_t> originalShape,
                                               ArrayRef<int64_t> targetShape,
                                               int64_t index) {
  SmallVector<int64_t, 4> dstSliceStrides =
      computeStrides(originalShape, targetShape);
  SmallVector<int64_t, 4> vectorOffsets = delinearize(dstSliceStrides, index);
  SmallVector<int64_t, 4> elementOffsets =
      computeElementOffsetsFromVectorSliceOffsets(targetShape, vectorOffsets);
  return elementOffsets;
}

/// Compute the indices of the slice `index` for a tranfer op.
static SmallVector<Value>
sliceTransferIndices(int64_t index, ArrayRef<int64_t> originalShape,
                     ArrayRef<int64_t> targetShape, ArrayRef<Value> indices,
                     AffineMap permutationMap, Location loc,
                     OpBuilder &builder) {
  MLIRContext *ctx = builder.getContext();
  auto isBroadcast = [](AffineExpr expr) {
    if (auto constExpr = expr.dyn_cast<AffineConstantExpr>())
      return constExpr.getValue() == 0;
    return false;
  };
  SmallVector<int64_t, 4> elementOffsets =
      getVectorOffset(originalShape, targetShape, index);
  // Compute 'sliceIndices' by adding 'sliceOffsets[i]' to 'indices[i]'.
  SmallVector<Value> slicedIndices(indices.begin(), indices.end());
  for (auto dim : llvm::enumerate(permutationMap.getResults())) {
    if (isBroadcast(dim.value()))
      continue;
    unsigned pos = dim.value().cast<AffineDimExpr>().getPosition();
    auto expr = getAffineDimExpr(0, builder.getContext()) +
                getAffineConstantExpr(elementOffsets[dim.index()], ctx);
    auto map = AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0, expr);
    slicedIndices[pos] = builder.create<AffineApplyOp>(loc, map, indices[pos]);
  }
  return slicedIndices;
}

// Clones `op` into a new operations that takes `operands` and returns
// `resultTypes`.
static Operation *cloneOpWithOperandsAndTypes(OpBuilder &builder, Location loc,
                                              Operation *op,
                                              ArrayRef<Value> operands,
                                              ArrayRef<Type> resultTypes) {
  OperationState res(loc, op->getName().getStringRef(), operands, resultTypes,
                     op->getAttrs());
  return builder.createOperation(res);
}

/// Return the target shape for unrolling for the given `op`. Return llvm::None
/// if the op shouldn't be or cannot be unrolled.
static Optional<SmallVector<int64_t, 4>>
getTargetShape(const vector::UnrollVectorOptions &options, Operation *op) {
  if (options.filterConstraint && failed(options.filterConstraint(op)))
    return llvm::None;
  assert(options.nativeShape &&
         "vector unrolling expects the native shape or native"
         "shape call back function to be set");
  auto unrollableVectorOp = dyn_cast<VectorUnrollOpInterface>(op);
  if (!unrollableVectorOp)
    return llvm::None;
  auto maybeUnrollShape = unrollableVectorOp.getShapeForUnroll();
  if (!maybeUnrollShape)
    return llvm::None;
  Optional<SmallVector<int64_t, 4>> targetShape = options.nativeShape(op);
  if (!targetShape)
    return llvm::None;
  auto maybeShapeRatio = shapeRatio(*maybeUnrollShape, *targetShape);
  if (!maybeShapeRatio ||
      llvm::all_of(*maybeShapeRatio, [](int64_t v) { return v == 1; }))
    return llvm::None;
  return targetShape;
}

namespace {

struct UnrollTransferReadPattern
    : public OpRewritePattern<vector::TransferReadOp> {
  UnrollTransferReadPattern(MLIRContext *context,
                            const vector::UnrollVectorOptions &options)
      : OpRewritePattern<vector::TransferReadOp>(context, /*benefit=*/1),
        options(options) {}
  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {
    // TODO: support 0-d corner case.
    if (readOp.getTransferRank() == 0)
      return failure();
    if (readOp.mask())
      return failure();
    auto targetShape = getTargetShape(options, readOp);
    if (!targetShape)
      return failure();
    auto sourceVectorType = readOp.getVectorType();
    SmallVector<int64_t, 4> strides(targetShape->size(), 1);
    Location loc = readOp.getLoc();
    ArrayRef<int64_t> originalSize = readOp.getVectorType().getShape();
    SmallVector<int64_t, 4> ratio = *shapeRatio(originalSize, *targetShape);
    // Compute shape ratio of 'shape' and 'sizes'.
    int64_t sliceCount = computeMaxLinearIndex(ratio);
    // Prepare the result vector;
    Value result = rewriter.create<arith::ConstantOp>(
        loc, sourceVectorType, rewriter.getZeroAttr(sourceVectorType));
    auto targetType =
        VectorType::get(*targetShape, sourceVectorType.getElementType());
    SmallVector<Value, 4> originalIndices(readOp.indices().begin(),
                                          readOp.indices().end());
    for (int64_t i = 0; i < sliceCount; i++) {
      SmallVector<Value, 4> indices =
          sliceTransferIndices(i, originalSize, *targetShape, originalIndices,
                               readOp.permutation_map(), loc, rewriter);
      auto slicedRead = rewriter.create<vector::TransferReadOp>(
          loc, targetType, readOp.source(), indices,
          readOp.permutation_mapAttr(), readOp.padding(), readOp.mask(),
          readOp.in_boundsAttr());

      SmallVector<int64_t, 4> elementOffsets =
          getVectorOffset(originalSize, *targetShape, i);
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
                             const vector::UnrollVectorOptions &options)
      : OpRewritePattern<vector::TransferWriteOp>(context, /*benefit=*/1),
        options(options) {}
  LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                PatternRewriter &rewriter) const override {
    // TODO: support 0-d corner case.
    if (writeOp.getTransferRank() == 0)
      return failure();

    if (writeOp.mask())
      return failure();
    auto targetShape = getTargetShape(options, writeOp);
    if (!targetShape)
      return failure();
    auto sourceVectorType = writeOp.getVectorType();
    SmallVector<int64_t, 4> strides(targetShape->size(), 1);
    Location loc = writeOp.getLoc();
    ArrayRef<int64_t> originalSize = sourceVectorType.getShape();
    SmallVector<int64_t, 4> ratio = *shapeRatio(originalSize, *targetShape);
    // Compute shape ratio of 'shape' and 'sizes'.
    int64_t sliceCount = computeMaxLinearIndex(ratio);
    SmallVector<Value, 4> originalIndices(writeOp.indices().begin(),
                                          writeOp.indices().end());
    Value resultTensor;
    for (int64_t i = 0; i < sliceCount; i++) {
      SmallVector<int64_t, 4> elementOffsets =
          getVectorOffset(originalSize, *targetShape, i);
      Value slicedVector = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, writeOp.vector(), elementOffsets, *targetShape, strides);

      SmallVector<Value, 4> indices =
          sliceTransferIndices(i, originalSize, *targetShape, originalIndices,
                               writeOp.permutation_map(), loc, rewriter);
      Operation *slicedWrite = rewriter.create<vector::TransferWriteOp>(
          loc, slicedVector, resultTensor ? resultTensor : writeOp.source(),
          indices, writeOp.permutation_mapAttr(), writeOp.in_boundsAttr());
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

struct UnrollContractionPattern
    : public OpRewritePattern<vector::ContractionOp> {
  struct OffsetMapInfo {
    static SmallVector<int64_t> getEmptyKey() { return {int64_t(-1)}; }

    static SmallVector<int64_t> getTombstoneKey() { return {int64_t(-2)}; }

    static unsigned getHashValue(const SmallVector<int64_t> &v) {
      return static_cast<unsigned>(
          llvm::hash_combine_range(v.begin(), v.end()));
    }

    static bool isEqual(const SmallVector<int64_t> &lhs,
                        const SmallVector<int64_t> &rhs) {
      return lhs == rhs;
    }
  };
  UnrollContractionPattern(MLIRContext *context,
                           const vector::UnrollVectorOptions &options)
      : OpRewritePattern<vector::ContractionOp>(context, /*benefit=*/1),
        options(options) {}

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {
    auto targetShape = getTargetShape(options, contractOp);
    if (!targetShape)
      return failure();
    auto dstVecType = contractOp.getResultType().cast<VectorType>();
    SmallVector<int64_t, 4> originalSize = *contractOp.getShapeForUnroll();
    SmallVector<int64_t, 4> ratio = *shapeRatio(originalSize, *targetShape);

    // Compute shape ratio of 'shape' and 'sizes'.
    int64_t sliceCount = computeMaxLinearIndex(ratio);
    Location loc = contractOp.getLoc();
    unsigned accIndex = vector::ContractionOp::getAccOperandIndex();
    AffineMap dstAffineMap = contractOp.getIndexingMaps()[accIndex];
    llvm::MapVector<
        SmallVector<int64_t>, Value,
        llvm::DenseMap<SmallVector<int64_t>, unsigned, OffsetMapInfo>>
        accCache;
    for (int64_t i = 0; i < sliceCount; i++) {
      SmallVector<int64_t, 4> offsets =
          getVectorOffset(originalSize, *targetShape, i);
      SmallVector<Value, 4> slicesOperands(contractOp.getNumOperands());

      // Helper to coompute the new shape of each operand and extract the slice.
      auto extractOperand = [&](unsigned index, Value operand,
                                AffineMap permutationMap,
                                ArrayRef<int64_t> operandOffets) {
        SmallVector<int64_t> operandShape = applyPermutationMap(
            permutationMap, ArrayRef<int64_t>(*targetShape));
        SmallVector<int64_t, 4> operandStrides(operandOffets.size(), 1);
        slicesOperands[index] = rewriter.create<vector::ExtractStridedSliceOp>(
            loc, operand, operandOffets, operandShape, operandStrides);
      };

      // Extract the new lhs operand.
      AffineMap lhsPermutationMap = contractOp.getIndexingMaps()[0];
      SmallVector<int64_t> lhsOffets =
          applyPermutationMap(lhsPermutationMap, ArrayRef<int64_t>(offsets));
      extractOperand(0, contractOp.lhs(), lhsPermutationMap, lhsOffets);
      // If there is a mask associated to lhs, extract it as well.
      if (slicesOperands.size() > 3)
        extractOperand(3, contractOp.masks()[0], lhsPermutationMap, lhsOffets);

      // Extract the new rhs operand.
      AffineMap rhsPermutationMap = contractOp.getIndexingMaps()[1];
      SmallVector<int64_t> rhsOffets =
          applyPermutationMap(rhsPermutationMap, ArrayRef<int64_t>(offsets));
      extractOperand(1, contractOp.rhs(), rhsPermutationMap, rhsOffets);
      // If there is a mask associated to rhs, extract it as well.
      if (slicesOperands.size() > 4)
        extractOperand(4, contractOp.masks()[1], rhsPermutationMap, rhsOffets);

      AffineMap accPermutationMap = contractOp.getIndexingMaps()[2];
      SmallVector<int64_t> accOffets =
          applyPermutationMap(accPermutationMap, ArrayRef<int64_t>(offsets));
      // If a version of the accumulator has already been computed, use it
      // otherwise extract the first version from the original operand.
      auto accIt = accCache.find(accOffets);
      if (accIt != accCache.end())
        slicesOperands[2] = accIt->second;
      else
        extractOperand(2, contractOp.acc(), accPermutationMap, accOffets);

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

struct UnrollElementwisePattern : public RewritePattern {
  UnrollElementwisePattern(MLIRContext *context,
                           const vector::UnrollVectorOptions &options)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context),
        options(options) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!OpTrait::hasElementwiseMappableTraits(op) || op->getNumResults() != 1)
      return failure();
    auto targetShape = getTargetShape(options, op);
    if (!targetShape)
      return failure();
    auto dstVecType = op->getResult(0).getType().cast<VectorType>();
    SmallVector<int64_t, 4> originalSize =
        *cast<VectorUnrollOpInterface>(op).getShapeForUnroll();
    SmallVector<int64_t, 4> ratio = *shapeRatio(originalSize, *targetShape);
    int64_t sliceCount = computeMaxLinearIndex(ratio);
    Location loc = op->getLoc();
    // Prepare the result vector.
    Value result = rewriter.create<arith::ConstantOp>(
        loc, dstVecType, rewriter.getZeroAttr(dstVecType));
    SmallVector<int64_t, 4> strides(targetShape->size(), 1);
    VectorType newVecType =
        VectorType::get(*targetShape, dstVecType.getElementType());
    for (int64_t i = 0; i < sliceCount; i++) {
      SmallVector<int64_t, 4> offsets =
          getVectorOffset(originalSize, *targetShape, i);
      SmallVector<Value, 4> extractOperands;
      for (OpOperand &operand : op->getOpOperands()) {
        auto vecType = operand.get().getType().template dyn_cast<VectorType>();
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

/// Canonicalize an extract_map using the result of a pointwise operation.
/// Transforms:
/// %v = arith.addf %a, %b : vector32xf32>
/// %dv = vector.extract_map %v[%id] : vector<32xf32> to vector<1xf32>
/// to:
/// %da = vector.extract_map %a[%id] : vector<32xf32> to vector<1xf32>
/// %db = vector.extract_map %a[%id] : vector<32xf32> to vector<1xf32>
/// %dv = arith.addf %da, %db : vector<1xf32>
struct PointwiseExtractPattern : public OpRewritePattern<vector::ExtractMapOp> {
  using OpRewritePattern<vector::ExtractMapOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::ExtractMapOp extract,
                                PatternRewriter &rewriter) const override {
    Operation *definedOp = extract.vector().getDefiningOp();
    if (!definedOp || !OpTrait::hasElementwiseMappableTraits(definedOp) ||
        definedOp->getNumResults() != 1)
      return failure();
    Location loc = extract.getLoc();
    SmallVector<Value, 4> extractOperands;
    for (OpOperand &operand : definedOp->getOpOperands()) {
      auto vecType = operand.get().getType().template dyn_cast<VectorType>();
      if (!vecType) {
        extractOperands.push_back(operand.get());
        continue;
      }
      extractOperands.push_back(rewriter.create<vector::ExtractMapOp>(
          loc,
          VectorType::get(extract.getResultType().getShape(),
                          vecType.getElementType()),
          operand.get(), extract.ids()));
    }
    Operation *newOp = cloneOpWithOperandsAndTypes(
        rewriter, loc, definedOp, extractOperands, extract.getResultType());
    rewriter.replaceOp(extract, newOp->getResult(0));
    return success();
  }
};

/// Canonicalize an extract_map using the result of a contract operation.
/// This propagate the extract_map to operands.
struct ContractExtractPattern : public OpRewritePattern<vector::ExtractMapOp> {
  using OpRewritePattern<vector::ExtractMapOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::ExtractMapOp extract,
                                PatternRewriter &rewriter) const override {
    Operation *definedOp = extract.vector().getDefiningOp();
    auto contract = dyn_cast_or_null<vector::ContractionOp>(definedOp);
    if (!contract)
      return failure();
    Location loc = contract.getLoc();
    unsigned accIndex = vector::ContractionOp::getAccOperandIndex();
    AffineMap affineMap = contract.getIndexingMaps()[accIndex];
    // Create a map of the dimensions distributed based on the acc affine map.
    // Only parallel dimensions are being distributed, reduction dimensions are
    // untouched.
    DenseMap<int64_t, int64_t> map;
    for (unsigned i : llvm::seq(unsigned(0), affineMap.getNumResults()))
      map[affineMap.getDimPosition(i)] = extract.getResultType().getDimSize(i);
    SmallVector<Value, 4> extractOperands;
    for (auto it : llvm::enumerate(contract.getIndexingMaps())) {
      // For each operands calculate the new vector type after distribution.
      Value operand = contract->getOperand(it.index());
      auto vecType = operand.getType().cast<VectorType>();
      SmallVector<int64_t> operandShape(vecType.getShape().begin(),
                                        vecType.getShape().end());
      for (unsigned i : llvm::seq(unsigned(0), it.value().getNumResults())) {
        unsigned dim = it.value().getDimPosition(i);
        auto distributedDim = map.find(dim);
        // If the dimension is not in the map it means it is a reduction and
        // doesn't get distributed.
        if (distributedDim == map.end())
          continue;
        operandShape[i] = distributedDim->second;
      }
      VectorType newVecType =
          VectorType::get(operandShape, vecType.getElementType());
      extractOperands.push_back(rewriter.create<vector::ExtractMapOp>(
          loc, newVecType, operand, extract.ids()));
    }
    Operation *newOp =
        cloneOpWithOperandsAndTypes(rewriter, loc, definedOp, extractOperands,
                                    extract.getResult().getType());
    rewriter.replaceOp(extract, newOp->getResult(0));
    return success();
  }
};

/// Converts TransferRead op used by ExtractMap op into a smaller dimension
/// TransferRead.
/// Example:
/// ```
/// %a = vector.transfer_read %A[%c0, %c0, %c0], %cf0:
///   memref<64x64x64xf32>, vector<64x4x32xf32>
/// %e = vector.extract_map %a[%id] : vector<64x4x32xf32> to vector<2x4x1xf32>
/// ```
/// to:
/// ```
/// %id1 = affine.apply affine_map<()[s0] -> (s0 * 2)> (%id)
/// %e = vector.transfer_read %A[%id1, %c0, %id1], %cf0 :
///   memref<64x64x64xf32>, vector<2x4x1xf32>
/// ```
struct TransferReadExtractPattern
    : public OpRewritePattern<vector::TransferReadOp> {
  TransferReadExtractPattern(MLIRContext *context)
      : OpRewritePattern<vector::TransferReadOp>(context) {}
  LogicalResult matchAndRewrite(vector::TransferReadOp read,
                                PatternRewriter &rewriter) const override {
    // TODO: support 0-d corner case.
    if (read.getTransferRank() == 0)
      return failure();

    if (!read.getResult().hasOneUse())
      return failure();
    auto extract =
        dyn_cast<vector::ExtractMapOp>(*read.getResult().getUsers().begin());
    if (!extract)
      return failure();
    if (read.mask())
      return failure();

    SmallVector<Value, 4> indices(read.indices().begin(), read.indices().end());
    AffineMap indexMap = extract.map().compose(read.permutation_map());
    unsigned idCount = 0;
    ImplicitLocOpBuilder lb(read.getLoc(), rewriter);
    for (auto it :
         llvm::zip(indexMap.getResults(), extract.map().getResults())) {
      AffineExpr d0, d1;
      bindDims(read.getContext(), d0, d1);
      auto indexExpr = std::get<0>(it).dyn_cast<AffineDimExpr>();
      if (!indexExpr)
        continue;
      unsigned indexPos = indexExpr.getPosition();
      unsigned vectorPos = std::get<1>(it).cast<AffineDimExpr>().getPosition();
      auto scale = getAffineConstantExpr(
          extract.getResultType().getDimSize(vectorPos), read.getContext());
      indices[indexPos] = makeComposedAffineApply(
          rewriter, read.getLoc(), d0 + scale * d1,
          {indices[indexPos], extract.ids()[idCount++]});
    }
    Value newRead = lb.create<vector::TransferReadOp>(
        extract.getType(), read.source(), indices, read.permutation_mapAttr(),
        read.padding(), read.mask(), read.in_boundsAttr());
    Value dest = lb.create<arith::ConstantOp>(
        read.getType(), rewriter.getZeroAttr(read.getType()));
    newRead = lb.create<vector::InsertMapOp>(newRead, dest, extract.ids());
    rewriter.replaceOp(read, newRead);
    return success();
  }
};

struct TransferWriteInsertPattern
    : public OpRewritePattern<vector::TransferWriteOp> {
  TransferWriteInsertPattern(MLIRContext *context)
      : OpRewritePattern<vector::TransferWriteOp>(context) {}
  LogicalResult matchAndRewrite(vector::TransferWriteOp write,
                                PatternRewriter &rewriter) const override {
    // TODO: support 0-d corner case.
    if (write.getTransferRank() == 0)
      return failure();

    auto insert = write.vector().getDefiningOp<vector::InsertMapOp>();
    if (!insert)
      return failure();
    if (write.mask())
      return failure();
    SmallVector<Value, 4> indices(write.indices().begin(),
                                  write.indices().end());
    AffineMap indexMap = insert.map().compose(write.permutation_map());
    unsigned idCount = 0;
    Location loc = write.getLoc();
    for (auto it :
         llvm::zip(indexMap.getResults(), insert.map().getResults())) {
      AffineExpr d0, d1;
      bindDims(write.getContext(), d0, d1);
      auto indexExpr = std::get<0>(it).dyn_cast<AffineDimExpr>();
      if (!indexExpr)
        continue;
      unsigned indexPos = indexExpr.getPosition();
      unsigned vectorPos = std::get<1>(it).cast<AffineDimExpr>().getPosition();
      auto scale = getAffineConstantExpr(
          insert.getSourceVectorType().getDimSize(vectorPos),
          write.getContext());
      indices[indexPos] =
          makeComposedAffineApply(rewriter, loc, d0 + scale * d1,
                                  {indices[indexPos], insert.ids()[idCount++]});
    }
    rewriter.create<vector::TransferWriteOp>(
        loc, insert.vector(), write.source(), indices,
        write.permutation_mapAttr(), write.in_boundsAttr());
    rewriter.eraseOp(write);
    return success();
  }
};

} // namespace

void mlir::vector::populateVectorUnrollPatterns(
    RewritePatternSet &patterns, const UnrollVectorOptions &options) {
  patterns.add<UnrollTransferReadPattern, UnrollTransferWritePattern,
               UnrollContractionPattern, UnrollElementwisePattern>(
      patterns.getContext(), options);
}

void mlir::vector::populatePropagateVectorDistributionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<PointwiseExtractPattern, ContractExtractPattern,
               TransferReadExtractPattern, TransferWriteInsertPattern>(
      patterns.getContext());
}
