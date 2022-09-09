//===- VectorTransferOpTransforms.cpp - transfer op transforms ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements functions concerned with optimizing transfer_read and
// transfer_write ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "vector-transfer-opt"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

using namespace mlir;

/// Return the ancestor op in the region or nullptr if the region is not
/// an ancestor of the op.
static Operation *findAncestorOpInRegion(Region *region, Operation *op) {
  for (; op != nullptr && op->getParentRegion() != region;
       op = op->getParentOp())
    ;
  return op;
}

namespace {

class TransferOptimization {
public:
  TransferOptimization(Operation *op) : dominators(op), postDominators(op) {}
  void deadStoreOp(vector::TransferWriteOp);
  void storeToLoadForwarding(vector::TransferReadOp);
  void removeDeadOp() {
    for (Operation *op : opToErase)
      op->erase();
    opToErase.clear();
  }

private:
  bool isReachable(Operation *start, Operation *dest);
  DominanceInfo dominators;
  PostDominanceInfo postDominators;
  std::vector<Operation *> opToErase;
};

/// Return true if there is a path from start operation to dest operation,
/// otherwise return false. The operations have to be in the same region.
bool TransferOptimization::isReachable(Operation *start, Operation *dest) {
  assert(start->getParentRegion() == dest->getParentRegion() &&
         "This function only works for ops i the same region");
  // Simple case where the start op dominate the destination.
  if (dominators.dominates(start, dest))
    return true;
  Block *startBlock = start->getBlock();
  Block *destBlock = dest->getBlock();
  SmallVector<Block *, 32> worklist(startBlock->succ_begin(),
                                    startBlock->succ_end());
  SmallPtrSet<Block *, 32> visited;
  while (!worklist.empty()) {
    Block *bb = worklist.pop_back_val();
    if (!visited.insert(bb).second)
      continue;
    if (dominators.dominates(bb, destBlock))
      return true;
    worklist.append(bb->succ_begin(), bb->succ_end());
  }
  return false;
}

/// For transfer_write to overwrite fully another transfer_write must:
/// 1. Access the same memref with the same indices and vector type.
/// 2. Post-dominate the other transfer_write operation.
/// If several candidates are available, one must be post-dominated by all the
/// others since they are all post-dominating the same transfer_write. We only
/// consider the transfer_write post-dominated by all the other candidates as
/// this will be the first transfer_write executed after the potentially dead
/// transfer_write.
/// If we found such an overwriting transfer_write we know that the original
/// transfer_write is dead if all reads that can be reached from the potentially
/// dead transfer_write are dominated by the overwriting transfer_write.
void TransferOptimization::deadStoreOp(vector::TransferWriteOp write) {
  LLVM_DEBUG(DBGS() << "Candidate for dead store: " << *write.getOperation()
                    << "\n");
  llvm::SmallVector<Operation *, 8> reads;
  Operation *firstOverwriteCandidate = nullptr;
  for (auto *user : write.getSource().getUsers()) {
    if (user == write.getOperation())
      continue;
    if (auto nextWrite = dyn_cast<vector::TransferWriteOp>(user)) {
      // Check candidate that can override the store.
      if (checkSameValueWAW(nextWrite, write) &&
          postDominators.postDominates(nextWrite, write)) {
        if (firstOverwriteCandidate == nullptr ||
            postDominators.postDominates(firstOverwriteCandidate, nextWrite))
          firstOverwriteCandidate = nextWrite;
        else
          assert(
              postDominators.postDominates(nextWrite, firstOverwriteCandidate));
      }
    } else {
      if (auto read = dyn_cast<vector::TransferReadOp>(user)) {
        // Don't need to consider disjoint reads.
        if (vector::isDisjointTransferSet(
                cast<VectorTransferOpInterface>(write.getOperation()),
                cast<VectorTransferOpInterface>(read.getOperation())))
          continue;
      }
      reads.push_back(user);
    }
  }
  if (firstOverwriteCandidate == nullptr)
    return;
  Region *topRegion = firstOverwriteCandidate->getParentRegion();
  Operation *writeAncestor = findAncestorOpInRegion(topRegion, write);
  assert(writeAncestor &&
         "write op should be recursively part of the top region");

  for (Operation *read : reads) {
    Operation *readAncestor = findAncestorOpInRegion(topRegion, read);
    // TODO: if the read and write have the same ancestor we could recurse in
    // the region to know if the read is reachable with more precision.
    if (readAncestor == nullptr || !isReachable(writeAncestor, readAncestor))
      continue;
    if (!dominators.dominates(firstOverwriteCandidate, read)) {
      LLVM_DEBUG(DBGS() << "Store may not be dead due to op: " << *read
                        << "\n");
      return;
    }
  }
  LLVM_DEBUG(DBGS() << "Found dead store: " << *write.getOperation()
                    << " overwritten by: " << *firstOverwriteCandidate << "\n");
  opToErase.push_back(write.getOperation());
}

/// A transfer_write candidate to storeToLoad forwarding must:
/// 1. Access the same memref with the same indices and vector type as the
/// transfer_read.
/// 2. Dominate the transfer_read operation.
/// If several candidates are available, one must be dominated by all the others
/// since they are all dominating the same transfer_read. We only consider the
/// transfer_write dominated by all the other candidates as this will be the
/// last transfer_write executed before the transfer_read.
/// If we found such a candidate we can do the forwarding if all the other
/// potentially aliasing ops that may reach the transfer_read are post-dominated
/// by the transfer_write.
void TransferOptimization::storeToLoadForwarding(vector::TransferReadOp read) {
  if (read.hasOutOfBoundsDim())
    return;
  LLVM_DEBUG(DBGS() << "Candidate for Forwarding: " << *read.getOperation()
                    << "\n");
  SmallVector<Operation *, 8> blockingWrites;
  vector::TransferWriteOp lastwrite = nullptr;
  for (Operation *user : read.getSource().getUsers()) {
    if (isa<vector::TransferReadOp>(user))
      continue;
    if (auto write = dyn_cast<vector::TransferWriteOp>(user)) {
      // If there is a write, but we can prove that it is disjoint we can ignore
      // the write.
      if (vector::isDisjointTransferSet(
              cast<VectorTransferOpInterface>(write.getOperation()),
              cast<VectorTransferOpInterface>(read.getOperation())))
        continue;
      if (dominators.dominates(write, read) && checkSameValueRAW(write, read)) {
        if (lastwrite == nullptr || dominators.dominates(lastwrite, write))
          lastwrite = write;
        else
          assert(dominators.dominates(write, lastwrite));
        continue;
      }
    }
    blockingWrites.push_back(user);
  }

  if (lastwrite == nullptr)
    return;

  Region *topRegion = lastwrite->getParentRegion();
  Operation *readAncestor = findAncestorOpInRegion(topRegion, read);
  assert(readAncestor &&
         "read op should be recursively part of the top region");

  for (Operation *write : blockingWrites) {
    Operation *writeAncestor = findAncestorOpInRegion(topRegion, write);
    // TODO: if the store and read have the same ancestor we could recurse in
    // the region to know if the read is reachable with more precision.
    if (writeAncestor == nullptr || !isReachable(writeAncestor, readAncestor))
      continue;
    if (!postDominators.postDominates(lastwrite, write)) {
      LLVM_DEBUG(DBGS() << "Fail to do write to read forwarding due to op: "
                        << *write << "\n");
      return;
    }
  }

  LLVM_DEBUG(DBGS() << "Forward value from " << *lastwrite.getOperation()
                    << " to: " << *read.getOperation() << "\n");
  read.replaceAllUsesWith(lastwrite.getVector());
  opToErase.push_back(read.getOperation());
}

/// Drops unit dimensions from the input MemRefType.
static MemRefType dropUnitDims(MemRefType inputType, ArrayRef<int64_t> offsets,
                               ArrayRef<int64_t> sizes,
                               ArrayRef<int64_t> strides) {
  SmallVector<int64_t> targetShape = llvm::to_vector(
      llvm::make_filter_range(sizes, [](int64_t sz) { return sz != 1; }));
  Type rankReducedType = memref::SubViewOp::inferRankReducedResultType(
      targetShape, inputType, offsets, sizes, strides);
  return canonicalizeStridedLayout(rankReducedType.cast<MemRefType>());
}

/// Creates a rank-reducing memref.subview op that drops unit dims from its
/// input. Or just returns the input if it was already without unit dims.
static Value rankReducingSubviewDroppingUnitDims(PatternRewriter &rewriter,
                                                 mlir::Location loc,
                                                 Value input) {
  MemRefType inputType = input.getType().cast<MemRefType>();
  assert(inputType.hasStaticShape());
  SmallVector<int64_t> subViewOffsets(inputType.getRank(), 0);
  SmallVector<int64_t> subViewStrides(inputType.getRank(), 1);
  ArrayRef<int64_t> subViewSizes = inputType.getShape();
  MemRefType resultType =
      dropUnitDims(inputType, subViewOffsets, subViewSizes, subViewStrides);
  if (canonicalizeStridedLayout(resultType) ==
      canonicalizeStridedLayout(inputType))
    return input;
  return rewriter.create<memref::SubViewOp>(
      loc, resultType, input, subViewOffsets, subViewSizes, subViewStrides);
}

/// Returns the number of dims that aren't unit dims.
static int getReducedRank(ArrayRef<int64_t> shape) {
  return llvm::count_if(shape, [](int64_t dimSize) { return dimSize != 1; });
}

/// Returns true if all values are `arith.constant 0 : index`
static bool isZero(Value v) {
  auto cst = v.getDefiningOp<arith::ConstantIndexOp>();
  return cst && cst.value() == 0;
}

/// Rewrites vector.transfer_read ops where the source has unit dims, by
/// inserting a memref.subview dropping those unit dims.
class TransferReadDropUnitDimsPattern
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp transferReadOp,
                                PatternRewriter &rewriter) const override {
    auto loc = transferReadOp.getLoc();
    Value vector = transferReadOp.getVector();
    VectorType vectorType = vector.getType().cast<VectorType>();
    Value source = transferReadOp.getSource();
    MemRefType sourceType = source.getType().dyn_cast<MemRefType>();
    // TODO: support tensor types.
    if (!sourceType || !sourceType.hasStaticShape())
      return failure();
    if (sourceType.getNumElements() != vectorType.getNumElements())
      return failure();
    // TODO: generalize this pattern, relax the requirements here.
    if (transferReadOp.hasOutOfBoundsDim())
      return failure();
    if (!transferReadOp.getPermutationMap().isMinorIdentity())
      return failure();
    int reducedRank = getReducedRank(sourceType.getShape());
    if (reducedRank == sourceType.getRank())
      return failure(); // The source shape can't be further reduced.
    if (reducedRank != vectorType.getRank())
      return failure(); // This pattern requires the vector shape to match the
                        // reduced source shape.
    if (llvm::any_of(transferReadOp.getIndices(),
                     [](Value v) { return !isZero(v); }))
      return failure();
    Value reducedShapeSource =
        rankReducingSubviewDroppingUnitDims(rewriter, loc, source);
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> zeros(reducedRank, c0);
    auto identityMap = rewriter.getMultiDimIdentityMap(reducedRank);
    rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
        transferReadOp, vectorType, reducedShapeSource, zeros, identityMap);
    return success();
  }
};

/// Rewrites vector.transfer_write ops where the "source" (i.e. destination) has
/// unit dims, by inserting a memref.subview dropping those unit dims.
class TransferWriteDropUnitDimsPattern
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp transferWriteOp,
                                PatternRewriter &rewriter) const override {
    auto loc = transferWriteOp.getLoc();
    Value vector = transferWriteOp.getVector();
    VectorType vectorType = vector.getType().cast<VectorType>();
    Value source = transferWriteOp.getSource();
    MemRefType sourceType = source.getType().dyn_cast<MemRefType>();
    // TODO: support tensor type.
    if (!sourceType || !sourceType.hasStaticShape())
      return failure();
    if (sourceType.getNumElements() != vectorType.getNumElements())
      return failure();
    // TODO: generalize this pattern, relax the requirements here.
    if (transferWriteOp.hasOutOfBoundsDim())
      return failure();
    if (!transferWriteOp.getPermutationMap().isMinorIdentity())
      return failure();
    int reducedRank = getReducedRank(sourceType.getShape());
    if (reducedRank == sourceType.getRank())
      return failure(); // The source shape can't be further reduced.
    if (reducedRank != vectorType.getRank())
      return failure(); // This pattern requires the vector shape to match the
                        // reduced source shape.
    if (llvm::any_of(transferWriteOp.getIndices(),
                     [](Value v) { return !isZero(v); }))
      return failure();
    Value reducedShapeSource =
        rankReducingSubviewDroppingUnitDims(rewriter, loc, source);
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> zeros(reducedRank, c0);
    auto identityMap = rewriter.getMultiDimIdentityMap(reducedRank);
    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        transferWriteOp, vector, reducedShapeSource, zeros, identityMap);
    return success();
  }
};

/// Returns the position of the first inner dimension that has contiguous layout
/// with at least `requiredContiguousSize` contiguous elements.
/// When such a dimension is found, the return value satisfies:
///   0 <= return_value <= memrefType.getRank() - 1.
/// When no such dimension is found, the return value is memrefType.getRank().
static int64_t getContiguousInnerDim(MemRefType memrefType,
                                     int64_t requiredContiguousSize) {
  auto shape = memrefType.getShape();
  SmallVector<int64_t> strides;
  int64_t offset;
  int64_t innerDim = shape.size();
  if (succeeded(getStridesAndOffset(memrefType, strides, offset))) {
    int64_t innerSize = 1;
    while (true) {
      if (innerDim == 0)
        break;
      const int64_t nextDim = innerDim - 1;
      if (shape[nextDim] == ShapedType::kDynamicSize)
        break;
      if (strides[nextDim] != innerSize)
        break;
      innerSize *= shape[nextDim];
      innerDim = nextDim;
      if (innerSize >= requiredContiguousSize)
        break;
    }
  }
  return innerDim;
}

/// Creates a memref.collapse_shape collapsing all inner dimensions of the
/// input starting at `firstDimToCollapse`.
static Value collapseInnerDims(PatternRewriter &rewriter, mlir::Location loc,
                               Value input, int64_t firstDimToCollapse) {
  ShapedType inputType = input.getType().cast<ShapedType>();
  if (inputType.getRank() == 1)
    return input;
  SmallVector<ReassociationIndices> reassociation;
  for (int64_t i = 0; i < firstDimToCollapse; ++i)
    reassociation.push_back(ReassociationIndices{i});
  ReassociationIndices collapsedIndices;
  for (int64_t i = firstDimToCollapse; i < inputType.getRank(); ++i)
    collapsedIndices.push_back(i);
  reassociation.push_back(collapsedIndices);
  return rewriter.create<memref::CollapseShapeOp>(loc, input, reassociation);
}

/// Checks that the indices corresponding to dimensions starting at
/// `firstDimToCollapse` are constant 0, and writes to `outIndices`
/// the truncated indices where `firstDimToCollapse` is now the innermost dim.
static LogicalResult
checkAndCollapseInnerZeroIndices(ValueRange indices, int64_t firstDimToCollapse,
                                 SmallVector<Value> &outIndices) {
  int64_t rank = indices.size();
  if (firstDimToCollapse >= rank)
    return failure();
  for (int64_t i = firstDimToCollapse; i < rank; ++i) {
    arith::ConstantIndexOp cst =
        indices[i].getDefiningOp<arith::ConstantIndexOp>();
    if (!cst || cst.value() != 0)
      return failure();
  }
  outIndices = indices;
  outIndices.resize(firstDimToCollapse + 1);
  return success();
}

/// Rewrites contiguous row-major vector.transfer_read ops by inserting
/// memref.collapse_shape on the source so that the resulting
/// vector.transfer_read has a 1D source. Requires the source shape to be
/// already reduced i.e. without unit dims.
class FlattenContiguousRowMajorTransferReadPattern
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp transferReadOp,
                                PatternRewriter &rewriter) const override {
    auto loc = transferReadOp.getLoc();
    Value vector = transferReadOp.getVector();
    VectorType vectorType = vector.getType().cast<VectorType>();
    Value source = transferReadOp.getSource();
    MemRefType sourceType = source.getType().dyn_cast<MemRefType>();
    // Contiguity check is valid on tensors only.
    if (!sourceType)
      return failure();
    if (vectorType.getRank() <= 1)
      // Already 0D/1D, nothing to do.
      return failure();
    int64_t firstContiguousInnerDim =
        getContiguousInnerDim(sourceType, vectorType.getNumElements());
    if (firstContiguousInnerDim >= sourceType.getRank() - 1)
      return failure();
    // TODO: generalize this pattern, relax the requirements here.
    if (transferReadOp.hasOutOfBoundsDim())
      return failure();
    if (!transferReadOp.getPermutationMap().isMinorIdentity())
      return failure();
    if (transferReadOp.getMask())
      return failure();
    SmallVector<Value> collapsedIndices;
    if (failed(checkAndCollapseInnerZeroIndices(transferReadOp.getIndices(),
                                                firstContiguousInnerDim,
                                                collapsedIndices)))
      return failure();
    Value collapsedSource =
        collapseInnerDims(rewriter, loc, source, firstContiguousInnerDim);
    MemRefType collapsedSourceType =
        collapsedSource.getType().dyn_cast<MemRefType>();
    int64_t collapsedRank = collapsedSourceType.getRank();
    assert(collapsedRank == firstContiguousInnerDim + 1);
    SmallVector<AffineExpr, 1> dimExprs{
        getAffineDimExpr(firstContiguousInnerDim, rewriter.getContext())};
    auto collapsedMap =
        AffineMap::get(collapsedRank, 0, dimExprs, rewriter.getContext());
    VectorType flatVectorType = VectorType::get({vectorType.getNumElements()},
                                                vectorType.getElementType());
    vector::TransferReadOp flatRead = rewriter.create<vector::TransferReadOp>(
        loc, flatVectorType, collapsedSource, collapsedIndices, collapsedMap);
    flatRead.setInBoundsAttr(rewriter.getBoolArrayAttr({true}));
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
        transferReadOp, vector.getType().cast<VectorType>(), flatRead);
    return success();
  }
};

/// Rewrites contiguous row-major vector.transfer_write ops by inserting
/// memref.collapse_shape on the source so that the resulting
/// vector.transfer_write has a 1D source. Requires the source shape to be
/// already reduced i.e. without unit dims.
class FlattenContiguousRowMajorTransferWritePattern
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp transferWriteOp,
                                PatternRewriter &rewriter) const override {
    auto loc = transferWriteOp.getLoc();
    Value vector = transferWriteOp.getVector();
    VectorType vectorType = vector.getType().cast<VectorType>();
    Value source = transferWriteOp.getSource();
    MemRefType sourceType = source.getType().dyn_cast<MemRefType>();
    // Contiguity check is valid on tensors only.
    if (!sourceType)
      return failure();
    if (vectorType.getRank() <= 1)
      // Already 0D/1D, nothing to do.
      return failure();
    int64_t firstContiguousInnerDim =
        getContiguousInnerDim(sourceType, vectorType.getNumElements());
    if (firstContiguousInnerDim >= sourceType.getRank() - 1)
      return failure();
    // TODO: generalize this pattern, relax the requirements here.
    if (transferWriteOp.hasOutOfBoundsDim())
      return failure();
    if (!transferWriteOp.getPermutationMap().isMinorIdentity())
      return failure();
    if (transferWriteOp.getMask())
      return failure();
    SmallVector<Value> collapsedIndices;
    if (failed(checkAndCollapseInnerZeroIndices(transferWriteOp.getIndices(),
                                                firstContiguousInnerDim,
                                                collapsedIndices)))
      return failure();
    Value collapsedSource =
        collapseInnerDims(rewriter, loc, source, firstContiguousInnerDim);
    MemRefType collapsedSourceType =
        collapsedSource.getType().cast<MemRefType>();
    int64_t collapsedRank = collapsedSourceType.getRank();
    assert(collapsedRank == firstContiguousInnerDim + 1);
    SmallVector<AffineExpr, 1> dimExprs{
        getAffineDimExpr(firstContiguousInnerDim, rewriter.getContext())};
    auto collapsedMap =
        AffineMap::get(collapsedRank, 0, dimExprs, rewriter.getContext());
    VectorType flatVectorType = VectorType::get({vectorType.getNumElements()},
                                                vectorType.getElementType());
    Value flatVector =
        rewriter.create<vector::ShapeCastOp>(loc, flatVectorType, vector);
    vector::TransferWriteOp flatWrite =
        rewriter.create<vector::TransferWriteOp>(
            loc, flatVector, collapsedSource, collapsedIndices, collapsedMap);
    flatWrite.setInBoundsAttr(rewriter.getBoolArrayAttr({true}));
    rewriter.eraseOp(transferWriteOp);
    return success();
  }
};

} // namespace

void mlir::vector::transferOpflowOpt(Operation *rootOp) {
  TransferOptimization opt(rootOp);
  // Run store to load forwarding first since it can expose more dead store
  // opportunity.
  rootOp->walk([&](vector::TransferReadOp read) {
    if (read.getShapedType().isa<MemRefType>())
      opt.storeToLoadForwarding(read);
  });
  opt.removeDeadOp();
  rootOp->walk([&](vector::TransferWriteOp write) {
    if (write.getShapedType().isa<MemRefType>())
      opt.deadStoreOp(write);
  });
  opt.removeDeadOp();
}

void mlir::vector::populateVectorTransferDropUnitDimsPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns
      .add<TransferReadDropUnitDimsPattern, TransferWriteDropUnitDimsPattern>(
          patterns.getContext(), benefit);
  populateShapeCastFoldingPatterns(patterns);
}

void mlir::vector::populateFlattenVectorTransferPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<FlattenContiguousRowMajorTransferReadPattern,
               FlattenContiguousRowMajorTransferWritePattern>(
      patterns.getContext(), benefit);
  populateShapeCastFoldingPatterns(patterns, benefit);
}
