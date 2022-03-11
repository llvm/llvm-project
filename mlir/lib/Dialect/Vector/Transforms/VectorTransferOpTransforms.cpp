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
  for (auto *user : write.source().getUsers()) {
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
  for (Operation *user : read.source().getUsers()) {
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
  read.replaceAllUsesWith(lastwrite.vector());
  opToErase.push_back(read.getOperation());
}

/// Drops unit dimensions from the input MemRefType.
static MemRefType dropUnitDims(MemRefType inputType, ArrayRef<int64_t> offsets,
                               ArrayRef<int64_t> sizes,
                               ArrayRef<int64_t> strides) {
  Type rankReducedType = memref::SubViewOp::inferRankReducedResultType(
      0, inputType, offsets, sizes, strides);
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
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp transferReadOp,
                                PatternRewriter &rewriter) const override {
    auto loc = transferReadOp.getLoc();
    Value vector = transferReadOp.vector();
    VectorType vectorType = vector.getType().cast<VectorType>();
    Value source = transferReadOp.source();
    MemRefType sourceType = source.getType().dyn_cast<MemRefType>();
    // TODO: support tensor types.
    if (!sourceType || !sourceType.hasStaticShape())
      return failure();
    if (sourceType.getNumElements() != vectorType.getNumElements())
      return failure();
    // TODO: generalize this pattern, relax the requirements here.
    if (transferReadOp.hasOutOfBoundsDim())
      return failure();
    if (!transferReadOp.permutation_map().isMinorIdentity())
      return failure();
    int reducedRank = getReducedRank(sourceType.getShape());
    if (reducedRank == sourceType.getRank())
      return failure(); // The source shape can't be further reduced.
    if (reducedRank != vectorType.getRank())
      return failure(); // This pattern requires the vector shape to match the
                        // reduced source shape.
    if (llvm::any_of(transferReadOp.indices(),
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
  using OpRewritePattern<vector::TransferWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp transferWriteOp,
                                PatternRewriter &rewriter) const override {
    auto loc = transferWriteOp.getLoc();
    Value vector = transferWriteOp.vector();
    VectorType vectorType = vector.getType().cast<VectorType>();
    Value source = transferWriteOp.source();
    MemRefType sourceType = source.getType().dyn_cast<MemRefType>();
    // TODO: support tensor type.
    if (!sourceType || !sourceType.hasStaticShape())
      return failure();
    if (sourceType.getNumElements() != vectorType.getNumElements())
      return failure();
    // TODO: generalize this pattern, relax the requirements here.
    if (transferWriteOp.hasOutOfBoundsDim())
      return failure();
    if (!transferWriteOp.permutation_map().isMinorIdentity())
      return failure();
    int reducedRank = getReducedRank(sourceType.getShape());
    if (reducedRank == sourceType.getRank())
      return failure(); // The source shape can't be further reduced.
    if (reducedRank != vectorType.getRank())
      return failure(); // This pattern requires the vector shape to match the
                        // reduced source shape.
    if (llvm::any_of(transferWriteOp.indices(),
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

/// Creates a memref.collapse_shape collapsing all of the dimensions of the
/// input into a 1D shape.
// TODO: move helper function
static Value collapseContiguousRowMajorMemRefTo1D(PatternRewriter &rewriter,
                                                  mlir::Location loc,
                                                  Value input) {
  Value rankReducedInput =
      rankReducingSubviewDroppingUnitDims(rewriter, loc, input);
  ShapedType rankReducedInputType =
      rankReducedInput.getType().cast<ShapedType>();
  if (rankReducedInputType.getRank() == 1)
    return rankReducedInput;
  ReassociationIndices indices;
  for (int i = 0; i < rankReducedInputType.getRank(); ++i)
    indices.push_back(i);
  return rewriter.create<memref::CollapseShapeOp>(
      loc, rankReducedInput, std::array<ReassociationIndices, 1>{indices});
}

/// Rewrites contiguous row-major vector.transfer_read ops by inserting
/// memref.collapse_shape on the source so that the resulting
/// vector.transfer_read has a 1D source. Requires the source shape to be
/// already reduced i.e. without unit dims.
class FlattenContiguousRowMajorTransferReadPattern
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp transferReadOp,
                                PatternRewriter &rewriter) const override {
    auto loc = transferReadOp.getLoc();
    Value vector = transferReadOp.vector();
    VectorType vectorType = vector.getType().cast<VectorType>();
    Value source = transferReadOp.source();
    MemRefType sourceType = source.getType().dyn_cast<MemRefType>();
    // Contiguity check is valid on tensors only.
    if (!sourceType)
      return failure();
    if (vectorType.getRank() <= 1)
      // Already 0D/1D, nothing to do.
      return failure();
    if (!isStaticShapeAndContiguousRowMajor(sourceType))
      return failure();
    if (getReducedRank(sourceType.getShape()) != sourceType.getRank())
      // This pattern requires the source to already be rank-reduced.
      return failure();
    if (sourceType.getNumElements() != vectorType.getNumElements())
      return failure();
    // TODO: generalize this pattern, relax the requirements here.
    if (transferReadOp.hasOutOfBoundsDim())
      return failure();
    if (!transferReadOp.permutation_map().isMinorIdentity())
      return failure();
    if (transferReadOp.mask())
      return failure();
    if (llvm::any_of(transferReadOp.indices(),
                     [](Value v) { return !isZero(v); }))
      return failure();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto identityMap1D = rewriter.getMultiDimIdentityMap(1);
    VectorType vectorType1d = VectorType::get({sourceType.getNumElements()},
                                              sourceType.getElementType());
    Value source1d =
        collapseContiguousRowMajorMemRefTo1D(rewriter, loc, source);
    Value read1d = rewriter.create<vector::TransferReadOp>(
        loc, vectorType1d, source1d, ValueRange{c0}, identityMap1D);
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
        transferReadOp, vector.getType().cast<VectorType>(), read1d);
    return success();
  }
};

/// Rewrites contiguous row-major vector.transfer_write ops by inserting
/// memref.collapse_shape on the source so that the resulting
/// vector.transfer_write has a 1D source. Requires the source shape to be
/// already reduced i.e. without unit dims.
class FlattenContiguousRowMajorTransferWritePattern
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern<vector::TransferWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp transferWriteOp,
                                PatternRewriter &rewriter) const override {
    auto loc = transferWriteOp.getLoc();
    Value vector = transferWriteOp.vector();
    VectorType vectorType = vector.getType().cast<VectorType>();
    Value source = transferWriteOp.source();
    MemRefType sourceType = source.getType().dyn_cast<MemRefType>();
    // Contiguity check is valid on tensors only.
    if (!sourceType)
      return failure();
    if (vectorType.getRank() <= 1)
      // Already 0D/1D, nothing to do.
      return failure();
    if (!isStaticShapeAndContiguousRowMajor(sourceType))
      return failure();
    if (getReducedRank(sourceType.getShape()) != sourceType.getRank())
      // This pattern requires the source to already be rank-reduced.
      return failure();
    if (sourceType.getNumElements() != vectorType.getNumElements())
      return failure();
    // TODO: generalize this pattern, relax the requirements here.
    if (transferWriteOp.hasOutOfBoundsDim())
      return failure();
    if (!transferWriteOp.permutation_map().isMinorIdentity())
      return failure();
    if (transferWriteOp.mask())
      return failure();
    if (llvm::any_of(transferWriteOp.indices(),
                     [](Value v) { return !isZero(v); }))
      return failure();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto identityMap1D = rewriter.getMultiDimIdentityMap(1);
    VectorType vectorType1d = VectorType::get({sourceType.getNumElements()},
                                              sourceType.getElementType());
    Value source1d =
        collapseContiguousRowMajorMemRefTo1D(rewriter, loc, source);
    Value vector1d =
        rewriter.create<vector::ShapeCastOp>(loc, vectorType1d, vector);
    rewriter.create<vector::TransferWriteOp>(loc, vector1d, source1d,
                                             ValueRange{c0}, identityMap1D);
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
    RewritePatternSet &patterns) {
  patterns
      .add<TransferReadDropUnitDimsPattern, TransferWriteDropUnitDimsPattern>(
          patterns.getContext());
  populateShapeCastFoldingPatterns(patterns);
}

void mlir::vector::populateFlattenVectorTransferPatterns(
    RewritePatternSet &patterns) {
  patterns.add<FlattenContiguousRowMajorTransferReadPattern,
               FlattenContiguousRowMajorTransferWritePattern>(
      patterns.getContext());
  populateShapeCastFoldingPatterns(patterns);
}
