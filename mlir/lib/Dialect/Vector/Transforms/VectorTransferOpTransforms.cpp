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

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
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
  TransferOptimization(RewriterBase &rewriter, Operation *op)
      : rewriter(rewriter), dominators(op), postDominators(op) {}
  void deadStoreOp(vector::TransferWriteOp);
  void storeToLoadForwarding(vector::TransferReadOp);
  void removeDeadOp() {
    for (Operation *op : opToErase)
      rewriter.eraseOp(op);
    opToErase.clear();
  }

private:
  RewriterBase &rewriter;
  bool isReachable(Operation *start, Operation *dest);
  DominanceInfo dominators;
  PostDominanceInfo postDominators;
  std::vector<Operation *> opToErase;
};

} // namespace
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
  llvm::SmallVector<Operation *, 8> blockingAccesses;
  Operation *firstOverwriteCandidate = nullptr;
  Value source = write.getSource();
  // Skip subview ops.
  while (auto subView = source.getDefiningOp<memref::SubViewOp>())
    source = subView.getSource();
  llvm::SmallVector<Operation *, 32> users(source.getUsers().begin(),
                                           source.getUsers().end());
  llvm::SmallDenseSet<Operation *, 32> processed;
  while (!users.empty()) {
    Operation *user = users.pop_back_val();
    // If the user has already been processed skip.
    if (!processed.insert(user).second)
      continue;
    if (auto subView = dyn_cast<memref::SubViewOp>(user)) {
      users.append(subView->getUsers().begin(), subView->getUsers().end());
      continue;
    }
    if (isMemoryEffectFree(user))
      continue;
    if (user == write.getOperation())
      continue;
    if (auto nextWrite = dyn_cast<vector::TransferWriteOp>(user)) {
      // Check candidate that can override the store.
      if (write.getSource() == nextWrite.getSource() &&
          checkSameValueWAW(nextWrite, write) &&
          postDominators.postDominates(nextWrite, write)) {
        if (firstOverwriteCandidate == nullptr ||
            postDominators.postDominates(firstOverwriteCandidate, nextWrite))
          firstOverwriteCandidate = nextWrite;
        else
          assert(
              postDominators.postDominates(nextWrite, firstOverwriteCandidate));
        continue;
      }
    }
    if (auto transferOp = dyn_cast<VectorTransferOpInterface>(user)) {
      // Don't need to consider disjoint accesses.
      if (vector::isDisjointTransferSet(
              cast<VectorTransferOpInterface>(write.getOperation()),
              cast<VectorTransferOpInterface>(transferOp.getOperation()),
              /*testDynamicValueUsingBounds=*/true))
        continue;
    }
    blockingAccesses.push_back(user);
  }
  if (firstOverwriteCandidate == nullptr)
    return;
  Region *topRegion = firstOverwriteCandidate->getParentRegion();
  Operation *writeAncestor = findAncestorOpInRegion(topRegion, write);
  assert(writeAncestor &&
         "write op should be recursively part of the top region");

  for (Operation *access : blockingAccesses) {
    Operation *accessAncestor = findAncestorOpInRegion(topRegion, access);
    // TODO: if the access and write have the same ancestor we could recurse in
    // the region to know if the access is reachable with more precision.
    if (accessAncestor == nullptr ||
        !isReachable(writeAncestor, accessAncestor))
      continue;
    if (!dominators.dominates(firstOverwriteCandidate, accessAncestor)) {
      LLVM_DEBUG(DBGS() << "Store may not be dead due to op: "
                        << *accessAncestor << "\n");
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
  Value source = read.getSource();
  // Skip subview ops.
  while (auto subView = source.getDefiningOp<memref::SubViewOp>())
    source = subView.getSource();
  llvm::SmallVector<Operation *, 32> users(source.getUsers().begin(),
                                           source.getUsers().end());
  llvm::SmallDenseSet<Operation *, 32> processed;
  while (!users.empty()) {
    Operation *user = users.pop_back_val();
    // If the user has already been processed skip.
    if (!processed.insert(user).second)
      continue;
    if (auto subView = dyn_cast<memref::SubViewOp>(user)) {
      users.append(subView->getUsers().begin(), subView->getUsers().end());
      continue;
    }
    if (auto collapsed = dyn_cast<memref::CollapseShapeOp>(user)) {
      users.append(collapsed->getUsers().begin(), collapsed->getUsers().end());
      continue;
    }
    if (isMemoryEffectFree(user) || isa<vector::TransferReadOp>(user))
      continue;
    if (auto write = dyn_cast<vector::TransferWriteOp>(user)) {
      // If there is a write, but we can prove that it is disjoint we can ignore
      // the write.
      if (vector::isDisjointTransferSet(
              cast<VectorTransferOpInterface>(write.getOperation()),
              cast<VectorTransferOpInterface>(read.getOperation()),
              /*testDynamicValueUsingBounds=*/true))
        continue;
      if (write.getSource() == read.getSource() &&
          dominators.dominates(write, read) && checkSameValueRAW(write, read)) {
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

/// Converts OpFoldResults to int64_t shape without unit dims.
static SmallVector<int64_t> getReducedShape(ArrayRef<OpFoldResult> mixedSizes) {
  SmallVector<int64_t> reducedShape;
  for (const auto size : mixedSizes) {
    if (llvm::dyn_cast_if_present<Value>(size)) {
      reducedShape.push_back(ShapedType::kDynamic);
      continue;
    }

    auto value = cast<IntegerAttr>(size.get<Attribute>()).getValue();
    if (value == 1)
      continue;
    reducedShape.push_back(value.getSExtValue());
  }
  return reducedShape;
}

/// Drops unit dimensions from the input MemRefType.
static MemRefType dropUnitDims(MemRefType inputType,
                               ArrayRef<OpFoldResult> offsets,
                               ArrayRef<OpFoldResult> sizes,
                               ArrayRef<OpFoldResult> strides) {
  auto targetShape = getReducedShape(sizes);
  Type rankReducedType = memref::SubViewOp::inferRankReducedResultType(
      targetShape, inputType, offsets, sizes, strides);
  return canonicalizeStridedLayout(cast<MemRefType>(rankReducedType));
}

/// Creates a rank-reducing memref.subview op that drops unit dims from its
/// input. Or just returns the input if it was already without unit dims.
static Value rankReducingSubviewDroppingUnitDims(PatternRewriter &rewriter,
                                                 mlir::Location loc,
                                                 Value input) {
  MemRefType inputType = cast<MemRefType>(input.getType());
  SmallVector<OpFoldResult> offsets(inputType.getRank(),
                                    rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes = memref::getMixedSizes(rewriter, loc, input);
  SmallVector<OpFoldResult> strides(inputType.getRank(),
                                    rewriter.getIndexAttr(1));
  MemRefType resultType = dropUnitDims(inputType, offsets, sizes, strides);

  if (canonicalizeStridedLayout(resultType) ==
      canonicalizeStridedLayout(inputType))
    return input;
  return rewriter.create<memref::SubViewOp>(loc, resultType, input, offsets,
                                            sizes, strides);
}

/// Returns the number of dims that aren't unit dims.
static int getReducedRank(ArrayRef<int64_t> shape) {
  return llvm::count_if(shape, [](int64_t dimSize) { return dimSize != 1; });
}

/// Trims non-scalable one dimensions from `oldType` and returns the result
/// type.
static VectorType trimNonScalableUnitDims(VectorType oldType) {
  SmallVector<int64_t> newShape;
  SmallVector<bool> newScalableDims;
  for (auto [dimIdx, dimSize] : llvm::enumerate(oldType.getShape())) {
    if (dimSize == 1 && !oldType.getScalableDims()[dimIdx])
      continue;
    newShape.push_back(dimSize);
    newScalableDims.push_back(oldType.getScalableDims()[dimIdx]);
  }
  return VectorType::get(newShape, oldType.getElementType(), newScalableDims);
}

// Rewrites vector.create_mask 'op' to drop non-scalable one dimensions.
static FailureOr<Value>
createMaskDropNonScalableUnitDims(PatternRewriter &rewriter, Location loc,
                                  vector::CreateMaskOp op) {
  auto type = op.getType();
  VectorType reducedType = trimNonScalableUnitDims(type);
  if (reducedType.getRank() == type.getRank())
    return failure();

  SmallVector<Value> reducedOperands;
  for (auto [dim, dimIsScalable, operand] : llvm::zip_equal(
           type.getShape(), type.getScalableDims(), op.getOperands())) {
    if (dim == 1 && !dimIsScalable) {
      // If the mask for the unit dim is not a constant of 1, do nothing.
      auto constant = operand.getDefiningOp<arith::ConstantIndexOp>();
      if (!constant || (constant.value() != 1))
        return failure();
      continue;
    }
    reducedOperands.push_back(operand);
  }
  return rewriter
      .create<vector::CreateMaskOp>(loc, reducedType, reducedOperands)
      .getResult();
}

namespace {

/// Rewrites `vector.transfer_read` ops where the source has unit dims, by
/// inserting a memref.subview dropping those unit dims. The vector shapes are
/// also reduced accordingly.
class TransferReadDropUnitDimsPattern
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp transferReadOp,
                                PatternRewriter &rewriter) const override {
    auto loc = transferReadOp.getLoc();
    Value vector = transferReadOp.getVector();
    VectorType vectorType = cast<VectorType>(vector.getType());
    Value source = transferReadOp.getSource();
    MemRefType sourceType = dyn_cast<MemRefType>(source.getType());
    // TODO: support tensor types.
    if (!sourceType)
      return failure();
    // TODO: generalize this pattern, relax the requirements here.
    if (transferReadOp.hasOutOfBoundsDim())
      return failure();
    if (!transferReadOp.getPermutationMap().isMinorIdentity())
      return failure();
    // Check if the source shape can be further reduced.
    int reducedRank = getReducedRank(sourceType.getShape());
    if (reducedRank == sourceType.getRank())
      return failure();
    // Check if the reduced vector shape matches the reduced source shape.
    // Otherwise, this case is not supported yet.
    VectorType reducedVectorType = trimNonScalableUnitDims(vectorType);
    if (reducedRank != reducedVectorType.getRank())
      return failure();
    if (llvm::any_of(transferReadOp.getIndices(), [](Value v) {
          return getConstantIntValue(v) != static_cast<int64_t>(0);
        }))
      return failure();

    Value maskOp = transferReadOp.getMask();
    if (maskOp) {
      auto createMaskOp = maskOp.getDefiningOp<vector::CreateMaskOp>();
      if (!createMaskOp)
        return rewriter.notifyMatchFailure(
            transferReadOp, "unsupported mask op, only 'vector.create_mask' is "
                            "currently supported");
      FailureOr<Value> rankReducedCreateMask =
          createMaskDropNonScalableUnitDims(rewriter, loc, createMaskOp);
      if (failed(rankReducedCreateMask))
        return failure();
      maskOp = *rankReducedCreateMask;
    }

    Value reducedShapeSource =
        rankReducingSubviewDroppingUnitDims(rewriter, loc, source);
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> zeros(reducedRank, c0);
    auto identityMap = rewriter.getMultiDimIdentityMap(reducedRank);
    SmallVector<bool> inBounds(reducedVectorType.getRank(), true);
    auto newTransferReadOp = rewriter.create<vector::TransferReadOp>(
        loc, reducedVectorType, reducedShapeSource, zeros, identityMap,
        transferReadOp.getPadding(), maskOp,
        rewriter.getBoolArrayAttr(inBounds));
    auto shapeCast = rewriter.createOrFold<vector::ShapeCastOp>(
        loc, vectorType, newTransferReadOp);
    rewriter.replaceOp(transferReadOp, shapeCast);

    return success();
  }
};

/// Rewrites `vector.transfer_write` ops where the "source" (i.e. destination)
/// has unit dims, by inserting a `memref.subview` dropping those unit dims. The
/// vector shapes are also reduced accordingly.
class TransferWriteDropUnitDimsPattern
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp transferWriteOp,
                                PatternRewriter &rewriter) const override {
    auto loc = transferWriteOp.getLoc();
    Value vector = transferWriteOp.getVector();
    VectorType vectorType = cast<VectorType>(vector.getType());
    Value source = transferWriteOp.getSource();
    MemRefType sourceType = dyn_cast<MemRefType>(source.getType());
    // TODO: support tensor type.
    if (!sourceType)
      return failure();
    // TODO: generalize this pattern, relax the requirements here.
    if (transferWriteOp.hasOutOfBoundsDim())
      return failure();
    if (!transferWriteOp.getPermutationMap().isMinorIdentity())
      return failure();
    // Check if the destination shape can be further reduced.
    int reducedRank = getReducedRank(sourceType.getShape());
    if (reducedRank == sourceType.getRank())
      return failure();
    // Check if the reduced vector shape matches the reduced destination shape.
    // Otherwise, this case is not supported yet.
    VectorType reducedVectorType = trimNonScalableUnitDims(vectorType);
    if (reducedRank != reducedVectorType.getRank())
      return failure();
    if (llvm::any_of(transferWriteOp.getIndices(), [](Value v) {
          return getConstantIntValue(v) != static_cast<int64_t>(0);
        }))
      return failure();

    Value maskOp = transferWriteOp.getMask();
    if (maskOp) {
      auto createMaskOp = maskOp.getDefiningOp<vector::CreateMaskOp>();
      if (!createMaskOp)
        return rewriter.notifyMatchFailure(
            transferWriteOp,
            "unsupported mask op, only 'vector.create_mask' is "
            "currently supported");
      FailureOr<Value> rankReducedCreateMask =
          createMaskDropNonScalableUnitDims(rewriter, loc, createMaskOp);
      if (failed(rankReducedCreateMask))
        return failure();
      maskOp = *rankReducedCreateMask;
    }
    Value reducedShapeSource =
        rankReducingSubviewDroppingUnitDims(rewriter, loc, source);
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> zeros(reducedRank, c0);
    auto identityMap = rewriter.getMultiDimIdentityMap(reducedRank);
    SmallVector<bool> inBounds(reducedVectorType.getRank(), true);
    auto shapeCast = rewriter.createOrFold<vector::ShapeCastOp>(
        loc, reducedVectorType, vector);
    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        transferWriteOp, Type(), shapeCast, reducedShapeSource, zeros,
        identityMap, maskOp, rewriter.getBoolArrayAttr(inBounds));

    return success();
  }
};

} // namespace

/// Creates a memref.collapse_shape collapsing all inner dimensions of the
/// input starting at `firstDimToCollapse`.
static Value collapseInnerDims(PatternRewriter &rewriter, mlir::Location loc,
                               Value input, int64_t firstDimToCollapse) {
  ShapedType inputType = cast<ShapedType>(input.getType());
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
/// TODO: Extract the logic that writes to outIndices so that this method
/// simply checks one pre-condition.
static LogicalResult
checkAndCollapseInnerZeroIndices(ValueRange indices, int64_t firstDimToCollapse,
                                 SmallVector<Value> &outIndices) {
  int64_t rank = indices.size();
  if (firstDimToCollapse >= rank)
    return failure();
  for (int64_t i = firstDimToCollapse; i < rank; ++i) {
    std::optional<int64_t> cst = getConstantIntValue(indices[i]);
    if (!cst || cst.value() != 0)
      return failure();
  }
  outIndices = indices;
  outIndices.resize(firstDimToCollapse + 1);
  return success();
}

namespace {

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
    VectorType vectorType = cast<VectorType>(vector.getType());
    auto source = transferReadOp.getSource();
    MemRefType sourceType = dyn_cast<MemRefType>(source.getType());

    // 0. Check pre-conditions
    // Contiguity check is valid on tensors only.
    if (!sourceType)
      return failure();
    // If this is already 0D/1D, there's nothing to do.
    if (vectorType.getRank() <= 1)
      return failure();
    if (!vector::isContiguousSlice(sourceType, vectorType))
      return failure();
    // TODO: generalize this pattern, relax the requirements here.
    if (transferReadOp.hasOutOfBoundsDim())
      return failure();
    if (!transferReadOp.getPermutationMap().isMinorIdentity())
      return failure();
    if (transferReadOp.getMask())
      return failure();

    SmallVector<Value> collapsedIndices;
    int64_t firstDimToCollapse = sourceType.getRank() - vectorType.getRank();

    // 1. Collapse the source memref
    Value collapsedSource =
        collapseInnerDims(rewriter, loc, source, firstDimToCollapse);
    MemRefType collapsedSourceType =
        dyn_cast<MemRefType>(collapsedSource.getType());
    int64_t collapsedRank = collapsedSourceType.getRank();
    assert(collapsedRank == firstDimToCollapse + 1);

    // 2. Generate input args for a new vector.transfer_read that will read
    // from the collapsed memref.
    // 2.1. New dim exprs + affine map
    SmallVector<AffineExpr, 1> dimExprs{
        getAffineDimExpr(firstDimToCollapse, rewriter.getContext())};
    auto collapsedMap =
        AffineMap::get(collapsedRank, 0, dimExprs, rewriter.getContext());

    // 2.2 New indices
    // If all the collapsed indices are zero then no extra logic is needed.
    // Otherwise, a new offset/index has to be computed.
    if (failed(checkAndCollapseInnerZeroIndices(transferReadOp.getIndices(),
                                                firstDimToCollapse,
                                                collapsedIndices))) {
      // Copy all the leading indices
      collapsedIndices = transferReadOp.getIndices();
      collapsedIndices.resize(firstDimToCollapse);

      // Compute the remaining trailing index/offset required for reading from
      // the collapsed memref:
      //
      //    offset = 0
      //    for (i = firstDimToCollapse; i < outputRank; ++i)
      //      offset += sourceType.getDimSize(i) * transferReadOp.indices[i]
      //
      // For this example:
      //   %2 = vector.transfer_read %arg4[%c0, %arg0, %c0] (...) :
      //      memref<1x43x2xi32>, vector<1x2xi32>
      // which would be collapsed to:
      //   %1 = vector.transfer_read %collapse_shape[%c0, %offset] (...) :
      //      memref<1x86xi32>, vector<2xi32>
      // one would get the following offset:
      //    %offset = %arg0 * 43
      AffineExpr offsetExpr, idxExpr;
      bindSymbols(rewriter.getContext(), offsetExpr, idxExpr);

      int64_t outputRank = transferReadOp.getIndices().size();
      OpFoldResult offset =
          rewriter.create<arith::ConstantIndexOp>(loc, 0).getResult();

      for (int64_t i = firstDimToCollapse; i < outputRank; ++i) {
        int64_t dim = dyn_cast<ShapedType>(source.getType()).getDimSize(i);
        offset = affine::makeComposedFoldedAffineApply(
            rewriter, loc, offsetExpr + dim * idxExpr,
            {offset, transferReadOp.getIndices()[i]});
      }
      if (offset.is<Value>()) {
        collapsedIndices.push_back(offset.get<Value>());
      } else {
        collapsedIndices.push_back(rewriter.create<arith::ConstantIndexOp>(
            loc, *getConstantIntValue(offset)));
      }
    }

    // 3. Create new vector.transfer_read that reads from the collapsed memref
    VectorType flatVectorType = VectorType::get({vectorType.getNumElements()},
                                                vectorType.getElementType());
    vector::TransferReadOp flatRead = rewriter.create<vector::TransferReadOp>(
        loc, flatVectorType, collapsedSource, collapsedIndices, collapsedMap);
    flatRead.setInBoundsAttr(rewriter.getBoolArrayAttr({true}));

    // 4. Replace the old transfer_read with the new one reading from the
    // collapsed shape
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
        transferReadOp, cast<VectorType>(vector.getType()), flatRead);
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
    VectorType vectorType = cast<VectorType>(vector.getType());
    Value source = transferWriteOp.getSource();
    MemRefType sourceType = dyn_cast<MemRefType>(source.getType());
    // Contiguity check is valid on tensors only.
    if (!sourceType)
      return failure();
    if (vectorType.getRank() <= 1)
      // Already 0D/1D, nothing to do.
      return failure();
    if (!vector::isContiguousSlice(sourceType, vectorType))
      return failure();
    int64_t firstContiguousInnerDim =
        sourceType.getRank() - vectorType.getRank();
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
        cast<MemRefType>(collapsedSource.getType());
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

/// Base class for `vector.extract/vector.extract_element(vector.transfer_read)`
/// to `memref.load` patterns. The `match` method is shared for both
/// `vector.extract` and `vector.extract_element`.
template <class VectorExtractOp>
class RewriteScalarExtractOfTransferReadBase
    : public OpRewritePattern<VectorExtractOp> {
  using Base = OpRewritePattern<VectorExtractOp>;

public:
  RewriteScalarExtractOfTransferReadBase(MLIRContext *context,
                                         PatternBenefit benefit,
                                         bool allowMultipleUses)
      : Base::OpRewritePattern(context, benefit),
        allowMultipleUses(allowMultipleUses) {}

  LogicalResult match(VectorExtractOp extractOp) const override {
    auto xferOp =
        extractOp.getVector().template getDefiningOp<vector::TransferReadOp>();
    if (!xferOp)
      return failure();
    // Check that we are extracting a scalar and not a sub-vector.
    if (isa<VectorType>(extractOp.getResult().getType()))
      return failure();
    // If multiple uses are not allowed, check if xfer has a single use.
    if (!allowMultipleUses && !xferOp.getResult().hasOneUse())
      return failure();
    // If multiple uses are allowed, check if all the xfer uses are extract ops.
    if (allowMultipleUses &&
        !llvm::all_of(xferOp->getUses(), [](OpOperand &use) {
          return isa<vector::ExtractOp, vector::ExtractElementOp>(
              use.getOwner());
        }))
      return failure();
    // Mask not supported.
    if (xferOp.getMask())
      return failure();
    // Map not supported.
    if (!xferOp.getPermutationMap().isMinorIdentity())
      return failure();
    // Cannot rewrite if the indices may be out of bounds.
    if (xferOp.hasOutOfBoundsDim())
      return failure();
    return success();
  }

private:
  bool allowMultipleUses;
};

/// Rewrite `vector.extractelement(vector.transfer_read)` to `memref.load`.
///
/// All the users of the transfer op must be either `vector.extractelement` or
/// `vector.extract` ops. If `allowMultipleUses` is set to true, rewrite
/// transfer ops with any number of users. Otherwise, rewrite only if the
/// extract op is the single user of the transfer op. Rewriting a single
/// vector load with multiple scalar loads may negatively affect performance.
class RewriteScalarExtractElementOfTransferRead
    : public RewriteScalarExtractOfTransferReadBase<vector::ExtractElementOp> {
  using RewriteScalarExtractOfTransferReadBase::
      RewriteScalarExtractOfTransferReadBase;

  void rewrite(vector::ExtractElementOp extractOp,
               PatternRewriter &rewriter) const override {
    // Construct scalar load.
    auto loc = extractOp.getLoc();
    auto xferOp = extractOp.getVector().getDefiningOp<vector::TransferReadOp>();
    SmallVector<Value> newIndices(xferOp.getIndices().begin(),
                                  xferOp.getIndices().end());
    if (extractOp.getPosition()) {
      AffineExpr sym0, sym1;
      bindSymbols(extractOp.getContext(), sym0, sym1);
      OpFoldResult ofr = affine::makeComposedFoldedAffineApply(
          rewriter, loc, sym0 + sym1,
          {newIndices[newIndices.size() - 1], extractOp.getPosition()});
      if (ofr.is<Value>()) {
        newIndices[newIndices.size() - 1] = ofr.get<Value>();
      } else {
        newIndices[newIndices.size() - 1] =
            rewriter.create<arith::ConstantIndexOp>(loc,
                                                    *getConstantIntValue(ofr));
      }
    }
    if (isa<MemRefType>(xferOp.getSource().getType())) {
      rewriter.replaceOpWithNewOp<memref::LoadOp>(extractOp, xferOp.getSource(),
                                                  newIndices);
    } else {
      rewriter.replaceOpWithNewOp<tensor::ExtractOp>(
          extractOp, xferOp.getSource(), newIndices);
    }
  }
};

/// Rewrite `vector.extractelement(vector.transfer_read)` to `memref.load`.
/// Rewrite `vector.extract(vector.transfer_read)` to `memref.load`.
///
/// All the users of the transfer op must be either `vector.extractelement` or
/// `vector.extract` ops. If `allowMultipleUses` is set to true, rewrite
/// transfer ops with any number of users. Otherwise, rewrite only if the
/// extract op is the single user of the transfer op. Rewriting a single
/// vector load with multiple scalar loads may negatively affect performance.
class RewriteScalarExtractOfTransferRead
    : public RewriteScalarExtractOfTransferReadBase<vector::ExtractOp> {
  using RewriteScalarExtractOfTransferReadBase::
      RewriteScalarExtractOfTransferReadBase;

  void rewrite(vector::ExtractOp extractOp,
               PatternRewriter &rewriter) const override {
    // Construct scalar load.
    auto xferOp = extractOp.getVector().getDefiningOp<vector::TransferReadOp>();
    SmallVector<Value> newIndices(xferOp.getIndices().begin(),
                                  xferOp.getIndices().end());
    for (auto [i, pos] : llvm::enumerate(extractOp.getMixedPosition())) {
      assert(pos.is<Attribute>() && "Unexpected non-constant index");
      int64_t offset = cast<IntegerAttr>(pos.get<Attribute>()).getInt();
      int64_t idx = newIndices.size() - extractOp.getNumIndices() + i;
      OpFoldResult ofr = affine::makeComposedFoldedAffineApply(
          rewriter, extractOp.getLoc(),
          rewriter.getAffineSymbolExpr(0) + offset, {newIndices[idx]});
      if (ofr.is<Value>()) {
        newIndices[idx] = ofr.get<Value>();
      } else {
        newIndices[idx] = rewriter.create<arith::ConstantIndexOp>(
            extractOp.getLoc(), *getConstantIntValue(ofr));
      }
    }
    if (isa<MemRefType>(xferOp.getSource().getType())) {
      rewriter.replaceOpWithNewOp<memref::LoadOp>(extractOp, xferOp.getSource(),
                                                  newIndices);
    } else {
      rewriter.replaceOpWithNewOp<tensor::ExtractOp>(
          extractOp, xferOp.getSource(), newIndices);
    }
  }
};

/// Rewrite transfer_writes of vectors of size 1 (e.g., vector<1x1xf32>)
/// to memref.store.
class RewriteScalarWrite : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp xferOp,
                                PatternRewriter &rewriter) const override {
    // Must be a scalar write.
    auto vecType = xferOp.getVectorType();
    if (!llvm::all_of(vecType.getShape(), [](int64_t sz) { return sz == 1; }))
      return failure();
    // Mask not supported.
    if (xferOp.getMask())
      return failure();
    // Map not supported.
    if (!xferOp.getPermutationMap().isMinorIdentity())
      return failure();
    // Only float and integer element types are supported.
    Value scalar;
    if (vecType.getRank() == 0) {
      // vector.extract does not support vector<f32> etc., so use
      // vector.extractelement instead.
      scalar = rewriter.create<vector::ExtractElementOp>(xferOp.getLoc(),
                                                         xferOp.getVector());
    } else {
      SmallVector<int64_t> pos(vecType.getRank(), 0);
      scalar = rewriter.create<vector::ExtractOp>(xferOp.getLoc(),
                                                  xferOp.getVector(), pos);
    }
    // Construct a scalar store.
    if (isa<MemRefType>(xferOp.getSource().getType())) {
      rewriter.replaceOpWithNewOp<memref::StoreOp>(
          xferOp, scalar, xferOp.getSource(), xferOp.getIndices());
    } else {
      rewriter.replaceOpWithNewOp<tensor::InsertOp>(
          xferOp, scalar, xferOp.getSource(), xferOp.getIndices());
    }
    return success();
  }
};

} // namespace

void mlir::vector::transferOpflowOpt(RewriterBase &rewriter,
                                     Operation *rootOp) {
  TransferOptimization opt(rewriter, rootOp);
  // Run store to load forwarding first since it can expose more dead store
  // opportunity.
  rootOp->walk([&](vector::TransferReadOp read) {
    if (isa<MemRefType>(read.getShapedType()))
      opt.storeToLoadForwarding(read);
  });
  opt.removeDeadOp();
  rootOp->walk([&](vector::TransferWriteOp write) {
    if (isa<MemRefType>(write.getShapedType()))
      opt.deadStoreOp(write);
  });
  opt.removeDeadOp();
}

void mlir::vector::populateScalarVectorTransferLoweringPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit,
    bool allowMultipleUses) {
  patterns.add<RewriteScalarExtractElementOfTransferRead,
               RewriteScalarExtractOfTransferRead>(patterns.getContext(),
                                                   benefit, allowMultipleUses);
  patterns.add<RewriteScalarWrite>(patterns.getContext(), benefit);
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
  populateDropUnitDimWithShapeCastPatterns(patterns, benefit);
}
