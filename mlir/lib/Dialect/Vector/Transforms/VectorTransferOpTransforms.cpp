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
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "vector-transfer-opt"

using namespace mlir;

/// Return the ancestor op in the region or nullptr if the region is not
/// an ancestor of the op.
static Operation *findAncestorOpInRegion(Region *region, Operation *op) {
  LDBG() << "    Finding ancestor of " << *op << " in region";
  for (; op != nullptr && op->getParentRegion() != region;
       op = op->getParentOp())
    ;
  if (op) {
    LDBG() << "    -> Ancestor: " << *op;
  } else {
    LDBG() << "    -> Ancestor: nullptr";
  }
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
    LDBG() << "Removing " << opToErase.size() << " dead operations";
    for (Operation *op : opToErase) {
      LDBG() << "  -> Erasing: " << *op;
      rewriter.eraseOp(op);
    }
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
  LDBG() << "    Checking reachability from " << *start << " to " << *dest;
  assert(start->getParentRegion() == dest->getParentRegion() &&
         "This function only works for ops i the same region");
  // Simple case where the start op dominate the destination.
  if (dominators.dominates(start, dest)) {
    LDBG() << "    -> Start dominates dest, reachable";
    return true;
  }
  bool blockReachable = start->getBlock()->isReachable(dest->getBlock());
  LDBG() << "    -> Block reachable: " << blockReachable;
  return blockReachable;
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
  LDBG() << "=== Starting deadStoreOp analysis for: " << *write.getOperation();
  llvm::SmallVector<Operation *, 8> blockingAccesses;
  Operation *firstOverwriteCandidate = nullptr;
  Value source = memref::skipViewLikeOps(cast<MemrefValue>(write.getBase()));
  LDBG() << "Source memref (after skipping view-like ops): " << source;
  llvm::SmallVector<Operation *, 32> users(source.getUsers().begin(),
                                           source.getUsers().end());
  LDBG() << "Found " << users.size() << " users of source memref";
  llvm::SmallDenseSet<Operation *, 32> processed;
  while (!users.empty()) {
    Operation *user = users.pop_back_val();
    LDBG() << "Processing user: " << *user;
    // If the user has already been processed skip.
    if (!processed.insert(user).second) {
      LDBG() << "  -> Already processed, skipping";
      continue;
    }
    if (auto viewLike = dyn_cast<ViewLikeOpInterface>(user)) {
      LDBG() << "  -> View-like operation, following to destination";
      Value viewDest = viewLike.getViewDest();
      users.append(viewDest.getUsers().begin(), viewDest.getUsers().end());
      continue;
    }
    if (isMemoryEffectFree(user)) {
      LDBG() << "  -> Memory effect free, skipping";
      continue;
    }
    if (user == write.getOperation()) {
      LDBG() << "  -> Same as write operation, skipping";
      continue;
    }
    if (auto nextWrite = dyn_cast<vector::TransferWriteOp>(user)) {
      LDBG() << "  -> Found transfer_write candidate: " << *nextWrite;
      // Check candidate that can override the store.
      bool sameView = memref::isSameViewOrTrivialAlias(
          cast<MemrefValue>(nextWrite.getBase()),
          cast<MemrefValue>(write.getBase()));
      bool sameValue = checkSameValueWAW(nextWrite, write);
      bool postDominates = postDominators.postDominates(nextWrite, write);
      LDBG() << "    -> Same view: " << sameView
             << ", Same value: " << sameValue
             << ", Post-dominates: " << postDominates;

      if (sameView && sameValue && postDominates) {
        LDBG() << "    -> Valid overwrite candidate found";
        if (firstOverwriteCandidate == nullptr ||
            postDominators.postDominates(firstOverwriteCandidate, nextWrite)) {
          LDBG() << "    -> New first overwrite candidate: " << *nextWrite;
          firstOverwriteCandidate = nextWrite;
        } else {
          LDBG() << "    -> Keeping existing first overwrite candidate";
          assert(
              postDominators.postDominates(nextWrite, firstOverwriteCandidate));
        }
        continue;
      }
      LDBG() << "    -> Not a valid overwrite candidate";
    }
    if (auto transferOp = dyn_cast<VectorTransferOpInterface>(user)) {
      LDBG() << "  -> Found vector transfer operation: " << *transferOp;
      // Don't need to consider disjoint accesses.
      bool isDisjoint = vector::isDisjointTransferSet(
          cast<VectorTransferOpInterface>(write.getOperation()),
          cast<VectorTransferOpInterface>(transferOp.getOperation()),
          /*testDynamicValueUsingBounds=*/true);
      LDBG() << "    -> Is disjoint: " << isDisjoint;
      if (isDisjoint) {
        LDBG() << "    -> Skipping disjoint access";
        continue;
      }
    }
    LDBG() << "  -> Adding to blocking accesses: " << *user;
    blockingAccesses.push_back(user);
  }
  LDBG() << "Finished processing users. Found " << blockingAccesses.size()
         << " blocking accesses";

  if (firstOverwriteCandidate == nullptr) {
    LDBG() << "No overwrite candidate found, store is not dead";
    return;
  }

  LDBG() << "First overwrite candidate: " << *firstOverwriteCandidate;
  Region *topRegion = firstOverwriteCandidate->getParentRegion();
  Operation *writeAncestor = findAncestorOpInRegion(topRegion, write);
  assert(writeAncestor &&
         "write op should be recursively part of the top region");
  LDBG() << "Write ancestor in top region: " << *writeAncestor;

  LDBG() << "Checking " << blockingAccesses.size()
         << " blocking accesses for reachability";
  for (Operation *access : blockingAccesses) {
    LDBG() << "Checking blocking access: " << *access;
    Operation *accessAncestor = findAncestorOpInRegion(topRegion, access);
    // TODO: if the access and write have the same ancestor we could recurse in
    // the region to know if the access is reachable with more precision.
    if (accessAncestor == nullptr) {
      LDBG() << "  -> No ancestor in top region, skipping";
      continue;
    }

    bool isReachableFromWrite = isReachable(writeAncestor, accessAncestor);
    LDBG() << "  -> Is reachable from write: " << isReachableFromWrite;
    if (!isReachableFromWrite) {
      LDBG() << "  -> Not reachable, skipping";
      continue;
    }

    bool overwriteDominatesAccess =
        dominators.dominates(firstOverwriteCandidate, accessAncestor);
    LDBG() << "  -> Overwrite dominates access: " << overwriteDominatesAccess;
    if (!overwriteDominatesAccess) {
      LDBG() << "Store may not be dead due to op: " << *accessAncestor;
      return;
    }
    LDBG() << "  -> Access is dominated by overwrite, continuing";
  }
  LDBG() << "Found dead store: " << *write.getOperation()
         << " overwritten by: " << *firstOverwriteCandidate;
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
  LDBG() << "=== Starting storeToLoadForwarding analysis for: "
         << *read.getOperation();
  if (read.hasOutOfBoundsDim()) {
    LDBG() << "Read has out-of-bounds dimensions, skipping";
    return;
  }
  SmallVector<Operation *, 8> blockingWrites;
  vector::TransferWriteOp lastwrite = nullptr;
  Value source = memref::skipViewLikeOps(cast<MemrefValue>(read.getBase()));
  LDBG() << "Source memref (after skipping view-like ops): " << source;
  llvm::SmallVector<Operation *, 32> users(source.getUsers().begin(),
                                           source.getUsers().end());
  LDBG() << "Found " << users.size() << " users of source memref";
  llvm::SmallDenseSet<Operation *, 32> processed;
  while (!users.empty()) {
    Operation *user = users.pop_back_val();
    LDBG() << "Processing user: " << *user;
    // If the user has already been processed skip.
    if (!processed.insert(user).second) {
      LDBG() << "  -> Already processed, skipping";
      continue;
    }
    if (auto viewLike = dyn_cast<ViewLikeOpInterface>(user)) {
      LDBG() << "  -> View-like operation, following to destination";
      Value viewDest = viewLike.getViewDest();
      users.append(viewDest.getUsers().begin(), viewDest.getUsers().end());
      continue;
    }
    if (isMemoryEffectFree(user) || isa<vector::TransferReadOp>(user)) {
      LDBG() << "  -> Memory effect free or transfer_read, skipping";
      continue;
    }
    if (auto write = dyn_cast<vector::TransferWriteOp>(user)) {
      LDBG() << "  -> Found transfer_write candidate: " << *write;
      // If there is a write, but we can prove that it is disjoint we can ignore
      // the write.
      bool isDisjoint = vector::isDisjointTransferSet(
          cast<VectorTransferOpInterface>(write.getOperation()),
          cast<VectorTransferOpInterface>(read.getOperation()),
          /*testDynamicValueUsingBounds=*/true);
      LDBG() << "    -> Is disjoint: " << isDisjoint;
      if (isDisjoint) {
        LDBG() << "    -> Skipping disjoint write";
        continue;
      }

      bool sameView =
          memref::isSameViewOrTrivialAlias(cast<MemrefValue>(read.getBase()),
                                           cast<MemrefValue>(write.getBase()));
      bool dominates = dominators.dominates(write, read);
      bool sameValue = checkSameValueRAW(write, read);
      LDBG() << "    -> Same view: " << sameView << ", Dominates: " << dominates
             << ", Same value: " << sameValue;

      if (sameView && dominates && sameValue) {
        LDBG() << "    -> Valid forwarding candidate found";
        if (lastwrite == nullptr || dominators.dominates(lastwrite, write)) {
          LDBG() << "    -> New last write candidate: " << *write;
          lastwrite = write;
        } else {
          LDBG() << "    -> Keeping existing last write candidate";
          assert(dominators.dominates(write, lastwrite));
        }
        continue;
      }
      LDBG() << "    -> Not a valid forwarding candidate";
    }
    LDBG() << "  -> Adding to blocking writes: " << *user;
    blockingWrites.push_back(user);
  }
  LDBG() << "Finished processing users. Found " << blockingWrites.size()
         << " blocking writes";

  if (lastwrite == nullptr) {
    LDBG() << "No last write candidate found, cannot forward";
    return;
  }

  LDBG() << "Last write candidate: " << *lastwrite;
  Region *topRegion = lastwrite->getParentRegion();
  Operation *readAncestor = findAncestorOpInRegion(topRegion, read);
  assert(readAncestor &&
         "read op should be recursively part of the top region");
  LDBG() << "Read ancestor in top region: " << *readAncestor;

  LDBG() << "Checking " << blockingWrites.size()
         << " blocking writes for post-dominance";
  for (Operation *write : blockingWrites) {
    LDBG() << "Checking blocking write: " << *write;
    Operation *writeAncestor = findAncestorOpInRegion(topRegion, write);
    if (writeAncestor) {
      LDBG() << "  -> Write ancestor: " << *writeAncestor;
    } else {
      LDBG() << "  -> Write ancestor: nullptr";
    }

    // TODO: if the store and read have the same ancestor we could recurse in
    // the region to know if the read is reachable with more precision.
    if (writeAncestor == nullptr) {
      LDBG() << "  -> No ancestor in top region, skipping";
      continue;
    }

    bool isReachableToRead = isReachable(writeAncestor, readAncestor);
    LDBG() << "  -> Is reachable to read: " << isReachableToRead;
    if (!isReachableToRead) {
      LDBG() << "  -> Not reachable, skipping";
      continue;
    }

    bool lastWritePostDominates =
        postDominators.postDominates(lastwrite, write);
    LDBG() << "  -> Last write post-dominates blocking write: "
           << lastWritePostDominates;
    if (!lastWritePostDominates) {
      LDBG() << "Fail to do write to read forwarding due to op: " << *write;
      return;
    }
    LDBG() << "  -> Blocking write is post-dominated, continuing";
  }

  LDBG() << "Forward value from " << *lastwrite.getOperation()
         << " to: " << *read.getOperation();
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

    auto value = cast<IntegerAttr>(cast<Attribute>(size)).getValue();
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
  MemRefType rankReducedType = memref::SubViewOp::inferRankReducedResultType(
      targetShape, inputType, offsets, sizes, strides);
  return rankReducedType.canonicalizeStridedLayout();
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

  if (resultType.canonicalizeStridedLayout() ==
      inputType.canonicalizeStridedLayout())
    return input;
  return memref::SubViewOp::create(rewriter, loc, resultType, input, offsets,
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
  return vector::CreateMaskOp::create(rewriter, loc, reducedType,
                                      reducedOperands)
      .getResult();
}

namespace {

/// Rewrites `vector.transfer_read` ops where the source has unit dims, by
/// inserting a memref.subview dropping those unit dims. The vector shapes are
/// also reduced accordingly.
class TransferReadDropUnitDimsPattern
    : public vector::MaskableOpRewritePattern<vector::TransferReadOp> {
  using MaskableOpRewritePattern::MaskableOpRewritePattern;

  FailureOr<Value>
  matchAndRewriteMaskableOp(vector::TransferReadOp transferReadOp,
                            vector::MaskingOpInterface maskingOp,
                            PatternRewriter &rewriter) const override {
    LDBG() << "=== TransferReadDropUnitDimsPattern: Analyzing "
           << *transferReadOp;
    auto loc = transferReadOp.getLoc();
    Value vector = transferReadOp.getVector();
    VectorType vectorType = cast<VectorType>(vector.getType());
    Value source = transferReadOp.getBase();
    MemRefType sourceType = dyn_cast<MemRefType>(source.getType());
    // TODO: support tensor types.
    if (!sourceType) {
      LDBG() << "  -> Not a MemRefType, skipping";
      return failure();
    }
    // TODO: generalize this pattern, relax the requirements here.
    if (transferReadOp.hasOutOfBoundsDim()) {
      LDBG() << "  -> Has out-of-bounds dimensions, skipping";
      return failure();
    }
    if (!transferReadOp.getPermutationMap().isMinorIdentity()) {
      LDBG() << "  -> Not minor identity permutation map, skipping";
      return failure();
    }
    // Check if the source shape can be further reduced.
    int reducedRank = getReducedRank(sourceType.getShape());
    LDBG() << "  -> Source rank: " << sourceType.getRank()
           << ", Reduced rank: " << reducedRank;
    if (reducedRank == sourceType.getRank()) {
      LDBG() << "  -> No unit dimensions to drop, skipping";
      return failure();
    }
    // TODO: Extend vector.mask to support 0-d vectors. In the meantime, bail
    // out.
    if (reducedRank == 0 && maskingOp) {
      LDBG() << "  -> 0-d vector with masking not supported, skipping";
      return failure();
    }
    // Check if the reduced vector shape matches the reduced source shape.
    // Otherwise, this case is not supported yet.
    VectorType reducedVectorType = trimNonScalableUnitDims(vectorType);
    LDBG() << "  -> Vector type: " << vectorType
           << ", Reduced vector type: " << reducedVectorType;
    if (reducedRank != reducedVectorType.getRank()) {
      LDBG() << "  -> Reduced ranks don't match, skipping";
      return failure();
    }
    if (llvm::any_of(transferReadOp.getIndices(), [](Value v) {
          return getConstantIntValue(v) != static_cast<int64_t>(0);
        })) {
      LDBG() << "  -> Non-zero indices found, skipping";
      return failure();
    }

    Value maskOp = transferReadOp.getMask();
    if (maskOp) {
      LDBG() << "  -> Processing mask operation";
      auto createMaskOp = maskOp.getDefiningOp<vector::CreateMaskOp>();
      if (!createMaskOp) {
        LDBG()
            << "  -> Unsupported mask op, only 'vector.create_mask' supported";
        return rewriter.notifyMatchFailure(
            transferReadOp, "unsupported mask op, only 'vector.create_mask' is "
                            "currently supported");
      }
      FailureOr<Value> rankReducedCreateMask =
          createMaskDropNonScalableUnitDims(rewriter, loc, createMaskOp);
      if (failed(rankReducedCreateMask)) {
        LDBG() << "  -> Failed to reduce mask dimensions";
        return failure();
      }
      maskOp = *rankReducedCreateMask;
      LDBG() << "  -> Successfully reduced mask dimensions";
    }

    LDBG() << "  -> Creating rank-reduced subview and new transfer_read";
    Value reducedShapeSource =
        rankReducingSubviewDroppingUnitDims(rewriter, loc, source);
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    SmallVector<Value> zeros(reducedRank, c0);
    auto identityMap = rewriter.getMultiDimIdentityMap(reducedRank);
    SmallVector<bool> inBounds(reducedVectorType.getRank(), true);
    Operation *newTransferReadOp = vector::TransferReadOp::create(
        rewriter, loc, reducedVectorType, reducedShapeSource, zeros,
        identityMap, transferReadOp.getPadding(), maskOp,
        rewriter.getBoolArrayAttr(inBounds));
    LDBG() << "  -> Created new transfer_read: " << *newTransferReadOp;

    if (maskingOp) {
      LDBG() << "  -> Applying masking operation";
      auto shapeCastMask = rewriter.createOrFold<vector::ShapeCastOp>(
          loc, reducedVectorType.cloneWith(std::nullopt, rewriter.getI1Type()),
          maskingOp.getMask());
      newTransferReadOp = mlir::vector::maskOperation(
          rewriter, newTransferReadOp, shapeCastMask);
    }

    auto shapeCast = rewriter.createOrFold<vector::ShapeCastOp>(
        loc, vectorType, newTransferReadOp->getResults()[0]);
    LDBG() << "  -> Created shape cast: " << *shapeCast.getDefiningOp();
    LDBG() << "  -> Pattern match successful, returning result";

    return shapeCast;
  }
};

/// Rewrites `vector.transfer_write` ops where the "source" (i.e. destination)
/// has unit dims, by inserting a `memref.subview` dropping those unit dims. The
/// vector shapes are also reduced accordingly.
class TransferWriteDropUnitDimsPattern
    : public vector::MaskableOpRewritePattern<vector::TransferWriteOp> {
  using MaskableOpRewritePattern::MaskableOpRewritePattern;

  FailureOr<Value>
  matchAndRewriteMaskableOp(vector::TransferWriteOp transferWriteOp,
                            vector::MaskingOpInterface maskingOp,
                            PatternRewriter &rewriter) const override {
    LDBG() << "=== TransferWriteDropUnitDimsPattern: Analyzing "
           << *transferWriteOp;
    auto loc = transferWriteOp.getLoc();
    Value vector = transferWriteOp.getVector();
    VectorType vectorType = cast<VectorType>(vector.getType());
    Value source = transferWriteOp.getBase();
    MemRefType sourceType = dyn_cast<MemRefType>(source.getType());
    // TODO: support tensor type.
    if (!sourceType) {
      LDBG() << "  -> Not a MemRefType, skipping";
      return failure();
    }
    // TODO: generalize this pattern, relax the requirements here.
    if (transferWriteOp.hasOutOfBoundsDim()) {
      LDBG() << "  -> Has out-of-bounds dimensions, skipping";
      return failure();
    }
    if (!transferWriteOp.getPermutationMap().isMinorIdentity()) {
      LDBG() << "  -> Not minor identity permutation map, skipping";
      return failure();
    }
    // Check if the destination shape can be further reduced.
    int reducedRank = getReducedRank(sourceType.getShape());
    LDBG() << "  -> Source rank: " << sourceType.getRank()
           << ", Reduced rank: " << reducedRank;
    if (reducedRank == sourceType.getRank()) {
      LDBG() << "  -> No unit dimensions to drop, skipping";
      return failure();
    }
    // TODO: Extend vector.mask to support 0-d vectors. In the meantime, bail
    // out.
    if (reducedRank == 0 && maskingOp) {
      LDBG() << "  -> 0-d vector with masking not supported, skipping";
      return failure();
    }
    // Check if the reduced vector shape matches the reduced destination shape.
    // Otherwise, this case is not supported yet.
    VectorType reducedVectorType = trimNonScalableUnitDims(vectorType);
    LDBG() << "  -> Vector type: " << vectorType
           << ", Reduced vector type: " << reducedVectorType;
    if (reducedRank != reducedVectorType.getRank()) {
      LDBG() << "  -> Reduced ranks don't match, skipping";
      return failure();
    }
    if (llvm::any_of(transferWriteOp.getIndices(), [](Value v) {
          return getConstantIntValue(v) != static_cast<int64_t>(0);
        })) {
      LDBG() << "  -> Non-zero indices found, skipping";
      return failure();
    }

    Value maskOp = transferWriteOp.getMask();
    if (maskOp) {
      LDBG() << "  -> Processing mask operation";
      auto createMaskOp = maskOp.getDefiningOp<vector::CreateMaskOp>();
      if (!createMaskOp) {
        LDBG()
            << "  -> Unsupported mask op, only 'vector.create_mask' supported";
        return rewriter.notifyMatchFailure(
            transferWriteOp,
            "unsupported mask op, only 'vector.create_mask' is "
            "currently supported");
      }
      FailureOr<Value> rankReducedCreateMask =
          createMaskDropNonScalableUnitDims(rewriter, loc, createMaskOp);
      if (failed(rankReducedCreateMask)) {
        LDBG() << "  -> Failed to reduce mask dimensions";
        return failure();
      }
      maskOp = *rankReducedCreateMask;
      LDBG() << "  -> Successfully reduced mask dimensions";
    }
    LDBG() << "  -> Creating rank-reduced subview and new transfer_write";
    Value reducedShapeSource =
        rankReducingSubviewDroppingUnitDims(rewriter, loc, source);
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    SmallVector<Value> zeros(reducedRank, c0);
    auto identityMap = rewriter.getMultiDimIdentityMap(reducedRank);
    SmallVector<bool> inBounds(reducedVectorType.getRank(), true);
    auto shapeCastSrc = rewriter.createOrFold<vector::ShapeCastOp>(
        loc, reducedVectorType, vector);
    Operation *newXferWrite = vector::TransferWriteOp::create(
        rewriter, loc, Type(), shapeCastSrc, reducedShapeSource, zeros,
        identityMap, maskOp, rewriter.getBoolArrayAttr(inBounds));
    LDBG() << "  -> Created new transfer_write: " << *newXferWrite;

    if (maskingOp) {
      LDBG() << "  -> Applying masking operation";
      auto shapeCastMask = rewriter.createOrFold<vector::ShapeCastOp>(
          loc, reducedVectorType.cloneWith(std::nullopt, rewriter.getI1Type()),
          maskingOp.getMask());
      newXferWrite =
          mlir::vector::maskOperation(rewriter, newXferWrite, shapeCastMask);
    }

    if (transferWriteOp.hasPureTensorSemantics()) {
      LDBG() << "  -> Pattern match successful (tensor semantics), returning "
                "result";
      return newXferWrite->getResults()[0];
    }

    // With Memref semantics, there's no return value. Use empty value to signal
    // success.
    LDBG() << "  -> Pattern match successful (memref semantics)";
    return Value();
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
  return memref::CollapseShapeOp::create(rewriter, loc, input, reassociation);
}

/// Returns the new indices that collapses the inner dimensions starting from
/// the `firstDimToCollapse` dimension.
static SmallVector<Value> getCollapsedIndices(RewriterBase &rewriter,
                                              Location loc,
                                              ArrayRef<int64_t> shape,
                                              ValueRange indices,
                                              int64_t firstDimToCollapse) {
  assert(firstDimToCollapse < static_cast<int64_t>(indices.size()));

  // If all the collapsed indices are zero then no extra logic is needed.
  // Otherwise, a new offset/index has to be computed.
  SmallVector<Value> indicesAfterCollapsing(
      indices.begin(), indices.begin() + firstDimToCollapse);
  SmallVector<Value> indicesToCollapse(indices.begin() + firstDimToCollapse,
                                       indices.end());
  if (llvm::all_of(indicesToCollapse, isZeroInteger)) {
    indicesAfterCollapsing.push_back(indicesToCollapse[0]);
    return indicesAfterCollapsing;
  }

  // Compute the remaining trailing index/offset required for reading from
  // the collapsed memref:
  //
  //    offset = 0
  //    for (i = firstDimToCollapse; i < outputRank; ++i)
  //      offset += sourceType.getDimSize(i) * transferReadOp.indices[i]
  //
  // For this example:
  //   %2 = vector.transfer_read/write %arg4[%c0, %arg0, %c0] (...) :
  //      memref<1x43x2xi32>, vector<1x2xi32>
  // which would be collapsed to:
  //   %1 = vector.transfer_read/write %collapse_shape[%c0, %offset] (...) :
  //      memref<1x86xi32>, vector<2xi32>
  // one would get the following offset:
  //    %offset = %arg0 * 43
  OpFoldResult collapsedOffset =
      arith::ConstantIndexOp::create(rewriter, loc, 0).getResult();

  auto collapsedStrides = computeSuffixProduct(
      ArrayRef<int64_t>(shape.begin() + firstDimToCollapse, shape.end()));

  // Compute the collapsed offset.
  auto &&[collapsedExpr, collapsedVals] =
      computeLinearIndex(collapsedOffset, collapsedStrides, indicesToCollapse);
  collapsedOffset = affine::makeComposedFoldedAffineApply(
      rewriter, loc, collapsedExpr, collapsedVals);

  if (auto value = dyn_cast<Value>(collapsedOffset)) {
    indicesAfterCollapsing.push_back(value);
  } else {
    indicesAfterCollapsing.push_back(arith::ConstantIndexOp::create(
        rewriter, loc, *getConstantIntValue(collapsedOffset)));
  }

  return indicesAfterCollapsing;
}

namespace {
/// Rewrites contiguous row-major vector.transfer_read ops by inserting
/// memref.collapse_shape on the source so that the resulting
/// vector.transfer_read has a 1D source. Requires the source shape to be
/// already reduced i.e. without unit dims.
///
/// If `targetVectorBitwidth` is provided, the flattening will only happen if
/// the trailing dimension of the vector read is smaller than the provided
/// bitwidth.
class FlattenContiguousRowMajorTransferReadPattern
    : public OpRewritePattern<vector::TransferReadOp> {
public:
  FlattenContiguousRowMajorTransferReadPattern(MLIRContext *context,
                                               unsigned vectorBitwidth,
                                               PatternBenefit benefit)
      : OpRewritePattern<vector::TransferReadOp>(context, benefit),
        targetVectorBitwidth(vectorBitwidth) {}

  LogicalResult matchAndRewrite(vector::TransferReadOp transferReadOp,
                                PatternRewriter &rewriter) const override {
    LDBG() << "=== FlattenContiguousRowMajorTransferReadPattern: Analyzing "
           << *transferReadOp;
    auto loc = transferReadOp.getLoc();
    Value vector = transferReadOp.getVector();
    VectorType vectorType = cast<VectorType>(vector.getType());
    auto source = transferReadOp.getBase();
    MemRefType sourceType = dyn_cast<MemRefType>(source.getType());

    // 0. Check pre-conditions
    // Contiguity check is valid on tensors only.
    if (!sourceType) {
      LDBG() << "  -> Not a MemRefType, skipping";
      return failure();
    }
    // If this is already 0D/1D, there's nothing to do.
    if (vectorType.getRank() <= 1) {
      LDBG() << "  -> Already 0D/1D, skipping";
      return failure();
    }
    if (!vectorType.getElementType().isSignlessIntOrFloat()) {
      LDBG() << "  -> Not signless int or float, skipping";
      return failure();
    }
    unsigned trailingVectorDimBitwidth =
        vectorType.getShape().back() * vectorType.getElementTypeBitWidth();
    LDBG() << "  -> Trailing vector dim bitwidth: " << trailingVectorDimBitwidth
           << ", target: " << targetVectorBitwidth;
    if (trailingVectorDimBitwidth >= targetVectorBitwidth) {
      LDBG() << "  -> Trailing dim bitwidth >= target, skipping";
      return failure();
    }
    if (!vector::isContiguousSlice(sourceType, vectorType)) {
      LDBG() << "  -> Not contiguous slice, skipping";
      return failure();
    }
    // TODO: generalize this pattern, relax the requirements here.
    if (transferReadOp.hasOutOfBoundsDim()) {
      LDBG() << "  -> Has out-of-bounds dimensions, skipping";
      return failure();
    }
    if (!transferReadOp.getPermutationMap().isMinorIdentity()) {
      LDBG() << "  -> Not minor identity permutation map, skipping";
      return failure();
    }
    if (transferReadOp.getMask()) {
      LDBG() << "  -> Has mask, skipping";
      return failure();
    }

    // Determine the first memref dimension to collapse - just enough so we can
    // read a flattened vector.
    int64_t firstDimToCollapse =
        sourceType.getRank() -
        vectorType.getShape().drop_while([](auto v) { return v == 1; }).size();
    LDBG() << "  -> First dimension to collapse: " << firstDimToCollapse;

    // 1. Collapse the source memref
    LDBG() << "  -> Collapsing source memref";
    Value collapsedSource =
        collapseInnerDims(rewriter, loc, source, firstDimToCollapse);
    MemRefType collapsedSourceType =
        cast<MemRefType>(collapsedSource.getType());
    int64_t collapsedRank = collapsedSourceType.getRank();
    assert(collapsedRank == firstDimToCollapse + 1);
    LDBG() << "  -> Collapsed source type: " << collapsedSourceType;

    // 2. Generate input args for a new vector.transfer_read that will read
    // from the collapsed memref.
    // 2.1. New dim exprs + affine map
    SmallVector<AffineExpr, 1> dimExprs{
        getAffineDimExpr(firstDimToCollapse, rewriter.getContext())};
    auto collapsedMap =
        AffineMap::get(collapsedRank, 0, dimExprs, rewriter.getContext());

    // 2.2 New indices
    SmallVector<Value> collapsedIndices =
        getCollapsedIndices(rewriter, loc, sourceType.getShape(),
                            transferReadOp.getIndices(), firstDimToCollapse);

    // 3. Create new vector.transfer_read that reads from the collapsed memref
    VectorType flatVectorType = VectorType::get({vectorType.getNumElements()},
                                                vectorType.getElementType());
    LDBG() << "  -> Creating flattened vector type: " << flatVectorType;
    vector::TransferReadOp flatRead = vector::TransferReadOp::create(
        rewriter, loc, flatVectorType, collapsedSource, collapsedIndices,
        transferReadOp.getPadding(), collapsedMap);
    flatRead.setInBoundsAttr(rewriter.getBoolArrayAttr({true}));
    LDBG() << "  -> Created flat transfer_read: " << *flatRead;

    // 4. Replace the old transfer_read with the new one reading from the
    // collapsed shape
    LDBG() << "  -> Replacing with shape cast";
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
        transferReadOp, cast<VectorType>(vector.getType()), flatRead);
    LDBG() << "  -> Pattern match successful";
    return success();
  }

private:
  // Minimum bitwidth that the trailing vector dimension should have after
  // flattening.
  unsigned targetVectorBitwidth;
};

/// Rewrites contiguous row-major vector.transfer_write ops by inserting
/// memref.collapse_shape on the source so that the resulting
/// vector.transfer_write has a 1D source. Requires the source shape to be
/// already reduced i.e. without unit dims.
///
/// If `targetVectorBitwidth` is provided, the flattening will only happen if
/// the trailing dimension of the vector read is smaller than the provided
/// bitwidth.
class FlattenContiguousRowMajorTransferWritePattern
    : public OpRewritePattern<vector::TransferWriteOp> {
public:
  FlattenContiguousRowMajorTransferWritePattern(MLIRContext *context,
                                                unsigned vectorBitwidth,
                                                PatternBenefit benefit)
      : OpRewritePattern<vector::TransferWriteOp>(context, benefit),
        targetVectorBitwidth(vectorBitwidth) {}

  LogicalResult matchAndRewrite(vector::TransferWriteOp transferWriteOp,
                                PatternRewriter &rewriter) const override {
    auto loc = transferWriteOp.getLoc();
    Value vector = transferWriteOp.getVector();
    VectorType vectorType = cast<VectorType>(vector.getType());
    Value source = transferWriteOp.getBase();
    MemRefType sourceType = dyn_cast<MemRefType>(source.getType());

    // 0. Check pre-conditions
    // Contiguity check is valid on tensors only.
    if (!sourceType)
      return failure();
    // If this is already 0D/1D, there's nothing to do.
    if (vectorType.getRank() <= 1)
      // Already 0D/1D, nothing to do.
      return failure();
    if (!vectorType.getElementType().isSignlessIntOrFloat())
      return failure();
    unsigned trailingVectorDimBitwidth =
        vectorType.getShape().back() * vectorType.getElementTypeBitWidth();
    if (trailingVectorDimBitwidth >= targetVectorBitwidth)
      return failure();
    if (!vector::isContiguousSlice(sourceType, vectorType))
      return failure();
    // TODO: generalize this pattern, relax the requirements here.
    if (transferWriteOp.hasOutOfBoundsDim())
      return failure();
    if (!transferWriteOp.getPermutationMap().isMinorIdentity())
      return failure();
    if (transferWriteOp.getMask())
      return failure();

    // Determine the first memref dimension to collapse - just enough so we can
    // read a flattened vector.
    int64_t firstDimToCollapse =
        sourceType.getRank() -
        vectorType.getShape().drop_while([](auto v) { return v == 1; }).size();

    // 1. Collapse the source memref
    Value collapsedSource =
        collapseInnerDims(rewriter, loc, source, firstDimToCollapse);
    MemRefType collapsedSourceType =
        cast<MemRefType>(collapsedSource.getType());
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
    SmallVector<Value> collapsedIndices =
        getCollapsedIndices(rewriter, loc, sourceType.getShape(),
                            transferWriteOp.getIndices(), firstDimToCollapse);

    // 3. Create new vector.transfer_write that writes to the collapsed memref
    VectorType flatVectorType = VectorType::get({vectorType.getNumElements()},
                                                vectorType.getElementType());
    Value flatVector =
        vector::ShapeCastOp::create(rewriter, loc, flatVectorType, vector);
    vector::TransferWriteOp flatWrite = vector::TransferWriteOp::create(
        rewriter, loc, flatVector, collapsedSource, collapsedIndices,
        collapsedMap);
    flatWrite.setInBoundsAttr(rewriter.getBoolArrayAttr({true}));

    // 4. Replace the old transfer_write with the new one writing the
    // collapsed shape
    rewriter.eraseOp(transferWriteOp);
    return success();
  }

private:
  // Minimum bitwidth that the trailing vector dimension should have after
  // flattening.
  unsigned targetVectorBitwidth;
};

/// Rewrite `vector.extract(vector.transfer_read)` to `memref.load`.
///
/// All the users of the transfer op must be `vector.extract` ops. If
/// `allowMultipleUses` is set to true, rewrite transfer ops with any number of
/// users. Otherwise, rewrite only if the extract op is the single user of the
/// transfer op. Rewriting a single vector load with multiple scalar loads may
/// negatively affect performance.
class RewriteScalarExtractOfTransferRead
    : public OpRewritePattern<vector::ExtractOp> {
public:
  RewriteScalarExtractOfTransferRead(MLIRContext *context,
                                     PatternBenefit benefit,
                                     bool allowMultipleUses)
      : OpRewritePattern(context, benefit),
        allowMultipleUses(allowMultipleUses) {}

  LogicalResult matchAndRewrite(vector::ExtractOp extractOp,
                                PatternRewriter &rewriter) const override {
    // Match phase.
    auto xferOp = extractOp.getVector().getDefiningOp<vector::TransferReadOp>();
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
          return isa<vector::ExtractOp>(use.getOwner());
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

    // Rewrite phase: construct scalar load.
    SmallVector<Value> newIndices(xferOp.getIndices().begin(),
                                  xferOp.getIndices().end());
    for (auto [i, pos] : llvm::enumerate(extractOp.getMixedPosition())) {
      int64_t idx = newIndices.size() - extractOp.getNumIndices() + i;

      // Compute affine expression `newIndices[idx] + pos` where `pos` can be
      // either a constant or a value.
      OpFoldResult composedIdx;
      if (auto attr = dyn_cast<Attribute>(pos)) {
        int64_t offset = cast<IntegerAttr>(attr).getInt();
        composedIdx = affine::makeComposedFoldedAffineApply(
            rewriter, extractOp.getLoc(),
            rewriter.getAffineSymbolExpr(0) + offset, {newIndices[idx]});
      } else {
        Value dynamicOffset = cast<Value>(pos);
        AffineExpr sym0, sym1;
        bindSymbols(rewriter.getContext(), sym0, sym1);
        composedIdx = affine::makeComposedFoldedAffineApply(
            rewriter, extractOp.getLoc(), sym0 + sym1,
            {newIndices[idx], dynamicOffset});
      }

      // Update the corresponding index with the folded result.
      if (auto value = dyn_cast<Value>(composedIdx)) {
        newIndices[idx] = value;
      } else {
        newIndices[idx] = arith::ConstantIndexOp::create(
            rewriter, extractOp.getLoc(), *getConstantIntValue(composedIdx));
      }
    }
    if (isa<MemRefType>(xferOp.getBase().getType())) {
      rewriter.replaceOpWithNewOp<memref::LoadOp>(extractOp, xferOp.getBase(),
                                                  newIndices);
    } else {
      rewriter.replaceOpWithNewOp<tensor::ExtractOp>(
          extractOp, xferOp.getBase(), newIndices);
    }

    return success();
  }

private:
  bool allowMultipleUses;
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
    Value scalar = vector::ExtractOp::create(rewriter, xferOp.getLoc(),
                                             xferOp.getVector());
    // Construct a scalar store.
    if (isa<MemRefType>(xferOp.getBase().getType())) {
      rewriter.replaceOpWithNewOp<memref::StoreOp>(
          xferOp, scalar, xferOp.getBase(), xferOp.getIndices());
    } else {
      rewriter.replaceOpWithNewOp<tensor::InsertOp>(
          xferOp, scalar, xferOp.getBase(), xferOp.getIndices());
    }
    return success();
  }
};

} // namespace

void mlir::vector::transferOpflowOpt(RewriterBase &rewriter,
                                     Operation *rootOp) {
  LDBG() << "=== Starting transferOpflowOpt on root operation: "
         << OpWithFlags(rootOp, OpPrintingFlags().skipRegions());
  TransferOptimization opt(rewriter, rootOp);

  // Run store to load forwarding first since it can expose more dead store
  // opportunity.
  LDBG() << "Phase 1: Store-to-load forwarding";
  int readCount = 0;
  rootOp->walk([&](vector::TransferReadOp read) {
    if (isa<MemRefType>(read.getShapedType())) {
      LDBG() << "Processing transfer_read #" << ++readCount << ": " << *read;
      opt.storeToLoadForwarding(read);
    }
  });
  LDBG() << "Phase 1 complete. Removing dead operations from forwarding";
  opt.removeDeadOp();

  LDBG() << "Phase 2: Dead store elimination";
  int writeCount = 0;
  rootOp->walk([&](vector::TransferWriteOp write) {
    if (isa<MemRefType>(write.getShapedType())) {
      LDBG() << "Processing transfer_write #" << ++writeCount << ": " << *write;
      opt.deadStoreOp(write);
    }
  });
  LDBG() << "Phase 2 complete. Removing dead operations from dead store "
            "elimination";
  opt.removeDeadOp();
  LDBG() << "=== transferOpflowOpt complete";
}

void mlir::vector::populateScalarVectorTransferLoweringPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit,
    bool allowMultipleUses) {
  patterns.add<RewriteScalarExtractOfTransferRead>(patterns.getContext(),
                                                   benefit, allowMultipleUses);
  patterns.add<RewriteScalarWrite>(patterns.getContext(), benefit);
}

void mlir::vector::populateVectorTransferDropUnitDimsPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns
      .add<TransferReadDropUnitDimsPattern, TransferWriteDropUnitDimsPattern>(
          patterns.getContext(), benefit);
}

void mlir::vector::populateFlattenVectorTransferPatterns(
    RewritePatternSet &patterns, unsigned targetVectorBitwidth,
    PatternBenefit benefit) {
  patterns.add<FlattenContiguousRowMajorTransferReadPattern,
               FlattenContiguousRowMajorTransferWritePattern>(
      patterns.getContext(), targetVectorBitwidth, benefit);
  populateDropUnitDimWithShapeCastPatterns(patterns, benefit);
}
