//===- Fusion.cpp - Implementation of linalg Fusion -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the linalg dialect Fusion pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include <set>
#include <optional>

#define DEBUG_TYPE "linalg-fusion"

using namespace mlir;
using namespace mlir::linalg;

/// Implements a simple high-level fusion pass on linalg structured operations.
///
/// In each block, linalg ops are processed in reverse textual order.
/// Given a linalg op `O`, fusion occurs by:
///   1. inspecting the linalg ops that write into the views read by `O`. There
///      are 2 cases:
///      a) buffer case: use the SSA value of the views and a simple alias
///         analysis on subview ops to determine producer-consumer dependences;
///      b) tensor case: use SSA use-def chains on extract_slice ops;
///   2. greedily fuse the linalg ops that produce the subview/extract_slice.
///   3. inspect the fused ops and determine whether they have other remaining
///      LinalgOp uses. If not, then erase the original producing linalg op.
///
/// More advanced use cases, analyses as well as profitability heuristics are
/// left for future work.

struct ShapeDimension {
  Value shape;
  unsigned dimension;
};

// Given an `op`, returns the first (`shape`, `dimension`) pair that identifies
// the loop range at `loopDepth`. The semantics of the loopToOperandRangesMaps
// guarantees at least one such dimension is found. If multiple candidates exist
// they must agree by construction (i.e. have the same size) and we just return
// the first one.
static ShapeDimension
getShapeDefiningLoopRange(LinalgOp op, unsigned loopDepth,
                          bool fromSubViewOpOnly = false) {
  // Iterate over the inputs and outputs in order.
  // Extract the subranges from the linearized ranges.
  for (OpOperand &opOperand : op->getOpOperands()) {
    // The method `getRangeFromOperandShape` requires using SubViewOp or
    // ExtractSliceOps. If the value isn't defined from there continue.
    // todo: The method should be adapted to get the values from
    // `ViewInterface`. The interface needs a `getOrCreateRanges` method which
    // currently returns a `linalg.range`. The fix here is to move this op to
    // `std` dialect and add the method to `ViewInterface`.
    if (fromSubViewOpOnly &&
        !isa_and_nonnull<memref::SubViewOp, tensor::ExtractSliceOp>(
            opOperand.get().getDefiningOp()))
      continue;

    AffineMap map = op.getMatchingIndexingMap(&opOperand);
    LLVM_DEBUG(llvm::dbgs() << "getShapeDefiningLoopRange I/O idx: "
                            << opOperand.getOperandNumber() << "\n");
    LLVM_DEBUG(llvm::dbgs()
               << "getShapeDefiningLoopRange map: " << map << "\n");
    SmallVector<Value, 8> shapeRanges(map.getNumResults(), nullptr);
    for (const auto &en : llvm::enumerate(map.getResults())) {
      auto dimExpr = en.value().dyn_cast<AffineDimExpr>();
      if (!dimExpr)
        continue;
      if (loopDepth == en.value().cast<AffineDimExpr>().getPosition()) {
        LLVM_DEBUG(llvm::dbgs() << "getShapeDefiningLoopRange loopDepth: "
                                << loopDepth << "\n");
        LLVM_DEBUG(llvm::dbgs() << "getShapeDefiningLoopRange shape: "
                                << opOperand.get() << "\n");
        return ShapeDimension{opOperand.get(),
                              static_cast<unsigned>(en.index())};
      }
    }
  }
  llvm_unreachable("Expect to be able to extract a shape defining loop range");
}

static SmallVector<Value> getTiledOperands(LinalgOp producer) {
  return producer->getOperands();
}

/// Fuses the producer by cloning the `producer`. The `fusedLoopsAndRanges`
/// provides the loop range information for the fused loops. The rest are
/// obtained from the producer itself, since they are not tiled + fused.
static LinalgOp fuse(OpBuilder &b, LinalgOp producer,
                     const DenseMap<unsigned, Range> &fusedLoopsAndRanges) {
  SmallVector<OpFoldResult> ivs, tileSizes, sizeBounds;
  SmallVector<Range> loopRanges;
  Location loc = producer.getLoc();

  for (unsigned i = 0, e = producer.getNumLoops(); i < e; ++i) {
    auto shapeDim = getShapeDefiningLoopRange(producer, i);
    OpFoldResult dim =
        createFoldedDimOp(b, loc, shapeDim.shape, shapeDim.dimension);
    sizeBounds.push_back(dim);
    auto it = fusedLoopsAndRanges.find(i);
    if (it != fusedLoopsAndRanges.end()) {
      ivs.push_back(it->second.offset);
      tileSizes.push_back(it->second.size);
      loopRanges.push_back(it->second);
      LLVM_DEBUG(llvm::dbgs() << "tiled loop#" << i << " with LoopRange "
                              << loopRanges.back() << "\n");
    } else {
      tileSizes.push_back(b.getIndexAttr(0));
      loopRanges.push_back(Range{b.getIndexAttr(0), dim, b.getIndexAttr(1)});
      LLVM_DEBUG(llvm::dbgs() << "full loop#" << i << " with LoopRange "
                              << loopRanges.back() << "\n");
    }
  }

  SmallVector<Value, 8> clonedShapes;
  clonedShapes.reserve(producer->getNumOperands());

  // Compute subranges for all tensor input/output operands.
  clonedShapes.append(makeTiledShapes(
      b, loc, producer, getTiledOperands(producer), ivs, tileSizes, sizeBounds,
      /**omitPartialTileCheck=*/false));

  // Iterate over the results in order.
  // Extract the subtensor type from the linearized range.
  // Since we do not enforce any canonicalizations on the fly, this is always
  // fully dynamic at construction time.
  SmallVector<Type, 4> resultTypes;
  resultTypes.reserve(producer->getNumResults());
  for (OpOperand *operand : producer.getDpsInitOperands()) {
    auto tensorType = dyn_cast<RankedTensorType>(operand->get().getType());
    if (!tensorType)
      continue;
    unsigned rank = tensorType.getRank();
    SmallVector<int64_t, 4> staticOffsetsVector(
        rank, ShapedType::kDynamic);
    SmallVector<int64_t, 4> staticSizesVector(rank, ShapedType::kDynamic);
    SmallVector<int64_t, 4> staticStridesVector(
        rank, ShapedType::kDynamic);
    resultTypes.push_back(tensor::ExtractSliceOp::inferResultType(
        tensorType, staticOffsetsVector, staticSizesVector,
        staticStridesVector));
  }

  LinalgOp clonedOp = clone(b, producer, resultTypes, clonedShapes);

  // Shift all IndexOp results by the tile offset.
  SmallVector<OpFoldResult> allIvs = llvm::to_vector(
      llvm::map_range(loopRanges, [&](Range range) { return range.offset; }));
  offsetIndices(b, clonedOp, allIvs);

  return clonedOp;
}

/// Get the loop range for a dimension `dim` based on the `shapedOperand`. It is
/// expected to be defined by a subview op or an extract_slice op.
static Range getRangeFromOperandShape(OpBuilder &b, Location loc,
                                      Value shapedOperand, unsigned dim) {
  Operation *shapeProducingOp = shapedOperand.getDefiningOp();
  if (auto subViewOp = dyn_cast<memref::SubViewOp>(shapeProducingOp))
    return subViewOp.getOrCreateRanges(b, loc)[dim];
  if (auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(shapeProducingOp))
    return sliceOp.getOrCreateRanges(b, loc)[dim];
  llvm_unreachable("SubviewOp or ExtractSliceOp expected");
}

/// Fuses the producer into the loop immediately enclosing the consumer.
/// This is achieved by "recomputing" the producer at the time it
/// is needed just before the consumer.
static LinalgOp fuse(OpBuilder &b, LinalgOp producerOp, AffineMap producerMap,
                     OpOperand &consumerOpOperand) {
  LLVM_DEBUG(llvm::dbgs() << "Producer map: " << producerMap << "\n");
  DenseMap<unsigned, Range> fusedLoopsAndRanges;
  Value shapedOperand = consumerOpOperand.get();
  for (const auto &en : llvm::enumerate(producerMap.getResults())) {
    unsigned posInProducerLoop = en.value().cast<AffineDimExpr>().getPosition();
    fusedLoopsAndRanges[posInProducerLoop] = getRangeFromOperandShape(
        b, consumerOpOperand.getOwner()->getLoc(), shapedOperand, en.index());
  }
  return fuse(b, producerOp, fusedLoopsAndRanges);
}

/// Walk back use-def chain through scf::For yields.
/// Sets `producer` and `outputIndex` if it finds a producer LinalgOp

// TODO(ravishankarm, ntv): This can be moved into the dependence graphs
// dependence tracking since the dependence tracking is similar to what is done
// w.r.t to buffers.
static void getProducerOfTensor(Value tensor, OpResult &opResult) {
  if (!isa<RankedTensorType>(tensor.getType()))
    return;

  while (true) {
    LLVM_DEBUG(llvm::dbgs() << "\ngetProducerOfTensor: " << tensor);
    if (auto linalgOp = tensor.getDefiningOp<LinalgOp>()) {
      opResult = cast<OpResult>(tensor);
      return;
    }
    if (auto sliceOp = tensor.getDefiningOp<tensor::ExtractSliceOp>()) {
      tensor = sliceOp.getSource();
      continue;
    }
    if (auto blockArg = dyn_cast<BlockArgument>(tensor)) {
      if (auto forOp = blockArg.getDefiningOp<scf::ForOp>()) {
        tensor = *(forOp.getIterOperands().begin() + blockArg.getArgNumber());
        continue;
      }
    }
    return;
  }
}

FailureOr<FusionInfo>
mlir::linalg::fuseProducerOfTensor(OpBuilder &b, OpOperand &consumerOpOperand) {
  Value inputTensor = consumerOpOperand.get();
  OpResult producerOpResult;
  getProducerOfTensor(inputTensor, producerOpResult);
  if (!producerOpResult) {
    LLVM_DEBUG(llvm::dbgs() << "\nUnable to find producer");
    return failure();
  }
  return fuseProducerOfTensor(b, producerOpResult, consumerOpOperand);
}

FailureOr<FusionInfo>
mlir::linalg::fuseProducerOfTensor(OpBuilder &b, OpResult producerOpResult,
                                   OpOperand &consumerOpOperand) {
  auto producerOp = dyn_cast<LinalgOp>(producerOpResult.getOwner());
  if (!producerOp)
    return failure();

  LinalgOp consumerOp = dyn_cast<LinalgOp>(consumerOpOperand.getOwner());
  if (!consumerOp)
    return failure();

  Value inputTensor = consumerOpOperand.get();

  // Must be an extract_slice op to guarantee there are loops we can fuse into.
  auto sliceOp = inputTensor.getDefiningOp<tensor::ExtractSliceOp>();
  if (!sliceOp) {
    LLVM_DEBUG(llvm::dbgs()
               << "\nNot fusable, not an extract_slice op: " << inputTensor);
    return failure();
  }

  // If producer is already in the same block as consumer, we are done.
  if (consumerOpOperand.get().getParentBlock() ==
      producerOpResult.getParentBlock())
    return failure();

  // Insert fused `producer` just before `consumer`.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(consumerOp);
  LLVM_DEBUG(llvm::dbgs() << "Fuse into consumer: " << *consumerOp << "\n");
  OpOperand *opOperand =
      producerOp.getDpsInitOperand(producerOpResult.getResultNumber());
  LinalgOp fusedProducer =
      fuse(b, producerOp, producerOp.getMatchingIndexingMap(opOperand),
           consumerOpOperand);

  // Replace use.
  // Canonicalizations are not guaranteed to have happened before constructing
  // `fusedProducer`. In the tensor case this can result in temporary type
  // mismatches. Insert a `tensor.cast` op to propagate the transformation
  // invariant that types are compatible.
  Value def = fusedProducer->getResult(producerOpResult.getResultNumber());
  Type consumerType = consumerOpOperand.get().getType();
  if (consumerType != def.getType())
    def = b.create<tensor::CastOp>(fusedProducer.getLoc(), consumerType, def);
  consumerOpOperand.set(def);
  return FusionInfo{cast<LinalgOp>(producerOpResult.getOwner()), fusedProducer};
}
