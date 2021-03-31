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

#include "PassDetail.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/EDSC/Intrinsics.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include <set>

#define DEBUG_TYPE "linalg-fusion"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::linalg;

using llvm::dbgs;

/// Implements a simple high-level fusion pass on linalg structured operations.
///
/// In each block, linalg ops are processed in reverse textual order.
/// Given a linalg op `O`, fusion occurs by:
///   1. inspecting the linalg ops that write into the views read by `O`. There
///      are 2 cases:
///      a) buffer case: use the SSA value of the views and a simple alias
///         analysis on subview ops to determine producer-consumer dependences;
///      b) tensor case: use SSA use-def chains on subtensor ops;
///   2. greedily fuse the linalg ops that produce the subview/subtensor.
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
  auto maps = op.indexing_maps();
  // Iterate over the inputs and outputs in order.
  // Extract the subranges from the linearized ranges.
  for (auto en : llvm::enumerate(op.getShapedOperands())) {
    // The method `getRangeFromOperandShape` requires using SubViewOp or
    // SubTensorOps. If the value isnt defined from there continue.
    // todo: The method should be adapted to get the values from
    // `ViewInterface`. The interface needs a `getOrCreateRanges` method which
    // currently returns a `linalg.range`. The fix here is to move this op to
    // `std` dialect and add the method to `ViewInterface`.
    if (fromSubViewOpOnly && !isa_and_nonnull<memref::SubViewOp, SubTensorOp>(
                                 en.value().getDefiningOp()))
      continue;

    unsigned idx = en.index();
    auto map = maps[idx].cast<AffineMapAttr>().getValue();
    LLVM_DEBUG(llvm::dbgs()
               << "getShapeDefiningLoopRange I/O idx: " << idx << "\n");
    LLVM_DEBUG(llvm::dbgs()
               << "getShapeDefiningLoopRange map: " << map << "\n");
    Value shape = en.value();
    SmallVector<Value, 8> shapeRanges(map.getNumResults(), nullptr);
    for (auto en2 : llvm::enumerate(map.getResults())) {
      auto dimExpr = en2.value().dyn_cast<AffineDimExpr>();
      if (!dimExpr)
        continue;
      if (loopDepth == en2.value().cast<AffineDimExpr>().getPosition()) {
        LLVM_DEBUG(llvm::dbgs() << "getShapeDefiningLoopRange loopDepth: "
                                << loopDepth << "\n");
        LLVM_DEBUG(llvm::dbgs()
                   << "getShapeDefiningLoopRange shape: " << shape << "\n");
        return ShapeDimension{shape, static_cast<unsigned>(en2.index())};
      }
    }
  }
  llvm_unreachable("Expect to be able to extract a shape defining loop range");
}

/// Fuses the producer by cloning the `producer`. The `fusedLoopsAndRanges`
/// provides the loop range information for the fused loops. The rest are
/// obtained from the producer itself, since they are not tiled + fused.
static LinalgOp fuse(OpBuilder &builder, LinalgOp producer,
                     const DenseMap<unsigned, Range> &fusedLoopsAndRanges) {
  SmallVector<Value, 8> ivs, tileSizes, sizeBounds;
  SmallVector<Range, 8> loopRanges;
  auto zero = std_constant_index(0);
  auto one = std_constant_index(1);
  Location loc = producer.getLoc();

  for (unsigned i = 0, e = producer.getNumLoops(); i < e; ++i) {
    auto it = fusedLoopsAndRanges.find(i);
    if (it != fusedLoopsAndRanges.end()) {
      ivs.push_back(it->second.offset);
      tileSizes.push_back(it->second.size);
      sizeBounds.push_back(nullptr);
      loopRanges.push_back(it->second);
      LLVM_DEBUG(llvm::dbgs() << "tiled loop#" << i << " with LoopRange "
                              << loopRanges.back() << "\n");
    } else {
      auto shapeDim = getShapeDefiningLoopRange(producer, i);
      Value dim = memref_dim(shapeDim.shape, shapeDim.dimension);
      tileSizes.push_back(zero);
      sizeBounds.push_back(dim);
      loopRanges.push_back(Range{zero, dim, one});
      LLVM_DEBUG(llvm::dbgs() << "full loop#" << i << " with LoopRange "
                              << loopRanges.back() << "\n");
    }
  }

  SmallVector<Value, 8> clonedShapes;
  clonedShapes.reserve(producer.getNumShapedOperands());

  // Compute subranges for all tensor input/output operands.
  auto tiledOperands = llvm::to_vector<4>(producer.getShapedOperands());
  clonedShapes.append(makeTiledShapes(builder, loc, producer, tiledOperands,
                                      ivs, tileSizes, sizeBounds));

  // Append the other operands.
  auto operands = producer.getAssumedNonShapedOperands();
  clonedShapes.append(operands.begin(), operands.end());

  // Iterate over the results in order.
  // Extract the subtensor type from the linearized range.
  // Since we do not enforce any canonicalizations on the fly, this is always
  // fully dynamic at construction time.
  SmallVector<Type, 4> resultTypes;
  resultTypes.reserve(producer->getNumResults());
  for (RankedTensorType t : producer.getOutputTensorTypes()) {
    unsigned rank = t.getRank();
    SmallVector<int64_t, 4> staticOffsetsVector(
        rank, ShapedType::kDynamicStrideOrOffset);
    SmallVector<int64_t, 4> staticSizesVector(rank, ShapedType::kDynamicSize);
    SmallVector<int64_t, 4> staticStridesVector(
        rank, ShapedType::kDynamicStrideOrOffset);
    resultTypes.push_back(SubTensorOp::inferResultType(
        t.cast<RankedTensorType>(), staticOffsetsVector, staticSizesVector,
        staticStridesVector));
  }

  Operation *clonedOp = producer.clone(builder, loc, resultTypes, clonedShapes);
  // When the producer is an IndexedGenericOp, we have to transform its block
  // IV arguments according to the tiling of the consumer, i.e. offset them by
  // the values computed in `loopRanges`.
  if (auto indexedGenericOp = dyn_cast<IndexedGenericOp>(clonedOp)) {
    auto &block = indexedGenericOp.region().front();
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(&block);
    for (unsigned i = 0, e = indexedGenericOp.getNumLoops(); i < e; ++i) {
      Value oldIndex = block.getArgument(i);
      // TODO: replace by an affine_apply.
      AddIOp newIndex = builder.create<AddIOp>(indexedGenericOp.getLoc(),
                                               oldIndex, loopRanges[i].offset);
      oldIndex.replaceAllUsesExcept(newIndex,
                                    SmallPtrSet<Operation *, 1>{newIndex});
    }
  }

  return clonedOp;
}

/// Get the loop range for a dimension `dim` based on the `shapedOperand`. It is
/// expected to be defined by a subview op or a subtensor op.
static Range getRangeFromOperandShape(OpBuilder &b, Location loc,
                                      Value shapedOperand, unsigned dim) {
  Operation *shapeProducingOp = shapedOperand.getDefiningOp();
  if (auto subViewOp = dyn_cast<memref::SubViewOp>(shapeProducingOp))
    return subViewOp.getOrCreateRanges(b, loc)[dim];
  if (auto subTensorOp = dyn_cast<SubTensorOp>(shapeProducingOp))
    return subTensorOp.getOrCreateRanges(b, loc)[dim];
  llvm_unreachable("SubviewOp or SubTensorOp expected");
}

/// Fuses the producer of `producerIdx` into the loop immediately enclosing
/// `consumer`. This is achieved by "recomputing" the `producer` at the time it
/// is needed just before the `consumer.
///
/// Depending on the type of `consumer.getShapedOperand(consumerIdx)`, there are
/// 2 cases:
///   1. Buffer case: `producerIdx` is the index of the buffer in
///      `producer.getOutputBuffers()`.
///   2. Tensor case: `producerIdx` is the index of the tensor in
///      `producer.getResults()`.
static LinalgOp fuse(OpBuilder &b, LinalgOp producerOp, AffineMap producerMap,
                     OpOperand &consumerOpOperand) {
  LLVM_DEBUG(llvm::dbgs() << "Producer map: " << producerMap << "\n");
  DenseMap<unsigned, Range> fusedLoopsAndRanges;
  Value shapedOperand = consumerOpOperand.get();
  for (auto en : llvm::enumerate(producerMap.getResults())) {
    unsigned posInProducerLoop = en.value().cast<AffineDimExpr>().getPosition();
    fusedLoopsAndRanges[posInProducerLoop] = getRangeFromOperandShape(
        b, consumerOpOperand.getOwner()->getLoc(), shapedOperand, en.index());
  }
  return fuse(b, producerOp, fusedLoopsAndRanges);
}

// Encode structural fusion safety preconditions.
// Some of these will be lifted in the future with better analysis.
static bool isStructurallyFusableProducer(LinalgOp producer, Value consumedView,
                                          LinalgOp consumer) {
  assert(producer.hasBufferSemantics() &&
         "expected linalg op with buffer semantics");
  assert(consumer.hasBufferSemantics() &&
         "expected linalg op with buffer semantics");
  if (producer.getNumOutputs() != 1) {
    LLVM_DEBUG(llvm::dbgs() << "\nNot structurally fusable (multi-output)");
    return false;
  }
  // Only fuse when the producer block dominates.
  DominanceInfo dom(producer.getOperation());
  if (!dom.dominates(producer->getBlock(), consumer->getBlock())) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "\nNot structurally fusable (producer block does not dominate)");
    return false;
  }
  return true;
}

bool mlir::linalg::isProducerLastWriteOfView(const LinalgDependenceGraph &graph,
                                             LinalgOp consumer,
                                             Value consumedView,
                                             LinalgOp producer) {
  assert(producer.hasBufferSemantics() &&
         "expected linalg op with buffer semantics");
  assert(consumer.hasBufferSemantics() &&
         "expected linalg op with buffer semantics");
  // Make some simple structural checks that alleviate the need for more
  // complex analyses.
  if (!isStructurallyFusableProducer(producer, consumedView, consumer)) {
    LLVM_DEBUG(llvm::dbgs() << "\n***Not static last write due to structure:\t"
                            << *producer.getOperation());
    return false;
  }
  // Check for any interleaved write to consumedView.
  if (!graph.findCoveringWrites(producer, consumer, consumedView).empty()) {
    LLVM_DEBUG(llvm::dbgs() << "\n***Not fusable due to interleaved write:\t"
                            << *producer.getOperation());
    return false;
  }
  return true;
}

bool mlir::linalg::isFusableInto(const LinalgDependenceGraph &graph,
                                 LinalgOp consumer, Value consumedView,
                                 LinalgOp producer) {
  assert(producer.hasBufferSemantics() &&
         "expected linalg op with buffer semantics");
  assert(consumer.hasBufferSemantics() &&
         "expected linalg op with buffer semantics");
  if (!isProducerLastWriteOfView(graph, consumer, consumedView, producer))
    return false;
  // Check for any fusion-preventing dependence to any shape read/written that
  // would violate dependences.
  if (!graph.findCoveringDependences(producer, consumer).empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "\n***Not fusable due to an interleaved dependence:\t"
               << *producer.getOperation());
    return false;
  }
  if (auto convOp = dyn_cast<linalg::ConvOp>(producer.getOperation())) {
    // TODO: add a level of indirection to linalg.generic.
    if (convOp.padding())
      return false;
  }
  if (auto convOp = dyn_cast<linalg::ConvOp>(consumer.getOperation())) {
    // TODO: add a level of indirection to linalg.generic.
    if (convOp.padding())
      return false;
  }
  return true;
}

/// For `consumer` with buffer semantics, find the Linalg operation on buffers
/// that is the last writer of `consumerOpOperand`. For now the fusable
/// dependence is returned as an instance of the `dependenceGraph`.
static Optional<LinalgDependenceGraph::LinalgDependenceGraphElem>
findFusableProducer(OpOperand &consumerOpOperand,
                    const LinalgDependenceGraph &dependenceGraph) {
  LinalgOp consumerOp = dyn_cast<LinalgOp>(consumerOpOperand.getOwner());
  if (!consumerOp)
    return {};

  // Only consider RAW and WAW atm.
  for (auto depType : {
           LinalgDependenceGraph::DependenceType::RAW,
           LinalgDependenceGraph::DependenceType::WAW,
       }) {
    for (auto dependence : llvm::make_filter_range(
             dependenceGraph.getDependencesInto(consumerOp, depType),
             [&](LinalgDependenceGraph::LinalgDependenceGraphElem elem) {
               Value v = elem.getIndexingValue();
               Optional<unsigned> operandNum =
                   elem.getIndexingOpViewOperandNum();
               return isa<LinalgOp>(elem.getDependentOp()) &&
                      v == consumerOpOperand.get() && operandNum &&
                      operandNum.getValue() ==
                          consumerOpOperand.getOperandNumber();
             })) {
      // Consumer consumes this view, `isStructurallyFusableProducer` also
      // checks whether it is a strict subview of the producer view.
      auto producer = cast<LinalgOp>(dependence.getDependentOp());
      LLVM_DEBUG(llvm::dbgs()
                 << "\n"
                 << LinalgDependenceGraph::getDependenceTypeStr(depType)
                 << "producer: " << *dependence.getDependentOp()
                 << " view: " << dependence.getDependentValue() << "\n");

      // If the producer and consumer have tensor semantics, the only dependence
      // between them is through a RAW dependence and they are fusable by
      // construction. For buffer semantics need additional checks.
      if (producer.hasBufferSemantics() && consumerOp.hasBufferSemantics() &&
          isFusableInto(dependenceGraph, consumerOp, consumerOpOperand.get(),
                        producer))
        return dependence;
      if (producer.hasTensorSemantics() && consumerOp.hasTensorSemantics()) {
        assert(dependence.dependenceType ==
               LinalgDependenceGraph::DependenceType::RAW);
        return dependence;
      }
    }
  }
  return {};
}

Optional<FusionInfo>
mlir::linalg::fuseProducerOfBuffer(OpBuilder &b, OpOperand &consumerOpOperand,
                                   const LinalgDependenceGraph &graph) {
  Optional<LinalgDependenceGraph::LinalgDependenceGraphElem> fusableDependence =
      findFusableProducer(consumerOpOperand, graph);
  if (!fusableDependence)
    return llvm::None;

  LinalgOp producerOp = dyn_cast<LinalgOp>(fusableDependence->getDependentOp());
  if (!producerOp)
    return llvm::None;

  // If producer is already in the same block as consumer, we are done.
  if (consumerOpOperand.get().getParentBlock() ==
      fusableDependence->getDependentValue().getParentBlock())
    return llvm::None;

  Optional<AffineMap> producerMap =
      fusableDependence->getDependentOpViewIndexingMap();
  if (!producerMap)
    return llvm::None;

  // Must be a subview or a slice to guarantee there are loops we can fuse
  // into.
  auto subView = consumerOpOperand.get().getDefiningOp<memref::SubViewOp>();
  if (!subView) {
    LLVM_DEBUG(llvm::dbgs() << "\nNot fusable (not a subview)");
    return llvm::None;
  }

  // Fuse `producer` just before `consumer`.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(consumerOpOperand.getOwner());
  ScopedContext scope(b, consumerOpOperand.getOwner()->getLoc());
  LLVM_DEBUG(llvm::dbgs() << "Fuse into consumer: "
                          << *consumerOpOperand.getOwner() << "\n");

  auto fusedProducer = fuse(b, producerOp, *producerMap, consumerOpOperand);
  return FusionInfo{producerOp, fusedProducer};
}

/// Walk back use-def chain through scf::For yields.
/// Sets `producer` and `outputIndex` if it finds a producer LinalgOp

// TODO(ravishankarm, ntv): This can be moved into the dependence graphs
// dependence tracking since the dependence tracking is similar to what is done
// w.r.t to buffers.
static void getProducerOfTensor(Value tensor, OpResult &opResult) {
  if (!tensor.getType().isa<RankedTensorType>())
    return;

  while (true) {
    LLVM_DEBUG(llvm::dbgs() << "\ngetProducerOfTensor: " << tensor);
    if (auto linalgOp = tensor.getDefiningOp<LinalgOp>()) {
      opResult = tensor.cast<OpResult>();
      return;
    }
    if (auto subTensorOp = tensor.getDefiningOp<SubTensorOp>()) {
      tensor = subTensorOp.source();
      continue;
    }
    if (auto blockArg = tensor.dyn_cast<BlockArgument>()) {
      if (auto forOp = blockArg.getDefiningOp<scf::ForOp>()) {
        tensor = *(forOp.getIterOperands().begin() + blockArg.getArgNumber());
        continue;
      }
    }
    return;
  }
}

Optional<FusionInfo>
mlir::linalg::fuseProducerOfTensor(OpBuilder &b, OpOperand &consumerOpOperand) {
  Value inputTensor = consumerOpOperand.get();
  OpResult producerOpResult;
  getProducerOfTensor(inputTensor, producerOpResult);
  if (!producerOpResult) {
    LLVM_DEBUG(llvm::dbgs() << "\nUnable to find producer");
    return {};
  }
  return fuseProducerOfTensor(b, producerOpResult, consumerOpOperand);
}

Optional<FusionInfo>
mlir::linalg::fuseProducerOfTensor(OpBuilder &b, OpResult producerOpResult,
                                   OpOperand &consumerOpOperand) {
  auto producerOp = dyn_cast<LinalgOp>(producerOpResult.getOwner());
  if (!producerOp)
    return llvm::None;

  LinalgOp consumerOp = dyn_cast<LinalgOp>(consumerOpOperand.getOwner());
  if (!consumerOp)
    return llvm::None;

  Value inputTensor = consumerOpOperand.get();

  // Must be a subtensor to guarantee there are loops we can fuse into.
  auto subTensor = inputTensor.getDefiningOp<SubTensorOp>();
  if (!subTensor) {
    LLVM_DEBUG(llvm::dbgs()
               << "\nNot fusable, not a subtensor: " << inputTensor);
    return {};
  }

  // If producer is already in the same block as consumer, we are done.
  if (consumerOpOperand.get().getParentBlock() ==
      producerOpResult.getParentBlock())
    return {};

  // Insert fused `producer` just before `consumer`.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(consumerOp);
  ScopedContext scope(b, consumerOp->getLoc());
  LLVM_DEBUG(llvm::dbgs() << "Fuse into consumer: " << *consumerOp << "\n");
  LinalgOp fusedProducer =
      fuse(b, producerOp,
           producerOp.getOutputIndexingMap(producerOpResult.getResultNumber()),
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

/// Prune all dimensions that are of reduction iterator type from `map`.
static AffineMap pruneReductionDimsFromMap(ArrayRef<Attribute> iteratorTypes,
                                           AffineMap map) {
  llvm::SmallDenseSet<unsigned> projectedDims;
  for (auto attr : llvm::enumerate(iteratorTypes)) {
    if (!isParallelIterator(attr.value()))
      projectedDims.insert(attr.index());
  }
  return getProjectedMap(map, projectedDims);
}

/// Returns the mapping from iterations in the consumer that write to the same
/// location as the iterations in the producer. To do so use
/// - indexing map of the fused view in the consumer : consumerIndexMap
/// - indexing map of the fused view in the producer : producerIndexMap
///     consumerLoopToProducerLoop =
///       inverse(producerIndexMap).compose(consumerIndexMap)
static Optional<AffineMap> getConsumerLoopToProducerLoopMap(
    LinalgDependenceGraph::LinalgDependenceGraphElem dependence) {
  auto producer = dyn_cast<LinalgOp>(dependence.getDependentOp());
  if (!producer)
    return None;

  Optional<AffineMap> producerIndexingMap =
      dependence.getDependentOpViewIndexingMap();
  Optional<AffineMap> consumerIndexingMap =
      dependence.getIndexingOpViewIndexingMap();
  if (!producerIndexingMap || !consumerIndexingMap)
    return None;

  AffineMap prunedProducerIndexingMap = pruneReductionDimsFromMap(
      producer.iterator_types().getValue(), *producerIndexingMap);
  if (!prunedProducerIndexingMap.isPermutation())
    return None;

  if (consumerIndexingMap->getNumResults() !=
      prunedProducerIndexingMap.getNumResults())
    return None;

  LLVM_DEBUG({
    llvm::dbgs() << "\t producerMap : ";
    producerIndexingMap->print(llvm::dbgs());
    llvm::dbgs() << "  pruned : ";
    prunedProducerIndexingMap.print(llvm::dbgs());
    llvm::dbgs() << "\n";
    llvm::dbgs() << "\t consumerMap : ";
    consumerIndexingMap->print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  AffineMap invProducerIndexMap = inversePermutation(prunedProducerIndexingMap);
  if (!invProducerIndexMap)
    return None;

  return invProducerIndexMap.compose(*consumerIndexingMap);
}

/// Given a projected permutation `map`, returns true if the map changes the
/// order in which the fused loop dimension appear.
static bool doesTransposeAccess(AffineMap map,
                                const std::set<unsigned> &fusableLoops) {
  Optional<unsigned> lastFusableLoop;
  for (unsigned pos : llvm::map_range(map.getResults(), [](AffineExpr expr) {
         return expr.cast<AffineDimExpr>().getPosition();
       })) {
    if (!fusableLoops.count(pos))
      continue;
    if (!lastFusableLoop) {
      lastFusableLoop = pos;
      continue;
    }
    if (pos <= lastFusableLoop.getValue())
      return true;
    lastFusableLoop = pos;
  }
  return false;
}

/// Returns the positions of the loop in `op` that can be tiled based on the
/// operations that are to be fused with it. For example, in a
///
///   linalg.matmul ins(%a, %b : ...) outs(%c : ...)
///
/// if the producer of %a needs to be fused with this op, only the `i` loop of
/// the matmul can be tiled while fusing. If producer of %a, and %b are to be
/// fused, then no loops can be tiled while fusing. The conditions used are:
/// 1. Only parallel loops can be used for tile + fuse. Find the number of
///    common outer parallel loops between the op and its producers being fused.
/// 2. Of the parallel loops only some can be fused. Only those loops can be
///    fused such where the fusable loops iteration space only touches one tile
///    of the fused operation. This is because the producer (which is writing
///    the fused subview) has update semantics.
///
/// Since an inverse computation is needed, we need to consider the projection
/// of the producerIndexMap w.r.t the parallel loops.  The actual fusable loops
/// are the dimensions of the consumerLoopToProducerLoop map that correspond to
/// parallel loops and appear in the result of the map
///
/// Example 1:
///   linalg.fill(%c, %cst)
///   linalg.matmul ins(%a, %b) outs(%c)
///     Number of parallel loops : 2
///     producerIndexMap = affine_map<(i, j) ->(i , j)>
///     consumerIndexMap = affine_map<(i, j, k) -> (i, j)>
///     consumerLoopToProducerLoop = affine_map<(i, j, k) -> (i, j)>
///     Fused dimensions : i, j
///
/// Example 2:
///   linalg.matmul ins(%a, %b) outs(%c)
///   linalg.generic {indexing_maps = [affine_map<(i, j) -> (j, i)>, ...
///                   iterator_types = ["parallel", "parallel"]}
///     ins(%c) ...
///
///     Number of parallel loops = 2:
///     producerIndexMap (projected to parallel loops) =
///       affine_map<(i, j) -> (i, j)>
///     consumerLoopToProducerLoop2 = affine_map<(i, j) -> (j, i)>
///     Fused dimensions : i, j
///
/// Example 3:
///   linalg.copy(%s, %b)
///   linalg.matmul ins(%a, %b) outs(%c)
///
///   Number of parallel loops = 2
///   produceIndexMap : affine_map<(i, j) -> (i, j)>
///   consumerLoopToProduceLoops = affine_map<(i, j, k) -> (k, j)>
///     submap with only parallel loops = affine_map<(i, j) -> (j)>
///   Fused dimensions : j
static std::set<unsigned>
collectFusableLoops(ArrayRef<LinalgOp> ops,
                    const FusableOpDependencesTy &fusableDependences) {
  assert(!ops.empty());
  auto getNumOuterParallelLoops = [](LinalgOp linalgOp) {
    return linalgOp.iterator_types()
        .getValue()
        .take_while([](Attribute attr) -> bool {
          return attr.cast<StringAttr>().getValue() ==
                 getParallelIteratorTypeName();
        })
        .size();
  };

  size_t numOuterParallelLoops = getNumOuterParallelLoops(ops.back());
  for (auto op : ops.drop_back()) {
    numOuterParallelLoops =
        std::min(numOuterParallelLoops, getNumOuterParallelLoops(op));
  }

  std::set<unsigned> fusableLoops;
  auto range = llvm::seq<unsigned>(0, numOuterParallelLoops);
  fusableLoops.insert(range.begin(), range.end());

  for (auto op : reverse(ops)) {
    for (auto dependence : fusableDependences.lookup(op)) {
      LLVM_DEBUG({
        llvm::dbgs() << "\t fusable :";
        for (unsigned i : fusableLoops)
          llvm::dbgs() << " " << i;
        llvm::dbgs() << "\n";
      });

      Optional<AffineMap> consumerLoopToProducerLoop =
          getConsumerLoopToProducerLoopMap(dependence);
      if (!consumerLoopToProducerLoop) {
        op.emitRemark("failed to get map from consumer loop to producer loop");
        return {};
      }
      // todo: This condition is only an implementation limitation. When fusing
      // the operation, if the accesses in the producer/consumer are transposes
      // of each other, the loop bounds for the tiled producer can be
      // manipulated accordingly. This requires some additional bookkeeping in
      // the implementation of tile+fuse that is deferred to later.
      if (doesTransposeAccess(*consumerLoopToProducerLoop, fusableLoops)) {
        op.emitRemark("unhandled fusion when fusion requires permutation");
        return {};
      }

      std::set<unsigned> candidates;
      for (AffineExpr expr : consumerLoopToProducerLoop->getResults()) {
        unsigned position = expr.cast<AffineDimExpr>().getPosition();
        if (fusableLoops.count(position))
          candidates.insert(position);
      }
      LLVM_DEBUG({
        llvm::dbgs() << "\t candidates :";
        for (unsigned i : candidates)
          llvm::dbgs() << " " << i;
        llvm::dbgs() << "\n";
      });
      if (candidates.empty())
        return {};
      std::swap(candidates, fusableLoops);
    }
  }

  return fusableLoops;
}

/// Find all dependences that are fusable.
FusableOpDependencesTy mlir::linalg::findAllFusableDependences(
    ArrayRef<LinalgOp> ops, const LinalgDependenceGraph &dependenceGraph) {
  FusableOpDependencesTy fusableDependences;
  DenseMap<Operation *, SmallVector<AffineMap, 1>> fusedProducerIndexingMap;
  for (LinalgOp op : reverse(ops)) {
    for (OpOperand &opOperand : op.getShapedOpOperands()) {
      Optional<LinalgDependenceGraph::LinalgDependenceGraphElem>
          fusableDependence = findFusableProducer(opOperand, dependenceGraph);
      if (!fusableDependence)
        continue;
      LinalgOp producerOp =
          dyn_cast<LinalgOp>(fusableDependence->getDependentOp());
      if (!producerOp)
        continue;
      // Do not fuse dependences that are to operations not in the same basic
      // block. This avoid moving fused operations across loops that might
      // themselves carry dependency making the fusion illegal.
      if (producerOp->getBlock() != op->getBlock())
        continue;

      // Make sure that the indexing map of the view used for fusion in the
      // producer is a projected permutation.
      Optional<AffineMap> producerMap =
          fusableDependence->getDependentOpViewIndexingMap();
      Optional<AffineMap> consumerMap =
          fusableDependence->getIndexingOpViewIndexingMap();
      assert(
          consumerMap &&
          "unable to find indexing map of operand/result of indexing OpView");
      fusedProducerIndexingMap[producerOp.getOperation()].push_back(
          *consumerMap);
      if (!producerMap || !producerMap->isProjectedPermutation() ||
          !consumerMap->isProjectedPermutation())
        continue;

      fusableDependences[producerOp.getOperation()].push_back(
          *fusableDependence);
    }
  }
  // TODO: Currently fusion would not be legal if the fusable dependence is to
  // the same producer but different indexing map in the consumer. Fix this, but
  // in the meanwhile disallow such a fusion.
  for (auto useIndexingMapsList : fusedProducerIndexingMap) {
    AffineMap map1 = useIndexingMapsList.second.front();
    for (AffineMap map2 :
         ArrayRef<AffineMap>(useIndexingMapsList.second).drop_front()) {
      if (map1 != map2) {
        fusableDependences.erase(useIndexingMapsList.first);
        break;
      }
    }
  }
  return fusableDependences;
}

/// Tile the fused loops in the root operation, by setting the tile sizes for
/// all other loops to zero (those will be tiled later).
static Optional<TiledLinalgOp> tileRootOperation(
    OpBuilder &builder, LinalgOp op, ArrayRef<Value> tileSizeVector,
    const LinalgTilingOptions &options, const std::set<unsigned> &fusedLoops) {
  SmallVector<Value, 4> tileSizes(tileSizeVector.begin(), tileSizeVector.end());
  auto zero = std_constant_index(0);
  for (unsigned i = 0, e = tileSizes.size(); i != e; ++i)
    if (!fusedLoops.count(i))
      tileSizes[i] = zero;
  LinalgTilingOptions tileFusedLoopsOptions = options;
  tileFusedLoopsOptions.setTileSizes(tileSizes);
  return tileLinalgOp(builder, op, tileFusedLoopsOptions);
}

/// Fuse the operations in `fusionCandidates` with `tiledOp`. Latter is expected
/// to be a tiled operation such that it is valid to fuse all operations in
/// `fusionCandidates`, i.e. move the operation within the inter-tile loops of
/// `tiledOp`.
static SmallVector<LinalgOp, 1>
fuseOperations(OpBuilder &builder, LinalgOp rootOp, LinalgOp tiledOp,
               ArrayRef<LinalgOp> fusionCandidates,
               const FusableOpDependencesTy &fusableDependences,
               const std::set<unsigned> &fusedLoops) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(tiledOp);
  DenseMap<unsigned, Range> fusedLoopsAndRanges;
  for (unsigned loop : fusedLoops) {
    ShapeDimension shapeDim = getShapeDefiningLoopRange(tiledOp, loop, true);
    fusedLoopsAndRanges[loop] = getRangeFromOperandShape(
        builder, tiledOp.getLoc(), shapeDim.shape, shapeDim.dimension);
  }

  SmallVector<LinalgOp, 1> fusedOps(fusionCandidates.size());
  DenseMap<Operation *, LinalgOp> origOpToFusedOp;
  origOpToFusedOp[rootOp.getOperation()] = tiledOp;
  for (auto candidate : enumerate(llvm::reverse(fusionCandidates))) {
    LinalgOp origOp = candidate.value();
    LinalgOp fusedOp = fuse(builder, origOp, fusedLoopsAndRanges);
    origOpToFusedOp[origOp.getOperation()] = fusedOp;
    fusedOps[fusionCandidates.size() - candidate.index() - 1] = fusedOp;
    // If the producer consumer operations are linalg operations on tensors, the
    // dependence is due to value produced (as a return tensor) by the producer
    // and used in the consumer. The returned value of the fused op needs to be
    // made the operand of the tiled/fused consumer operation. By construction
    // the value returned by the producer is the value used by the consumer.
    for (auto &dependence : fusableDependences.lookup(origOp.getOperation())) {
      if (origOp.hasTensorSemantics() &&
          dependence.dependenceType ==
              LinalgDependenceGraph::DependenceType::RAW) {
        unsigned resultIndex =
            dependence.getDependentOpViewResultNum().getValue();
        LinalgOp consumer = origOpToFusedOp.lookup(dependence.getIndexingOp());
        if (!consumer)
          continue;
        Value replacementValue = fusedOp.getOperation()->getResult(resultIndex);
        consumer.getOperation()->setOperand(
            dependence.getIndexingOpViewOperandNum().getValue(),
            replacementValue);
      }
    }
    builder.setInsertionPoint(fusedOp);
  }
  return fusedOps;
}

template <typename LoopType>
static Optional<TiledAndFusedLinalgOps>
tileAndFuseLinalgOpsImpl(OpBuilder &builder, ArrayRef<LinalgOp> ops,
                         const LinalgDependenceGraph &dependenceGraph,
                         const LinalgTilingOptions &tilingOptions) {
  if (ops.size() < 2)
    return llvm::None;
  LinalgOp rootOp = ops.back();
  if (!llvm::all_of(
          ops,
          [](LinalgOp linalgOp) { return linalgOp.hasBufferSemantics(); }) &&
      !llvm::all_of(ops, [](LinalgOp linalgOp) {
        return linalgOp.hasTensorSemantics();
      })) {
    rootOp.emitError(
        "unable to fuse operations that have tensor semantics with operations "
        "that have buffer semantics and viceversa.");
    return llvm::None;
  }
  // TODO: Support interchange with tile + fuse. This might actually help do
  // better fusion.
  if (!tilingOptions.interchangeVector.empty()) {
    rootOp.emitRemark("unable to handle tile and fuse with interchange");
    return llvm::None;
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(rootOp);
  ScopedContext scope(builder, rootOp.getLoc());

  // Find all the producers.
  FusableOpDependencesTy fusableDependences =
      findAllFusableDependences(ops, dependenceGraph);
  if (fusableDependences.empty())
    return llvm::None;

  TiledAndFusedLinalgOps ret;
  // Find the loops that can be tiled and fused.
  ret.fusedLoopDims = collectFusableLoops(ops, fusableDependences);

  // If there are no fusable dependences or there are no tile+fusable loops,
  // just return.
  if (ret.fusedLoopDims.empty()) {
    return llvm::None;
  }

  // Tile the fused loops in the last operation in the list.
  SmallVector<Value, 4> tileSizeVector =
      tilingOptions.tileSizeComputationFunction(builder, rootOp);
  Optional<TiledLinalgOp> tiledRootOp = tileRootOperation(
      builder, rootOp, tileSizeVector, tilingOptions, ret.fusedLoopDims);
  if (!tiledRootOp) {
    rootOp.emitRemark("failed to tile the fused loops");
    return llvm::None;
  }
  ret.op = tiledRootOp->op;
  ret.fusedLoops.assign(tiledRootOp->loops.begin(), tiledRootOp->loops.end());

  // Fuse the other operations into the fused inter-tile loops produced above.
  ret.fusedProducers = fuseOperations(builder, rootOp, ret.op, ops.drop_back(),
                                      fusableDependences, ret.fusedLoopDims);

  return ret;
}

Optional<TiledAndFusedLinalgOps>
mlir::linalg::tileAndFuseLinalgOps(OpBuilder &builder, ArrayRef<LinalgOp> ops,
                                   const LinalgDependenceGraph &dependenceGraph,
                                   const LinalgTilingOptions &tilingOptions) {
  switch (tilingOptions.loopType) {
  case LinalgTilingLoopType::Loops:
    return tileAndFuseLinalgOpsImpl<scf::ForOp>(builder, ops, dependenceGraph,
                                                tilingOptions);
  case LinalgTilingLoopType::ParallelLoops:
    return tileAndFuseLinalgOpsImpl<scf::ParallelOp>(
        builder, ops, dependenceGraph, tilingOptions);
  default:;
  }
  return llvm::None;
}
