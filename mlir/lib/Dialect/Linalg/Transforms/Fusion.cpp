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
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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

// Fill `offset`, `sizes` and `strides` used to iterate over the shape indexed
// by `permutationMap`.
static void inferShapeComponents(AffineMap permutationMap,
                                 ArrayRef<Range> loopRanges,
                                 SmallVectorImpl<Value> &offsets,
                                 SmallVectorImpl<Value> &sizes,
                                 SmallVectorImpl<Value> &strides) {
  assert(permutationMap.isProjectedPermutation() &&
         "expected some subset of a permutation map");
  SmallVector<Range, 4> shapeRanges(permutationMap.getNumResults());
  unsigned idx = 0;
  for (AffineExpr e : permutationMap.getResults()) {
    // loopToOperandRangesMaps are permutations-only, just swap indices.
    unsigned loopPos = e.cast<AffineDimExpr>().getPosition();
    shapeRanges[idx++] = loopRanges[loopPos];
  }
  // Construct a new subshape for the tile.
  unsigned rank = shapeRanges.size();
  offsets.reserve(rank);
  sizes.reserve(rank);
  strides.reserve(rank);
  for (auto r : shapeRanges) {
    offsets.push_back(r.offset);
    sizes.push_back(r.size);
    strides.push_back(r.stride);
  }
}

// Return a cloned version of `op` that operates on `loopRanges`, assumed to be
// a subset of the original loop ranges of `op`.
// This is achieved by applying the `loopToOperandRangesMaps` permutation maps
// to the `loopRanges` in order to obtain view ranges.
static LinalgOp cloneWithLoopRanges(OpBuilder &b, Location loc, LinalgOp op,
                                    ArrayRef<Range> loopRanges) {
  SmallVector<Value, 8> clonedShapes;
  clonedShapes.reserve(op.getNumShapedOperands());

  // Iterate over the shape operands in order.
  // Extract the subranges from the linearized ranges.
  for (auto en : llvm::enumerate(op.getShapedOperands())) {
    unsigned shapedOperandIdx = en.index();
    AffineMap map = op.getIndexingMap(shapedOperandIdx);
    LLVM_DEBUG(llvm::dbgs() << "shapedOperandIdx: " << shapedOperandIdx
                            << " with indexingMap: " << map << "\n");
    SmallVector<Value, 4> offsets, sizes, strides;
    inferShapeComponents(map, loopRanges, offsets, sizes, strides);
    Value shape = en.value();
    Value sub = shape.getType().isa<MemRefType>()
                    ? b.create<SubViewOp>(loc, shape, offsets, sizes, strides)
                          .getResult()
                    : b.create<SubTensorOp>(loc, shape, offsets, sizes, strides)
                          .getResult();
    clonedShapes.push_back(sub);
  }
  // Append the other operands.
  auto operands = op.getAssumedNonShapedOperands();
  clonedShapes.append(operands.begin(), operands.end());

  // Iterate over the results in order.
  // Extract the subtensor type from the linearized range.
  // Since we do not enforce any canonicalizations on the fly, this is always
  // fully dynamic at construction time.
  SmallVector<Type, 4> resultTypes;
  resultTypes.reserve(op.getOperation()->getNumResults());
  for (RankedTensorType t : op.getOutputTensorTypes()) {
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

  Operation *clonedOp = op.clone(b, loc, resultTypes, clonedShapes);
  // When the producer is an IndexedGenericOp, we have to transform its block
  // IV arguments according to the tiling of the consumer, i.e. offset them by
  // the values computed in `loopRanges`.
  if (auto indexedGenericOp = dyn_cast<IndexedGenericOp>(clonedOp)) {
    auto &block = indexedGenericOp.region().front();
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(&block);
    for (unsigned i = 0, e = indexedGenericOp.getNumLoops(); i < e; ++i) {
      Value oldIndex = block.getArgument(i);
      // TODO: replace by an affine_apply.
      AddIOp newIndex = b.create<AddIOp>(indexedGenericOp.getLoc(), oldIndex,
                                         loopRanges[i].offset);
      oldIndex.replaceAllUsesExcept(newIndex,
                                    SmallPtrSet<Operation *, 1>{newIndex});
    }
  }

  return clonedOp;
}

struct ShapeDimension {
  Value shape;
  unsigned dimension;
};

// Given an `op`, returns the first (`shape`, `dimension`) pair that identifies
// the loop range at `loopDepth`. The semantics of the loopToOperandRangesMaps
// guarantees at least one such dimension is found. If multiple candidates exist
// they must agree by construction (i.e. have the same size) and we just return
// the first one.
static ShapeDimension getShapeDefiningLoopRange(LinalgOp op,
                                                unsigned loopDepth) {
  auto maps = op.indexing_maps();
  // Iterate over the inputs and outputs in order.
  // Extract the subranges from the linearized ranges.
  SmallVector<Value, 8> ios(op.getInputsAndOutputBuffers());
  for (auto en : llvm::enumerate(ios)) {
    unsigned idx = en.index();
    auto map = maps[idx].cast<AffineMapAttr>().getValue();
    LLVM_DEBUG(llvm::dbgs()
               << "getShapeDefiningLoopRange I/O idx: " << idx << "\n");
    LLVM_DEBUG(llvm::dbgs()
               << "getShapeDefiningLoopRange map: " << map << "\n");
    Value shape = en.value();
    SmallVector<Value, 8> shapeRanges(map.getNumResults(), nullptr);
    for (auto en2 : llvm::enumerate(map.getResults())) {
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
static LinalgOp fuse(OpBuilder &b, LinalgOp producer, unsigned producerIdx,
                     LinalgOp consumer, unsigned consumerIdx) {
  Operation *shapeProducingOp =
      consumer.getShapedOperand(consumerIdx).getDefiningOp();
  assert((isa<SubViewOp>(shapeProducingOp) ||
          isa<SubTensorOp>(shapeProducingOp)) &&
         "SubviewOp or SubTensorOp expected");

  // loopToOperandRangesMaps are permutations-only by construction:
  //   we can always identify a data dimension with a (at least one) loop
  //   dimension.
  // TODO: extend this with range inference.
  AffineMap producerMap = producer.getOutputIndexingMap(producerIdx);
  LLVM_DEBUG(llvm::dbgs() << "Producer Idx: " << producerIdx
                          << ", producer map: " << producerMap << "\n");

  unsigned nPar = producer.getNumParallelLoops();
  unsigned nRed = producer.getNumReductionLoops();
  unsigned nWin = producer.getNumWindowLoops();
  SmallVector<Range, 8> loopRanges(nPar + nRed + nWin);

  // Iterate over dimensions identified by the producer map for `producerIdx`.
  // This defines a subset of the loop ranges that we need to complete later.
  auto loc = consumer.getLoc();
  for (auto en : llvm::enumerate(producerMap.getResults())) {
    unsigned posInProducerLoop = en.value().cast<AffineDimExpr>().getPosition();
    loopRanges[posInProducerLoop] =
        isa<SubViewOp>(shapeProducingOp)
            ? cast<SubViewOp>(shapeProducingOp)
                  .getOrCreateRanges(b, loc)[en.index()]
            : cast<SubTensorOp>(shapeProducingOp)
                  .getOrCreateRanges(b, loc)[en.index()];
  }

  // Iterate over all dimensions. For the dimensions not identified by the
  // producer map for `producerIdx`, we need to explicitly compute the shape
  // that defines the loop ranges using the `producer`.
  for (unsigned i = 0, nLoops = loopRanges.size(); i < nLoops; ++i) {
    if (loopRanges[i].offset)
      LLVM_DEBUG(llvm::dbgs()
                 << "existing LoopRange: " << loopRanges[i] << "\n");
    else {
      auto shapeDim = getShapeDefiningLoopRange(producer, i);
      loopRanges[i] = Range{std_constant_index(0),
                            std_dim(shapeDim.shape, shapeDim.dimension),
                            std_constant_index(1)};
      LLVM_DEBUG(llvm::dbgs() << "new LoopRange: " << loopRanges[i] << "\n");
    }
  }

  return cloneWithLoopRanges(b, loc, producer, loopRanges);
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
  if (!dom.dominates(producer.getOperation()->getBlock(),
                     consumer.getOperation()->getBlock())) {
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

static bool isSameSubView(Value a, Value b) {
  if (a == b)
    return true;
  auto sva = a.getDefiningOp<SubViewOp>();
  auto svb = b.getDefiningOp<SubViewOp>();
  if (!sva || !svb)
    return false;
  if (!isSameSubView(sva.getViewSource(), svb.getViewSource()))
    return false;
  if (sva.getType() != svb.getType())
    return false;
  if (sva.getNumOperands() != svb.getNumOperands())
    return false;
  if (sva.static_offsets() != svb.static_offsets())
    return false;
  if (sva.static_sizes() != svb.static_sizes())
    return false;
  if (sva.static_strides() != svb.static_strides())
    return false;
  /// Skip the "source" operand.
  for (unsigned idx = 1, e = sva.getNumOperands(); idx != e; ++idx)
    if (sva.getOperand(idx) != svb.getOperand(idx))
      return false;
  return true;
}

static Optional<LinalgDependenceGraph::LinalgDependenceGraphElem>
findFusableProducer(LinalgOp consumer, unsigned consumerIdx,
                    const LinalgDependenceGraph &dependenceGraph) {
  // Only consider RAW and WAW atm.
  for (auto depType : {
           LinalgDependenceGraph::DependenceType::RAW,
           LinalgDependenceGraph::DependenceType::WAW,
       }) {
    for (auto dependence : llvm::make_filter_range(
             dependenceGraph.getDependencesInto(consumer, depType),
             [consumerIdx](
                 LinalgDependenceGraph::LinalgDependenceGraphElem elem) {
               return elem.indexingOpView.operandIndex == consumerIdx;
             })) {
      auto producer = cast<LinalgOp>(dependence.dependentOpView.op);

      // Check that the dependence is indeed on the input `consumerIdx` view.
      auto consumedView =
          consumer.getBuffer(dependence.indexingOpView.operandIndex);
      if (!isSameSubView(consumer.getBuffer(consumerIdx), consumedView))
        continue;

      // Consumer consumes this view, `isStructurallyFusableProducer` also
      // checks whether it is a strict subview of the producer view.
      auto producedView =
          producer.getBuffer(dependence.dependentOpView.operandIndex);
      LLVM_DEBUG(llvm::dbgs()
                 << "\n"
                 << LinalgDependenceGraph::getDependenceTypeStr(depType)
                 << "producer: " << *producer.getOperation()
                 << " view: " << producedView << " output index: "
                 << dependence.dependentOpView.operandIndex -
                        producer.getNumInputs()
                 << "\n");
      (void)producedView;

      // Simple fusability checks.
      if (!isFusableInto(dependenceGraph, consumer, consumedView, producer))
        continue;

      return dependence;
    }
  }
  return {};
}

Optional<FusionInfo>
mlir::linalg::fuseProducerOfBuffer(OpBuilder &b, LinalgOp consumer,
                                   unsigned consumerIdx,
                                   const LinalgDependenceGraph &graph) {
  Optional<LinalgDependenceGraph::LinalgDependenceGraphElem> fusableDependence =
      findFusableProducer(consumer, consumerIdx, graph);
  if (!fusableDependence)
    return {};

  LinalgOp producerOp = cast<LinalgOp>(fusableDependence->dependentOpView.op);
  // If producer is already in the same block as consumer, we are done.
  if (consumer.getOperation()->getBlock() ==
      producerOp.getOperation()->getBlock())
    return {};

  unsigned producerIdx = fusableDependence->dependentOpView.operandIndex -
                         producerOp.getNumInputs();
  Value consumerView = consumer.getShapedOperand(consumerIdx);

  // Must be a subview or a slice to guarantee there are loops we can fuse
  // into.
  auto subView = consumerView.getDefiningOp<SubViewOp>();
  auto slice = consumerView.getDefiningOp<SliceOp>();
  if (!subView && !slice) {
    LLVM_DEBUG(llvm::dbgs() << "\nNot fusable (not a subview or slice)");
    return {};
  }

  // Fuse `producer` just before `consumer`.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(consumer.getOperation());
  ScopedContext scope(b, consumer.getLoc());
  LLVM_DEBUG(llvm::dbgs() << "Fuse into consumer: " << *consumer << "\n");

  auto fusedProducer = fuse(b, producerOp, producerIdx, consumer, consumerIdx);
  return FusionInfo{producerOp, fusedProducer};
}

/// Walk back use-def chain through scf::For yields.
/// Sets `producer` and `outputIndex` if it finds a producer LinalgOp
static void getProducerOfTensor(Value tensor, LinalgOp &producer,
                                unsigned &outputIndex) {
  if (!tensor.getType().isa<RankedTensorType>())
    return;

  while (true) {
    if (auto linalgOp = tensor.getDefiningOp<LinalgOp>()) {
      producer = linalgOp;
      outputIndex = tensor.cast<OpResult>().getResultNumber();
      return;
    }
    if (auto subTensorOp = tensor.getDefiningOp<SubTensorOp>()) {
      tensor = subTensorOp.source();
      continue;
    }
    if (auto blockArg = tensor.dyn_cast<BlockArgument>()) {
      if (auto forOp = blockArg.getDefiningOp<scf::ForOp>()) {
        tensor = forOp.getResult(blockArg.getArgNumber());
        continue;
      }
    }
    return;
  }
}

Optional<FusionInfo> mlir::linalg::fuseProducerOfTensor(OpBuilder &b,
                                                        LinalgOp consumer,
                                                        unsigned consumerIdx) {
  Value inputTensor = consumer.getInput(consumerIdx);
  LinalgOp producerOp;
  unsigned producerIdx;
  getProducerOfTensor(inputTensor, producerOp, producerIdx);

  // Must be a subtensor to guarantee there are loops we can fuse into.
  auto subTensor = inputTensor.getDefiningOp<SubTensorOp>();
  if (!subTensor || !producerOp) {
    LLVM_DEBUG(llvm::dbgs() << "\nNot fusable (not a subtensor)");
    return {};
  }

  // If producer is already in the same block as consumer, we are done.
  if (consumer.getOperation()->getBlock() ==
      producerOp.getOperation()->getBlock())
    return {};

  // Insert fused `producer` just before `consumer`.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(consumer.getOperation());
  ScopedContext scope(b, consumer.getLoc());
  LLVM_DEBUG(llvm::dbgs() << "Fuse into consumer: " << *consumer << "\n");
  LinalgOp fusedProducer =
      fuse(b, producerOp, producerIdx, consumer, consumerIdx);

  // Replace use.
  // Canonicalizations are not guaranteed to have happened before constructing
  // `fusedProducer`. In the tensor case this can result in temporary type
  // mismatches. Insert a `tensor_cast` op to propagate the transformation
  // invariant that types are compatible.
  Value def = fusedProducer.getOperation()->getResult(producerIdx);
  OpOperand &use = consumer.getOperation()->getOpOperand(consumerIdx);
  Type consumerType = use.get().getType();
  if (consumerType != def.getType())
    def = b.create<TensorCastOp>(fusedProducer.getLoc(), consumerType, def);
  use.set(def);
  return FusionInfo{producerOp, fusedProducer};
}

/// Prune all dimensions that are of reduction iterator type from `map`.
static AffineMap pruneReductionDimsFromMap(ArrayRef<Attribute> iteratorTypes,
                                           AffineMap map) {
  SmallVector<unsigned, 2> projectedDims;
  for (auto attr : llvm::enumerate(iteratorTypes)) {
    if (!isParallelIterator(attr.value()))
      projectedDims.push_back(attr.index());
  }
  return getProjectedMap(map, projectedDims);
}

using FusableOpDependencesTy = llvm::MapVector<
    Operation *,
    SmallVector<LinalgDependenceGraph::LinalgDependenceGraphElem, 1>>;

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
///    the fused subview) has update semantics. To compute this,
///    a. Find the mapping from iterations in the consumer that write to the
///       same location as the iterations in the producer. To do so use
///       - indexing map of the fused view in the consumer : consumerIndexMap
///       - indexing map of the fused view in the producer : producerIndexMap
///       consumerLoopToProducerLoop =
///         inverse(producerIndexMap).compose(consumerIndexMap)
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
collectTileAndFuseLoops(LinalgOp op,
                        const FusableOpDependencesTy &fusableDependences) {
  auto getNumOuterParallelLoops = [](LinalgOp linalgOp) {
    return linalgOp.iterator_types()
        .getValue()
        .take_while([](Attribute attr) -> bool {
          return attr.cast<StringAttr>().getValue() ==
                 getParallelIteratorTypeName();
        })
        .size();
  };

  LLVM_DEBUG({
    llvm::dbgs() << "Op : ";
    op.getOperation()->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n";
  });

  size_t numOuterParallelLoops = getNumOuterParallelLoops(op);
  for (auto dependence : fusableDependences) {
    linalg::LinalgOp producer = cast<linalg::LinalgOp>(dependence.first);
    numOuterParallelLoops =
        std::min(numOuterParallelLoops, getNumOuterParallelLoops(producer));
  }

  std::set<unsigned> fusableLoops;
  auto range = llvm::seq<unsigned>(0, numOuterParallelLoops);
  fusableLoops.insert(range.begin(), range.end());
  for (auto dependence : fusableDependences) {
    LLVM_DEBUG({
      llvm::dbgs() << "\t fusable :";
      for (unsigned i : fusableLoops)
        llvm::dbgs() << " " << i;
      llvm::dbgs() << "\n";
    });
    linalg::LinalgOp producer = cast<linalg::LinalgOp>(dependence.first);

    assert(!dependence.second.empty() &&
           "unexpected producer but not dependences");
    AffineMap producerIndexingMap = producer.getIndexingMap(
        dependence.second.front().dependentOpView.operandIndex);
    AffineMap prunedProducerIndexingMap = pruneReductionDimsFromMap(
        producer.iterator_types().getValue(), producerIndexingMap);
    if (!prunedProducerIndexingMap.isPermutation())
      return {};

    AffineMap consumerIndexingMap = op.getIndexingMap(
        dependence.second.front().indexingOpView.operandIndex);
    if (consumerIndexingMap.getNumResults() !=
        prunedProducerIndexingMap.getNumResults())
      return {};

    LLVM_DEBUG({
      llvm::dbgs() << "\t producerMap : ";
      producerIndexingMap.print(llvm::dbgs());
      llvm::dbgs() << "  pruned : ";
      prunedProducerIndexingMap.print(llvm::dbgs());
      llvm::dbgs() << "\n";
      llvm::dbgs() << "\t consumerMap : ";
      consumerIndexingMap.print(llvm::dbgs());
      llvm::dbgs() << "\n";
    });

    AffineMap invProducerIndexMap =
        inversePermutation(prunedProducerIndexingMap);
    if (!invProducerIndexMap)
      return {};

    AffineMap consumerLoopToProducerLoop =
        invProducerIndexMap.compose(consumerIndexingMap);

    LLVM_DEBUG({
      llvm::dbgs() << "\t consumerLoopToProducerLoop : ";
      consumerLoopToProducerLoop.print(llvm::dbgs());
    });

    std::set<unsigned> candidates;
    for (AffineExpr expr : consumerLoopToProducerLoop.getResults()) {
      AffineDimExpr dimExpr = expr.dyn_cast<AffineDimExpr>();
      if (!dimExpr)
        continue;
      unsigned position = dimExpr.getPosition();
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

  return fusableLoops;
}

/// Find all dependences that are to be fusable.
static FusableOpDependencesTy
findAllFusableDependences(LinalgOp op,
                          const LinalgDependenceGraph &dependenceGraph,
                          const LinalgFusionOptions &fusionOptions) {
  FusableOpDependencesTy fusableDependences;
  // TODO: Currently fusion would not be legal if the fusable dependence is to
  // the same producer but different indexing map in the consumer. Fix this, but
  // in the meanwhile disallow such a fusion.
  DenseMap<Operation *, AffineMap> fusedProducerIndexingMap;
  for (auto operandIndex : fusionOptions.indicesToFuse) {
    auto fusableDependence =
        findFusableProducer(op, operandIndex, dependenceGraph);
    if (!fusableDependence)
      return FusableOpDependencesTy{};
    LinalgOp producerOp = cast<LinalgOp>(fusableDependence->dependentOpView.op);
    // Do not fuse dependences that are to operations not in the same basic
    // block. This avoid moving fused operations across loops that might
    // themselves carry dependency making the fusion illegal.
    if (producerOp.getOperation()->getBlock() !=
        op.getOperation()->getBlock()) {
      op.emitRemark("unhandled fusion of ops in different basic blocks");
      return FusableOpDependencesTy{};
    }
    // Make sure that the indexing map of the view used for fusion in the
    // producer is a projected permutation.
    unsigned producerIdx = fusableDependence->dependentOpView.operandIndex;
    AffineMap producerMap = producerOp.getIndexingMap(producerIdx);
    if (!producerMap.isProjectedPermutation()) {
      op.emitRemark("unhandled non permutation indexing map for fused view in "
                    "producer for operand at index ")
          << operandIndex;
      return FusableOpDependencesTy{};
    }

    unsigned consumerIdx = fusableDependence->indexingOpView.operandIndex;
    AffineMap consumerMap = op.getIndexingMap(consumerIdx);
    if (!consumerMap.isProjectedPermutation()) {
      op.emitRemark(
          "unhandled case where indexing map for fused view in the consumer is "
          "not a projected permutation while fusing at index ")
          << operandIndex;
      return FusableOpDependencesTy{};
    }

    // Check if the producer is already a fusion candidate. Cannot fuse this
    // dependence if it has a different indexing map when used in the consumer.
    if (fusedProducerIndexingMap.count(producerOp.getOperation()) &&
        fusedProducerIndexingMap[producerOp.getOperation()] != consumerMap) {
      op.emitRemark("unhandled fusion to the same producer but with different "
                    "indexing maps");
      return FusableOpDependencesTy{};
    }
    fusedProducerIndexingMap[producerOp.getOperation()] = consumerMap;

    fusableDependences[producerOp.getOperation()].push_back(*fusableDependence);
  }
  return fusableDependences;
}

static bool isZero(Value v) {
  if (auto cst = v.getDefiningOp<ConstantIndexOp>())
    return cst.getValue() == 0;
  return false;
}

template <typename LoopType>
static Optional<TiledAndFusedLinalgOps>
tileAndFuseLinalgOpsImpl(PatternRewriter &rewriter, LinalgOp op,
                         const LinalgDependenceGraph &dependenceGraph,
                         const LinalgTilingOptions &tilingOptions,
                         const LinalgFusionOptions &fusionOptions) {
  assert(op.hasBufferSemantics() && "expected linalg op with buffer semantics");
  // Some of the tiling options might not be supportable with tile and fuse.
  // TODO: Support interchange with tile + fuse.
  if (!tilingOptions.interchangeVector.empty()) {
    op.emitError("unable to handle tile and fuse with interchange");
    return llvm::None;
  }

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);
  ScopedContext scope(rewriter, op.getLoc());

  // Find all the producers.
  FusableOpDependencesTy fusableDependences =
      findAllFusableDependences(op, dependenceGraph, fusionOptions);
  if (fusableDependences.empty())
    return llvm::None;

  // Enforce the convention that "tiling by zero" skips tiling a particular
  // dimension. This convention is significantly simpler to handle instead of
  // adjusting affine maps to account for missing dimensions.
  auto nLoops = op.getNumLoops();
  SmallVector<Value, 4> tileSizeVector =
      tilingOptions.tileSizeComputationFunction(rewriter, op);
  if (tileSizeVector.size() < nLoops) {
    auto zero = std_constant_index(0);
    tileSizeVector.append(nLoops - tileSizeVector.size(), zero);
  }

  TiledAndFusedLinalgOps ret;

  // Find the loops that can be tiled and fused.
  std::set<unsigned> tileFuseLoops =
      collectTileAndFuseLoops(op, fusableDependences);

  // If there are no fusable dependences or there are no tile+fusable loops,
  // just return.
  if (tileFuseLoops.empty()) {
    return llvm::None;
  }

  // Get the tile sizes for the first and second tiling steps. For the first
  // step the tile size are set to zero for the loops that arent
  // fused. Similarly for the second step, the tile sizes are set to zero for
  // the loops that are fused. For example, if for the following input
  //
  // ```
  //   linalg.add ins(%a, %b) outs(%c)
  //   linalg.matmul ins(%d, %c) outs(%e)
  // ```
  //
  // if the tile sizes of the `{i, j, k}` loops where given as `{ti, tj, tk}`
  // respectively, and since only `j` can be tiled and fused. The tile sizes
  // would be `{0, t_j, 0}` for the first tiling that tiles just the fusable
  // loops. The second tiling would be use tile sizes of `{t_i, 0, t_k}` to tile
  // the tiled matmul generated by the first tiling step.
  SmallVector<Value, 4> tileAndFuseSizes, tileSizes;
  for (auto tileSize : enumerate(tileSizeVector)) {
    auto zero = std_constant_index(0);
    if (tileFuseLoops.count(tileSize.index())) {
      tileAndFuseSizes.push_back(tileSize.value());
      tileSizes.push_back(zero);
    } else {
      tileSizes.push_back(tileSize.value());
      tileAndFuseSizes.push_back(zero);
    }
  }

  // Tile for the loops that can be fused.
  LinalgTilingOptions firstTilingOptions = tilingOptions;
  firstTilingOptions.setTileSizes(tileAndFuseSizes);
  Optional<TiledLinalgOp> firstTiledOp =
      tileLinalgOp(rewriter, op, firstTilingOptions);
  if (!firstTiledOp)
    return llvm::None;
  ret.op = firstTiledOp->op;
  ret.fusedLoops.assign(firstTiledOp->loops.begin(), firstTiledOp->loops.end());

  rewriter.setInsertionPoint(ret.op);
  // Fuse the operands.
  for (auto dependence : fusableDependences) {
    LinalgOp producerOp = cast<LinalgOp>(dependence.first);
    unsigned producerIdx =
        dependence.second.front().dependentOpView.operandIndex;
    unsigned consumerIdx =
        dependence.second.front().indexingOpView.operandIndex;
    LinalgOp fusedOp = fuse(rewriter, producerOp,
                            producerOp.getOutputIndex(producerIdx).getValue(),
                            ret.op, consumerIdx);
    ret.fusedProducers.push_back(fusedOp);
    ret.originalProducers.push_back(producerOp);
  }

  if (!llvm::all_of(tileSizes, isZero)) {
    // Tile the remaining loops of the root operation.
    LinalgTilingOptions secondTilingOptions = tilingOptions;
    // The distribution is done only for the tile+fused loops.
    secondTilingOptions.distribution = llvm::None;
    secondTilingOptions.setTileSizes(tileSizes);
    Optional<TiledLinalgOp> secondTiledOp =
        tileLinalgOp(rewriter, ret.op, secondTilingOptions);
    if (!secondTiledOp)
      return llvm::None;
    ret.unfusedLoops.assign(secondTiledOp->loops.begin(),
                            secondTiledOp->loops.end());
    rewriter.eraseOp(ret.op);
    ret.op = secondTiledOp->op;
  }

  return ret;
}

Optional<TiledAndFusedLinalgOps>
mlir::linalg::tileAndFuseLinalgOps(PatternRewriter &rewriter, LinalgOp op,
                                   const LinalgDependenceGraph &dependenceGraph,
                                   const LinalgTilingOptions &tilingOptions,
                                   const LinalgFusionOptions &fusionOptions) {
  switch (tilingOptions.loopType) {
  case LinalgTilingLoopType::Loops:
    return tileAndFuseLinalgOpsImpl<scf::ForOp>(rewriter, op, dependenceGraph,
                                                tilingOptions, fusionOptions);
  case LinalgTilingLoopType::ParallelLoops:
    return tileAndFuseLinalgOpsImpl<scf::ParallelOp>(
        rewriter, op, dependenceGraph, tilingOptions, fusionOptions);
  default:;
  }
  return llvm::None;
}
