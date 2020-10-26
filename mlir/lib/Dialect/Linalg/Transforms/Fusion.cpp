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
#include "mlir/Dialect/Linalg/EDSC/FoldedIntrinsics.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/FoldUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-fusion"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::linalg;

using folded_std_constant_index = FoldedValueBuilder<ConstantIndexOp>;

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
    LLVM_DEBUG(dbgs() << "shapedOperandIdx: " << shapedOperandIdx
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
    LLVM_DEBUG(dbgs() << "getShapeDefiningLoopRange I/O idx: " << idx << "\n");
    LLVM_DEBUG(dbgs() << "getShapeDefiningLoopRange map: " << map << "\n");
    Value shape = en.value();
    SmallVector<Value, 8> shapeRanges(map.getNumResults(), nullptr);
    for (auto en2 : llvm::enumerate(map.getResults())) {
      if (loopDepth == en2.value().cast<AffineDimExpr>().getPosition()) {
        LLVM_DEBUG(dbgs() << "getShapeDefiningLoopRange loopDepth: "
                          << loopDepth << "\n");
        LLVM_DEBUG(dbgs() << "getShapeDefiningLoopRange shape: " << shape
                          << "\n");
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
                     LinalgOp consumer, unsigned consumerIdx,
                     OperationFolder *folder = nullptr) {
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
  LLVM_DEBUG(dbgs() << "Producer Idx: " << producerIdx
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
      loopRanges[i] = Range{folded_std_constant_index(folder, 0),
                            std_dim(shapeDim.shape, shapeDim.dimension),
                            folded_std_constant_index(folder, 1)};
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
    LLVM_DEBUG(dbgs() << "\nNot structurally fusable (multi-output)");
    return false;
  }
  // Only fuse when the producer block dominates.
  DominanceInfo dom(producer.getOperation());
  if (!dom.dominates(producer.getOperation()->getBlock(),
                     consumer.getOperation()->getBlock())) {
    LLVM_DEBUG(
        dbgs()
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
    LLVM_DEBUG(dbgs() << "\n***Not static last write due to structure:\t"
                      << *producer.getOperation());
    return false;
  }
  // Check for any interleaved write to consumedView.
  if (!graph.findCoveringWrites(producer, consumer, consumedView).empty()) {
    LLVM_DEBUG(dbgs() << "\n***Not fusable due to interleaved write:\t"
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
    LLVM_DEBUG(dbgs() << "\n***Not fusable due to an interleaved dependence:\t"
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
    for (auto dependence :
         dependenceGraph.getDependencesInto(consumer, depType)) {
      auto producer = cast<LinalgOp>(dependence.dependentOpView.op);

      // Check that the dependence is indeed on the input `consumerIdx` view.
      auto consumedView = dependence.indexingView;
      if (!isSameSubView(consumer.getBuffer(consumerIdx), consumedView))
        continue;

      // Consumer consumes this view, `isStructurallyFusableProducer` also
      // checks whether it is a strict subview of the producer view.
      auto producedView = dependence.dependentOpView.view;
      auto producerIdx =
          producer.getIndexOfOutputBuffer(producedView).getValue();
      // `consumerIdx` and `producerIdx` exist by construction.
      LLVM_DEBUG(dbgs() << "\n"
                        << LinalgDependenceGraph::getDependenceTypeStr(depType)
                        << "producer: " << *producer.getOperation() << " view: "
                        << producedView << " output index: " << producerIdx);
      (void)producerIdx;

      // Simple fusability checks.
      if (!isFusableInto(dependenceGraph, consumer, consumedView, producer))
        continue;

      return dependence;
    }
  }
  return {};
}

Optional<FusionInfo> mlir::linalg::fuseProducerOfBuffer(
    OpBuilder &b, LinalgOp consumer, unsigned consumerIdx,
    const LinalgDependenceGraph &graph, OperationFolder *folder) {
  Optional<LinalgDependenceGraph::LinalgDependenceGraphElem> fusableDependence =
      findFusableProducer(consumer, consumerIdx, graph);
  if (!fusableDependence)
    return {};

  LinalgOp producerOp = cast<LinalgOp>(fusableDependence->dependentOpView.op);
  Value producerView = fusableDependence->dependentOpView.view;
  Value consumerView = fusableDependence->indexingView;

  // Must be a subview or a slice to guarantee there are loops we can fuse
  // into.
  auto subView = consumerView.getDefiningOp<SubViewOp>();
  auto slice = consumerView.getDefiningOp<SliceOp>();
  if (!subView && !slice) {
    LLVM_DEBUG(dbgs() << "\nNot fusable (not a subview or slice)");
    return {};
  }

  // Fuse `producer` just before `consumer`.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(consumer.getOperation());
  ScopedContext scope(b, consumer.getLoc());
  LLVM_DEBUG(dbgs() << "Fuse into consumer: " << *consumer << "\n");
  Optional<unsigned> producerIdxOpt =
      producerOp.getIndexOfOutputBuffer(producerView);
  assert(producerIdxOpt.hasValue() && "incorrect operand index");
  unsigned producerIdx = producerIdxOpt.getValue();

  auto fusedProducer =
      fuse(b, producerOp, producerIdx, consumer, consumerIdx, folder);
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

Optional<FusionInfo>
mlir::linalg::fuseProducerOfTensor(OpBuilder &b, LinalgOp consumer,
                                   unsigned consumerIdx,
                                   OperationFolder *folder) {
  Value inputTensor = consumer.getInput(consumerIdx);
  LinalgOp producerOp;
  unsigned producerIdx;
  getProducerOfTensor(inputTensor, producerOp, producerIdx);

  // Must be a subtensor to guarantee there are loops we can fuse into.
  auto subTensor = inputTensor.getDefiningOp<SubTensorOp>();
  if (!subTensor || !producerOp) {
    LLVM_DEBUG(dbgs() << "\nNot fusable (not a subtensor)");
    return {};
  }

  // Insert fused `producer` just before `consumer`.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(consumer.getOperation());
  ScopedContext scope(b, consumer.getLoc());
  LLVM_DEBUG(dbgs() << "Fuse into consumer: " << *consumer << "\n");
  LinalgOp fusedProducer =
      fuse(b, producerOp, producerIdx, consumer, consumerIdx, folder);

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

/// Returns the positions of the loop in `op` that can be tiled based on the
/// operations that are to be fused with it. For example, in a
///
///   linalg.matmul ins(%a, %b : ...) outs(%c : ...)
///
/// if the producer of %a needs to be fused with this op, only the `i` loop of
/// the matmul can be tiled while fusing. If producer of %a, and %b are to be
/// fused, then no loops can be tiled while fusing.
static DenseSet<unsigned> collectTileAndFuseLoops(
    LinalgOp op, ArrayRef<LinalgDependenceGraph::LinalgDependenceGraphElem>
                     fusableDependences) {
  // 1. Only parallel loops can be used for tile + fuse. Find the number of
  // common outer parallel loops between the op and its producers being fused.
  auto getNumOuterParallelLoops = [](LinalgOp linalgOp) {
    return linalgOp.iterator_types()
        .getValue()
        .take_while([](Attribute attr) -> bool {
          return attr.cast<StringAttr>().getValue() ==
                 getParallelIteratorTypeName();
        })
        .size();
  };

  size_t numOuterParallelLoops = getNumOuterParallelLoops(op);
  for (auto dependence : fusableDependences) {
    numOuterParallelLoops =
        std::min(numOuterParallelLoops, getNumOuterParallelLoops(cast<LinalgOp>(
                                            dependence.dependentOpView.op)));
  }

  // Need to compute what tiled loops can be "fused". Given the precondition
  // that all indexing map for the producer view is a projected permutation, we
  // can assert that the producer iterates over the dimensions of the "fused
  // view" only once. To be used a fused loop the producer should use this loop
  // to access the fused view. For example, consider
  //
  // ```
  //   linalg.add ins(%a, %b) outs(%c)
  //   linalg.matmul ins(%d, %c) outs(%e)
  // ```
  //
  // if `linalg.add` has the semantics of `c = a + b`, then the following
  // tile+fuse code is correct.
  //
  // ```
  // for j ... += TSj
  //   %sa = subview %a[0, %j][...]
  //   %sb = subview %b[0, %j][...]
  //   %sc = subview %c[0, %j][...]
  //   %sd = subview %d[0, 0][...]
  //   %se = subview %e[0, %j][...]
  //   linalg.add ins(%sa, %sb) outs(%sc)
  //   linalg.matmul ins(%sd, %sc) outs(%se)
  // ```
  //
  // On the other hand tiling along i would be incorrect
  //
  // ```
  // for %i .. += TSi
  //   %sa = subview %a[%i, 0][...]
  //   %sb = subview %b[%i, 0][...]
  //   %sc = subview %c[%i, 0][...]
  //   %sc2 = subview %c[0, 0][...]
  //   %sd = subview %d[%i, 0][...]
  //   %se = subview %e[%i, 0][...]
  //   linalg.add ins(%sa, %sb) outs(%sc)
  //   linalg.matmul ins(%sd, %sc2) outs(%se)
  // ```
  //
  // The write to the subview `%sc` in `linalg.add` is performed after the read
  // from it using `%sc2` violating the RAW dependence of the original code. To
  // find such loops indexing map of the fused view in the consumer op is
  // used. For the above example, this indexing map is
  //
  //   affine_map<(d0, d1, d2) -> (d2, d1)>
  //
  // Since d0 is not in the result expressions of this map, it is not treated as
  // tile + fuse loop, (but d1 is).
  //
  // TODO: The above is probably restrictive and there might be a generalization
  // of these that might allow for more fusion opportunities. Explore based on
  // needs.
  SmallVector<DenseSet<unsigned>, 1> commonTilableLoops;
  for (auto dependence : fusableDependences) {
    unsigned consumerIdx =
        op.getIndexOfShapedOperand(dependence.indexingView).getValue();
    AffineMap consumerAccess = op.getIndexingMap(consumerIdx);
    // Previously asserted that the consumerAccess map is a projected
    // permutation, so all results are known to be AffineDimExprs. To remove
    // this restriction walk the expression to find which dimensions of the
    // consumer loop appear in the `consumerAccess`.
    DenseSet<unsigned> positions;
    for (auto expr : consumerAccess.getResults())
      positions.insert(expr.cast<AffineDimExpr>().getPosition());
    commonTilableLoops.emplace_back(std::move(positions));
  }

  // 2. Of the outer parallel loops, only those loops can be tiled + fused as
  // computed above for all the fused dependences can be used to tile and fuse.
  DenseSet<unsigned> tilableParallelLoops;
  for (auto index : llvm::seq<unsigned>(0, numOuterParallelLoops)) {
    if (llvm::all_of(commonTilableLoops,
                     [&](const DenseSet<unsigned> &tilableLoops) {
                       return tilableLoops.count(index);
                     }))
      tilableParallelLoops.insert(index);
  }
  return tilableParallelLoops;
}

/// Find all dependences that are to be fusable.
static Optional<
    SmallVector<LinalgDependenceGraph::LinalgDependenceGraphElem, 1>>
findAllFusableDependences(LinalgOp op,
                          const LinalgDependenceGraph &dependenceGraph,
                          const LinalgFusionOptions &fusionOptions) {
  SmallVector<LinalgDependenceGraph::LinalgDependenceGraphElem, 1>
      fusableDependences;
  for (auto operand : llvm::enumerate(op.getInputsAndOutputBuffers())) {
    if (fusionOptions.indicesToFuse &&
        !fusionOptions.indicesToFuse->count(operand.index()))
      continue;
    Optional<LinalgDependenceGraph::LinalgDependenceGraphElem>
        fusableDependence =
            findFusableProducer(op, operand.index(), dependenceGraph);
    if (!fusableDependence)
      continue;
    // Make sure that the indexing map of the view used for fusion in the
    // producer is a projected permutation.
    LinalgOp producerOp = cast<LinalgOp>(fusableDependence->dependentOpView.op);
    Value producerView = fusableDependence->dependentOpView.view;
    unsigned producerIdx =
        producerOp.getIndexOfOutputBuffer(producerView).getValue();
    AffineMap producerMap = producerOp.getOutputIndexingMap(producerIdx);
    if (!producerMap.isProjectedPermutation()) {
      op.emitError("unhandled non permutation indexing map for fused view in "
                   "producer for operand at index ")
          << operand.index();
      return llvm::None;
    }
    Value consumerView = fusableDependence->indexingView;
    unsigned consumerIdx = op.getIndexOfShapedOperand(consumerView).getValue();
    if (!op.getIndexingMap(consumerIdx).isProjectedPermutation()) {
      op.emitError(
          "unhandled case where indexing map for fused view in the consumer is "
          "not a projected permuration while fusing at index ")
          << operand.index();
      return llvm::None;
    }
    fusableDependences.push_back(*fusableDependence);
    if (!fusionOptions.indicesToFuse)
      break;
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
  Optional<SmallVector<LinalgDependenceGraph::LinalgDependenceGraphElem, 1>>
      fusableDependencesOpt =
          findAllFusableDependences(op, dependenceGraph, fusionOptions);
  if (!fusableDependencesOpt)
    return llvm::None;
  ArrayRef<LinalgDependenceGraph::LinalgDependenceGraphElem> fusableDependences(
      *fusableDependencesOpt);

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
  DenseSet<unsigned> tileFuseLoops =
      collectTileAndFuseLoops(op, fusableDependences);

  // If there are no fusable dependences or there are no tile+fusable loops,
  // just return.
  if (fusableDependences.empty() || tileFuseLoops.empty()) {
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
  for (auto producer : enumerate(fusableDependences)) {
    LinalgOp producerOp = cast<LinalgOp>(producer.value().dependentOpView.op);
    unsigned producerIdx =
        producerOp.getIndexOfOutputBuffer(producer.value().dependentOpView.view)
            .getValue();
    unsigned consumerIdx =
        op.getIndexOfShapedOperand(producer.value().indexingView).getValue();
    LinalgOp fusedOp =
        fuse(rewriter, producerOp, producerIdx, ret.op, consumerIdx);
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

static void fuseLinalgOpsGreedily(FuncOp f) {
  LLVM_DEBUG(f.print(dbgs() << "\nBefore linalg-fusion: \n"));

  OpBuilder b(f);
  OperationFolder folder(f.getContext());
  DenseSet<Operation *> eraseSet;

  // Save original Linalg ops, we only want to make a pass over those.
  SmallVector<Operation *, 8> linalgOps;
  f.walk([&](LinalgOp op) {
    // TODO: support multi-results.
    if (op.getOperation()->getNumResults() <= 1)
      linalgOps.push_back(op);
  });

  // Tile and Fuse for tensors inputs (TODO: all tensor operands).
  for (auto *op : llvm::reverse(linalgOps)) {
    LinalgOp linalgOp = cast<LinalgOp>(op);
    for (auto en : llvm::enumerate(linalgOp.getShapedOperands())) {
      if (en.value().getType().isa<MemRefType>()) {
        // TODO: LinalgDependenceGraph should be able to update itself.
        // The current naive and expensive reconstruction of the graph should be
        // removed.
        linalg::Aliases aliases;
        linalg::LinalgDependenceGraph graph(aliases, linalgOps);
        if (auto info =
                fuseProducerOfBuffer(b, op, en.index(), graph, &folder)) {
          auto *originalOp = info->originalProducer.getOperation();
          eraseSet.insert(originalOp);
          auto *originalOpInLinalgOpsVector =
              std::find(linalgOps.begin(), linalgOps.end(), originalOp);
          *originalOpInLinalgOpsVector = info->fusedProducer.getOperation();
        }
      } else {
        assert(en.value().getType().isa<RankedTensorType>());
        // Tile and Fuse tensor input (TODO: init_tensors too).
        if (en.index() >= linalgOp.getNumInputs())
          continue;
        if (auto info = fuseProducerOfTensor(b, op, en.index(), &folder)) {
          auto *originalOp = info->originalProducer.getOperation();
          auto *originalOpInLinalgOpsVector =
              std::find(linalgOps.begin(), linalgOps.end(), originalOp);
          *originalOpInLinalgOpsVector = info->fusedProducer.getOperation();
          // Don't mark for erasure in the tensor case, let DCE handle this.
        }
      }
    }
  }
  // The `fuseProducerOfBuffer` function performs structural checks and in
  // particular that no covering read or write exist between the consumer and
  // the producer. As a consequence, the only fusions that may occur preserve
  // subsequent dependences and are guaranteed by construction to produce the
  // whole view. We may thus erase the producer once it is fused.
  for (auto *e : eraseSet)
    e->erase();

  LLVM_DEBUG(f.print(dbgs() << "\nAfter linalg-fusion: \n"));
}

namespace {
struct LinalgFusionPass : public LinalgFusionBase<LinalgFusionPass> {
  void runOnFunction() override { fuseLinalgOpsGreedily(getFunction()); }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createLinalgFusionPass() {
  return std::make_unique<LinalgFusionPass>();
}
