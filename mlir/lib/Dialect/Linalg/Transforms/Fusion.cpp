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

/// Implements a simple high-level fusion pass of linalg library operations.
///
/// In each block, linalg ops are processed in reverse textual order.
/// Given a linalg op `O`, fusion occurs by:
///   1. inspecting the linalg ops that write into the views read by `O`. This
///      uses the SSA value of the views and a simple subview/slice analysis to
///      determine producer-consumer dependences;
///   2. greedily fuse the linalg ops that produce subview
///   3. inspect the fused ops and determine whether they have other remaining
///      LinalgOp uses. If not, then erase the original producing linalg op.
///
/// More advanced use cases, analyses as well as profitability heuristics are
/// left for future work.

// Return a cloned version of `op` that operates on `loopRanges`, assumed to be
// a subset of the original loop ranges of `op`.
// This is achieved by applying the `loopToOperandRangesMaps` permutation maps
// to the `loopRanges` in order to obtain view ranges.
static LinalgOp cloneWithLoopRanges(OpBuilder &b, Location loc, LinalgOp op,
                                    ArrayRef<Range> loopRanges) {
  assert(op.hasBufferSemantics() && "expected linalg op with buffer semantics");
  auto maps = op.indexing_maps();
  SmallVector<Value, 8> clonedViews;
  clonedViews.reserve(op.getNumInputsAndOutputs());
  // Iterate over the inputs and outputs in order.
  // Extract the subranges from the linearized ranges.
  SmallVector<Value, 8> ios(op.getInputsAndOutputBuffers());
  for (auto en : llvm::enumerate(ios)) {
    unsigned idx = en.index();
    auto map = maps[idx].cast<AffineMapAttr>().getValue();
    LLVM_DEBUG(dbgs() << "map: " << map << "\n");
    Value view = en.value();
    SmallVector<Range, 4> viewRanges(map.getNumResults());
    for (auto en2 : llvm::enumerate(map.getResults())) {
      unsigned d = en2.index();
      // loopToOperandRangesMaps are permutations-only.
      unsigned loopPos = en2.value().cast<AffineDimExpr>().getPosition();
      viewRanges[d] = loopRanges[loopPos];
      LLVM_DEBUG(dbgs() << "\ni,j: " << en.index() << ", " << en2.index()
                        << "\t"
                        << "loopPos: " << loopPos << "\t" << viewRanges[d]);
    }
    // Construct a new subview for the tile.
    unsigned rank = viewRanges.size();
    SmallVector<Value, 4> offsets, sizes, strides;
    offsets.reserve(rank);
    sizes.reserve(rank);
    strides.reserve(rank);
    for (auto r : viewRanges) {
      offsets.push_back(r.offset);
      sizes.push_back(r.size);
      strides.push_back(r.stride);
    }
    clonedViews.push_back(
        b.create<SubViewOp>(loc, view, offsets, sizes, strides));
  }
  auto operands = op.getAssumedNonShapedOperands();
  clonedViews.append(operands.begin(), operands.end());

  Operation *clonedOp = op.clone(b, loc, /*resultTypes*/ {}, clonedViews);
  // When the producer is an IndexedGenercOp, we have to transform its block
  // IV arguments according to the tiling of the consumer, i.e. offset them by
  // the values computed in `loopRanges`.
  if (auto indexedGenericOp = dyn_cast<IndexedGenericOp>(clonedOp)) {
    auto &block = indexedGenericOp.region().front();

    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(&block);
    for (unsigned i = 0, e = indexedGenericOp.getNumLoops(); i < e; ++i) {
      Value oldIndex = block.getArgument(i);
      AddIOp newIndex = b.create<AddIOp>(indexedGenericOp.getLoc(), oldIndex,
                                         loopRanges[i].offset);
      oldIndex.replaceAllUsesExcept(newIndex,
                                    SmallPtrSet<Operation *, 1>{newIndex});
    }
  }
  return clonedOp;
}

struct ViewDimension {
  Value view;
  unsigned dimension;
};

// Given an `op`, returns the first (`view`, `dimension`) pair that identifies
// the loop range at `loopDepth`. The semantics of the loopToOperandRangesMaps
// guarantees at least one such dimension is found. If multiple candidates exist
// they must agree by construction (i.e. have the same size) and we just return
// the first one.
static ViewDimension getViewDefiningLoopRange(LinalgOp op, unsigned loopDepth) {
  assert(op.hasBufferSemantics() && "expected linalg op with buffer semantics");
  auto maps = op.indexing_maps();
  // Iterate over the inputs and outputs in order.
  // Extract the subranges from the linearized ranges.
  SmallVector<Value, 8> ios(op.getInputsAndOutputBuffers());
  for (auto en : llvm::enumerate(ios)) {
    unsigned idx = en.index();
    auto map = maps[idx].cast<AffineMapAttr>().getValue();
    LLVM_DEBUG(dbgs() << "getViewDefiningLoopRange I/O idx: " << idx << "\n");
    LLVM_DEBUG(dbgs() << "getViewDefiningLoopRange map: " << map << "\n");
    Value view = en.value();
    SmallVector<Value, 8> viewRanges(map.getNumResults(), nullptr);
    for (auto en2 : llvm::enumerate(map.getResults())) {
      if (loopDepth == en2.value().cast<AffineDimExpr>().getPosition()) {
        LLVM_DEBUG(dbgs() << "getViewDefiningLoopRange loopDepth: " << loopDepth
                          << "\n");
        LLVM_DEBUG(dbgs() << "getViewDefiningLoopRange view: " << view << "\n");
        return ViewDimension{view, static_cast<unsigned>(en2.index())};
      }
    }
  }
  llvm_unreachable("Expect to be able to extract a view defining loop range");
}

static LinalgOp fuse(OpBuilder &b, LinalgOp producer, unsigned producerIdx,
                     LinalgOp consumer, unsigned consumerIdx,
                     OperationFolder *folder = nullptr) {
  assert(producer.hasBufferSemantics() &&
         "expected linalg op with buffer semantics");
  assert(consumer.hasBufferSemantics() &&
         "expected linalg op with buffer semantics");

  auto subView = dyn_cast_or_null<SubViewOp>(
      consumer.getBuffer(consumerIdx).getDefiningOp());
  auto slice = dyn_cast_or_null<SliceOp>(
      consumer.getBuffer(consumerIdx).getDefiningOp());
  assert(subView || slice);
  (void)subView;
  (void)slice;

  // loopToOperandRangesMaps are permutations-only by construction:
  //   we can always identify a data dimension with a (at least one) loop
  //   dimension.
  AffineMap producerMap =
      producer.indexing_maps()[producerIdx].cast<AffineMapAttr>().getValue();
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
        subView.getOrCreateRanges(b, loc)[en.index()];
  }

  // Iterate over all dimensions. For the dimensions not identified by the
  // producer map for `producerIdx`, we need to explicitly compute the view that
  // defines the loop ranges using the `producer`.
  for (unsigned i = 0, nLoops = loopRanges.size(); i < nLoops; ++i) {
    if (loopRanges[i].offset)
      LLVM_DEBUG(llvm::dbgs()
                 << "existing LoopRange: " << loopRanges[i] << "\n");
    else {
      auto viewDim = getViewDefiningLoopRange(producer, i);
      loopRanges[i] = Range{folded_std_constant_index(folder, 0),
                            std_dim(viewDim.view, viewDim.dimension),
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
  // Check for any fusion-preventing dependence to any view read/written that
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
  /// Skip the "viewSource" operand.
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

Optional<FusionInfo> mlir::linalg::fuseProducerOf(
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
      producerOp.getIndexOfInputAndOutputBuffer(producerView);
  assert(producerIdxOpt.hasValue() && "incorrect operand index");
  unsigned producerIdx = producerIdxOpt.getValue();

  auto fusedProducer =
      fuse(b, producerOp, producerIdx, consumer, consumerIdx, folder);
  return FusionInfo{producerOp, fusedProducer};
}

/// Returns the positions of the loop in `op` that can be tiled based on the
/// operations that are to be fused with it. For example, in a
///
///   linalg. matmul ins(%a, %b : ...) outs(%c : ...)
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
        op.getIndexOfInputAndOutputBuffer(dependence.indexingView).getValue();
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
        producerOp.getIndexOfInputAndOutputBuffer(producerView).getValue();
    AffineMap producerMap = producerOp.getIndexingMap(producerIdx);
    if (!producerMap.isProjectedPermutation()) {
      op.emitError("unhandled non permutation indexing map for fused view in "
                   "producer for operand at index ")
          << operand.index();
      return llvm::None;
    }
    Value consumerView = fusableDependence->indexingView;
    unsigned consumerIdx =
        op.getIndexOfInputAndOutputBuffer(consumerView).getValue();
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
    unsigned producerIdx = producerOp
                               .getIndexOfInputAndOutputBuffer(
                                   producer.value().dependentOpView.view)
                               .getValue();
    unsigned consumerIdx =
        op.getIndexOfInputAndOutputBuffer(producer.value().indexingView)
            .getValue();
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
    if (op.hasBufferSemantics())
      linalgOps.push_back(op);
  });

  // TODO: LinalgDependenceGraph should be able to update itself.
  // The current naive and expensive reconstruction of the graph should be
  // removed.
  for (auto *op : llvm::reverse(linalgOps)) {
    for (unsigned id = 0, e = LinalgOp(op).getNumInputsAndOutputBuffers();
         id < e; ++id) {
      linalg::Aliases aliases;
      linalg::LinalgDependenceGraph graph(aliases, linalgOps);
      if (auto info = fuseProducerOf(b, op, id, graph, &folder)) {
        auto *originalOp = info->originalProducer.getOperation();
        eraseSet.insert(originalOp);
        auto *originalOpInLinalgOpsVector =
            std::find(linalgOps.begin(), linalgOps.end(), originalOp);
        *originalOpInLinalgOpsVector = info->fusedProducer.getOperation();
      }
    }
  }
  // The `fuseProducerOf` function performs structural checks and in particular
  // that no covering read or write exist between the consumer and the producer.
  // As a consequence, the only fusions that may occur preserve subsequent
  // dependences and are guaranteed by construction to produce the whole view.
  // We may thus erase the producer once it is fused.
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
