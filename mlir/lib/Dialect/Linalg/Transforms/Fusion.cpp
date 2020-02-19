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

#include "mlir/Analysis/Dominance.h"
#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/EDSC/Intrinsics.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/STLExtras.h"
#include "mlir/Transforms/FoldUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-fusion"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::linalg;

using folded_std_constant_index = folded::ValueBuilder<ConstantIndexOp>;

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

static llvm::cl::OptionCategory clOptionsCategory(DEBUG_TYPE " options");
static llvm::cl::list<unsigned> clTileSizes(
    "linalg-fusion-tile-sizes",
    llvm::cl::desc(
        "Tile sizes by which to tile linalg operations during linalg fusion"),
    llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated,
    llvm::cl::cat(clOptionsCategory));

// Return a cloned version of `op` that operates on `loopRanges`, assumed to be
// a subset of the original loop ranges of `op`.
// This is achieved by applying the `loopToOperandRangesMaps` permutation maps
// to the `loopRanges` in order to obtain view ranges.
static LinalgOp cloneWithLoopRanges(OpBuilder &b, Location loc, LinalgOp op,
                                    ArrayRef<SubViewOp::Range> loopRanges) {
  assert(op.hasBufferSemantics() && "expected linalg op with buffer semantics");
  auto maps = loopToOperandRangesMaps(op);
  SmallVector<Value, 8> clonedViews;
  clonedViews.reserve(op.getNumInputsAndOutputs());
  // Iterate over the inputs and outputs in order.
  // Extract the subranges from the linearized ranges.
  SmallVector<Value, 8> ios(op.getInputsAndOutputBuffers());
  for (auto en : llvm::enumerate(ios)) {
    unsigned idx = en.index();
    auto map = maps[idx];
    LLVM_DEBUG(dbgs() << "map: " << map << "\n");
    Value view = en.value();
    SmallVector<SubViewOp::Range, 4> viewRanges(map.getNumResults());
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
  auto operands = getAssumedNonViewOperands(op);
  clonedViews.append(operands.begin(), operands.end());
  return op.clone(b, loc, clonedViews);
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
  auto maps = loopToOperandRangesMaps(op);
  // Iterate over the inputs and outputs in order.
  // Extract the subranges from the linearized ranges.
  SmallVector<Value, 8> ios(op.getInputsAndOutputBuffers());
  for (auto en : llvm::enumerate(ios)) {
    unsigned idx = en.index();
    auto map = maps[idx];
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

static LinalgOp fuse(Value producedView, LinalgOp producer, LinalgOp consumer,
                     unsigned consumerIdx, unsigned producerIdx,
                     OperationFolder *folder) {
  assert(producer.hasBufferSemantics() &&
         "expected linalg op with buffer semantics");
  assert(consumer.hasBufferSemantics() &&
         "expected linalg op with buffer semantics");
  auto subView = dyn_cast_or_null<SubViewOp>(
      consumer.getInput(consumerIdx).getDefiningOp());
  auto slice =
      dyn_cast_or_null<SliceOp>(consumer.getInput(consumerIdx).getDefiningOp());
  assert(subView || slice);
  (void)subView;
  (void)slice;

  // loopToOperandRangesMaps are permutations-only by construction:
  //   we can always identify a data dimension with a (at least one) loop
  //   dimension.
  AffineMap producerMap =
      loopToOperandRangesMaps(producer)[producer.getNumInputs() + producerIdx];
  LLVM_DEBUG(dbgs() << "Producer Idx: " << producerIdx
                    << ", producer map: " << producerMap << "\n");

  unsigned nPar = producer.getNumParallelLoops();
  unsigned nRed = producer.getNumReductionLoops();
  unsigned nWin = producer.getNumWindowLoops();
  SmallVector<SubViewOp::Range, 8> loopRanges(nPar + nRed + nWin);

  // Iterate over dimensions identified by the producer map for `producerIdx`.
  // This defines a subset of the loop ranges that we need to complete later.
  for (auto en : llvm::enumerate(producerMap.getResults())) {
    unsigned posInProducerLoop = en.value().cast<AffineDimExpr>().getPosition();
    loopRanges[posInProducerLoop] = subView.getRanges()[en.index()];
  }

  OpBuilder b(consumer.getOperation());
  auto loc = consumer.getLoc();
  // Iterate over all dimensions. For the dimensions not identified by the
  // producer map for `producerIdx`, we need to explicitly compute the view that
  // defines the loop ranges using the `producer`.
  for (unsigned i = 0, nLoops = loopRanges.size(); i < nLoops; ++i) {
    if (loopRanges[i].offset)
      LLVM_DEBUG(llvm::dbgs()
                 << "existing LoopRange: " << loopRanges[i] << "\n");
    else {
      auto viewDim = getViewDefiningLoopRange(producer, i);
      loopRanges[i] = SubViewOp::Range{
          folded_std_constant_index(folder, 0),
          std_dim(viewDim.view, viewDim.dimension),
          folded_std_constant_index(folder, 1)
      };
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
  return true;
}

// Only consider RAW atm.
Optional<FusionInfo> mlir::linalg::fuseProducerOf(
    OpBuilder &b, LinalgOp consumer, unsigned consumerIdx,
    const LinalgDependenceGraph &graph, OperationFolder *folder) {
  assert(consumer.hasBufferSemantics() &&
         "expected linalg op with buffer semantics");
  LLVM_DEBUG(dbgs() << "\nStart examining consumer: "
                    << *consumer.getOperation());
  for (auto dependence : graph.getDependencesInto(
           consumer, LinalgDependenceGraph::DependenceType::RAW)) {
    LLVM_DEBUG(dbgs() << "\n***Consider producer:\t"
                      << *dependence.dependentOpView.op << "\n");
    auto producer = cast<LinalgOp>(dependence.dependentOpView.op);
    if (isa<linalg::IndexedGenericOp>(dependence.dependentOpView.op)) {
      LLVM_DEBUG(dbgs() << "Not fusing indexed_generic producer");
      continue;
    }

    // Check that the dependence is indeed on the input `consumerIdx` view.
    auto consumedView = dependence.indexingView;
    if (consumer.getInput(consumerIdx) != consumedView)
      continue;

    // Consumer consumes this view, `isStructurallyFusableProducer` also checks
    // whether it is a strict subview of the producer view.
    auto producedView = dependence.dependentOpView.view;
    auto producerIdx = producer.getIndexOfOutputBuffer(producedView).getValue();
    // `consumerIdx` and `producerIdx` exist by construction.
    LLVM_DEBUG(dbgs() << "\nRAW producer: " << *producer.getOperation()
                      << " view: " << producedView
                      << " output index: " << producerIdx);

    // Must be a subview or a slice to guarantee there are loops we can fuse
    // into.
    auto subView = dyn_cast_or_null<SubViewOp>(consumedView.getDefiningOp());
    auto slice = dyn_cast_or_null<SliceOp>(consumedView.getDefiningOp());
    if (!subView && !slice) {
      LLVM_DEBUG(dbgs() << "\nNot fusable (not a subview or slice)");
      continue;
    }

    // Simple fusability checks.
    if (!isFusableInto(graph, consumer, consumedView, producer))
      continue;

    // Fuse `producer` just before `consumer`.
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(consumer.getOperation());
    ScopedContext scope(b, consumer.getLoc());
    LLVM_DEBUG(dbgs() << "Fuse into consumer: " << *consumer << "\n");
    auto fusedProducer = fuse(producedView, producer, consumer, consumerIdx,
                              producerIdx, folder);

    return FusionInfo{producer, fusedProducer};
  }
  return llvm::None;
}

/// Checks if two Generic ops are fusible, when one is a producer and another is
/// a consumer (with the result of the producer being the `consumerIdx` operand
/// of the consumer).
static bool areTensorOpsFusible(LinalgOp producer, LinalgOp consumer,
                                unsigned consumerIdx) {
  // Verify that the producer and consumer are ops on tensors.
  if (!producer.hasTensorSemantics() || !consumer.hasTensorSemantics())
    return false;

  auto producerOp = dyn_cast<linalg::GenericOp>(producer.getOperation());
  auto consumerOp = dyn_cast<linalg::GenericOp>(consumer.getOperation());
  // Verify that
  // - the producer and consumers are generic ops,
  // - only handle cases where the producer has a single return value,
  // - the producer return value should be the same as argument at `consumerIdx`
  //   of the consumer,
  // - the producer has all "parallel" iterator type.
  // - only handle ops that use regions for specifying the scalar operations.
  if (!producerOp || !consumerOp || producerOp.getNumOutputs() != 1 ||
      producerOp.getResult(0) != consumerOp.getOperand(consumerIdx) ||
      producerOp.getNumParallelLoops() != producerOp.getNumLoops() ||
      producerOp.fun() || consumerOp.fun())
    return false;

  // Get the consumer index map. The number of results of the consumer index map
  // must match the number of loops of the producer.
  AffineMap consumerIndexMap = consumerOp.getIndexingMap(consumerIdx);
  if (consumerIndexMap.getNumResults() != producerOp.getNumLoops())
    return false;

  // Finally the index_map for the result must be invertible. For now just
  // verify it is a permutation.
  AffineMap producerResultIndexMap = producerOp.getOutputIndexingMap(0);
  return producerResultIndexMap.isPermutation();
}

/// Computes the indexing maps for arguments of a producer generic op when the
/// result of the producer is fused with the consumer.
/// - consumerIndexMap is the indexing_map for the argument in the consumer op
///   that is the result of the producer op.
/// - invProducerResultIndexMap is the inverse of the indexing_map for the
///   result in the producer op.
/// - producerArgIndexMap is the indexing_map of the argument of the producer
///   op.
/// The result is the indexing_map to use for the producer argument when the
/// producer and consumer ops are fused.
static AffineMap computeProducerArgMap(AffineMap consumerIndexMap,
                                       AffineMap invProducerResultIndexMap,
                                       AffineMap producerArgIndexMap) {
  // t1 is map from producer result tensor index -> producer arg tensor index.
  auto t1 = producerArgIndexMap.compose(invProducerResultIndexMap);
  // The return is map from consumer loop -> producer arg tensor index,
  // i.e. indexing_map for the producer argument in the fused operation.
  return t1.compose(consumerIndexMap);
}

Optional<LinalgOp> mlir::linalg::fuseTensorOps(OpBuilder &b, LinalgOp producer,
                                               LinalgOp consumer,
                                               unsigned consumerIdx,
                                               OperationFolder *folder) {
  if (!areTensorOpsFusible(producer, consumer, consumerIdx))
    return {};

  MLIRContext *context = b.getContext();
  auto producerOp = cast<linalg::GenericOp>(producer.getOperation());
  auto consumerOp = cast<linalg::GenericOp>(consumer.getOperation());
  AffineMap consumerIndexMap = consumerOp.getIndexingMap(consumerIdx);
  AffineMap invProducerResultIndexMap =
      inversePermutation(producerOp.getOutputIndexingMap(0));

  // Compute the fused op operandslist by replacing the operand corresponding to
  // the result of the producer, with the operands of the producer.
  unsigned fusedArgsIn =
      producerOp.getNumInputs() + consumerOp.getNumInputs() - 1;
  auto fusedArgsOut = consumerOp.getNumOutputs();
  SmallVector<Value, 2> fusedOperandsList(consumerOp.getOperands());
  fusedOperandsList.erase(std::next(fusedOperandsList.begin(), consumerIdx));
  fusedOperandsList.reserve(fusedArgsIn + fusedArgsOut);
  fusedOperandsList.insert(
      std::next(fusedOperandsList.begin(), consumerIdx),
      producerOp.operand_begin(),
      std::next(producerOp.operand_begin(), producerOp.getNumInputs()));

  // Compute the fused indexing_maps of the operands/results of the fused op.
  SmallVector<Attribute, 2> fusedIndexingMapAttrs;
  fusedIndexingMapAttrs.reserve(fusedArgsIn + fusedArgsOut);
  fusedIndexingMapAttrs.append(consumerOp.indexing_maps().begin(),
                               consumerOp.indexing_maps().end());
  fusedIndexingMapAttrs.erase(
      std::next(fusedIndexingMapAttrs.begin(), consumerIdx));
  auto *insertPos = std::next(fusedIndexingMapAttrs.begin(), consumerIdx);
  for (auto producerArgIndexAttr :
       llvm::enumerate(producerOp.indexing_maps())) {
    if (producerArgIndexAttr.index() == producerOp.getNumInputs())
      break;
    auto composedIndexMap = computeProducerArgMap(
        consumerIndexMap, invProducerResultIndexMap,
        producerArgIndexAttr.value().cast<AffineMapAttr>().getValue());
    insertPos = std::next(fusedIndexingMapAttrs.insert(
        insertPos, AffineMapAttr::get(composedIndexMap)));
  }

  // Generate the fused op.
  auto fusedLinalgOp = b.create<GenericOp>(
      UnknownLoc::get(context), consumerOp.getResultTypes(), fusedOperandsList,
      b.getI64IntegerAttr(fusedArgsIn), b.getI64IntegerAttr(fusedArgsOut),
      b.getArrayAttr(fusedIndexingMapAttrs), consumerOp.iterator_types(),
      /*doc=*/nullptr,
      /*fun=*/nullptr,
      /*library_call=*/nullptr);

  // Build the region of the fused op.
  auto &fusedOpRegion = fusedLinalgOp.region();
  Block &producerOpBlock = producerOp.region().front();
  Block &consumerOpBlock = consumerOp.region().front();
  Block *fusedBlock = new Block();
  fusedOpRegion.push_back(fusedBlock);
  BlockAndValueMapping mapper;
  // Map the arguments for the unmodified args from the consumer.
  for (auto consumerOpArg : llvm::enumerate(consumerOpBlock.getArguments())) {
    if (consumerOpArg.index() == consumerIdx) {
      // Map the arguments for the args from the producer.
      for (auto producerOpArg : producerOpBlock.getArguments())
        mapper.map(producerOpArg,
                   fusedBlock->addArgument(producerOpArg.getType()));
      continue;
    }
    mapper.map(consumerOpArg.value(),
               fusedBlock->addArgument(consumerOpArg.value().getType()));
  }

  // Add operations from producer (except the yield operation) to the fused op.
  for (auto &op : producerOpBlock.getOperations()) {
    if (auto yieldOp = dyn_cast<YieldOp>(op)) {
      // Lookup the value the yield operation is mapped to.
      Value yieldVal = yieldOp.getOperand(0);
      auto clonedVal = mapper.lookup(yieldVal);
      mapper.map(consumerOpBlock.getArgument(consumerIdx), clonedVal);
      continue;
    }
    fusedBlock->push_back(op.clone(mapper));
  }
  for (auto &op : consumerOpBlock.getOperations())
    fusedBlock->push_back(op.clone(mapper));

  return cast<LinalgOp>(fusedLinalgOp.getOperation());
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

  // TODO(pifon, ntv): LinalgDependenceGraph should be able to update itself.
  // The current naive and expensive reconstruction of the graph should be
  // removed.
  for (auto *op : llvm::reverse(linalgOps)) {
    for (unsigned id = 0, e = LinalgOp(op).getNumInputs(); id < e; ++id) {
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

/// Patterns to fuse a generic op, with the producer of its operands.
struct FuseGenericTensorOps : public OpRewritePattern<GenericOp> {
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(GenericOp op,
                                     PatternRewriter &rewriter) const override {
    if (!op.hasTensorSemantics())
      return matchFailure();

    // Find the first operand that is defined by another generic op on tensors.
    for (auto operand : llvm::enumerate(op.getOperation()->getOperands())) {
      auto definingOp =
          dyn_cast_or_null<GenericOp>(operand.value().getDefiningOp());
      if (!definingOp || !definingOp.hasTensorSemantics())
        continue;
      auto fusedOp =
          fuseTensorOps(rewriter, cast<LinalgOp>(definingOp.getOperation()),
                        cast<LinalgOp>(op.getOperation()), operand.index());
      if (!fusedOp)
        continue;
      rewriter.replaceOp(op, fusedOp.getValue().getOperation()->getResults());
      return matchSuccess();
    }
    return matchFailure();
  }
};

/// Pass that fuses generic ops on tensors. Used only for testing.
struct FusionOfTensorOpsPass : public OperationPass<FusionOfTensorOpsPass> {
  void runOnOperation() override {
    OwningRewritePatternList patterns;
    Operation *op = getOperation();
    patterns.insert<FuseGenericTensorOps>(op->getContext());
    applyPatternsGreedily(op->getRegions(), patterns);
  };
};

struct LinalgFusionPass : public FunctionPass<LinalgFusionPass> {
  void runOnFunction() override { fuseLinalgOpsGreedily(getFunction()); }
};
} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createLinalgFusionPass() {
  return std::make_unique<LinalgFusionPass>();
}

static PassRegistration<LinalgFusionPass>
    pass("linalg-fusion", "Fuse operations in the linalg dialect");

static PassRegistration<FusionOfTensorOpsPass>
    tensorOpsPass("linalg-fusion-for-tensor-ops",
                  "Fuse operations on RankedTensorType in linalg dialect");
