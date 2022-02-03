//===- ElementwiseOpFusion.cpp - Implementation of linalg Fusion ---------===///
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the linalg dialect Fusion on tensors operations pass.
//
//===----------------------------------------------------------------------===//
#include <utility>

#include "PassDetail.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::linalg;

/// Append to `fusedOpIndexingMapAttrs` the indexing maps for the operands of
/// the `producer` to use in the fused operation given the indexing map of the
/// result of the producer in the consumer.
static AffineMap getIndexingMapOfProducerOperandsInCoordinatesOfFusedOp(
    OpOperand *producerOpOperand, AffineMap producerResultIndexMap,
    AffineMap fusedConsumerArgIndexMap) {
  // The indexing map in the consumer op (fusedConsumerArgIndexMap) is a map
  // from consumer loop -> consumer arg tensor index/producer result tensor
  // index. The fused loop is same as the consumer loop. For each producer arg
  // the indexing map to be computed is a map from consumer loop -> producer
  // arg tensor index.
  // producerResultIndexMap is a map from producer loop -> tensor index.
  // Compute the inverse to get map from tensor index -> producer loop.
  // The inverse is a map from producer result tensor index -> producer loop.
  AffineMap invProducerResultIndexMap =
      inversePermutation(producerResultIndexMap);
  assert(invProducerResultIndexMap &&
         "expected producer result indexig map to be invertible");

  LinalgOp producer = cast<LinalgOp>(producerOpOperand->getOwner());
  // argMap is a map from producer loop -> producer arg tensor index.
  AffineMap argMap = producer.getTiedIndexingMap(producerOpOperand);

  // Compose argMap with invProducerResultIndexMap to get a map from
  // producer result tensor index -> producer arg tensor index.
  AffineMap t1 = argMap.compose(invProducerResultIndexMap);

  // Compose t1 with fusedConsumerArgIndexMap gives an indexing map from
  // consumer loop/ fused loop -> producer arg tensor index.
  return t1.compose(fusedConsumerArgIndexMap);
}

/// Conditions for elementwise fusion of generic operations.
static bool areElementwiseOpsFusable(GenericOp producer, GenericOp consumer,
                                     OpOperand *consumerOpOperand) {
  // Producer and consumer must have tensor semantics.
  if (!producer.hasTensorSemantics() || !consumer.hasTensorSemantics())
    return false;

  // Verify that
  // - the producer has all "parallel" iterator type.
  if (producer.getNumParallelLoops() != producer.getNumLoops())
    return false;

  // Only allow fusing the producer of an input operand for now.
  // TODO: allow fusing the producer of an output operand.
  if (!consumer.isInputTensor(consumerOpOperand))
    return false;

  // Get the consumer index map. The number of results of the consumer index
  // map must match the number of loops of the producer.
  AffineMap consumerIndexMap = consumer.getTiedIndexingMap(consumerOpOperand);
  if (consumerIndexMap.getNumResults() != producer.getNumLoops())
    return false;

  // Currently support only operations with single result.
  if (producer.getNumOutputs() != 1)
    return false;

  // Finally the index_map for the result must be invertible. For now just
  // verify it is a permutation.
  AffineMap producerResultIndexMap =
      producer.getTiedIndexingMap(producer.getOutputOperand(0));
  if (!producerResultIndexMap.isPermutation())
    return false;

  // Ensure that the fusion does not remove size information required to
  // get the loop bounds. For non-reduction generics, this is trivially the
  // case due to the output operand. For reductions, we need to check that after
  // the fusion, each loop dimension has at least one input that defines it.
  if ((consumer.getNumReductionLoops())) {
    BitVector coveredDims(consumer.getNumLoops(), false);

    auto addToCoveredDims = [&](AffineMap map) {
      for (auto result : map.getResults())
        if (auto dimExpr = result.dyn_cast<AffineDimExpr>())
          coveredDims[dimExpr.getPosition()] = true;
    };

    for (auto pair :
         llvm::zip(consumer->getOperands(), consumer.getIndexingMaps())) {
      Value operand = std::get<0>(pair);
      if (operand == consumerOpOperand->get())
        continue;
      AffineMap operandMap = std::get<1>(pair);
      addToCoveredDims(operandMap);
    }

    for (OpOperand *operand : producer.getInputOperands()) {
      AffineMap newIndexingMap =
          getIndexingMapOfProducerOperandsInCoordinatesOfFusedOp(
              operand, producerResultIndexMap, consumerIndexMap);
      addToCoveredDims(newIndexingMap);
    }
    if (!coveredDims.all())
      return false;
  }

  return true;
}

/// Generate the region of the fused tensor operation. The region of the fused
/// op must be empty.
static void
generateFusedElementwiseOpRegion(PatternRewriter &rewriter, GenericOp fusedOp,
                                 AffineMap consumerToProducerLoopsMap,
                                 OpOperand *consumerOpOperand,
                                 unsigned nloops) {
  auto producer = cast<GenericOp>(consumerOpOperand->get().getDefiningOp());
  auto consumer = cast<GenericOp>(consumerOpOperand->getOwner());
  // Build the region of the fused op.
  Block &producerBlock = producer->getRegion(0).front();
  Block &consumerBlock = consumer->getRegion(0).front();
  Block *fusedBlock = new Block();
  fusedOp.region().push_back(fusedBlock);
  BlockAndValueMapping mapper;
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(fusedBlock);

  // 2. Add an index operation for every fused loop dimension and use the
  // `consumerToProducerLoopsMap` to map the producer indices.
  if (producer.hasIndexSemantics()) {
    // Add an index operation for every fused loop dimension.
    unsigned numFusedOpLoops =
        std::max(producer.getNumLoops(), consumer.getNumLoops());
    SmallVector<Value> fusedIndices;
    fusedIndices.reserve(numFusedOpLoops);
    llvm::transform(llvm::seq<uint64_t>(0, numFusedOpLoops),
                    std::back_inserter(fusedIndices), [&](uint64_t dim) {
                      return rewriter.create<IndexOp>(producer.getLoc(), dim);
                    });
    for (IndexOp indexOp :
         llvm::make_early_inc_range(producerBlock.getOps<IndexOp>())) {
      Value newIndex = rewriter.create<mlir::AffineApplyOp>(
          producer.getLoc(),
          consumerToProducerLoopsMap.getSubMap(indexOp.dim()), fusedIndices);
      mapper.map(indexOp.getResult(), newIndex);
    }
  }
  // TODO: allow fusing the producer of an output operand.
  assert(consumer.isInputTensor(consumerOpOperand) &&
         "expected producer of input operand");
  // 3. Consumer input operands up to consumerIdx (exclusive).
  for (BlockArgument bbArg : consumerBlock.getArguments().take_front(
           consumerOpOperand->getOperandNumber())) // input assumption.
    mapper.map(bbArg, fusedBlock->addArgument(bbArg.getType(), bbArg.getLoc()));

  // Replacing consumerIdx requires getting the cloned, yielded, value from
  // the (cloned) producer block. This happens in step 9.

  // 4. Splice in producer's input operands.
  for (BlockArgument bbArg :
       producerBlock.getArguments().take_front(producer.getNumInputs()))
    mapper.map(bbArg, fusedBlock->addArgument(bbArg.getType(), bbArg.getLoc()));

  // 4.b. Producer output operand/map that is fused needs to be mapped to the
  // producer bbArg if it is an "initTensor" (i.e. its value is actually read).
  assert(producer->getNumResults() == 1 && "expected single result producer");
  if (producer.isInitTensor(producer.getOutputOperand(0))) {
    BlockArgument bbArg = producerBlock.getArguments()
                              .drop_front(producer.getNumInputs())
                              // TODO: bbArg index of
                              .front();
    mapper.map(bbArg, fusedBlock->addArgument(bbArg.getType(), bbArg.getLoc()));
  }
  // 5. Remaining consumer's input operands (drop past index `consumerIdx`).
  for (BlockArgument bbArg :
       consumerBlock.getArguments()
           .take_front(consumer.getNumInputs())
           .drop_front(consumerOpOperand->getOperandNumber() + 1))
    mapper.map(bbArg, fusedBlock->addArgument(bbArg.getType(), bbArg.getLoc()));
  // 6. All of consumer's output operands.
  for (BlockArgument bbArg :
       consumerBlock.getArguments().take_back(consumer.getNumOutputs()))
    mapper.map(bbArg, fusedBlock->addArgument(bbArg.getType(), bbArg.getLoc()));
  // 7. All of producer's output operands except the one fused.
  // TODO: allow fusion of multi-result producers.
  assert(producer->getNumResults() == 1 && "expected single result producer");

  // 8. Clone all producer operations except for the yield and index operations
  // to the fused operation.
  for (auto &op : producerBlock.without_terminator()) {
    if (!isa<IndexOp>(op))
      rewriter.clone(op, mapper);
  }
  // 9. Now we can map the consumerBlock's `consumerIdx` block argument. Just
  // forward the yield operand.
  auto yieldOp = cast<linalg::YieldOp>(producerBlock.getTerminator());
  // TODO: allow fusion of multi-result producers.
  assert(producer->getNumResults() == 1 && "expected single result producer");
  unsigned producerResultNumber = 0;
  Value replacement =
      mapper.lookupOrDefault(yieldOp.getOperand(producerResultNumber));
  // Sanity checks, if replacement is not already in the mapper then it must be
  // produced outside.
  if (replacement == yieldOp.getOperand(producerResultNumber)) {
    if (auto bb = replacement.dyn_cast<BlockArgument>())
      assert(bb.getOwner() != &producerBlock &&
             "yielded block argument must have been mapped");
    else
      assert(!producer->isAncestor(replacement.getDefiningOp()) &&
             "yielded value must have been mapped");
  }
  mapper.map(consumerBlock.getArgument(consumerOpOperand->getOperandNumber()),
             replacement);
  // 10. Clone operations from the consumer to the fused op.
  for (auto &op : consumerBlock.getOperations())
    rewriter.clone(op, mapper);

  // Sanity checks.
  assert(fusedBlock->getNumArguments() == fusedOp.getNumOperands() &&
         "Ill-formed GenericOp region");
}

static Optional<SmallVector<Value>>
fuseElementwiseOpsImpl(GenericOp producer, OpOperand *consumerOpOperand,
                       const ControlElementwiseOpsFusionFn &controlFn,
                       PatternRewriter &rewriter) {
  auto consumer = cast<GenericOp>(consumerOpOperand->getOwner());
  if (!areElementwiseOpsFusable(producer, consumer, consumerOpOperand) ||
      !controlFn(producer->getResult(0), *consumerOpOperand))
    return llvm::None;

  // TODO: allow fusing the producer of an output operand.
  assert(consumer.isInputTensor(consumerOpOperand) &&
         "expected producer of input operand");

  // Compute the fused operands list and indexing maps.
  SmallVector<Value> fusedOperands;
  SmallVector<AffineMap> fusedIndexMaps;
  fusedOperands.reserve(producer->getNumOperands() +
                        consumer->getNumOperands());
  fusedIndexMaps.reserve(producer->getNumOperands() +
                         consumer->getNumOperands());
  // In the following, numbering matches that of `generateFusedTensorOpRegion`.
  // 3. Consumer input operands/maps up to consumerIdx (exclusive).
  SmallVector<OpOperand *> consumerInputs = consumer.getInputOperands();
  SmallVector<OpOperand *>::iterator it =
      llvm::find(consumerInputs, consumerOpOperand);
  assert(it != consumerInputs.end() && "expected to find the consumer operand");
  for (OpOperand *opOperand : llvm::make_range(consumerInputs.begin(), it)) {
    fusedOperands.push_back(opOperand->get());
    fusedIndexMaps.push_back(consumer.getTiedIndexingMap(opOperand));
  }
  // 4. Splice in producer's input operands/maps.
  assert(producer->getNumResults() == 1 && "expected single result producer");
  AffineMap producerResultIndexMap =
      producer.getTiedIndexingMap(producer.getOutputOperand(0));
  for (OpOperand *opOperand : producer.getInputOperands()) {
    fusedOperands.push_back(opOperand->get());
    // Compute indexing maps for the producer args in the fused operation.
    AffineMap map = getIndexingMapOfProducerOperandsInCoordinatesOfFusedOp(
        opOperand, producerResultIndexMap,
        consumer.getTiedIndexingMap(consumerOpOperand));
    fusedIndexMaps.push_back(map);
  }
  // 4.b. Producer output operand/map that is fused needs to be passed if it is
  // an "initTensor" (i.e. its value is actually read).
  assert(producer->getNumResults() == 1 && "expected single result producer");
  if (producer.isInitTensor(producer.getOutputOperand(0))) {
    fusedOperands.push_back(producer.getOutputOperand(0)->get());
    // Compute indexing maps for the producer args in the fused operation.
    AffineMap map = getIndexingMapOfProducerOperandsInCoordinatesOfFusedOp(
        producer.getOutputOperand(0), producerResultIndexMap,
        consumer.getTiedIndexingMap(consumerOpOperand));
    fusedIndexMaps.push_back(map);
  }
  // 5. Remaining consumer's input operands/maps (drop past index
  // `consumerIdx`).
  for (OpOperand *opOperand :
       llvm::make_range(std::next(it), consumerInputs.end())) {
    fusedOperands.push_back(opOperand->get());
    fusedIndexMaps.push_back(consumer.getTiedIndexingMap(opOperand));
  }
  // 6. All of consumer's output operands (skip operands: added by the builder).
  for (OpOperand *opOperand : consumer.getOutputOperands())
    fusedIndexMaps.push_back(consumer.getTiedIndexingMap(opOperand));
  // 7. All of producer's output operands/maps except the one fused.
  // TODO: allow fusion of multi-result producers.
  assert(producer->getNumResults() == 1 && "expected single result producer");

  // Generate the fused op.
  SmallVector<Value> consumerOutputs = consumer.getOutputOperands();
  auto fusedOp = rewriter.create<GenericOp>(
      consumer.getLoc(), consumer->getResultTypes(),
      /*inputs=*/fusedOperands,
      // TODO: handle outputs.
      consumerOutputs, rewriter.getAffineMapArrayAttr(fusedIndexMaps),
      consumer.iterator_types(),
      /*doc=*/nullptr,
      /*library_call=*/nullptr);
  if (!fusedOp.getShapesToLoopsMap()) {
    // Fused op has invalid indexing maps. Typically this means something is off
    // in the input, but going ahead here would result in verification errors.
    // So cleanup and abort.
    rewriter.eraseOp(fusedOp);
    return llvm::None;
  }

  // Construct an AffineMap from consumer loops to producer loops.
  // consumer loop -> tensor index
  AffineMap consumerResultIndexMap =
      consumer.getTiedIndexingMap(consumerOpOperand);
  // tensor index -> producer loop
  AffineMap invProducerResultIndexMap =
      inversePermutation(producerResultIndexMap);
  assert(invProducerResultIndexMap &&
         "expected producer result indexig map to be invertible");
  // consumer loop -> producer loop
  AffineMap consumerToProducerLoopsMap =
      invProducerResultIndexMap.compose(consumerResultIndexMap);

  generateFusedElementwiseOpRegion(rewriter, fusedOp,
                                   consumerToProducerLoopsMap,
                                   consumerOpOperand, consumer.getNumLoops());
  return SmallVector<Value>(fusedOp->getResults());
}

/// Linearize the expressions in `sourceMap` based on the `reassociationMaps`
/// provided, given the shape of the source tensor that corresponds to the
/// `sourceMap`. Note that this implicitly assumes that the tensors dimensions
/// are "row-major" ordered logically.
///
/// For example:
///
/// %0 = op ... : tensor<?x?x4x5xf32>
/// with output index_map `affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>`
///
/// and reshape:
/// %1 = tensor.collapse_shape %0 [[0], [0, 1, 2]] :
///        tensor<?x?x4x5xf32> into tensor<?x?xf32>
///
/// would be rewritten into:
/// %0 = op ... : tensor<?x?x4x5xf32>
/// with output index_map
///   `affine_map<(d0, d1, d2, d3) -> (d0, d1 * 20 + d2 * 5 + d3)>`
template <typename TensorReshapeOp>
static AffineMap linearizeCollapsedDims(AffineMap sourceMap,
                                        TensorReshapeOp reshapeOp) {
  constexpr bool isExpanding =
      std::is_same<TensorReshapeOp, tensor::ExpandShapeOp>::value;
  ArrayRef<int64_t> sourceShape =
      (isExpanding ? reshapeOp.getResultType().getShape()
                   : reshapeOp.getSrcType().getShape());
  SmallVector<AffineExpr> resultExprs;
  ArrayRef<AffineExpr> sourceExprs = sourceMap.getResults();
  MLIRContext *context = sourceMap.getContext();

  // Compute the result exprs based on the reassociation maps.
  for (auto &indices : reshapeOp.getReassociationIndices()) {
    // Assume that they are in-order and contiguous (already checked in
    // verifier).
    assert(!indices.empty());
    SmallVector<int64_t> sizes;
    SmallVector<AffineExpr> dimExprs;
    for (auto en : llvm::zip(sourceShape.slice(indices[0], indices.size()),
                             sourceExprs.slice(indices[0], indices.size()))) {
      if (std::get<0>(en) == 1)
        continue;
      sizes.push_back(std::get<0>(en));
      dimExprs.push_back(std::get<1>(en));
    }
    AffineExpr linearizedExpr =
        makeCanonicalStridedLayoutExpr(sizes, dimExprs, context);
    resultExprs.push_back(linearizedExpr);
  }
  // The new affine map cannot drop unused dimension but some new symbols may
  // have been added. Create a map with at least as many dimensions/symbols as
  // the original affine map.
  int64_t maxDim = -1;
  int64_t maxSym = -1;
  getMaxDimAndSymbol<SmallVector<AffineExpr>>({resultExprs}, maxDim, maxSym);
  unsigned numDims = std::max(unsigned(maxDim + 1), sourceMap.getNumDims());
  unsigned numSyms = std::max(unsigned(maxSym + 1), sourceMap.getNumSymbols());
  return AffineMap::get(numDims, numSyms, resultExprs, context);
}

// tensor::ExpandShapeOp is fusable with its consumer (i.e. reshape as a
// producer). Fusing when operand has higher rank will require use of mods and
// divs in the indexing maps of the fused op which would make it non-invertible.
static bool isTensorReshapeOpFoldableByLinearization(
    tensor::ExpandShapeOp expandOp, AffineMap useIndexMap, bool asProducer) {
  if (!asProducer)
    return false;
  return useIndexMap.isPermutation();
}

// tensor::CollapseShapeOp is fusable with its producer (i.e. reshape as a
// consumer).
static bool
isTensorReshapeOpFoldableByLinearization(tensor::CollapseShapeOp collapseOp,
                                         AffineMap useIndexMap,
                                         bool asProducer) {
  if (asProducer)
    return false;
  return useIndexMap.isPermutation();
}

/// Check if the reshape operation is only expansion into/collapsing of
/// unit-dimension.
template <typename TensorReshapeOp>
static bool isUnitDimExpansionOnly(TensorReshapeOp reshapeOp) {
  constexpr bool isExpanding =
      std::is_same<TensorReshapeOp, tensor::ExpandShapeOp>::value;
  ArrayRef<int64_t> expandedShape =
      (isExpanding ? reshapeOp.getResultType().getShape()
                   : reshapeOp.getSrcType().getShape());
  for (auto &indices : reshapeOp.getReassociationIndices()) {
    unsigned numUnitDims = 0;
    for (int64_t position : indices)
      if (expandedShape[position] == 1)
        numUnitDims++;
    if (numUnitDims != indices.size() - 1)
      return false;
  }
  return true;
}

/// Conditions for folding a generic operation with a reshape op by expanding
/// the iteration space dimensionality for tensor operations. These are
/// preconditions assumed by `foldReshapeByDimExpansion` which implements the
/// following fusion pattern.
///
///  Consider
///
///  %c = linalg.generic ins(%a, %b : memref<?x?x?xf32>, memref<?x?xf32>)
///         indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d0, d2)>,
///                          affine_map<(d0, d1, d2) -> (d1, d2)>,
///                          affine_map<(d0, d1, d2) -> (d0, d2, d1)>]
///  %d = tensor.expand_shape %c [[0, 1], [2], [3, 4, 5]]
///       : tensor<?x?x?xf32> into tensor<?x?x?x?x?x?xf32>
///
///  The reshape can be folded into the `genericOp` if its loop dimensionality
///  is increased to match the result (operand) of the tensor_expand_shape.
///  The indexing_map of the fused tensor in the `genericOp` and the
///  reassociation map helps compute the indexing maps of the modified op.
///  For the above example, based on the reassociation map it
///  can be concluded that
///
///  - The loop used to access the first dimension of the fused tensor is split
///    into two.
///  - The loop used to access the second dimension of the fused tensor is kept
///    as is.
///  - The loop used to access the third dimension of the fused tensor is split
///    into three.
///
///  i.e. (e0, e1, e2, e3, e4) is the domain of the indexing map of the modified
///  op, then
///
///   d0 -> e0, e1
///   d1 -> e2, e3, e4
///   d2 -> e5
///
///  substituting this, the generic op can be rewritten as
///
///  %d = linalg.generic ins(%0, %1 : )
///        indexing_maps =
///         [affine_map<(e0, e1, e2, e3, e4, e5) -> (e2, e3, e4, e0, e1, e5)>,
///          affine_map<(e0, e1, e2, e3, e4, e5) -> (e2, e3, e4, e5)>,
///          affine_map<(e0, e1, e2, e3, e4, e5) -> (e0, e1, e5, e2, e3, e4)>]
///
///  Since operands to the linalg generic are now 5D, reshapes can be introduced
///  to make it consistent
///
///  %0 = tensor.expand_shape %a [[0, 1, 2], [3, 4], [5]]
///       : tensor<?x?x?xf32> into tensor<?x?x?x?x?x?xf32>
///  %1 = tensor.expand_shape %b [[0, 1, 2], [3]]
///       : tensor<?x?x?xf32> into tensor<?x?x?x?xf32>
///
///  The added reshapes are again expanding patterns, so they will get fused
///  with its producers if possible.
static bool isFusableWithReshapeByDimExpansion(GenericOp genericOp,
                                               OpOperand *fusableOpOperand) {
  // Is fusable only if:
  // - All the indexing maps for operands and results are projected
  //   permutations.
  // - The fused tensor is not a scalar.
  // - All the loops are parallel loops.
  return genericOp.hasTensorSemantics() &&
         llvm::all_of(genericOp.indexing_maps().getValue(),
                      [](Attribute attr) {
                        return attr.cast<AffineMapAttr>()
                            .getValue()
                            .isProjectedPermutation();
                      }) &&
         genericOp.getTiedIndexingMap(fusableOpOperand).getNumResults() > 0 &&
         llvm::all_of(genericOp.iterator_types(), [](Attribute attr) {
           return attr.cast<StringAttr>().getValue() ==
                  getParallelIteratorTypeName();
         });
}

namespace {
/// Information needed to expand a generic operation to fold the reshape with
/// it.
class ExpansionInfo {
public:
  // Computes the mapping from original dimensions of the op to the dimensions
  // of the expanded op given the `indexingMap` of the fused operand/result of
  // the generic op, the `reassocationMaps` of the reshape op and the shape of
  // the expanded op.
  LogicalResult compute(LinalgOp linalgOp, OpOperand *fusableOpOperand,
                        ArrayRef<AffineMap> reassociationMaps,
                        ArrayRef<int64_t> expandedShape,
                        ArrayRef<int64_t> collapsedShape,
                        PatternRewriter &rewriter);
  unsigned getOrigOpNumDims() const { return reassociation.size(); }
  unsigned getExpandedOpNumDims() const { return expandedOpNumDims; }
  ReassociationIndicesRef getExpandedDims(unsigned i) const {
    return reassociation[i];
  }
  ArrayRef<int64_t> getExpandedShapeOfDim(unsigned i) const {
    return expandedShapeMap[i];
  }
  ArrayRef<int64_t> getOriginalShape() const { return originalLoopExtent; }

private:
  /// Reassociation from the dimensions in the original operation to the
  /// dimension of the expanded operation.
  SmallVector<ReassociationIndices> reassociation;
  /// Mapping from extent of loops in the original operation, to the extent of
  /// loops in the expanded operation.
  SmallVector<SmallVector<int64_t>> expandedShapeMap;
  /// Extent of the loop in the original operation.
  SmallVector<int64_t> originalLoopExtent;
  unsigned expandedOpNumDims;
};
} // namespace

LogicalResult ExpansionInfo::compute(LinalgOp linalgOp,
                                     OpOperand *fusableOpOperand,
                                     ArrayRef<AffineMap> reassociationMaps,
                                     ArrayRef<int64_t> expandedShape,
                                     ArrayRef<int64_t> collapsedShape,
                                     PatternRewriter &rewriter) {
  if (reassociationMaps.empty())
    return failure();
  AffineMap fusedIndexMap = linalgOp.getTiedIndexingMap(fusableOpOperand);

  Optional<SmallVector<int64_t, 4>> originalLoopRange =
      linalgOp.getStaticLoopRanges();
  if (!originalLoopRange)
    return rewriter.notifyMatchFailure(linalgOp, "unable to find loop range");
  originalLoopExtent.assign(originalLoopRange->begin(),
                            originalLoopRange->end());

  reassociation.clear();
  expandedShapeMap.clear();
  // Compute the number of dimension in the expanded op that correspond to each
  // dimension of the original op.
  SmallVector<unsigned> numExpandedDims(fusedIndexMap.getNumDims(), 1);
  expandedShapeMap.resize(fusedIndexMap.getNumDims());
  for (const auto &resultExpr : llvm::enumerate(fusedIndexMap.getResults())) {
    unsigned pos = resultExpr.value().cast<AffineDimExpr>().getPosition();
    AffineMap foldedDims = reassociationMaps[resultExpr.index()];
    numExpandedDims[pos] = foldedDims.getNumResults();
    ArrayRef<int64_t> shape =
        expandedShape.slice(foldedDims.getDimPosition(0), numExpandedDims[pos]);
    expandedShapeMap[pos].assign(shape.begin(), shape.end());
  }
  // The remaining dimensions remain the same.
  for (unsigned i : llvm::seq<unsigned>(0, fusedIndexMap.getNumDims()))
    if (expandedShapeMap[i].empty())
      expandedShapeMap[i] = {originalLoopExtent[i]};

  // Compute reassociation map from the original op to the expanded op.
  unsigned sum = 0;
  reassociation.reserve(fusedIndexMap.getNumDims());
  for (const auto &numFoldedDim : llvm::enumerate(numExpandedDims)) {
    auto seq = llvm::seq<int64_t>(sum, sum + numFoldedDim.value());
    reassociation.emplace_back(seq.begin(), seq.end());
    sum += numFoldedDim.value();
  }
  expandedOpNumDims = sum;
  return success();
}

/// Epanding the body of a linalg operation requires adaptations of the accessed
/// loop indices. Specifically, access of indices in the original operation need
/// to be replaced with linearizations of indices in the expanded op. That
/// requires the shape of the expanded dimensions to be static (at least all but
/// the most significant). For now check that these are all statically sized.
/// Note that this could be extended to handle dynamic case, but the
/// implementation below uses `affine.apply` which seems to have issues when the
/// shapes are not static.
LogicalResult isGenericOpExpandable(GenericOp genericOp,
                                    const ExpansionInfo &expansionInfo,
                                    PatternRewriter &rewriter) {
  if (!genericOp.hasIndexSemantics())
    return success();
  for (unsigned i : llvm::seq<unsigned>(0, expansionInfo.getOrigOpNumDims())) {
    ArrayRef<int64_t> expandedShape = expansionInfo.getExpandedShapeOfDim(i);
    if (expandedShape.size() == 1)
      continue;
    for (int64_t shape : expandedShape.drop_front()) {
      if (ShapedType::isDynamic(shape)) {
        return rewriter.notifyMatchFailure(
            genericOp, "cannot expand due to index semantics and dynamic dims");
      }
    }
  }
  return success();
}

/// Return the indexing map to use in the expanded op for a given the
/// `indexingMap` of the original operation.
static AffineMap
getIndexingMapInExpandedOp(OpBuilder &builder, AffineMap indexingMap,
                           const ExpansionInfo &expansionInfo) {
  SmallVector<AffineExpr> newExprs;
  for (AffineExpr expr : indexingMap.getResults()) {
    unsigned pos = expr.cast<AffineDimExpr>().getPosition();
    SmallVector<AffineExpr, 4> expandedExprs = llvm::to_vector<4>(
        llvm::map_range(expansionInfo.getExpandedDims(pos), [&](int64_t v) {
          return builder.getAffineDimExpr(static_cast<unsigned>(v));
        }));
    newExprs.append(expandedExprs.begin(), expandedExprs.end());
  }
  return AffineMap::get(expansionInfo.getExpandedOpNumDims(),
                        indexingMap.getNumSymbols(), newExprs,
                        builder.getContext());
}

/// Return the type of the operand/result to use in the expanded op given the
/// type in the original op.
static RankedTensorType getExpandedType(RankedTensorType originalType,
                                        AffineMap indexingMap,
                                        const ExpansionInfo &expansionInfo) {
  SmallVector<int64_t> expandedShape;
  for (AffineExpr expr : indexingMap.getResults()) {
    unsigned dim = expr.cast<AffineDimExpr>().getPosition();
    auto dimExpansion = expansionInfo.getExpandedShapeOfDim(dim);
    expandedShape.append(dimExpansion.begin(), dimExpansion.end());
  }
  return RankedTensorType::get(expandedShape, originalType.getElementType());
}

/// Returns the reassociation maps to use in the `tensor.expand_shape`
/// operation to convert the operands of the original operation to operands of
/// the expanded operation. The same method is used to compute the
/// `tensor.collapse_shape` used to collapse the result of the expanded
/// op to get the value that can replace all uses of the results of the original
/// op.
static SmallVector<ReassociationIndices>
getReassociationForExpansion(AffineMap indexingMap,
                             const ExpansionInfo &expansionInfo) {
  SmallVector<ReassociationIndices> reassociation;
  unsigned numReshapeDims = 0;
  for (AffineExpr expr : indexingMap.getResults()) {
    unsigned dim = expr.cast<AffineDimExpr>().getPosition();
    auto numExpandedDims = expansionInfo.getExpandedDims(dim).size();
    SmallVector<int64_t, 2> indices = llvm::to_vector<2>(
        llvm::seq<int64_t>(numReshapeDims, numReshapeDims + numExpandedDims));
    reassociation.emplace_back(std::move(indices));
    numReshapeDims += numExpandedDims;
  }
  return reassociation;
}

/// Update the body of an expanded linalg operation having index semantics. The
/// indices of the original operation need to be recovered by linearizing the
/// indices of the correspoding dimensions of the expanded operation. For now it
/// is assumed that the shapes of the expanded operation needed for
/// linearization are static.
static void updateExpandedGenericOpRegion(PatternRewriter &rewriter,
                                          Location loc, Region &fusedRegion,
                                          const ExpansionInfo &expansionInfo) {
  // Replace the original indices by the linearization of the expanded indices.
  for (IndexOp indexOp :
       llvm::make_early_inc_range(fusedRegion.front().getOps<IndexOp>())) {
    ArrayRef<int64_t> expandedDims =
        expansionInfo.getExpandedDims(indexOp.dim());
    assert(!expandedDims.empty() && "expected valid expansion info");

    // Skip index operations that are not affected by the expansion.
    if (expandedDims.size() == 1 &&
        expandedDims.front() == (int64_t)indexOp.dim())
      continue;

    // Linearize the expanded indices of the original index dimension.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(indexOp);
    ArrayRef<int64_t> expandedDimsShape =
        expansionInfo.getExpandedShapeOfDim(indexOp.dim()).drop_front();
    SmallVector<Value> expandedIndices;
    expandedIndices.reserve(expandedDims.size() - 1);
    llvm::transform(
        expandedDims.drop_front(), std::back_inserter(expandedIndices),
        [&](int64_t dim) { return rewriter.create<IndexOp>(loc, dim); });
    Value newIndex = rewriter.create<IndexOp>(loc, expandedDims.front());
    for (auto it : llvm::zip(expandedDimsShape, expandedIndices)) {
      assert(!ShapedType::isDynamic(std::get<0>(it)));
      AffineExpr idx, acc;
      bindDims(rewriter.getContext(), idx, acc);
      newIndex = rewriter.create<AffineApplyOp>(
          indexOp.getLoc(), idx + acc * std::get<0>(it),
          ValueRange{std::get<1>(it), newIndex});
    }
    rewriter.replaceOp(indexOp, newIndex);
  }
}

/// Implements the fusion of a tensor_collapse_shape or a tensor_expand_shape op
/// and a generic op as explained in `isFusableWithReshapeByExpansion`. Assumes
/// that those conditions have been satisfied.
static Optional<SmallVector<Value>>
fuseWithReshapeByExpansion(GenericOp genericOp, Operation *reshapeOp,
                           OpOperand *fusableOpOperand,
                           PatternRewriter &rewriter) {
  assert(isFusableWithReshapeByDimExpansion(genericOp, fusableOpOperand) &&
         "preconditions for fuse operation failed");
  // Check if reshape is expanding or collapsing.
  auto expandingReshapeOp = dyn_cast<tensor::ExpandShapeOp>(*reshapeOp);
  auto collapsingReshapeOp = dyn_cast<tensor::CollapseShapeOp>(*reshapeOp);
  bool isExpanding = (expandingReshapeOp != nullptr);
  RankedTensorType expandedType = isExpanding
                                      ? expandingReshapeOp.getResultType()
                                      : collapsingReshapeOp.getSrcType();
  RankedTensorType collapsedType = isExpanding
                                       ? expandingReshapeOp.getSrcType()
                                       : collapsingReshapeOp.getResultType();

  ExpansionInfo expansionInfo;
  if (failed(expansionInfo.compute(
          genericOp, fusableOpOperand,
          isExpanding ? expandingReshapeOp.getReassociationMaps()
                      : collapsingReshapeOp.getReassociationMaps(),
          expandedType.getShape(), collapsedType.getShape(), rewriter)))
    return llvm::None;

  if (failed(isGenericOpExpandable(genericOp, expansionInfo, rewriter)))
    return llvm::None;

  SmallVector<AffineMap, 4> expandedOpIndexingMaps = llvm::to_vector<4>(
      llvm::map_range(genericOp.getIndexingMaps(), [&](AffineMap m) {
        return getIndexingMapInExpandedOp(rewriter, m, expansionInfo);
      }));

  SmallVector<Value> expandedOpOperands;
  expandedOpOperands.reserve(genericOp.getNumInputs());
  for (OpOperand *opOperand : genericOp.getInputOperands()) {
    if (opOperand == fusableOpOperand) {
      expandedOpOperands.push_back(isExpanding ? expandingReshapeOp.src()
                                               : collapsingReshapeOp.src());
      continue;
    }
    if (genericOp.isInputTensor(opOperand)) {
      AffineMap indexingMap = genericOp.getTiedIndexingMap(opOperand);
      auto opOperandType = opOperand->get().getType().cast<RankedTensorType>();
      RankedTensorType expandedOperandType =
          getExpandedType(opOperandType, indexingMap, expansionInfo);
      if (expandedOperandType != opOperand->get().getType()) {
        // Reshape the operand to get the right type.
        SmallVector<ReassociationIndices> reassociation =
            getReassociationForExpansion(indexingMap, expansionInfo);
        if (failed(reshapeLikeShapesAreCompatible(
                [&](const Twine &msg) {
                  return rewriter.notifyMatchFailure(genericOp, msg);
                },
                opOperandType.getShape(), expandedOperandType.getShape(),
                reassociation,
                /*isExpandingReshape=*/true)))
          return llvm::None;
        expandedOpOperands.push_back(rewriter.create<tensor::ExpandShapeOp>(
            genericOp.getLoc(), expandedOperandType, opOperand->get(),
            reassociation));
        continue;
      }
    }
    expandedOpOperands.push_back(opOperand->get());
  }

  Location loc = genericOp.getLoc();
  SmallVector<Value> outputs;
  for (OpOperand *opOperand : genericOp.getOutputOperands()) {
    AffineMap indexingMap = genericOp.getTiedIndexingMap(opOperand);
    auto opOperandType = opOperand->get().getType().cast<RankedTensorType>();
    RankedTensorType expandedOutputType =
        getExpandedType(opOperandType, indexingMap, expansionInfo);
    if (expandedOutputType != opOperand->get().getType()) {
      SmallVector<ReassociationIndices> reassociation =
          getReassociationForExpansion(indexingMap, expansionInfo);
      if (failed(reshapeLikeShapesAreCompatible(
              [&](const Twine &msg) {
                return rewriter.notifyMatchFailure(genericOp, msg);
              },
              opOperandType.getShape(), expandedOutputType.getShape(),
              reassociation,
              /*isExpandingReshape=*/true)))
        return llvm::None;
      outputs.push_back(rewriter.create<tensor::ExpandShapeOp>(
          genericOp.getLoc(), expandedOutputType, opOperand->get(),
          reassociation));
    }
  }

  // The iterator types of the expanded op are all parallel.
  SmallVector<StringRef> iteratorTypes(expansionInfo.getExpandedOpNumDims(),
                                       getParallelIteratorTypeName());

  TypeRange resultTypes = ValueRange(outputs).getTypes();
  auto fusedOp =
      rewriter.create<GenericOp>(genericOp.getLoc(), resultTypes,
                                 /*inputs=*/expandedOpOperands, outputs,
                                 expandedOpIndexingMaps, iteratorTypes);
  Region &fusedRegion = fusedOp->getRegion(0);
  Region &originalRegion = genericOp->getRegion(0);
  rewriter.cloneRegionBefore(originalRegion, fusedRegion, fusedRegion.begin());

  // Update the index accesses after the expansion.
  updateExpandedGenericOpRegion(rewriter, loc, fusedRegion, expansionInfo);

  // Reshape the result values to their original shape if this is a collapsing
  // reshape folded into its consumer.
  SmallVector<Value> resultVals;
  for (OpResult opResult : genericOp->getOpResults()) {
    int64_t resultNumber = opResult.getResultNumber();
    if (!isExpanding && resultTypes[resultNumber] != opResult.getType()) {
      SmallVector<ReassociationIndices> reassociation =
          getReassociationForExpansion(
              genericOp.getTiedIndexingMap(
                  genericOp.getOutputOperand(resultNumber)),
              expansionInfo);
      resultVals.push_back(rewriter.create<tensor::CollapseShapeOp>(
          genericOp.getLoc(), opResult.getType(),
          fusedOp->getResult(resultNumber), reassociation));
    } else {
      resultVals.push_back(fusedOp->getResult(resultNumber));
    }
  }
  // Assuming a single result.
  return resultVals;
}

namespace {

/// Pattern to fold tensor_expand_shape op with its consumer by using the source
/// of the reshape op as the operand in the consumer (instead of the result of
/// the tensor_collapse_shape). The corresponding index map in the consumer
/// needs to be modified to linearize the folded dimension.
///
/// For example,
///
/// #map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
/// %0 = tensor.expand_shape %arg0 [[0], [1, 2], [3]]
///      tensor<?x?x?xf32> into tensor<?x?x4x?xf32>
/// %1 = linalg.generic { indexing_maps = [#map0, #map0, #map0], ... }
///        ins(%0, %arg1 : tensor<?x?x4x?xf32>, tensor<?x?x4x?xf32>) ...
///        -> tensor<?x?x4x?xf32>
///
/// can be folded into
///
/// #map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1 * 4 + d2, d3)>
/// #map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
/// %0 = linalg.generic { indexing_maps = [#map0, #map1, #map1] ... }
///        ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x4x?xf32>) ...
///        -> tensor<?x?x4x?xf32>
template <bool foldUnitDimReshapesOnly, typename TensorReshapeOp>
struct FoldProducerReshapeOpByLinearization
    : public OpRewritePattern<GenericOp> {
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!genericOp.hasTensorSemantics())
      return failure();
    SmallVector<OpOperand *> inputOperands = genericOp.getInputOperands();
    for (const auto &en : llvm::enumerate(inputOperands)) {
      auto reshapeOp = en.value()->get().getDefiningOp<TensorReshapeOp>();
      if (!reshapeOp)
        continue;

      if (!isTensorReshapeOpFoldableByLinearization(
              reshapeOp, genericOp.getTiedIndexingMap(en.value()),
              /*asProducer =*/true) ||
          (foldUnitDimReshapesOnly && !isUnitDimExpansionOnly(reshapeOp)))
        continue;

      // Compute the fused operands list,
      SmallVector<Value> fusedOperands = genericOp.getInputOperands();
      fusedOperands[en.index()] = reshapeOp.src();
      SmallVector<Value> outputOperands = genericOp.getOutputOperands();
      llvm::append_range(fusedOperands, outputOperands);

      // Compute indexing_maps for the fused operation. The indexing_maps for
      // the operands of the consumers that arent fused are the same.
      SmallVector<AffineMap> fusedIndexMaps = genericOp.getIndexingMaps();

      // Compute the indexing map to use for the result of the producer.
      AffineMap modifiedMap =
          linearizeCollapsedDims(fusedIndexMaps[en.index()], reshapeOp);
      // The modified map cannot have symbols.
      if (modifiedMap.getNumSymbols())
        return failure();
      for (AffineExpr expr : modifiedMap.getResults()) {
        if (!expr.isPureAffine())
          return failure();
      }
      fusedIndexMaps[en.index()] = modifiedMap;

      // Further check that the resulting index maps can be fused and
      // inverted. Without this the resultant op is not legal.
      if (!inversePermutation(concatAffineMaps(fusedIndexMaps))) {
        return rewriter.notifyMatchFailure(
            genericOp, "fused op loop bound computation failed");
      }

      rewriter.startRootUpdate(genericOp);
      genericOp->setOperands(fusedOperands);
      genericOp.indexing_mapsAttr(
          rewriter.getAffineMapArrayAttr(fusedIndexMaps));
      rewriter.finalizeRootUpdate(genericOp);
      return success();
    }
    return failure();
  }
};

static SmallVector<ReassociationIndices>
getReassociationIndices(ArrayRef<AffineMap> maps) {
  SmallVector<ReassociationIndices> reassociation;
  for (AffineMap map : maps) {
    ReassociationIndices indices;
    for (unsigned i = 0, e = map.getNumResults(); i < e; i++) {
      unsigned pos = map.getResult(i).cast<AffineDimExpr>().getPosition();
      indices.push_back(pos);
    }
    reassociation.push_back(indices);
  }
  return reassociation;
}

/// Pattern to move rank reducing reshape after an elementwise linalg generic
/// op. This is useful to expose more fusion opportunities between named ops and
/// generic ops. This can only be done if there is no broadcast or permuation
/// within the dimensions we need to merge.
///
/// For example,
///
///  %0 = tensor.expand_shape %A [[0, 1], [2]]
///      : tensor<12544x16xf32> into tensor<112x112x16xf32>
///  %2 = linalg.generic {indexing_maps = [
///    affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
///    affine_map<(d0, d1, d2) -> (d2)>,
///    affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types =
///    ["parallel", "parallel", "parallel"]} {
///  } -> tensor<112x112x16xf32>
///
///  into
///
///  %2 = linalg.generic {indexing_maps = [
///    affine_map<(d0, d1) -> (d0, d1)>,
///    affine_map<(d0, d1) -> (d1)>,
///    affine_map<(d0, d1) -> (d0, d1)>],
///    iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1
///    : tensor<12544x16xf32>, tensor<16xf32>) outs(%1 : tensor<12544x16xf32>) {
///  } -> tensor<12544x16xf32>
///  %3 = tensor.expand_shape %2 [[0, 1], [2]]
///    : tensor<12544x16xf32> into tensor<112x112x16xf32>
struct PushExpandingReshape : public OpRewritePattern<GenericOp> {
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    // Only apply to elementwise linalg on tensor.
    if (!genericOp.hasTensorSemantics() || genericOp.hasIndexSemantics() ||
        genericOp.getNumParallelLoops() != genericOp.getNumLoops())
      return failure();
    // Only support identity output maps. It could be extended to permuations if
    // needed.
    if (llvm::any_of(genericOp.getOutputOperands(), [&](OpOperand *opOperand) {
          return !genericOp.getTiedIndexingMap(opOperand).isIdentity();
        }))
      return failure();
    int64_t destRank = genericOp.getNumParallelLoops();
    SmallVector<Value> newOperands = genericOp.getInputOperands();
    tensor::ExpandShapeOp reshapeFound;
    // 1. Look for tensor_expand_shape operands and figure out save the
    // dimensions merged.
    SmallVector<OpOperand *> inputOperands = genericOp.getInputOperands();
    for (const auto &en : llvm::enumerate(inputOperands)) {
      auto reshapeOp =
          en.value()->get().template getDefiningOp<tensor::ExpandShapeOp>();
      if (!reshapeOp)
        continue;
      // TODO: We could support non-identity map as long as the merged
      // dimensions are still contiguous.
      if (!genericOp.getTiedIndexingMap(en.value()).isIdentity())
        continue;
      if (reshapeFound) {
        // Only support a second reshape op if it has the same reassociate maps.
        if (reshapeFound.getReassociationMaps() ==
            reshapeOp.getReassociationMaps())
          newOperands[en.index()] = reshapeOp.src();
        continue;
      }
      reshapeFound = reshapeOp;
      newOperands[en.index()] = reshapeOp.src();
    }
    if (!reshapeFound)
      return failure();

    // Calculate the reassociation indices and rassociated reverse map.
    SmallVector<ReassociationIndices> reassociation =
        getReassociationIndices(reshapeFound.getReassociationMaps());
    SmallVector<unsigned> remap(destRank);
    for (auto &indices : llvm::enumerate(reassociation)) {
      for (int64_t index : indices.value()) {
        remap[index] = indices.index();
      }
    }
    // 2. Verify that we can merge the dimensions in the linalg and that we
    // don't need to create new reshapes operands. Inserting new reshape
    // operands would defeat the purpose of the transformation.
    for (const auto &en : llvm::enumerate(inputOperands)) {
      if (en.value()->get() == newOperands[en.index()]) {
        AffineMap map = genericOp.getTiedIndexingMap(en.value());
        for (unsigned i : llvm::seq(unsigned(0), map.getNumResults())) {
          if (reassociation[remap[map.getDimPosition(i)]].size() > 1)
            return failure();
        }
      }
    }

    // 3. Calculate the affine map remapping and the reassociation to apply to
    // output tensors.
    SmallVector<AffineMap> newMaps;
    unsigned newRank = reassociation.size();
    for (auto map : genericOp.getIndexingMaps()) {
      SmallVector<AffineExpr> newExprs;
      for (auto expr : map.getResults()) {
        unsigned position = expr.template cast<AffineDimExpr>().getPosition();
        // Skip dimension merged except for the last of the group.
        if (reassociation[remap[position]].back() == position) {
          newExprs.push_back(
              getAffineDimExpr(remap[position], genericOp.getContext()));
        }
      }
      newMaps.push_back(
          AffineMap::get(newRank, 0, newExprs, genericOp.getContext()));
    }

    // 4. Reshape the output tensors.
    SmallVector<Value> newOutputs;
    SmallVector<Type> newOutputTypes;
    for (auto output : genericOp.outputs()) {
      auto newOutputType = RankedTensorType::get(
          reshapeFound.getSrcType().getShape(),
          output.getType().template cast<RankedTensorType>().getElementType());
      Value newOutput = rewriter.create<tensor::CollapseShapeOp>(
          genericOp->getLoc(), newOutputType, output, reassociation);
      newOutputTypes.push_back(newOutputType);
      newOutputs.push_back(newOutput);
    }
    // 5. Create a new generic op with lowerer rank.
    SmallVector<StringRef> iteratorTypes(newRank,
                                         getParallelIteratorTypeName());
    auto newOp = rewriter.create<GenericOp>(genericOp->getLoc(), newOutputTypes,
                                            newOperands, newOutputs, newMaps,
                                            iteratorTypes);
    rewriter.inlineRegionBefore(genericOp.region(), newOp.region(),
                                newOp.region().begin());
    // 6. Reshape the so that the type matches the uses.
    SmallVector<Value> newResults;
    for (const auto &result : llvm::enumerate(newOp->getResults())) {
      newResults.push_back(rewriter.create<tensor::ExpandShapeOp>(
          genericOp->getLoc(), genericOp.getOutputTensorTypes()[result.index()],
          result.value(), reassociation));
    }
    rewriter.replaceOp(genericOp, newResults);
    return success();
  }
};

/// Pattern to fuse a tensor_collapse_shape op with its consumer generic op,
/// when the reshape op is collapsing dimensions. The dimensionality of the loop
/// in the consumer is expanded.
class FoldWithProducerReshapeOpByExpansion
    : public OpRewritePattern<GenericOp> {
public:
  FoldWithProducerReshapeOpByExpansion(
      MLIRContext *context, ControlElementwiseOpsFusionFn foldReshapes,
      PatternBenefit benefit = 1)
      : OpRewritePattern<GenericOp>(context, benefit),
        controlFoldingReshapes(std::move(foldReshapes)) {}

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    for (OpOperand *opOperand : genericOp.getInputTensorOperands()) {
      tensor::CollapseShapeOp reshapeOp =
          opOperand->get().getDefiningOp<tensor::CollapseShapeOp>();
      if (!reshapeOp)
        continue;
      // Fold only if
      // - The tensor reshape op is folding.
      // - All constraints of fusing with reshape by expansion are met.
      if (!isFusableWithReshapeByDimExpansion(genericOp, opOperand) ||
          (!controlFoldingReshapes(reshapeOp->getResult(0), *opOperand)))
        continue;

      Optional<SmallVector<Value>> replacementValues =
          fuseWithReshapeByExpansion(genericOp, reshapeOp, opOperand, rewriter);
      if (!replacementValues)
        return failure();
      rewriter.replaceOp(genericOp, replacementValues.getValue());
      return success();
    }
    return failure();
  }

private:
  ControlElementwiseOpsFusionFn controlFoldingReshapes;
};

/// Pattern to fold tensor_collapse_shape or tensor_expand_shape op with its
/// producer. The corresponding index map in the consumer needs to be modified
/// to linearize the folded dimension.
template <bool foldUnitDimReshapesOnly, typename TensorReshapeOp>
struct FoldConsumerReshapeOpByLinearization
    : public OpRewritePattern<TensorReshapeOp> {
  using OpRewritePattern<TensorReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TensorReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    GenericOp producer = reshapeOp.src().template getDefiningOp<GenericOp>();
    if (!producer || !producer.hasTensorSemantics() ||
        producer.getNumOutputs() != 1 ||
        !isTensorReshapeOpFoldableByLinearization(
            reshapeOp,
            producer.getTiedIndexingMap(producer.getOutputOperand(0)),
            /*asProducer =*/false) ||
        (foldUnitDimReshapesOnly && !isUnitDimExpansionOnly(reshapeOp)))
      return failure();
    // The indexing_maps for the operands of the fused operation are same as
    // those for the operands of the producer.
    SmallVector<AffineMap> fusedIndexMaps = producer.getIndexingMaps();

    // Compute the indexing map to use for the operand of the producer.
    AffineMap modifiedMap = linearizeCollapsedDims(
        producer.getTiedIndexingMap(producer.getOutputOperand(0)), reshapeOp);
    for (AffineExpr expr : modifiedMap.getResults()) {
      if (!expr.isPureAffine()) {
        return rewriter.notifyMatchFailure(
            producer, "fused op indexing map is not affine");
      }
    }
    fusedIndexMaps.back() = modifiedMap;

    // Further check that the resulting index maps can be fused and
    // inverted. Without this the resultant op is not legal.
    if (!inversePermutation(concatAffineMaps(fusedIndexMaps))) {
      return rewriter.notifyMatchFailure(
          producer, "fused op loop bound computation failed");
    }

    Location loc = producer.getLoc();
    SmallVector<Value> inputOperands = producer.getInputOperands();
    Value output = rewriter.create<TensorReshapeOp>(
        loc, producer.getOutputOperand(0)->get(),
        reshapeOp.getReassociationExprs());
    auto fusedOp = rewriter.create<GenericOp>(
        loc, reshapeOp.getResultType(),
        /*inputs=*/inputOperands,
        // TODO: handle outputs.
        /*outputs=*/output, rewriter.getAffineMapArrayAttr(fusedIndexMaps),
        producer.iterator_types(),
        /*doc=*/nullptr,
        /*library_call=*/nullptr);
    auto &fusedRegion = fusedOp->getRegion(0);
    rewriter.cloneRegionBefore(producer->getRegion(0), fusedRegion,
                               fusedRegion.begin());
    rewriter.replaceOp(reshapeOp, fusedOp->getResults());
    return success();
  }
};

/// Pattern to fold a tensor_expand_shape op with its producer generic op
/// by expanding the dimensionality of the loop in the producer op.
struct FoldReshapeWithGenericOpByExpansion
    : public OpRewritePattern<tensor::ExpandShapeOp> {

  FoldReshapeWithGenericOpByExpansion(
      MLIRContext *context, ControlElementwiseOpsFusionFn foldReshapes,
      PatternBenefit benefit = 1)
      : OpRewritePattern<tensor::ExpandShapeOp>(context, benefit),
        controlFoldingReshapes(std::move(foldReshapes)) {}

  LogicalResult matchAndRewrite(tensor::ExpandShapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    // Fold only if all constraints of fusing with reshape by expansion are met.
    GenericOp producer = reshapeOp.src().getDefiningOp<GenericOp>();
    if (!producer || producer.getNumOutputs() != 1 ||
        !isFusableWithReshapeByDimExpansion(producer,
                                            producer.getOutputOperand(0)) ||
        !controlFoldingReshapes(producer->getResult(0),
                                reshapeOp->getOpOperand(0)))
      return failure();
    Optional<SmallVector<Value>> replacementValues = fuseWithReshapeByExpansion(
        producer, reshapeOp, producer.getOutputOperand(0), rewriter);
    if (!replacementValues)
      return failure();
    rewriter.replaceOp(reshapeOp, replacementValues.getValue());
    return success();
  }

private:
  ControlElementwiseOpsFusionFn controlFoldingReshapes;
};

/// Pattern to fold a generic op with a splat constant/scalar constant. Does not
/// handle cases where the constant is not single-valued.
class FoldScalarOrSplatConstant : public OpRewritePattern<GenericOp> {
public:
  FoldScalarOrSplatConstant(MLIRContext *context,
                            ControlElementwiseOpsFusionFn &fun,
                            PatternBenefit benefit = 1)
      : OpRewritePattern<GenericOp>(context, benefit), controlFn(fun) {}

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!genericOp.hasTensorSemantics())
      return failure();
    for (OpOperand *opOperand : genericOp.getInputOperands()) {
      Operation *def = opOperand->get().getDefiningOp();
      Attribute constantAttr;
      auto isScalarOrSplatConstantOp = [&constantAttr](Operation *def) -> bool {
        {
          DenseElementsAttr splatAttr;
          if (matchPattern(def, m_Constant<DenseElementsAttr>(&splatAttr)) &&
              splatAttr.isSplat() &&
              splatAttr.getType().getElementType().isIntOrFloat()) {
            constantAttr = splatAttr.getSplatValue<Attribute>();
            return true;
          }
        }
        {
          IntegerAttr intAttr;
          if (matchPattern(def, m_Constant<IntegerAttr>(&intAttr))) {
            constantAttr = intAttr;
            return true;
          }
        }
        {
          FloatAttr floatAttr;
          if (matchPattern(def, m_Constant<FloatAttr>(&floatAttr))) {
            constantAttr = floatAttr;
            return true;
          }
        }
        return false;
      };

      auto resultValue = opOperand->get().dyn_cast<OpResult>();
      if (!def || !resultValue || !isScalarOrSplatConstantOp(def) ||
          !controlFn(resultValue, *opOperand))
        continue;

      // The operands and the indexing_maps of the fused operation the same as
      // the operands and indexing_maps of the generic operations with the
      // values at the constant index dropped.
      SmallVector<AffineMap> fusedIndexMaps;
      SmallVector<Value> fusedOperands;
      SmallVector<Location> fusedLocs{genericOp.getLoc()};
      fusedIndexMaps.reserve(genericOp.getNumInputsAndOutputs());
      fusedOperands.reserve(genericOp.getNumInputs());
      fusedLocs.reserve(fusedLocs.size() + genericOp.getNumInputs());
      for (OpOperand *inputOperand : genericOp.getInputOperands()) {
        if (inputOperand == opOperand)
          continue;
        Value inputValue = inputOperand->get();
        fusedIndexMaps.push_back(genericOp.getTiedIndexingMap(inputOperand));
        fusedOperands.push_back(inputValue);
        fusedLocs.push_back(inputValue.getLoc());
      }
      for (OpOperand *outputOperand : genericOp.getOutputOperands())
        fusedIndexMaps.push_back(genericOp.getTiedIndexingMap(outputOperand));

      // Check if the operation shapes to loops map is computable.
      if (!inversePermutation(concatAffineMaps(fusedIndexMaps))) {
        return rewriter.notifyMatchFailure(
            genericOp, "fused op loop bound computation failed");
      }

      // Create a constant scalar value from the splat constant.
      Value scalarConstant = rewriter.create<arith::ConstantOp>(
          def->getLoc(), constantAttr, constantAttr.getType());

      SmallVector<Value> outputOperands = genericOp.getOutputOperands();
      auto fusedOp = rewriter.create<GenericOp>(
          rewriter.getFusedLoc(fusedLocs), genericOp->getResultTypes(),
          /*inputs=*/fusedOperands,
          /*outputs=*/outputOperands,
          rewriter.getAffineMapArrayAttr(fusedIndexMaps),
          genericOp.iterator_types(),
          /*doc=*/nullptr,
          /*library_call=*/nullptr);

      // Map the block argument corresponding to the replaced argument with the
      // scalar constant.
      Region &region = genericOp->getRegion(0);
      Block &entryBlock = *region.begin();
      BlockAndValueMapping mapping;
      mapping.map(entryBlock.getArgument(opOperand->getOperandNumber()),
                  scalarConstant);
      Region &fusedRegion = fusedOp->getRegion(0);
      rewriter.cloneRegionBefore(region, fusedRegion, fusedRegion.begin(),
                                 mapping);
      rewriter.replaceOp(genericOp, fusedOp->getResults());
      return success();
    }
    return failure();
  }

private:
  ControlElementwiseOpsFusionFn controlFn;
};

/// Base class for constant folding linalg.generic ops with N inputs, 1 output,
/// and permutation indexing maps.
///
/// `ConcreteType` should provide methods with signatures
///
/// ```c++
///   bool matchIndexingMaps(GenericOp genericOp) const;
///   RegionComputationFn getRegionComputeFn(GenericOp) const;
/// ```
///
/// The latter inspects the region and returns the computation inside as a
/// functor. The functor will be invoked with constant elements for all inputs
/// and should return the corresponding computea constant element for output.
template <typename ConcreteType>
class FoldConstantBase : public OpRewritePattern<GenericOp> {
public:
  struct APIntOrFloat {
    Optional<APInt> apInt;
    Optional<APFloat> apFloat;
  };
  struct APIntOrFloatArray {
    SmallVector<APInt> apInts;
    SmallVector<APFloat> apFloats;
  };
  using RegionComputationFn =
      std::function<APIntOrFloat(const APIntOrFloatArray &)>;

  FoldConstantBase(MLIRContext *context,
                   const ControlElementwiseOpsFusionFn &controlFn,
                   PatternBenefit benefit = 1)
      : OpRewritePattern<GenericOp>(context, benefit), controlFn(controlFn) {}

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (genericOp.hasBufferSemantics())
      return failure();

    // Only support ops generating one output for now.
    if (genericOp.getNumOutputs() != 1)
      return failure();

    auto outputType = genericOp.getResultTypes().front().dyn_cast<ShapedType>();
    // Require the output types to be static give we are generating constants.
    if (!outputType || !outputType.hasStaticShape())
      return failure();

    if (!llvm::all_of(genericOp.getInputOperands(), [](OpOperand *operand) {
          return operand->get().getType().isa<ShapedType>();
        }))
      return failure();

    // Make sure all element types are the same.
    auto getOperandElementType = [](OpOperand *operand) {
      return operand->get().getType().cast<ShapedType>().getElementType();
    };
    if (!llvm::is_splat(llvm::map_range(genericOp.getInputAndOutputOperands(),
                                        getOperandElementType)))
      return failure();

    // We can only handle the case where we have int/float elements.
    auto elementType = outputType.getElementType();
    if (!elementType.isIntOrFloat())
      return failure();

    // Require all indexing maps to be permutations for now. This is common and
    // it simplifies input/output access greatly: we can do the data shuffling
    // entirely in the compiler, without needing to turn all indices into
    // Values, and then do affine apply on them, and then match back the
    // constant again.
    if (!llvm::all_of(genericOp.getIndexingMaps(),
                      [](AffineMap map) { return map.isPermutation(); }))
      return failure();

    for (OpOperand *operand : genericOp.getOutputOperands()) {
      if (genericOp.payloadUsesValueFromOperand(operand))
        return failure();
    }

    // Further check the indexing maps are okay for the ConcreteType.
    if (!static_cast<const ConcreteType *>(this)->matchIndexingMaps(genericOp))
      return failure();

    // Defer to the concrete type to check the region and discover the
    // computation inside.
    RegionComputationFn computeFn =
        static_cast<const ConcreteType *>(this)->getRegionComputeFn(genericOp);
    if (!computeFn)
      return failure();

    // All inputs should be constants.
    int numInputs = genericOp.getNumInputs();
    SmallVector<DenseIntOrFPElementsAttr> inputValues(numInputs);
    for (const auto &operand : llvm::enumerate(genericOp.getInputOperands())) {
      if (!matchPattern(operand.value()->get(),
                        m_Constant(&inputValues[operand.index()])))
        return failure();
    }

    // Identified this as a potential candidate for folding. Now check the
    // policy to see whether we are allowed to proceed.
    for (int i = 0; i < numInputs; ++i) {
      OpOperand *consumer = genericOp.getInputOperand(i);
      OpResult producer = consumer->get().cast<OpResult>();
      if (!controlFn(producer, *consumer))
        return failure();
    }

    auto linalgOp = cast<LinalgOp>(genericOp.getOperation());
    SmallVector<int64_t, 4> loopBounds = linalgOp.computeStaticLoopSizes();
    int64_t numElements = outputType.getNumElements();

    // Use APInt/APFloat instead of Attribute here for constructing the output.
    // This helps to avoid blowing up compiler memory usage: Attributes would
    // unify the following cases but they have lifetime as the MLIRContext.
    SmallVector<APInt> intOutputValues;
    SmallVector<APFloat> fpOutputValues;
    if (elementType.template isa<FloatType>())
      fpOutputValues.resize(numElements, APFloat(0.f));
    else
      intOutputValues.resize(numElements);

    // Return the constant dim positions from the given permutation map.
    auto getDimPositions = [](AffineMap map) {
      SmallVector<unsigned> dims;
      dims.reserve(map.getNumResults());
      for (AffineExpr result : map.getResults()) {
        dims.push_back(result.cast<AffineDimExpr>().getPosition());
      }
      return dims;
    };

    SmallVector<SmallVector<unsigned>> inputDims;
    for (int i = 0; i < numInputs; ++i)
      inputDims.push_back(getDimPositions(genericOp.getIndexingMaps()[i]));
    auto outputDims = getDimPositions(genericOp.getIndexingMaps().back());
    auto outputShape = outputType.getShape();

    // Allocate small vectors for index delinearization. Initial values do not
    // matter here as they will be overwritten later.
    SmallVector<uint64_t> indices(loopBounds.size(), 0);
    SmallVector<uint64_t> dstIndices(loopBounds.size(), 0);
    SmallVector<SmallVector<uint64_t>> srcIndices(
        numInputs, SmallVector<uint64_t>(loopBounds.size(), 0));
    SmallVector<uint64_t> srcLinearIndices(numInputs, 0);
    uint64_t dstLinearIndex = 0;

    // Allocate spaces for compute function inputs. Initial values do not matter
    // here as they will be overwritten later.
    APIntOrFloatArray computeFnInputs;

    auto inputShapes = llvm::to_vector<4>(
        llvm::map_range(genericOp.getInputOperands(), [](OpOperand *operand) {
          return operand->get().getType().cast<ShapedType>().getShape();
        }));

    // Given a `linearIndex`, remap it to a linear index to access linalg op
    // inputs/ouputs. This mutates `indices`, `srcIndices`, `dstIndices`,
    // `srcLinearIndices`, `dstLinearIndex` in place.
    auto computeRemappedLinearIndex = [&](int linearIndex) {
      int totalCount = linearIndex;
      for (int dim = loopBounds.size() - 1; dim >= 0; --dim) {
        indices[dim] = totalCount % loopBounds[dim];
        totalCount /= loopBounds[dim];
      }

      for (int dim = loopBounds.size() - 1; dim >= 0; --dim) {
        for (int i = 0; i < numInputs; ++i)
          srcIndices[i][dim] = indices[inputDims[i][dim]];
        dstIndices[dim] = indices[outputDims[dim]];
      }

      dstLinearIndex = dstIndices.front();
      for (int i = 0; i < numInputs; ++i)
        srcLinearIndices[i] = srcIndices[i].front();

      for (int dim = 1; dim < outputType.getRank(); ++dim) {
        dstLinearIndex = dstLinearIndex * outputShape[dim] + dstIndices[dim];
        for (int i = 0; i < numInputs; ++i)
          srcLinearIndices[i] =
              srcLinearIndices[i] * inputShapes[i][dim] + srcIndices[i][dim];
      }
    };

    bool isFloat = elementType.isa<FloatType>();
    if (isFloat) {
      SmallVector<DenseElementsAttr::iterator_range<APFloat>> inFpRanges;
      for (int i = 0; i < numInputs; ++i)
        inFpRanges.push_back(inputValues[i].getValues<APFloat>());

      computeFnInputs.apFloats.resize(numInputs, APFloat(0.f));

      // Transpose the input constant. Because we don't know its rank in
      // advance, we need to loop over the range [0, element count) and
      // delinearize the index.
      for (int linearIndex = 0; linearIndex < numElements; ++linearIndex) {
        computeRemappedLinearIndex(linearIndex);

        // Collect constant elements for all inputs at this loop iteration.
        for (int i = 0; i < numInputs; ++i)
          computeFnInputs.apFloats[i] = inFpRanges[i][srcLinearIndices[i]];

        // Invoke the computation to get the corresponding constant output
        // element.
        fpOutputValues[dstLinearIndex] = *computeFn(computeFnInputs).apFloat;
      }
    } else {
      SmallVector<DenseElementsAttr::iterator_range<APInt>> inIntRanges;
      for (int i = 0; i < numInputs; ++i)
        inIntRanges.push_back(inputValues[i].getValues<APInt>());

      computeFnInputs.apInts.resize(numInputs);

      // Transpose the input constant. Because we don't know its rank in
      // advance, we need to loop over the range [0, element count) and
      // delinearize the index.
      for (int linearIndex = 0; linearIndex < numElements; ++linearIndex) {
        computeRemappedLinearIndex(linearIndex);

        // Collect constant elements for all inputs at this loop iteration.
        for (int i = 0; i < numInputs; ++i)
          computeFnInputs.apInts[i] = inIntRanges[i][srcLinearIndices[i]];

        // Invoke the computation to get the corresponding constant output
        // element.
        intOutputValues[dstLinearIndex] = *computeFn(computeFnInputs).apInt;
      }
    }

    DenseElementsAttr outputAttr =
        isFloat ? DenseElementsAttr::get(outputType, fpOutputValues)
                : DenseElementsAttr::get(outputType, intOutputValues);

    rewriter.replaceOpWithNewOp<ConstantOp>(genericOp, outputAttr);
    return success();
  }

private:
  ControlElementwiseOpsFusionFn controlFn;
};

// Folds linalg.generic ops that are actually transposes on constant values.
struct FoldConstantTranspose : public FoldConstantBase<FoldConstantTranspose> {
  using FoldConstantBase::FoldConstantBase;

  bool matchIndexingMaps(GenericOp genericOp) const {
    // We should have one input and one output.
    return genericOp.getIndexingMaps().size() == 2;
  }

  RegionComputationFn getRegionComputeFn(GenericOp genericOp) const {
    // Make sure the region only contains a yield op.
    Block &body = genericOp.region().front();
    if (!llvm::hasSingleElement(body))
      return nullptr;
    auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (!yieldOp)
      return nullptr;

    // The yield op should return the block argument corresponds to the input.
    for (Value yieldVal : yieldOp.values()) {
      auto yieldArg = yieldVal.dyn_cast<BlockArgument>();
      if (!yieldArg || yieldArg.getOwner() != &body)
        return nullptr;
      if (yieldArg.getArgNumber() != 0)
        return nullptr;
    }

    // No computation; just return the orginal value.
    return [](const APIntOrFloatArray &inputs) {
      if (inputs.apFloats.empty())
        return APIntOrFloat{inputs.apInts.front(), llvm::None};
      return APIntOrFloat{llvm::None, inputs.apFloats.front()};
    };
  }

  ControlElementwiseOpsFusionFn controlFn;
};

} // namespace

static Optional<SmallVector<Value>>
fuseElementwiseOps(PatternRewriter &rewriter, OpOperand *consumerOpOperand,
                   GenericOp producer,
                   const ControlElementwiseOpsFusionFn &controlFn) {
  if (producer->getNumResults() != 1)
    return llvm::None;

  return fuseElementwiseOpsImpl(producer, consumerOpOperand, controlFn,
                                rewriter);
}

bool mlir::linalg::skipUnitDimReshape(const OpResult &producer,
                                      OpOperand &consumer) {
  if (auto producerCollapseOp =
          dyn_cast<tensor::CollapseShapeOp>(producer.getOwner())) {
    return !isUnitDimExpansionOnly(producerCollapseOp);
  }
  if (auto consumerExpandOp =
          dyn_cast<tensor::ExpandShapeOp>(consumer.getOwner())) {
    return !isUnitDimExpansionOnly(consumerExpandOp);
  }
  return true;
}

namespace {
/// Patterns to fuse a generic op, with the producer of its operands.
class FuseElementwiseOps : public OpRewritePattern<GenericOp> {
public:
  FuseElementwiseOps(MLIRContext *context, ControlElementwiseOpsFusionFn &fun,
                     PatternBenefit benefit = 1)
      : OpRewritePattern<GenericOp>(context, benefit), controlFn(fun) {}

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    // Find the first operand that is defined by another generic op on tensors.
    for (OpOperand *opOperand : genericOp.getInputAndOutputOperands()) {
      auto producer =
          dyn_cast_or_null<GenericOp>(opOperand->get().getDefiningOp());
      if (!producer || !producer.hasTensorSemantics())
        continue;
      Optional<SmallVector<Value>> fusedOpResults =
          fuseElementwiseOps(rewriter, opOperand, producer, controlFn);
      if (fusedOpResults) {
        rewriter.replaceOp(genericOp, *fusedOpResults);
        return success();
      }
    }
    return failure();
  }

private:
  ControlElementwiseOpsFusionFn controlFn;
};

/// Pass that fuses generic ops on tensors. Used only for testing.
struct LinalgElementwiseOpFusionPass
    : public LinalgElementwiseOpFusionBase<LinalgElementwiseOpFusionPass> {
  void runOnOperation() override {
    Operation *op = getOperation();
    RewritePatternSet patterns(op->getContext());
    ControlElementwiseOpsFusionFn allowFoldingFn =
        [](const OpResult &producer, const OpOperand &consumer) {
          return true;
        };
    populateElementwiseOpsFusionPatterns(
        patterns,
        LinalgElementwiseFusionOptions().setControlFoldingReshapes(
            allowFoldingUnitDimReshapes ? allowFoldingFn : skipUnitDimReshape));

    // Use TopDownTraversal for compile time reasons
    GreedyRewriteConfig grc;
    grc.useTopDownTraversal = true;
    (void)applyPatternsAndFoldGreedily(op->getRegions(), std::move(patterns),
                                       grc);
  }
};

/// Pass to test folding of reshape ops with generic ops by linearization.
struct FoldReshapeOpsByLinearizationPass
    : public LinalgFoldReshapeOpsByLinearizationBase<
          FoldReshapeOpsByLinearizationPass> {
  void runOnOperation() override {
    Operation *op = getOperation();
    RewritePatternSet patterns(op->getContext());
    populateFoldReshapeOpsByLinearizationPatterns(patterns);
    if (allowFoldingUnitDimReshapes) {
      populateFoldUnitDimsReshapeOpsByLinearizationPatterns(patterns);
    }
    (void)applyPatternsAndFoldGreedily(op->getRegions(), std::move(patterns));
  }
};

/// Forces `outs` operands of linalg operations to use `linalg.init_tensor` if
/// the value of the `outs` operand is not used within the op.  This is only
/// implemented for `linalg.generic` operations for now, but should hold for all
/// linalg structured ops.
struct RemoveOutsDependency : public OpRewritePattern<GenericOp> {
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.startRootUpdate(op);
    bool modifiedOutput = false;
    Location loc = op.getLoc();
    for (OpOperand *opOperand : op.getOutputOperands()) {
      if (!op.payloadUsesValueFromOperand(opOperand)) {
        Value operandVal = opOperand->get();
        auto operandType = operandVal.getType().dyn_cast<RankedTensorType>();
        if (!operandType)
          continue;

        // If outs is already an `init_tensor` operation, nothing to do.
        auto definingOp = operandVal.getDefiningOp<InitTensorOp>();
        if (definingOp)
          continue;
        modifiedOutput = true;
        SmallVector<Value> dynamicDims;
        for (const auto &dim : llvm::enumerate(operandType.getShape())) {
          if (dim.value() != ShapedType::kDynamicSize)
            continue;
          dynamicDims.push_back(rewriter.createOrFold<tensor::DimOp>(
              loc, operandVal, dim.index()));
        }
        Value initTensor = rewriter.create<InitTensorOp>(
            loc, dynamicDims, operandType.getShape(),
            operandType.getElementType());
        op->setOperand(opOperand->getOperandNumber(), initTensor);
      }
    }
    if (!modifiedOutput) {
      rewriter.cancelRootUpdate(op);
      return failure();
    }
    rewriter.finalizeRootUpdate(op);
    return success();
  }
};

} // namespace

void mlir::linalg::populateFoldReshapeOpsByLinearizationPatterns(
    RewritePatternSet &patterns) {
  patterns
      .add<FoldProducerReshapeOpByLinearization<false, tensor::CollapseShapeOp>,
           FoldProducerReshapeOpByLinearization<false, tensor::ExpandShapeOp>,
           FoldConsumerReshapeOpByLinearization<false, tensor::CollapseShapeOp>,
           FoldConsumerReshapeOpByLinearization<false, tensor::ExpandShapeOp>>(
          patterns.getContext());
}

void mlir::linalg::populateFoldUnitDimsReshapeOpsByLinearizationPatterns(
    RewritePatternSet &patterns) {
  patterns
      .add<FoldProducerReshapeOpByLinearization<true, tensor::CollapseShapeOp>,
           FoldProducerReshapeOpByLinearization<true, tensor::ExpandShapeOp>,
           FoldConsumerReshapeOpByLinearization<true, tensor::CollapseShapeOp>,
           FoldConsumerReshapeOpByLinearization<true, tensor::ExpandShapeOp>>(
          patterns.getContext());
}

void mlir::linalg::populateFoldReshapeOpsByExpansionPatterns(
    RewritePatternSet &patterns,
    const ControlElementwiseOpsFusionFn &controlFoldingReshapes) {
  patterns.add<FoldReshapeWithGenericOpByExpansion>(patterns.getContext(),
                                                    controlFoldingReshapes);
  patterns.add<FoldWithProducerReshapeOpByExpansion>(patterns.getContext(),
                                                     controlFoldingReshapes);
}

void mlir::linalg::populateElementwiseOpsFusionPatterns(
    RewritePatternSet &patterns, LinalgElementwiseFusionOptions options) {
  auto *context = patterns.getContext();
  patterns.add<FuseElementwiseOps, FoldScalarOrSplatConstant,
               FoldConstantTranspose>(context,
                                      options.controlElementwiseOpsFusionFn);
  patterns.add<RemoveOutsDependency>(context);
  populateFoldReshapeOpsByExpansionPatterns(patterns,
                                            options.controlFoldingReshapesFn);
  AffineApplyOp::getCanonicalizationPatterns(patterns, context);
  GenericOp::getCanonicalizationPatterns(patterns, context);
  tensor::ExpandShapeOp::getCanonicalizationPatterns(patterns, context);
  tensor::CollapseShapeOp::getCanonicalizationPatterns(patterns, context);
  context->getLoadedDialect<LinalgDialect>()->getCanonicalizationPatterns(
      patterns);
}

void mlir::linalg::populatePushReshapeOpsPatterns(RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  patterns.add<PushExpandingReshape>(context);
}

std::unique_ptr<Pass> mlir::createLinalgElementwiseOpFusionPass() {
  return std::make_unique<LinalgElementwiseOpFusionPass>();
}

std::unique_ptr<Pass> mlir::createFoldReshapeOpsByLinearizationPass() {
  return std::make_unique<FoldReshapeOpsByLinearizationPass>();
}
