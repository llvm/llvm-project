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
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::linalg;

//===---------------------------------------------------------------------===//
// Methods and patterns that fuse elementwise `linalg.generic` operations.
//===---------------------------------------------------------------------===//

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
         "expected producer result indexing map to be invertible");

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
                       const ControlFusionFn &controlFn,
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

static Optional<SmallVector<Value>>
fuseElementwiseOps(PatternRewriter &rewriter, OpOperand *consumerOpOperand,
                   GenericOp producer, const ControlFusionFn &controlFn) {
  if (producer->getNumResults() != 1)
    return llvm::None;

  return fuseElementwiseOpsImpl(producer, consumerOpOperand, controlFn,
                                rewriter);
}

namespace {
/// Patterns to fuse a generic op, with the producer of its operands.
class FuseElementwiseOps : public OpRewritePattern<GenericOp> {
public:
  FuseElementwiseOps(MLIRContext *context, ControlFusionFn fun,
                     PatternBenefit benefit = 1)
      : OpRewritePattern<GenericOp>(context, benefit),
        controlFn(std::move(fun)) {}

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
  ControlFusionFn controlFn;
};
} // namespace

//===---------------------------------------------------------------------===//
// Methods and patterns that fuse reshape ops with elementwise operations by
// linearization of indexing maps.
//===---------------------------------------------------------------------===//

// TODO(ravishankarm): The indexing maps
// these produce in the general case are detrimental to transformations.
// These patterns are on deprecation path in favor of using fusion by
// collapsing, which covers the only legitimate use case of this pattern of
// folding unit-extent dims.

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
} // namespace

//===---------------------------------------------------------------------===//
// Methods and patterns that fuse reshape ops with elementwise operations by
// expanding the dimensionality of the elementwise operations.
//===---------------------------------------------------------------------===//

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
static LogicalResult isGenericOpExpandable(GenericOp genericOp,
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

/// Pattern to fuse a tensor_collapse_shape op with its consumer generic op,
/// when the reshape op is collapsing dimensions. The dimensionality of the loop
/// in the consumer is expanded.
class FoldWithProducerReshapeOpByExpansion
    : public OpRewritePattern<GenericOp> {
public:
  FoldWithProducerReshapeOpByExpansion(MLIRContext *context,
                                       ControlFusionFn foldReshapes,
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
  ControlFusionFn controlFoldingReshapes;
};

/// Pattern to fold a tensor_expand_shape op with its producer generic op
/// by expanding the dimensionality of the loop in the producer op.
struct FoldReshapeWithGenericOpByExpansion
    : public OpRewritePattern<tensor::ExpandShapeOp> {

  FoldReshapeWithGenericOpByExpansion(MLIRContext *context,
                                      ControlFusionFn foldReshapes,
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
  ControlFusionFn controlFoldingReshapes;
};
} // namespace

//===---------------------------------------------------------------------===//
// Methods and patterns to fuse reshape with linalg.generic operations by
// contraction of dimensions.
//===---------------------------------------------------------------------===//

/// For a given list of indices in the range of the `indexingMap` that are
/// folded, return the indices of the corresponding domain. Return `llvm::None`
/// on failure. Ensures that all the elements of the returned reassociation are
/// distinct.
static ReassociationIndices
getDomainReassociation(AffineMap indexingMap,
                       ReassociationIndicesRef rangeReassociation) {
  assert(indexingMap.isProjectedPermutation() &&
         "expected projected permutation");

  ReassociationIndices domainReassociation = llvm::to_vector<4>(
      llvm::map_range(rangeReassociation, [&](int64_t pos) -> int64_t {
        return indexingMap.getResults()[pos]
            .cast<AffineDimExpr>()
            .getPosition();
      }));
  // The projected permutation semantics ensures that there is no repetition of
  // the domain indices.
  return domainReassociation;
}

/// For a given `dimSequence`, check if the sequence is conserved in the
/// `indexingMap`. `indexingMap` is expected to be a projected permutation.
/// Non-existence of the sequence returns true as well.
static bool isDimSequencePreserved(AffineMap indexingMap,
                                   ReassociationIndicesRef dimSequence) {
  assert(!dimSequence.empty() &&
         "expected non-empty list for dimension sequence");
  assert(indexingMap.isProjectedPermutation() &&
         "expected indexing map to be projected permutation");

  llvm::SmallDenseSet<unsigned, 4> sequenceElements;
  sequenceElements.insert(dimSequence.begin(), dimSequence.end());

  unsigned dimSequenceStart = dimSequence[0];
  for (const auto &expr : enumerate(indexingMap.getResults())) {
    unsigned dimInMapStart = expr.value().cast<AffineDimExpr>().getPosition();
    // 1.  Check if this start of the sequence.
    if (dimInMapStart == dimSequenceStart) {
      if (expr.index() + dimSequence.size() > indexingMap.getNumResults())
        return false;
      // 1a. Check if sequence is preserved.
      for (const auto &dimInSequence : enumerate(dimSequence)) {
        unsigned dimInMap =
            indexingMap.getResult(expr.index() + dimInSequence.index())
                .cast<AffineDimExpr>()
                .getPosition();
        if (dimInMap != dimInSequence.value())
          return false;
      }
      // Found the sequence. Projected permutation
      // enforces that all AffineDimExprs in the result are unique, so no
      // further checks are needed.
      return true;
    }
    // 2. If position in the expr (which is of type AffineDimExpr) is part
    // of sequence, return false here. This implies the entire sequence does not
    // exist in the indexing map.
    if (sequenceElements.count(dimInMapStart))
      return false;
  }
  // 3. No element of sequence found. Return true.
  return true;
}

// Return the list of dimensions of the iteration domain that can be
// collapsed to allow for fusion with the a producer that is an expand_shape
// operation. If all dimensions created by expansion can be collapsed in the
// iteration space then the reshape is defunct.
//
// Example:
//
// ```mlir
// #map = affine_map<(d0, d1) -> (d0, d1)>
// %1 = tensor.expand_shape %0 [[0, 1]] : tensor<?xf32> into tensor<?x4xf32>
// %2 = linalg.init_tensor [..] : tensor<?x4xf32>
// %3 = linalg.generic {
//     indexing_maps = [#map, #map],
//     iterator_types = ["parallel" ,"parallel"]}
//     ins(%1 : tensor<?x4xf32>) outs(%2 : tensor<?x4xf32>) {.. }
// ```
//
// can be fused by collapsing the dimensions of the iteration space.
//
// ```mlir
// #map = affine_map<(d0) -> (d0)>
// %2 = linalg.init_tensor [..] : tensor<?xf32>
// %3 = linalg.generic {
//     indexing_maps = [#map, #map],
//     iterator_types = ["parallel"]}
//     ins(%1 : tensor<?xf32>) outs(%2 : tensor<?xf32>) {.. }
// %4 = tensor.expand_shape %3 [[0, 1]] : tensor<?xf32> into tensor<?x4xf32>
// ```
//
// In the following example,
//
// ```mlir
// #map0 = affine_map<(d0, d1) -> (d0, d1)>
// #map1 = affine_map<(d0, d1) -> (d1, d0)>
// %1 = tensor.expand_shape %0 [[0, 1]] : tensor<?xf32> into tensor<?x4xf32>
// %2 = linalg.init_tensor [..] : tensor<4x?xf32>
// %2 = linalg.generic {
//     indexing_maps = [#map0, #map1],
//     iterator_types = ["parallel" ,"parallel"]}
//     ins(%1 : tensor<?x4xf32>) outs(%2 : tensor<4x?xf32>) {.. }
// ```
//
// the reshape cannot be fused with the generic op by collapsing the op
// dimensions since the indexing maps will have to contain mods and divs
// to preserve the accesses pattern. When no dimensions of the iteration
// space are collapsable and empty vector is returned.
static SmallVector<ReassociationIndices>
getCollapsableIterationSpaceDims(GenericOp genericOp, OpOperand *fusableOperand,
                                 ArrayRef<ReassociationIndices> reassociation) {
  // Some basic checks for this fusion to be valid.
  if (!genericOp.hasTensorSemantics() || genericOp.getNumOutputs() != 1)
    return {};

  if (!llvm::all_of(genericOp.getIndexingMaps(), [](AffineMap map) {
        return map.isProjectedPermutation();
      })) {
    return {};
  }

  // Compute all the loops with the reduction iterator types.
  SmallVector<int64_t> reductionDims;
  for (auto iteratorType : llvm::enumerate(genericOp.iterator_types())) {
    if (isReductionIterator(iteratorType.value())) {
      reductionDims.push_back(iteratorType.index());
    }
  }

  llvm::SmallDenseSet<unsigned, 4> processedIterationDims;
  AffineMap indexingMap = genericOp.getTiedIndexingMap(fusableOperand);
  auto iteratorTypes = genericOp.iterator_types().getValue();
  SmallVector<ReassociationIndices> iterationSpaceReassociation;
  for (ReassociationIndicesRef foldedRangeDims : reassociation) {
    assert(!foldedRangeDims.empty() && "unexpected empty reassociation");

    // Ignore dims that are not folded.
    if (foldedRangeDims.size() == 1)
      continue;

    ReassociationIndices foldedIterationSpaceDims =
        getDomainReassociation(indexingMap, foldedRangeDims);

    // Check that the folded iteration dims do not contain already processed
    // dims.
    if (llvm::any_of(foldedIterationSpaceDims, [&](int64_t dim) {
          return processedIterationDims.count(dim);
        }))
      continue;

    // Check that all folded iterator types are all parallel or all reductions.
    Attribute startIteratorType = iteratorTypes[foldedIterationSpaceDims[0]];
    if (!isParallelIterator(startIteratorType) &&
        !isReductionIterator(startIteratorType))
      continue;
    if (llvm::any_of(foldedIterationSpaceDims, [&](int64_t dim) {
          return iteratorTypes[dim] != startIteratorType;
        }))
      continue;

    // If the folded dimensions correspond to a "reduction" iterator type,
    // the folded dimensions need to be "in-order". Strictly speaking this is
    // not necessary, for reductions that are associative and commutative,  but
    // using a more strict definition of reduction for now.
    if (isReductionIterator(startIteratorType)) {
      bool isContiguous = false;
      for (auto startDim : llvm::enumerate(reductionDims)) {
        // Move window in `reductionDims` to start of the folded iteration dims.
        if (startDim.value() != foldedIterationSpaceDims[0])
          continue;
        // If sizes doesnt match, trivial not contiguous. This condition should
        // not be hit.
        if (startDim.index() + foldedIterationSpaceDims.size() >
            reductionDims.size())
          break;
        // Check that the contiguity is maintained.
        isContiguous = true;
        for (auto foldedDim : llvm::enumerate(foldedIterationSpaceDims)) {
          if (reductionDims[foldedDim.index() + startDim.index()] !=
              foldedDim.value()) {
            isContiguous = false;
            break;
          }
        }
        break;
      }
      if (!isContiguous)
        continue;
    }

    // Check that the sequence is preserved in all indexing maps.
    if (llvm::any_of(genericOp.getIndexingMaps(), [&](AffineMap indexingMap) {
          return !isDimSequencePreserved(indexingMap, foldedIterationSpaceDims);
        }))
      continue;

    processedIterationDims.insert(foldedIterationSpaceDims.begin(),
                                  foldedIterationSpaceDims.end());
    iterationSpaceReassociation.emplace_back(
        std::move(foldedIterationSpaceDims));
  }

  return iterationSpaceReassociation;
}

/// Helper class to carry state while collapsing the `linalg.generic` op.
namespace {
class CollapsingInfo {
public:
  LogicalResult initialize(unsigned origNumLoops,
                           ArrayRef<ReassociationIndices> foldedIterationDims) {
    llvm::SmallDenseSet<int64_t, 4> processedDims;
    // Find all the dims that are folded.
    for (ReassociationIndicesRef foldedIterationDim : foldedIterationDims) {
      if (foldedIterationDim.empty())
        continue;
      // If the folded dims contain dims already folded, that's illegal
      // specification. Repetition within a list is also illegal.
      for (auto dim : foldedIterationDim) {
        if (dim >= origNumLoops)
          return failure();
        if (processedDims.count(dim))
          return failure();
        processedDims.insert(dim);
      }
      collapsedOpToOrigOpIterationDim.emplace_back(foldedIterationDim.begin(),
                                                   foldedIterationDim.end());
    }
    if (processedDims.size() > origNumLoops)
      return failure();

    // Add all the preserved dims of the original op as single
    // elements to `collapsedOpToOrigOpIterationDim`.
    for (auto dim : llvm::seq<int64_t>(0, origNumLoops)) {
      if (processedDims.count(dim))
        continue;
      collapsedOpToOrigOpIterationDim.emplace_back(ReassociationIndices{dim});
    }

    llvm::sort(collapsedOpToOrigOpIterationDim,
               [&](ReassociationIndicesRef lhs, ReassociationIndicesRef rhs) {
                 return lhs[0] < rhs[0];
               });
    origOpToCollapsedOpIterationDim.resize(origNumLoops);
    for (auto foldedDims : llvm::enumerate(collapsedOpToOrigOpIterationDim)) {
      for (auto dim : enumerate(foldedDims.value()))
        origOpToCollapsedOpIterationDim[dim.value()] =
            std::make_pair<int64_t, unsigned>(foldedDims.index(), dim.index());
    }
    return success();
  }

  /// Return mapping from collapsed loop domain to original loop domain.
  ArrayRef<ReassociationIndices> getCollapsedOpToOrigOpMapping() const {
    return collapsedOpToOrigOpIterationDim;
  }

  /// Return mapping from original loop domain to collapsed loop domain. The
  /// mapping is a pair. First value is the dimension in the collapsed loop that
  /// the original loop is mapped to. Second is the relative position in folded
  /// list of this domain. For example if the original loop domain is 3D, and
  /// the collapsed loop domain is folding all of it, i.e.
  ///
  /// ```
  /// collapsedOpToOrigOpMapping = [[0, 1, 2] [3, 4]]`
  /// ```
  ///
  /// then
  ///
  /// ```
  ///  origOpToCollapsedOpMapping[0] = {0, 0};
  ///  origOpToCollapsedOpMapping[1] = {0, 1};
  ///  origOpToCollapsedOpMapping[2] = {0, 2};
  ///  origOpToCollapsedOpMapping[3] = {1, 0};
  ///  origOpToCollapsedOpMapping[4] = {1, 1};
  /// ```
  ///
  ArrayRef<std::pair<int64_t, unsigned>> getOrigOpToCollapsedOpMapping() const {
    return origOpToCollapsedOpIterationDim;
  }

  /// Return the collapsed op iteration domain rank.
  unsigned getCollapsedOpIterationRank() const {
    return collapsedOpToOrigOpIterationDim.size();
  }

private:
  /// Map from the iteration domain index in collapsed op to the iteration
  /// domain indices in the original op.
  SmallVector<ReassociationIndices> collapsedOpToOrigOpIterationDim;

  /// Map from iteration domain index in the original op to the iteration domain
  /// index in the collapsed op.
  SmallVector<std::pair<int64_t, unsigned>> origOpToCollapsedOpIterationDim;
};
} // namespace

/// Get the iterator types for the collapsed operation given the original
/// iterator types and collapsed dimensions.
static SmallVector<StringRef>
getCollapsedOpIteratorTypes(ArrayRef<Attribute> iteratorTypes,
                            const CollapsingInfo &collapsingInfo) {
  SmallVector<StringRef> collapsedIteratorTypes;
  for (ReassociationIndicesRef foldedIterDims :
       collapsingInfo.getCollapsedOpToOrigOpMapping()) {
    assert(!foldedIterDims.empty() &&
           "reassociation indices expected to have non-empty sets");
    // Just pick the iterator type of the first folded dim. Pre-condition checks
    // expected to have checked that iterator types of all folded dimensions are
    // the same.
    collapsedIteratorTypes.push_back(
        iteratorTypes[foldedIterDims[0]].cast<StringAttr>().getValue());
  }
  return collapsedIteratorTypes;
}

/// Compute the indexing map in the collapsed op that corresponds to the given
/// `indexingMap` of the original operation.
static AffineMap
getCollapsedOpIndexingMap(AffineMap indexingMap,
                          const CollapsingInfo &collapsingInfo) {
  MLIRContext *context = indexingMap.getContext();
  assert(indexingMap.isProjectedPermutation() &&
         "expected indexing map to be projected permutation");
  SmallVector<AffineExpr> resultExprs;
  auto origOpToCollapsedOpMapping =
      collapsingInfo.getOrigOpToCollapsedOpMapping();
  for (auto expr : indexingMap.getResults()) {
    unsigned dim = expr.cast<AffineDimExpr>().getPosition();
    // If the dim is not the first of the collapsed dim, do nothing.
    if (origOpToCollapsedOpMapping[dim].second != 0)
      continue;
    // The next n-dims are guaranteed to be collapsed. So just use the
    // iteration dimension of the collapsed op.
    resultExprs.push_back(
        getAffineDimExpr(origOpToCollapsedOpMapping[dim].first, context));
  }
  return AffineMap::get(collapsingInfo.getCollapsedOpIterationRank(), 0,
                        resultExprs, context);
}

/// Return the `reassociation` indices to use to collapse the operand when the
/// iteration space of a generic op is collapsed.
static SmallVector<ReassociationIndices>
getOperandReassociation(AffineMap indexingMap,
                        const CollapsingInfo &collapsingInfo) {
  unsigned counter = 0;
  SmallVector<ReassociationIndices> operandReassociation;
  auto origOpToCollapsedOpMapping =
      collapsingInfo.getOrigOpToCollapsedOpMapping();
  auto collapsedOpToOrigOpMapping =
      collapsingInfo.getCollapsedOpToOrigOpMapping();
  while (counter < indexingMap.getNumResults()) {
    unsigned dim =
        indexingMap.getResult(counter).cast<AffineDimExpr>().getPosition();
    if (origOpToCollapsedOpMapping[dim].second == 0) {
      // This is the start of a collapsed dimensions of the iteration that
      // is gauranteed to be preserved in the indexing map. The number of folded
      // dims is obtained from the collapsed op to original op mapping.
      unsigned numFoldedDims =
          collapsedOpToOrigOpMapping[origOpToCollapsedOpMapping[dim].first]
              .size();
      auto range = llvm::seq<unsigned>(counter, counter + numFoldedDims);
      operandReassociation.emplace_back(range.begin(), range.end());
      counter += numFoldedDims;
    }
  }
  return operandReassociation;
}

/// Get the new value to use for a given `OpOperand` in the collapsed operation.
static Value getCollapsedOpOperand(Location loc, GenericOp genericOp,
                                   OpOperand *opOperand,
                                   const CollapsingInfo &collapsingInfo,
                                   OpBuilder &builder) {
  AffineMap indexingMap = genericOp.getTiedIndexingMap(opOperand);
  SmallVector<ReassociationIndices> operandReassociation =
      getOperandReassociation(indexingMap, collapsingInfo);

  // If the number of entries in the reassocation for the operand is same as the
  // number of results of the indexing map, then nothing to do for this operand.
  Value operand = opOperand->get();
  if (operandReassociation.size() == indexingMap.getNumResults())
    return operand;

  // Insert a reshape to collapse the dimensions.
  auto reshapeOp = builder.create<tensor::CollapseShapeOp>(
      loc, operand, operandReassociation);
  return reshapeOp.getResult();
}

/// Modify the `linalg.index` operations in the original generic op, to its
/// value in the collapsed operation.
void generateCollapsedIndexingRegion(Location loc, Block *block,
                                     const CollapsingInfo &collapsingInfo,
                                     ValueRange loopRange,
                                     PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(block);

  // Collect all the original index ops.
  auto indexOps = llvm::to_vector(block->getOps<linalg::IndexOp>());

  // For each folded dimension list resolve the original induction variable
  // values in terms of the folded dimension induction variable.
  //   i_{folded} = (i_0 * d1 + i1) * d2 + i2.
  // can be inverted to
  //   i2 = i_{folded} % d2
  //   i1 = (i_{folded} / d2) % d1
  //   i0 = i_{folded} / (d1 * d2)
  llvm::DenseMap<unsigned, Value> indexReplacementVals;
  for (auto &foldedDims :
       enumerate(collapsingInfo.getCollapsedOpToOrigOpMapping())) {
    ReassociationIndicesRef foldedDimsRef(foldedDims.value());
    Value newIndexVal =
        rewriter.create<linalg::IndexOp>(loc, foldedDims.index());
    for (auto dim : llvm::reverse(foldedDimsRef.drop_front())) {
      indexReplacementVals[dim] =
          rewriter.create<arith::RemUIOp>(loc, newIndexVal, loopRange[dim]);
      newIndexVal =
          rewriter.create<arith::DivUIOp>(loc, newIndexVal, loopRange[dim]);
    }
    indexReplacementVals[foldedDims.value().front()] = newIndexVal;
  }

  for (auto indexOp : indexOps) {
    auto dim = indexOp.dim();
    rewriter.replaceOp(indexOp, indexReplacementVals[dim]);
  }
}

/// Implementation of fusion with reshape operation by collapsing dimensions.
static FailureOr<SmallVector<Value>> collapseGenericOpIterationDims(
    GenericOp genericOp, ArrayRef<ReassociationIndices> foldedIterationDims,
    OpOperand *fusableOpOperand, PatternRewriter &rewriter) {
  // Bail on trivial no-op cases.
  if (genericOp.getNumLoops() <= 1 || foldedIterationDims.empty() ||
      llvm::all_of(foldedIterationDims, [](ReassociationIndicesRef foldedDims) {
        return foldedDims.size() <= 1;
      }))
    return failure();

  CollapsingInfo collapsingInfo;
  if (failed(collapsingInfo.initialize(genericOp.getNumLoops(),
                                       foldedIterationDims))) {
    return rewriter.notifyMatchFailure(
        genericOp, "illegal to collapse specified dimensions");
  }

  // Get the iterator types for the operand.
  SmallVector<StringRef> iteratorTypes = getCollapsedOpIteratorTypes(
      genericOp.iterator_types().getValue(), collapsingInfo);

  // Get the indexing maps.
  auto indexingMaps = llvm::to_vector(
      llvm::map_range(genericOp.getIndexingMaps(), [&](AffineMap map) {
        return getCollapsedOpIndexingMap(map, collapsingInfo);
      }));

  Location loc = genericOp->getLoc();

  // Get the input operands.
  auto inputOperands = llvm::to_vector(
      llvm::map_range(genericOp.getInputOperands(), [&](OpOperand *opOperand) {
        return getCollapsedOpOperand(loc, genericOp, opOperand, collapsingInfo,
                                     rewriter);
      }));

  // Get the output operands and result types.
  SmallVector<Type> resultTypes;
  SmallVector<Value> outputOperands;
  resultTypes.reserve(genericOp.getNumOutputs());
  outputOperands.reserve(genericOp.getNumOutputs());
  for (OpOperand *output : genericOp.getOutputOperands()) {
    Value newOutput =
        getCollapsedOpOperand(loc, genericOp, output, collapsingInfo, rewriter);
    outputOperands.push_back(newOutput);
    resultTypes.push_back(newOutput.getType());
  }

  // Create the generic op.
  auto collapsedGenericOp = rewriter.create<linalg::GenericOp>(
      loc, resultTypes, inputOperands, outputOperands, indexingMaps,
      iteratorTypes, [](OpBuilder &builder, Location loc, ValueRange args) {});
  Block *origOpBlock = &genericOp->getRegion(0).front();
  Block *collapsedOpBlock = &collapsedGenericOp->getRegion(0).front();
  rewriter.mergeBlocks(origOpBlock, collapsedOpBlock,
                       collapsedOpBlock->getArguments());

  if (collapsedGenericOp.hasIndexSemantics()) {
    // Collect the loop range of the generic op.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(collapsedGenericOp);
    SmallVector<Range> loopRanges =
        cast<LinalgOp>(genericOp.getOperation())
            .createLoopRanges(rewriter, genericOp.getLoc());
    assert(llvm::all_of(loopRanges,
                        [](Range range) {
                          return matchPattern(range.offset, m_Zero()) &&
                                 matchPattern(range.stride, m_One());
                        }) &&
           "expected all loop ranges to have zero start and unit stride");
    SmallVector<Value> loopBound = llvm::to_vector(
        llvm::map_range(loopRanges, [](Range range) { return range.size; }));
    generateCollapsedIndexingRegion(loc,
                                    &collapsedGenericOp->getRegion(0).front(),
                                    collapsingInfo, loopBound, rewriter);
  }

  // Insert expanding reshape for the result to get back the original result
  // type.
  SmallVector<Value> results;
  for (const auto &originalResult : llvm::enumerate(genericOp->getResults())) {
    Value collapsedOpResult =
        collapsedGenericOp->getResult(originalResult.index());
    auto originalResultType =
        originalResult.value().getType().cast<ShapedType>();
    auto collapsedOpResultType = collapsedOpResult.getType().cast<ShapedType>();
    if (collapsedOpResultType.getRank() != originalResultType.getRank()) {
      AffineMap indexingMap =
          genericOp.getTiedIndexingMapForResult(originalResult.value());
      SmallVector<ReassociationIndices> reassociation =
          getOperandReassociation(indexingMap, collapsingInfo);
      Value result = rewriter.create<tensor::ExpandShapeOp>(
          loc, originalResultType, collapsedOpResult, reassociation);
      results.push_back(result);
    } else {
      results.push_back(collapsedOpResult);
    }
  }
  return results;
}

namespace {

/// Pattern to fuse a tensor.expand_shape op with its consumer generic op by
/// contracting dimensions of the loop.
class FoldWithProducerReshapeOpByCollapsing
    : public OpRewritePattern<GenericOp> {
public:
  FoldWithProducerReshapeOpByCollapsing(MLIRContext *context,
                                        ControlFusionFn foldReshapes,
                                        PatternBenefit benefit = 1)
      : OpRewritePattern<GenericOp>(context, benefit),
        controlFoldingReshapes(std::move(foldReshapes)) {}

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    for (OpOperand *opOperand : genericOp.getInputTensorOperands()) {
      tensor::ExpandShapeOp reshapeOp =
          opOperand->get().getDefiningOp<tensor::ExpandShapeOp>();
      if (!reshapeOp)
        continue;

      SmallVector<ReassociationIndices> collapsableIterationDims =
          getCollapsableIterationSpaceDims(genericOp, opOperand,
                                           reshapeOp.getReassociationIndices());
      if (collapsableIterationDims.empty() ||
          !controlFoldingReshapes(reshapeOp->getResult(0), *opOperand)) {
        continue;
      }

      Optional<SmallVector<Value>> replacements =
          collapseGenericOpIterationDims(genericOp, collapsableIterationDims,
                                         opOperand, rewriter);
      if (!replacements) {
        return rewriter.notifyMatchFailure(
            genericOp, "failed to do the fusion by collapsing transformation");
      }

      rewriter.replaceOp(genericOp, replacements.getValue());
      return success();
    }
    return failure();
  }

private:
  ControlFusionFn controlFoldingReshapes;
};
} // namespace

//===---------------------------------------------------------------------===//
// Methods and patterns to convert tensor.expand_shape -> linalg.generic
// into linalg.generic -> tensor.expand_shape, i.e. push the reshape down.
//===---------------------------------------------------------------------===//

// TODO(ravishankarm): This pattern is to be deprecated in favor of fusion by
// collapsing that provides a more general functionality. This pattern is very
// specific to a particular use case. The fusion by collapsing can provide the
// same control to clients using the control function there.

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

namespace {
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
} // namespace

//===---------------------------------------------------------------------===//
// Methods and patterns that fuse constants with linalg.generic operations.
//===---------------------------------------------------------------------===//

namespace {
/// Pattern to fold a generic op with a splat constant/scalar constant. Does not
/// handle cases where the constant is not single-valued.
class FoldScalarOrSplatConstant : public OpRewritePattern<GenericOp> {
public:
  FoldScalarOrSplatConstant(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<GenericOp>(context, benefit) {}

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
      if (!def || !resultValue || !isScalarOrSplatConstantOp(def))
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
};

} // namespace

//===---------------------------------------------------------------------===//
// Miscellaneous patterns that help fusion.
//===---------------------------------------------------------------------===//

namespace {
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

        // If outs is sparse, leave it to the sparse compiler.
        if (sparse_tensor::getSparseTensorEncoding(operandVal.getType()))
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

/// Fold linalg.fill into linalg.generic
struct FoldFillWithGenericOp : public OpRewritePattern<GenericOp> {
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!genericOp.hasTensorSemantics())
      return failure();
    bool fillFound = false;
    Block &payload = genericOp.region().front();
    for (OpOperand *opOperand : genericOp.getInputOperands()) {
      if (!genericOp.payloadUsesValueFromOperand(opOperand))
        continue;
      FillOp fillOp = opOperand->get().getDefiningOp<FillOp>();
      if (!fillOp)
        continue;
      fillFound = true;
      payload.getArgument(opOperand->getOperandNumber())
          .replaceAllUsesWith(fillOp.value());
    }
    return success(fillFound);
  }
};
} // namespace
//===---------------------------------------------------------------------===//
// Methods that add patterns described in this file to a pattern list.
//===---------------------------------------------------------------------===//

void mlir::linalg::populateFoldReshapeOpsByLinearizationPatterns(
    RewritePatternSet &patterns) {
  patterns.add<
      FoldProducerReshapeOpByLinearization<false, tensor::CollapseShapeOp>,
      FoldProducerReshapeOpByLinearization<false, tensor::ExpandShapeOp>,
      FoldConsumerReshapeOpByLinearization<false, tensor::CollapseShapeOp>>(
      patterns.getContext());
}

void mlir::linalg::populateFoldUnitDimsReshapeOpsByLinearizationPatterns(
    RewritePatternSet &patterns) {
  patterns
      .add<FoldProducerReshapeOpByLinearization<true, tensor::CollapseShapeOp>,
           FoldProducerReshapeOpByLinearization<true, tensor::ExpandShapeOp>,
           FoldConsumerReshapeOpByLinearization<true, tensor::CollapseShapeOp>>(
          patterns.getContext());
}

void mlir::linalg::populateFoldReshapeOpsByExpansionPatterns(
    RewritePatternSet &patterns,
    const ControlFusionFn &controlFoldingReshapes) {
  patterns.add<FoldReshapeWithGenericOpByExpansion>(patterns.getContext(),
                                                    controlFoldingReshapes);
  patterns.add<FoldWithProducerReshapeOpByExpansion>(patterns.getContext(),
                                                     controlFoldingReshapes);
}

void mlir::linalg::populateFoldReshapeOpsByCollapsingPatterns(
    RewritePatternSet &patterns,
    const ControlFusionFn &controlFoldingReshapes) {
  patterns.add<FoldWithProducerReshapeOpByCollapsing>(patterns.getContext(),
                                                      controlFoldingReshapes);
}

void mlir::linalg::populateElementwiseOpsFusionPatterns(
    RewritePatternSet &patterns,
    const ControlFusionFn &controlElementwiseOpsFusion) {
  auto *context = patterns.getContext();
  patterns.add<FuseElementwiseOps>(context, controlElementwiseOpsFusion);
  patterns.add<FoldFillWithGenericOp, FoldScalarOrSplatConstant,
               RemoveOutsDependency>(context);
}

void mlir::linalg::populatePushReshapeOpsPatterns(RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  patterns.add<PushExpandingReshape>(context);
}

//===---------------------------------------------------------------------===//
// Passes
//===---------------------------------------------------------------------===//

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

/// Pass that fuses generic ops on tensors. Used only for testing.
// TODO(ravishankarm): This pass is to be deprecated. The efficacy of the
// patterns added here heavily depends on the cost function used. Having an
// opinionated pass of this form is not recommended. Deprecate this pass in
// favor of test passes that check the functionality of each of the patterns
// added here individually.
struct LinalgElementwiseOpFusionPass
    : public LinalgElementwiseOpFusionBase<LinalgElementwiseOpFusionPass> {
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();
    RewritePatternSet patterns(context);

    // Add folding with reshape by expansion patterns.
    ControlFusionFn defaultControlFn = [](const OpResult &producer,
                                          const OpOperand &consumer) {
      return producer.hasOneUse();
    };

    // Add elementwise op fusion patterns.
    populateElementwiseOpsFusionPatterns(patterns, defaultControlFn);

    populateFoldReshapeOpsByExpansionPatterns(
        patterns,
        allowFoldingUnitDimReshapes ? defaultControlFn : skipUnitDimReshape);

    // Add the sparse tensor rewriting patterns.
    populateSparseTensorRewriting(patterns);

    // General canonicalization patterns.
    AffineApplyOp::getCanonicalizationPatterns(patterns, context);
    GenericOp::getCanonicalizationPatterns(patterns, context);
    tensor::ExpandShapeOp::getCanonicalizationPatterns(patterns, context);
    tensor::CollapseShapeOp::getCanonicalizationPatterns(patterns, context);
    context->getLoadedDialect<LinalgDialect>()->getCanonicalizationPatterns(
        patterns);

    // Add constant folding patterns.
    populateConstantFoldLinalgOperations(patterns, defaultControlFn);

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

} // namespace

std::unique_ptr<Pass> mlir::createLinalgElementwiseOpFusionPass() {
  return std::make_unique<LinalgElementwiseOpFusionPass>();
}

std::unique_ptr<Pass> mlir::createFoldReshapeOpsByLinearizationPass() {
  return std::make_unique<FoldReshapeOpsByLinearizationPass>();
}
