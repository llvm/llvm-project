//===- Fusion.cpp - Implementation of linalg Fusion -----------------------===//
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
#include "PassDetail.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::linalg;

/// Implementation of fusion of generic ops and indexed_generic ops.
// struct FuseGenericOpsOnTensors {
static bool areTensorOpsFusable(LinalgOp producer, LinalgOp consumer,
                                unsigned consumerIdx) {
  // Producer and consumer must have tensor semantics.
  if (!producer.hasTensorSemantics() || !consumer.hasTensorSemantics())
    return false;

  // Verify that
  // - the producer has all "parallel" iterator type.
  if (producer.getNumParallelLoops() != producer.getNumLoops())
    return false;

  // Get the consumer index map. The number of results of the consumer index
  // map must match the number of loops of the producer.
  AffineMap consumerIndexMap = consumer.getIndexingMap(consumerIdx);
  if (consumerIndexMap.getNumResults() != producer.getNumLoops())
    return false;

  // Finally the index_map for the result must be invertible. For now just
  // verify it is a permutation.
  AffineMap producerResultIndexMap = producer.getOutputIndexingMap(0);
  return producerResultIndexMap.isPermutation();
}

/// Append to `fusedOpIndexingMapAttrs` the indexing maps for the operands of
/// the `producer` to use in the fused operation given the indexing map of the
/// result of the producer in the consumer.
static void getIndexingMapOfProducerOperandsInFusedOp(
    LinalgOp producer, AffineMap fusedConsumerArgIndexMap,
    SmallVectorImpl<Attribute> &fusedOpIndexingMapAttrs) {
  // The indexing map in the consumer op (fusedConsumerArgIndexMap) is a map
  // from consumer loop -> consumer arg tensor index/producer result tensor
  // index. The fused loop is same as the consumer loop. For each producer arg
  // the indexing map to be computed is a map from consumer loop -> producer
  // arg tensor index.

  AffineMap producerResultIndexMap = producer.getOutputIndexingMap(0);
  // producerResultIndexMap is a map from producer loop -> tensor index.
  // Compute the inverse to get map from tensor index -> producer loop.
  // The inverse is a map from producer result tensor index -> producer loop.
  AffineMap invProducerResultIndexMap =
      inversePermutation(producerResultIndexMap);
  assert(invProducerResultIndexMap &&
         "expected producer result indexig map to be invertible");
  for (unsigned argNum : llvm::seq<unsigned>(0, producer.getNumInputs())) {
    // argMap is a map from producer loop -> producer arg tensor index.
    AffineMap argMap = producer.getInputIndexingMap(argNum);

    // Compose argMap with invProducerResultIndexMap to get a map from
    // producer result tensor index -> producer arg tensor index.
    AffineMap t1 = argMap.compose(invProducerResultIndexMap);

    // Compose t1 with fusedConsumerArgIndexMap gives an indexing map from
    // consumer loop/ fused loop -> producer arg tensor index.
    AffineMap indexingMap = t1.compose(fusedConsumerArgIndexMap);
    fusedOpIndexingMapAttrs.push_back(AffineMapAttr::get(indexingMap));
  }
}

/// Generate the region of the fused tensor operation. The region of the fused
/// op must be empty.
static void generateFusedTensorOpRegion(PatternRewriter &rewriter,
                                        Operation *fusedOp, LinalgOp producer,
                                        LinalgOp consumer,
                                        AffineMap consumerToProducerLoopsMap,
                                        unsigned consumerIdx, unsigned nloops) {
  // Build the region of the fused op.
  Block &producerBlock = producer.getOperation()->getRegion(0).front();
  Block &consumerBlock = consumer.getOperation()->getRegion(0).front();
  Block *fusedBlock = new Block();
  fusedOp->getRegion(0).push_back(fusedBlock);
  BlockAndValueMapping mapper;
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(fusedBlock);

  // The block arguments are
  // [index_0, index_1, ... ,
  //   consumer_operand_0, ... , consumer_operand_(`consumerIdx`-1),
  //   producer_operand_0, ... , producer_operand_(n-1)],
  //   consumer_operand_(`consumerIdx`), .. consumer_operand_(m-1)]
  // , where n is the number of producer's operand and m is the number
  // consumer's operand.
  // If both `numProducerIndices` and `numConsumerIndices` are zero, this is a
  // generic op. In this case, there are no indices in block arguments.
  unsigned numProducerIndices = isa<IndexedGenericOp>(producer.getOperation())
                                    ? producer.getNumLoops()
                                    : 0;
  unsigned numConsumerIndices = isa<IndexedGenericOp>(consumer.getOperation())
                                    ? consumer.getNumLoops()
                                    : 0;
  unsigned numFusedOpIndices =
      (isa<IndexedGenericOp>(producer.getOperation()) ||
       isa<IndexedGenericOp>(consumer.getOperation()))
          ? std::max(producer.getNumLoops(), consumer.getNumLoops())
          : 0;
  // Firstly, add all the indices to the block arguments.
  for (unsigned i = 0, e = numFusedOpIndices; i < e; ++i)
    fusedBlock->addArgument(rewriter.getIndexType());
  // Map the arguments for the unmodified args from the consumer.
  for (auto consumerArg : llvm::enumerate(consumerBlock.getArguments())) {
    if (consumerArg.index() == consumerIdx + numConsumerIndices) {
      // Map the arguments for the args from the producer.
      for (auto producerArg : llvm::enumerate(producerBlock.getArguments())) {
        // If producer is an indexed_generic op, map the indices from consumer
        // loop to producer loop (because the fusedOp is built based on
        // consumer's perspective).
        if (producerArg.index() < numProducerIndices) {
          auto newIndex = rewriter.create<mlir::AffineApplyOp>(
              producer.getLoc(),
              consumerToProducerLoopsMap.getSubMap(producerArg.index()),
              fusedBlock->getArguments().take_front(numFusedOpIndices));
          mapper.map(producerArg.value(), newIndex);
        } else {
          mapper.map(producerArg.value(),
                     fusedBlock->addArgument(producerArg.value().getType()));
        }
      }
      continue;
    }

    // If consumer is an indexed_generic op, map the indices to the block
    // arguments directly. Otherwise, add the same type of argument and map to
    // it.
    if (consumerArg.index() < numConsumerIndices) {
      mapper.map(consumerArg.value(),
                 fusedBlock->getArgument(consumerArg.index()));
    } else {
      mapper.map(consumerArg.value(),
                 fusedBlock->addArgument(consumerArg.value().getType()));
    }
  }

  // Add operations from producer (except the yield operation) to the fused
  // op.
  for (auto &op : producerBlock.getOperations()) {
    if (auto yieldOp = dyn_cast<linalg::YieldOp>(op)) {
      // Lookup the value the yield operation is mapped to.
      Value yieldVal = yieldOp.getOperand(0);
      if (Value clonedVal = mapper.lookupOrNull(yieldVal))
        mapper.map(consumerBlock.getArgument(consumerIdx + numConsumerIndices),
                   clonedVal);
      continue;
    }
    rewriter.clone(op, mapper);
  }
  for (auto &op : consumerBlock.getOperations())
    rewriter.clone(op, mapper);
}

static Optional<SmallVector<Value, 1>>
fuseTensorOpsImpl(LinalgOp producer, LinalgOp consumer, unsigned consumerIdx,
                  PatternRewriter &rewriter,
                  OperationFolder *folder = nullptr) {
  if (!areTensorOpsFusable(producer, consumer, consumerIdx))
    return llvm::None;

  unsigned numFusedOperands =
      producer.getNumInputs() + consumer.getNumInputs() - 1;

  // Compute the fused operands list,
  SmallVector<Value, 2> fusedOperands;
  fusedOperands.reserve(numFusedOperands);
  auto consumerOperands = consumer.getInputs();
  auto producerOperands = producer.getInputs();
  fusedOperands.assign(consumerOperands.begin(),
                       std::next(consumerOperands.begin(), consumerIdx));
  fusedOperands.append(producerOperands.begin(), producerOperands.end());
  fusedOperands.append(std::next(consumerOperands.begin(), consumerIdx + 1),
                       consumerOperands.end());

  // Compute indexing_maps for the fused operation. The indexing_maps for the
  // operands of the consumers that arent fused are the same. The
  // indexing_maps for the producers need to be computed based on the
  // indexing_map of the operand at consumerIdx in the consumer.
  SmallVector<Attribute, 4> fusedIndexMaps;
  auto consumerIndexMaps = consumer.indexing_maps();
  fusedIndexMaps.reserve(fusedOperands.size() + consumer.getNumOutputs());
  fusedIndexMaps.assign(consumerIndexMaps.begin(),
                        std::next(consumerIndexMaps.begin(), consumerIdx));
  // Compute indexing maps for the producer args in the fused operation.
  getIndexingMapOfProducerOperandsInFusedOp(
      producer, consumer.getInputIndexingMap(consumerIdx), fusedIndexMaps);

  // Append the indexing maps for the remaining consumer operands.
  fusedIndexMaps.append(std::next(consumerIndexMaps.begin(), consumerIdx + 1),
                        consumerIndexMaps.end());

  // Generate the fused op.
  // Tensor-level fusion is only on ops without initTensors and outputBuffers.
  LinalgOp fusedOp;
  if (isa<GenericOp>(producer.getOperation()) &&
      isa<GenericOp>(consumer.getOperation())) {
    fusedOp = rewriter
                  .create<GenericOp>(consumer.getLoc(),
                                     consumer.getOperation()->getResultTypes(),
                                     /*inputs=*/fusedOperands,
                                     /*outputBuffers=*/ValueRange{},
                                     /*initTensors=*/ValueRange{},
                                     rewriter.getArrayAttr(fusedIndexMaps),
                                     consumer.iterator_types(),
                                     /*doc=*/nullptr,
                                     /*library_call=*/nullptr,
                                     /*symbol_source=*/nullptr)
                  .getOperation();
  } else {
    fusedOp =
        rewriter
            .create<IndexedGenericOp>(consumer.getLoc(),
                                      consumer.getOperation()->getResultTypes(),
                                      /*inputs=*/fusedOperands,
                                      /*outputBuffers=*/ValueRange{},
                                      /*initTensors=*/ValueRange{},
                                      rewriter.getArrayAttr(fusedIndexMaps),
                                      consumer.iterator_types(),
                                      /*doc=*/nullptr,
                                      /*library_call=*/nullptr,
                                      /*symbol_source=*/nullptr)
            .getOperation();
  }

  // Construct an AffineMap from consumer loops to producer loops.
  // consumer loop -> tensor index
  AffineMap consumerResultIndexMap = consumer.getInputIndexingMap(consumerIdx);
  // producer loop -> tensor index
  AffineMap producerResultIndexMap = producer.getOutputIndexingMap(0);
  // tensor index -> producer loop
  AffineMap invProducerResultIndexMap =
      inversePermutation(producerResultIndexMap);
  assert(invProducerResultIndexMap &&
         "expected producer result indexig map to be invertible");
  // consumer loop -> producer loop
  AffineMap consumerToProducerLoopsMap =
      invProducerResultIndexMap.compose(consumerResultIndexMap);

  generateFusedTensorOpRegion(rewriter, fusedOp.getOperation(), producer,
                              consumer, consumerToProducerLoopsMap, consumerIdx,
                              consumer.getNumLoops());
  return SmallVector<Value, 1>(fusedOp.getOperation()->getResults());
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
/// %1 = linalg.tensor_reshape %0 [affine_map<(i, j, k, l) -> (i)>,
///                                affine_map<(i, j, k, l) -> (j, k, l)>] :
///        tensor<?x?x4x5xf32> into tensor<?x?xf32>
///
/// would be rewritten into:
/// %0 = op ... : tensor<?x?x4x5xf32>
/// with output index_map
///   `affine_map<(d0, d1, d2, d3) -> (d0, d1 * 20 + d2 * 5 + d3)>`
static AffineMap linearizeCollapsedDims(AffineMap sourceMap,
                                        ArrayRef<int64_t> sourceShape,
                                        ArrayRef<AffineMap> reassociationMaps) {
  SmallVector<AffineExpr, 4> resultExprs;
  resultExprs.reserve(reassociationMaps.size());
  ArrayRef<AffineExpr> sourceExprs = sourceMap.getResults();
  MLIRContext *context = sourceMap.getContext();

  // Compute the result exprs based on the reassociation maps.
  for (AffineMap map : reassociationMaps) {
    ArrayRef<AffineExpr> collapsedDims = map.getResults();
    // Assume that they are in-order and contiguous (already checked in
    // verifier).
    assert(!collapsedDims.empty());
    unsigned startDim =
        collapsedDims.front().cast<AffineDimExpr>().getPosition();
    AffineExpr linearizedExpr = makeCanonicalStridedLayoutExpr(
        sourceShape.slice(startDim, collapsedDims.size()),
        sourceExprs.slice(startDim, collapsedDims.size()), context);
    resultExprs.push_back(linearizedExpr);
  }
  return AffineMap::get(sourceMap.getNumDims(), sourceMap.getNumSymbols(),
                        resultExprs, context);
}

/// Checks if the `reshapeOp` can be fused with it consumer (if `asProducer` is
/// true) or its producer (if `asProducer` is false) given the indexing map at
/// its use.
static bool isTensorReshapeOpFoldableByLinearization(TensorReshapeOp reshapeOp,
                                                     AffineMap useIndexMap,
                                                     bool asProducer) {
  RankedTensorType returnType = reshapeOp.getResultType();
  RankedTensorType operandType = reshapeOp.getSrcType();
  // Reshape is fusable with its consumer (i.e. reshape as a producer) when its
  // operand is of lesser rank than the result. Fusing when operand has higher
  // rank will require use of mods and divs in the indexing maps of the fused op
  // which would make it non-invertible. Similarly reshape is fused with its
  // producer (i.e. reshape as consumer) only if the return type has lesser
  // rank.
  if ((asProducer && reshapeOp.getSrcType().hasStaticShape() &&
       returnType.getRank() < operandType.getRank()) ||
      (!asProducer && reshapeOp.getResultType().hasStaticShape() &&
       operandType.getRank() < returnType.getRank()))
    return false;
  return useIndexMap.isPermutation();
}

/// Based on the type of `op` create a linalg op of the same type, i.e. if `op`
/// is a linalg.generic operation, the create a `linalg.generic` operation with
/// the given `args`. Expects `op` to be `linalg.generic` or
/// `linalg.indexed_generic`.
template <typename... Args>
static LinalgOp createLinalgOpOfSameType(LinalgOp op, PatternRewriter &rewriter,
                                         Args... args) {
  if (isa<GenericOp>(op.getOperation()))
    return cast<LinalgOp>(rewriter.create<GenericOp>(args...).getOperation());
  if (isa<IndexedGenericOp>(op.getOperation()))
    return cast<LinalgOp>(
        rewriter.create<IndexedGenericOp>(args...).getOperation());
  llvm_unreachable(
      "expected only linalg.generic or linalg.indexed_generic ops");
  return nullptr;
}

/// Conditions for folding a generic/indexed-generic operation with a reshape op
/// by expanding the iteration space dimensionality for tensor operations. These
/// are preconditions assumed by `foldReshapeByDimExpansion` which implements
/// the following fusion pattern.
///
///  Consider
///
///  %c = linalg.generic ins(%a, %b : memref<?x?x?xf32>, memref<?x?xf32>)
///         indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d0, d2)>,
///                          affine_map<(d0, d1, d2) -> (d1, d2)>,
///                          affine_map<(d0, d1, d2) -> (d0, d2, d1)>]
///  %d = linalg.tensor_reshape %c
///         [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1)>,
///          affine_map<(d0, d1, d2, d3, d4, d5) -> (d2)>,
///          affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>]
///       : tensor<?x?x?xf32> into tensor<?x?x?x?x?x?xf32>
///
///  The reshape can be folded into the `linalgOp` if the
///  generic/indexed-generic op loop dimensionality is increased to match the
///  result (operand) of the tensor_reshape when the reshape is expanding
///  (folding). The indexing_map of the fused tensor in the `linalgOp` and the
///  reassociation map helps compute the indexing maps of the modified op. For
///  the above example, based on the reassociation map it can be concluded that
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
///  %0 = linalg.tensor_reshape %a
///         [affine_map<(e0, e1, e2, e3, e4, e5) -> (e0, e1, e2),
///          affine_map<(e0, e1, e2, e3, e4, e5) -> (e3, e4),
///          affine_map<(e0, e1, e2, e3, e4, e5) -> (e5)]
///       : tensor<?x?x?xf32> into tensor<?x?x?x?x?x?xf32>
///  %1 = linalg.tensor_reshape %b
///         [affine_map<(e0, e1, e2, e3) -> (e0, e1, e2),
///          affine_map<(e0, e1, e2, e3) -> (e3)]
///       : tensor<?x?x?xf32> into tensor<?x?x?x?xf32>
///
///  The added reshapes are again expanding patterns, so they will get fused
///  with its producers if possible.
static bool isFusableWithReshapeByDimExpansion(LinalgOp linalgOp,
                                               unsigned fusedTensorIndex) {
  // Is fusable only if:
  // - The linalgOp is a generic op, or an indexed_generic.
  // - All the indexing maps for operands in linalgOp are projected
  //   permutations.
  // - The indexing map at the position representing the fused tensor is a
  //   permutation.
  // - All the loops in linalgOp are parallel loops.
  return isa<GenericOp, IndexedGenericOp>(linalgOp.getOperation()) &&
         linalgOp.hasTensorSemantics() &&
         llvm::all_of(linalgOp.indexing_maps().getValue().take_front(
                          linalgOp.getNumInputs()),
                      [](Attribute attr) {
                        return attr.cast<AffineMapAttr>()
                            .getValue()
                            .isProjectedPermutation();
                      }) &&
         linalgOp.getIndexingMap(fusedTensorIndex).isPermutation() &&
         llvm::all_of(linalgOp.iterator_types(), [](Attribute attr) {
           return attr.cast<StringAttr>().getValue() ==
                  getParallelIteratorTypeName();
         });
}

/// Implements the fusion of a tensor_reshape op and a generic/indexed_generic
/// op as explained in `isFusableWithReshapeByExpansion`. Assumes that those
/// conditions have been satisfied.
static Optional<SmallVector<Value, 1>>
fuseWithReshapeByExpansion(LinalgOp linalgOp, TensorReshapeOp reshapeOp,
                           unsigned fusedTensorIndex, PatternRewriter &rewriter,
                           OperationFolder *folder = nullptr) {
  assert(isFusableWithReshapeByDimExpansion(linalgOp, fusedTensorIndex) &&
         "preconditions for fuse operation failed");
  // Check if reshape is expanding or collapsing.
  bool isExpanding =
      reshapeOp.getSrcType().getRank() < reshapeOp.getResultType().getRank();
  RankedTensorType expandedType =
      isExpanding ? reshapeOp.getResultType() : reshapeOp.getSrcType();
  RankedTensorType foldedType =
      isExpanding ? reshapeOp.getSrcType() : reshapeOp.getResultType();
  AffineMap fusedIndexMap = linalgOp.getIndexingMap(fusedTensorIndex);

  // The reshape is folding/expanding consecutive dimensions. Given the indexing
  // map of the fused tensor find the number of dimensions each of the loops of
  // the original op is expanded into. Also record the shape of the expanded
  // dimensions.
  ArrayRef<int64_t> expandedShape = expandedType.getShape();
  SmallVector<unsigned, 4> numFoldedDims(foldedType.getRank(), 0);
  SmallVector<SmallVector<int64_t, 4>, 4> expandedDimsShape(
      foldedType.getRank());
  auto reassociationMaps = reshapeOp.getReassociationMaps();
  for (auto resultExpr : llvm::enumerate(fusedIndexMap.getResults())) {
    unsigned pos = resultExpr.value().cast<AffineDimExpr>().getPosition();
    AffineMap foldedDims = reassociationMaps[resultExpr.index()];
    numFoldedDims[pos] = foldedDims.getNumResults();
    ArrayRef<int64_t> shape = expandedShape.slice(
        foldedDims.getResult(0).cast<AffineDimExpr>().getPosition(),
        numFoldedDims[pos]);
    expandedDimsShape[pos].assign(shape.begin(), shape.end());
  }

  if (isa<IndexedGenericOp>(linalgOp.getOperation())) {
    // For indexed generic op, the region contains arguments that represent the
    // induction variable value of the loops. In the fused op these values are
    // obtained by linearizing the expanded dimensions. For now just check that
    // the extents used in the linearization (all the expanded dims except the
    // front) are statically know. For dynamic case, we would need shape
    // information on these dimensions to get these.
    for (auto &expandedShape : expandedDimsShape) {
      for (int64_t expandedDimShape : llvm::make_range(
               std::next(expandedShape.begin()), expandedShape.end())) {
        if (ShapedType::isDynamic(expandedDimShape)) {
          linalgOp.emitError(
              "unable to fuse indexed generic op where the expanded dim is "
              "dynamic");
          return llvm::None;
        }
      }
    }
  }

  // The remapping of the indices is then the prefix sum (inclusive) of the
  // numFoldedDims.
  SmallVector<unsigned, 4> remapping(numFoldedDims.size() + 1, 0);
  unsigned sum = 0;
  for (auto numFoldedDim : llvm::enumerate(numFoldedDims)) {
    sum += numFoldedDim.value();
    remapping[numFoldedDim.index() + 1] = sum;
  }

  SmallVector<AffineMap, 4> expandedOpIndexingMaps;
  // Compute the modified indexing maps by replacing every loop (AffineDimExpr)
  // in the original indexing map with the sequence of loops that it is expanded
  // to.
  for (AffineMap indexingMap : linalgOp.getIndexingMaps()) {
    SmallVector<AffineExpr, 4> newExprs;
    for (AffineExpr expr : indexingMap.getResults()) {
      unsigned pos = expr.cast<AffineDimExpr>().getPosition();
      for (unsigned newPos :
           llvm::seq<unsigned>(remapping[pos], remapping[pos + 1])) {
        newExprs.push_back(rewriter.getAffineDimExpr(newPos));
      }
    }
    expandedOpIndexingMaps.push_back(
        AffineMap::get(remapping.back(), indexingMap.getNumSymbols(), newExprs,
                       rewriter.getContext()));
  }

  // The operands of the expanded op are computed by reshaping the original
  // operands. The reshape depends on the ordering of the loop used to access
  // the tensor in the original operation, and are expanded into as many
  // dimensions as the loop is expanded into (as computed by `remapping`).
  auto getReshapeInfo =
      [&](AffineMap operandIndexingMap,
          SmallVectorImpl<ReassociationIndices> &reassociation,
          SmallVectorImpl<int64_t> &expandedOpOperandShape) {
        unsigned reshapeDims = 0;
        for (AffineExpr expr : operandIndexingMap.getResults()) {
          unsigned origDim = expr.cast<AffineDimExpr>().getPosition();
          auto foldedDims = llvm::seq<int64_t>(
              reshapeDims, reshapeDims + numFoldedDims[origDim]);
          reassociation.emplace_back(foldedDims.begin(), foldedDims.end());
          expandedOpOperandShape.append(expandedDimsShape[origDim].begin(),
                                        expandedDimsShape[origDim].end());
          reshapeDims += numFoldedDims[origDim];
        }
      };
  SmallVector<Value, 4> expandedOpOperands;
  for (auto operand : llvm::enumerate(linalgOp.getInputs())) {
    if (operand.index() == fusedTensorIndex) {
      expandedOpOperands.push_back(reshapeOp.src());
      continue;
    }
    AffineMap indexingMap = linalgOp.getIndexingMap(operand.index());
    SmallVector<ReassociationIndices, 4> reassociation;
    SmallVector<int64_t, 4> expandedOperandShape;
    getReshapeInfo(indexingMap, reassociation, expandedOperandShape);
    Type expandedOperandType = RankedTensorType::get(
        expandedOperandShape,
        operand.value().getType().cast<ShapedType>().getElementType());
    if (expandedOperandType != operand.value().getType()) {
      expandedOpOperands.push_back(rewriter.create<TensorReshapeOp>(
          linalgOp.getLoc(), expandedOperandType, operand.value(),
          reassociation));
    } else {
      expandedOpOperands.push_back(operand.value());
    }
  }
  SmallVector<Type, 1> resultTypes;
  SmallVector<SmallVector<ReassociationIndices, 4>, 1> resultReassociation;
  for (auto result : llvm::enumerate(linalgOp.getOperation()->getResults())) {
    AffineMap indexingMap =
        linalgOp.getIndexingMap(linalgOp.getNumInputs() + result.index());
    SmallVector<ReassociationIndices, 4> reassociation;
    SmallVector<int64_t, 4> expandedResultShape;
    getReshapeInfo(indexingMap, reassociation, expandedResultShape);
    resultTypes.push_back(RankedTensorType::get(
        expandedResultShape,
        result.value().getType().cast<ShapedType>().getElementType()));
    resultReassociation.emplace_back(std::move(reassociation));
  }

  // The iterator types of the expanded op are all parallel.
  SmallVector<StringRef, 4> iteratorTypes(remapping.back(),
                                          getParallelIteratorTypeName());

  LinalgOp fusedOp = createLinalgOpOfSameType(
      linalgOp, rewriter, linalgOp.getLoc(), resultTypes,
      /*inputs=*/expandedOpOperands,
      /*outputBuffers=*/ValueRange{},
      /*initTensors=*/ValueRange{}, expandedOpIndexingMaps, iteratorTypes);
  Region &fusedRegion = fusedOp.getOperation()->getRegion(0);
  Region &originalRegion = linalgOp.getOperation()->getRegion(0);

  if (isa<GenericOp>(linalgOp.getOperation())) {
    rewriter.cloneRegionBefore(originalRegion, fusedRegion,
                               fusedRegion.begin());
  } else {
    assert(isa<IndexedGenericOp>(linalgOp.getOperation()));
    // Create an entry block in the fused Region with same number of arguments
    // as the fused op
    Block *fusedEntryBlock = new Block;
    fusedRegion.push_back(fusedEntryBlock);
    rewriter.cloneRegionBefore(originalRegion, fusedRegion, fusedRegion.end());

    // Merge the entry block of the fused op with the cloned blocks. For this
    // compute the value for arguments of the region in the original operation
    // in terms of the arguments of the fused op. Since the original operation
    // is expanded, the expanded dimensions need to be folded back to get the
    // replacement value for the arguments corresponding to interation index.
    // For now this expects that all the loop ranges are constants, which is
    // true if the shapes are all static. This has already been checked in the
    // precondition.
    using namespace edsc::op;
    using namespace edsc::intrinsics;
    OpBuilder::InsertionGuard guard(rewriter);
    SmallVector<Value, 4> argReplacements(originalRegion.getNumArguments());
    rewriter.setInsertionPointToStart(fusedEntryBlock);
    edsc::ScopedContext scopedContext(rewriter, fusedOp.getLoc());
    IndexType indexType = rewriter.getIndexType();
    for (unsigned i : llvm::seq<unsigned>(0, numFoldedDims.size())) {
      Value linearizedIndex = fusedEntryBlock->addArgument(indexType);
      for (unsigned foldedDim = remapping[i] + 1; foldedDim != remapping[i + 1];
           foldedDim++) {
        int64_t expandedDimExtent =
            expandedDimsShape[i][foldedDim - remapping[i]];
        assert(!ShapedType::isDynamic(expandedDimExtent));
        linearizedIndex =
            linearizedIndex * std_constant_index(expandedDimExtent);
        linearizedIndex =
            linearizedIndex + fusedEntryBlock->addArgument(indexType);
      }
      argReplacements[i] = linearizedIndex;
    }
    for (unsigned i :
         llvm::seq<unsigned>(numFoldedDims.size(), argReplacements.size())) {
      argReplacements[i] =
          fusedEntryBlock->addArgument(originalRegion.getArgument(i).getType());
    }
    rewriter.mergeBlocks(fusedEntryBlock->getNextNode(), fusedEntryBlock,
                         argReplacements);
  }

  // Reshape the result values to their original shape if this is a collapsing
  // reshape folded into its consumer.
  SmallVector<Value, 1> resultVals;
  for (auto result : llvm::enumerate(linalgOp.getOperation()->getResults())) {
    if (!isExpanding &&
        resultTypes[result.index()] != result.value().getType()) {
      resultVals.push_back(rewriter.create<TensorReshapeOp>(
          linalgOp.getLoc(), result.value().getType(),
          fusedOp.getOperation()->getResult(result.index()),
          resultReassociation[result.index()]));
    } else {
      resultVals.push_back(fusedOp.getOperation()->getResult(result.index()));
    }
  }
  // Assuming a single result.
  return resultVals;
}

namespace {

/// Pattern to fold tensor_reshape op with its consumer by using the source of
/// the reshape op as the operand in the consumer (instead of the result of the
/// tensor_reshapeop) when the tensor_reshape op is collapsing. The
/// corresponding index map in the consumer needs to be modified to linearize
/// the folded dimension.
///
/// For example,
///
/// #map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
/// %0 = linalg.tensor_reshape %arg0
///        [affine_map<(i, j, k, l) -> (i)>, affine_map<(i, j, k, l) -> (j, k)>,
///         affine_map<(i, j, k, l) -> (l)>]
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
template <typename LinalgOpTy>
struct FoldProducerReshapeOpByLinearization
    : public OpRewritePattern<LinalgOpTy> {
  using OpRewritePattern<LinalgOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(LinalgOpTy op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasTensorSemantics())
      return failure();
    LinalgOp linalgOp = cast<LinalgOp>(op.getOperation());
    for (auto operand : llvm::enumerate(linalgOp.getInputs())) {
      TensorReshapeOp reshapeOp =
          operand.value().getDefiningOp<TensorReshapeOp>();
      if (!reshapeOp ||
          !isTensorReshapeOpFoldableByLinearization(
              reshapeOp, linalgOp.getInputIndexingMap(operand.index()),
              /*asProducer =*/true))
        continue;

      // Compute the fused operands list,
      SmallVector<Value, 2> fusedOperands(linalgOp.getInputs());
      fusedOperands[operand.index()] = reshapeOp.src();

      // Compute indexing_maps for the fused operation. The indexing_maps for
      // the operands of the consumers that arent fused are the same.
      SmallVector<AffineMap, 4> fusedIndexMaps = llvm::to_vector<4>(
          op.indexing_maps().template getAsValueRange<AffineMapAttr>());

      // Accepted consumer maps are either identity or permutation.
      auto invMap = inversePermutation(fusedIndexMaps[operand.index()]);

      // Compute the indexing map to use for the result of the producer.
      AffineMap modifiedMap =
          linearizeCollapsedDims(invMap, reshapeOp.getResultType().getShape(),
                                 reshapeOp.getReassociationMaps());
      for (AffineExpr expr : modifiedMap.getResults()) {
        if (!expr.isPureAffine())
          return failure();
      }
      fusedIndexMaps[operand.index()] = modifiedMap;

      // Further check that the resulting index maps can be fused and
      // inverted. Without this the resultant op is not legal.
      if (!inversePermutation(concatAffineMaps(fusedIndexMaps)))
        return op.emitRemark("fused op loop bound computation failed");

      rewriter.startRootUpdate(op);
      op.getOperation()->setOperands(fusedOperands);
      op.indexing_mapsAttr(rewriter.getAffineMapArrayAttr(fusedIndexMaps));
      rewriter.finalizeRootUpdate(op);
      if (reshapeOp.use_empty())
        rewriter.eraseOp(reshapeOp);
      return success();
    }
    return op.emitRemark("no fusion candidates found");
  }
};

/// Pattern to fuse a tensor_reshape op with its consumer
/// generic/indexed_generic op, when the reshape op is collapsing
/// dimensions. The dimensionality of the loop in the consumer is expanded.
template <typename GenericOpTy>
struct FoldWithProducerReshapeOpByExpansion
    : public OpRewritePattern<GenericOpTy> {
  using OpRewritePattern<GenericOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOpTy genericOp,
                                PatternRewriter &rewriter) const override {
    LinalgOp linalgOp = cast<LinalgOp>(genericOp.getOperation());
    for (auto operand : llvm::enumerate(linalgOp.getInputs())) {
      TensorReshapeOp reshapeOp =
          operand.value().getDefiningOp<TensorReshapeOp>();
      if (!reshapeOp)
        continue;

      // Fold only if
      // - The tensor reshape op is folding.
      // - All constraints of fusing with reshape by expansion are met.
      if (reshapeOp.getSrcType().getRank() <
              reshapeOp.getResultType().getRank() ||
          !isFusableWithReshapeByDimExpansion(linalgOp, operand.index()))
        continue;

      Optional<SmallVector<Value, 1>> replacementValues =
          fuseWithReshapeByExpansion(linalgOp, reshapeOp, operand.index(),
                                     rewriter);
      if (!replacementValues)
        return failure();
      rewriter.replaceOp(genericOp, replacementValues.getValue());
      if (reshapeOp.use_empty())
        rewriter.eraseOp(reshapeOp);
      return success();
    }
    return failure();
  }
};

/// Pattern to fold tensor_reshape op with its producer. The corresponding index
/// map in the consumer needs to be modified to linearize the folded dimension.
struct FoldConsumerReshapeOpByLinearization
    : public OpRewritePattern<TensorReshapeOp> {
  using OpRewritePattern<TensorReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TensorReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    LinalgOp producer = reshapeOp.src().getDefiningOp<LinalgOp>();
    if (!producer ||
        !isa<GenericOp, IndexedGenericOp>(producer.getOperation()) ||
        !producer.hasTensorSemantics() || producer.getNumOutputs() != 1 ||
        !isTensorReshapeOpFoldableByLinearization(
            reshapeOp, producer.getOutputIndexingMap(0), /*asProducer =*/false))
      return failure();
    // The indexing_maps for the operands of the fused operation are same as
    // those for the operands of the producer.
    SmallVector<AffineMap, 4> fusedIndexMaps = llvm::to_vector<4>(
        producer.indexing_maps().getAsValueRange<AffineMapAttr>());

    auto invMap = inversePermutation(producer.getOutputIndexingMap(0));

    // Compute the indexing map to use for the operand of the producer.
    AffineMap modifiedMap =
        linearizeCollapsedDims(invMap, reshapeOp.getSrcType().getShape(),
                               reshapeOp.getReassociationMaps());
    for (AffineExpr expr : modifiedMap.getResults()) {
      if (!expr.isPureAffine())
        return reshapeOp.emitRemark("fused op indexing map is not affine");
    }
    fusedIndexMaps.back() = modifiedMap;

    // Further check that the resulting index maps can be fused and
    // inverted. Without this the resultant op is not legal.
    if (!inversePermutation(concatAffineMaps(fusedIndexMaps)))
      return reshapeOp.emitRemark("fused op loop bound computation failed");

    LinalgOp fusedOp = createLinalgOpOfSameType(
        producer, rewriter, rewriter.getUnknownLoc(), reshapeOp.getResultType(),
        /*inputs=*/producer.getInputs(),
        /*outputBuffers=*/ValueRange{},
        /*initTensors=*/ValueRange{}, // no init tensors for now.
        rewriter.getAffineMapArrayAttr(fusedIndexMaps),
        producer.iterator_types(),
        /*doc=*/nullptr,
        /*library_call=*/nullptr,
        /*symbol_source=*/nullptr);
    auto &fusedRegion = fusedOp.getOperation()->getRegion(0);
    rewriter.cloneRegionBefore(producer.getOperation()->getRegion(0),
                               fusedRegion, fusedRegion.begin());
    rewriter.replaceOp(reshapeOp, fusedOp.getOperation()->getResults());
    if (producer.use_empty())
      rewriter.eraseOp(producer);
    return success();
  }
};

/// Pattern to fold a tensor_reshape op with its producer generic op if the
/// tensor_reshape op is expanding, by expanding the dimensionality of the loop
/// in the producer op.
struct FoldReshapeWithGenericOpByExpansion
    : public OpRewritePattern<TensorReshapeOp> {
  using OpRewritePattern<TensorReshapeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TensorReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    // Fold only if
    // - The tensor reshape op is a expanding case.
    // - All constraints of fusing with reshape by expansion are met.
    if (reshapeOp.getSrcType().getRank() > reshapeOp.getResultType().getRank())
      return failure();
    LinalgOp producer = reshapeOp.src().getDefiningOp<LinalgOp>();
    if (!producer || producer.getNumOutputs() != 1 ||
        !isFusableWithReshapeByDimExpansion(producer, producer.getNumInputs()))
      return failure();
    Optional<SmallVector<Value, 1>> replacementValues =
        fuseWithReshapeByExpansion(producer, reshapeOp, producer.getNumInputs(),
                                   rewriter);
    if (!replacementValues)
      return failure();
    rewriter.replaceOp(reshapeOp, replacementValues.getValue());
    if (producer.use_empty())
      rewriter.eraseOp(producer);
    return success();
  }
};

/// Pattern to fold a GenericOp/IndexedGenericOp with a splat constant.
template <typename LinalgOpTy>
struct FoldSplatConstants : public OpRewritePattern<LinalgOpTy> {
  using OpRewritePattern<LinalgOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(LinalgOpTy op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasTensorSemantics())
      return failure();
    LinalgOp linalgOp = cast<LinalgOp>(op.getOperation());
    for (auto operand : llvm::enumerate(linalgOp.getInputs())) {
      ConstantOp constantOp = operand.value().getDefiningOp<ConstantOp>();
      if (!constantOp ||
          !constantOp.value().cast<DenseElementsAttr>().isSplat())
        continue;

      // The indexing_maps for the operands of the fused operation are same as
      // those for the operands of the linalgOp without the indexing map at
      // operand.index()
      SmallVector<AffineMap, 4> fusedIndexMaps = llvm::to_vector<4>(
          linalgOp.indexing_maps().getAsValueRange<AffineMapAttr>());
      fusedIndexMaps.erase(std::next(fusedIndexMaps.begin(), operand.index()));

      // The operands list is same as the linalgOp with the argument for
      // constant index dropped.
      SmallVector<Value, 4> fusedOperands(linalgOp.getInputs());
      fusedOperands.erase(std::next(fusedOperands.begin(), operand.index()));

      // Create a constant scalar value from the splat constant.
      Value scalarConstant = rewriter.create<ConstantOp>(
          constantOp.getLoc(),
          constantOp.value().cast<DenseElementsAttr>().getSplatValue());

      LinalgOp fusedOp = createLinalgOpOfSameType(
          linalgOp, rewriter, rewriter.getUnknownLoc(),
          linalgOp.getOperation()->getResultTypes(),
          /*inputs=*/fusedOperands,
          /*outputBuffers=*/ValueRange{},
          /*initTensors=*/ValueRange{}, // no init tensors for now.
          rewriter.getAffineMapArrayAttr(fusedIndexMaps),
          linalgOp.iterator_types(),
          /*doc=*/nullptr,
          /*library_call=*/nullptr,
          /*symbol_source=*/nullptr);

      // Map the block argument corresponding to the replaced argument with the
      // scalar constant.
      Region &linalgOpRegion = linalgOp.getOperation()->getRegion(0);
      Block &entryBlock = *linalgOpRegion.begin();
      unsigned argIndex = entryBlock.getNumArguments() -
                          linalgOp.getNumInputs() + operand.index();
      BlockAndValueMapping mapping;
      mapping.map(entryBlock.getArgument(argIndex), scalarConstant);
      Region &fusedRegion = fusedOp.getOperation()->getRegion(0);
      rewriter.cloneRegionBefore(linalgOpRegion, fusedRegion,
                                 fusedRegion.begin(), mapping);
      rewriter.replaceOp(linalgOp, fusedOp.getOperation()->getResults());
      if (constantOp.use_empty())
        rewriter.eraseOp(constantOp);
      return success();
    }
    return failure();
  }
};
} // namespace

Optional<SmallVector<Value, 1>>
mlir::linalg::fuseTensorOps(PatternRewriter &rewriter, Operation *consumer,
                            unsigned consumerIdx, OperationFolder *folder) {
  if (consumerIdx >= consumer->getNumOperands())
    return llvm::None;
  Operation *producer = consumer->getOperand(consumerIdx).getDefiningOp();
  if (!producer || producer->getNumResults() != 1)
    return llvm::None;

  // Fuse when consumer is GenericOp or IndexedGenericOp.
  if (!isa<GenericOp, IndexedGenericOp>(consumer) ||
      !isa<GenericOp, IndexedGenericOp>(producer))
    return llvm::None;

  return fuseTensorOpsImpl(cast<LinalgOp>(producer), cast<LinalgOp>(consumer),
                           consumerIdx, rewriter, folder);
}

namespace {
/// Patterns to fuse a generic op, with the producer of its operands.
template <typename LinalgOpTy>
struct FuseTensorOps : public OpRewritePattern<LinalgOpTy> {
  using OpRewritePattern<LinalgOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(LinalgOpTy op,
                                PatternRewriter &rewriter) const override {
    // Find the first operand that is defined by another generic op on tensors.
    for (auto operandNum :
         llvm::seq<unsigned>(0, op.getOperation()->getNumOperands())) {
      Operation *producer =
          op.getOperation()->getOperand(operandNum).getDefiningOp();
      if (!producer)
        continue;
      Optional<SmallVector<Value, 1>> fusedOpResults =
          fuseTensorOps(rewriter, op, operandNum);
      if (fusedOpResults) {
        rewriter.replaceOp(op, *fusedOpResults);
        if (producer->use_empty())
          rewriter.eraseOp(producer);
        return success();
      }
    }
    return failure();
  }
};

/// Pass that fuses generic ops on tensors. Used only for testing.
struct FusionOfTensorOpsPass
    : public LinalgFusionOfTensorOpsBase<FusionOfTensorOpsPass> {
  void runOnOperation() override {
    OwningRewritePatternList patterns;
    Operation *op = getOperation();
    populateLinalgTensorOpsFusionPatterns(op->getContext(), patterns);
    applyPatternsAndFoldGreedily(op->getRegions(), std::move(patterns));
  }
};

/// Pass to test folding of reshape op with generic/indexed_generic ops by
/// linearization.
struct FoldReshapeOpsByLinearizationPass
    : public LinalgFoldReshapeOpsByLinearizationBase<
          FoldReshapeOpsByLinearizationPass> {
  void runOnOperation() override {
    OwningRewritePatternList patterns;
    Operation *op = getOperation();
    populateFoldReshapeOpsByLinearizationPatterns(op->getContext(), patterns);
    applyPatternsAndFoldGreedily(op->getRegions(), std::move(patterns));
  }
};

} // namespace

void mlir::populateFoldReshapeOpsByLinearizationPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns) {
  patterns.insert<FoldProducerReshapeOpByLinearization<GenericOp>,
                  FoldProducerReshapeOpByLinearization<IndexedGenericOp>,
                  FoldConsumerReshapeOpByLinearization>(context);
}

void mlir::populateFoldReshapeOpsByExpansionPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns) {
  patterns.insert<FoldReshapeWithGenericOpByExpansion,
                  FoldWithProducerReshapeOpByExpansion<GenericOp>,
                  FoldWithProducerReshapeOpByExpansion<IndexedGenericOp>>(
      context);
}

void mlir::populateLinalgTensorOpsFusionPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns) {
  patterns.insert<FuseTensorOps<GenericOp>, FuseTensorOps<IndexedGenericOp>,
                  FoldSplatConstants<GenericOp>,
                  FoldSplatConstants<IndexedGenericOp>>(context);
  populateFoldReshapeOpsByExpansionPatterns(context, patterns);
  GenericOp::getCanonicalizationPatterns(patterns, context);
  IndexedGenericOp::getCanonicalizationPatterns(patterns, context);
  TensorReshapeOp::getCanonicalizationPatterns(patterns, context);
}

std::unique_ptr<Pass> mlir::createLinalgFusionOfTensorOpsPass() {
  return std::make_unique<FusionOfTensorOpsPass>();
}

std::unique_ptr<Pass> mlir::createFoldReshapeOpsByLinearizationPass() {
  return std::make_unique<FoldReshapeOpsByLinearizationPass>();
}
