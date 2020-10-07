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

using namespace mlir;
using namespace mlir::linalg;

namespace {

/// Implementation of fusion of generic ops and indexed_generic ops.
struct FuseGenericOpsOnTensors {
  static bool isFusible(LinalgOp producer, LinalgOp consumer,
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

  static LinalgOp fuse(LinalgOp producer, LinalgOp consumer,
                       unsigned consumerIdx, PatternRewriter &rewriter,
                       OperationFolder *folder = nullptr) {
    if (!isFusible(producer, consumer, consumerIdx))
      return nullptr;

    unsigned numFusedOperands = producer.getOperation()->getNumOperands() +
                                consumer.getOperation()->getNumOperands() - 1;

    // Compute the fused operands list,
    SmallVector<Value, 2> fusedOperands;
    fusedOperands.reserve(numFusedOperands);
    auto consumerOperands = consumer.getOperation()->getOperands();
    auto producerOperands = producer.getOperation()->getOperands();
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
    fusedIndexMaps.reserve(fusedOperands.size() +
                           consumer.getOperation()->getNumResults());
    fusedIndexMaps.assign(consumerIndexMaps.begin(),
                          std::next(consumerIndexMaps.begin(), consumerIdx));
    // Compute indexing maps for the producer args in the fused operation.
    computeProducerOperandIndex(
        producer, consumer.getInputIndexingMap(consumerIdx), fusedIndexMaps);

    // Append the indexing maps for the remaining consumer operands.
    fusedIndexMaps.append(std::next(consumerIndexMaps.begin(), consumerIdx + 1),
                          consumerIndexMaps.end());

    // Generate the fused op.
    // Tensor-level fusion is only on ops without initTensors and outputBuffers.
    LinalgOp fusedOp;
    if (isa<GenericOp>(producer.getOperation()) &&
        isa<GenericOp>(consumer.getOperation())) {
      fusedOp =
          rewriter
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
              .create<IndexedGenericOp>(
                  consumer.getLoc(), consumer.getOperation()->getResultTypes(),
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
    AffineMap consumerResultIndexMap =
        consumer.getInputIndexingMap(consumerIdx);
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

    generateFusedRegion(rewriter, fusedOp, producer, consumer,
                        consumerToProducerLoopsMap, consumerIdx,
                        consumer.getNumLoops());
    return fusedOp;
  }

private:
  /// Append to `fusedOpIndexingMapAttrs` the indexing maps for the operands of
  /// the `producer` to use in the fused operation given the indexing map of the
  /// result of the producer in the consumer.
  static void computeProducerOperandIndex(
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

  /// Generate the region of the fused operation. The region of the fused op
  /// must be empty.
  static void generateFusedRegion(PatternRewriter &rewriter, Operation *fusedOp,
                                  LinalgOp producer, LinalgOp consumer,
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
    unsigned numProducerIndices =
        isa<IndexedGenericOp>(producer.getOperation()) ? nloops : 0;
    unsigned numConsumerIndices =
        isa<IndexedGenericOp>(consumer.getOperation()) ? nloops : 0;
    // Firstly, add all the indices to the block arguments.
    for (unsigned i = 0, e = std::max(numProducerIndices, numConsumerIndices);
         i < e; ++i)
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
                fusedBlock->getArguments().take_front(nloops));
            mapper.map(producerArg.value(), newIndex);
          } else {
            mapper.map(producerArg.value(),
                       fusedBlock->addArgument(producerArg.value().getType()));
          }
        }
        continue;
      }

      // If consumer is an indexed_generic op, map the indices to the block
      // arguments directly. Otherwise, add the same type of arugment and map to
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
          mapper.map(
              consumerBlock.getArgument(consumerIdx + numConsumerIndices),
              clonedVal);
        continue;
      }
      rewriter.clone(op, mapper);
    }
    for (auto &op : consumerBlock.getOperations())
      rewriter.clone(op, mapper);
  }
};
} // namespace

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
static bool isTensorReshapeOpFusible(TensorReshapeOp reshapeOp,
                                     AffineMap useIndexMap, bool asProducer) {
  RankedTensorType returnType = reshapeOp.getResultType();
  RankedTensorType operandType = reshapeOp.getSrcType();
  // Reshape is fusible with its consumer (i.e. reshape as a producer) when its
  // operand is of lesser rank than the result. Fusing when operand has higher
  // rank will require use of mods and divs in the indexing maps of the fused op
  // which would make it non-invertible. Similarly reshape is fused with its
  // producer (i.e. reshape as consumer) only if the return type has lesser
  // rank.
  if ((asProducer && returnType.getRank() < operandType.getRank()) ||
      (!asProducer && operandType.getRank() < returnType.getRank()))
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

namespace {

/// Implementation of fusion on tensor ops when producer is a TensorReshapeOp.
struct FuseTensorReshapeOpAsProducer {
  static bool isFusible(TensorReshapeOp producer, LinalgOp consumer,
                        unsigned consumerIdx) {
    return isa<GenericOp, IndexedGenericOp>(consumer.getOperation()) &&
           consumer.hasTensorSemantics() &&
           isTensorReshapeOpFusible(producer,
                                    consumer.getInputIndexingMap(consumerIdx),
                                    /*asProducer=*/true);
  }

  static LinalgOp fuse(TensorReshapeOp producer, LinalgOp consumer,
                       unsigned consumerIdx, PatternRewriter &rewriter,
                       OperationFolder *folder = nullptr) {
    if (producer.src().getDefiningOp<ConstantOp>())
      return nullptr;

    if (!isFusible(producer, consumer, consumerIdx))
      return nullptr;

    // Compute the fused operands list,
    Operation *consumerOp = consumer.getOperation();
    SmallVector<Value, 2> fusedOperands(consumerOp->getOperands());
    fusedOperands[consumerIdx] = producer.src();

    // Compute indexing_maps for the fused operation. The indexing_maps for the
    // operands of the consumers that arent fused are the same.
    SmallVector<AffineMap, 4> fusedIndexMaps =
        llvm::to_vector<4>(llvm::map_range(
            consumer.indexing_maps(), [](Attribute attr) -> AffineMap {
              return attr.cast<AffineMapAttr>().getValue();
            }));

    // Accepted consumer maps are either identity or permutation.
    auto invMap = inversePermutation(fusedIndexMaps[consumerIdx]);

    // Compute the indexing map to use for the operand of the producer.
    AffineMap modifiedMap =
        linearizeCollapsedDims(invMap, producer.getResultType().getShape(),
                               producer.getReassociationMaps());
    for (AffineExpr expr : modifiedMap.getResults()) {
      if (!expr.isPureAffine())
        return nullptr;
    }
    fusedIndexMaps[consumerIdx] = modifiedMap;

    // Further check that the resulting index maps can be fused and
    // inverted. Without this the resultant op is not legal.
    if (!inversePermutation(concatAffineMaps(fusedIndexMaps)))
      return nullptr;

    SmallVector<Attribute, 4> indexMapAttrs = llvm::to_vector<4>(
        llvm::map_range(fusedIndexMaps, [](AffineMap map) -> Attribute {
          return AffineMapAttr::get(map);
        }));
    LinalgOp fusedOp = createLinalgOpOfSameType(
        consumer, rewriter, rewriter.getUnknownLoc(),
        consumerOp->getResultTypes(),
        /*inputs=*/fusedOperands,
        /*outputBuffers=*/ValueRange{},
        /*initTensors=*/ValueRange{}, // no init tensors for now.
        rewriter.getArrayAttr(indexMapAttrs), consumer.iterator_types(),
        /*doc=*/nullptr,
        /*library_call=*/nullptr,
        /*symbol_source=*/nullptr);
    auto &fusedRegion = fusedOp.getOperation()->getRegion(0);
    rewriter.cloneRegionBefore(consumerOp->getRegion(0), fusedRegion,
                               fusedRegion.begin());
    return fusedOp;
  }
};

/// Implementation of fusion on tensor ops when consumer is a TensorReshapeOp.
struct FuseTensorReshapeOpAsConsumer {
  static bool isCollapsingAndFusible(LinalgOp producer,
                                     TensorReshapeOp consumer,
                                     unsigned consumerIdx) {
    return isa<GenericOp, IndexedGenericOp>(producer.getOperation()) &&
           producer.hasTensorSemantics() &&
           isTensorReshapeOpFusible(consumer, producer.getOutputIndexingMap(0),
                                    /*asProducer=*/false);
  }

  static LinalgOp fuseCollapsingCase(LinalgOp producer,
                                     TensorReshapeOp consumer,
                                     unsigned consumerIdx,
                                     PatternRewriter &rewriter) {
    // The indexing_maps for the operands of the fused operation are same as
    // those for the operands of the producer.
    SmallVector<AffineMap, 4> fusedIndexMaps =
        llvm::to_vector<4>(llvm::map_range(
            producer.indexing_maps(), [](Attribute attr) -> AffineMap {
              return attr.cast<AffineMapAttr>().getValue();
            }));

    auto invMap = inversePermutation(producer.getOutputIndexingMap(0));

    // Compute the indexing map to use for the operand of the producer.
    AffineMap modifiedMap =
        linearizeCollapsedDims(invMap, consumer.getSrcType().getShape(),
                               consumer.getReassociationMaps());
    for (AffineExpr expr : modifiedMap.getResults()) {
      if (!expr.isPureAffine())
        return nullptr;
    }
    fusedIndexMaps.back() = modifiedMap;

    // Further check that the resulting index maps can be fused and
    // inverted. Without this the resultant op is not legal.
    if (!inversePermutation(concatAffineMaps(fusedIndexMaps)))
      return nullptr;

    SmallVector<Attribute, 4> indexMapAttrs = llvm::to_vector<4>(
        llvm::map_range(fusedIndexMaps, [](AffineMap map) -> Attribute {
          return AffineMapAttr::get(map);
        }));

    Operation *producerOp = producer.getOperation();
    LinalgOp fusedOp = createLinalgOpOfSameType(
        producer, rewriter, rewriter.getUnknownLoc(), consumer.getResultType(),
        /*inputs=*/producerOp->getOperands(),
        /*outputBuffers=*/ValueRange{},
        /*initTensors=*/ValueRange{}, // no init tensors for now.
        rewriter.getArrayAttr(indexMapAttrs), producer.iterator_types(),
        /*doc=*/nullptr,
        /*library_call=*/nullptr,
        /*symbol_source=*/nullptr);
    auto &fusedRegion = fusedOp.getOperation()->getRegion(0);
    rewriter.cloneRegionBefore(producerOp->getRegion(0), fusedRegion,
                               fusedRegion.begin());
    return fusedOp;
  }

  static bool isExpandingAndFusible(LinalgOp producer, TensorReshapeOp consumer,
                                    unsigned consumerIdx) {
    // Is fusible only if:
    //   1) The producer is a generic op.
    //   2) The producer has tensor semantics.
    //   3) The tensor reshape op is a expanding case.
    //   4) All the shapes are the same for the generic op.
    //   5) All the indexing maps in producer are identity.
    //   6) All the loops in producer are parallel loops.
    //   7) The producer has a single user.
    auto types = producer.getInputOutputShapedTypes();
    assert(!types.empty());
    return isa<GenericOp>(producer.getOperation()) &&
           producer.hasTensorSemantics() &&
           consumer.getSrcType().getRank() <
               consumer.getResultType().getRank() &&
           std::equal(types.begin() + 1, types.end(), types.begin()) &&
           llvm::all_of(producer.getIndexingMaps(),
                        [](AffineMap map) { return map.isIdentity(); }) &&
           llvm::all_of(producer.iterator_types(),
                        [](Attribute attr) {
                          return attr.cast<StringAttr>().getValue() ==
                                 getParallelIteratorTypeName();
                        }) &&
           producer.getOperation()->hasOneUse();
  }

  static LinalgOp fuseExpandingCase(LinalgOp producer, TensorReshapeOp consumer,
                                    unsigned consumerIdx,
                                    PatternRewriter &rewriter) {
    Location loc = producer.getLoc();
    auto dstShape = consumer.getResultType().cast<ShapedType>().getShape();
    SmallVector<Value, 4> args;
    for (auto arg : producer.getOperation()->getOperands()) {
      auto type = RankedTensorType::get(
          dstShape, arg.getType().cast<ShapedType>().getElementType());
      args.push_back(rewriter.createOrFold<linalg::TensorReshapeOp>(
          loc, type, arg, consumer.reassociation()));
    }

    SmallVector<Type, 4> resultTypes;
    for (auto t : producer.getOutputTensorTypes()) {
      Type type = RankedTensorType::get(dstShape,
                                        t.cast<ShapedType>().getElementType());
      resultTypes.push_back(type);
    }

    int rank = dstShape.size();
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultTypes, /*inputs=*/args,
        /*outputBuffers=*/ValueRange{},
        /*initTensors=*/ValueRange{},
        SmallVector<AffineMap, 3>(args.size() + resultTypes.size(),
                                  rewriter.getMultiDimIdentityMap(rank)),
        SmallVector<StringRef, 3>(rank, getParallelIteratorTypeName()));
    Region &region = genericOp.getRegion();
    rewriter.cloneRegionBefore(producer.getOperation()->getRegion(0), region,
                               region.begin());
    return cast<LinalgOp>(genericOp.getOperation());
  }

  static LinalgOp fuse(LinalgOp producer, TensorReshapeOp consumer,
                       unsigned consumerIdx, PatternRewriter &rewriter,
                       OperationFolder *folder = nullptr) {
    if (isCollapsingAndFusible(producer, consumer, consumerIdx))
      return fuseCollapsingCase(producer, consumer, consumerIdx, rewriter);
    if (isExpandingAndFusible(producer, consumer, consumerIdx))
      return fuseExpandingCase(producer, consumer, consumerIdx, rewriter);
    return nullptr;
  }
};

/// Implementation of fusion on tensor ops when producer is a splat constant.
struct FuseConstantOpAsProducer {
  static bool isFusible(ConstantOp producer, LinalgOp consumer,
                        unsigned consumerIdx) {
    return isa<GenericOp, IndexedGenericOp>(consumer.getOperation()) &&
           consumer.hasTensorSemantics() &&
           producer.getResult().getType().isa<RankedTensorType>() &&
           producer.value().cast<DenseElementsAttr>().isSplat();
  }

  static LinalgOp fuse(ConstantOp producer, LinalgOp consumer,
                       unsigned consumerIdx, PatternRewriter &rewriter,
                       OperationFolder *folder = nullptr) {
    if (!isFusible(producer, consumer, consumerIdx))
      return nullptr;

    // The indexing_maps for the operands of the fused operation are same as
    // those for the operands of the consumer without the indexing map at
    // consumerIdx
    SmallVector<AffineMap, 4> fusedIndexMaps =
        llvm::to_vector<4>(llvm::map_range(
            consumer.indexing_maps(), [](Attribute attr) -> AffineMap {
              return attr.cast<AffineMapAttr>().getValue();
            }));
    fusedIndexMaps.erase(std::next(fusedIndexMaps.begin(), consumerIdx));

    // The operands list is same as the consumer with the argument for constant
    // index dropped.
    Operation *consumerOp = consumer.getOperation();
    SmallVector<Value, 4> fusedOperands(consumerOp->getOperands());
    fusedOperands.erase(std::next(fusedOperands.begin(), consumerIdx));

    // Create a constant scalar value from the splat constant.
    Value scalarConstant = rewriter.create<ConstantOp>(
        producer.getLoc(),
        producer.value().cast<DenseElementsAttr>().getSplatValue());

    LinalgOp fusedOp = createLinalgOpOfSameType(
        consumer, rewriter, rewriter.getUnknownLoc(),
        consumerOp->getResultTypes(),
        /*inputs=*/fusedOperands,
        /*outputBuffers=*/ValueRange{},
        /*initTensors=*/ValueRange{}, // no init tensors for now.
        rewriter.getAffineMapArrayAttr(fusedIndexMaps),
        consumer.iterator_types(),
        /*doc=*/nullptr,
        /*library_call=*/nullptr,
        /*symbol_source=*/nullptr);

    // Map the block argument corresponding to the replaced argument with the
    // scalar constant.
    Region &consumerRegion = consumerOp->getRegion(0);
    Block &entryBlock = *consumerRegion.begin();
    unsigned argIndex = entryBlock.getNumArguments() -
                        consumerOp->getNumOperands() + consumerIdx;
    BlockAndValueMapping mapping;
    mapping.map(entryBlock.getArgument(argIndex), scalarConstant);
    Region &fusedRegion = fusedOp.getOperation()->getRegion(0);
    rewriter.cloneRegionBefore(consumerRegion, fusedRegion, fusedRegion.begin(),
                               mapping);
    return fusedOp;
  }
};
} // namespace

Operation *mlir::linalg::fuseTensorOps(PatternRewriter &rewriter,
                                       Operation *consumer,
                                       unsigned consumerIdx,
                                       OperationFolder *folder) {
  if (consumerIdx >= consumer->getNumOperands())
    return nullptr;
  Operation *producer = consumer->getOperand(consumerIdx).getDefiningOp();
  if (!producer || producer->getNumResults() != 1)
    return nullptr;

  // Fuse when consumer is GenericOp or IndexedGenericOp.
  if (isa<GenericOp, IndexedGenericOp>(consumer)) {
    if (isa<GenericOp, IndexedGenericOp>(producer))
      return FuseGenericOpsOnTensors::fuse(cast<LinalgOp>(producer),
                                           cast<LinalgOp>(consumer),
                                           consumerIdx, rewriter, folder);
    if (auto reshapeOpProducer = dyn_cast<TensorReshapeOp>(producer))
      return FuseTensorReshapeOpAsProducer::fuse(reshapeOpProducer,
                                                 cast<LinalgOp>(consumer),
                                                 consumerIdx, rewriter, folder);
    if (auto constantOpProducer = dyn_cast<ConstantOp>(producer))
      return FuseConstantOpAsProducer::fuse(constantOpProducer,
                                            cast<LinalgOp>(consumer),
                                            consumerIdx, rewriter, folder);
    return nullptr;
  }

  if (isa<GenericOp, IndexedGenericOp>(producer)) {
    // Fuse when consumer is a TensorReshapeOp.
    if (TensorReshapeOp reshapeOp = dyn_cast<TensorReshapeOp>(consumer)) {
      return FuseTensorReshapeOpAsConsumer::fuse(
          cast<LinalgOp>(producer), reshapeOp, consumerIdx, rewriter, folder);
    }
  }

  return nullptr;
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
      if (Operation *fusedOp = fuseTensorOps(rewriter, op, operandNum)) {
        rewriter.replaceOp(op, fusedOp->getResults());
        if (producer && llvm::all_of(producer->getResults(),
                                     [](Value val) { return val.use_empty(); }))
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
    applyPatternsAndFoldGreedily(op->getRegions(), patterns);
  };
};
} // namespace

void mlir::populateLinalgTensorOpsFusionPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns) {
  patterns.insert<FuseTensorOps<GenericOp>, FuseTensorOps<IndexedGenericOp>,
                  FuseTensorOps<TensorReshapeOp>>(context);
}

std::unique_ptr<Pass> mlir::createLinalgFusionOfTensorOpsPass() {
  return std::make_unique<FusionOfTensorOpsPass>();
}
