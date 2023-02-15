//===- ConvertToDestinationStyle.cpp - Convert non-DPS to DPS ops ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains patterns to convert non-DPS ops to DPS ops. New
// tensor.empty ops are inserted as a destination. Such tensor.empty can be
// eliminated with "empty tensor elimination", allowing them to bufferize
// without an allocation (assuming there are no further conflicts).
//
//===----------------------------------------------------------------------===//
//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::tensor;

// Implements backtracking to traverse indices of the output buffer while
// iterating over op.elements().
static Value createInserts(RewriterBase &rewriter, Location loc, int dim,
                           Value destination, ArrayRef<int64_t> shape,
                           ArrayRef<Value> constants,
                           OperandRange::iterator &elementIt,
                           SmallVectorImpl<Value> &indices) {
  if (dim == static_cast<int>(shape.size()) - 1) {
    for (int i = 0; i < shape.back(); ++i) {
      indices.back() = constants[i];
      destination = rewriter.create<tensor::InsertOp>(loc, *elementIt,
                                                      destination, indices);
      ++elementIt;
    }
    return destination;
  }
  for (int i = 0; i < shape[dim]; ++i) {
    indices[dim] = constants[i];
    destination = createInserts(rewriter, loc, dim + 1, destination, shape,
                                constants, elementIt, indices);
  }
  return destination;
}

namespace {

/// Lower tensor.from_elements to a sequence of chained tensor.insert.
struct FromElementsOpConverter : public OpRewritePattern<FromElementsOp> {
  using OpRewritePattern<FromElementsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FromElementsOp elementsOp,
                                PatternRewriter &rewriter) const override {
    Location loc = elementsOp.getLoc();
    RankedTensorType tensorType = elementsOp.getType().cast<RankedTensorType>();
    auto shape = tensorType.getShape();

    // Create tensor.empty.
    auto emptyOp = rewriter.create<EmptyOp>(loc, tensorType, ValueRange());

    // Case: tensor<elem_type>.
    if (shape.empty()) {
      rewriter.replaceOpWithNewOp<tensor::InsertOp>(
          elementsOp, elementsOp.getElements().front(), emptyOp.getResult(),
          ValueRange());
      return success();
    }

    // Create constants for the range of possible indices [0, max{shape_i}).
    auto maxDim = *std::max_element(shape.begin(), shape.end());
    SmallVector<Value, 2> constants;
    constants.reserve(maxDim);
    for (int i = 0; i < maxDim; ++i)
      constants.push_back(rewriter.create<arith::ConstantIndexOp>(loc, i));

    // Traverse all elements and create tensor.insert ops.
    auto elementIt = elementsOp.getElements().begin();
    SmallVector<Value, 2> indices(tensorType.getRank(), constants[0]);
    Value result = createInserts(rewriter, loc, /*dim=*/0, emptyOp.getResult(),
                                 shape, constants, elementIt, indices);

    // Replace tensor.from_elements.
    rewriter.replaceOp(elementsOp, result);
    return success();
  }
};

/// Lower tensor.generate to linalg.generic.
struct GenerateOpConverter : public OpRewritePattern<GenerateOp> {
  using OpRewritePattern<GenerateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenerateOp generateOp,
                                PatternRewriter &rewriter) const override {
    // Only ops with exactly one block are supported.
    if (!generateOp.getBody().hasOneBlock())
      return failure();

    Location loc = generateOp.getLoc();
    RankedTensorType tensorType = generateOp.getType().cast<RankedTensorType>();

    // Create tensor.empty.
    auto emptyOp = rewriter.create<EmptyOp>(loc, tensorType,
                                            generateOp.getDynamicExtents());

    // Create linalg.generic.
    SmallVector<utils::IteratorType> iteratorTypes(
        tensorType.getRank(), utils::IteratorType::parallel);
    SmallVector<AffineMap> indexingMaps(
        1, rewriter.getMultiDimIdentityMap(tensorType.getRank()));
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, tensorType, /*inputs=*/ValueRange(),
        /*outputs=*/ValueRange{emptyOp.getResult()}, /*indexingMaps=*/
        indexingMaps, iteratorTypes);
    Block *body = rewriter.createBlock(&genericOp->getRegion(0), {},
                                       tensorType.getElementType(), loc);
    rewriter.setInsertionPointToStart(body);
    SmallVector<Value> bbArgReplacements;
    for (int64_t i = 0; i < tensorType.getRank(); ++i)
      bbArgReplacements.push_back(rewriter.create<linalg::IndexOp>(loc, i));
    rewriter.mergeBlocks(&generateOp.getBody().front(), body,
                         bbArgReplacements);

    // Update terminator.
    auto yieldOp = cast<tensor::YieldOp>(body->getTerminator());
    rewriter.replaceOpWithNewOp<linalg::YieldOp>(yieldOp, yieldOp.getValue());

    // Replace tensor.generate.
    rewriter.replaceOp(generateOp, genericOp->getResult(0));
    return success();
  }
};
} // namespace

static Operation *movePaddingToFillOrGenericOp(RewriterBase &rewriter,
                                               Location loc, PadOp padOp,
                                               Value dest) {
  OpBuilder::InsertionGuard g(rewriter);
  RankedTensorType resultType = padOp.getResultType();

  // Examine the yielded value to decide if a linalg.generic is neede or a
  // linalg.fill is sufficient.
  Value yieldedValue =
      cast<tensor::YieldOp>(padOp.getBody()->getTerminator()).getValue();
  Attribute constYieldedValue;
  // Is the yielded value a bbArg defined outside of the PadOp?
  bool outsideBbArg =
      yieldedValue.isa<BlockArgument>() &&
      yieldedValue.cast<BlockArgument>().getOwner()->getParentOp() !=
          padOp.getOperation();
  // Is the yielded value an OpResult defined outside of the PadOp?
  bool outsideOpResult =
      yieldedValue.isa<OpResult>() &&
      yieldedValue.getDefiningOp()->getParentOp() != padOp.getOperation();
  bool invariantYieldedValue = outsideBbArg || outsideOpResult;
  if (matchPattern(yieldedValue, m_Constant(&constYieldedValue))) {
    // Padding with a constant: Create linalg.fill.
    Dialect *arithDialect =
        rewriter.getContext()->getLoadedDialect<arith::ArithDialect>();
    Value fillValue =
        arithDialect
            ->materializeConstant(rewriter, constYieldedValue,
                                  yieldedValue.getType(), yieldedValue.getLoc())
            ->getResult(0);
    auto fillOp = rewriter.create<linalg::FillOp>(loc, ValueRange(fillValue),
                                                  ValueRange(dest));
    return fillOp;
  }

  if (invariantYieldedValue) {
    // Padding with an invariant value.
    auto fillOp = rewriter.create<linalg::FillOp>(loc, ValueRange(yieldedValue),
                                                  ValueRange(dest));
    return fillOp;
  }

  // Create linalg.generic.
  SmallVector<utils::IteratorType> iteratorTypes(resultType.getRank(),
                                                 utils::IteratorType::parallel);
  SmallVector<AffineMap> indexingMaps(
      1, rewriter.getMultiDimIdentityMap(resultType.getRank()));
  auto genericOp = rewriter.create<linalg::GenericOp>(
      loc, resultType, /*inputs=*/ValueRange(),
      /*outputs=*/ValueRange{dest}, /*indexingMaps=*/
      indexingMaps, iteratorTypes);
  Block *body = rewriter.createBlock(&genericOp->getRegion(0), {},
                                     resultType.getElementType(), loc);
  rewriter.setInsertionPointToStart(body);
  SmallVector<Value> bbArgReplacements;
  for (int64_t i = 0; i < resultType.getRank(); ++i)
    bbArgReplacements.push_back(rewriter.create<linalg::IndexOp>(loc, i));
  rewriter.mergeBlocks(padOp.getBody(), body, bbArgReplacements);

  // Update terminator.
  auto yieldOp = cast<tensor::YieldOp>(body->getTerminator());
  rewriter.replaceOpWithNewOp<linalg::YieldOp>(yieldOp, yieldOp.getValue());
  return genericOp;
}

static SmallVector<Value> reifyOrComputeDynamicSizes(OpBuilder &b,
                                                     Value value) {
  auto tensorType = value.getType().cast<RankedTensorType>();
  if (tensorType.hasStaticShape())
    return {};

  // Try to reify dynamic sizes.
  if (auto reifiableOp =
          value.getDefiningOp<ReifyRankedShapedTypeOpInterface>()) {
    ReifiedRankedShapedTypeDims reifiedShape;
    if (succeeded(reifiableOp.reifyResultShapes(b, reifiedShape))) {
      SmallVector<Value> dynSizes;
      for (int64_t i = 0; i < tensorType.getRank(); ++i) {
        if (tensorType.isDynamicDim(i))
          dynSizes.push_back(
              reifiedShape[value.cast<OpResult>().getResultNumber()][i]);
      }
      return dynSizes;
    }
  }

  // Create tensor.dim ops.
  SmallVector<Value> dynSizes;
  for (int64_t i = 0; i < tensorType.getRank(); ++i) {
    if (tensorType.isDynamicDim(i))
      dynSizes.push_back(
          b.create<DimOp>(value.getLoc(), value,
                          b.create<arith::ConstantIndexOp>(value.getLoc(), i)));
  }
  return dynSizes;
}

static Value createAllocationForTensor(RewriterBase &rewriter, Location loc,
                                       Value value,
                                       Attribute memorySpace = {}) {
  OpBuilder::InsertionGuard g(rewriter);
  auto tensorType = value.getType().cast<RankedTensorType>();

  // Create buffer allocation.
  auto memrefType = bufferization::getMemRefTypeWithStaticIdentityLayout(
                        tensorType, memorySpace)
                        .cast<MemRefType>();
  SmallVector<Value> dynamicSizes = reifyOrComputeDynamicSizes(rewriter, value);
  Value alloc = rewriter.create<memref::AllocOp>(loc, memrefType, dynamicSizes);

  // Place deallocation at the end of the block.
  rewriter.setInsertionPoint(rewriter.getInsertionBlock()->getTerminator());
  rewriter.create<memref::DeallocOp>(loc, alloc);

  return alloc;
}

Value linalg::bufferizeToAllocation(RewriterBase &rewriter, PadOp padOp,
                                    Attribute memorySpace) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(padOp);
  Location loc = padOp.getLoc();

  // Create buffer allocation.
  Value alloc =
      createAllocationForTensor(rewriter, loc, padOp.getResult(), memorySpace);
  rewriter.setInsertionPointAfter(alloc.getDefiningOp());

  // Create linalg.fill or linalg.generic.
  Operation *fillOp = movePaddingToFillOrGenericOp(rewriter, loc, padOp, alloc);
  rewriter.setInsertionPointAfter(fillOp);

  // Create memref.tensor_store.
  SmallVector<OpFoldResult> sizes =
      getMixedSizes(rewriter, loc, padOp.getSource());
  SmallVector<OpFoldResult> strides(padOp.getResultType().getRank(),
                                    rewriter.getIndexAttr(1));
  Value subview = rewriter.create<memref::SubViewOp>(
      loc, alloc, /*offsets=*/padOp.getMixedLowPad(), sizes, strides);
  rewriter.create<memref::TensorStoreOp>(loc, padOp.getSource(), subview);

  // Create bufferization.to_tensor with "restrict" and "writable". The returned
  // tensor is a new buffer allocation, so it does not alias with any buffer.
  Value toTensorOp = rewriter.create<bufferization::ToTensorOp>(
      loc, alloc, /*restrict=*/true, /*writable=*/true);
  rewriter.replaceOp(padOp, toTensorOp);
  return toTensorOp;
}

namespace {
/// Lower tensor.pad to linalg.generic + tensor.insert_slice.
struct PadOpConverter : public OpRewritePattern<PadOp> {
  using OpRewritePattern<PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PadOp padOp,
                                PatternRewriter &rewriter) const override {
    // Only ops with exactly one block are supported.
    if (!padOp.getBodyRegion().hasOneBlock())
      return failure();

    // Create tensor.empty.
    Location loc = padOp.getLoc();
    RankedTensorType resultType = padOp.getResultType();
    ReifiedRankedShapedTypeDims reifiedShape;
    if (failed(cast<ReifyRankedShapedTypeOpInterface>(padOp.getOperation())
                   .reifyResultShapes(rewriter, reifiedShape)))
      return rewriter.notifyMatchFailure(
          padOp, "failed to reify tensor.pad op result shape");
    SmallVector<Value> dynamicSizes;
    for (int64_t i = 0; i < resultType.getRank(); ++i)
      if (resultType.isDynamicDim(i))
        dynamicSizes.push_back(reifiedShape[0][i]);
    auto emptyOp = rewriter.create<EmptyOp>(loc, resultType, dynamicSizes);

    // Create linalg.fill or linalg.generic.
    Operation *fillOp =
        movePaddingToFillOrGenericOp(rewriter, loc, padOp, emptyOp.getResult());
    rewriter.setInsertionPointAfter(fillOp);

    // Create tensor::InsertSliceOp.
    SmallVector<OpFoldResult> sliceSizes =
        getMixedSizes(rewriter, loc, padOp.getSource());
    SmallVector<OpFoldResult> sliceStrides(resultType.getRank(),
                                           rewriter.getIndexAttr(1));
    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        padOp, padOp.getSource(), fillOp->getResult(0),
        /*offsets=*/padOp.getMixedLowPad(), sliceSizes, sliceStrides);

    return success();
  }
};
} // namespace

Value linalg::bufferizeToAllocation(RewriterBase &rewriter, Value value,
                                    Attribute memorySpace) {
  // Call specialized overload for certain ops.
  if (auto padOp = value.getDefiningOp<PadOp>())
    return bufferizeToAllocation(rewriter, padOp, memorySpace);

  // Collect all uses.
  SmallVector<OpOperand *> uses = llvm::to_vector(
      llvm::map_range(value.getUses(), [](OpOperand &use) { return &use; }));

  OpBuilder::InsertionGuard g(rewriter);
  if (auto bbArg = value.dyn_cast<BlockArgument>()) {
    rewriter.setInsertionPointToStart(bbArg.getOwner());
  } else {
    rewriter.setInsertionPointAfter(value.getDefiningOp());
  }
  Location loc = value.getLoc();

  // Create buffer allocation.
  Value alloc = createAllocationForTensor(rewriter, loc, value, memorySpace);

  // Create memref.tensor_store.
  rewriter.setInsertionPointAfter(alloc.getDefiningOp());
  rewriter.create<memref::TensorStoreOp>(loc, value, alloc);

  // Create bufferization.to_tensor with "restrict" and "writable". The returned
  // tensor is a new buffer allocation, so it does not alias with any buffer.
  Value toTensorOp = rewriter.create<bufferization::ToTensorOp>(
      loc, alloc, /*restrict=*/true, /*writable=*/true);
  for (OpOperand *use : uses) {
    rewriter.updateRootInPlace(use->getOwner(),
                               [&]() { use->set(toTensorOp); });
  }

  return toTensorOp;
}

void linalg::populateConvertToDestinationStylePatterns(
    RewritePatternSet &patterns) {
  patterns.insert<FromElementsOpConverter, GenerateOpConverter, PadOpConverter>(
      patterns.getContext());
}
