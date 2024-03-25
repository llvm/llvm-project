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
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"
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

/// Create a memcpy from the given source tensor to the given destination
/// memref. The copy op type can be specified in the `options`.
static void createMemcpy(OpBuilder &b, Location loc, Value tensorSource,
                         Value memrefDest,
                         const linalg::BufferizeToAllocationOptions &options) {
  auto tensorType = dyn_cast<RankedTensorType>(tensorSource.getType());
  assert(tensorType && "expected ranked tensor");
  assert(memrefDest.getType().isa<MemRefType>() && "expected ranked memref");

  switch (options.memcpyOp) {
  case linalg::BufferizeToAllocationOptions::MemcpyOp::
      MaterializeInDestination: {
    // Note: This is the preferred way of memcpy'ing because no layout map
    // and/or memory space must be specified for the source.
    auto materializeOp = b.create<bufferization::MaterializeInDestinationOp>(
        loc, tensorSource, memrefDest);
    materializeOp.setWritable(true);
  } break;
  case linalg::BufferizeToAllocationOptions::MemcpyOp::MemrefCopy: {
    // TODO: Support custom memory space on source.
    // We do not know the layout map of the source yet, so use a fully dynamic
    // layout for best compatibility.
    Value toMemref = b.create<bufferization::ToMemrefOp>(
        loc, bufferization::getMemRefTypeWithFullyDynamicLayout(tensorType),
        tensorSource, /*readOnly=*/true);
    b.create<memref::CopyOp>(loc, toMemref, memrefDest);
  } break;
  case linalg::BufferizeToAllocationOptions::MemcpyOp::LinalgCopy: {
    // TODO: Support custom memory space on source.
    // We do not know the layout map of the source yet, so use a fully dynamic
    // layout for best compatibility.
    Value toMemref = b.create<bufferization::ToMemrefOp>(
        loc, bufferization::getMemRefTypeWithFullyDynamicLayout(tensorType),
        tensorSource, /*readOnly=*/true);
    b.create<linalg::CopyOp>(loc, toMemref, memrefDest);
  } break;
  };
}

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
      isa<BlockArgument>(yieldedValue) &&
      cast<BlockArgument>(yieldedValue).getOwner()->getParentOp() !=
          padOp.getOperation();
  // Is the yielded value an OpResult defined outside of the PadOp?
  bool outsideOpResult =
      isa<OpResult>(yieldedValue) &&
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
  auto tensorType = cast<RankedTensorType>(value.getType());
  if (tensorType.hasStaticShape())
    return {};

  // Try to reify dynamic sizes.
  ReifiedRankedShapedTypeDims reifiedShape;
  if (isa<OpResult>(value) &&
      succeeded(reifyResultShapes(b, value.getDefiningOp(), reifiedShape))) {
    SmallVector<Value> dynSizes;
    for (int64_t i = 0; i < tensorType.getRank(); ++i) {
      if (tensorType.isDynamicDim(i))
        dynSizes.push_back(
            reifiedShape[cast<OpResult>(value).getResultNumber()][i]
                .get<Value>());
    }
    return dynSizes;
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

static Value
createAllocationForTensor(RewriterBase &rewriter, Location loc, Value value,
                          const linalg::BufferizeToAllocationOptions &options,
                          Attribute memorySpace = {}) {
  OpBuilder::InsertionGuard g(rewriter);
  auto tensorType = cast<RankedTensorType>(value.getType());

  // Create buffer allocation.
  auto memrefType =
      cast<MemRefType>(bufferization::getMemRefTypeWithStaticIdentityLayout(
          tensorType, memorySpace));
  SmallVector<Value> dynamicSizes = reifyOrComputeDynamicSizes(rewriter, value);

  Value alloc;
  if (options.allocOp ==
      linalg::BufferizeToAllocationOptions::AllocOp::MemrefAlloc) {
    alloc = rewriter.create<memref::AllocOp>(loc, memrefType, dynamicSizes);
    if (options.emitDealloc) {
      // Place deallocation at the end of the block.
      rewriter.setInsertionPoint(rewriter.getInsertionBlock()->getTerminator());
      rewriter.create<memref::DeallocOp>(loc, alloc);
    }
  } else if (options.allocOp ==
             linalg::BufferizeToAllocationOptions::AllocOp::MemrefAlloca) {
    alloc = rewriter.create<memref::AllocaOp>(loc, memrefType, dynamicSizes);
    // No dealloc is needed.
  }

  return alloc;
}

Value linalg::bufferizeToAllocation(
    RewriterBase &rewriter, const linalg::BufferizeToAllocationOptions &options,
    PadOp padOp, Attribute memorySpace, Operation *insertionPoint) {
  // tensor.pad does not have a destination operand.
  assert(!options.bufferizeDestinationOnly && "invalid options");

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(insertionPoint ? insertionPoint : padOp);
  Location loc = padOp.getLoc();

  // Create buffer allocation.
  Value alloc = createAllocationForTensor(rewriter, loc, padOp.getResult(),
                                          options, memorySpace);
  rewriter.setInsertionPoint(padOp);

  if (!padOp.hasZeroLowPad() || !padOp.hasZeroHighPad()) {
    // Create linalg.fill or linalg.generic. Not needed if there is no padding.
    Operation *fillOp =
        movePaddingToFillOrGenericOp(rewriter, loc, padOp, alloc);
    rewriter.setInsertionPointAfter(fillOp);
  }

  // Create memcpy.
  SmallVector<OpFoldResult> sizes =
      getMixedSizes(rewriter, loc, padOp.getSource());
  SmallVector<OpFoldResult> strides(padOp.getResultType().getRank(),
                                    rewriter.getIndexAttr(1));
  Value subview = rewriter.create<memref::SubViewOp>(
      loc, alloc, /*offsets=*/padOp.getMixedLowPad(), sizes, strides);
  createMemcpy(rewriter, loc, padOp.getSource(), subview, options);

  // Create bufferization.to_tensor with "restrict" and "writable". The returned
  // tensor is a new buffer allocation, so it does not alias with any buffer.
  Value toTensorOp = rewriter.create<bufferization::ToTensorOp>(
      loc, alloc, /*restrict=*/true, /*writable=*/true);
  rewriter.replaceOp(padOp, toTensorOp);
  return alloc;
}

Value linalg::bufferizeToAllocation(
    RewriterBase &rewriter, const linalg::BufferizeToAllocationOptions &options,
    vector::MaskOp maskOp, Attribute memorySpace, Operation *insertionPoint) {
  assert(llvm::range_size(maskOp.getMaskBlock()->without_terminator()) == 1 &&
         "expected single masked op");
  OpBuilder::InsertionGuard g(rewriter);
  bufferization::BufferizationOptions bufferizationOptions;
  Operation *yieldOp = maskOp.getMaskRegion().front().getTerminator();
  assert(isa<vector::YieldOp>(yieldOp) && "expected yield op terminator");

  // Bufferize maskable op. By default, place the buffer allocation right before
  // the mask op.
  Value alloc = bufferizeToAllocation(
      rewriter, options, maskOp.getMaskableOp(), memorySpace,
      /*insertionPoint=*/insertionPoint ? insertionPoint : maskOp);

  if (options.bufferizeDestinationOnly)
    return alloc;

  // Bufferize terminator.
  rewriter.setInsertionPoint(yieldOp);
  if (failed(cast<bufferization::BufferizableOpInterface>(yieldOp).bufferize(
          rewriter, bufferizationOptions)))
    return nullptr;

  // Erase dead to_tensor ops inside of the mask op. This is necessary because
  // there only be one op (apart from the terminator) inside the mask op.
  // TODO: Remove dead to_tensor ops more aggressively during bufferization.
  SmallVector<Operation *> toTensorOps;
  maskOp.walk([&](bufferization::ToTensorOp toTensorOp) {
    if (toTensorOp->getUses().empty())
      toTensorOps.push_back(toTensorOp.getOperation());
  });
  for (Operation *op : toTensorOps)
    rewriter.eraseOp(op);

  // Bufferize mask op.
  SmallVector<OpOperand *> resultUses;
  for (Value result : maskOp.getResults())
    if (isa<TensorType>(result.getType()))
      for (OpOperand &use : result.getUses())
        resultUses.push_back(&use);
  rewriter.setInsertionPoint(maskOp);
  if (failed(cast<bufferization::BufferizableOpInterface>(maskOp.getOperation())
                 .bufferize(rewriter, bufferizationOptions)))
    return nullptr;

  // Set "restrict" attribute, indicating that no other tensor aliases with
  // this tensor. That is because we just allocated a new buffer for the tensor.
  for (OpOperand *resultUse : resultUses) {
    auto toTensorOp =
        resultUse->get().getDefiningOp<bufferization::ToTensorOp>();
    assert(toTensorOp && "expected to_tensor op");
    rewriter.modifyOpInPlace(toTensorOp, [&]() {
      toTensorOp.setRestrict(true);
      toTensorOp.setWritable(true);
    });
  }

  return alloc;
}

Value linalg::bufferizeToAllocation(
    RewriterBase &rewriter, const linalg::BufferizeToAllocationOptions &options,
    bufferization::AllocTensorOp allocTensorOp, Attribute memorySpace,
    Operation *insertionPoint) {
  Location loc = allocTensorOp.getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(insertionPoint ? insertionPoint : allocTensorOp);
  bufferization::BufferizationOptions bufferizationOptions;

  // Create buffer allocation.
  Value alloc = createAllocationForTensor(
      rewriter, loc, allocTensorOp.getResult(), options, memorySpace);

  // Create bufferization.to_tensor with "restrict" and "writable". The returned
  // tensor is a new buffer allocation, so it does not alias with any buffer.
  Value toTensorOp = rewriter.create<bufferization::ToTensorOp>(
      loc, alloc, /*restrict=*/true, /*writable=*/true);
  rewriter.replaceOp(allocTensorOp, toTensorOp);
  return alloc;
}

/// Lower tensor.from_elements to a sequence of chained tensor.insert.
FailureOr<Operation *> mlir::linalg::rewriteInDestinationPassingStyle(
    RewriterBase &rewriter, tensor::FromElementsOp fromElementsOp) {
  Location loc = fromElementsOp.getLoc();
  RankedTensorType tensorType =
      cast<RankedTensorType>(fromElementsOp.getType());
  auto shape = tensorType.getShape();

  // Create tensor.empty.
  auto emptyOp = rewriter.create<EmptyOp>(loc, tensorType, ValueRange());

  // Case: tensor<elem_type>.
  if (shape.empty()) {
    Operation *res = rewriter.replaceOpWithNewOp<tensor::InsertOp>(
        fromElementsOp, fromElementsOp.getElements().front(),
        emptyOp.getResult(), ValueRange());
    return res;
  }

  // Create constants for the range of possible indices [0, max{shape_i}).
  auto maxDim = *llvm::max_element(shape);
  SmallVector<Value, 2> constants;
  constants.reserve(maxDim);
  for (int i = 0; i < maxDim; ++i)
    constants.push_back(rewriter.create<arith::ConstantIndexOp>(loc, i));

  // Traverse all elements and create tensor.insert ops.
  auto elementIt = fromElementsOp.getElements().begin();
  SmallVector<Value, 2> indices(tensorType.getRank(), constants[0]);
  Value result = createInserts(rewriter, loc, /*dim=*/0, emptyOp.getResult(),
                               shape, constants, elementIt, indices);

  // Replace tensor.from_elements.
  rewriter.replaceOp(fromElementsOp, result);
  return result.getDefiningOp();
}

/// Lower tensor.generate to linalg.generic.
FailureOr<Operation *>
mlir::linalg::rewriteInDestinationPassingStyle(RewriterBase &rewriter,
                                               tensor::GenerateOp generateOp) {
  // Only ops with exactly one block are supported.
  if (!generateOp.getBody().hasOneBlock())
    return failure();

  Location loc = generateOp.getLoc();
  RankedTensorType tensorType = cast<RankedTensorType>(generateOp.getType());

  // Create tensor.empty.
  auto emptyOp =
      rewriter.create<EmptyOp>(loc, tensorType, generateOp.getDynamicExtents());

  // Create linalg.generic.
  SmallVector<utils::IteratorType> iteratorTypes(tensorType.getRank(),
                                                 utils::IteratorType::parallel);
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
  rewriter.mergeBlocks(&generateOp.getBody().front(), body, bbArgReplacements);

  // Update terminator.
  auto yieldOp = cast<tensor::YieldOp>(body->getTerminator());
  rewriter.replaceOpWithNewOp<linalg::YieldOp>(yieldOp, yieldOp.getValue());

  // Replace tensor.generate.
  rewriter.replaceOp(generateOp, genericOp->getResult(0));
  return genericOp.getOperation();
}

/// Lower tensor.pad to linalg.generic + tensor.insert_slice.
FailureOr<Operation *>
mlir::linalg::rewriteInDestinationPassingStyle(RewriterBase &rewriter,
                                               tensor::PadOp padOp) {
  // Only ops with exactly one block are supported.
  if (!padOp.getBodyRegion().hasOneBlock())
    return failure();

  // Create tensor.empty.
  Location loc = padOp.getLoc();
  RankedTensorType resultType = padOp.getResultType();
  ReifiedRankedShapedTypeDims reifiedShape;
  if (failed(reifyResultShapes(rewriter, padOp, reifiedShape)))
    return rewriter.notifyMatchFailure(
        padOp, "failed to reify tensor.pad op result shape");
  SmallVector<Value> dynamicSizes;
  for (int64_t i = 0; i < resultType.getRank(); ++i)
    if (resultType.isDynamicDim(i))
      dynamicSizes.push_back(reifiedShape[0][i].get<Value>());

  // If the `padOp` has a nofold attribute and all paddings are known to be 0,
  // explicitly insert a `linalg.copy`.
  if (padOp.getNofoldAttr() &&
      llvm::all_of(padOp.getMixedLowPad(), isZeroIndex) &&
      llvm::all_of(padOp.getMixedHighPad(), isZeroIndex)) {
    using bufferization::AllocTensorOp;
    Value allocated =
        rewriter.create<AllocTensorOp>(loc, resultType, dynamicSizes);
    auto copyOp = rewriter.replaceOpWithNewOp<linalg::CopyOp>(
        padOp, padOp.getSource(), allocated);
    return copyOp.getOperation();
  }

  Value empty = rewriter.create<EmptyOp>(loc, resultType, dynamicSizes);
  // Create linalg.fill or linalg.generic.
  Operation *fillOp = movePaddingToFillOrGenericOp(rewriter, loc, padOp, empty);
  rewriter.setInsertionPointAfter(fillOp);

  // Create tensor::InsertSliceOp.
  SmallVector<OpFoldResult> sliceSizes =
      getMixedSizes(rewriter, loc, padOp.getSource());
  SmallVector<OpFoldResult> sliceStrides(resultType.getRank(),
                                         rewriter.getIndexAttr(1));
  auto insertSliceOp = rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
      padOp, padOp.getSource(), fillOp->getResult(0),
      /*offsets=*/padOp.getMixedLowPad(), sliceSizes, sliceStrides);
  return insertSliceOp.getOperation();
}

Value linalg::bufferizeToAllocation(
    RewriterBase &rewriter, const linalg::BufferizeToAllocationOptions &options,
    Operation *op, Attribute memorySpace, Operation *insertionPoint) {
  using namespace bufferization;

  // Call specialized overload for certain ops.
  if (auto padOp = dyn_cast<tensor::PadOp>(op))
    return bufferizeToAllocation(rewriter, options, padOp, memorySpace);
  if (auto maskOp = dyn_cast<vector::MaskOp>(op))
    return bufferizeToAllocation(rewriter, options, maskOp, memorySpace);
  if (auto allocTensorOp = dyn_cast<bufferization::AllocTensorOp>(op))
    return bufferizeToAllocation(rewriter, options, allocTensorOp, memorySpace);

  // Only bufferizable ops are supported.
  auto bufferizableOp = dyn_cast<BufferizableOpInterface>(op);
  if (!bufferizableOp)
    return nullptr;
  BufferizationOptions bufferizationOptions;
  AnalysisState state(bufferizationOptions);

#ifndef NDEBUG
  if (!options.bufferizeDestinationOnly) {
    // Ops with nested tensor ops are not supported yet. At the moment, this
    // function just bufferizes the given op itself, but not its body.
    op->walk([&](Operation *nestedOp) {
      if (op == nestedOp)
        return;
      if (llvm::any_of(nestedOp->getOperands(),
                       [](Value v) { return v.getType().isa<TensorType>(); }))
        llvm_unreachable("ops with nested tensor ops are not supported yet");
      if (llvm::any_of(nestedOp->getResults(),
                       [](Value v) { return v.getType().isa<TensorType>(); }))
        llvm_unreachable("ops with nested tensor ops are not supported yet");
    });
  }
#endif // NDEBUG

  // Gather tensor results.
  SmallVector<OpResult> tensorResults;
  for (OpResult result : op->getResults()) {
    if (!result.getType().isa<TensorType>())
      continue;
    // Unranked tensors are not supported
    if (!isa<RankedTensorType>(result.getType()))
      return nullptr;
    // Ops that bufferize to an allocation are not supported.
    if (bufferizableOp.bufferizesToAllocation(result))
      return nullptr;
    tensorResults.push_back(result);
  }

  // Gather all operands that should bufferize to a new allocation. I.e.,
  // bufferize out-of-place.
  SmallVector<OpOperand *> outOfPlaceOperands, resultUses;
  auto addOutOfPlaceOperand = [&](OpOperand *operand) {
    if (!llvm::is_contained(outOfPlaceOperands, operand))
      outOfPlaceOperands.push_back(operand);
  };
  for (OpResult result : tensorResults) {
    AliasingOpOperandList aliasingOperands =
        state.getAliasingOpOperands(result);
    for (const AliasingOpOperand &operand : aliasingOperands) {
      addOutOfPlaceOperand(operand.opOperand);
      for (OpOperand &resultUse : result.getUses())
        resultUses.push_back(&resultUse);
    }
  }
  for (OpOperand &operand : op->getOpOperands()) {
    if (!state.bufferizesToMemoryWrite(operand))
      continue;
    if (!isa<RankedTensorType>(operand.get().getType()))
      continue;
    addOutOfPlaceOperand(&operand);
  }
  // TODO: Support multiple buffers.
  if (outOfPlaceOperands.size() != 1)
    return nullptr;

  // Allocate buffers.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(insertionPoint ? insertionPoint : op);
  SmallVector<Value> allocs;
  for (OpOperand *operand : outOfPlaceOperands) {
    Value alloc = createAllocationForTensor(
        rewriter, op->getLoc(), operand->get(), options, memorySpace);
    allocs.push_back(alloc);
    if (!state.findDefinitions(operand->get()).empty()) {
      // Initialize buffer with a copy of the operand data. Not needed if the
      // tensor is uninitialized.
      createMemcpy(rewriter, op->getLoc(), operand->get(), alloc, options);
    }
    rewriter.modifyOpInPlace(op, [&]() {
      auto toTensorOp = rewriter.create<ToTensorOp>(op->getLoc(), alloc);
      operand->set(toTensorOp);
      if (options.bufferizeDestinationOnly) {
        rewriter.modifyOpInPlace(toTensorOp, [&]() {
          toTensorOp.setRestrict(true);
          toTensorOp.setWritable(true);
        });
      }
    });
  }

  if (options.bufferizeDestinationOnly)
    return allocs.front();

  // Bufferize the op.
  rewriter.setInsertionPoint(op);
  if (failed(bufferizableOp.bufferize(rewriter, bufferizationOptions)))
    return nullptr;

  // Set "restrict" attribute, indicating that no other tensor aliases with
  // this tensor. That is because we just allocated a new buffer for the tensor.
  for (OpOperand *resultUse : resultUses) {
    auto toTensorOp = resultUse->get().getDefiningOp<ToTensorOp>();
    assert(toTensorOp && "expected to_tensor op");
    rewriter.modifyOpInPlace(toTensorOp, [&]() {
      toTensorOp.setRestrict(true);
      toTensorOp.setWritable(true);
    });
  }
  return allocs.front();
}

namespace {

template <typename OpTy>
LogicalResult rewriteOpInDestinationPassingStyle(OpTy op,
                                                 PatternRewriter &rewriter) {
  return linalg::rewriteInDestinationPassingStyle(rewriter, op);
}

} // namespace

void linalg::populateConvertToDestinationStylePatterns(
    RewritePatternSet &patterns) {
  patterns.add(rewriteOpInDestinationPassingStyle<tensor::FromElementsOp>);
  patterns.add(rewriteOpInDestinationPassingStyle<tensor::GenerateOp>);
  patterns.add(rewriteOpInDestinationPassingStyle<tensor::PadOp>);
}
