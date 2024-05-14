//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::tensor;

namespace mlir {
namespace tensor {
namespace {

struct CastOpInterface
    : public BufferizableOpInterface::ExternalModel<CastOpInterface,
                                                    tensor::CastOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {{op->getResult(0), BufferRelation::Equivalent}};
  }

  FailureOr<BaseMemRefType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                SmallVector<Value> &invocationStack) const {
    auto castOp = cast<tensor::CastOp>(op);
    auto maybeSrcBufferType = bufferization::getBufferType(
        castOp.getSource(), options, invocationStack);
    if (failed(maybeSrcBufferType))
      return failure();
    Attribute memorySpace = maybeSrcBufferType->getMemorySpace();

    // Note: `getMemRefTypeWithFullyDynamicLayout` returns an unranked memref
    // type in case the input is an unranked tensor type.

    // Case 1: Casting an unranked tensor
    if (isa<UnrankedTensorType>(castOp.getSource().getType())) {
      // When casting to a ranked tensor, we cannot infer any static offset or
      // strides from the source. Assume fully dynamic.
      return getMemRefTypeWithFullyDynamicLayout(castOp.getType(), memorySpace);
    }

    // Case 2: Casting to an unranked tensor type
    if (isa<UnrankedTensorType>(castOp.getType())) {
      return getMemRefTypeWithFullyDynamicLayout(castOp.getType(), memorySpace);
    }

    // Case 3: Ranked tensor -> ranked tensor. The offsets and strides do not
    // change.
    auto rankedResultType = cast<RankedTensorType>(castOp.getType());
    return MemRefType::get(
        rankedResultType.getShape(), rankedResultType.getElementType(),
        llvm::cast<MemRefType>(*maybeSrcBufferType).getLayout(), memorySpace);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto castOp = cast<tensor::CastOp>(op);

    // The result buffer still has the old (pre-cast) type.
    FailureOr<Value> resultBuffer =
        getBuffer(rewriter, castOp.getSource(), options);
    if (failed(resultBuffer))
      return failure();

    // Compute the new type.
    auto resultMemRefType =
        bufferization::getBufferType(castOp.getResult(), options);
    if (failed(resultMemRefType))
      return failure();
    if (resultBuffer->getType() == *resultMemRefType) {
      // This cast is a no-op.
      replaceOpWithBufferizedValues(rewriter, op, *resultBuffer);
      return success();
    }

    // Replace the op with a memref.cast.
    assert(memref::CastOp::areCastCompatible(resultBuffer->getType(),
                                             *resultMemRefType) &&
           "CallOp::bufferize: cast incompatible");
    replaceOpWithNewBufferizedOp<memref::CastOp>(
        rewriter, op, *resultMemRefType, *resultBuffer);

    return success();
  }
};

/// Bufferization of tensor.collapse_shape. Replace with memref.collapse_shape.
struct CollapseShapeOpInterface
    : public BufferizableOpInterface::ExternalModel<CollapseShapeOpInterface,
                                                    tensor::CollapseShapeOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    // tensor.collapse_shape may reallocate, at which point the source buffer is
    // copied. I.e., there will be a memory read side effect on the bufferized
    // source. This function conservatively returns "true" because whether a
    // copy will be created or not is not known at this point.
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    // TODO: CollapseShapeOp may allocate at runtime.
    return {{op->getOpResult(0), BufferRelation::Equivalent}};
  }

  FailureOr<BaseMemRefType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                SmallVector<Value> &invocationStack) const {
    auto collapseShapeOp = cast<tensor::CollapseShapeOp>(op);
    auto maybeSrcBufferType = bufferization::getBufferType(
        collapseShapeOp.getSrc(), options, invocationStack);
    if (failed(maybeSrcBufferType))
      return failure();
    auto srcBufferType = llvm::cast<MemRefType>(*maybeSrcBufferType);
    bool canBeCollapsed = memref::CollapseShapeOp::isGuaranteedCollapsible(
        srcBufferType, collapseShapeOp.getReassociationIndices());

    if (!canBeCollapsed) {
      // If dims cannot be collapsed, this op bufferizes to a new allocation.
      RankedTensorType tensorResultType = collapseShapeOp.getResultType();
      return bufferization::getMemRefTypeWithStaticIdentityLayout(
          tensorResultType, srcBufferType.getMemorySpace());
    }

    return memref::CollapseShapeOp::computeCollapsedType(
        srcBufferType, collapseShapeOp.getReassociationIndices());
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto collapseShapeOp = cast<tensor::CollapseShapeOp>(op);
    RankedTensorType tensorResultType = collapseShapeOp.getResultType();
    FailureOr<Value> maybeBuffer =
        getBuffer(rewriter, collapseShapeOp.getSrc(), options);
    if (failed(maybeBuffer))
      return failure();
    Value buffer = *maybeBuffer;
    auto bufferType = cast<MemRefType>(buffer.getType());

    if (tensorResultType.getRank() == 0) {
      // 0-d collapses must go through a different op builder.
      MemRefType resultType;

      if (bufferType.getLayout().isIdentity()) {
        // Standard layout: result type has no offset.
        MemRefLayoutAttrInterface layout;
        resultType = MemRefType::get({}, tensorResultType.getElementType(),
                                     layout, bufferType.getMemorySpace());
      } else {
        // Source memref has a layout map: result type has the same offset as
        // the source type.
        SmallVector<int64_t> strides;
        int64_t offset;
        if (failed(getStridesAndOffset(bufferType, strides, offset)))
          return failure();
        resultType = MemRefType::get(
            {}, tensorResultType.getElementType(),
            StridedLayoutAttr::get(op->getContext(), offset, {}),
            bufferType.getMemorySpace());
      }

      replaceOpWithNewBufferizedOp<memref::CollapseShapeOp>(
          rewriter, op, resultType, buffer, collapseShapeOp.getReassociation());
      return success();
    }

    // If the dims are not collapsible (due to an incompatible source layout
    // map), force an out-of-place bufferization, i.e., a buffer copy. This
    // newly allocated buffer will have no layout map and thus be collapsible.
    bool canBeCollapsed = memref::CollapseShapeOp::isGuaranteedCollapsible(
        bufferType, collapseShapeOp.getReassociationIndices());
    if (!canBeCollapsed) {
      // TODO: Create alloc_tensor ops during TensorCopyInsertion.
      AnalysisState analysisState(options);
      FailureOr<Value> tensorAlloc = allocateTensorForShapedValue(
          rewriter, op->getLoc(), collapseShapeOp.getSrc(), options);
      if (failed(tensorAlloc))
        return failure();
      auto memrefType =
          MemRefType::get(collapseShapeOp.getSrcType().getShape(),
                          collapseShapeOp.getSrcType().getElementType(),
                          AffineMap(), bufferType.getMemorySpace());
      buffer = rewriter.create<bufferization::ToMemrefOp>(
          op->getLoc(), memrefType, *tensorAlloc);
    }

    // Result type is inferred by the builder.
    replaceOpWithNewBufferizedOp<memref::CollapseShapeOp>(
        rewriter, op, buffer, collapseShapeOp.getReassociationIndices());
    return success();
  }
};

/// Bufferization of tensor.dim. Replace with memref.dim.
struct DimOpInterface
    : public BufferizableOpInterface::ExternalModel<DimOpInterface,
                                                    tensor::DimOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    // The op reads the tensor's metadata but not its contents.
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto dimOp = cast<tensor::DimOp>(op);
    FailureOr<Value> v = getBuffer(rewriter, dimOp.getSource(), options);
    if (failed(v))
      return failure();
    replaceOpWithNewBufferizedOp<memref::DimOp>(rewriter, op, *v,
                                                dimOp.getIndex());
    return success();
  }
};

/// Bufferization of "tensor.empty". Replace with "bufferization.alloc_tensor".
struct EmptyOpInterface
    : public BufferizableOpInterface::ExternalModel<EmptyOpInterface,
                                                    tensor::EmptyOp> {
  bool bufferizesToAllocation(Operation *op, Value value) const { return true; }

  bool resultBufferizesToMemoryWrite(Operation *op, OpResult opResult,
                                     const AnalysisState &state) const {
    // The returned tensor does not have specified contents.
    return false;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto emptyOp = cast<tensor::EmptyOp>(op);

    // Optimization: Fold away the op if it has no uses.
    if (op->getUses().empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    // Allocate a tensor. This emits a "bufferization.alloc_tensor" op.
    FailureOr<Value> allocTensor = allocateTensorForShapedValue(
        rewriter, op->getLoc(), emptyOp.getResult(), options, /*copy=*/false);
    if (failed(allocTensor))
      return failure();
    rewriter.replaceOp(op, *allocTensor);
    return success();
  }
};

/// Bufferization of tensor.expand_shape. Replace with memref.expand_shape.
struct ExpandShapeOpInterface
    : public BufferizableOpInterface::ExternalModel<ExpandShapeOpInterface,
                                                    tensor::ExpandShapeOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    // In contrast to tensor.collapse_shape, this op can always be bufferized
    // without a copy.
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {{op->getOpResult(0), BufferRelation::Equivalent}};
  }

  FailureOr<BaseMemRefType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                SmallVector<Value> &invocationStack) const {
    auto expandShapeOp = cast<tensor::ExpandShapeOp>(op);
    auto maybeSrcBufferType = bufferization::getBufferType(
        expandShapeOp.getSrc(), options, invocationStack);
    if (failed(maybeSrcBufferType))
      return failure();
    auto srcBufferType = llvm::cast<MemRefType>(*maybeSrcBufferType);
    auto maybeResultType = memref::ExpandShapeOp::computeExpandedType(
        srcBufferType, expandShapeOp.getResultType().getShape(),
        expandShapeOp.getReassociationIndices());
    if (failed(maybeResultType))
      return failure();
    return *maybeResultType;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto expandShapeOp = cast<tensor::ExpandShapeOp>(op);
    auto tensorResultType = expandShapeOp.getResultType();
    FailureOr<Value> buffer =
        getBuffer(rewriter, expandShapeOp.getSrc(), options);
    if (failed(buffer))
      return failure();

    // Memref result type is inferred by the builder based on reassociation
    // indices and result shape.
    // TODO: Instead of inferring the output shape argument of
    // memref.expand_shape op, use output_shape argument of tensor.expand_shape
    // op.
    replaceOpWithNewBufferizedOp<memref::ExpandShapeOp>(
        rewriter, op, tensorResultType.getShape(), *buffer,
        expandShapeOp.getReassociationIndices());
    return success();
  }
};

/// Bufferization of tensor.extract_slice. Replace with memref.subview.
struct ExtractSliceOpInterface
    : public BufferizableOpInterface::ExternalModel<ExtractSliceOpInterface,
                                                    tensor::ExtractSliceOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {{op->getOpResult(0), BufferRelation::Unknown}};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto extractSliceOp = cast<tensor::ExtractSliceOp>(op);
    SmallVector<OpFoldResult> mixedOffsets = extractSliceOp.getMixedOffsets();
    SmallVector<OpFoldResult> mixedSizes = extractSliceOp.getMixedSizes();
    SmallVector<OpFoldResult> mixedStrides = extractSliceOp.getMixedStrides();
    Location loc = extractSliceOp.getLoc();

    // Get source buffer.
    FailureOr<Value> srcMemref =
        getBuffer(rewriter, extractSliceOp.getSource(), options);
    if (failed(srcMemref))
      return failure();

    // Take a subview of the source buffer.
    auto resultMemrefType =
        bufferization::getBufferType(extractSliceOp.getResult(), options);
    if (failed(resultMemrefType))
      return failure();
    Value subView = rewriter.create<memref::SubViewOp>(
        loc, llvm::cast<MemRefType>(*resultMemrefType), *srcMemref, mixedOffsets,
        mixedSizes, mixedStrides);

    replaceOpWithBufferizedValues(rewriter, op, subView);
    return success();
  }

  FailureOr<BaseMemRefType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                SmallVector<Value> &invocationStack) const {
    auto extractSliceOp = cast<tensor::ExtractSliceOp>(op);
    assert(value == extractSliceOp.getResult() && "invalid value");
    auto srcMemrefType = bufferization::getBufferType(
        extractSliceOp.getSource(), options, invocationStack);
    if (failed(srcMemrefType))
      return failure();
    SmallVector<OpFoldResult> mixedOffsets = extractSliceOp.getMixedOffsets();
    SmallVector<OpFoldResult> mixedSizes = extractSliceOp.getMixedSizes();
    SmallVector<OpFoldResult> mixedStrides = extractSliceOp.getMixedStrides();
    return cast<BaseMemRefType>(memref::SubViewOp::inferRankReducedResultType(
        extractSliceOp.getType().getShape(), llvm::cast<MemRefType>(*srcMemrefType),
        mixedOffsets, mixedSizes, mixedStrides));
  }
};

/// Bufferization of tensor.extract. Replace with memref.load.
struct ExtractOpInterface
    : public BufferizableOpInterface::ExternalModel<ExtractOpInterface,
                                                    tensor::ExtractOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto extractOp = cast<tensor::ExtractOp>(op);
    FailureOr<Value> srcMemref =
        getBuffer(rewriter, extractOp.getTensor(), options);
    if (failed(srcMemref))
      return failure();
    replaceOpWithNewBufferizedOp<memref::LoadOp>(rewriter, op, *srcMemref,
                                                 extractOp.getIndices());
    return success();
  }
};

// Implements backtracking to traverse indices of the output buffer while
// iterating over op.elements().
static void createStores(RewriterBase &rewriter, Location loc, int dim,
                         Value buffer, ArrayRef<int64_t> shape,
                         ArrayRef<Value> constants,
                         OperandRange::iterator &elementIt,
                         SmallVectorImpl<Value> &indices) {
  if (dim == static_cast<int>(shape.size()) - 1) {
    for (int i = 0; i < shape.back(); ++i) {
      indices.back() = constants[i];
      rewriter.create<memref::StoreOp>(loc, *elementIt, buffer, indices);
      ++elementIt;
    }
    return;
  }
  for (int i = 0; i < shape[dim]; ++i) {
    indices[dim] = constants[i];
    createStores(rewriter, loc, dim + 1, buffer, shape, constants, elementIt,
                 indices);
  }
}

/// Bufferization of tensor.from_elements.
struct FromElementsOpInterface
    : public BufferizableOpInterface::ExternalModel<FromElementsOpInterface,
                                                    tensor::FromElementsOp> {

  bool bufferizesToAllocation(Operation *op, Value value) const { return true; }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto fromElementsOp = cast<tensor::FromElementsOp>(op);
    auto tensorType = cast<RankedTensorType>(fromElementsOp.getType());

    // TODO: Implement memory space for this op.
    if (options.defaultMemorySpaceFn(tensorType) != Attribute())
      return op->emitError("memory space not implemented yet");

    // Allocate a buffer for the result.
    Location loc = op->getLoc();
    auto shape = tensorType.getShape();
    // TODO: Create alloc_tensor ops during TensorCopyInsertion.
    FailureOr<Value> tensorAlloc = allocateTensorForShapedValue(
        rewriter, loc, fromElementsOp.getResult(), options,
        /*copy=*/false);
    if (failed(tensorAlloc))
      return failure();
    auto memrefType =
        MemRefType::get(tensorType.getShape(), tensorType.getElementType());
    Value buffer = rewriter.create<bufferization::ToMemrefOp>(
        op->getLoc(), memrefType, *tensorAlloc);

    // Case: tensor<0xelem_type>.
    if (fromElementsOp.getElements().empty()) {
      replaceOpWithBufferizedValues(rewriter, op, buffer);
      return success();
    }

    // Case: tensor<elem_type>.
    if (shape.empty()) {
      rewriter.create<memref::StoreOp>(
          loc, fromElementsOp.getElements().front(), buffer);
      replaceOpWithBufferizedValues(rewriter, op, buffer);
      return success();
    }

    // Create constants for the range of possible indices [0, max{shape_i}).
    auto maxDim = *llvm::max_element(shape);
    SmallVector<Value, 2> constants;
    constants.reserve(maxDim);
    for (int i = 0; i < maxDim; ++i)
      constants.push_back(rewriter.create<arith::ConstantIndexOp>(loc, i));

    // Traverse all `elements` and create `memref.store` ops.
    auto elementIt = fromElementsOp.getElements().begin();
    SmallVector<Value, 2> indices(tensorType.getRank(), constants[0]);
    createStores(rewriter, loc, /*dim=*/0, buffer, shape, constants, elementIt,
                 indices);

    replaceOpWithBufferizedValues(rewriter, op, buffer);

    return success();
  }
};

/// Lower the body of a tensor.generate like op (one index-typed bbArg per dim).
/// Such ops are lowered to linalg.map with the given tensor as a destination.
///
/// Example:
/// ```
/// %r = tensor.generate %x, %y {
///   ^bb0(%arg0: index, %arg1: index):
///   %0 = "some_op"(%arg0, %arg1) : (index, index) -> (index)
///   tensor.yield %0 : index
/// } : tensor<?x?xindex>
/// ```
///
/// Is lowered to:
/// ```
/// linalg.map ins() outs(%dest) {
///   %d0 = linalg.index 0 : index
///   %d1 = linalg.index 1 : index
///   %0 = "some_op"(%d0, %d1) : (index, index) -> (index)
///   linalg.yield %0 : index
/// }
/// ```
static Value lowerGenerateLikeOpBody(RewriterBase &rewriter, Location loc,
                                     Value tensorDestination,
                                     ValueRange dynamicSizes,
                                     Region &generateBody) {
  assert(generateBody.hasOneBlock() && "expected body with single block");
  auto tensorType = cast<RankedTensorType>(tensorDestination.getType());
  assert(generateBody.getNumArguments() == tensorType.getRank() &&
         "rank mismatch");

  // Create linalg::MapOp.
  OpBuilder::InsertionGuard g(rewriter);
  auto linalgOp =
      rewriter.create<linalg::MapOp>(loc, tensorType, /*inputs=*/ValueRange(),
                                     /*init=*/tensorDestination);
  Block &linalgBody = linalgOp.getMapper().emplaceBlock();

  // Create linalg::IndexOps.
  rewriter.setInsertionPointToStart(&linalgBody);
  SmallVector<Value> indices;
  for (int64_t dim = 0; dim < tensorType.getRank(); ++dim)
    indices.push_back(rewriter.create<linalg::IndexOp>(loc, dim));

  // Move over body.
  rewriter.mergeBlocks(&generateBody.front(), &linalgBody, indices);
  auto yieldOp = cast<tensor::YieldOp>(linalgBody.getTerminator());
  rewriter.replaceOpWithNewOp<linalg::YieldOp>(yieldOp, yieldOp.getValue());

  return linalgOp.getResult()[0];
}

/// Bufferization of tensor.generate.
struct GenerateOpInterface
    : public BufferizableOpInterface::ExternalModel<GenerateOpInterface,
                                                    tensor::GenerateOp> {

  bool bufferizesToAllocation(Operation *op, Value value) const { return true; }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto generateOp = cast<tensor::GenerateOp>(op);

    auto type = generateOp.getResult().getType();

    // TODO: Implement memory space for this op.
    if (options.defaultMemorySpaceFn(type) != Attribute())
      return op->emitError("memory space not implemented yet");

    // Allocate memory.
    Location loc = op->getLoc();
    FailureOr<Value> tensorAlloc = allocateTensorForShapedValue(
        rewriter, loc, generateOp.getResult(), options,
        /*copy=*/false);
    if (failed(tensorAlloc))
      return failure();

    Value result = lowerGenerateLikeOpBody(rewriter, loc, *tensorAlloc,
                                           generateOp.getDynamicExtents(),
                                           generateOp.getBody());
    rewriter.replaceOp(generateOp, result);

    return success();
  }
};

/// Bufferization of tensor.insert. Replace with memref.store.
///
/// Note: DstBufferizableOpInterfaceExternalModel provides many default method
/// implementations for DestinationStyle ops.
struct InsertOpInterface
    : public DstBufferizableOpInterfaceExternalModel<InsertOpInterface,
                                                     tensor::InsertOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto insertOp = cast<tensor::InsertOp>(op);
    FailureOr<Value> destMemref =
        getBuffer(rewriter, insertOp.getDest(), options);
    if (failed(destMemref))
      return failure();
    rewriter.create<memref::StoreOp>(insertOp.getLoc(), insertOp.getScalar(),
                                     *destMemref, insertOp.getIndices());
    replaceOpWithBufferizedValues(rewriter, op, *destMemref);
    return success();
  }
};

/// Bufferization of tensor.insert_slice. Replace with a memory copy. Under
/// certain circumstances, this op can also be a no-op.
///
/// Note: DstBufferizableOpInterfaceExternalModel provides many default method
/// implementations for DestinationStyle ops.
struct InsertSliceOpInterface
    : public DstBufferizableOpInterfaceExternalModel<InsertSliceOpInterface,
                                                     tensor::InsertSliceOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    auto insertSliceOp = cast<tensor::InsertSliceOp>(op);
    RankedTensorType destType = insertSliceOp.getDestType();

    // The source is always read.
    if (opOperand == insertSliceOp.getSourceMutable())
      return true;

    // For the destination, it depends...
    assert(opOperand == insertSliceOp.getDestMutable() && "expected dest");

    // Dest is not read if it is entirely overwritten. E.g.:
    // tensor.insert_slice %a into %t[0][10][1] : ... into tensor<10xf32>
    bool allOffsetsZero =
        llvm::all_of(insertSliceOp.getMixedOffsets(), [](OpFoldResult ofr) {
          return isConstantIntValue(ofr, 0);
        });
    bool sizesMatchDestSizes = llvm::all_of(
        llvm::enumerate(insertSliceOp.getMixedSizes()), [&](const auto &it) {
          return getConstantIntValue(it.value()) ==
                 destType.getDimSize(it.index());
        });
    bool allStridesOne =
        llvm::all_of(insertSliceOp.getMixedStrides(), [](OpFoldResult ofr) {
          return isConstantIntValue(ofr, 1);
        });
    return !(allOffsetsZero && sizesMatchDestSizes && allStridesOne);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    // insert_slice ops arise from tiling and bufferizing them out-of-place is
    // generally a deal breaker. When used with loops, this ends up cloning the
    // whole tensor on every single iteration and is a symptom of a
    // catastrophically bad scheduling decision.
    // TODO: be very loud about it or even consider failing the pass.
    auto insertSliceOp = cast<tensor::InsertSliceOp>(op);
    SmallVector<OpFoldResult> mixedOffsets = insertSliceOp.getMixedOffsets();
    SmallVector<OpFoldResult> mixedSizes = insertSliceOp.getMixedSizes();
    SmallVector<OpFoldResult> mixedStrides = insertSliceOp.getMixedStrides();
    Location loc = insertSliceOp.getLoc();

    // Get destination buffer.
    FailureOr<Value> dstMemref =
        getBuffer(rewriter, insertSliceOp.getDest(), options);
    if (failed(dstMemref))
      return failure();

    // Take a subview of the destination buffer.
    auto dstMemrefType = cast<MemRefType>(dstMemref->getType());
    auto subviewMemRefType =
        cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
            insertSliceOp.getSourceType().getShape(), dstMemrefType,
            mixedOffsets, mixedSizes, mixedStrides));
    Value subView = rewriter.create<memref::SubViewOp>(
        loc, subviewMemRefType, *dstMemref, mixedOffsets, mixedSizes,
        mixedStrides);

    // Copy tensor. If this tensor.insert_slice has a matching
    // tensor.extract_slice, the copy operation will eventually fold away.
    FailureOr<Value> srcMemref =
        getBuffer(rewriter, insertSliceOp.getSource(), options);
    if (failed(srcMemref))
      return failure();
    if (failed(options.createMemCpy(rewriter, loc, *srcMemref, subView)))
      return failure();

    replaceOpWithBufferizedValues(rewriter, op, *dstMemref);
    return success();
  }
};

/// Bufferization of tensor.pad. Replace with bufferization.alloc_tensor +
/// linalg.map + insert_slice.
/// For best performance, vectorize before bufferization (better performance in
/// case of padding with a constant).
struct PadOpInterface
    : public BufferizableOpInterface::ExternalModel<PadOpInterface,
                                                    tensor::PadOp> {
  bool bufferizesToAllocation(Operation *op, Value value) const { return true; }

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {};
  }

  FailureOr<BaseMemRefType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                SmallVector<Value> &invocationStack) const {
    // Infer memory space from the source tensor.
    auto padOp = cast<tensor::PadOp>(op);
    auto maybeSrcBufferType = bufferization::getBufferType(
        padOp.getSource(), options, invocationStack);
    if (failed(maybeSrcBufferType))
      return failure();
    MemRefLayoutAttrInterface layout;
    return MemRefType::get(padOp.getResultType().getShape(),
                           padOp.getResultType().getElementType(), layout,
                           maybeSrcBufferType->getMemorySpace());
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto padOp = cast<tensor::PadOp>(op);
    Location loc = padOp.getLoc();
    RankedTensorType resultType = padOp.getResultType();
    RankedTensorType srcType = padOp.getSourceType();

    auto toValue = [&](OpFoldResult ofr) {
      if (ofr.is<Value>())
        return ofr.get<Value>();
      return rewriter
          .create<arith::ConstantIndexOp>(loc, *getConstantIntValue(ofr))
          .getResult();
    };

    // Compute dynamic result dimensions.
    SmallVector<OpFoldResult> mixedLowPad = padOp.getMixedLowPad();
    SmallVector<OpFoldResult> mixedHighPad = padOp.getMixedHighPad();
    SmallVector<Value> dynamicSizes;
    for (int64_t i = 0; i < resultType.getRank(); ++i) {
      if (!resultType.isDynamicDim(i))
        continue;
      Value srcDim = rewriter.create<tensor::DimOp>(loc, padOp.getSource(), i);
      Value lowPad = toValue(mixedLowPad[i]);
      Value highPad = toValue(mixedHighPad[i]);
      AffineExpr s0, s1, s2;
      bindSymbols(op->getContext(), s0, s1, s2);
      AffineExpr sumExpr = s0 + s1 + s2;
      Value sum = rewriter.create<affine::AffineApplyOp>(
          loc, sumExpr, ValueRange{srcDim, lowPad, highPad});
      dynamicSizes.push_back(sum);
    }

    // Allocate a buffer for the padded result.
    FailureOr<Value> tensorAlloc =
        allocateTensorForShapedValue(rewriter, loc, padOp.getResult(), options,
                                     /*copy=*/false);
    if (failed(tensorAlloc))
      return failure();

    // tensor::PadOp is like tensor::GenerateOp: The only difference is that
    // only a part of the generated tensor is needed. For simplicity, we reuse
    // the same functionality here.
    Value filledBuffer = lowerGenerateLikeOpBody(
        rewriter, loc, *tensorAlloc, dynamicSizes, padOp.getBodyRegion());

    // Create tensor::InsertSliceOp.
    SmallVector<OpFoldResult> sliceSizes =
        getMixedSizes(rewriter, loc, padOp.getSource());
    SmallVector<OpFoldResult> sliceStrides(srcType.getRank(),
                                           rewriter.getIndexAttr(1));
    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        padOp, padOp.getSource(), filledBuffer,
        /*offsets=*/padOp.getMixedLowPad(), sliceSizes, sliceStrides);

    return success();
  }
};

/// Bufferization of tensor.rank. Replace with memref.rank.
struct RankOpInterface
    : public BufferizableOpInterface::ExternalModel<RankOpInterface,
                                                    tensor::RankOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    // The op reads the tensor's metadata but not its contents.
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto rankOp = cast<tensor::RankOp>(op);
    FailureOr<Value> v = getBuffer(rewriter, rankOp.getTensor(), options);
    if (failed(v))
      return failure();
    replaceOpWithNewBufferizedOp<memref::RankOp>(rewriter, op, rankOp.getType(),
                                                 *v);
    return success();
  }
};

/// Bufferization of tensor.reshape. Replace with memref.reshape.
struct ReshapeOpInterface
    : public BufferizableOpInterface::ExternalModel<ReshapeOpInterface,
                                                    tensor::ReshapeOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    // Depending on the layout map, the source buffer may have to be copied.
    auto reshapeOp = cast<tensor::ReshapeOp>(op);
    return opOperand == reshapeOp.getShapeMutable();
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {{op->getOpResult(0), BufferRelation::Equivalent}};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto reshapeOp = cast<tensor::ReshapeOp>(op);
    FailureOr<Value> srcBuffer =
        getBuffer(rewriter, reshapeOp.getSource(), options);
    FailureOr<Value> shapeBuffer =
        getBuffer(rewriter, reshapeOp.getShape(), options);
    if (failed(srcBuffer) || failed(shapeBuffer))
      return failure();
    auto maybeResultMemRefType =
        bufferization::getBufferType(reshapeOp.getResult(), options);
    if (failed(maybeResultMemRefType))
      return failure();

    // memref.reshape requires the source buffer to have an identity layout.
    // If the source memref does not have an identity layout, copy the source
    // into a new buffer with an identity layout.
    auto srcType = llvm::dyn_cast<MemRefType>(srcBuffer->getType());
    if (srcType && !srcType.getLayout().isIdentity()) {
      FailureOr<Value> tensorAlloc = allocateTensorForShapedValue(
          rewriter, op->getLoc(), reshapeOp.getSource(), options);
      if (failed(tensorAlloc))
        return failure();
      auto memrefType = MemRefType::get(
          srcType.getShape(), srcType.getElementType(), AffineMap(),
          cast<BaseMemRefType>(srcBuffer->getType()).getMemorySpace());
      srcBuffer = rewriter
                      .create<bufferization::ToMemrefOp>(
                          op->getLoc(), memrefType, *tensorAlloc)
                      .getResult();
    }

    replaceOpWithNewBufferizedOp<memref::ReshapeOp>(
        rewriter, op, maybeResultMemRefType.value(), *srcBuffer, *shapeBuffer);
    return success();
  }

  FailureOr<BaseMemRefType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                SmallVector<Value> &invocationStack) const {
    auto reshapeOp = cast<tensor::ReshapeOp>(op);
    assert(value == reshapeOp.getResult() && "unexpected value provided");
    auto maybeSourceBufferType = bufferization::getBufferType(
        reshapeOp.getSource(), options, invocationStack);
    if (failed(maybeSourceBufferType))
      return failure();
    return getMemRefTypeWithStaticIdentityLayout(
        reshapeOp.getResult().getType(),
        cast<BaseMemRefType>(maybeSourceBufferType.value()).getMemorySpace());
  }
};

/// Analysis of ParallelInsertSliceOp.
struct ParallelInsertSliceOpInterface
    : public BufferizableOpInterface::ExternalModel<
          ParallelInsertSliceOpInterface, ParallelInsertSliceOp> {
  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {};
  }

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    auto parallelInsertSliceOp = cast<ParallelInsertSliceOp>(op);
    return opOperand == parallelInsertSliceOp.getDestMutable();
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    OpBuilder::InsertionGuard g(rewriter);
    auto parallelInsertSliceOp = cast<ParallelInsertSliceOp>(op);
    ParallelCombiningOpInterface parallelCombiningParent =
        parallelInsertSliceOp.getParallelCombiningParent();

    // Bufferize the op outside of the parallel combining terminator.
    rewriter.setInsertionPoint(parallelCombiningParent);

    // Get source and destination buffers.
    FailureOr<Value> destBuffer =
        getBuffer(rewriter, parallelInsertSliceOp.getDest(), options);
    if (failed(destBuffer))
      return failure();
    FailureOr<Value> srcBuffer =
        getBuffer(rewriter, parallelInsertSliceOp.getSource(), options);
    if (failed(srcBuffer))
      return failure();

    // Take a subview of the destination buffer.
    auto destBufferType = cast<MemRefType>(destBuffer->getType());
    auto subviewMemRefType =
        cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
            parallelInsertSliceOp.getSourceType().getShape(), destBufferType,
            parallelInsertSliceOp.getMixedOffsets(),
            parallelInsertSliceOp.getMixedSizes(),
            parallelInsertSliceOp.getMixedStrides()));
    Value subview = rewriter.create<memref::SubViewOp>(
        parallelInsertSliceOp.getLoc(), subviewMemRefType, *destBuffer,
        parallelInsertSliceOp.getMixedOffsets(),
        parallelInsertSliceOp.getMixedSizes(),
        parallelInsertSliceOp.getMixedStrides());

    // This memcpy will fold away if everything bufferizes in-place.
    if (failed(options.createMemCpy(rewriter, parallelInsertSliceOp.getLoc(),
                                    *srcBuffer, subview)))
      return failure();

    // In case the source was allocated in the same block, make sure that the
    // deallocation op (if any) appears after the memcpy. By default, deallocs
    // are placed before the terminator, but this does not work for ForallOp
    // because the terminator does more than just yielding a value.
    //
    // Note: This is not a problem for the destination buffer because these are
    // assumed to always bufferize in-place.
    for (Operation *user : srcBuffer->getUsers()) {
      if (hasEffect<MemoryEffects::Free>(user)) {
        if (user->getBlock() == parallelCombiningParent->getBlock())
          rewriter.moveOpBefore(user, user->getBlock()->getTerminator());
        break;
      }
    }

    // Delete the op.
    rewriter.eraseOp(op);
    return success();
  }
};

/// Bufferization of tensor.splat. Bufferizes to a new allocation that is filled
/// with a linalg.map. Similar to tensor.generate.
struct SplatOpInterface
    : public BufferizableOpInterface::ExternalModel<SplatOpInterface,
                                                    tensor::SplatOp> {

  bool bufferizesToAllocation(Operation *op, Value value) const { return true; }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    OpBuilder::InsertionGuard g(rewriter);
    auto splatOp = cast<tensor::SplatOp>(op);

    // Allocate memory.
    Location loc = op->getLoc();
    FailureOr<Value> tensorAlloc = allocateTensorForShapedValue(
        rewriter, loc, splatOp.getResult(), options,
        /*copy=*/false);
    if (failed(tensorAlloc))
      return failure();

    // Create linalg::MapOp.
    auto tensorType = cast<RankedTensorType>(tensorAlloc->getType());

    // TODO: Implement memory space for this op.
    if (options.defaultMemorySpaceFn(tensorType) != Attribute())
      return op->emitError("memory space not implemented yet");

    auto linalgOp =
        rewriter.create<linalg::MapOp>(loc, tensorType, /*inputs=*/ValueRange(),
                                       /*init=*/*tensorAlloc);
    Block &linalgBody = linalgOp.getMapper().emplaceBlock();

    // Create linalg::IndexOps.
    rewriter.setInsertionPointToStart(&linalgBody);
    rewriter.create<linalg::YieldOp>(loc, splatOp.getInput());
    rewriter.replaceOp(splatOp, linalgOp.getResult()[0]);

    return success();
  }
};

} // namespace
} // namespace tensor
} // namespace mlir

void mlir::tensor::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, tensor::TensorDialect *dialect) {
    CastOp::attachInterface<CastOpInterface>(*ctx);
    CollapseShapeOp::attachInterface<CollapseShapeOpInterface>(*ctx);
    DimOp::attachInterface<DimOpInterface>(*ctx);
    EmptyOp::attachInterface<EmptyOpInterface>(*ctx);
    ExpandShapeOp::attachInterface<ExpandShapeOpInterface>(*ctx);
    ExtractSliceOp::attachInterface<ExtractSliceOpInterface>(*ctx);
    ExtractOp::attachInterface<ExtractOpInterface>(*ctx);
    FromElementsOp::attachInterface<FromElementsOpInterface>(*ctx);
    GenerateOp::attachInterface<GenerateOpInterface>(*ctx);
    InsertOp::attachInterface<InsertOpInterface>(*ctx);
    InsertSliceOp::attachInterface<InsertSliceOpInterface>(*ctx);
    PadOp::attachInterface<PadOpInterface>(*ctx);
    ParallelInsertSliceOp::attachInterface<ParallelInsertSliceOpInterface>(
        *ctx);
    RankOp::attachInterface<RankOpInterface>(*ctx);
    ReshapeOp::attachInterface<ReshapeOpInterface>(*ctx);
    SplatOp::attachInterface<SplatOpInterface>(*ctx);

    // Load additional dialects of which ops may get created.
    ctx->loadDialect<arith::ArithDialect, linalg::LinalgDialect>();
  });

  // Bufferization requires SubsetInsertionOpInterface models. Make sure that
  // they are registered.
  tensor::registerSubsetOpInterfaceExternalModels(registry);
}
