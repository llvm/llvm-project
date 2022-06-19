//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {op->getResult(0)};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto castOp = cast<tensor::CastOp>(op);

    // The result buffer still has the old (pre-cast) type.
    Value resultBuffer = getBuffer(rewriter, castOp.getSource(), options);
    auto sourceMemRefType = resultBuffer.getType().cast<BaseMemRefType>();
    Attribute memorySpace = sourceMemRefType.getMemorySpace();
    TensorType resultTensorType =
        castOp.getResult().getType().cast<TensorType>();
    MemRefLayoutAttrInterface layout;

    if (auto rankedMemRefType = sourceMemRefType.dyn_cast<MemRefType>())
      if (resultTensorType.isa<RankedTensorType>())
        layout = rankedMemRefType.getLayout();

    // Compute the new memref type.
    Type resultMemRefType =
        getMemRefType(resultTensorType, options, layout, memorySpace);

    // Replace the op with a memref.cast.
    assert(memref::CastOp::areCastCompatible(resultBuffer.getType(),
                                             resultMemRefType) &&
           "CallOp::bufferize: cast incompatible");
    replaceOpWithNewBufferizedOp<memref::CastOp>(rewriter, op, resultMemRefType,
                                                 resultBuffer);

    return success();
  }
};

/// Bufferization of tensor.collapse_shape. Replace with memref.collapse_shape.
struct CollapseShapeOpInterface
    : public BufferizableOpInterface::ExternalModel<CollapseShapeOpInterface,
                                                    tensor::CollapseShapeOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    if (&opOperand == &op->getOpOperand(0) /*src*/)
      return {op->getOpResult(0)};
    return {};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto collapseShapeOp = cast<tensor::CollapseShapeOp>(op);
    RankedTensorType tensorResultType = collapseShapeOp.getResultType();
    Value buffer = getBuffer(rewriter, collapseShapeOp.getSrc(), options);
    auto bufferType = buffer.getType().cast<MemRefType>();

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
        AffineMap resultLayout =
            makeStridedLinearLayoutMap({}, offset, op->getContext());
        resultType =
            MemRefType::get({}, tensorResultType.getElementType(), resultLayout,
                            bufferType.getMemorySpaceAsInt());
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
      Value tensorAlloc = allocateTensorForShapedValue(
          rewriter, op->getLoc(), collapseShapeOp.getSrc(),
          analysisState.isTensorYielded(collapseShapeOp.getResult()));
      auto memrefType =
          MemRefType::get(collapseShapeOp.getSrcType().getShape(),
                          collapseShapeOp.getSrcType().getElementType(),
                          AffineMap(), bufferType.getMemorySpaceAsInt());
      buffer = rewriter.create<bufferization::ToMemrefOp>(
          op->getLoc(), memrefType, tensorAlloc);
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
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto dimOp = cast<tensor::DimOp>(op);
    auto v = getBuffer(rewriter, dimOp.getSource(), options);
    replaceOpWithNewBufferizedOp<memref::DimOp>(rewriter, op, v,
                                                dimOp.getIndex());
    return success();
  }
};

/// Bufferization of tensor.expand_shape. Replace with memref.expand_shape.
struct ExpandShapeOpInterface
    : public BufferizableOpInterface::ExternalModel<ExpandShapeOpInterface,
                                                    tensor::ExpandShapeOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    if (&opOperand == &op->getOpOperand(0) /*src*/)
      return {op->getOpResult(0)};
    return {};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto expandShapeOp = cast<tensor::ExpandShapeOp>(op);
    auto tensorResultType = expandShapeOp.getResultType();
    auto buffer = getBuffer(rewriter, expandShapeOp.getSrc(), options);

    // Memref result type is inferred by the builder based on reassociation
    // indices and result shape.
    replaceOpWithNewBufferizedOp<memref::ExpandShapeOp>(
        rewriter, op, tensorResultType.getShape(), buffer,
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

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    if (&opOperand == &op->getOpOperand(0) /*source*/)
      return {op->getOpResult(0)};
    return {};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::None;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto extractSliceOp = cast<tensor::ExtractSliceOp>(op);
    Location loc = extractSliceOp.getLoc();

    // Even if this op was decided to bufferize out-of-place, do not insert the
    // buffer copy yet. This is done later in this function.
    auto srcMemref = getBuffer(rewriter, extractSliceOp.getSource(), options);
    auto srcMemrefType = srcMemref.getType().cast<MemRefType>();
    auto dstTensorType =
        extractSliceOp.getResult().getType().cast<RankedTensorType>();

    // Expand offsets, sizes and strides to the full rank to handle the
    // rank-reducing case.
    SmallVector<OpFoldResult> mixedOffsets = extractSliceOp.getMixedOffsets();
    SmallVector<OpFoldResult> mixedSizes = extractSliceOp.getMixedSizes();
    SmallVector<OpFoldResult> mixedStrides = extractSliceOp.getMixedStrides();
    OffsetSizeAndStrideOpInterface::expandToRank(
        srcMemref, mixedOffsets, mixedSizes, mixedStrides,
        [&](Value target, int64_t dim) -> OpFoldResult {
          auto shapedType = target.getType().cast<ShapedType>();
          if (shapedType.isDynamicDim(dim))
            return rewriter.create<memref::DimOp>(loc, target, dim).result();
          return rewriter.getIndexAttr(shapedType.getDimSize(dim));
        });
    // Bufferize to subview.
    auto subviewMemRefType = memref::SubViewOp::inferRankReducedResultType(
                                 dstTensorType.getRank(), srcMemrefType,
                                 mixedOffsets, mixedSizes, mixedStrides)
                                 .cast<MemRefType>();
    Value subView = rewriter.create<memref::SubViewOp>(
        loc, subviewMemRefType, srcMemref, mixedOffsets, mixedSizes,
        mixedStrides);

    replaceOpWithBufferizedValues(rewriter, op, subView);
    return success();
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

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto extractOp = cast<tensor::ExtractOp>(op);
    Value srcMemref = getBuffer(rewriter, extractOp.getTensor(), options);
    replaceOpWithNewBufferizedOp<memref::LoadOp>(rewriter, op, srcMemref,
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
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto fromElementsOp = cast<tensor::FromElementsOp>(op);

    // Allocate a buffer for the result.
    Location loc = op->getLoc();
    auto tensorType = fromElementsOp.getType().cast<RankedTensorType>();
    auto shape = tensorType.getShape();
    // TODO: Create alloc_tensor ops during TensorCopyInsertion.
    AnalysisState analysisState(options);
    Value tensorAlloc = allocateTensorForShapedValue(
        rewriter, loc, fromElementsOp.getResult(),
        analysisState.isTensorYielded(fromElementsOp.getResult()),
        /*copy=*/false);
    auto memrefType =
        MemRefType::get(tensorType.getShape(), tensorType.getElementType());
    Value buffer = rewriter.create<bufferization::ToMemrefOp>(
        op->getLoc(), memrefType, tensorAlloc);

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
    auto maxDim = *std::max_element(shape.begin(), shape.end());
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

/// Bufferization of tensor.generate.
struct GenerateOpInterface
    : public BufferizableOpInterface::ExternalModel<GenerateOpInterface,
                                                    tensor::GenerateOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto generateOp = cast<tensor::GenerateOp>(op);
    auto tensorType = generateOp.getType().cast<RankedTensorType>();
    // Allocate memory.
    Location loc = op->getLoc();
    // TODO: Create alloc_tensor ops during TensorCopyInsertion.
    AnalysisState analysisState(options);
    Value tensorAlloc = allocateTensorForShapedValue(
        rewriter, loc, generateOp.getResult(),
        analysisState.isTensorYielded(generateOp.getResult()),
        /*copy=*/false);
    auto memrefType =
        MemRefType::get(tensorType.getShape(), tensorType.getElementType());
    Value buffer = rewriter.create<bufferization::ToMemrefOp>(
        op->getLoc(), memrefType, tensorAlloc);

    // Collect loop bounds.
    int64_t rank = memrefType.getRank();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Value, 4> lowerBounds(rank, zero);
    SmallVector<Value, 4> steps(rank, one);
    SmallVector<Value, 4> upperBounds;
    int nextDynamicIndex = 0;
    for (int i = 0; i < rank; i++) {
      Value upperBound =
          memrefType.isDynamicDim(i)
              ? generateOp.getDynamicExtents()[nextDynamicIndex++]
              : rewriter.create<arith::ConstantIndexOp>(
                    loc, memrefType.getDimSize(i));
      upperBounds.push_back(upperBound);
    }

    // Generate tensor elements with a parallel loop that stores into
    // each element of the resulting memref. We use mergeBlockBefore to "move"
    // this op's body into the scf.parallel's body.
    auto parallel =
        rewriter.create<scf::ParallelOp>(loc, lowerBounds, upperBounds, steps);
    Block *parallelBody = parallel.getBody();
    rewriter.mergeBlockBefore(&generateOp.getBody().front(),
                              parallelBody->getTerminator(),
                              parallelBody->getArguments());
    // Replace the inlined yield op with a store op. The scf.parallel's builder
    // already populated an scf.yield at the end, so we don't need to worry
    // about creating that.
    Operation *elementYield = parallelBody->getTerminator()->getPrevNode();
    rewriter.setInsertionPointAfter(elementYield);
    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        elementYield, elementYield->getOperands()[0], buffer,
        parallelBody->getArguments());

    replaceOpWithBufferizedValues(rewriter, op, buffer);
    return success();
  }
};

/// Bufferization of tensor.insert. Replace with memref.store.
struct InsertOpInterface
    : public BufferizableOpInterface::ExternalModel<InsertOpInterface,
                                                    tensor::InsertOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return true;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    assert(&opOperand == &op->getOpOperand(1) /*dest*/ &&
           "expected dest OpOperand");
    return {op->getOpResult(0)};
  }

  SmallVector<OpOperand *>
  getAliasingOpOperand(Operation *op, OpResult opResult,
                       const AnalysisState &state) const {
    return {&op->getOpOperand(1) /*dest*/};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto insertOp = cast<tensor::InsertOp>(op);
    Value destMemref = getBuffer(rewriter, insertOp.getDest(), options);
    rewriter.create<memref::StoreOp>(insertOp.getLoc(), insertOp.getScalar(),
                                     destMemref, insertOp.getIndices());
    replaceOpWithBufferizedValues(rewriter, op, destMemref);
    return success();
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::Equivalent;
  }
};

/// Return true if the (ExtractSliceOp, InsertSliceOp) pair match (i.e.
/// equivalent operand / result and same offset/sizes/strides specification).
///
/// This is one particular type of relationship between ops on tensors that
/// reduce to an equivalence on buffers. This should be generalized and
/// exposed as interfaces on the proper types.
static bool areEquivalentExtractSliceOps(const AnalysisState &state,
                                         ExtractSliceOp st, InsertSliceOp sti) {
  if (!st || !sti)
    return false;
  if (sti != sti &&
      !state.areEquivalentBufferizedValues(st.getSource(), sti.getDest()))
    return false;
  if (!sameOffsetsSizesAndStrides(st, sti, isEqualConstantIntOrValue))
    return false;
  return true;
}

/// Return true if `value` is originating from an ExtractSliceOp that matches
/// the given InsertSliceOp.
static bool hasMatchingExtractSliceOp(const AnalysisState &state, Value value,
                                      InsertSliceOp insertOp) {
  auto condition = [&](Value val) {
    if (auto extractOp = val.getDefiningOp<ExtractSliceOp>())
      if (areEquivalentExtractSliceOps(state, extractOp, insertOp))
        return true;
    return false;
  };

  return llvm::all_of(state.findValueInReverseUseDefChain(value, condition),
                      condition);
}

/// Bufferization of tensor.insert_slice. Replace with a memory copy. Under
/// certain circumstances, this op can also be a no-op.
struct InsertSliceOpInterface
    : public BufferizableOpInterface::ExternalModel<InsertSliceOpInterface,
                                                    tensor::InsertSliceOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return &opOperand == &op->getOpOperand(1) /*dest*/;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    if (&opOperand == &op->getOpOperand(1) /*dest*/)
      return {op->getResult(0)};
    return {};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::Equivalent;
  }

  bool isNotConflicting(Operation *op, OpOperand *uRead,
                        OpOperand *uConflictingWrite,
                        const AnalysisState &state) const {
    Operation *readingOp = uRead->getOwner();
    Operation *conflictingWritingOp = uConflictingWrite->getOwner();

    // Special rules for matching ExtractSliceOp/InsertSliceOp pairs. If
    // uRead is an InsertSliceOp...
    if (auto insertSliceOp = dyn_cast<InsertSliceOp>(readingOp)) {
      // As an example, consider the following IR.
      //
      // %0 = tensor.extract_slice %t[%a, %b][%c, %d][1, 1] {inplace = [true] }
      // %1 = linalg.fill %cst, %0 {inplace= [true] }
      // %2 = tensor.insert_slice %1 into %t[%a, %b][%c, %d][1, 1]
      //     {inplace= [true] }

      // TODO: Use insertSliceOp.getDestOpOperand etc. when available.
      if (uRead == &insertSliceOp->getOpOperand(1) /*dest*/ &&
          hasMatchingExtractSliceOp(state, uConflictingWrite->get(),
                                    insertSliceOp))
        // Case 1: The main insight is that InsertSliceOp reads only part of
        // the destination tensor. The overwritten area is not read. If
        // uConflictingWrite writes into exactly the memory location that is
        // being read by uRead, this is not a conflict.
        //
        // In the above example:
        // uRead             = OpOperand 1 (%t) of tensor.insert_slice
        // uConflictingWrite = OpOperand 1 (%0) of linalg.fill
        //
        // The read of %t does not conflict with the write of the FillOp
        // (same aliases!) because the area that the FillOp operates on is
        // exactly the one that is *not* read via %t.
        return true;

      if (uRead == &insertSliceOp->getOpOperand(0) /*source*/ &&
          uConflictingWrite == &insertSliceOp->getOpOperand(1) /*dest*/ &&
          hasMatchingExtractSliceOp(state, uRead->get(), insertSliceOp))
        // Case 2: The read of the source tensor and the write to the dest
        // tensor via an InsertSliceOp is not a conflict if the read is
        // reading exactly that part of an equivalent tensor that the
        // InsertSliceOp is writing.
        //
        // In the above example:
        // uRead             = OpOperand 0 (%1) of tensor.insert_slice
        // uConflictingWrite = OpOperand 1 (%t) of tensor.insert_slice
        return true;
    }

    // If uConflictingWrite is an InsertSliceOp...
    if (auto insertSliceOp = dyn_cast<InsertSliceOp>(conflictingWritingOp))
      // As an example, consider the following IR.
      //
      // %0 = tensor.extract_slice %t[%a, %b][%c, %d][1, 1] {inplace = [true] }
      // %1 = linalg.fill %cst, %0 {inplace= [true] }
      // %2 = tensor.insert_slice %1 into %t[%a, %b][%c, %d][1, 1]
      //     {inplace= [true] }
      // %3 = vector.transfer_read %1, %cst
      //
      // In the above example:
      // uRead             = OpOperand 0 (%1) of vector.transfer_read
      // uConflictingWrite = OpOperand 1 (%t) of tensor.insert_slice
      // lastWrite         = %1
      //
      // This is not a conflict because the InsertSliceOp overwrites the
      // memory segment of %1 with the exact same data. (Effectively, there
      // is no memory write here.)
      if (uConflictingWrite == &insertSliceOp->getOpOperand(1) /*dest*/ &&
          state.areEquivalentBufferizedValues(uRead->get(),
                                              insertSliceOp.getSource()) &&
          hasMatchingExtractSliceOp(state, insertSliceOp.getSource(),
                                    insertSliceOp))
        return true;

    return false;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    // insert_slice ops arise from tiling and bufferizing them out-of-place is
    // generally a deal breaker. When used with loops, this ends up cloning the
    // whole tensor on every single iteration and is a symptom of a
    // catastrophically bad scheduling decision.
    // TODO: be very loud about it or even consider failing the pass.
    auto insertSliceOp = cast<tensor::InsertSliceOp>(op);
    Location loc = insertSliceOp.getLoc();
    Value dstMemref = getBuffer(rewriter, insertSliceOp.getDest(), options);

    // Expand offsets, sizes and strides to the full rank to handle the
    // rank-reducing case.
    SmallVector<OpFoldResult> mixedOffsets = insertSliceOp.getMixedOffsets();
    SmallVector<OpFoldResult> mixedSizes = insertSliceOp.getMixedSizes();
    SmallVector<OpFoldResult> mixedStrides = insertSliceOp.getMixedStrides();
    OffsetSizeAndStrideOpInterface::expandToRank(
        dstMemref, mixedOffsets, mixedSizes, mixedStrides,
        [&](Value target, int64_t dim) -> OpFoldResult {
          auto shapedType = target.getType().cast<ShapedType>();
          if (shapedType.isDynamicDim(dim))
            return rewriter.create<memref::DimOp>(loc, target, dim).result();
          return rewriter.getIndexAttr(shapedType.getDimSize(dim));
        });
    // Take a subview of the dst.
    auto dstMemrefType = dstMemref.getType().cast<MemRefType>();
    auto subviewMemRefType =
        memref::SubViewOp::inferRankReducedResultType(
            insertSliceOp.getSourceType().getRank(), dstMemrefType,
            mixedOffsets, mixedSizes, mixedStrides)
            .cast<MemRefType>();
    Value subView = rewriter.create<memref::SubViewOp>(
        loc, subviewMemRefType, dstMemref, mixedOffsets, mixedSizes,
        mixedStrides);

    // Copy tensor. If this tensor.insert_slice has a matching
    // tensor.extract_slice, the copy operation will eventually fold away.
    auto srcMemref = getBuffer(rewriter, insertSliceOp.getSource(), options);
    if (failed(options.createMemCpy(rewriter, loc, srcMemref, subView)))
      return failure();

    replaceOpWithBufferizedValues(rewriter, op, dstMemref);
    return success();
  }
};

/// Bufferization of tensor.rank. Replace with memref.rank.
struct RankOpInterface
    : public BufferizableOpInterface::ExternalModel<RankOpInterface,
                                                    tensor::RankOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto rankOp = cast<tensor::RankOp>(op);
    auto v = getBuffer(rewriter, rankOp.getTensor(), options);
    replaceOpWithNewBufferizedOp<memref::RankOp>(rewriter, op, rankOp.getType(),
                                                 v);
    return success();
  }
};

/// Bufferization of tensor.reshape. Replace with memref.reshape.
struct ReshapeOpInterface
    : public BufferizableOpInterface::ExternalModel<ReshapeOpInterface,
                                                    tensor::ReshapeOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    if (&opOperand == &op->getOpOperand(1) /* shape */)
      return true;
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {op->getOpResult(0)};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto reshapeOp = cast<tensor::ReshapeOp>(op);
    Value srcBuffer = getBuffer(rewriter, reshapeOp.getSource(), options);
    Value shapeBuffer = getBuffer(rewriter, reshapeOp.getShape(), options);
    auto resultTensorType = reshapeOp.getResult().getType().cast<TensorType>();
    auto resultMemRefType = getMemRefType(resultTensorType, options);
    replaceOpWithNewBufferizedOp<memref::ReshapeOp>(
        rewriter, op, resultMemRefType, srcBuffer, shapeBuffer);
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
    ExpandShapeOp::attachInterface<ExpandShapeOpInterface>(*ctx);
    ExtractSliceOp::attachInterface<ExtractSliceOpInterface>(*ctx);
    ExtractOp::attachInterface<ExtractOpInterface>(*ctx);
    FromElementsOp::attachInterface<FromElementsOpInterface>(*ctx);
    GenerateOp::attachInterface<GenerateOpInterface>(*ctx);
    InsertOp::attachInterface<InsertOpInterface>(*ctx);
    InsertSliceOp::attachInterface<InsertSliceOpInterface>(*ctx);
    RankOp::attachInterface<RankOpInterface>(*ctx);
    ReshapeOp::attachInterface<ReshapeOpInterface>(*ctx);
  });
}
