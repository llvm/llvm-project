//===- TensorInterfaceImpl.cpp - Tensor Impl. of BufferizableOpInterface --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/ComprehensiveBufferize/TensorInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace mlir {
namespace linalg {
namespace comprehensive_bufferize {
namespace tensor_ext {

using tensor::ExtractSliceOp;
using tensor::InsertSliceOp;

struct CastOpInterface
    : public BufferizableOpInterface::ExternalModel<CastOpInterface,
                                                    tensor::CastOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const BufferizationState &state) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    return false;
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    return op->getResult(0);
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const BufferizationState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationState &state) const {
    auto castOp = cast<tensor::CastOp>(op);

    // The result buffer still has the old (pre-cast) type.
    FailureOr<Value> resultBuffer =
        state.getBuffer(rewriter, castOp->getOpOperand(0) /*source*/);
    if (failed(resultBuffer))
      return failure();
    auto sourceMemRefType = resultBuffer->getType().cast<BaseMemRefType>();
    Attribute memorySpace = sourceMemRefType.getMemorySpace();
    TensorType resultTensorType =
        castOp.getResult().getType().cast<TensorType>();
    MemRefLayoutAttrInterface layout;

    if (auto rankedMemRefType = sourceMemRefType.dyn_cast<MemRefType>())
      if (resultTensorType.isa<RankedTensorType>())
        layout = rankedMemRefType.getLayout();

    // Compute the new memref type.
    Type resultMemRefType;
    if (resultTensorType.isa<RankedTensorType>()) {
      resultMemRefType =
          getContiguousMemRefType(resultTensorType, layout, memorySpace);
    } else {
      resultMemRefType =
          getUnrankedMemRefType(resultTensorType.getElementType(), memorySpace);
    }

    // Replace the op with a memref.cast.
    assert(memref::CastOp::areCastCompatible(resultBuffer->getType(),
                                             resultMemRefType) &&
           "CallOp::bufferize: cast incompatible");
    replaceOpWithNewBufferizedOp<memref::CastOp>(rewriter, op, resultMemRefType,
                                                 *resultBuffer);

    return success();
  }
};

/// Bufferization of tensor.dim. Replace with memref.dim.
struct DimOpInterface
    : public BufferizableOpInterface::ExternalModel<DimOpInterface,
                                                    tensor::DimOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const BufferizationState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    return false;
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    return OpResult();
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationState &state) const {
    auto dimOp = cast<tensor::DimOp>(op);
    Value v = *state.getBuffer(rewriter, dimOp->getOpOperand(0) /*source*/);
    replaceOpWithNewBufferizedOp<memref::DimOp>(rewriter, op, v, dimOp.index());
    return success();
  }
};

/// Bufferization of tensor.extract_slice. Replace with memref.subview.
struct ExtractSliceOpInterface
    : public BufferizableOpInterface::ExternalModel<ExtractSliceOpInterface,
                                                    tensor::ExtractSliceOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const BufferizationState &state) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    return false;
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    return &opOperand == &op->getOpOperand(0) /*source*/
               ? op->getResult(0)
               : OpResult();
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const BufferizationState &state) const {
    return BufferRelation::None;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationState &state) const {
    auto extractSliceOp = cast<tensor::ExtractSliceOp>(op);
    Location loc = extractSliceOp.getLoc();
    Value srcMemref =
        *state.getBuffer(rewriter, extractSliceOp->getOpOperand(0) /*source*/,
                         /*forceInPlace=*/true);
    auto srcMemrefType = srcMemref.getType().cast<MemRefType>();
    auto dstTensorType =
        extractSliceOp.result().getType().cast<RankedTensorType>();

    // If not inplaceable, alloc.
    bool inplace = state.isInPlace(extractSliceOp->getOpOperand(0));
    Value alloc;
    if (!inplace) {
      FailureOr<Value> allocOrFailure =
          createAlloc(rewriter, loc, extractSliceOp.result(),
                      state.getOptions().createDeallocs, state.getOptions());
      if (failed(allocOrFailure))
        return failure();
      alloc = *allocOrFailure;
    }

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

    // If not inplaceable, copy.
    if (!inplace) {
      // Do not copy if the copied data is never read.
      if (state.isValueRead(extractSliceOp.result()))
        if (failed(createMemCpy(rewriter, extractSliceOp.getLoc(), subView,
                                alloc, state.getOptions())))
          return failure();
      subView = alloc;
    }

    replaceOpWithBufferizedValues(rewriter, op, subView);
    return success();
  }
};

/// Bufferization of tensor.extract. Replace with memref.load.
struct ExtractOpInterface
    : public BufferizableOpInterface::ExternalModel<ExtractOpInterface,
                                                    tensor::ExtractOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const BufferizationState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    return false;
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    return OpResult();
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationState &state) const {
    auto extractOp = cast<tensor::ExtractOp>(op);
    Value srcMemref =
        *state.getBuffer(rewriter, extractOp->getOpOperand(0) /*tensor*/);
    replaceOpWithNewBufferizedOp<memref::LoadOp>(rewriter, op, srcMemref,
                                                 extractOp.indices());
    return success();
  }
};

/// Bufferization of tensor.insert. Replace with memref.store.
struct InsertOpInterface
    : public BufferizableOpInterface::ExternalModel<InsertOpInterface,
                                                    tensor::InsertOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const BufferizationState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    return true;
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    assert(&opOperand == &op->getOpOperand(1) /*dest*/ &&
           "expected dest OpOperand");
    return op->getOpResult(0);
  }

  SmallVector<OpOperand *>
  getAliasingOpOperand(Operation *op, OpResult opResult,
                       const BufferizationState &state) const {
    return {&op->getOpOperand(1) /*dest*/};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationState &state) const {
    auto insertOp = cast<tensor::InsertOp>(op);
    FailureOr<Value> destMemref =
        state.getBuffer(rewriter, insertOp->getOpOperand(1) /*dest*/);
    if (failed(destMemref))
      return failure();
    rewriter.create<memref::StoreOp>(insertOp.getLoc(), insertOp.scalar(),
                                     *destMemref, insertOp.indices());
    replaceOpWithBufferizedValues(rewriter, op, *destMemref);
    return success();
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const BufferizationState &state) const {
    return BufferRelation::Equivalent;
  }
};

/// Return true if the (ExtractSliceOp, InsertSliceOp) pair match (i.e.
/// equivalent operand / result and same offset/sizes/strides specification).
///
/// This is one particular type of relationship between ops on tensors that
/// reduce to an equivalence on buffers. This should be generalized and
/// exposed as interfaces on the proper types.
static bool areEquivalentExtractSliceOps(const BufferizationState &state,
                                         ExtractSliceOp st, InsertSliceOp sti) {
  if (!st || !sti)
    return false;
  if (sti != sti &&
      !state.areEquivalentBufferizedValues(st.source(), sti.dest()))
    return false;
  if (!sameOffsetsSizesAndStrides(st, sti, isEqualConstantIntOrValue))
    return false;
  return true;
}

/// Return true if `value` is originating from an ExtractSliceOp that matches
/// the given InsertSliceOp.
static bool hasMatchingExtractSliceOp(const BufferizationState &state,
                                      Value value, InsertSliceOp insertOp) {
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
                              const BufferizationState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    return &opOperand == &op->getOpOperand(1) /*dest*/;
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    return &opOperand == &op->getOpOperand(1) /*dest*/
               ? op->getResult(0)
               : OpResult();
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const BufferizationState &state) const {
    return BufferRelation::Equivalent;
  }

  bool isNotConflicting(Operation *op, OpOperand *uRead,
                        OpOperand *uConflictingWrite,
                        const BufferizationState &state) const {
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
                                              insertSliceOp.source()) &&
          hasMatchingExtractSliceOp(state, insertSliceOp.source(),
                                    insertSliceOp))
        return true;

    return false;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationState &state) const {
    // insert_slice ops arise from tiling and bufferizing them out-of-place is
    // generally a deal breaker. When used with loops, this ends up cloning the
    // whole tensor on every single iteration and is a symptom of a
    // catastrophically bad scheduling decision.
    // TODO: be very loud about it or even consider failing the pass.
    auto insertSliceOp = cast<tensor::InsertSliceOp>(op);
    Location loc = insertSliceOp.getLoc();

    // When bufferizing out-of-place, `getResultBuffer` allocates.
    FailureOr<Value> dstMemref =
        state.getBuffer(rewriter, insertSliceOp->getOpOperand(1) /*dest*/);
    if (failed(dstMemref))
      return failure();

    // Expand offsets, sizes and strides to the full rank to handle the
    // rank-reducing case.
    SmallVector<OpFoldResult> mixedOffsets = insertSliceOp.getMixedOffsets();
    SmallVector<OpFoldResult> mixedSizes = insertSliceOp.getMixedSizes();
    SmallVector<OpFoldResult> mixedStrides = insertSliceOp.getMixedStrides();
    OffsetSizeAndStrideOpInterface::expandToRank(
        *dstMemref, mixedOffsets, mixedSizes, mixedStrides,
        [&](Value target, int64_t dim) -> OpFoldResult {
          auto shapedType = target.getType().cast<ShapedType>();
          if (shapedType.isDynamicDim(dim))
            return rewriter.create<memref::DimOp>(loc, target, dim).result();
          return rewriter.getIndexAttr(shapedType.getDimSize(dim));
        });
    // Take a subview of the dst.
    auto dstMemrefType = dstMemref->getType().cast<MemRefType>();
    auto subviewMemRefType =
        memref::SubViewOp::inferRankReducedResultType(
            insertSliceOp.getSourceType().getRank(), dstMemrefType,
            mixedOffsets, mixedSizes, mixedStrides)
            .cast<MemRefType>();
    Value subView = rewriter.create<memref::SubViewOp>(
        loc, subviewMemRefType, *dstMemref, mixedOffsets, mixedSizes,
        mixedStrides);

    // Copy tensor. If this tensor.insert_slice has a matching
    // tensor.extract_slice, the copy operation will eventually fold away.
    Value srcMemref =
        *state.getBuffer(rewriter, insertSliceOp->getOpOperand(0) /*source*/);
    if (failed(createMemCpy(rewriter, loc, srcMemref, subView,
                            state.getOptions())))
      return failure();

    replaceOpWithBufferizedValues(rewriter, op, *dstMemref);
    return success();
  }
};

} // namespace tensor_ext
} // namespace comprehensive_bufferize
} // namespace linalg
} // namespace mlir

void mlir::linalg::comprehensive_bufferize::tensor_ext::
    registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addOpInterface<tensor::CastOp, tensor_ext::CastOpInterface>();
  registry.addOpInterface<tensor::DimOp, tensor_ext::DimOpInterface>();
  registry.addOpInterface<tensor::ExtractSliceOp,
                          tensor_ext::ExtractSliceOpInterface>();
  registry.addOpInterface<tensor::ExtractOp, tensor_ext::ExtractOpInterface>();
  registry.addOpInterface<tensor::InsertOp, tensor_ext::InsertOpInterface>();
  registry.addOpInterface<tensor::InsertSliceOp,
                          tensor_ext::InsertSliceOpInterface>();
}
