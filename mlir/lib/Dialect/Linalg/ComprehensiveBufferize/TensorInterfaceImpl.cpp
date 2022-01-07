//===- TensorInterfaceImpl.cpp - Tensor Impl. of BufferizableOpInterface --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/ComprehensiveBufferize/TensorInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/BufferizableOpInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

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
                                const BufferizationAliasInfo &aliasInfo,
                                const BufferizationState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationState &state) const {
    auto castOp = cast<tensor::CastOp>(op);

    Value resultBuffer = state.getResultBuffer(rewriter, castOp->getResult(0));
    if (!resultBuffer)
      return failure();
    Type sourceType = resultBuffer.getType();
    auto rankedMemRefType = sourceType.dyn_cast<MemRefType>();
    auto unrankedMemRefType = sourceType.dyn_cast<UnrankedMemRefType>();
    assert(rankedMemRefType || unrankedMemRefType);
    Attribute memorySpace = rankedMemRefType
                                ? rankedMemRefType.getMemorySpace()
                                : unrankedMemRefType.getMemorySpace();
    TensorType tensorType = castOp.getResult().getType().cast<TensorType>();
    MemRefLayoutAttrInterface layout =
        rankedMemRefType && tensorType.isa<RankedTensorType>()
            ? rankedMemRefType.getLayout()
            : MemRefLayoutAttrInterface();
    Type memRefType = getContiguousOrUnrankedMemRefType(
        castOp.getResult().getType(), layout, memorySpace);
    replaceOpWithNewBufferizedOp<memref::CastOp>(rewriter, op, memRefType,
                                                 resultBuffer);
    return success();
  }
};

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
    if (!dimOp.source().getType().isa<RankedTensorType>())
      return dimOp.emitError("unranked tensor not supported");
    Value v = state.lookupBuffer(rewriter, dimOp.source());
    replaceOpWithNewBufferizedOp<memref::DimOp>(rewriter, op, v, dimOp.index());
    return success();
  }
};

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
                                const BufferizationAliasInfo &aliasInfo,
                                const BufferizationState &state) const {
    return BufferRelation::None;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationState &state) const {
    auto extractSliceOp = cast<tensor::ExtractSliceOp>(op);
    Location loc = extractSliceOp.getLoc();
    Value srcMemref = state.lookupBuffer(rewriter, extractSliceOp.source());
    auto srcMemrefType = srcMemref.getType().cast<MemRefType>();
    auto dstTensorType =
        extractSliceOp.result().getType().cast<RankedTensorType>();

    // If not inplaceable, alloc.
    bool inplace = state.isInPlace(extractSliceOp->getResult(0));
    Value alloc;
    if (!inplace)
      alloc = state.createAlloc(rewriter, loc, extractSliceOp.result(),
                                state.getOptions().createDeallocs);

    // Bufferize to subview.
    auto subviewMemRefType =
        memref::SubViewOp::inferRankReducedResultType(
            dstTensorType.getRank(), srcMemrefType,
            extractSliceOp.getMixedOffsets(), extractSliceOp.getMixedSizes(),
            extractSliceOp.getMixedStrides())
            .cast<MemRefType>();
    Value subView = rewriter.create<memref::SubViewOp>(
        loc, subviewMemRefType, srcMemref, extractSliceOp.getMixedOffsets(),
        extractSliceOp.getMixedSizes(), extractSliceOp.getMixedStrides());

    /// If not inplaceable, copy.
    if (!inplace) {
      // Do not copy if the copied data is never read.
      if (state.isValueRead(extractSliceOp.result()))
        state.createMemCpy(rewriter, extractSliceOp.getLoc(), subView, alloc);
      subView = alloc;
    }

    replaceOpWithBufferizedValues(rewriter, op, subView);
    return success();
  }
};

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
    Value srcMemref = state.lookupBuffer(rewriter, extractOp.tensor());
    replaceOpWithNewBufferizedOp<memref::LoadOp>(rewriter, op, srcMemref,
                                                 extractOp.indices());
    return success();
  }
};

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
    Location loc = insertOp.getLoc();
    Value destMemref =
        state.getResultBuffer(rewriter, insertOp->getOpResult(0));
    rewriter.create<memref::StoreOp>(loc, insertOp.scalar(), destMemref,
                                     insertOp.indices());
    replaceOpWithBufferizedValues(rewriter, op, destMemref);
    return success();
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const BufferizationAliasInfo &aliasInfo,
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
static bool
areEquivalentExtractSliceOps(const BufferizationAliasInfo &aliasInfo,
                             ExtractSliceOp st, InsertSliceOp sti) {
  if (!st || !sti)
    return false;
  if (!aliasInfo.areEquivalentBufferizedValues(st.source(), sti.dest()))
    return false;
  if (!sameOffsetsSizesAndStrides(st, sti, isEqualConstantIntOrValue))
    return false;
  return true;
}

/// Return true if `value` is originating from an ExtractSliceOp that matches
/// the given InsertSliceOp.
static bool hasMatchingExtractSliceOp(const BufferizationAliasInfo &aliasInfo,
                                      const BufferizationState &state,
                                      Value value, InsertSliceOp insertOp) {
  auto condition = [&](Value val) {
    if (auto extractOp = val.getDefiningOp<ExtractSliceOp>())
      if (areEquivalentExtractSliceOps(aliasInfo, extractOp, insertOp))
        return true;
    return false;
  };

  return llvm::all_of(state.findValueInReverseUseDefChain(value, condition),
                      condition);
}

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
                                const BufferizationAliasInfo &aliasInfo,
                                const BufferizationState &state) const {
    return BufferRelation::Equivalent;
  }

  bool isNotConflicting(Operation *op, OpOperand *uRead,
                        OpOperand *uConflictingWrite,
                        const BufferizationState &state,
                        const BufferizationAliasInfo &aliasInfo) const {
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
          hasMatchingExtractSliceOp(aliasInfo, state, uConflictingWrite->get(),
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
          hasMatchingExtractSliceOp(aliasInfo, state, uRead->get(),
                                    insertSliceOp))
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
          aliasInfo.areEquivalentBufferizedValues(uRead->get(),
                                                  insertSliceOp.source()) &&
          hasMatchingExtractSliceOp(aliasInfo, state, insertSliceOp.source(),
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
    Value dstMemref =
        state.getResultBuffer(rewriter, insertSliceOp->getResult(0));
    if (!dstMemref)
      return failure();

    // Take a subview of the dst.
    auto dstMemrefType = dstMemref.getType().cast<MemRefType>();
    auto subviewMemRefType =
        memref::SubViewOp::inferRankReducedResultType(
            insertSliceOp.getSourceType().getRank(), dstMemrefType,
            insertSliceOp.getMixedOffsets(), insertSliceOp.getMixedSizes(),
            insertSliceOp.getMixedStrides())
            .cast<MemRefType>();
    Value subView = rewriter.create<memref::SubViewOp>(
        loc, subviewMemRefType, dstMemref, insertSliceOp.getMixedOffsets(),
        insertSliceOp.getMixedSizes(), insertSliceOp.getMixedStrides());

    // Copy tensor. If this tensor.insert_slice has a matching
    // tensor.extract_slice, the copy operation will eventually fold away.
    Value srcMemref = state.lookupBuffer(rewriter, insertSliceOp.source());
    state.createMemCpy(rewriter, insertSliceOp.getLoc(), srcMemref, subView);

    replaceOpWithBufferizedValues(rewriter, op, dstMemref);
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
