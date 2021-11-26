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

namespace {
/// Extra bufferization state that is required for bufferization of tensor ops.
struct TensorBufferizationState : public DialectBufferizationState {
  /// InsertSliceOps that bufferize inplace and do not require a copy.
  DenseSet<Operation *> insertSliceOpsWithoutCopy;
};
} // namespace

static TensorBufferizationState &
getTensorBufferizationState(BufferizationState &state) {
  return state.getDialectState<TensorBufferizationState>(
      tensor::TensorDialect::getDialectNamespace());
}

struct CastOpInterface
    : public BufferizableOpInterface::ExternalModel<CastOpInterface,
                                                    tensor::CastOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand) const {
    return false;
  }

  SmallVector<OpOperand *> getAliasingOpOperand(Operation *op,
                                                OpResult opResult) const {
    return {&op->getOpOperand(0)};
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand) const {
    return op->getResult(0);
  }

  BufferRelation bufferRelation(Operation *op, OpOperand &opOperand) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    auto castOp = cast<tensor::CastOp>(op);

    // Take a guard before anything else.
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(castOp);

    Value resultBuffer = getResultBuffer(b, castOp->getResult(0), state);
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
    Value res =
        b.create<memref::CastOp>(castOp.getLoc(), memRefType, resultBuffer);
    state.aliasInfo.insertNewBufferEquivalence(res, castOp.getResult());
    state.mapBuffer(castOp.getResult(), res);
    return success();
  }
};

struct DimOpInterface
    : public BufferizableOpInterface::ExternalModel<DimOpInterface,
                                                    tensor::DimOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand) const {
    return false;
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand) const {
    return OpResult();
  }

  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    auto dimOp = cast<tensor::DimOp>(op);

    // Take a guard before anything else.
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(dimOp);

    if (dimOp.source().getType().isa<RankedTensorType>()) {
      Value v = state.lookupBuffer(dimOp.source());
      dimOp.result().replaceAllUsesWith(
          b.create<memref::DimOp>(dimOp.getLoc(), v, dimOp.index()));
    }
    return success();
  }
};

struct ExtractSliceOpInterface
    : public BufferizableOpInterface::ExternalModel<ExtractSliceOpInterface,
                                                    tensor::ExtractSliceOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand) const {
    return false;
  }

  SmallVector<OpOperand *> getAliasingOpOperand(Operation *op,
                                                OpResult opResult) const {
    return {&op->getOpOperand(0) /*source*/};
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand) const {
    return &opOperand == &op->getOpOperand(0) /*source*/
               ? op->getResult(0)
               : OpResult();
  }

  BufferRelation bufferRelation(Operation *op, OpOperand &opOperand) const {
    return BufferRelation::None;
  }

  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    auto extractSliceOp = cast<tensor::ExtractSliceOp>(op);

    // Take a guard before anything else.
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(extractSliceOp);

    Location loc = extractSliceOp.getLoc();
    Value srcMemref = state.lookupBuffer(extractSliceOp.source());
    auto srcMemrefType = srcMemref.getType().cast<MemRefType>();
    auto dstTensorType =
        extractSliceOp.result().getType().cast<RankedTensorType>();

    // If not inplaceable, alloc.
    bool inplace = state.aliasInfo.isInPlace(extractSliceOp->getResult(0));
    Value alloc;
    if (!inplace)
      alloc = state.createAllocDeallocFn(b, loc, extractSliceOp.result());

    // Bufferize to subview.
    auto subviewMemRefType =
        memref::SubViewOp::inferRankReducedResultType(
            dstTensorType.getRank(), srcMemrefType,
            extractSliceOp.getMixedOffsets(), extractSliceOp.getMixedSizes(),
            extractSliceOp.getMixedStrides())
            .cast<MemRefType>();
    Value subView = b.create<memref::SubViewOp>(
        loc, subviewMemRefType, srcMemref, extractSliceOp.getMixedOffsets(),
        extractSliceOp.getMixedSizes(), extractSliceOp.getMixedStrides());
    // Insert new alias.
    state.aliasInfo.insertNewBufferAlias(subView, srcMemref);

    /// If not inplaceable, copy.
    if (!inplace) {
      // Do not copy if the copied data is never read.
      if (isValueRead(extractSliceOp.result()))
        state.allocationFns.memCpyFn(b, extractSliceOp.getLoc(), subView,
                                     alloc);
      subView = alloc;
    }

    state.mapBuffer(extractSliceOp.result(), subView);
    return success();
  }
};

struct ExtractOpInterface
    : public BufferizableOpInterface::ExternalModel<ExtractOpInterface,
                                                    tensor::ExtractOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand) const {
    return false;
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand) const {
    return OpResult();
  }

  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    auto extractOp = cast<tensor::ExtractOp>(op);

    // Take a guard before anything else.
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(extractOp);

    Location loc = extractOp.getLoc();
    Value srcMemref = state.lookupBuffer(extractOp.tensor());
    Value l = b.create<memref::LoadOp>(loc, srcMemref, extractOp.indices());
    extractOp.replaceAllUsesWith(l);
    return success();
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

/// Return true if the source of a `insertSliceOp` bufferizes to an
/// equivalent ExtractSliceOp that bufferizes inplace.
static bool isSourceEquivalentToAMatchingInplaceExtractSliceOp(
    const BufferizationAliasInfo &aliasInfo, InsertSliceOp insertSliceOp) {
  bool foundOp = false;
  aliasInfo.applyOnEquivalenceClass(insertSliceOp.source(), [&](Value value) {
    auto extractSliceOp = value.getDefiningOp<ExtractSliceOp>();
    if (extractSliceOp &&
        areEquivalentExtractSliceOps(aliasInfo, extractSliceOp,
                                     insertSliceOp) &&
        aliasInfo.isInPlace(extractSliceOp->getResult(0))) {
      foundOp = true;
    }
  });
  return foundOp;
}

/// Return true if `value` is originating from an ExtractSliceOp that matches
/// the given InsertSliceOp.
static bool hasMatchingExtractSliceOp(const BufferizationAliasInfo &aliasInfo,
                                      Value value, InsertSliceOp insertOp) {
  auto condition = [&](Value val) {
    if (auto extractOp = val.getDefiningOp<ExtractSliceOp>())
      if (areEquivalentExtractSliceOps(aliasInfo, extractOp, insertOp))
        return true;
    return false;
  };

  return llvm::all_of(findValueInReverseUseDefChain(value, condition),
                      condition);
}

struct InsertSliceOpInterface
    : public BufferizableOpInterface::ExternalModel<InsertSliceOpInterface,
                                                    tensor::InsertSliceOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand) const {
    return &opOperand == &op->getOpOperand(1) /*dest*/;
  }

  SmallVector<OpOperand *> getAliasingOpOperand(Operation *op,
                                                OpResult opResult) const {
    return {&op->getOpOperand(1) /*dest*/};
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand) const {
    return &opOperand == &op->getOpOperand(1) /*dest*/
               ? op->getResult(0)
               : OpResult();
  }

  BufferRelation bufferRelation(Operation *op, OpOperand &opOperand) const {
    return BufferRelation::Equivalent;
  }

  bool isNotConflicting(Operation *op, OpOperand *uRead,
                        OpOperand *uConflictingWrite,
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
          hasMatchingExtractSliceOp(aliasInfo, uConflictingWrite->get(),
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
          hasMatchingExtractSliceOp(aliasInfo, uRead->get(), insertSliceOp))
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
          hasMatchingExtractSliceOp(aliasInfo, insertSliceOp.source(),
                                    insertSliceOp))
        return true;

    return false;
  }

  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    // insert_slice ops arise from tiling and bufferizing them out-of-place is
    // generally a deal breaker. When used with loops, this ends up cloning the
    // whole tensor on every single iteration and is a symptom of a
    // catastrophically bad scheduling decision.
    // TODO: be very loud about it or even consider failing the pass.
    auto insertSliceOp = cast<tensor::InsertSliceOp>(op);
    TensorBufferizationState &tensorState = getTensorBufferizationState(state);

    // Take a guard before anything else.
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(insertSliceOp);
    Location loc = insertSliceOp.getLoc();

    // When bufferizing out-of-place, `getResultBuffer` allocates.
    Value dstMemref = getResultBuffer(b, insertSliceOp->getResult(0), state);
    if (!dstMemref)
      return failure();

    bool needCopy =
        !tensorState.insertSliceOpsWithoutCopy.contains(insertSliceOp);
    if (needCopy) {
      // Take a subview of the dst.
      auto dstMemrefType = dstMemref.getType().cast<MemRefType>();
      auto subviewMemRefType =
          memref::SubViewOp::inferRankReducedResultType(
              insertSliceOp.getSourceType().getRank(), dstMemrefType,
              insertSliceOp.getMixedOffsets(), insertSliceOp.getMixedSizes(),
              insertSliceOp.getMixedStrides())
              .cast<MemRefType>();
      Value subView = b.create<memref::SubViewOp>(
          loc, subviewMemRefType, dstMemref, insertSliceOp.getMixedOffsets(),
          insertSliceOp.getMixedSizes(), insertSliceOp.getMixedStrides());
      // Insert new alias.
      state.aliasInfo.insertNewBufferAlias(subView, dstMemref);
      // Copy tensor.
      Value srcMemref = state.lookupBuffer(insertSliceOp.source());
      state.allocationFns.memCpyFn(b, insertSliceOp.getLoc(), srcMemref,
                                   subView);
    }

    state.mapBuffer(insertSliceOp.result(), dstMemref);
    return success();
  }
};

} // namespace tensor_ext
} // namespace comprehensive_bufferize
} // namespace linalg
} // namespace mlir

LogicalResult mlir::linalg::comprehensive_bufferize::tensor_ext::
    InplaceInsertSliceOpAnalysis::run(FuncOp funcOp, BufferizationState &state,
                                      SmallVector<Operation *> &newOps) {
  auto &tensorState = getTensorBufferizationState(state);
  funcOp.walk([&](InsertSliceOp insertSliceOp) {
    // A copy of the source buffer is needed if either:
    //   - The producer of `source` is not inplace. This is the case where a
    //     slice is computed out of place into the inplace full tensor.
    //   - The result is not inplace. This is the case where the whole tensor is
    //     cloned and the clone needs to be updated.
    if (isSourceEquivalentToAMatchingInplaceExtractSliceOp(state.aliasInfo,
                                                           insertSliceOp) &&
        state.aliasInfo.isInPlace(insertSliceOp->getResult(0)))
      tensorState.insertSliceOpsWithoutCopy.insert(insertSliceOp);
  });
  return success();
}

void mlir::linalg::comprehensive_bufferize::tensor_ext::
    registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addOpInterface<tensor::CastOp, tensor_ext::CastOpInterface>();
  registry.addOpInterface<tensor::DimOp, tensor_ext::DimOpInterface>();
  registry.addOpInterface<tensor::ExtractSliceOp,
                          tensor_ext::ExtractSliceOpInterface>();
  registry.addOpInterface<tensor::ExtractOp, tensor_ext::ExtractOpInterface>();
  registry.addOpInterface<tensor::InsertSliceOp,
                          tensor_ext::InsertSliceOpInterface>();
}
