//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
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
                              const BufferizationState &state) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    return false;
  }

  SmallVector<OpResult>
  getAliasingOpResult(Operation *op, OpOperand &opOperand,
                      const BufferizationState &state) const {
    return {op->getResult(0)};
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
    Type resultMemRefType = getMemRefType(resultTensorType, state.getOptions(),
                                          layout, memorySpace);

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

  SmallVector<OpResult>
  getAliasingOpResult(Operation *op, OpOperand &opOperand,
                      const BufferizationState &state) const {
    return {};
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

  SmallVector<OpResult>
  getAliasingOpResult(Operation *op, OpOperand &opOperand,
                      const BufferizationState &state) const {
    if (&opOperand == &op->getOpOperand(0) /*source*/)
      return {op->getOpResult(0)};
    return {};
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

  SmallVector<OpResult>
  getAliasingOpResult(Operation *op, OpOperand &opOperand,
                      const BufferizationState &state) const {
    return {};
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
                          const BufferizationState &state) const {
    auto fromElementsOp = cast<tensor::FromElementsOp>(op);

    // Allocate a buffer for the result.
    Location loc = op->getLoc();
    auto tensorType = fromElementsOp.getType().cast<RankedTensorType>();
    auto shape = tensorType.getShape();
    MemRefType resultType = getContiguousMemRefType(tensorType);
    FailureOr<Value> maybeBuffer =
        createAlloc(rewriter, loc, resultType, {},
                    /*deallocMemref=*/state.getOptions().createDeallocs,
                    state.getOptions());
    if (failed(maybeBuffer))
      return failure();
    Value buffer = *maybeBuffer;

    // Case: tensor<0xelem_type>.
    if (fromElementsOp.elements().empty()) {
      replaceOpWithBufferizedValues(rewriter, op, buffer);
      return success();
    }

    // Case: tensor<elem_type>.
    if (shape.empty()) {
      rewriter.create<memref::StoreOp>(loc, fromElementsOp.elements().front(),
                                       buffer);
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
    auto elementIt = fromElementsOp.elements().begin();
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
                          const BufferizationState &state) const {
    auto generateOp = cast<tensor::GenerateOp>(op);

    // Allocate memory.
    Location loc = op->getLoc();
    MemRefType memrefType =
        getContiguousMemRefType(generateOp.getType().cast<RankedTensorType>());
    FailureOr<Value> maybeResult =
        createAlloc(rewriter, loc, memrefType, generateOp.dynamicExtents(),
                    /*deallocMemref=*/state.getOptions().createDeallocs,
                    state.getOptions());
    if (failed(maybeResult))
      return failure();
    Value result = *maybeResult;

    // Collect loop bounds.
    int64_t rank = memrefType.getRank();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Value, 4> lowerBounds(rank, zero);
    SmallVector<Value, 4> steps(rank, one);
    SmallVector<Value, 4> upperBounds;
    int nextDynamicIndex = 0;
    for (int i = 0; i < rank; i++) {
      Value upperBound = memrefType.isDynamicDim(i)
                             ? generateOp.dynamicExtents()[nextDynamicIndex++]
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
    rewriter.mergeBlockBefore(generateOp.getBody(),
                              parallelBody->getTerminator(),
                              parallelBody->getArguments());
    // Replace the inlined yield op with a store op. The scf.parallel's builder
    // already populated an scf.yield at the end, so we don't need to worry
    // about creating that.
    Operation *elementYield = parallelBody->getTerminator()->getPrevNode();
    rewriter.setInsertionPointAfter(elementYield);
    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        elementYield, elementYield->getOperands()[0], result,
        parallelBody->getArguments());

    replaceOpWithBufferizedValues(rewriter, op, result);
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

  SmallVector<OpResult>
  getAliasingOpResult(Operation *op, OpOperand &opOperand,
                      const BufferizationState &state) const {
    assert(&opOperand == &op->getOpOperand(1) /*dest*/ &&
           "expected dest OpOperand");
    return {op->getOpResult(0)};
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

  SmallVector<OpResult>
  getAliasingOpResult(Operation *op, OpOperand &opOperand,
                      const BufferizationState &state) const {
    if (&opOperand == &op->getOpOperand(1) /*dest*/)
      return {op->getResult(0)};
    return {};
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

/// Bufferization of tensor.rank. Replace with memref.rank.
struct RankOpInterface
    : public BufferizableOpInterface::ExternalModel<RankOpInterface,
                                                    tensor::RankOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const BufferizationState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    return false;
  }

  SmallVector<OpResult>
  getAliasingOpResult(Operation *op, OpOperand &opOperand,
                      const BufferizationState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationState &state) const {
    auto rankOp = cast<tensor::RankOp>(op);
    Value v = *state.getBuffer(rewriter, rankOp->getOpOperand(0) /*source*/);
    replaceOpWithNewBufferizedOp<memref::RankOp>(rewriter, op, rankOp.getType(),
                                                 v);
    return success();
  }
};

} // namespace
} // namespace tensor
} // namespace mlir

void mlir::tensor::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addOpInterface<CastOp, CastOpInterface>();
  registry.addOpInterface<DimOp, DimOpInterface>();
  registry.addOpInterface<ExtractSliceOp, ExtractSliceOpInterface>();
  registry.addOpInterface<ExtractOp, ExtractOpInterface>();
  registry.addOpInterface<FromElementsOp, FromElementsOpInterface>();
  registry.addOpInterface<GenerateOp, GenerateOpInterface>();
  registry.addOpInterface<InsertOp, InsertOpInterface>();
  registry.addOpInterface<InsertSliceOp, InsertSliceOpInterface>();
  registry.addOpInterface<RankOp, RankOpInterface>();
}
