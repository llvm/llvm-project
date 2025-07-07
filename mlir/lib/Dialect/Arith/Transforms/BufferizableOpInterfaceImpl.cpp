//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace {
/// Bufferization of arith.constant. Replace with memref.get_global.
struct ConstantOpInterface
    : public BufferizableOpInterface::ExternalModel<ConstantOpInterface,
                                                    arith::ConstantOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto constantOp = cast<arith::ConstantOp>(op);
    auto type = dyn_cast<RankedTensorType>(constantOp.getType());

    // Only ranked tensors are supported.
    if (!type)
      return failure();

    Attribute memorySpace;
    if (auto memSpace = options.defaultMemorySpaceFn(type))
      memorySpace = *memSpace;
    else
      return constantOp->emitError("could not infer memory space");

    // Only constants inside a module are supported.
    auto moduleOp = constantOp->getParentOfType<ModuleOp>();
    if (!moduleOp)
      return failure();

    // Create global memory segment and replace tensor with memref pointing to
    // that memory segment.
    FailureOr<memref::GlobalOp> globalOp =
        getGlobalFor(constantOp, state.getSymbolTables(),
                     options.bufferAlignment, memorySpace);
    if (failed(globalOp))
      return failure();
    memref::GlobalOp globalMemref = *globalOp;
    replaceOpWithNewBufferizedOp<memref::GetGlobalOp>(
        rewriter, op, globalMemref.getType(), globalMemref.getName());

    return success();
  }

  bool isWritable(Operation *op, Value value,
                  const AnalysisState &state) const {
    // Memory locations returned by memref::GetGlobalOp may not be written to.
    assert(isa<OpResult>(value));
    return false;
  }
};

struct IndexCastOpInterface
    : public BufferizableOpInterface::ExternalModel<IndexCastOpInterface,
                                                    arith::IndexCastOp> {
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

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto castOp = cast<arith::IndexCastOp>(op);
    auto resultTensorType = cast<TensorType>(castOp.getType());

    FailureOr<Value> source =
        getBuffer(rewriter, castOp.getIn(), options, state);
    if (failed(source))
      return failure();
    auto sourceType = cast<BaseMemRefType>(source->getType());

    // Result type should have same layout and address space as the source type.
    BaseMemRefType resultType;
    if (auto rankedMemRefType = dyn_cast<MemRefType>(sourceType)) {
      resultType = MemRefType::get(
          rankedMemRefType.getShape(), resultTensorType.getElementType(),
          rankedMemRefType.getLayout(), rankedMemRefType.getMemorySpace());
    } else {
      auto unrankedMemrefType = cast<UnrankedMemRefType>(sourceType);
      resultType = UnrankedMemRefType::get(resultTensorType.getElementType(),
                                           unrankedMemrefType.getMemorySpace());
    }

    replaceOpWithNewBufferizedOp<arith::IndexCastOp>(rewriter, op, resultType,
                                                     *source);
    return success();
  }
};

/// Bufferization of arith.select. Just replace the operands.
struct SelectOpInterface
    : public BufferizableOpInterface::ExternalModel<SelectOpInterface,
                                                    arith::SelectOp> {
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
    return {{op->getOpResult(0) /*result*/, BufferRelation::Equivalent,
             /*isDefinite=*/false}};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto selectOp = cast<arith::SelectOp>(op);
    Location loc = selectOp.getLoc();

    // Elementwise conditions are not supported yet. To bufferize such an op,
    // it could be lowered to an elementwise "linalg.generic" with a new
    // "tensor.empty" out tensor, followed by "empty tensor elimination". Such
    // IR will bufferize.
    if (!selectOp.getCondition().getType().isInteger(1))
      return op->emitOpError("only i1 condition values are supported");

    // TODO: It would be more efficient to copy the result of the `select` op
    // instead of its OpOperands. In the worst case, 2 copies are inserted at
    // the moment (one for each tensor). When copying the op result, only one
    // copy would be needed.
    FailureOr<Value> maybeTrueBuffer =
        getBuffer(rewriter, selectOp.getTrueValue(), options, state);
    FailureOr<Value> maybeFalseBuffer =
        getBuffer(rewriter, selectOp.getFalseValue(), options, state);
    if (failed(maybeTrueBuffer) || failed(maybeFalseBuffer))
      return failure();
    Value trueBuffer = *maybeTrueBuffer;
    Value falseBuffer = *maybeFalseBuffer;

    // The "true" and the "false" operands must have the same type. If the
    // buffers have different types, they differ only in their layout map. Cast
    // both of them to the most dynamic MemRef type.
    if (trueBuffer.getType() != falseBuffer.getType()) {
      auto targetType = bufferization::detail::asMemRefType(
          bufferization::getBufferType(selectOp.getResult(), options, state));
      if (failed(targetType))
        return failure();
      if (trueBuffer.getType() != *targetType)
        trueBuffer =
            rewriter.create<memref::CastOp>(loc, *targetType, trueBuffer);
      if (falseBuffer.getType() != *targetType)
        falseBuffer =
            rewriter.create<memref::CastOp>(loc, *targetType, falseBuffer);
    }

    replaceOpWithNewBufferizedOp<arith::SelectOp>(
        rewriter, op, selectOp.getCondition(), trueBuffer, falseBuffer);
    return success();
  }

  FailureOr<BufferLikeType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                const BufferizationState &state,
                SmallVector<Value> &invocationStack) const {
    auto selectOp = cast<arith::SelectOp>(op);
    assert(value == selectOp.getResult() && "invalid value");
    auto trueType =
        bufferization::detail::asMemRefType(bufferization::getBufferType(
            selectOp.getTrueValue(), options, state, invocationStack));
    auto falseType =
        bufferization::detail::asMemRefType(bufferization::getBufferType(
            selectOp.getFalseValue(), options, state, invocationStack));
    if (failed(trueType) || failed(falseType))
      return failure();
    if (*trueType == *falseType)
      return cast<BufferLikeType>(*trueType);
    if (trueType->getMemorySpace() != falseType->getMemorySpace())
      return op->emitError("inconsistent memory space on true/false operands");

    // If the buffers have different types, they differ only in their layout
    // map.
    auto memrefType = llvm::cast<MemRefType>(*trueType);
    return cast<BufferLikeType>(getMemRefTypeWithFullyDynamicLayout(
        RankedTensorType::get(memrefType.getShape(),
                              memrefType.getElementType()),
        memrefType.getMemorySpace()));
  }
};

} // namespace

void mlir::arith::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, ArithDialect *dialect) {
    ConstantOp::attachInterface<ConstantOpInterface>(*ctx);
    IndexCastOp::attachInterface<IndexCastOpInterface>(*ctx);
    SelectOp::attachInterface<SelectOpInterface>(*ctx);
  });
}
