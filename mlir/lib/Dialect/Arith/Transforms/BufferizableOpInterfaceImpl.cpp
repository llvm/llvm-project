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
                          const BufferizationOptions &options) const {
    auto constantOp = cast<arith::ConstantOp>(op);

    // TODO: Implement memory space for this op. E.g., by adding a memory_space
    // attribute to ConstantOp.
    if (options.defaultMemorySpace != Attribute())
      return op->emitError("memory space not implemented yet");

    // Only ranked tensors are supported.
    if (!constantOp.getType().isa<RankedTensorType>())
      return failure();

    // Only constants inside a module are supported.
    auto moduleOp = constantOp->getParentOfType<ModuleOp>();
    if (!moduleOp)
      return failure();

    // Create global memory segment and replace tensor with memref pointing to
    // that memory segment.
    FailureOr<memref::GlobalOp> globalOp =
        getGlobalFor(constantOp, options.bufferAlignment);
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
    assert(value.isa<OpResult>());
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

  AliasingOpResultList getAliasingOpResults(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {{op->getResult(0), BufferRelation::Equivalent}};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto castOp = cast<arith::IndexCastOp>(op);
    auto resultTensorType = castOp.getType().cast<TensorType>();

    FailureOr<Value> source = getBuffer(rewriter, castOp.getIn(), options);
    if (failed(source))
      return failure();
    auto sourceType = source->getType().cast<BaseMemRefType>();

    // Result type should have same layout and address space as the source type.
    BaseMemRefType resultType;
    if (auto rankedMemRefType = sourceType.dyn_cast<MemRefType>()) {
      resultType = MemRefType::get(
          rankedMemRefType.getShape(), resultTensorType.getElementType(),
          rankedMemRefType.getLayout(), rankedMemRefType.getMemorySpace());
    } else {
      auto unrankedMemrefType = sourceType.cast<UnrankedMemRefType>();
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

  AliasingOpResultList getAliasingOpResults(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {{op->getOpResult(0) /*result*/, BufferRelation::Equivalent,
             /*isDefinite=*/false}};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto selectOp = cast<arith::SelectOp>(op);
    Location loc = selectOp.getLoc();

    // TODO: It would be more efficient to copy the result of the `select` op
    // instead of its OpOperands. In the worst case, 2 copies are inserted at
    // the moment (one for each tensor). When copying the op result, only one
    // copy would be needed.
    FailureOr<Value> maybeTrueBuffer =
        getBuffer(rewriter, selectOp.getTrueValue(), options);
    FailureOr<Value> maybeFalseBuffer =
        getBuffer(rewriter, selectOp.getFalseValue(), options);
    if (failed(maybeTrueBuffer) || failed(maybeFalseBuffer))
      return failure();
    Value trueBuffer = *maybeTrueBuffer;
    Value falseBuffer = *maybeFalseBuffer;

    // The "true" and the "false" operands must have the same type. If the
    // buffers have different types, they differ only in their layout map. Cast
    // both of them to the most dynamic MemRef type.
    if (trueBuffer.getType() != falseBuffer.getType()) {
      auto targetType =
          bufferization::getBufferType(selectOp.getResult(), options);
      if (failed(targetType))
        return failure();
      trueBuffer =
          rewriter.create<memref::CastOp>(loc, *targetType, trueBuffer);
      falseBuffer =
          rewriter.create<memref::CastOp>(loc, *targetType, falseBuffer);
    }

    replaceOpWithNewBufferizedOp<arith::SelectOp>(
        rewriter, op, selectOp.getCondition(), trueBuffer, falseBuffer);
    return success();
  }

  FailureOr<BaseMemRefType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                const DenseMap<Value, BaseMemRefType> &fixedTypes) const {
    auto selectOp = cast<arith::SelectOp>(op);
    assert(value == selectOp.getResult() && "invalid value");
    auto trueType = bufferization::getBufferType(selectOp.getTrueValue(),
                                                 options, fixedTypes);
    auto falseType = bufferization::getBufferType(selectOp.getFalseValue(),
                                                  options, fixedTypes);
    if (failed(trueType) || failed(falseType))
      return failure();
    if (*trueType == *falseType)
      return *trueType;
    if (trueType->getMemorySpace() != falseType->getMemorySpace())
      return op->emitError("inconsistent memory space on true/false operands");

    // If the buffers have different types, they differ only in their layout
    // map.
    auto memrefType = trueType->cast<MemRefType>();
    return getMemRefTypeWithFullyDynamicLayout(
        RankedTensorType::get(memrefType.getShape(),
                              memrefType.getElementType()),
        memrefType.getMemorySpace());
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
