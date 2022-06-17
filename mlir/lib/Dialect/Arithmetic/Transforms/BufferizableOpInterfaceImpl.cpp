//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
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
                          BufferizationState &state) const {
    auto constantOp = cast<arith::ConstantOp>(op);

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
        getGlobalFor(constantOp, state.getOptions().bufferAlignment);
    if (failed(globalOp))
      return failure();
    memref::GlobalOp globalMemref = globalOp.getValue();
    replaceOpWithNewBufferizedOp<memref::GetGlobalOp>(
        rewriter, op, globalMemref.type(), globalMemref.getName());

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

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {op->getResult(0)};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          BufferizationState &state) const {
    auto castOp = cast<arith::IndexCastOp>(op);
    auto resultTensorType = castOp.getType().cast<TensorType>();

    Value source = state.getBuffer(rewriter, castOp.getIn());
    auto sourceType = source.getType().cast<BaseMemRefType>();

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
                                                     source);
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

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {op->getOpResult(0) /*result*/};
  }

  SmallVector<OpOperand *>
  getAliasingOpOperand(Operation *op, OpResult opResult,
                       const AnalysisState &state) const {
    return {&op->getOpOperand(1) /*true_value*/,
            &op->getOpOperand(2) /*false_value*/};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          BufferizationState &state) const {
    auto selectOp = cast<arith::SelectOp>(op);
    Location loc = selectOp.getLoc();

    // TODO: It would be more efficient to copy the result of the `select` op
    // instead of its OpOperands. In the worst case, 2 copies are inserted at
    // the moment (one for each tensor). When copying the op result, only one
    // copy would be needed.
    Value trueBuffer = state.getBuffer(rewriter, selectOp.getTrueValue());
    Value falseBuffer = state.getBuffer(rewriter, selectOp.getFalseValue());

    // The "true" and the "false" operands must have the same type. If the
    // buffers have different types, they differ only in their layout map. Cast
    // both of them to the most dynamic MemRef type.
    if (trueBuffer.getType() != falseBuffer.getType()) {
      auto trueType = trueBuffer.getType().cast<MemRefType>();
      int64_t dynamicOffset = ShapedType::kDynamicStrideOrOffset;
      SmallVector<int64_t> dynamicStrides(trueType.getRank(),
                                          ShapedType::kDynamicStrideOrOffset);
      AffineMap stridedLayout = makeStridedLinearLayoutMap(
          dynamicStrides, dynamicOffset, op->getContext());
      auto castedType =
          MemRefType::get(trueType.getShape(), trueType.getElementType(),
                          stridedLayout, trueType.getMemorySpaceAsInt());
      trueBuffer = rewriter.create<memref::CastOp>(loc, castedType, trueBuffer);
      falseBuffer =
          rewriter.create<memref::CastOp>(loc, castedType, falseBuffer);
    }

    replaceOpWithNewBufferizedOp<arith::SelectOp>(
        rewriter, op, selectOp.getCondition(), trueBuffer, falseBuffer);
    return success();
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::None;
  }
};

} // namespace

void mlir::arith::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, ArithmeticDialect *dialect) {
    ConstantOp::attachInterface<ConstantOpInterface>(*ctx);
    IndexCastOp::attachInterface<IndexCastOpInterface>(*ctx);
    SelectOp::attachInterface<SelectOpInterface>(*ctx);
  });
}
