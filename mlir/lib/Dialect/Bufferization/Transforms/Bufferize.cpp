//===- Bufferize.cpp - Bufferization utilities ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::bufferization;

//===----------------------------------------------------------------------===//
// BufferizeTypeConverter
//===----------------------------------------------------------------------===//

static Value materializeToTensor(OpBuilder &builder, TensorType type,
                                 ValueRange inputs, Location loc) {
  assert(inputs.size() == 1);
  assert(inputs[0].getType().isa<BaseMemRefType>());
  return builder.create<bufferization::ToTensorOp>(loc, type, inputs[0]);
}

/// Registers conversions into BufferizeTypeConverter
BufferizeTypeConverter::BufferizeTypeConverter() {
  // Keep all types unchanged.
  addConversion([](Type type) { return type; });
  // Convert RankedTensorType to MemRefType.
  addConversion([](RankedTensorType type) -> Type {
    return MemRefType::get(type.getShape(), type.getElementType());
  });
  // Convert UnrankedTensorType to UnrankedMemRefType.
  addConversion([](UnrankedTensorType type) -> Type {
    return UnrankedMemRefType::get(type.getElementType(), 0);
  });
  addArgumentMaterialization(materializeToTensor);
  addSourceMaterialization(materializeToTensor);
  addTargetMaterialization([](OpBuilder &builder, BaseMemRefType type,
                              ValueRange inputs, Location loc) -> Value {
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<TensorType>());
    return builder.create<bufferization::ToMemrefOp>(loc, type, inputs[0]);
  });
}

void mlir::bufferization::populateBufferizeMaterializationLegality(
    ConversionTarget &target) {
  target.addLegalOp<bufferization::ToTensorOp, bufferization::ToMemrefOp>();
}

namespace {
// In a finalizing bufferize conversion, we know that all tensors have been
// converted to memrefs, thus, this op becomes an identity.
class BufferizeToTensorOp
    : public OpConversionPattern<bufferization::ToTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(bufferization::ToTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.memref());
    return success();
  }
};
} // namespace

namespace {
// In a finalizing bufferize conversion, we know that all tensors have been
// converted to memrefs, thus, this op becomes an identity.
class BufferizeToMemrefOp
    : public OpConversionPattern<bufferization::ToMemrefOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(bufferization::ToMemrefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.tensor());
    return success();
  }
};
} // namespace

void mlir::bufferization::populateEliminateBufferizeMaterializationsPatterns(
    BufferizeTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<BufferizeToTensorOp, BufferizeToMemrefOp>(typeConverter,
                                                         patterns.getContext());
  populateBufferizationOpFoldingPatterns(patterns, patterns.getContext());
}

namespace {
struct FinalizingBufferizePass
    : public FinalizingBufferizeBase<FinalizingBufferizePass> {
  using FinalizingBufferizeBase<
      FinalizingBufferizePass>::FinalizingBufferizeBase;

  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();

    BufferizeTypeConverter typeConverter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    populateEliminateBufferizeMaterializationsPatterns(typeConverter, patterns);

    // If all result types are legal, and all block arguments are legal (ensured
    // by func conversion above), then all types in the program are legal.
    //
    // We also check that the operand types are legal to avoid creating invalid
    // IR. For example, this prevents
    // populateEliminateBufferizeMaterializationsPatterns from updating the
    // types of the operands to a return op without updating the enclosing
    // function.
    target.markUnknownOpDynamicallyLegal(
        [&](Operation *op) { return typeConverter.isLegal(op); });

    if (failed(applyFullConversion(func, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::bufferization::createFinalizingBufferizePass() {
  return std::make_unique<FinalizingBufferizePass>();
}

//===----------------------------------------------------------------------===//
// BufferizableOpInterface-based Bufferization
//===----------------------------------------------------------------------===//

static bool isaTensor(Type t) { return t.isa<TensorType>(); }

/// Return true if the given op has a tensor result or a tensor operand.
static bool hasTensorSemantics(Operation *op) {
  bool hasTensorResult = any_of(op->getResultTypes(), isaTensor);
  bool hasTensorOperand = any_of(op->getOperandTypes(), isaTensor);
  return hasTensorResult || hasTensorOperand;
}

/// Rewrite pattern that bufferizes bufferizable ops.
struct BufferizationPattern
    : public OpInterfaceRewritePattern<BufferizableOpInterface> {
  BufferizationPattern(MLIRContext *context, const BufferizationState &state,
                       PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern<BufferizableOpInterface>(context, benefit),
        state(state) {}

  LogicalResult matchAndRewrite(BufferizableOpInterface bufferizableOp,
                                PatternRewriter &rewriter) const override {
    // No tensors => no buffers.
    if (!hasTensorSemantics(bufferizableOp.getOperation()))
      return failure();
    if (!state.getOptions().isOpAllowed(bufferizableOp.getOperation()))
      return failure();
    return bufferizableOp.bufferize(rewriter, state);
  }

private:
  const BufferizationState &state;
};

/// Check the result of bufferization. Return an error if an op was not
/// bufferized, unless partial bufferization is allowed.
static LogicalResult
checkBufferizationResult(Operation *op, const BufferizationOptions &options) {
  if (!options.allowUnknownOps) {
    // Check if all ops were bufferized.
    LogicalResult status = success();
    op->walk([&](Operation *op) {
      if (!hasTensorSemantics(op))
        return WalkResult::advance();

      // Bufferization dialect ops will canonicalize away if all other ops are
      // bufferized.
      if (isa<bufferization::ToMemrefOp, bufferization::ToTensorOp>(op))
        return WalkResult::advance();

      // Ops that are not in the allow list can be ignored.
      if (!options.isOpAllowed(op))
        return WalkResult::advance();

      // Ops without any uses and no side effects will fold away.
      if (op->getUses().empty() && MemoryEffectOpInterface::hasNoEffect(op))
        return WalkResult::advance();

      status = op->emitError("op was not bufferized");
      return WalkResult::interrupt();
    });

    if (failed(status))
      return status;
  }

  return success();
}

LogicalResult bufferization::bufferizeOp(Operation *op,
                                         const BufferizationState &state) {
  // Bufferize the op and its nested ops.
  RewritePatternSet patterns(op->getContext());
  patterns.add<BufferizationPattern>(op->getContext(), state);
  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
    return failure();

  return checkBufferizationResult(op, state.getOptions());
}
