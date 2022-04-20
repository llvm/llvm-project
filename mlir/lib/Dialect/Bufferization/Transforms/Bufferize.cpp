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
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

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
    assert(inputs.size() == 1 && "expected exactly one input");

    if (auto inputType = inputs[0].getType().dyn_cast<MemRefType>()) {
      // MemRef to MemRef cast.
      assert(inputType != type && "expected different types");
      // Unranked to ranked and ranked to unranked casts must be explicit.
      auto rankedDestType = type.dyn_cast<MemRefType>();
      if (!rankedDestType)
        return nullptr;
      FailureOr<Value> replacement =
          castOrReallocMemRefValue(builder, inputs[0], rankedDestType);
      if (failed(replacement))
        return nullptr;
      return *replacement;
    }

    if (inputs[0].getType().isa<TensorType>()) {
      // Tensor to MemRef cast.
      return builder.create<bufferization::ToMemrefOp>(loc, type, inputs[0]);
    }

    llvm_unreachable("only tensor/memref input types supported");
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

struct OneShotBufferizePass
    : public OneShotBufferizeBase<OneShotBufferizePass> {
  OneShotBufferizePass() : OneShotBufferizeBase<OneShotBufferizePass>() {}

  explicit OneShotBufferizePass(const OneShotBufferizationOptions &options)
      : options(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<bufferization::BufferizationDialect, memref::MemRefDialect>();
    registerAllocationOpInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    OneShotBufferizationOptions opt;
    if (!options) {
      // Make new bufferization options if none were provided when creating the
      // pass.
      opt.allowReturnAllocs = allowReturnAllocs;
      opt.allowUnknownOps = allowUnknownOps;
      opt.analysisFuzzerSeed = analysisFuzzerSeed;
      opt.createDeallocs = createDeallocs;
      opt.fullyDynamicLayoutMaps = fullyDynamicLayoutMaps;
      opt.printConflicts = printConflicts;
      opt.testAnalysisOnly = testAnalysisOnly;

      BufferizationOptions::OpFilterEntry::FilterFn filterFn =
          [&](Operation *op) {
            // Disallow non-func dialect ops. I.e., no ops related to function
            // calls.
            if (isa<func::FuncDialect>(op->getDialect()))
              return false;
            // Filter may be specified via options.
            if (this->dialectFilter.hasValue())
              return llvm::find(this->dialectFilter,
                                op->getDialect()->getNamespace()) !=
                     this->dialectFilter.end();
            // No filter specified: All other ops are allowed.
            return true;
          };
      opt.allowOperationInFilter(filterFn);
    } else {
      opt = *options;
    }

    ModuleOp moduleOp = getOperation();
    if (failed(runOneShotBufferize(moduleOp, opt))) {
      signalPassFailure();
      return;
    }

    if (opt.testAnalysisOnly)
      return;

    OpPassManager cleanupPipeline("builtin.module");
    cleanupPipeline.addPass(createCanonicalizerPass());
    cleanupPipeline.addPass(createCSEPass());
    cleanupPipeline.addPass(createLoopInvariantCodeMotionPass());
    (void)runPipeline(cleanupPipeline, moduleOp);
  }

private:
  llvm::Optional<OneShotBufferizationOptions> options;
};
} // namespace

std::unique_ptr<Pass> mlir::bufferization::createOneShotBufferizePass() {
  return std::make_unique<OneShotBufferizePass>();
}

std::unique_ptr<Pass> mlir::bufferization::createOneShotBufferizePass(
    const OneShotBufferizationOptions &options) {
  return std::make_unique<OneShotBufferizePass>(options);
}

std::unique_ptr<OperationPass<func::FuncOp>>
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
  BufferizationPattern(MLIRContext *context, BufferizationState &state,
                       PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern<BufferizableOpInterface>(context, benefit),
        state(&state) {}

  LogicalResult matchAndRewrite(BufferizableOpInterface bufferizableOp,
                                PatternRewriter &rewriter) const override {
    const BufferizationOptions &options = state->getOptions();

    // No tensors => no buffers.
    if (!hasTensorSemantics(bufferizableOp.getOperation()))
      return failure();
    if (!options.isOpAllowed(bufferizableOp.getOperation()))
      return failure();
    return bufferizableOp.bufferize(rewriter, *state);
  }

private:
  BufferizationState *const state;
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

LogicalResult
bufferization::finalizeBuffers(Operation *op,
                               const BufferizationOptions &options) {
  // Hoist buffers.
  if (failed(hoistBufferAllocations(op, options)))
    return failure();

  // Deallocate buffers that escape block boundaries ("leaking buffers") with
  // the buffer deallocation pass.
  bool hasLeakingAlloc = false;
  if (failed(createAllocDeallocOps(op, options, /*onlyLeakingAllocs=*/true,
                                   &hasLeakingAlloc)))
    return failure();
  if (options.createDeallocs && hasLeakingAlloc &&
      failed(deallocateBuffers(op)))
    return failure();

  // Deallocate all remaining buffers at the end of the block.
  if (failed(createAllocDeallocOps(op, options)))
    return failure();

  return success();
}

LogicalResult bufferization::bufferizeOp(Operation *op,
                                         const AnalysisState &analysisState) {
  BufferizationState bufferizationState(analysisState);
  if (failed(bufferizeOp(op, bufferizationState)))
    return failure();
  if (failed(finalizeBuffers(op, analysisState.getOptions())))
    return failure();
  return success();
}

LogicalResult
bufferization::bufferizeOp(Operation *op,
                           BufferizationState &bufferizationState) {
  // Bufferize the op and its nested ops.
  RewritePatternSet patterns(op->getContext());
  patterns.add<BufferizationPattern>(patterns.getContext(), bufferizationState);

  // Bufferize ops top-to-bottom. When creating a new op, we should ideally
  // know the exact memref type of all operands. Otherwise, we have to use a
  // memref type with a fully dynamic layout map, which has to canonicalize
  // away. This is less efficient.
  //
  // Note: If "fullyDynamicLayoutMaps = false", we may have to insert buffer
  // copies to fold ("finalize") to_memref(to_tensor(x)) ops with non-cast-
  // compatible layout maps when doing a traversal other than top-to-bottom.
  // There are currently no canonicalization patterns to fold these away.
  GreedyRewriteConfig config;
  config.useTopDownTraversal = true;

  // TODO: Perform a preorder walk instead of the greedy pattern rewriter. This
  // would be more efficient because every bufferization pattern is guaranteed
  // to apply only a single time (otherwise, an assertion would be triggered).
  // However, there are restrictions wrt. erasing ops during a preorder walk,
  // which would likely require a larger refactoring.
  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns), config)))
    return failure();

  if (failed(checkBufferizationResult(op, bufferizationState.getOptions())))
    return failure();

  return success();
}

namespace {
/// This a "no analysis, always copy" AnalysisState. In the absence of an
/// analysis, a buffer must be copied each time it is written to. Therefore, all
/// OpOperands that bufferize to a memory write must bufferize out-of-place.
class AlwaysCopyAnalysisState : public AnalysisState {
public:
  AlwaysCopyAnalysisState(const BufferizationOptions &options)
      : AnalysisState(options) {}

  AlwaysCopyAnalysisState(const AlwaysCopyAnalysisState &) = delete;

  virtual ~AlwaysCopyAnalysisState() = default;

  /// Return `true` if the given OpResult has been decided to bufferize inplace.
  bool isInPlace(OpOperand &opOperand) const override {
    // OpOperands that bufferize to a memory write are out-of-place, i.e., an
    // alloc and copy is inserted.
    return !bufferizesToMemoryWrite(opOperand);
  }

  /// Return true if `v1` and `v2` bufferize to equivalent buffers.
  bool areEquivalentBufferizedValues(Value v1, Value v2) const override {
    // There is no analysis, so we do not know if the values are equivalent. The
    // conservative answer is "false".
    return false;
  }
};
} // namespace

LogicalResult bufferization::bufferizeOp(Operation *op,
                                         const BufferizationOptions &options) {
  AlwaysCopyAnalysisState state(options);
  return bufferizeOp(op, state);
}

BufferizationOptions bufferization::getPartialBufferizationOptions() {
  BufferizationOptions options;
  options.allowUnknownOps = true;
  options.createDeallocs = false;
  options.fullyDynamicLayoutMaps = false;
  return options;
}
