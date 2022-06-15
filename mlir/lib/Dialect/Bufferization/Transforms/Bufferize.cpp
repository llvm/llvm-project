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
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

static BufferizationOptions::LayoutMapOption
parseLayoutMapOption(const std::string &s) {
  if (s == "fully-dynamic-layout-map")
    return BufferizationOptions::LayoutMapOption::FullyDynamicLayoutMap;
  if (s == "identity-layout-map")
    return BufferizationOptions::LayoutMapOption::IdentityLayoutMap;
  if (s == "infer-layout-map")
    return BufferizationOptions::LayoutMapOption::InferLayoutMap;
  llvm_unreachable("invalid layout map option");
}

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
      opt.functionBoundaryTypeConversion =
          parseLayoutMapOption(functionBoundaryTypeConversion);
      opt.printConflicts = printConflicts;
      opt.testAnalysisOnly = testAnalysisOnly;
      opt.bufferizeFunctionBoundaries = bufferizeFunctionBoundaries;
      opt.unknownTypeConversion = parseLayoutMapOption(unknownTypeConversion);

      OpFilter::Entry::FilterFn filterFn =
          [&](Operation *op) {
            // Filter may be specified via options.
            if (this->dialectFilter.hasValue())
              return llvm::is_contained(this->dialectFilter,
                                        op->getDialect()->getNamespace());
            // No filter specified: All other ops are allowed.
            return true;
          };
      opt.opFilter.allowOperation(filterFn);
    } else {
      opt = *options;
    }

    ModuleOp moduleOp = getOperation();
    if (opt.bufferizeFunctionBoundaries) {
      if (failed(runOneShotModuleBufferize(moduleOp, opt))) {
        signalPassFailure();
        return;
      }
    } else {
      if (failed(runOneShotBufferize(moduleOp, opt))) {
        signalPassFailure();
        return;
      }
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

namespace {
struct BufferizationBufferizePass
    : public BufferizationBufferizeBase<BufferizationBufferizePass> {
  void runOnOperation() override {
    BufferizationOptions options = getPartialBufferizationOptions();
    options.opFilter.allowDialect<BufferizationDialect>();

    if (failed(bufferizeOp(getOperation(), options)))
      signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<bufferization::BufferizationDialect, memref::MemRefDialect>();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::bufferization::createBufferizationBufferizePass() {
  return std::make_unique<BufferizationBufferizePass>();
}

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
  if (auto funcOp = dyn_cast<FunctionOpInterface>(op)) {
    bool hasTensorArg = any_of(funcOp.getArgumentTypes(), isaTensor);
    bool hasTensorResult = any_of(funcOp.getResultTypes(), isaTensor);
    return hasTensorArg || hasTensorResult;
  }

  bool hasTensorResult = any_of(op->getResultTypes(), isaTensor);
  bool hasTensorOperand = any_of(op->getOperandTypes(), isaTensor);
  return hasTensorResult || hasTensorOperand;
}

LogicalResult bufferization::bufferizeOp(Operation *op,
                                         const AnalysisState &analysisState) {
  // Catch incorrect API usage.
  assert((analysisState.hasDialectState(
              func::FuncDialect::getDialectNamespace()) ||
          !analysisState.getOptions().bufferizeFunctionBoundaries) &&
         "must use ModuleBufferize to bufferize function boundaries");

  BufferizationState bufferizationState(analysisState);
  if (failed(bufferizeOp(op, bufferizationState)))
    return failure();
  return success();
}

namespace {
/// A rewriter that keeps track of extra information during bufferization.
class BufferizationRewriter : public IRRewriter {
public:
  BufferizationRewriter(MLIRContext *ctx, DenseSet<Operation *> &erasedOps,
                        DenseSet<Operation *> &toMemrefOps,
                        const BufferizationOptions &options,
                        const OpFilter *opFilter)
      : IRRewriter(ctx), erasedOps(erasedOps), toMemrefOps(toMemrefOps),
        options(options), opFilter(opFilter) {}

protected:
  void notifyOperationRemoved(Operation *op) override {
    IRRewriter::notifyOperationRemoved(op);
    erasedOps.insert(op);
    // Erase if present.
    toMemrefOps.erase(op);
  }

  void notifyOperationInserted(Operation *op) override {
    IRRewriter::notifyOperationInserted(op);

    // Keep track of to_memref ops.
    if (isa<ToMemrefOp>(op)) {
      toMemrefOps.insert(op);
      return;
    }

    // Skip to_tensor ops.
    if (isa<ToTensorOp>(op))
      return;

    // Skip non-tensor ops.
    if (!hasTensorSemantics(op))
      return;

    // Skip ops that are not allowed.
    if (!options.isOpAllowed(op) || (opFilter && !opFilter->isOpAllowed(op)))
      return;

    // Adding new bufferizable ops is not allowed during bufferization. Such ops
    // would not be analyzed and can lead to surprising behavior.
    llvm_unreachable(
        "creating new tensor ops is not allowed during bufferization");
  }

private:
  /// A set of all erased ops.
  DenseSet<Operation *> &erasedOps;

  /// A set of all to_memref ops.
  DenseSet<Operation *> &toMemrefOps;

  /// The bufferization options.
  /// Used for debug modes.
  LLVM_ATTRIBUTE_UNUSED
  const BufferizationOptions &options;

  const OpFilter *opFilter;
};
} // namespace

LogicalResult bufferization::bufferizeOp(Operation *op,
                                         BufferizationState &bufferizationState,
                                         const OpFilter *opFilter) {
  const auto &options = bufferizationState.getOptions();
  assert(options.unknownTypeConversion !=
             BufferizationOptions::LayoutMapOption::InferLayoutMap &&
         "invalid layout map option");

  // Keep track of to_memref ops.
  DenseSet<Operation *> toMemrefOps;
  op->walk([&](ToMemrefOp toMemrefOp) { toMemrefOps.insert(toMemrefOp); });

  // Gather all bufferizable ops in top-to-bottom order.
  //
  // We should ideally know the exact memref type of all operands when
  // bufferizing an op. (This is the case when bufferizing top-to-bottom.)
  // Otherwise, we have to use a memref type with a fully dynamic layout map to
  // avoid copies. We are currently missing patterns for layout maps to
  // canonicalize away (or canonicalize to more precise layouts).
  SmallVector<Operation *> worklist;
  op->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (hasTensorSemantics(op))
      worklist.push_back(op);
  });

  // Keep track of all erased ops.
  DenseSet<Operation *> erasedOps;

  // Bufferize all ops.
  BufferizationRewriter rewriter(op->getContext(), erasedOps, toMemrefOps,
                                 bufferizationState.getOptions(), opFilter);
  for (unsigned i = 0; i < worklist.size(); ++i) {
    Operation *op = worklist[i];
    // Skip ops that were erased.
    if (erasedOps.contains(op))
      continue;
    // Skip ops that are not bufferizable or not allowed.
    auto bufferizableOp = options.dynCastBufferizableOp(op);
    if (!bufferizableOp)
      continue;
    if (opFilter && !opFilter->isOpAllowed(op))
      continue;
    // Skip ops that no longer have tensor semantics.
    if (!hasTensorSemantics(op))
      continue;
    // Bufferize the op.
    rewriter.setInsertionPoint(op);
    if (failed(bufferizableOp.bufferize(rewriter, bufferizationState)))
      return op->emitError("failed to bufferize op");
  }

  // Fold all to_memref(to_tensor(x)) pairs.
  for (Operation *op : toMemrefOps) {
    rewriter.setInsertionPoint(op);
    (void)bufferization::foldToMemrefToTensorPair(rewriter,
                                                  cast<ToMemrefOp>(op));
  }

  /// Check the result of bufferization. Return an error if an op was not
  /// bufferized, unless partial bufferization is allowed.
  if (bufferizationState.getOptions().allowUnknownOps)
    return success();

  for (Operation *op : worklist) {
    // Skip ops that are entirely gone.
    if (erasedOps.contains(op))
      continue;
    // Ops that no longer have tensor semantics (because they were updated
    // in-place) are allowed.
    if (!hasTensorSemantics(op))
      continue;
    // Continue ops that are not allowed.
    if (!options.isOpAllowed(op))
      continue;
    if (opFilter && !opFilter->isOpAllowed(op))
      continue;
    // Ops without any uses and no side effects will fold away.
    if (op->getUses().empty() && MemoryEffectOpInterface::hasNoEffect(op))
      continue;
    return op->emitError("op was not bufferized");
  }

  return success();
}

LogicalResult bufferization::bufferizeOp(Operation *op,
                                         const BufferizationOptions &options) {
  AnalysisState state(options);
  return bufferizeOp(op, state);
}

BufferizationOptions bufferization::getPartialBufferizationOptions() {
  BufferizationOptions options;
  options.allowUnknownOps = true;
  options.createDeallocs = false;
  options.unknownTypeConversion =
      BufferizationOptions::LayoutMapOption::IdentityLayoutMap;
  return options;
}
