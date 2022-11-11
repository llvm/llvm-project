//===- Bufferize.cpp - Bufferization utilities ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace bufferization {
#define GEN_PASS_DEF_FINALIZINGBUFFERIZE
#define GEN_PASS_DEF_BUFFERIZATIONBUFFERIZE
#define GEN_PASS_DEF_ONESHOTBUFFERIZE
#include "mlir/Dialect/Bufferization/Transforms/Passes.h.inc"
} // namespace bufferization
} // namespace mlir

#define DEBUG_TYPE "bufferize"

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
    rewriter.replaceOp(op, adaptor.getMemref());
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
    rewriter.replaceOp(op, adaptor.getTensor());
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
    : public bufferization::impl::FinalizingBufferizeBase<
          FinalizingBufferizePass> {
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

static OneShotBufferizationOptions::AnalysisHeuristic
parseHeuristicOption(const std::string &s) {
  if (s == "bottom-up")
    return OneShotBufferizationOptions::AnalysisHeuristic::BottomUp;
  if (s == "top-down")
    return OneShotBufferizationOptions::AnalysisHeuristic::TopDown;
  llvm_unreachable("invalid analysisheuristic option");
}

struct OneShotBufferizePass
    : public bufferization::impl::OneShotBufferizeBase<OneShotBufferizePass> {
  OneShotBufferizePass() = default;

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
      opt.analysisHeuristic = parseHeuristicOption(analysisHeuristic);
      opt.copyBeforeWrite = copyBeforeWrite;
      opt.createDeallocs = createDeallocs;
      opt.functionBoundaryTypeConversion =
          parseLayoutMapOption(functionBoundaryTypeConversion);
      if (mustInferMemorySpace)
        opt.defaultMemorySpace = None;
      opt.printConflicts = printConflicts;
      opt.testAnalysisOnly = testAnalysisOnly;
      opt.bufferizeFunctionBoundaries = bufferizeFunctionBoundaries;

      // Configure type converter.
      BufferizationOptions::LayoutMapOption unknownTypeConversionOption =
          parseLayoutMapOption(unknownTypeConversion);
      opt.unknownTypeConverterFn = [=](Value value, unsigned memorySpace,
                                       const BufferizationOptions &options) {
        auto tensorType = value.getType().cast<TensorType>();
        if (unknownTypeConversionOption ==
            BufferizationOptions::LayoutMapOption::IdentityLayoutMap)
          return bufferization::getMemRefTypeWithStaticIdentityLayout(
              tensorType, memorySpace);
        assert(
            unknownTypeConversionOption ==
                BufferizationOptions::LayoutMapOption::FullyDynamicLayoutMap &&
            "invalid layout map option");
        return bufferization::getMemRefTypeWithFullyDynamicLayout(tensorType,
                                                                  memorySpace);
      };

      // Configure op filter.
      OpFilter::Entry::FilterFn filterFn = [&](Operation *op) {
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
    : public bufferization::impl::BufferizationBufferizeBase<
          BufferizationBufferizePass> {
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

namespace {
/// A rewriter that keeps track of extra information during bufferization.
class BufferizationRewriter : public IRRewriter {
public:
  BufferizationRewriter(MLIRContext *ctx, DenseSet<Operation *> &erasedOps,
                        DenseSet<Operation *> &toMemrefOps,
                        SmallVector<Operation *> &worklist,
                        const BufferizationOptions &options,
                        const OpFilter *opFilter)
      : IRRewriter(ctx), erasedOps(erasedOps), toMemrefOps(toMemrefOps),
        worklist(worklist), analysisState(options), opFilter(opFilter) {}

protected:
  void notifyOperationRemoved(Operation *op) override {
    IRRewriter::notifyOperationRemoved(op);
    erasedOps.insert(op);
    // Erase if present.
    toMemrefOps.erase(op);
  }

  void notifyOperationInserted(Operation *op) override {
    IRRewriter::notifyOperationInserted(op);
    erasedOps.erase(op);

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

    // Skip ops that are not allowed to be bufferized.
    auto const &options = analysisState.getOptions();
    if (!options.isOpAllowed(op) || (opFilter && !opFilter->isOpAllowed(op)))
      return;

    // Add op to worklist.
    worklist.push_back(op);
  }

private:
  /// A set of all erased ops.
  DenseSet<Operation *> &erasedOps;

  /// A set of all to_memref ops.
  DenseSet<Operation *> &toMemrefOps;

  /// The worklist of ops to be bufferized.
  SmallVector<Operation *> &worklist;

  /// The analysis state. Used for debug assertions and access to the
  /// bufferization options.
  const AnalysisState analysisState;

  /// An extra op filter for bufferization.
  const OpFilter *opFilter;
};
} // namespace

LogicalResult bufferization::bufferizeOp(Operation *op,
                                         const BufferizationOptions &options,
                                         bool copyBeforeWrite,
                                         const OpFilter *opFilter) {
  if (copyBeforeWrite) {
    AnalysisState state(options);
    if (failed(insertTensorCopies(op, state)))
      return failure();
  }

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
  //
  // FuncOps must be bufferized before their bodies, so add them to the worklist
  // first.
  SmallVector<Operation *> worklist;
  op->walk([&](func::FuncOp funcOp) {
    if (hasTensorSemantics(funcOp))
      worklist.push_back(funcOp);
  });
  op->walk<WalkOrder::PostOrder>([&](Operation *op) {
    if (hasTensorSemantics(op) && !isa<func::FuncOp>(op))
      worklist.push_back(op);
  });

  // Keep track of all erased ops.
  DenseSet<Operation *> erasedOps;

  // Bufferize all ops.
  BufferizationRewriter rewriter(op->getContext(), erasedOps, toMemrefOps,
                                 worklist, options, opFilter);
  for (unsigned i = 0; i < worklist.size(); ++i) {
    Operation *nextOp = worklist[i];
    // Skip ops that were erased.
    if (erasedOps.contains(nextOp))
      continue;
    // Skip ops that are not bufferizable or not allowed.
    auto bufferizableOp = options.dynCastBufferizableOp(nextOp);
    if (!bufferizableOp)
      continue;
    if (opFilter && !opFilter->isOpAllowed(nextOp))
      continue;
    // Skip ops that no longer have tensor semantics.
    if (!hasTensorSemantics(nextOp))
      continue;
    // Bufferize the op.
    LLVM_DEBUG(llvm::dbgs()
               << "//===-------------------------------------------===//\n"
               << "IR after bufferizing: " << nextOp->getName() << "\n");
    rewriter.setInsertionPoint(nextOp);
    if (failed(bufferizableOp.bufferize(rewriter, options))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "failed to bufferize\n"
                 << "//===-------------------------------------------===//\n");
      return nextOp->emitError("failed to bufferize op");
    }
    LLVM_DEBUG(llvm::dbgs()
               << *op
               << "\n//===-------------------------------------------===//\n");
  }

  // Fold all to_memref(to_tensor(x)) pairs.
  for (Operation *op : toMemrefOps) {
    rewriter.setInsertionPoint(op);
    (void)bufferization::foldToMemrefToTensorPair(rewriter,
                                                  cast<ToMemrefOp>(op));
  }

  /// Check the result of bufferization. Return an error if an op was not
  /// bufferized, unless partial bufferization is allowed.
  if (options.allowUnknownOps)
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
    // ToTensorOps/ToMemrefOps are allowed in the output.
    if (isa<ToTensorOp, ToMemrefOp>(op))
      continue;
    return op->emitError("op was not bufferized");
  }

  return success();
}

BufferizationOptions bufferization::getPartialBufferizationOptions() {
  BufferizationOptions options;
  options.allowUnknownOps = true;
  options.createDeallocs = false;
  options.enforceAliasingInvariants = false;
  options.unknownTypeConverterFn = [](Value value, unsigned memorySpace,
                                      const BufferizationOptions &options) {
    return getMemRefTypeWithStaticIdentityLayout(
        value.getType().cast<TensorType>(), memorySpace);
  };
  options.opFilter.allowDialect<BufferizationDialect>();
  return options;
}
