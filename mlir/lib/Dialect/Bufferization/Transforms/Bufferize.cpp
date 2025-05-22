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
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include <optional>

namespace mlir {
namespace bufferization {
#define GEN_PASS_DEF_ONESHOTBUFFERIZEPASS
#include "mlir/Dialect/Bufferization/Transforms/Passes.h.inc"
} // namespace bufferization
} // namespace mlir

#define DEBUG_TYPE "bufferize"

using namespace mlir;
using namespace mlir::bufferization;

namespace {

static OneShotBufferizationOptions::AnalysisHeuristic
parseHeuristicOption(const std::string &s) {
  if (s == "bottom-up")
    return OneShotBufferizationOptions::AnalysisHeuristic::BottomUp;
  if (s == "top-down")
    return OneShotBufferizationOptions::AnalysisHeuristic::TopDown;
  if (s == "bottom-up-from-terminators")
    return OneShotBufferizationOptions::AnalysisHeuristic::
        BottomUpFromTerminators;
  if (s == "fuzzer")
    return OneShotBufferizationOptions::AnalysisHeuristic::Fuzzer;
  llvm_unreachable("invalid analysisheuristic option");
}

struct OneShotBufferizePass
    : public bufferization::impl::OneShotBufferizePassBase<
          OneShotBufferizePass> {
  using Base::Base;

  void runOnOperation() override {
    OneShotBufferizationOptions opt;
    if (!options) {
      // Make new bufferization options if none were provided when creating the
      // pass.
      opt.allowReturnAllocsFromLoops = allowReturnAllocsFromLoops;
      opt.allowUnknownOps = allowUnknownOps;
      opt.analysisFuzzerSeed = analysisFuzzerSeed;
      opt.analysisHeuristic = parseHeuristicOption(analysisHeuristic);
      opt.copyBeforeWrite = copyBeforeWrite;
      opt.dumpAliasSets = dumpAliasSets;
      opt.setFunctionBoundaryTypeConversion(functionBoundaryTypeConversion);

      if (mustInferMemorySpace && useEncodingForMemorySpace) {
        emitError(getOperation()->getLoc())
            << "only one of 'must-infer-memory-space' and "
               "'use-encoding-for-memory-space' are allowed in "
            << getArgument();
        return signalPassFailure();
      }

      if (mustInferMemorySpace) {
        opt.defaultMemorySpaceFn =
            [](TensorType t) -> std::optional<Attribute> {
          return std::nullopt;
        };
      }

      if (useEncodingForMemorySpace) {
        opt.defaultMemorySpaceFn =
            [](TensorType t) -> std::optional<Attribute> {
          if (auto rtt = dyn_cast<RankedTensorType>(t))
            return rtt.getEncoding();
          return std::nullopt;
        };
      }

      opt.printConflicts = printConflicts;
      opt.bufferAlignment = bufferAlignment;
      opt.testAnalysisOnly = testAnalysisOnly;
      opt.bufferizeFunctionBoundaries = bufferizeFunctionBoundaries;
      opt.checkParallelRegions = checkParallelRegions;
      opt.noAnalysisFuncFilter = noAnalysisFuncFilter;

      // Configure type converter.
      LayoutMapOption unknownTypeConversionOption = unknownTypeConversion;
      if (unknownTypeConversionOption == LayoutMapOption::InferLayoutMap) {
        emitError(UnknownLoc::get(&getContext()),
                  "Invalid option: 'infer-layout-map' is not a valid value for "
                  "'unknown-type-conversion'");
        return signalPassFailure();
      }
      opt.unknownTypeConverterFn = [=](Value value, Attribute memorySpace,
                                       const BufferizationOptions &options) {
        auto tensorType = cast<TensorType>(value.getType());
        if (unknownTypeConversionOption == LayoutMapOption::IdentityLayoutMap)
          return bufferization::getMemRefTypeWithStaticIdentityLayout(
              tensorType, memorySpace);
        assert(unknownTypeConversionOption ==
                   LayoutMapOption::FullyDynamicLayoutMap &&
               "invalid layout map option");
        return bufferization::getMemRefTypeWithFullyDynamicLayout(tensorType,
                                                                  memorySpace);
      };

      // Configure op filter.
      OpFilter::Entry::FilterFn filterFn = [&](Operation *op) {
        // Filter may be specified via options.
        if (this->dialectFilter.hasValue() && !(*this->dialectFilter).empty())
          return llvm::is_contained(this->dialectFilter,
                                    op->getDialect()->getNamespace());
        // No filter specified: All other ops are allowed.
        return true;
      };
      opt.opFilter.allowOperation(filterFn);
    } else {
      opt = *options;
    }

    if (opt.copyBeforeWrite && opt.testAnalysisOnly) {
      // These two flags do not make sense together: "copy-before-write"
      // indicates that copies should be inserted before every memory write,
      // but "test-analysis-only" indicates that only the analysis should be
      // tested. (I.e., no IR is bufferized.)
      emitError(UnknownLoc::get(&getContext()),
                "Invalid option: 'copy-before-write' cannot be used with "
                "'test-analysis-only'");
      return signalPassFailure();
    }

    if (opt.printConflicts && !opt.testAnalysisOnly) {
      emitError(
          UnknownLoc::get(&getContext()),
          "Invalid option: 'print-conflicts' requires 'test-analysis-only'");
      return signalPassFailure();
    }

    if (opt.dumpAliasSets && !opt.testAnalysisOnly) {
      emitError(
          UnknownLoc::get(&getContext()),
          "Invalid option: 'dump-alias-sets' requires 'test-analysis-only'");
      return signalPassFailure();
    }

    BufferizationState state;

    BufferizationStatistics statistics;
    ModuleOp moduleOp = getOperation();
    if (opt.bufferizeFunctionBoundaries) {
      if (failed(
              runOneShotModuleBufferize(moduleOp, opt, state, &statistics))) {
        signalPassFailure();
        return;
      }
    } else {
      if (!opt.noAnalysisFuncFilter.empty()) {
        emitError(UnknownLoc::get(&getContext()),
                  "Invalid option: 'no-analysis-func-filter' requires "
                  "'bufferize-function-boundaries'");
        return signalPassFailure();
      }
      if (failed(runOneShotBufferize(moduleOp, opt, state, &statistics))) {
        signalPassFailure();
        return;
      }
    }

    // Set pass statistics.
    this->numBufferAlloc = statistics.numBufferAlloc;
    this->numTensorInPlace = statistics.numTensorInPlace;
    this->numTensorOutOfPlace = statistics.numTensorOutOfPlace;
  }

private:
  std::optional<OneShotBufferizationOptions> options;
};
} // namespace

//===----------------------------------------------------------------------===//
// BufferizableOpInterface-based Bufferization
//===----------------------------------------------------------------------===//

namespace {
/// A rewriter that keeps track of extra information during bufferization.
class BufferizationRewriter : public IRRewriter, public RewriterBase::Listener {
public:
  BufferizationRewriter(MLIRContext *ctx, DenseSet<Operation *> &erasedOps,
                        DenseSet<Operation *> &toBufferOps,
                        SmallVector<Operation *> &worklist,
                        const BufferizationOptions &options,
                        BufferizationStatistics *statistics)
      : IRRewriter(ctx), erasedOps(erasedOps), toBufferOps(toBufferOps),
        worklist(worklist), analysisState(options), statistics(statistics) {
    setListener(this);
  }

protected:
  void notifyOperationErased(Operation *op) override {
    erasedOps.insert(op);
    // Erase if present.
    toBufferOps.erase(op);
  }

  void notifyOperationInserted(Operation *op, InsertPoint previous) override {
    // We only care about newly created ops.
    if (previous.isSet())
      return;

    erasedOps.erase(op);

    // Gather statistics about allocs.
    if (statistics) {
      if (auto sideEffectingOp = dyn_cast<MemoryEffectOpInterface>(op))
        statistics->numBufferAlloc += static_cast<int64_t>(
            sideEffectingOp.hasEffect<MemoryEffects::Allocate>());
    }

    // Keep track of to_buffer ops.
    if (isa<ToBufferOp>(op)) {
      toBufferOps.insert(op);
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
    if (!options.isOpAllowed(op))
      return;

    // Add op to worklist.
    worklist.push_back(op);
  }

private:
  /// A set of all erased ops.
  DenseSet<Operation *> &erasedOps;

  /// A set of all to_buffer ops.
  DenseSet<Operation *> &toBufferOps;

  /// The worklist of ops to be bufferized.
  SmallVector<Operation *> &worklist;

  /// The analysis state. Used for debug assertions and access to the
  /// bufferization options.
  const AnalysisState analysisState;

  /// Bufferization statistics for debugging.
  BufferizationStatistics *statistics;
};
} // namespace

LogicalResult bufferization::bufferizeOp(Operation *op,
                                         const BufferizationOptions &options,
                                         BufferizationState &bufferizationState,
                                         BufferizationStatistics *statistics) {
  if (options.copyBeforeWrite) {
    AnalysisState state(options);
    if (failed(insertTensorCopies(op, state)))
      return failure();
  }

  // Keep track of to_buffer ops.
  DenseSet<Operation *> toBufferOps;
  op->walk([&](ToBufferOp toBufferOp) { toBufferOps.insert(toBufferOp); });

  // Gather all bufferizable ops in top-to-bottom order.
  //
  // We should ideally know the exact memref type of all operands when
  // bufferizing an op. (This is the case when bufferizing top-to-bottom.)
  // Otherwise, we have to use a memref type with a fully dynamic layout map to
  // avoid copies. We are currently missing patterns for layout maps to
  // canonicalize away (or canonicalize to more precise layouts).
  SmallVector<Operation *> worklist;
  op->walk<WalkOrder::PostOrder>([&](Operation *op) {
    if (options.isOpAllowed(op) && hasTensorSemantics(op))
      worklist.push_back(op);
  });

  // Keep track of all erased ops.
  DenseSet<Operation *> erasedOps;

  // Bufferize all ops.
  BufferizationRewriter rewriter(op->getContext(), erasedOps, toBufferOps,
                                 worklist, options, statistics);
  for (unsigned i = 0; i < worklist.size(); ++i) {
    Operation *nextOp = worklist[i];
    // Skip ops that were erased.
    if (erasedOps.contains(nextOp))
      continue;
    // Skip ops that are not bufferizable or not allowed.
    auto bufferizableOp = options.dynCastBufferizableOp(nextOp);
    if (!bufferizableOp)
      continue;
    // Skip ops that no longer have tensor semantics.
    if (!hasTensorSemantics(nextOp))
      continue;
    // Check for unsupported unstructured control flow.
    if (!bufferizableOp.supportsUnstructuredControlFlow())
      for (Region &r : nextOp->getRegions())
        if (r.getBlocks().size() > 1)
          return nextOp->emitOpError(
              "op or BufferizableOpInterface implementation does not support "
              "unstructured control flow, but at least one region has multiple "
              "blocks");

    // Bufferize the op.
    LLVM_DEBUG(llvm::dbgs()
               << "//===-------------------------------------------===//\n"
               << "IR after bufferizing: " << nextOp->getName() << "\n");
    rewriter.setInsertionPoint(nextOp);
    if (failed(
            bufferizableOp.bufferize(rewriter, options, bufferizationState))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "failed to bufferize\n"
                 << "//===-------------------------------------------===//\n");
      return nextOp->emitError("failed to bufferize op");
    }
    LLVM_DEBUG(llvm::dbgs()
               << *op
               << "\n//===-------------------------------------------===//\n");
  }

  // Return early if the top-level op is entirely gone.
  if (erasedOps.contains(op))
    return success();

  // Fold all to_buffer(to_tensor(x)) pairs.
  for (Operation *op : toBufferOps) {
    rewriter.setInsertionPoint(op);
    (void)bufferization::foldToBufferToTensorPair(
        rewriter, cast<ToBufferOp>(op), options);
  }

  // Remove all dead to_tensor ops.
  op->walk<WalkOrder::PostOrder>([&](ToTensorOp toTensorOp) {
    if (toTensorOp->getUses().empty()) {
      rewriter.eraseOp(toTensorOp);
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });

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
    // Ops without any uses and no side effects will fold away.
    if (op->getUses().empty() && isMemoryEffectFree(op))
      continue;
    // ToTensorOps/ToBufferOps are allowed in the output.
    if (isa<ToTensorOp, ToBufferOp>(op))
      continue;
    return op->emitError("op was not bufferized");
  }

  return success();
}

LogicalResult
bufferization::bufferizeBlockSignature(Block *block, RewriterBase &rewriter,
                                       const BufferizationOptions &options) {
  OpBuilder::InsertionGuard g(rewriter);
  auto bufferizableOp = options.dynCastBufferizableOp(block->getParentOp());
  if (!bufferizableOp)
    return failure();

  // Compute the new signature.
  SmallVector<Type> newTypes;
  for (BlockArgument &bbArg : block->getArguments()) {
    auto tensorType = dyn_cast<TensorType>(bbArg.getType());
    if (!tensorType) {
      newTypes.push_back(bbArg.getType());
      continue;
    }

    FailureOr<BaseMemRefType> memrefType =
        bufferization::getBufferType(bbArg, options);
    if (failed(memrefType))
      return failure();
    newTypes.push_back(*memrefType);
  }

  // Change the type of all block arguments.
  for (auto [bbArg, type] : llvm::zip(block->getArguments(), newTypes)) {
    if (bbArg.getType() == type)
      continue;

    // Collect all uses of the bbArg.
    SmallVector<OpOperand *> bbArgUses;
    for (OpOperand &use : bbArg.getUses())
      bbArgUses.push_back(&use);

    Type tensorType = bbArg.getType();
    // Change the bbArg type to memref.
    bbArg.setType(type);

    // Replace all uses of the original tensor bbArg.
    rewriter.setInsertionPointToStart(block);
    if (!bbArgUses.empty()) {
      Value toTensorOp = rewriter.create<bufferization::ToTensorOp>(
          bbArg.getLoc(), tensorType, bbArg);
      for (OpOperand *use : bbArgUses)
        use->set(toTensorOp);
    }
  }

  // Bufferize callers of the block.
  for (Operation *op : block->getUsers()) {
    auto branchOp = dyn_cast<BranchOpInterface>(op);
    if (!branchOp)
      return op->emitOpError("cannot bufferize ops with block references that "
                             "do not implement BranchOpInterface");

    auto it = llvm::find(op->getSuccessors(), block);
    assert(it != op->getSuccessors().end() && "could find successor");
    int64_t successorIdx = std::distance(op->getSuccessors().begin(), it);

    SuccessorOperands operands = branchOp.getSuccessorOperands(successorIdx);
    SmallVector<Value> newOperands;
    for (auto [operand, type] :
         llvm::zip(operands.getForwardedOperands(), newTypes)) {
      if (operand.getType() == type) {
        // Not a tensor type. Nothing to do for this operand.
        newOperands.push_back(operand);
        continue;
      }
      FailureOr<BaseMemRefType> operandBufferType =
          bufferization::getBufferType(operand, options);
      if (failed(operandBufferType))
        return failure();
      rewriter.setInsertionPointAfterValue(operand);
      Value bufferizedOperand = rewriter.create<bufferization::ToBufferOp>(
          operand.getLoc(), *operandBufferType, operand);
      // A cast is needed if the operand and the block argument have different
      // bufferized types.
      if (type != *operandBufferType)
        bufferizedOperand = rewriter.create<memref::CastOp>(
            operand.getLoc(), type, bufferizedOperand);
      newOperands.push_back(bufferizedOperand);
    }
    operands.getMutableForwardedOperands().assign(newOperands);
  }

  return success();
}
