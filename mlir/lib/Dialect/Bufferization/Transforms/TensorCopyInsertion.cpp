//===- TensorCopyInsertion.cpp - Resolve Bufferization Conflicts w/ Copies ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Transforms/TensorCopyInsertion.h"

#include "PassDetail.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::bufferization;

/// Resolve all operands that are also used inside of repetitive regions of the
/// same op. Such cases are not fully supported by One-Shot Bufferize.
///
/// E.g.:
/// %r = scf.for ... iter_args(%t = %tensor) -> tensor<?xf32> {
///   "some_use"(%tensor)
///   ...
/// }
///
/// Is converted to:
/// %tensor_copy = bufferization.alloc_tensor copy(%tensor)
/// %r = scf.for ... iter_args(%t = %tensor) -> tensor<?xf32> {
///   "some_use"(%tensor_copy)
///   ...
/// }
static void
resolveUsesInRepetitiveRegions(Operation *op,
                               const BufferizationOptions &options) {
  IRRewriter rewriter(op->getContext());
  AnalysisState state(options);

  // Look for repetitive ops (loops).
  op->walk([&](RegionBranchOpInterface regionBranchOp) {
    // Skip non-bufferizable ops.
    auto bufferizableOp = options.dynCastBufferizableOp(regionBranchOp);
    if (!bufferizableOp)
      return WalkResult::advance();

    // Find all operands that are also used inside of a repetitve region of this
    // op.
    for (OpOperand &opOperand : regionBranchOp->getOpOperands()) {
      Value operand = opOperand.get();
      // Skip non-tensor operands.
      if (!operand.getType().isa<TensorType>())
        continue;
      // Skip operands that do not bufferize to memory writes.
      if (!bufferizableOp.bufferizesToMemoryWrite(opOperand, state))
        continue;

      // Gather all uses inside repetitive regions.
      SmallVector<OpOperand *> usesInsideRegion;
      for (OpOperand &use : operand.getUses()) {
        Operation *owner = use.getOwner();
        if (!regionBranchOp->isProperAncestor(owner))
          continue;
        for (Region &r : regionBranchOp->getRegions()) {
          if (r.findAncestorOpInRegion(*owner) &&
              regionBranchOp.isRepetitiveRegion(r.getRegionNumber())) {
            usesInsideRegion.push_back(&use);
            break;
          }
        }
      }
      // Nothing to do if the operand is not used inside a repetitive region.
      if (usesInsideRegion.empty())
        continue;

      // Insert a tensor copy and replace all uses inside of repetitive regions.
      rewriter.setInsertionPoint(regionBranchOp);
      auto tensorCopy = rewriter.create<AllocTensorOp>(
          regionBranchOp->getLoc(), operand.getType().cast<TensorType>(),
          /*dynamicSizes=*/ValueRange(),
          /*copy=*/operand, /*memory_space=*/IntegerAttr());
      for (OpOperand *use : usesInsideRegion)
        use->set(tensorCopy);
    }

    return WalkResult::advance();
  });
}

LogicalResult mlir::bufferization::insertTensorCopies(
    Operation *op, const OneShotBufferizationOptions &options) {
  // Preprocessing: Resolve currently unsupported bufferization cases.
  resolveUsesInRepetitiveRegions(op, options);

  OneShotAnalysisState state(op, options);
  // Run normal One-Shot Bufferize analysis or One-Shot Module Bufferize
  // analysis depending on whether function boundary bufferization is enabled or
  // not.
  if (options.bufferizeFunctionBoundaries) {
    if (failed(analyzeModuleOp(cast<ModuleOp>(op), state)))
      return failure();
  } else {
    if (failed(analyzeOp(op, state)))
      return failure();
  }

  if (options.testAnalysisOnly)
    return success();

  return insertTensorCopies(op, state);
}

LogicalResult
mlir::bufferization::insertTensorCopies(Operation *op,
                                        const AnalysisState &state) {
  IRRewriter rewriter(op->getContext());
  StringRef escapeAttrName = BufferizationDialect::kEscapeAttrName;

  WalkResult result = op->walk([&](Operation *op) {
    auto bufferizableOp = state.getOptions().dynCastBufferizableOp(op);
    if (!bufferizableOp)
      return WalkResult::skip();

    // Find allocations without an `escape` attribute and add the attribute
    // based on analysis results.
    if (!op->hasAttr(escapeAttrName)) {
      SmallVector<bool> escapeAttrValue;
      bool foundTensorResult = false;
      for (OpResult opResult : op->getOpResults()) {
        if (!opResult.getType().isa<TensorType>() ||
            !bufferizableOp.bufferizesToAllocation(opResult)) {
          escapeAttrValue.push_back(false);
          continue;
        }
        foundTensorResult = true;
        bool escape = !state.getOptions().createDeallocs ||
                      state.isTensorYielded(opResult);
        escapeAttrValue.push_back(escape);
      }
      if (foundTensorResult)
        op->setAttr(escapeAttrName, rewriter.getBoolArrayAttr(escapeAttrValue));
    }

    // Find inplacability conflicts and resolve them. (Typically with explicit
    // tensor copies in the form of AllocTensorOps.)
    rewriter.setInsertionPoint(op);
    if (failed(bufferizableOp.resolveConflicts(rewriter, state)))
      return WalkResult::interrupt();

    return WalkResult::advance();
  });

  return failure(result.wasInterrupted());
}

namespace {
struct TensorCopyInsertionPass
    : TensorCopyInsertionBase<TensorCopyInsertionPass> {
  TensorCopyInsertionPass() : options(llvm::None) {}
  TensorCopyInsertionPass(const OneShotBufferizationOptions &options)
      : options(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect>();
  }

  void runOnOperation() override {
    if (options) {
      if (failed(insertTensorCopies(getOperation(), *options)))
        signalPassFailure();
    } else {
      OneShotBufferizationOptions options;
      options.allowReturnAllocs = allowReturnAllocs;
      options.bufferizeFunctionBoundaries = bufferizeFunctionBoundaries;
      options.createDeallocs = createDeallocs;
      if (mustInferMemorySpace)
        options.defaultMemorySpace = None;
      if (failed(insertTensorCopies(getOperation(), options)))
        signalPassFailure();
    }
  }

private:
  Optional<OneShotBufferizationOptions> options;
};
} // namespace

std::unique_ptr<Pass> mlir::bufferization::createTensorCopyInsertionPass() {
  return std::make_unique<TensorCopyInsertionPass>();
}

std::unique_ptr<Pass> mlir::bufferization::createTensorCopyInsertionPass(
    const OneShotBufferizationOptions &options) {
  return std::make_unique<TensorCopyInsertionPass>(options);
}
