//===- Generalization.cpp - linalg named ops to generic ops  --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Linalg generalization pass. It converts named
// Linalg ops to linalg.generic ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_LINALGGENERALIZATION
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "linalg-generalization"

using namespace mlir;
using namespace mlir::linalg;

static LogicalResult generalizeNamedOpPrecondition(LinalgOp linalgOp) {
  // Check if the operation is a LinalgOp but not a GenericOp.
  if (isa<GenericOp>(linalgOp))
    return failure();
  // Check if the operation has exactly one region.
  if (linalgOp->getNumRegions() != 1) {
    assert(linalgOp->getNumRegions() == 0 && "op with multiple regions");
    // TOD: Otherwise it needs to be built explicitly from the region builder.
    return failure();
  }
  return success();
}

FailureOr<GenericOp> mlir::linalg::generalizeNamedOp(RewriterBase &rewriter,
                                                     LinalgOp linalgOp) {
  if (failed(generalizeNamedOpPrecondition(linalgOp)))
    return rewriter.notifyMatchFailure(linalgOp, "preconditions not met");

  SmallVector<Value> inputs = linalgOp.getDpsInputOperands();
  SmallVector<Value> outputs = linalgOp.getDpsInitOperands();
  SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();
  SmallVector<utils::IteratorType> iterators = linalgOp.getIteratorTypesArray();
  SmallVector<Type> resultTypes = linalgOp.hasTensorSemantics()
                                      ? TypeRange(ValueRange(outputs))
                                      : TypeRange{};

  // All named ops have a region attached that can be inlined.
  assert(linalgOp->getNumRegions() == 1 &&
         "expect named op to have one region attached");
  GenericOp genericOp = rewriter.create<GenericOp>(
      linalgOp.getLoc(), resultTypes, inputs, outputs, indexingMaps, iterators);
  rewriter.inlineRegionBefore(linalgOp->getRegion(0), genericOp.getRegion(),
                              genericOp.getRegion().begin());
  rewriter.replaceOp(linalgOp, genericOp->getResults());
  return genericOp;
}

namespace {

struct LinalgGeneralizationPass
    : public impl::LinalgGeneralizationBase<LinalgGeneralizationPass> {
  void runOnOperation() override;
};

} // namespace

void LinalgGeneralizationPass::runOnOperation() {
  func::FuncOp func = getOperation();
  RewritePatternSet patterns(&getContext());
  populateLinalgNamedOpsGeneralizationPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(func.getBody(), std::move(patterns));
}

void mlir::linalg::populateLinalgNamedOpsGeneralizationPatterns(
    RewritePatternSet &patterns) {
  patterns.add<LinalgGeneralizationPattern>(patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLinalgGeneralizationPass() {
  return std::make_unique<LinalgGeneralizationPass>();
}
