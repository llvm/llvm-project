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

#include "PassDetail.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-generalization"

using namespace mlir;
using namespace mlir::linalg;

LogicalResult mlir::linalg::generalizeNamedOpPrecondition(Operation *op) {
  LinalgOp namedOp = dyn_cast<LinalgOp>(op);
  // Check if the operation is a LinalgOp but not a GenericOp.
  if (!namedOp || isa<GenericOp>(op))
    return failure();
  // Check if the operation has a region builder.
  if (!namedOp.getRegionBuilder())
    return failure();
  return success();
}

GenericOp mlir::linalg::generalizeNamedOp(PatternRewriter &rewriter,
                                          LinalgOp namedOp) {
  SmallVector<Value> inputOperands = namedOp.getInputOperands();
  SmallVector<Value> outputOperands = namedOp.getOutputOperands();
  SmallVector<AffineMap> indexingMaps = namedOp.getIndexingMaps();
  SmallVector<StringRef> iterators = llvm::to_vector<4>(
      namedOp.iterator_types().getAsValueRange<StringAttr>());
  SmallVector<RankedTensorType> resultTypes = namedOp.getOutputTensorTypes();
  SmallVector<Type> types(resultTypes.begin(), resultTypes.end());

  // All named ops have a region attached that can be inlined.
  assert(namedOp->getNumRegions() == 1 &&
         "expect named op to have one region attached");
  GenericOp genericOp =
      rewriter.create<GenericOp>(namedOp.getLoc(), types, inputOperands,
                                 outputOperands, indexingMaps, iterators);
  rewriter.inlineRegionBefore(namedOp->getRegion(0), genericOp.region(),
                              genericOp.region().begin());
  return genericOp;
}

namespace {

struct LinalgGeneralizationPass
    : public LinalgGeneralizationBase<LinalgGeneralizationPass> {
  void runOnFunction() override;
};

} // namespace

void LinalgGeneralizationPass::runOnFunction() {
  FuncOp func = getFunction();
  RewritePatternSet patterns(&getContext());
  populateLinalgNamedOpsGeneralizationPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(func.getBody(), std::move(patterns));
}

void mlir::linalg::populateLinalgNamedOpsGeneralizationPatterns(
    RewritePatternSet &patterns, const LinalgTransformationFilter &marker) {
  patterns.add<LinalgGeneralizationPattern>(patterns.getContext(), marker);
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createLinalgGeneralizationPass() {
  return std::make_unique<LinalgGeneralizationPass>();
}
