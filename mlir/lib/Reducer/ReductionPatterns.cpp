//===- ReductionPatterns.cpp - MLIR Reducer Patterns ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Reducer/ReductionPatternInterface.h"

using namespace mlir;

namespace {
/// A generic reduction pattern that replaces any operation returning a
/// VectorType with a ub.poison value of the same type.
struct GenericVectorPoisonReduction : public RewritePattern {
  GenericVectorPoisonReduction(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Prevent infinite loops by ignoring operations that are already poison.
    if (isa<ub::PoisonOp>(op))
      return failure();

    // Check if the operation has at least one vector result.
    bool hasVectorResult = llvm::any_of(
        op->getResultTypes(), [](Type t) { return isa<VectorType>(t); });

    if (!hasVectorResult)
      return failure();

    SmallVector<Value> replacements;
    for (auto [idx, type] : llvm::enumerate(op->getResultTypes())) {
      if (isa<VectorType>(type))
        replacements.push_back(
            ub::PoisonOp::create(rewriter, op->getLoc(), type));
      else
        replacements.push_back(
            op->getResult(idx)); // Preserve non-vector results.
    }

    rewriter.replaceOp(op, replacements);
    return success();
  }
};

/// Dialect interface to attach the reduction pattern to the Vector dialect.
struct VectorReductionInterface : public DialectReductionPatternInterface {
  VectorReductionInterface(Dialect *dialect)
      : DialectReductionPatternInterface(dialect) {}

  void populateReductionPatterns(RewritePatternSet &patterns) const override {
    patterns.add<GenericVectorPoisonReduction>(patterns.getContext());
  }
};
} // end anonymous namespace

namespace mlir {
void registerReducerExtension(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, vector::VectorDialect *dialect) {
    dialect->addInterfaces<VectorReductionInterface>();
  });
}
} // namespace mlir