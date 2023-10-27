//===- ReconcileUnrealizedCasts.cpp - Eliminate noop unrealized casts -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_RECONCILEUNREALIZEDCASTS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

/// Folds the DAGs of `unrealized_conversion_cast`s that have as exit types
/// the same as the input ones.
/// For example, the DAGs `A -> B -> C -> B -> A` and `A -> B -> C -> A`
/// represent a noop within the IR, and thus the initial input values can be
/// propagated.
/// The same does not hold for 'open' chains of casts, such as
/// `A -> B -> C`. In this last case there is no cycle among the types and thus
/// the conversion is incomplete. The same hold for 'closed' chains like
/// `A -> B -> A`, but with the result of type `B` being used by some non-cast
/// operations.
/// Bifurcations (that is when a chain starts in between of another one) are
/// also taken into considerations, and all the above considerations remain
/// valid.
/// Special corner cases such as dead casts or single casts with same input and
/// output types are also covered.
struct UnrealizedConversionCastPassthrough
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  using OpRewritePattern<UnrealizedConversionCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    // The nodes that either are not used by any operation or have at least
    // one user that is not an unrealized cast.
    DenseSet<UnrealizedConversionCastOp> exitNodes;

    // The nodes whose users are all unrealized casts
    DenseSet<UnrealizedConversionCastOp> intermediateNodes;

    // Stack used for the depth-first traversal of the use-def DAG.
    SmallVector<UnrealizedConversionCastOp, 2> visitStack;
    visitStack.push_back(op);

    while (!visitStack.empty()) {
      UnrealizedConversionCastOp current = visitStack.pop_back_val();
      auto users = current->getUsers();
      bool isLive = false;

      for (Operation *user : users) {
        if (auto other = dyn_cast<UnrealizedConversionCastOp>(user)) {
          if (other.getInputs() != current.getOutputs())
            return rewriter.notifyMatchFailure(
                op, "mismatching values propagation");
        } else {
          isLive = true;
        }

        // Continue traversing the DAG of unrealized casts
        if (auto other = dyn_cast<UnrealizedConversionCastOp>(user))
          visitStack.push_back(other);
      }

      // If the cast is live, then we need to check if the results of the last
      // cast have the same type of the root inputs. It this is the case (e.g.
      // `{A -> B, B -> A}`, but also `{A -> A}`), then the cycle is just a
      // no-op and the inputs can be forwarded. If it's not (e.g.
      // `{A -> B, B -> C}`, `{A -> B}`), then the cast chain is incomplete.

      bool isCycle = current.getResultTypes() == op.getInputs().getTypes();

      if (isLive && !isCycle)
        return rewriter.notifyMatchFailure(op,
                                           "live unrealized conversion cast");

      bool isExitNode = users.empty() || isLive;

      if (isExitNode) {
        exitNodes.insert(current);
      } else {
        intermediateNodes.insert(current);
      }
    }

    // Replace the sink nodes with the root input values
    for (UnrealizedConversionCastOp exitNode : exitNodes)
      rewriter.replaceOp(exitNode, op.getInputs());

    // Erase all the other casts belonging to the DAG
    for (UnrealizedConversionCastOp castOp : intermediateNodes)
      rewriter.eraseOp(castOp);

    return success();
  }
};

/// Pass to simplify and eliminate unrealized conversion casts.
struct ReconcileUnrealizedCasts
    : public impl::ReconcileUnrealizedCastsBase<ReconcileUnrealizedCasts> {
  ReconcileUnrealizedCasts() = default;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateReconcileUnrealizedCastsPatterns(patterns);
    ConversionTarget target(getContext());
    target.addIllegalOp<UnrealizedConversionCastOp>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::populateReconcileUnrealizedCastsPatterns(
    RewritePatternSet &patterns) {
  patterns.add<UnrealizedConversionCastPassthrough>(patterns.getContext());
}

std::unique_ptr<Pass> mlir::createReconcileUnrealizedCastsPass() {
  return std::make_unique<ReconcileUnrealizedCasts>();
}
