//===- ReconcileUnrealizedCasts.cpp - Eliminate noop unrealized casts -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DEF_RECONCILEUNREALIZEDCASTS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

/// Pass to simplify and eliminate unrealized conversion casts.
///
/// This pass processes unrealized_conversion_cast ops in a worklist-driven
/// fashion. For each matched cast op, if the chain of input casts eventually
/// reaches a cast op where the input types match the output types of the
/// matched op, replace the matched op with the inputs.
///
/// Example:
/// %1 = unrealized_conversion_cast %0 : !A to !B
/// %2 = unrealized_conversion_cast %1 : !B to !C
/// %3 = unrealized_conversion_cast %2 : !C to !A
///
/// In the above example, %0 can be used instead of %3 and all cast ops are
/// folded away.
struct ReconcileUnrealizedCasts
    : public impl::ReconcileUnrealizedCastsBase<ReconcileUnrealizedCasts> {
  ReconcileUnrealizedCasts() = default;

  void runOnOperation() override {
    // Gather all unrealized_conversion_cast ops.
    SetVector<UnrealizedConversionCastOp> worklist;
    getOperation()->walk(
        [&](UnrealizedConversionCastOp castOp) { worklist.insert(castOp); });

    // Helper function that adds all operands to the worklist that are an
    // unrealized_conversion_cast op result.
    auto enqueueOperands = [&](UnrealizedConversionCastOp castOp) {
      for (Value v : castOp.getInputs())
        if (auto inputCastOp = v.getDefiningOp<UnrealizedConversionCastOp>())
          worklist.insert(inputCastOp);
    };

    // Helper function that return the unrealized_conversion_cast op that
    // defines all inputs of the given op (in the same order). Return "nullptr"
    // if there is no such op.
    auto getInputCast =
        [](UnrealizedConversionCastOp castOp) -> UnrealizedConversionCastOp {
      if (castOp.getInputs().empty())
        return {};
      auto inputCastOp = castOp.getInputs()
                             .front()
                             .getDefiningOp<UnrealizedConversionCastOp>();
      if (!inputCastOp)
        return {};
      if (inputCastOp.getOutputs() != castOp.getInputs())
        return {};
      return inputCastOp;
    };

    // Process ops in the worklist bottom-to-top.
    while (!worklist.empty()) {
      UnrealizedConversionCastOp castOp = worklist.pop_back_val();
      if (castOp->use_empty()) {
        // DCE: If the op has no users, erase it. Add the operands to the
        // worklist to find additional DCE opportunities.
        enqueueOperands(castOp);
        castOp->erase();
        continue;
      }

      // Traverse the chain of input cast ops to see if an op with the same
      // input types can be found.
      UnrealizedConversionCastOp nextCast = castOp;
      while (nextCast) {
        if (nextCast.getInputs().getTypes() == castOp.getResultTypes()) {
          // Found a cast where the input types match the output types of the
          // matched op. We can directly use those inputs and the matched op can
          // be removed.
          enqueueOperands(castOp);
          castOp.replaceAllUsesWith(nextCast.getInputs());
          castOp->erase();
          break;
        }
        nextCast = getInputCast(nextCast);
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createReconcileUnrealizedCastsPass() {
  return std::make_unique<ReconcileUnrealizedCasts>();
}
