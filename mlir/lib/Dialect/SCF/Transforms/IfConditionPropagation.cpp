//===- IfConditionPropagation.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass for constant propagation of the condition of an
// `scf.if` into its then and else regions as true and false respectively.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"

using namespace mlir;

namespace mlir {
#define GEN_PASS_DEF_SCFIFCONDITIONPROPAGATION
#include "mlir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace mlir

/// Traverses the IR recursively (on region tree) and updates the uses of a
/// value also as the condition of an `scf.if` to either `true` or `false`
/// constants in the `then` and `else regions. This is done as a single
/// post-order sweep over the IR (without `walk`) for efficiency reasons. While
/// traversing, the function maintains the set of visited regions to quickly
/// identify whether the value belong to a region that is known to be nested in
/// the `then` or `else` branch of a specific loop.
static void propagateIfConditionsImpl(Operation *root,
                                      llvm::SmallPtrSet<Region *, 8> &visited) {
  if (auto scfIf = dyn_cast<scf::IfOp>(root)) {
    llvm::SmallPtrSet<Region *, 8> thenChildren, elseChildren;
    // Visit the "then" region, collect children.
    for (Block &block : scfIf.getThenRegion()) {
      for (Operation &op : block) {
        propagateIfConditionsImpl(&op, thenChildren);
      }
    }

    // Visit the "else" region, collect children.
    for (Block &block : scfIf.getElseRegion()) {
      for (Operation &op : block) {
        propagateIfConditionsImpl(&op, elseChildren);
      }
    }

    // Update uses to point to constants instead.
    OpBuilder builder(scfIf);
    Value trueValue = arith::ConstantIntOp::create(builder, scfIf.getLoc(),
                                                   /*value=*/true, /*width=*/1);
    Value falseValue =
        arith::ConstantIntOp::create(builder, scfIf.getLoc(),
                                     /*value=*/false, /*width=*/1);

    for (OpOperand &use : scfIf.getCondition().getUses()) {
      if (thenChildren.contains(use.getOwner()->getParentRegion()))
        use.set(trueValue);
      else if (elseChildren.contains(use.getOwner()->getParentRegion()))
        use.set(falseValue);
    }
    if (trueValue.getUses().empty())
      trueValue.getDefiningOp()->erase();
    if (falseValue.getUses().empty())
      falseValue.getDefiningOp()->erase();

    // Append the two lists of children and return them.
    visited.insert_range(thenChildren);
    visited.insert_range(elseChildren);
    return;
  }

  for (Region &region : root->getRegions()) {
    for (Block &block : region) {
      for (Operation &op : block) {
        propagateIfConditionsImpl(&op, visited);
      }
    }
  }
}

/// Traverses the IR recursively (on region tree) and updates the uses of a
/// value also as the condition of an `scf.if` to either `true` or `false`
/// constants in the `then` and `else regions
static void propagateIfConditions(Operation *root) {
  llvm::SmallPtrSet<Region *, 8> visited;
  propagateIfConditionsImpl(root, visited);
}

namespace {
/// Pass entrypoint.
struct SCFIfConditionPropagationPass
    : impl::SCFIfConditionPropagationBase<SCFIfConditionPropagationPass> {
  void runOnOperation() override { propagateIfConditions(getOperation()); }
};
} // namespace
