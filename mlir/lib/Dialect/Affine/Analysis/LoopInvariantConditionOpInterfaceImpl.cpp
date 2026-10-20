//===- LoopInvariantConditionOpInterfaceImpl.cpp - Impl. for affine.if ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/LoopInvariantConditionOpInterfaceImpl.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Interfaces/LoopInvariantConditionInterface.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

using namespace mlir;
using namespace mlir::affine;

namespace {

/// Returns true if ifOp's condition is provably true for every value the
/// induction variable of forOp can take.
/// Conservative: returns false whenever the
/// check cannot be fully carried out.
static bool conditionAlwaysHolds(AffineIfOp ifOp, AffineForOp forOp) {

  (void)forOp;

  IntegerSet set = ifOp.getIntegerSet();
  ValueRange operands = ifOp.getOperands();
  Builder b(ifOp.getContext());
  auto zero = ValueBoundsConstraintSet::Variable(
      AffineMap::get(set.getNumDims(), set.getNumSymbols(),
                     b.getAffineConstantExpr(0)),
      operands);

  for (unsigned i = 0, e = set.getNumConstraints(); i < e; ++i) {
    AffineMap constraintMap = AffineMap::get(
        set.getNumDims(), set.getNumSymbols(), set.getConstraint(i));
    auto constraintVar =
        ValueBoundsConstraintSet::Variable(constraintMap, operands);
    auto cmp = set.isEq(i) ? ValueBoundsConstraintSet::EQ
                           : ValueBoundsConstraintSet::GE;
    if (!ValueBoundsConstraintSet::compare(constraintVar, cmp, zero))
      return false;
  }
  return true;
}

/// External model implementation of LoopInvariantConditionOpInterface for
/// AffineIfOp.
struct AffineIfOpLoopInvariantConditionModel
    : public LoopInvariantConditionOpInterface::ExternalModel<
          AffineIfOpLoopInvariantConditionModel, AffineIfOp> {
  bool isRegionAlwaysEnteredInLoop(Operation *op, Region &region,
                                   LoopLikeOpInterface enclosingLoop) const {
    auto ifOp = cast<AffineIfOp>(op);
    if (&region != &ifOp.getThenRegion())
      return false; // Only the then-branch is handled for now.
    if (ifOp.getNumResults() != 0)
      return false; // TODO: support ifOp already having results.
    auto forOp = dyn_cast<AffineForOp>(enclosingLoop.getOperation());
    if (!forOp)
      return false; // Only affine.for is handled; not affine.parallel, etc.
    return conditionAlwaysHolds(ifOp, forOp);
  }
};

} // namespace

void mlir::affine::registerLoopInvariantConditionOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, AffineDialect *dialect) {
    AffineIfOp::attachInterface<AffineIfOpLoopInvariantConditionModel>(*ctx);
  });
}