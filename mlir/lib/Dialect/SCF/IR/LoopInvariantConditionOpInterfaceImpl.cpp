//===- LoopInvariantConditionOpInterfaceImpl.cpp - Impl. for scf.if -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/LoopInvariantConditionOpInterfaceImpl.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/LoopInvariantConditionInterface.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

using namespace mlir;

namespace {

using CmpOp = ValueBoundsConstraintSet::ComparisonOperator;

static ValueBoundsConstraintSet::Variable zeroVariable(MLIRContext *ctx) {
  Builder b(ctx);
  return ValueBoundsConstraintSet::Variable(
      AffineMap::get(0, 0, b.getAffineConstantExpr(0)), ValueRange{});
}

/// Returns true if value is provably non-negative. Used to check whether
/// an unsigned comparison can be safely reduced to its signed counterpart
static bool isProvablyNonNegative(Value value) {
  return ValueBoundsConstraintSet::compare(
      ValueBoundsConstraintSet::Variable(value), CmpOp::GE,
      zeroVariable(value.getContext()));
}

/// Returns true if cmpOp's relation is provably
/// always true.
/// Conservative: returns false whenever the relation cannot be proven.
static bool cmpAlwaysHolds(arith::CmpIOp cmpOp) {
  auto lhs = ValueBoundsConstraintSet::Variable(cmpOp.getLhs());
  auto rhs = ValueBoundsConstraintSet::Variable(cmpOp.getRhs());

  switch (cmpOp.getPredicate()) {
  case arith::CmpIPredicate::eq:
    return ValueBoundsConstraintSet::compare(lhs, CmpOp::EQ, rhs);
  case arith::CmpIPredicate::slt:
    return ValueBoundsConstraintSet::compare(lhs, CmpOp::LT, rhs);
  case arith::CmpIPredicate::sle:
    return ValueBoundsConstraintSet::compare(lhs, CmpOp::LE, rhs);
  case arith::CmpIPredicate::sgt:
    return ValueBoundsConstraintSet::compare(lhs, CmpOp::GT, rhs);
  case arith::CmpIPredicate::sge:
    return ValueBoundsConstraintSet::compare(lhs, CmpOp::GE, rhs);
  case arith::CmpIPredicate::ne:
    return ValueBoundsConstraintSet::compare(lhs, CmpOp::LT, rhs) ||
           ValueBoundsConstraintSet::compare(lhs, CmpOp::GT, rhs);
  case arith::CmpIPredicate::ult:
  case arith::CmpIPredicate::ule:
  case arith::CmpIPredicate::ugt:
  case arith::CmpIPredicate::uge:
    // Only safe to reduce to the signed operator once both operands are
    // proven non-negative
    if (!isProvablyNonNegative(cmpOp.getLhs()) ||
        !isProvablyNonNegative(cmpOp.getRhs()))
      return false;
    switch (cmpOp.getPredicate()) {
    case arith::CmpIPredicate::ult:
      return ValueBoundsConstraintSet::compare(lhs, CmpOp::LT, rhs);
    case arith::CmpIPredicate::ule:
      return ValueBoundsConstraintSet::compare(lhs, CmpOp::LE, rhs);
    case arith::CmpIPredicate::ugt:
      return ValueBoundsConstraintSet::compare(lhs, CmpOp::GT, rhs);
    case arith::CmpIPredicate::uge:
      return ValueBoundsConstraintSet::compare(lhs, CmpOp::GE, rhs);
    default:
      llvm_unreachable("unsigned predicate switch is exhaustive");
    }
  }
  llvm_unreachable("CmpIPredicate switch is exhaustive");
}

/// External model implementation of LoopInvariantConditionOpInterface for
/// scf.if
struct SCFIfOpLoopInvariantConditionModel
    : public LoopInvariantConditionOpInterface::ExternalModel<
          SCFIfOpLoopInvariantConditionModel, scf::IfOp> {
  bool isRegionAlwaysEnteredInLoop(Operation *op, Region &region,
                                   LoopLikeOpInterface enclosingLoop) const {
    auto ifOp = cast<scf::IfOp>(op);
    if (&region != &ifOp.getThenRegion())
      return false;
    if (ifOp.getNumResults() != 0)
      return false; // TODO: support ifOp already having results.

    Value cond = ifOp.getCondition();

    BoolAttr cst;
    if (matchPattern(cond, m_Constant(&cst)))
      return cst.getValue();

    if (auto cmpOp = cond.getDefiningOp<arith::CmpIOp>())
      return cmpAlwaysHolds(cmpOp);

    return false;
  }
};

} // namespace

void mlir::scf::registerLoopInvariantConditionOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, scf::SCFDialect *dialect) {
    scf::IfOp::attachInterface<SCFIfOpLoopInvariantConditionModel>(*ctx);
  });
}