//===- ValueBoundsOpInterfaceImpl.cpp - Impl. of ValueBoundsOpInterface ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

using namespace mlir;

namespace mlir {
namespace arith {
namespace {
using Cmp = ValueBoundsConstraintSet::ComparisonOperator;

struct ConstantOpInterface
    : public ValueBoundsOpInterface::ExternalModel<ConstantOpInterface,
                                                   ConstantOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto constantOp = cast<ConstantOp>(op);
    assert(value == constantOp.getResult() && "invalid value");

    if (auto attr = llvm::dyn_cast<IntegerAttr>(constantOp.getValue()))
      cstr.bound(value) == attr.getInt();
  }
};

struct ExtSIOpInterface
    : public ValueBoundsOpInterface::ExternalModel<ExtSIOpInterface, ExtSIOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto extSIOp = cast<ExtSIOp>(op);
    assert(value == extSIOp.getOut() && "invalid value");

    // Sign extension preserves the signed value (unlike zero extension where
    // the result may be negative), so the bound is an exact equality.
    cstr.bound(value) == cstr.getExpr(extSIOp.getIn());
  }
};

struct AddIOpInterface
    : public ValueBoundsOpInterface::ExternalModel<AddIOpInterface, AddIOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto addIOp = cast<AddIOp>(op);
    assert(value == addIOp.getResult() && "invalid value");

    // Note: `getExpr` has a side effect: it may add a new column to the
    // constraint system. The evaluation order of addition operands is
    // unspecified in C++. To make sure that all compilers produce the exact
    // same results (that can be FileCheck'd), it is important that `getExpr`
    // is called first and assigned to temporary variables, and the addition
    // is performed afterwards.
    AffineExpr lhs = cstr.getExpr(addIOp.getLhs());
    AffineExpr rhs = cstr.getExpr(addIOp.getRhs());
    cstr.bound(value) == lhs + rhs;
  }
};

struct DivUIOpInterface
    : public ValueBoundsOpInterface::ExternalModel<DivUIOpInterface,
                                                   arith::DivUIOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto divOp = cast<arith::DivUIOp>(op);
    assert(value == divOp.getResult() && "invalid value");

    AffineExpr lhs = cstr.getExpr(divOp.getLhs());
    AffineExpr rhs = cstr.getExpr(divOp.getRhs());
    OpFoldResult zero = Builder(op->getContext()).getIndexAttr(0);
    auto bounds = cstr.bound(value).when(
        {{divOp.getLhs(), Cmp::GE, zero}, {divOp.getRhs(), Cmp::GT, zero}});
    bounds >= 0;
    bounds == lhs.floorDiv(rhs);
  }
};

struct DivSIOpInterface
    : public ValueBoundsOpInterface::ExternalModel<DivSIOpInterface,
                                                   arith::DivSIOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto divOp = cast<arith::DivSIOp>(op);
    assert(value == divOp.getResult() && "invalid value");

    Value lhsValue = divOp.getLhs();
    Value rhsValue = divOp.getRhs();
    AffineExpr lhs = cstr.getExpr(lhsValue);
    AffineExpr rhs = cstr.getExpr(rhsValue);
    OpFoldResult zero = Builder(op->getContext()).getIndexAttr(0);

    // divsi rounds toward zero, unlike floorDiv/ceilDiv which round toward
    // negative/positive infinity respectively. Without knowing the sign, the
    // result is bounded between these expressions.
    cstr.bound(value) >= lhs.floorDiv(rhs);
    cstr.bound(value) <= lhs.ceilDiv(rhs);

    // Register each sign combination separately. At lhs == 0 both signs can
    // be proven, and the resulting equalities consistently describe zero.
    auto addBounds = [&](Cmp lhsSign, Cmp rhsSign, AffineExpr quotient,
                         bool nonNegative) {
      auto bounds = cstr.bound(value).when(
          {{lhsValue, lhsSign, zero}, {rhsValue, rhsSign, zero}});
      bounds == quotient;
      if (nonNegative)
        bounds >= 0;
      else
        bounds <= 0;
    };
    addBounds(Cmp::GE, Cmp::GT, lhs.floorDiv(rhs), true);
    addBounds(Cmp::LE, Cmp::LT, lhs.floorDiv(rhs), true);
    addBounds(Cmp::LE, Cmp::GT, lhs.ceilDiv(rhs), false);
    addBounds(Cmp::GE, Cmp::LT, lhs.ceilDiv(rhs), false);
  }
};

struct SubIOpInterface
    : public ValueBoundsOpInterface::ExternalModel<SubIOpInterface, SubIOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto subIOp = cast<SubIOp>(op);
    assert(value == subIOp.getResult() && "invalid value");

    AffineExpr lhs = cstr.getExpr(subIOp.getLhs());
    AffineExpr rhs = cstr.getExpr(subIOp.getRhs());
    cstr.bound(value) == lhs - rhs;
  }
};

struct MulIOpInterface
    : public ValueBoundsOpInterface::ExternalModel<MulIOpInterface, MulIOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto mulIOp = cast<MulIOp>(op);
    assert(value == mulIOp.getResult() && "invalid value");

    AffineExpr lhs = cstr.getExpr(mulIOp.getLhs());
    AffineExpr rhs = cstr.getExpr(mulIOp.getRhs());
    cstr.bound(value) == (lhs * rhs);
  }
};

struct FloorDivSIOpInterface
    : public ValueBoundsOpInterface::ExternalModel<FloorDivSIOpInterface,
                                                   FloorDivSIOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto divSIOp = cast<FloorDivSIOp>(op);
    assert(value == divSIOp.getResult() && "invalid value");

    AffineExpr lhs = cstr.getExpr(divSIOp.getLhs());
    AffineExpr rhs = cstr.getExpr(divSIOp.getRhs());
    cstr.bound(value) == lhs.floorDiv(rhs);
  }
};

struct CeilDivSIOpInterface
    : public ValueBoundsOpInterface::ExternalModel<CeilDivSIOpInterface,
                                                   CeilDivSIOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto divSIOp = cast<CeilDivSIOp>(op);
    assert(value == divSIOp.getResult() && "invalid value");

    AffineExpr lhs = cstr.getExpr(divSIOp.getLhs());
    AffineExpr rhs = cstr.getExpr(divSIOp.getRhs());
    cstr.bound(value) == lhs.ceilDiv(rhs);
  }
};

struct RemSIOpInterface
    : public ValueBoundsOpInterface::ExternalModel<RemSIOpInterface, RemSIOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto remSIOp = cast<RemSIOp>(op);
    assert(value == remSIOp.getResult() && "invalid value");

    Value lhsValue = remSIOp.getLhs();
    Value rhsValue = remSIOp.getRhs();
    AffineExpr rhs = cstr.getExpr(rhsValue);
    OpFoldResult zero = Builder(op->getContext()).getIndexAttr(0);

    // remsi has the sign of lhs and |result| < |rhs|. At lhs == 0 both
    // dividend conditions hold, consistently bounding the result to zero.
    cstr.bound(value).when({{lhsValue, Cmp::LE, zero}}) <= 0;
    cstr.bound(value).when({{lhsValue, Cmp::GE, zero}}) >= 0;
    cstr.bound(value).when(
        {{lhsValue, Cmp::LE, zero}, {rhsValue, Cmp::GT, zero}}) >= 1 - rhs;
    cstr.bound(value).when(
        {{lhsValue, Cmp::LE, zero}, {rhsValue, Cmp::LT, zero}}) >= rhs + 1;
    cstr.bound(value).when(
        {{lhsValue, Cmp::GE, zero}, {rhsValue, Cmp::GT, zero}}) <= rhs - 1;
    cstr.bound(value).when(
        {{lhsValue, Cmp::GE, zero}, {rhsValue, Cmp::LT, zero}}) <= -rhs - 1;
  }
};

struct RemUIOpInterface
    : public ValueBoundsOpInterface::ExternalModel<RemUIOpInterface, RemUIOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto remUIOp = cast<RemUIOp>(op);
    assert(value == remUIOp.getResult() && "invalid value");

    Value rhsValue = remUIOp.getRhs();
    AffineExpr rhs = cstr.getExpr(rhsValue);
    OpFoldResult zero = Builder(op->getContext()).getIndexAttr(0);
    // For a provably positive divisor, remui is always in [0, rhs - 1].
    auto bounds = cstr.bound(value).when({{rhsValue, Cmp::GT, zero}});
    bounds >= 0;
    bounds <= rhs - 1;
  }
};

struct SelectOpInterface
    : public ValueBoundsOpInterface::ExternalModel<SelectOpInterface,
                                                   SelectOp> {

  static void populateBounds(SelectOp selectOp, std::optional<int64_t> dim,
                             ValueBoundsConstraintSet &cstr) {
    Value value = selectOp.getResult();
    Value condition = selectOp.getCondition();
    Value trueValue = selectOp.getTrueValue();
    Value falseValue = selectOp.getFalseValue();

    if (isa<ShapedType>(condition.getType())) {
      // If the condition is a shaped type, the condition is applied
      // element-wise. All three operands must have the same shape.
      cstr.bound(value)[*dim] == cstr.getExpr(trueValue, dim);
      cstr.bound(value)[*dim] == cstr.getExpr(falseValue, dim);
      cstr.bound(value)[*dim] == cstr.getExpr(condition, dim);
      return;
    }

    // Each ordering is a sufficient condition for an interval containing both
    // possible results. Registration does not query or recursively analyze IR.
    auto addBounds = [&](Value lower, Value upper) {
      auto bounds = cstr.bound(value);
      if (dim)
        bounds[*dim];
      auto conditional = bounds.when({{{lower, dim}, Cmp::LE, {upper, dim}}});
      conditional >= cstr.getExpr(lower, dim);
      conditional <= cstr.getExpr(upper, dim);
    };
    addBounds(trueValue, falseValue);
    addBounds(falseValue, trueValue);
  }

  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    populateBounds(cast<SelectOp>(op), /*dim=*/std::nullopt, cstr);
  }

  void populateBoundsForShapedValueDim(Operation *op, Value value, int64_t dim,
                                       ValueBoundsConstraintSet &cstr) const {
    populateBounds(cast<SelectOp>(op), dim, cstr);
  }
};

struct MinSIOpInterface
    : public ValueBoundsOpInterface::ExternalModel<MinSIOpInterface, MinSIOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto minOp = cast<MinSIOp>(op);
    assert(value == minOp.getResult() && "invalid value");

    AffineExpr lhs = cstr.getExpr(minOp.getLhs());
    AffineExpr rhs = cstr.getExpr(minOp.getRhs());
    cstr.bound(value) <= lhs;
    cstr.bound(value) <= rhs;
  }
};

struct MaxSIOpInterface
    : public ValueBoundsOpInterface::ExternalModel<MaxSIOpInterface, MaxSIOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto maxOp = cast<MaxSIOp>(op);
    assert(value == maxOp.getResult() && "invalid value");

    AffineExpr lhs = cstr.getExpr(maxOp.getLhs());
    AffineExpr rhs = cstr.getExpr(maxOp.getRhs());
    cstr.bound(value) >= lhs;
    cstr.bound(value) >= rhs;
  }
};

struct MinUIOpInterface
    : public ValueBoundsOpInterface::ExternalModel<MinUIOpInterface,
                                                   arith::MinUIOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto minOp = cast<arith::MinUIOp>(op);
    assert(value == minOp.getResult() && "invalid value");

    // ValueBoundsConstraintSet models values as signed integers (e.g. an i8
    // 0xff is treated as -1, not 255). For an unsigned minimum it is enough
    // that a single operand is provably non-negative: minui(x, y) is in
    // [0, y] whenever y >= 0 (and symmetrically for x).
    OpFoldResult zero = Builder(op->getContext()).getIndexAttr(0);
    // Either non-negative operand bounds the unsigned minimum from above.
    for (Value operand : {minOp.getLhs(), minOp.getRhs()}) {
      auto bounds = cstr.bound(value).when({{operand, Cmp::GE, zero}});
      bounds >= 0;
      bounds <= cstr.getExpr(operand);
    }
  }
};

struct MaxUIOpInterface
    : public ValueBoundsOpInterface::ExternalModel<MaxUIOpInterface,
                                                   arith::MaxUIOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto maxOp = cast<arith::MaxUIOp>(op);
    assert(value == maxOp.getResult() && "invalid value");

    // See MinUIOpInterface comment.
    OpFoldResult zero = Builder(op->getContext()).getIndexAttr(0);
    for (Value operand : {maxOp.getLhs(), maxOp.getRhs()})
      cstr.bound(value).when({{operand, Cmp::GE, zero}}) >=
          cstr.getExpr(operand);
  }
};
} // namespace
} // namespace arith
} // namespace mlir

void mlir::arith::registerValueBoundsOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, arith::ArithDialect *dialect) {
    arith::ConstantOp::attachInterface<arith::ConstantOpInterface>(*ctx);
    arith::ExtSIOp::attachInterface<arith::ExtSIOpInterface>(*ctx);
    arith::AddIOp::attachInterface<arith::AddIOpInterface>(*ctx);
    arith::DivUIOp::attachInterface<arith::DivUIOpInterface>(*ctx);
    arith::DivSIOp::attachInterface<arith::DivSIOpInterface>(*ctx);
    arith::SubIOp::attachInterface<arith::SubIOpInterface>(*ctx);
    arith::MulIOp::attachInterface<arith::MulIOpInterface>(*ctx);
    arith::FloorDivSIOp::attachInterface<arith::FloorDivSIOpInterface>(*ctx);
    arith::CeilDivSIOp::attachInterface<arith::CeilDivSIOpInterface>(*ctx);
    arith::RemSIOp::attachInterface<arith::RemSIOpInterface>(*ctx);
    arith::RemUIOp::attachInterface<arith::RemUIOpInterface>(*ctx);
    arith::SelectOp::attachInterface<arith::SelectOpInterface>(*ctx);
    arith::MinSIOp::attachInterface<arith::MinSIOpInterface>(*ctx);
    arith::MaxSIOp::attachInterface<arith::MaxSIOpInterface>(*ctx);
    arith::MinUIOp::attachInterface<arith::MinUIOpInterface>(*ctx);
    arith::MaxUIOp::attachInterface<arith::MaxUIOpInterface>(*ctx);
  });
}
