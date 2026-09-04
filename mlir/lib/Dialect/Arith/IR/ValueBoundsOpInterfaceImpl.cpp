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

    bool lhsNonNegative =
        ValueBoundsConstraintSet::isProvablyNonNegative(divOp.getLhs(), cstr);
    bool rhsPositive =
        ValueBoundsConstraintSet::isProvablyPositive(divOp.getRhs(), cstr);
    if (!lhsNonNegative || !rhsPositive)
      return;

    AffineExpr lhs = cstr.getExpr(divOp.getLhs());
    AffineExpr rhs = cstr.getExpr(divOp.getRhs());
    cstr.bound(value) >= 0;
    cstr.bound(value) == lhs.floorDiv(rhs);
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

    bool lhsNonNegative =
        ValueBoundsConstraintSet::isProvablyNonNegative(lhsValue, cstr);
    bool lhsNonPositive =
        ValueBoundsConstraintSet::isProvablyNonPositive(lhsValue, cstr);
    bool rhsPositive =
        ValueBoundsConstraintSet::isProvablyPositive(rhsValue, cstr);
    bool rhsNegative =
        ValueBoundsConstraintSet::isProvablyNegative(rhsValue, cstr);

    AffineExpr lhs = cstr.getExpr(lhsValue);
    AffineExpr rhs = cstr.getExpr(rhsValue);

    // divsi rounds toward zero, unlike floorDiv/ceilDiv which round toward
    // negative/positive infinity respectively. When the result is non-negative,
    // divsi equals floorDiv(lhs, rhs); when negative, it equals ceilDiv(lhs,
    // rhs). Without knowing the sign, bound the result between those two
    // expressions, which is always correct.
    cstr.bound(value) >= lhs.floorDiv(rhs);
    cstr.bound(value) <= lhs.ceilDiv(rhs);

    // If the sign of the result is known, we can use the exact expression.
    if ((lhsNonNegative && rhsPositive) || (lhsNonPositive && rhsNegative)) {
      cstr.bound(value) == lhs.floorDiv(rhs);
      cstr.bound(value) >= 0;
    } else if ((lhsNonPositive && rhsPositive) ||
               (lhsNonNegative && rhsNegative)) {
      cstr.bound(value) == lhs.ceilDiv(rhs);
      cstr.bound(value) <= 0;
    }
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
    bool rhsPositive =
        ValueBoundsConstraintSet::isProvablyPositive(rhsValue, cstr);
    bool rhsNegative =
        ValueBoundsConstraintSet::isProvablyNegative(rhsValue, cstr);

    // The result of remsi has the same sign as the dividend (lhs) and also
    // fulfills |result| < |rhs|. The sign of lhs does not need to be a
    // compile-time constant: it is sufficient if the constraint set can prove
    // it. For lhs == 0 both branches may fire, which is consistent since the
    // result is then 0. f.e:
    //   lhs   rhs   result   bounds
    //   ----  ----  ------   --------------------------------------------------
    //    7     3      1      0 <= val && val <= rhs-1 = 2      -> [0, 2]
    //    7    -3      1      0 <= val && val <= -rhs-1 = 2     -> [0, 2]
    //   -7     3     -1      val <= 0 && val >= 1-rhs = -2     -> [-2, 0]
    //   -7    -3     -1      val <= 0 && val >= rhs+1 = -2     -> [-2, 0]
    //    0     3      0      both lhs branches fire (0<=val and val<=0) -> val
    //    == 0
    if (ValueBoundsConstraintSet::isProvablyNonPositive(lhsValue, cstr)) {
      cstr.bound(value) <= 0;
      if (rhsPositive)
        cstr.bound(value) >= 1 - rhs;
      if (rhsNegative)
        cstr.bound(value) >= rhs + 1;
    }
    if (ValueBoundsConstraintSet::isProvablyNonNegative(lhsValue, cstr)) {
      cstr.bound(value) >= 0;
      if (rhsPositive)
        cstr.bound(value) <= rhs - 1;
      if (rhsNegative)
        cstr.bound(value) <= -rhs - 1;
    }
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

    // remui computes an unsigned remainder, so for a provably positive divisor
    // the result is always in [0, rhs - 1].
    if (ValueBoundsConstraintSet::isProvablyPositive(rhsValue, cstr)) {
      cstr.bound(value) >= 0;
      cstr.bound(value) <= rhs - 1;
    }
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

    // Populate constraints for the true/false values (and all values on the
    // backward slice, as long as the current stop condition is not satisfied).
    cstr.populateConstraints(trueValue, dim);
    cstr.populateConstraints(falseValue, dim);
    auto boundsBuilder = cstr.bound(value);
    if (dim)
      boundsBuilder[*dim];

    // Compare yielded values.
    // If trueValue <= falseValue:
    // * result <= falseValue
    // * result >= trueValue
    if (cstr.populateAndCompare(
            /*lhs=*/{trueValue, dim},
            ValueBoundsConstraintSet::ComparisonOperator::LE,
            /*rhs=*/{falseValue, dim})) {
      if (dim) {
        cstr.bound(value)[*dim] >= cstr.getExpr(trueValue, dim);
        cstr.bound(value)[*dim] <= cstr.getExpr(falseValue, dim);
      } else {
        cstr.bound(value) >= trueValue;
        cstr.bound(value) <= falseValue;
      }
    }
    // If falseValue <= trueValue:
    // * result <= trueValue
    // * result >= falseValue
    if (cstr.populateAndCompare(
            /*lhs=*/{falseValue, dim},
            ValueBoundsConstraintSet::ComparisonOperator::LE,
            /*rhs=*/{trueValue, dim})) {
      if (dim) {
        cstr.bound(value)[*dim] >= cstr.getExpr(falseValue, dim);
        cstr.bound(value)[*dim] <= cstr.getExpr(trueValue, dim);
      } else {
        cstr.bound(value) >= falseValue;
        cstr.bound(value) <= trueValue;
      }
    }
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
    bool lhsNonNegative =
        ValueBoundsConstraintSet::isProvablyNonNegative(minOp.getLhs(), cstr);
    bool rhsNonNegative =
        ValueBoundsConstraintSet::isProvablyNonNegative(minOp.getRhs(), cstr);
    if (!lhsNonNegative && !rhsNonNegative)
      return;

    // A negative signed integer bit pattern reinterpreted as an
    // unsigned integer is greater than SIGNED_INT_MAX. If
    // one of the operands is signed non-negative, it is smaller than
    // or equal to SIGNED_INT_MAX in unsigned interpretation,
    // and `minui` will choose that operand over a negative signed
    // integer operand.
    cstr.bound(value) >= 0;
    // If an operand is provably non-negative, its signed and unsigned value
    // interpretations agree, so `minsi` and `minui` impose the same upper
    // bound: `result <= operand`.
    if (lhsNonNegative) {
      AffineExpr lhs = cstr.getExpr(minOp.getLhs());
      cstr.bound(value) <= lhs;
    }
    if (rhsNonNegative) {
      AffineExpr rhs = cstr.getExpr(minOp.getRhs());
      cstr.bound(value) <= rhs;
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

    // See MinUIOpInterface comment
    bool lhsNonNegative =
        ValueBoundsConstraintSet::isProvablyNonNegative(maxOp.getLhs(), cstr);
    bool rhsNonNegative =
        ValueBoundsConstraintSet::isProvablyNonNegative(maxOp.getRhs(), cstr);

    if (lhsNonNegative) {
      AffineExpr lhs = cstr.getExpr(maxOp.getLhs());
      cstr.bound(value) >= lhs;
    }
    if (rhsNonNegative) {
      AffineExpr rhs = cstr.getExpr(maxOp.getRhs());
      cstr.bound(value) >= rhs;
    }
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
