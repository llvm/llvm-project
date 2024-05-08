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
    cstr.bound(value) == lhs *rhs;
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
    if (cstr.compare(/*lhs=*/{trueValue, dim},
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
    if (cstr.compare(/*lhs=*/{falseValue, dim},
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
} // namespace
} // namespace arith
} // namespace mlir

void mlir::arith::registerValueBoundsOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, arith::ArithDialect *dialect) {
    arith::AddIOp::attachInterface<arith::AddIOpInterface>(*ctx);
    arith::ConstantOp::attachInterface<arith::ConstantOpInterface>(*ctx);
    arith::SubIOp::attachInterface<arith::SubIOpInterface>(*ctx);
    arith::MulIOp::attachInterface<arith::MulIOpInterface>(*ctx);
    arith::SelectOp::attachInterface<arith::SelectOpInterface>(*ctx);
  });
}
