//===- ValueBoundsOpInterfaceImpl.cpp - Impl. of ValueBoundsOpInterface ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

using namespace mlir;

namespace mlir {
namespace scf {
namespace {

struct ForOpInterface
    : public ValueBoundsOpInterface::ExternalModel<ForOpInterface, ForOp> {

  /// Populate bounds of values/dimensions for iter_args/OpResults. If the
  /// value/dimension size does not change in an iteration, we can deduce that
  /// it the same as the initial value/dimension.
  ///
  /// Example 1:
  /// %0 = scf.for ... iter_args(%arg0 = %t) -> tensor<?xf32> {
  ///   ...
  ///   %1 = tensor.insert %f into %arg0[...] : tensor<?xf32>
  ///   scf.yield %1 : tensor<?xf32>
  /// }
  /// --> bound(%0)[0] == bound(%t)[0]
  /// --> bound(%arg0)[0] == bound(%t)[0]
  ///
  /// Example 2:
  /// %0 = scf.for ... iter_args(%arg0 = %t) -> tensor<?xf32> {
  ///   %sz = tensor.dim %arg0 : tensor<?xf32>
  ///   %incr = arith.addi %sz, %c1 : index
  ///   %1 = tensor.empty(%incr) : tensor<?xf32>
  ///   scf.yield %1 : tensor<?xf32>
  /// }
  /// --> The yielded tensor dimension size changes with each iteration. Such
  ///     loops are not supported and no constraints are added.
  static void populateIterArgBounds(scf::ForOp forOp, Value value,
                                    std::optional<int64_t> dim,
                                    ValueBoundsConstraintSet &cstr) {
    // `value` is an iter_arg or an OpResult.
    int64_t iterArgIdx;
    if (auto iterArg = llvm::dyn_cast<BlockArgument>(value)) {
      iterArgIdx = iterArg.getArgNumber() - forOp.getNumInductionVars();
    } else {
      iterArgIdx = llvm::cast<OpResult>(value).getResultNumber();
    }

    Value yieldedValue = cast<scf::YieldOp>(forOp.getBody()->getTerminator())
                             .getOperand(iterArgIdx);
    Value iterArg = forOp.getRegionIterArg(iterArgIdx);
    Value initArg = forOp.getInitArgs()[iterArgIdx];

    // An EQ constraint can be added if the yielded value (dimension size)
    // equals the corresponding block argument (dimension size).
    if (cstr.populateAndCompare(
            /*lhs=*/{yieldedValue, dim},
            ValueBoundsConstraintSet::ComparisonOperator::EQ,
            /*rhs=*/{iterArg, dim})) {
      if (dim.has_value()) {
        cstr.bound(value)[*dim] == cstr.getExpr(initArg, dim);
      } else {
        cstr.bound(value) == cstr.getExpr(initArg);
      }
    }
  }

  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto forOp = cast<ForOp>(op);

    if (value == forOp.getInductionVar()) {
      // TODO: Take into account step size.
      cstr.bound(value) >= forOp.getLowerBound();
      cstr.bound(value) < forOp.getUpperBound();
      return;
    }

    // Handle iter_args and OpResults.
    populateIterArgBounds(forOp, value, std::nullopt, cstr);
  }

  void populateBoundsForShapedValueDim(Operation *op, Value value, int64_t dim,
                                       ValueBoundsConstraintSet &cstr) const {
    auto forOp = cast<ForOp>(op);
    // Handle iter_args and OpResults.
    populateIterArgBounds(forOp, value, dim, cstr);
  }
};

struct IfOpInterface
    : public ValueBoundsOpInterface::ExternalModel<IfOpInterface, IfOp> {

  static void populateBounds(scf::IfOp ifOp, Value value,
                             std::optional<int64_t> dim,
                             ValueBoundsConstraintSet &cstr) {
    unsigned int resultNum = cast<OpResult>(value).getResultNumber();
    Value thenValue = ifOp.thenYield().getResults()[resultNum];
    Value elseValue = ifOp.elseYield().getResults()[resultNum];

    auto boundsBuilder = cstr.bound(value);
    if (dim)
      boundsBuilder[*dim];

    // Compare yielded values.
    // If thenValue <= elseValue:
    // * result <= elseValue
    // * result >= thenValue
    if (cstr.populateAndCompare(
            /*lhs=*/{thenValue, dim},
            ValueBoundsConstraintSet::ComparisonOperator::LE,
            /*rhs=*/{elseValue, dim})) {
      if (dim) {
        cstr.bound(value)[*dim] >= cstr.getExpr(thenValue, dim);
        cstr.bound(value)[*dim] <= cstr.getExpr(elseValue, dim);
      } else {
        cstr.bound(value) >= thenValue;
        cstr.bound(value) <= elseValue;
      }
    }
    // If elseValue <= thenValue:
    // * result <= thenValue
    // * result >= elseValue
    if (cstr.populateAndCompare(
            /*lhs=*/{elseValue, dim},
            ValueBoundsConstraintSet::ComparisonOperator::LE,
            /*rhs=*/{thenValue, dim})) {
      if (dim) {
        cstr.bound(value)[*dim] >= cstr.getExpr(elseValue, dim);
        cstr.bound(value)[*dim] <= cstr.getExpr(thenValue, dim);
      } else {
        cstr.bound(value) >= elseValue;
        cstr.bound(value) <= thenValue;
      }
    }
  }

  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    populateBounds(cast<IfOp>(op), value, /*dim=*/std::nullopt, cstr);
  }

  void populateBoundsForShapedValueDim(Operation *op, Value value, int64_t dim,
                                       ValueBoundsConstraintSet &cstr) const {
    populateBounds(cast<IfOp>(op), value, dim, cstr);
  }
};

} // namespace
} // namespace scf
} // namespace mlir

void mlir::scf::registerValueBoundsOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, scf::SCFDialect *dialect) {
    scf::ForOp::attachInterface<scf::ForOpInterface>(*ctx);
    scf::IfOp::attachInterface<scf::IfOpInterface>(*ctx);
  });
}
