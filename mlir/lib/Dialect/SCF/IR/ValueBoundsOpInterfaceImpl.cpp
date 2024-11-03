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
using presburger::BoundType;

namespace mlir {
namespace scf {
namespace {

struct ForOpInterface
    : public ValueBoundsOpInterface::ExternalModel<ForOpInterface, ForOp> {

  /// Populate bounds of values/dimensions for iter_args/OpResults.
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

    // An EQ constraint can be added if the yielded value (dimension size)
    // equals the corresponding block argument (dimension size).
    Value yieldedValue = cast<scf::YieldOp>(forOp.getBody()->getTerminator())
                             .getOperand(iterArgIdx);
    Value iterArg = forOp.getRegionIterArg(iterArgIdx);
    Value initArg = forOp.getInitArgs()[iterArgIdx];

    auto addEqBound = [&]() {
      if (dim.has_value()) {
        cstr.bound(value)[*dim] == cstr.getExpr(initArg, dim);
      } else {
        cstr.bound(value) == initArg;
      }
    };

    if (yieldedValue == iterArg) {
      addEqBound();
      return;
    }

    // Compute EQ bound for yielded value.
    AffineMap bound;
    ValueDimList boundOperands;
    LogicalResult status = ValueBoundsConstraintSet::computeBound(
        bound, boundOperands, BoundType::EQ, yieldedValue, dim,
        [&](Value v, std::optional<int64_t> d) {
          // Stop when reaching a block argument of the loop body.
          if (auto bbArg = llvm::dyn_cast<BlockArgument>(v))
            return bbArg.getOwner()->getParentOp() == forOp;
          // Stop when reaching a value that is defined outside of the loop. It
          // is impossible to reach an iter_arg from there.
          Operation *op = v.getDefiningOp();
          return forOp.getRegion().findAncestorOpInRegion(*op) == nullptr;
        });
    if (failed(status))
      return;
    if (bound.getNumResults() != 1)
      return;

    // Check if computed bound equals the corresponding iter_arg.
    Value singleValue = nullptr;
    std::optional<int64_t> singleDim;
    if (auto dimExpr = bound.getResult(0).dyn_cast<AffineDimExpr>()) {
      int64_t idx = dimExpr.getPosition();
      singleValue = boundOperands[idx].first;
      singleDim = boundOperands[idx].second;
    } else if (auto symExpr = bound.getResult(0).dyn_cast<AffineSymbolExpr>()) {
      int64_t idx = symExpr.getPosition() + bound.getNumDims();
      singleValue = boundOperands[idx].first;
      singleDim = boundOperands[idx].second;
    }
    if (singleValue == iterArg && singleDim == dim)
      addEqBound();
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

} // namespace
} // namespace scf
} // namespace mlir

void mlir::scf::registerValueBoundsOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, scf::SCFDialect *dialect) {
    scf::ForOp::attachInterface<scf::ForOpInterface>(*ctx);
  });
}
