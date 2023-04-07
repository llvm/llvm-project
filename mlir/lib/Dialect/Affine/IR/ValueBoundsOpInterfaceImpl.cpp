//===- ValueBoundsOpInterfaceImpl.cpp - Impl. of ValueBoundsOpInterface ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

using namespace mlir;

namespace mlir {
namespace {

struct AffineApplyOpInterface
    : public ValueBoundsOpInterface::ExternalModel<AffineApplyOpInterface,
                                                   AffineApplyOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto applyOp = cast<AffineApplyOp>(op);
    assert(value == applyOp.getResult() && "invalid value");
    assert(applyOp.getAffineMap().getNumResults() == 1 &&
           "expected single result");

    // Align affine map result with dims/symbols in the constraint set.
    AffineExpr expr = applyOp.getAffineMap().getResult(0);
    SmallVector<AffineExpr> dimReplacements = llvm::to_vector(llvm::map_range(
        applyOp.getDimOperands(), [&](Value v) { return cstr.getExpr(v); }));
    SmallVector<AffineExpr> symReplacements = llvm::to_vector(llvm::map_range(
        applyOp.getSymbolOperands(), [&](Value v) { return cstr.getExpr(v); }));
    AffineExpr bound =
        expr.replaceDimsAndSymbols(dimReplacements, symReplacements);
    cstr.bound(value) == bound;
  }
};

struct AffineMinOpInterface
    : public ValueBoundsOpInterface::ExternalModel<AffineMinOpInterface,
                                                   AffineMinOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto minOp = cast<AffineMinOp>(op);
    assert(value == minOp.getResult() && "invalid value");

    // Align affine map results with dims/symbols in the constraint set.
    for (AffineExpr expr : minOp.getAffineMap().getResults()) {
      SmallVector<AffineExpr> dimReplacements = llvm::to_vector(llvm::map_range(
          minOp.getDimOperands(), [&](Value v) { return cstr.getExpr(v); }));
      SmallVector<AffineExpr> symReplacements = llvm::to_vector(llvm::map_range(
          minOp.getSymbolOperands(), [&](Value v) { return cstr.getExpr(v); }));
      AffineExpr bound =
          expr.replaceDimsAndSymbols(dimReplacements, symReplacements);
      cstr.bound(value) <= bound;
    }
  };
};

struct AffineMaxOpInterface
    : public ValueBoundsOpInterface::ExternalModel<AffineMaxOpInterface,
                                                   AffineMaxOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto maxOp = cast<AffineMaxOp>(op);
    assert(value == maxOp.getResult() && "invalid value");

    // Align affine map results with dims/symbols in the constraint set.
    for (AffineExpr expr : maxOp.getAffineMap().getResults()) {
      SmallVector<AffineExpr> dimReplacements = llvm::to_vector(llvm::map_range(
          maxOp.getDimOperands(), [&](Value v) { return cstr.getExpr(v); }));
      SmallVector<AffineExpr> symReplacements = llvm::to_vector(llvm::map_range(
          maxOp.getSymbolOperands(), [&](Value v) { return cstr.getExpr(v); }));
      AffineExpr bound =
          expr.replaceDimsAndSymbols(dimReplacements, symReplacements);
      cstr.bound(value) >= bound;
    }
  };
};

} // namespace
} // namespace mlir

void mlir::affine::registerValueBoundsOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, AffineDialect *dialect) {
    AffineApplyOp::attachInterface<AffineApplyOpInterface>(*ctx);
    AffineMaxOp::attachInterface<AffineMaxOpInterface>(*ctx);
    AffineMinOp::attachInterface<AffineMinOpInterface>(*ctx);
  });
}
