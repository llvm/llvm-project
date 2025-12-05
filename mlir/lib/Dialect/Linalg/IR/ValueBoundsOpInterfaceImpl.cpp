//===- ValueBoundsOpInterfaceImpl.cpp - Impl. of ValueBoundsOpInterface ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/ValueBoundsOpInterfaceImpl.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

using namespace mlir;

namespace mlir {
namespace linalg {
namespace {

struct IndexOpInterface
    : public ValueBoundsOpInterface::ExternalModel<IndexOpInterface, IndexOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto indexOp = cast<IndexOp>(op);
    auto linalgOp = indexOp->getParentOfType<LinalgOp>();
    assert(value == indexOp.getResult() && "invalid value");

    // index >= 0
    cstr.bound(value) >= 0;

    // index < dim size
    int64_t flatDimPos =
        cast<AffineDimExpr>(
            linalgOp.getShapesToLoopsMap().getResult(indexOp.getDim()))
            .getPosition();
    // Find the `flatDimPos`-th operand dimension.
    int64_t flatDimCtr = 0;
    for (Value operand : linalgOp->getOperands()) {
      assert(flatDimPos >= flatDimCtr && "invalid pos");
      auto shapedType = llvm::cast<ShapedType>(operand.getType());
      if (flatDimPos < flatDimCtr + shapedType.getRank()) {
        cstr.bound(value) < cstr.getExpr(operand, flatDimPos - flatDimCtr);
        break;
      }
      flatDimCtr += shapedType.getRank();
    }
  }
};

} // namespace
} // namespace linalg
} // namespace mlir

void mlir::linalg::registerValueBoundsOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, linalg::LinalgDialect *dialect) {
    IndexOp::attachInterface<IndexOpInterface>(*ctx);
    // Note: ValueBoundsOpInterface implementation is not required for ops that
    // implement `DestinationStyleOpInterface` (for querying shaped OpResults).
  });
}
