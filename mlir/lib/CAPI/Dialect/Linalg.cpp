//===- Linalg.cpp - C Interface for Linalg dialect ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/Linalg.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace mlir;
using namespace mlir::linalg;

/// Apply the special region builder for the builtin named Linalg op.
/// Assert that `op` is a builtin named Linalg op.
void mlirLinalgFillBuiltinNamedOpRegion(MlirOperation mlirOp) {
  Operation *op = unwrap(mlirOp);
  auto linalgOp = cast<LinalgOp>(op);
  auto *dialect = static_cast<LinalgDialect *>(linalgOp->getDialect());
  LinalgDialect::RegionBuilderFunType fun =
      dialect->getRegionBuilder(op->getName().getStringRef());

  assert(fun && "Expected a builtin named Linalg op.");
  assert(op->getNumRegions() == 1 && "Expected Linalg op with 1 region");
  assert(op->getRegion(0).getBlocks().empty() &&
         "Expected Linalg op with 0 blocks");

  SmallVector<Type, 8> argTypes;
  SmallVector<Location, 8> argLocs;
  for (OpOperand *opOperand : linalgOp.getInputAndOutputOperands()) {
    argTypes.push_back(getElementTypeOrSelf(opOperand->get().getType()));
    argLocs.push_back(opOperand->get().getLoc());
  }

  ImplicitLocOpBuilder b(op->getLoc(), op->getContext());
  Region &region = op->getRegion(0);
  Block *body = b.createBlock(&region, /*insertPt=*/{}, argTypes, argLocs);
  b.setInsertionPointToStart(body);
  fun(b, *body);
}

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Linalg, linalg, LinalgDialect)
