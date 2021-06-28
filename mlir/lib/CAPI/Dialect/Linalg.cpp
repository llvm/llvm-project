//===- Linalg.cpp - C Interface for Linalg dialect ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/Linalg.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"

using namespace mlir;
using namespace mlir::linalg;

/// Apply the special region builder for the builtin named Linalg op.
/// Assert that `op` is a builtin named Linalg op.
void mlirLinalgFillBuiltinNamedOpRegion(MlirDialect linalgDialect,
                                        MlirOperation mlirOp) {
  Operation *op = unwrap(mlirOp);

  LinalgDialect::RegionBuilderFunType fun =
      static_cast<LinalgDialect *>(unwrap(linalgDialect))
          ->getRegionBuilder(op->getName().getStringRef());
  assert(fun && "Expected a builtin named Linalg op.");
  assert(op->getNumRegions() == 1 && "Expected Linalg op with 1 region");
  assert(op->getRegion(0).getBlocks().empty() &&
         "Expected Linalg op with 0 blocks");

  SmallVector<Type, 8> argTypes;
  auto linalgOp = cast<LinalgOp>(op);
  for (OpOperand *opOperand : linalgOp.getInputAndOutputOperands())
    argTypes.push_back(getElementTypeOrSelf(opOperand->get().getType()));

  ImplicitLocOpBuilder b(op->getLoc(), op->getContext());
  Region &region = op->getRegion(0);
  Block *body = b.createBlock(&region, /*insertPt=*/{}, argTypes);
  b.setInsertionPointToStart(body);
  fun(b, *body);
}

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Linalg, linalg, LinalgDialect)
