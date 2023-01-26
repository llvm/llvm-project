//===- ShapedOpInterfaces.cpp - Interfaces for Shaped Ops -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/ShapedOpInterfaces.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// ShapedDimOpInterface
//===----------------------------------------------------------------------===//

LogicalResult mlir::detail::verifyShapedDimOpInterface(Operation *op) {
  if (op->getNumResults() != 1)
    return op->emitError("expected single op result");
  if (!op->getResult(0).getType().isIndex())
    return op->emitError("expect index result type");
  return success();
}

/// Include the definitions of the interface.
#include "mlir/Interfaces/ShapedOpInterfaces.cpp.inc"
