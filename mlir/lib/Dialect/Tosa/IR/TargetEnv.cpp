//===-------------- TosaTarget.cpp - TOSA Target utilities ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TargetEnv.h"

namespace mlir {
namespace tosa {

TargetEnvAttr lookupTargetEnv(Operation *op) {
  while (op) {
    op = SymbolTable::getNearestSymbolTable(op);
    if (!op)
      break;

    if (auto attr = op->getAttrOfType<TargetEnvAttr>(TargetEnvAttr::name))
      return attr;

    op = op->getParentOp();
  }

  return {};
}

TargetEnvAttr getDefaultTargetEnv(MLIRContext *context) {
  return TargetEnvAttr::get(context, Level::eightK,
                            {Profile::pro_int, Profile::pro_fp}, {});
}

TargetEnvAttr lookupTargetEnvOrDefault(Operation *op) {
  if (auto attr = lookupTargetEnv(op))
    return attr;

  return getDefaultTargetEnv(op->getContext());
}

} // namespace tosa
} // namespace mlir
