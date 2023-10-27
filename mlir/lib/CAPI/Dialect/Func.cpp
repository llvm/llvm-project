//===- Func.cpp - C Interface for Func dialect ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/Func.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Func, func, mlir::func::FuncDialect)

void mlirFuncSetArgAttr(MlirOperation op, intptr_t pos, MlirStringRef name,
                        MlirAttribute attr) {
  llvm::cast<mlir::func::FuncOp>(unwrap(op))
      .setArgAttr(pos, unwrap(name), unwrap(attr));
}
