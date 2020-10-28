//===- StandardDialect.cpp - C Interface for Standard dialect -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/StandardDialect.h"
#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

void mlirContextRegisterStandardDialect(MlirContext context) {
  unwrap(context)->getDialectRegistry().insert<mlir::StandardOpsDialect>();
}

MlirDialect mlirContextLoadStandardDialect(MlirContext context) {
  return wrap(unwrap(context)->getOrLoadDialect<mlir::StandardOpsDialect>());
}

MlirStringRef mlirStandardDialectGetNamespace() {
  return wrap(mlir::StandardOpsDialect::getDialectNamespace());
}
