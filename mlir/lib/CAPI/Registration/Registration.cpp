//===- Registration.cpp - C Interface for MLIR Registration ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Registration.h"

#include "mlir/CAPI/IR.h"
#include "mlir/InitAllDialects.h"

void mlirRegisterAllDialects(MlirContext context) {
  registerAllDialects(unwrap(context)->getDialectRegistry());
  // TODO: we may not want to eagerly load here.
  unwrap(context)->getDialectRegistry().loadAll(unwrap(context));
}
