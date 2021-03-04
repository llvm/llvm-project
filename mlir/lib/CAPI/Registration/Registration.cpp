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
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

void mlirRegisterAllDialects(MlirContext context) {
  mlir::registerAllDialects(*unwrap(context));
  // TODO: we may not want to eagerly load here.
  unwrap(context)->loadAllAvailableDialects();
}

void mlirRegisterAllLLVMTranslations(MlirContext context) {
  mlir::registerLLVMDialectTranslation(*unwrap(context));
}
