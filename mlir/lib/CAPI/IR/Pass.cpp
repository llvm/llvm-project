//===- Pass.cpp - C Interface for General Pass Management APIs ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Pass.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

/* ========================================================================== */
/* PassManager/OpPassManager APIs. */
/* ========================================================================== */

MlirPassManager mlirPassManagerCreate(MlirContext ctx) {
  return wrap(new PassManager(unwrap(ctx)));
}

void mlirPassManagerDestroy(MlirPassManager passManager) {
  delete unwrap(passManager);
}

MlirLogicalResult mlirPassManagerRun(MlirPassManager passManager,
                                     MlirModule module) {
  return wrap(unwrap(passManager)->run(unwrap(module)));
}

MlirOpPassManager mlirPassManagerGetNestedUnder(MlirPassManager passManager,
                                                MlirStringRef operationName) {
  return wrap(&unwrap(passManager)->nest(unwrap(operationName)));
}

MlirOpPassManager mlirOpPassManagerGetNestedUnder(MlirOpPassManager passManager,
                                                  MlirStringRef operationName) {
  return wrap(&unwrap(passManager)->nest(unwrap(operationName)));
}

void mlirPassManagerAddOwnedPass(MlirPassManager passManager, MlirPass pass) {
  unwrap(passManager)->addPass(std::unique_ptr<Pass>(unwrap(pass)));
}

void mlirOpPassManagerAddOwnedPass(MlirOpPassManager passManager,
                                   MlirPass pass) {
  unwrap(passManager)->addPass(std::unique_ptr<Pass>(unwrap(pass)));
}
