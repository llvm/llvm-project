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

//===----------------------------------------------------------------------===//
// PassManager/OpPassManager APIs.
//===----------------------------------------------------------------------===//

MlirPassManager mlirPassManagerCreate(MlirContext ctx) {
  return wrap(new PassManager(unwrap(ctx)));
}

void mlirPassManagerDestroy(MlirPassManager passManager) {
  delete unwrap(passManager);
}

MlirOpPassManager
mlirPassManagerGetAsOpPassManager(MlirPassManager passManager) {
  return wrap(static_cast<OpPassManager *>(unwrap(passManager)));
}

MlirLogicalResult mlirPassManagerRun(MlirPassManager passManager,
                                     MlirModule module) {
  return wrap(unwrap(passManager)->run(unwrap(module)));
}

void mlirPassManagerEnableIRPrinting(MlirPassManager passManager) {
  return unwrap(passManager)->enableIRPrinting();
}

void mlirPassManagerEnableVerifier(MlirPassManager passManager, bool enable) {
  unwrap(passManager)->enableVerifier(enable);
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

void mlirPrintPassPipeline(MlirOpPassManager passManager,
                           MlirStringCallback callback, void *userData) {
  detail::CallbackOstream stream(callback, userData);
  unwrap(passManager)->printAsTextualPipeline(stream);
}

MlirLogicalResult mlirParsePassPipeline(MlirOpPassManager passManager,
                                        MlirStringRef pipeline) {
  // TODO: errors are sent to std::errs() at the moment, we should pass in a
  // stream and redirect to a diagnostic.
  return wrap(mlir::parsePassPipeline(unwrap(pipeline), *unwrap(passManager)));
}
