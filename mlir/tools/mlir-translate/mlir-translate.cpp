//===- mlir-translate.cpp - MLIR Translate Driver -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that translates a file from/to MLIR using one
// of the registered translations.
//
//===----------------------------------------------------------------------===//

#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

using namespace mlir;

namespace mlir {
// Defined in the test directory, no public header.
void registerTestRoundtripSPIRV();
void registerTestRoundtripDebugSPIRV();
#ifdef MLIR_INCLUDE_TESTS
void registerTestToLLVMIR();
#endif
} // namespace mlir

static void registerTestTranslations() {
  registerTestRoundtripSPIRV();
  registerTestRoundtripDebugSPIRV();
#ifdef MLIR_INCLUDE_TESTS
  registerTestToLLVMIR();
#endif
}

int main(int argc, char **argv) {
  registerAllTranslations();
  registerTestTranslations();
  return failed(mlirTranslateMain(argc, argv, "MLIR Translation Testing Tool"));
}
