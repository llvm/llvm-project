//===- mlir-minimal-opt-canonicalize.cpp ------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  // Register only the canonicalize pass
  // This pulls in the pattern rewrite engine as well as the whole PDL
  // compiler/intepreter.
  mlir::registerCanonicalizerPass();

  mlir::DialectRegistry registry;
  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Minimal Standalone optimizer driver\n", registry));
}
