//===- mlir-minimal-opt.cpp -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-opt/MlirOptMain.h"

/// This test includes the minimal amount of components for mlir-opt, that is
/// the CoreIR, the printer/parser, the bytecode reader/writer, the
/// passmanagement infrastructure and all the instrumentation.
int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Minimal Standalone optimizer driver\n", registry));
}
