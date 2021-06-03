//===- mlir-reduce.cpp - The MLIR reducer ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the general framework of the MLIR reducer tool. It
// parses the command line arguments, parses the initial MLIR test case and sets
// up the testing environment. It  outputs the most reduced test case variant
// after executing the reduction passes.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-reduce/MlirReduceMain.h"

using namespace mlir;

namespace mlir {
namespace test {
#ifdef MLIR_INCLUDE_TESTS
void registerTestDialect(DialectRegistry &);
#endif
} // namespace test
} // namespace mlir

int main(int argc, char **argv) {
  registerAllPasses();

  DialectRegistry registry;
  registerAllDialects(registry);
#ifdef MLIR_INCLUDE_TESTS
  test::registerTestDialect(registry);
#endif
  MLIRContext context(registry);

  return failed(mlirReduceMain(argc, argv, context));
}
