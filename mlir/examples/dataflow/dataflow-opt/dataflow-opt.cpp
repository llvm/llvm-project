//===-- dataflow-opt.cpp - dataflow tutorial entry point ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top-level file for the dataflow tutorial.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace test {
void registerTestMetadataAnalysisPass();
};
} // namespace mlir

int main(int argc, char *argv[]) {
  // Register all MLIR core dialects.
  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  registerAllExtensions(registry);

  // Register test-metadata-analysis pass.
  mlir::test::registerTestMetadataAnalysisPass();
  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "dataflow-opt optimizer driver", registry));
}
