//===------ TestCompositePass.cpp --- composite test pass -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to test the composite pass utility.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace test {
void registerTestCompositePass() {
  registerPassPipeline(
      "test-composite-fixed-point-pass", "Test composite pass",
      [](OpPassManager &pm, StringRef optionsStr,
         function_ref<LogicalResult(const Twine &)> errorHandler) {
        if (!optionsStr.empty())
          return failure();

        pm.addPass(createCompositeFixedPointPass(
            "TestCompositePass", [](OpPassManager &p) {
              p.addPass(createCanonicalizerPass());
              p.addPass(createCSEPass());
            }));
        return success();
      },
      [](function_ref<void(const detail::PassOptions &)>) {});
}
} // namespace test
} // namespace mlir
