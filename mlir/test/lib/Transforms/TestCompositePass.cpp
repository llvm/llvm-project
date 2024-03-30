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
  registerPass([]() -> std::unique_ptr<Pass> {
    return createCompositePass("TestCompositePass", "test-composite-pass",
                               [](OpPassManager &p) {
                                 p.addPass(createCanonicalizerPass());
                                 p.addPass(createCSEPass());
                               });
  });
}
} // namespace test
} // namespace mlir
