//===- TestDynDialect.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a fake 'test_dyn' dynamic dialect that is used to test the
// registration of dynamic dialects.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/ExtensibleDialect.h"

using namespace mlir;

namespace test {
void registerTestDynDialect(DialectRegistry &registry) {
  registry.insertDynamic(
      "test_dyn", [](MLIRContext *ctx, DynamicDialect *testDyn) {
        auto opVerifier = [](Operation *op) -> LogicalResult {
          if (op->getNumOperands() == 0 && op->getNumResults() == 1 &&
              op->getNumRegions() == 0)
            return success();
          return op->emitError(
              "expected a single result, no operands and no regions");
        };

        auto opRegionVerifier = [](Operation *op) { return success(); };

        testDyn->registerDynamicOp(DynamicOpDefinition::get(
            "one_result", testDyn, opVerifier, opRegionVerifier));
      });
}
} // namespace test
