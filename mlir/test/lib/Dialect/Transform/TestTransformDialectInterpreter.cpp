//===- TestTransformDialectInterpreter.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a test pass that interprets Transform dialect operations in
// the module.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
template <typename Derived>
class OpPassWrapper : public PassWrapper<Derived, OperationPass<>> {};

struct TestTransformDialectEraseSchedulePass
    : public PassWrapper<TestTransformDialectEraseSchedulePass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestTransformDialectEraseSchedulePass)

  StringRef getArgument() const final {
    return "test-transform-dialect-erase-schedule";
  }

  StringRef getDescription() const final {
    return "erase transform dialect schedule from the IR";
  }

  void runOnOperation() override {
    getOperation()->walk<WalkOrder::PreOrder>([&](Operation *nestedOp) {
      if (isa<transform::TransformOpInterface>(nestedOp)) {
        nestedOp->erase();
        return WalkResult::skip();
      }
      return WalkResult::advance();
    });
  }
};
} // namespace

namespace mlir {
namespace test {
/// Registers the test pass for erasing transform dialect ops.
void registerTestTransformDialectEraseSchedulePass() {
  PassRegistration<TestTransformDialectEraseSchedulePass> reg;
}
} // namespace test
} // namespace mlir
