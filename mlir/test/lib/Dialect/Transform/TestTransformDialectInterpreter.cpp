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

#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
/// Simple pass that applies transform dialect ops directly contained in a
/// module.
class TestTransformDialectInterpreterPass
    : public PassWrapper<TestTransformDialectInterpreterPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestTransformDialectInterpreterPass)

  TestTransformDialectInterpreterPass() = default;
  TestTransformDialectInterpreterPass(
      const TestTransformDialectInterpreterPass &) {}

  StringRef getArgument() const override {
    return "test-transform-dialect-interpreter";
  }

  StringRef getDescription() const override {
    return "apply transform dialect operations one by one";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    for (auto op :
         module.getBody()->getOps<transform::TransformOpInterface>()) {
      if (failed(transform::applyTransforms(
              module, op,
              transform::TransformOptions().enableExpensiveChecks(
                  enableExpensiveChecks))))
        return signalPassFailure();
    }
  }

  Option<bool> enableExpensiveChecks{
      *this, "enable-expensive-checks", llvm::cl::init(false),
      llvm::cl::desc("perform expensive checks to better report errors in the "
                     "transform IR")};
};

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
/// Registers the test pass for applying transform dialect ops.
void registerTestTransformDialectInterpreterPass() {
  PassRegistration<TestTransformDialectInterpreterPass> reg;
}
} // namespace test
} // namespace mlir
