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

  StringRef getArgument() const override {
    return "test-transform-dialect-interpreter";
  }

  StringRef getDescription() const override {
    return "apply transform dialect operations one by one";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    transform::TransformState state(module);
    for (auto op :
         module.getBody()->getOps<transform::TransformOpInterface>()) {
      if (failed(state.applyTransform(op)))
        return signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
namespace test {
/// Registers the test pass for applying transform dialect ops.
void registerTestTransformDialectInterpreterPass() {
  PassRegistration<TestTransformDialectInterpreterPass> reg;
}
} // namespace test
} // namespace mlir
