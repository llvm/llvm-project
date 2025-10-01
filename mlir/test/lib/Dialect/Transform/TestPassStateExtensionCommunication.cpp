//===- TestPassStateExtensionCommunication.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a test pass that showcases how communication can be
// conducted between a regular mlir pass and transform ops through the
// transform state extension stateInitializer and stateExporter mechanism.
//
//===----------------------------------------------------------------------===//

#include "TestTransformStateExtension.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::test;

namespace {
template <typename Derived>
class OpPassWrapper : public PassWrapper<Derived, OperationPass<>> {};

struct TestPassStateExtensionCommunication
    : public PassWrapper<TestPassStateExtensionCommunication,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestPassStateExtensionCommunication)

  StringRef getArgument() const final {
    return "test-pass-state-extension-communication";
  }

  StringRef getDescription() const final {
    return "test state communciation between a mlir pass and transform ops";
  }

  static void printVector(const SmallVector<std::string> &opCollection,
                          const std::string &extraMessage = {}) {
    outs() << "Printing opCollection" << extraMessage
           << ", size: " << opCollection.size() << "\n";
    for (const auto &subVector : opCollection) {
      outs() << subVector << " ";
    }
    outs() << "\n";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Create an opCollection vector.
    SmallVector<std::string> opCollection = {"PASS-TRANSFORMOP-PASS "};
    printVector(opCollection, " before processing transform ops");

    auto stateInitializer =
        [&opCollection](mlir::transform::TransformState &state) -> void {
      TransformStateInitializerExtension *ext =
          state.getExtension<TransformStateInitializerExtension>();
      if (!ext)
        state.addExtension<TransformStateInitializerExtension>(0, opCollection);
    };

    auto stateExporter =
        [&opCollection](
            mlir::transform::TransformState &state) -> LogicalResult {
      TransformStateInitializerExtension *ext =
          state.getExtension<TransformStateInitializerExtension>();
      if (!ext) {
        errs() << "Target transform state extension not found!\n";
        return failure();
      }
      opCollection.clear();
      opCollection = ext->getRegisteredOps();
      return success();
    };

    // Process transform ops with stateInitializer and stateExporter.
    for (auto op : module.getBody()->getOps<transform::TransformOpInterface>())
      if (failed(transform::applyTransforms(
              module, op, {}, mlir::transform::TransformOptions(), false,
              stateInitializer, stateExporter)))
        return signalPassFailure();

    // Print the opCollection vector after processing transform ops.
    printVector(opCollection, " after processing transform ops");
  }
};
} // namespace

namespace mlir {
namespace test {
/// Registers the test pass here.
void registerTestPassStateExtensionCommunication() {
  PassRegistration<TestPassStateExtensionCommunication> reg;
}
} // namespace test
} // namespace mlir
