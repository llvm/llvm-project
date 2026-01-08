//===- TestInterfaces.cpp - Test interface generation and application -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace test;

namespace {
/// This test checks various aspects of Type interface generation and
/// application.
struct TestTypeInterfaces
    : public PassWrapper<TestTypeInterfaces, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTypeInterfaces)

  StringRef getArgument() const final { return "test-type-interfaces"; }
  StringRef getDescription() const final {
    return "Test type interface support.";
  }
  void runOnOperation() override {
    getOperation().walk([](Operation *op) {
      for (Type type : op->getResultTypes()) {
        if (auto testInterface = dyn_cast<TestTypeInterface>(type)) {
          testInterface.printTypeA(op->getLoc());
          testInterface.printTypeB(op->getLoc());
          testInterface.printTypeC(op->getLoc());
          testInterface.printTypeC(op->getLoc(), 42);
          testInterface.printTypeC(op->getLoc(), 3.14f);
          testInterface.printTypeD(op->getLoc());
          // Just check that we can assign the result to a variable of interface
          // type.
          TestTypeInterface result = testInterface.printTypeRet(op->getLoc());
          (void)result;
        }
        if (auto testType = dyn_cast<TestType>(type))
          testType.printTypeE(op->getLoc());
      }
    });
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestInterfaces() { PassRegistration<TestTypeInterfaces>(); }
} // namespace test
} // namespace mlir
