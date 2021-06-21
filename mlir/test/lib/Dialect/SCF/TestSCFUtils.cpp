//===- TestSCFUtils.cpp --- Pass to test independent SCF dialect utils ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to test SCF dialect utils.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SetVector.h"

using namespace mlir;

namespace {
class TestSCFForUtilsPass
    : public PassWrapper<TestSCFForUtilsPass, FunctionPass> {
public:
  StringRef getArgument() const final { return "test-scf-for-utils"; }
  StringRef getDescription() const final { return "test scf.for utils"; }
  explicit TestSCFForUtilsPass() = default;

  void runOnFunction() override {
    FuncOp func = getFunction();
    SmallVector<scf::ForOp, 4> toErase;

    func.walk([&](Operation *fakeRead) {
      if (fakeRead->getName().getStringRef() != "fake_read")
        return;
      auto *fakeCompute = fakeRead->getResult(0).use_begin()->getOwner();
      auto *fakeWrite = fakeCompute->getResult(0).use_begin()->getOwner();
      auto loop = fakeRead->getParentOfType<scf::ForOp>();

      OpBuilder b(loop);
      (void)loop.moveOutOfLoop({fakeRead});
      fakeWrite->moveAfter(loop);
      auto newLoop = cloneWithNewYields(b, loop, fakeRead->getResult(0),
                                        fakeCompute->getResult(0));
      fakeCompute->getResult(0).replaceAllUsesWith(
          newLoop.getResults().take_back()[0]);
      toErase.push_back(loop);
    });
    for (auto loop : llvm::reverse(toErase))
      loop.erase();
  }
};

class TestSCFIfUtilsPass
    : public PassWrapper<TestSCFIfUtilsPass, FunctionPass> {
public:
  StringRef getArgument() const final { return "test-scf-if-utils"; }
  StringRef getDescription() const final { return "test scf.if utils"; }
  explicit TestSCFIfUtilsPass() = default;

  void runOnFunction() override {
    int count = 0;
    FuncOp func = getFunction();
    func.walk([&](scf::IfOp ifOp) {
      auto strCount = std::to_string(count++);
      FuncOp thenFn, elseFn;
      OpBuilder b(ifOp);
      outlineIfOp(b, ifOp, &thenFn, std::string("outlined_then") + strCount,
                  &elseFn, std::string("outlined_else") + strCount);
    });
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestSCFUtilsPass() {
  PassRegistration<TestSCFForUtilsPass>();
  PassRegistration<TestSCFIfUtilsPass>();
}
} // namespace test
} // namespace mlir
