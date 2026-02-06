//===------ TestRemarkPipeline.cpp --- dynamic pipeline test pass --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to test the dynamic pipeline feature.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Remarks.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/WalkResult.h"

using namespace mlir;

namespace {

class TestRemarkPass : public PassWrapper<TestRemarkPass, OperationPass<>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestRemarkPass)

  StringRef getArgument() const final { return "test-remark"; }
  StringRef getDescription() const final {
    return "Tests the remark pipeline feature";
  }

  TestRemarkPass() = default;

  void runOnOperation() override {

    getOperation()->walk([](Operation *op) {
      if (isa<ModuleOp>(op))
        return WalkResult::advance();
      Location loc = op->getLoc();
      mlir::remark::missed(loc, remark::RemarkOpts::name("test-remark")
                                    .category("a-category-1-missed"))
          << remark::add("This is a test missed remark")
          << remark::reason("because we are testing the remark pipeline")
          << remark::suggest("try using the remark pipeline feature");
      mlir::remark::passed(
          loc,
          remark::RemarkOpts::name("test-remark").category("category-1-passed"))
          << remark::add("This is a test passed remark (should be dropped)")
          << remark::reason("because we are testing the remark pipeline")
          << remark::suggest("try using the remark pipeline feature");
      mlir::remark::passed(
          loc,
          remark::RemarkOpts::name("test-remark").category("category-1-passed"))
          << remark::add("This is a test passed remark")
          << remark::reason("because we are testing the remark pipeline")
          << remark::suggest("try using the remark pipeline feature");

      mlir::remark::failed(
          loc,
          remark::RemarkOpts::name("test-remark").category("category-2-failed"))
          << remark::add("This is a test failed remark")
          << remark::reason("because we are testing the remark pipeline")
          << remark::suggest("try using the remark pipeline feature");

      mlir::remark::analysis(loc, remark::RemarkOpts::name("test-remark")
                                      .category("category-2-analysis"))
          << remark::add("This is a test analysis remark");
      return WalkResult::advance();
    });
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestRemarkPass() { PassRegistration<TestRemarkPass>(); }
} // namespace test
} // namespace mlir
