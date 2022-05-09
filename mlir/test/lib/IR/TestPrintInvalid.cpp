//===- TestPrintInvalid.cpp - Test printing invalid ops -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass creates and prints to the standard output an invalid operation and
// a valid operation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
struct TestPrintInvalidPass
    : public PassWrapper<TestPrintInvalidPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestPrintInvalidPass)

  StringRef getArgument() const final { return "test-print-invalid"; }
  StringRef getDescription() const final {
    return "Test printing invalid ops.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    Location loc = getOperation().getLoc();
    OpBuilder builder(getOperation().getBodyRegion());
    auto funcOp = builder.create<func::FuncOp>(
        loc, "test", FunctionType::get(getOperation().getContext(), {}, {}));
    funcOp.addEntryBlock();
    // The created function is invalid because there is no return op.
    llvm::outs() << "Invalid operation:\n" << funcOp << "\n";
    builder.setInsertionPointToEnd(&funcOp.getBody().front());
    builder.create<func::ReturnOp>(loc);
    // Now this function is valid.
    llvm::outs() << "Valid operation:\n" << funcOp << "\n";
    funcOp.erase();
  }
};
} // namespace

namespace mlir {
void registerTestPrintInvalidPass() {
  PassRegistration<TestPrintInvalidPass>{};
}
} // namespace mlir
