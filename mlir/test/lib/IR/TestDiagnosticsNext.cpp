//===- TestDiagnosticsNext.cpp - Test Diagnostic Utilities ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains test passes for the next diagnostic manipulator.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace {
struct TestDiagnosticsNextPass
    : public PassWrapper<TestDiagnosticsNextPass,
                         InterfacePass<SymbolOpInterface>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestDiagnosticsNextPass)

  StringRef getArgument() const final { return "test-diagnostic-next"; }
  StringRef getDescription() const final {
    return "Test diagnostic next support.";
  }

  void runOnOperation() override {
    getOperation()->walk([](SymbolOpInterface op) {
      StringRef opName = op.getNameAttr();
      InFlightDiagnostic diag = op->emitRemark(opName) << "1" << next;
      diag << opName << "2" << next << opName << "3";
    });
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestDiagnosticsNextPass() {
  PassRegistration<TestDiagnosticsNextPass>{};
}
} // namespace test
} // namespace mlir
