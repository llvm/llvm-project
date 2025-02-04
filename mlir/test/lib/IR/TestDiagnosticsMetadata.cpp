//===- TestDiagnosticsMetadata.cpp - Test Diagnostic Metatdata ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains test passes for constructing and resolving dominance
// information.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;

namespace {
struct TestDiagnosticMetadataPass
    : public PassWrapper<TestDiagnosticMetadataPass,
                         InterfacePass<SymbolOpInterface>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestDiagnosticMetadataPass)

  StringRef getArgument() const final { return "test-diagnostic-metadata"; }
  StringRef getDescription() const final { return "Test diagnostic metadata."; }
  TestDiagnosticMetadataPass() = default;
  TestDiagnosticMetadataPass(const TestDiagnosticMetadataPass &) {}

  void runOnOperation() override {
    llvm::errs() << "Test '" << getOperation().getName() << "'\n";

    // Build a diagnostic handler that has filtering capabilities.
    ScopedDiagnosticHandler handler(&getContext(), [](mlir::Diagnostic &diag) {
      return mlir::success(
          llvm::none_of(diag.getMetadata(), [](mlir::DiagnosticArgument &arg) {
            return arg.getKind() == mlir::DiagnosticArgument::
                                        DiagnosticArgumentKind::String &&
                   arg.getAsString().contains("hello");
          }));
    });

    // Emit a diagnostic for every operation with a valid loc.
    getOperation()->walk([&](Operation *op) {
      if (StringAttr strAttr = op->getAttrOfType<StringAttr>("attr")) {
        if (strAttr.getValue() == "emit_error")
          emitError(op->getLoc(), "test diagnostic metadata")
              .getUnderlyingDiagnostic()
              ->getMetadata()
              .push_back(DiagnosticArgument("hello"));
      }
    });
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestDiagnosticsMetadataPass() {
  PassRegistration<TestDiagnosticMetadataPass>{};
}
} // namespace test
} // namespace mlir
