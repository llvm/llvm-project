//===- TestDiagnostics.cpp - Test Diagnostic Utilities --------------------===//
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

#include "mlir/Pass/Pass.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;

namespace {
struct TestDiagnosticFilterPass
    : public PassWrapper<TestDiagnosticFilterPass, OperationPass<FuncOp>> {
  StringRef getArgument() const final { return "test-diagnostic-filter"; }
  StringRef getDescription() const final {
    return "Test diagnostic filtering support.";
  }
  TestDiagnosticFilterPass() {}
  TestDiagnosticFilterPass(const TestDiagnosticFilterPass &) {}

  void runOnOperation() override {
    llvm::errs() << "Test '" << getOperation().getName() << "'\n";

    // Build a diagnostic handler that has filtering capabilities.
    auto filterFn = [&](Location loc) {
      // Ignore non-file locations.
      FileLineColLoc fileLoc = loc.dyn_cast<FileLineColLoc>();
      if (!fileLoc)
        return true;

      // Don't show file locations if their name contains a filter.
      return llvm::none_of(filters, [&](StringRef filter) {
        return fileLoc.getFilename().strref().contains(filter);
      });
    };
    llvm::SourceMgr sourceMgr;
    SourceMgrDiagnosticHandler handler(sourceMgr, &getContext(), llvm::errs(),
                                       filterFn);

    // Emit a diagnostic for every operation with a valid loc.
    getOperation()->walk([&](Operation *op) {
      if (LocationAttr locAttr = op->getAttrOfType<LocationAttr>("test.loc"))
        emitError(locAttr, "test diagnostic");
    });
  }

  ListOption<std::string> filters{
      *this, "filters", llvm::cl::MiscFlags::CommaSeparated,
      llvm::cl::desc("Specifies the diagnostic file name filters.")};
};

} // end anonymous namespace

namespace mlir {
namespace test {
void registerTestDiagnosticsPass() {
  PassRegistration<TestDiagnosticFilterPass>{};
}
} // namespace test
} // namespace mlir
