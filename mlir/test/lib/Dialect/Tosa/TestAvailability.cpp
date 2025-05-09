//===- TestAvailability.cpp - Pass to test Tosa op availability ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Printing op availability pass
//===----------------------------------------------------------------------===//

namespace {
/// A pass for testing Tosa op availability.
struct PrintOpAvailability
    : public PassWrapper<PrintOpAvailability, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrintOpAvailability)

  void runOnOperation() override;
  StringRef getArgument() const final { return "test-tosa-op-availability"; }
  StringRef getDescription() const final { return "Test Tosa op availability"; }
};
} // namespace

void PrintOpAvailability::runOnOperation() {
  auto f = getOperation();
  llvm::outs() << f.getName() << "\n";

  Dialect *tosaDialect = getContext().getLoadedDialect("tosa");

  f->walk([&](Operation *op) {
    if (op->getDialect() != tosaDialect)
      return WalkResult::advance();

    auto opName = op->getName();
    auto &os = llvm::outs();

    if (auto profile = dyn_cast<tosa::QueryProfileInterface>(op)) {
      os << opName << " profiles: [";
      for (const auto &profs : profile.getProfiles()) {
        os << " [";
        llvm::interleaveComma(profs, os, [&](tosa::Profile prof) {
          os << tosa::stringifyProfile(prof);
        });
        os << "]";
      }
      os << " ]\n";
    }

    if (auto extension = dyn_cast<tosa::QueryExtensionInterface>(op)) {
      os << opName << " extensions: [";
      for (const auto &exts : extension.getExtensions()) {
        os << " [";
        llvm::interleaveComma(exts, os, [&](tosa::Extension ext) {
          os << tosa::stringifyExtension(ext);
        });
        os << "]";
      }
      os << " ]\n";
    }

    os.flush();

    return WalkResult::advance();
  });
}

namespace mlir {
void registerPrintTosaAvailabilityPass() {
  PassRegistration<PrintOpAvailability>();
}
} // namespace mlir
