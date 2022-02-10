//===- Lifetimecheck.cpp - emit diagnostic checks for lifetime violations -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/CIR/Passes.h"

#include "PassDetail.h"
#include "mlir/Dialect/CIR/IR/CIRDialect.h"

using namespace mlir;

namespace {
struct LifetimeCheckPass : public LifetimeCheckBase<LifetimeCheckPass> {
  explicit LifetimeCheckPass(raw_ostream &os = llvm::errs()) : os(os) {}

  // Prints the resultant operation statistics post iterating over the module.
  void runOnOperation() override;

  // Print lifetime diagnostics
  void printDiagnostics();

private:
  raw_ostream &os;
};
} // namespace

void LifetimeCheckPass::runOnOperation() { printDiagnostics(); }
void LifetimeCheckPass::printDiagnostics() { os << "Hello Lifetime World\n"; }

std::unique_ptr<Pass> mlir::createLifetimeCheckPass() {
  return std::make_unique<LifetimeCheckPass>();
}