//===- PrintIR.cpp - Pass to dump IR on debug stream ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace {

#define GEN_PASS_DEF_PRINTIRPASS
#include "mlir/Transforms/Passes.h.inc"

struct PrintIRPass : public impl::PrintIRPassBase<PrintIRPass> {
  PrintIRPass() = default;

  void runOnOperation() override { getOperation()->dump(); }
};

} // namespace

std::unique_ptr<Pass> createPrintIRPass() {
  return std::make_unique<PrintIRPass>();
}

} // namespace mlir