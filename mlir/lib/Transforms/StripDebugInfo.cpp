//===- StripDebugInfo.cpp - Pass to strip debug information ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
struct StripDebugInfo : public StripDebugInfoBase<StripDebugInfo> {
  void runOnOperation() override;
};
} // end anonymous namespace

void StripDebugInfo::runOnOperation() {
  // Strip the debug info from all operations.
  auto unknownLoc = UnknownLoc::get(&getContext());
  getOperation()->walk([&](Operation *op) { op->setLoc(unknownLoc); });
}

/// Creates a pass to strip debug information from a function.
std::unique_ptr<Pass> mlir::createStripDebugInfoPass() {
  return std::make_unique<StripDebugInfo>();
}
