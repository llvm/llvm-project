//===- StripDebugInfo.cpp - Pass to strip debug information ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DEF_STRIPDEBUGINFOPASS
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct StripDebugInfoPass
    : public impl::StripDebugInfoPassBase<StripDebugInfoPass> {
  using StripDebugInfoPassBase::StripDebugInfoPassBase;

  void runOnOperation() override;
};
} // namespace

void StripDebugInfoPass::runOnOperation() {
  auto unknownLoc = UnknownLoc::get(&getContext());

  // Strip the debug info from all operations.
  getOperation()->walk([&](Operation *op) {
    op->setLoc(unknownLoc);
    // Strip block arguments debug info.
    for (Region &region : op->getRegions()) {
      for (Block &block : region.getBlocks()) {
        for (BlockArgument &arg : block.getArguments()) {
          arg.setLoc(unknownLoc);
        }
      }
    }
  });
}
