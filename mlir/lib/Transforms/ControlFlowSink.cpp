//===- ControlFlowSink.cpp - Code to perform control-flow sinking ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a basic control-flow sink pass. Control-flow sinking
// moves operations whose only uses are in conditionally-executed blocks in to
// those blocks so that they aren't executed on paths where their results are
// not needed.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/ControlFlowSinkUtils.h"

namespace mlir {
#define GEN_PASS_DEF_CONTROLFLOWSINKPASS
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
/// A control-flow sink pass.
struct ControlFlowSink : public impl::ControlFlowSinkPassBase<ControlFlowSink> {
  void runOnOperation() override;
};
} // end anonymous namespace

void ControlFlowSink::runOnOperation() {
  auto &domInfo = getAnalysis<DominanceInfo>();
  getOperation()->walk([&](RegionBranchOpInterface branch) {
    SmallVector<Region *> regionsToSink;
    // Get the regions are that known to be executed at most once.
    getSinglyExecutedRegionsToSink(branch, regionsToSink);
    // Sink memory-effect-free operations.
    numSunk = controlFlowSink(
        regionsToSink, domInfo,
        [](Operation *op, Region *) {
          if (!isMemoryEffectFree(op))
            return false;
          // A non-pure operation whose result is currently a valid affine
          // dimension or symbol must remain at the top level of its affine
          // scope to stay valid. Sinking such an operation into a
          // conditionally executed region moves its definition into a nested
          // region, breaking that invariant and producing invalid IR (e.g.
          // `index.remu` used by `affine.load`).
          if (isPure(op))
            return true;
          for (Value result : op->getResults())
            if (affine::isValidDim(result) || affine::isValidSymbol(result))
              return false;
          return true;
        },
        [](Operation *op, Region *region) {
          // Move the operation to the beginning of the region's entry block.
          // This guarantees the preservation of SSA dominance of all of the
          // operation's uses are in the region.
          op->moveBefore(&region->front(), region->front().begin());
        });
  });
}
