//===- SCCP.cpp - Sparse Conditional Constant Propagation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass performs a sparse conditional constant propagation
// in MLIR. It identifies values known to be constant, propagates that
// information throughout the IR, and replaces them. This is done with an
// optimistic dataflow analysis that assumes that all values are constant until
// proven otherwise.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Passes.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/FoldUtils.h"

namespace mlir {
#define GEN_PASS_DEF_SCCPPASS
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::dataflow;

//===----------------------------------------------------------------------===//
// SCCP Rewrites
//===----------------------------------------------------------------------===//

/// Replace the given value with a constant if the corresponding lattice
/// represents a constant. Returns success if the value was replaced, failure
/// otherwise.
static LogicalResult replaceWithConstant(DataFlowSolver &solver,
                                         OpBuilder &builder,
                                         OperationFolder &folder, Value value) {
  auto *lattice = solver.lookupState<Lattice<ConstantValue>>(value);
  if (!lattice || lattice->isUninitialized())
    return failure();
  const ConstantValue &latticeValue = lattice->getValue();
  if (!latticeValue.getConstantValue())
    return failure();

  // Attempt to materialize a constant for the given value.
  Dialect *dialect = latticeValue.getConstantDialect();
  Value constant = folder.getOrCreateConstant(builder, dialect,
                                              latticeValue.getConstantValue(),
                                              value.getType(), value.getLoc());
  if (!constant)
    return failure();

  value.replaceAllUsesWith(constant);
  return success();
}

/// Rewrite the given regions using the computing analysis. This replaces the
/// uses of all values that have been computed to be constant, and erases as
/// many newly dead operations.
static void rewrite(DataFlowSolver &solver, MLIRContext *context,
                    MutableArrayRef<Region> initialRegions) {
  SmallVector<Block *> worklist;
  auto addToWorklist = [&](MutableArrayRef<Region> regions) {
    for (Region &region : regions)
      for (Block &block : llvm::reverse(region))
        worklist.push_back(&block);
  };

  // An operation folder used to create and unique constants.
  OperationFolder folder(context);
  OpBuilder builder(context);

  addToWorklist(initialRegions);
  while (!worklist.empty()) {
    Block *block = worklist.pop_back_val();

    for (Operation &op : llvm::make_early_inc_range(*block)) {
      builder.setInsertionPoint(&op);

      // Replace any result with constants.
      bool replacedAll = op.getNumResults() != 0;
      for (Value res : op.getResults())
        replacedAll &=
            succeeded(replaceWithConstant(solver, builder, folder, res));

      // If all of the results of the operation were replaced, try to erase
      // the operation completely.
      if (replacedAll && wouldOpBeTriviallyDead(&op)) {
        assert(op.use_empty() && "expected all uses to be replaced");
        op.erase();
        continue;
      }

      // Add any the regions of this operation to the worklist.
      addToWorklist(op.getRegions());
    }

    // Replace any block arguments with constants.
    builder.setInsertionPointToStart(block);
    for (BlockArgument arg : block->getArguments())
      (void)replaceWithConstant(solver, builder, folder, arg);
  }
}

//===----------------------------------------------------------------------===//
// SCCP Pass
//===----------------------------------------------------------------------===//

namespace {
struct SCCPPass : public impl::SCCPPassBase<SCCPPass> {
  using SCCPPassBase::SCCPPassBase;

  void runOnOperation() override;
};
} // namespace

void SCCPPass::runOnOperation() {
  Operation *op = getOperation();

  DataFlowSolver solver;
  solver.load<DeadCodeAnalysis>();
  solver.load<SparseConstantPropagation>();
  if (failed(solver.initializeAndRun(op)))
    return signalPassFailure();
  rewrite(solver, op->getContext(), op->getRegions());
}
