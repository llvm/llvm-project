//===- TestIntRangeInference.cpp - Create consts from range inference ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// TODO: This pass is needed to test integer range inference until that
// functionality has been integrated into SCCP.
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/FoldUtils.h"

using namespace mlir;
using namespace mlir::dataflow;

/// Patterned after SCCP
static LogicalResult replaceWithConstant(DataFlowSolver &solver, OpBuilder &b,
                                         OperationFolder &folder, Value value) {
  auto *maybeInferredRange =
      solver.lookupState<IntegerValueRangeLattice>(value);
  if (!maybeInferredRange || maybeInferredRange->isUninitialized())
    return failure();
  const ConstantIntRanges &inferredRange =
      maybeInferredRange->getValue().getValue();
  Optional<APInt> maybeConstValue = inferredRange.getConstantValue();
  if (!maybeConstValue.hasValue())
    return failure();

  Operation *maybeDefiningOp = value.getDefiningOp();
  Dialect *valueDialect =
      maybeDefiningOp ? maybeDefiningOp->getDialect()
                      : value.getParentRegion()->getParentOp()->getDialect();
  Attribute constAttr = b.getIntegerAttr(value.getType(), *maybeConstValue);
  Value constant = folder.getOrCreateConstant(b, valueDialect, constAttr,
                                              value.getType(), value.getLoc());
  if (!constant)
    return failure();

  value.replaceAllUsesWith(constant);
  return success();
}

static void rewrite(DataFlowSolver &solver, MLIRContext *context,
                    MutableArrayRef<Region> initialRegions) {
  SmallVector<Block *> worklist;
  auto addToWorklist = [&](MutableArrayRef<Region> regions) {
    for (Region &region : regions)
      for (Block &block : llvm::reverse(region))
        worklist.push_back(&block);
  };

  OpBuilder builder(context);
  OperationFolder folder(context);

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

namespace {
struct TestIntRangeInference
    : PassWrapper<TestIntRangeInference, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestIntRangeInference)

  StringRef getArgument() const final { return "test-int-range-inference"; }
  StringRef getDescription() const final {
    return "Test integer range inference analysis";
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    DataFlowSolver solver;
    solver.load<DeadCodeAnalysis>();
    solver.load<IntegerRangeAnalysis>();
    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();
    rewrite(solver, op->getContext(), op->getRegions());
  }
};
} // end anonymous namespace

namespace mlir {
namespace test {
void registerTestIntRangeInference() {
  PassRegistration<TestIntRangeInference>();
}
} // end namespace test
} // end namespace mlir
