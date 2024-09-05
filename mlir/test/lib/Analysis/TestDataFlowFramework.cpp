//===- TestDataFlowFramework.cpp - Test data-flow analysis framework ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include <optional>

using namespace mlir;

namespace {
/// This analysis state represents an integer that is XOR'd with other states.
class FooState : public AnalysisState {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FooState)

  using AnalysisState::AnalysisState;

  /// Returns true if the state is uninitialized.
  bool isUninitialized() const { return !state; }

  /// Print the integer value or "none" if uninitialized.
  void print(raw_ostream &os) const override {
    if (state)
      os << *state;
    else
      os << "none";
  }

  /// Join the state with another. If either is unintialized, take the
  /// initialized value. Otherwise, XOR the integer values.
  ChangeResult join(const FooState &rhs) {
    if (rhs.isUninitialized())
      return ChangeResult::NoChange;
    return join(*rhs.state);
  }
  ChangeResult join(uint64_t value) {
    if (isUninitialized()) {
      state = value;
      return ChangeResult::Change;
    }
    uint64_t before = *state;
    state = before ^ value;
    return before == *state ? ChangeResult::NoChange : ChangeResult::Change;
  }

  /// Set the value of the state directly.
  ChangeResult set(const FooState &rhs) {
    if (state == rhs.state)
      return ChangeResult::NoChange;
    state = rhs.state;
    return ChangeResult::Change;
  }

  /// Returns the integer value of the state.
  uint64_t getValue() const { return *state; }

private:
  /// An optional integer value.
  std::optional<uint64_t> state;
};

/// This analysis computes `FooState` across operations and control-flow edges.
/// If an op specifies a `foo` integer attribute, the contained value is XOR'd
/// with the value before the operation.
class FooAnalysis : public DataFlowAnalysis {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FooAnalysis)

  using DataFlowAnalysis::DataFlowAnalysis;

  LogicalResult initialize(Operation *top) override;
  LogicalResult visit(ProgramPoint point) override;

private:
  void visitBlock(Block *block);
  void visitOperation(Operation *op);
};

struct TestFooAnalysisPass
    : public PassWrapper<TestFooAnalysisPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestFooAnalysisPass)

  StringRef getArgument() const override { return "test-foo-analysis"; }

  void runOnOperation() override;
};
} // namespace

LogicalResult FooAnalysis::initialize(Operation *top) {
  if (top->getNumRegions() != 1)
    return top->emitError("expected a single region top-level op");

  if (top->getRegion(0).getBlocks().empty())
    return top->emitError("expected at least one block in the region");

  // Initialize the top-level state.
  (void)getOrCreate<FooState>(&top->getRegion(0).front())->join(0);

  // Visit all nested blocks and operations.
  for (Block &block : top->getRegion(0)) {
    visitBlock(&block);
    for (Operation &op : block) {
      if (op.getNumRegions())
        return op.emitError("unexpected op with regions");
      visitOperation(&op);
    }
  }
  return success();
}

LogicalResult FooAnalysis::visit(ProgramPoint point) {
  if (auto *op = llvm::dyn_cast_if_present<Operation *>(point))
    visitOperation(op);
  else
    visitBlock(point.get<Block *>());
  return success();
}

void FooAnalysis::visitBlock(Block *block) {
  if (block->isEntryBlock()) {
    // This is the initial state. Let the framework default-initialize it.
    return;
  }
  FooState *state = getOrCreate<FooState>(block);
  ChangeResult result = ChangeResult::NoChange;
  for (Block *pred : block->getPredecessors()) {
    // Join the state at the terminators of all predecessors.
    const FooState *predState =
        getOrCreateFor<FooState>(block, pred->getTerminator());
    result |= state->join(*predState);
  }
  propagateIfChanged(state, result);
}

void FooAnalysis::visitOperation(Operation *op) {
  FooState *state = getOrCreate<FooState>(op);
  ChangeResult result = ChangeResult::NoChange;

  // Copy the state across the operation.
  const FooState *prevState;
  if (Operation *prev = op->getPrevNode())
    prevState = getOrCreateFor<FooState>(op, prev);
  else
    prevState = getOrCreateFor<FooState>(op, op->getBlock());
  result |= state->set(*prevState);

  // Modify the state with the attribute, if specified.
  if (auto attr = op->getAttrOfType<IntegerAttr>("foo")) {
    uint64_t value = attr.getUInt();
    result |= state->join(value);
  }
  propagateIfChanged(state, result);
}

void TestFooAnalysisPass::runOnOperation() {
  func::FuncOp func = getOperation();
  DataFlowSolver solver;
  solver.load<FooAnalysis>();
  if (failed(solver.initializeAndRun(func)))
    return signalPassFailure();

  raw_ostream &os = llvm::errs();
  os << "function: @" << func.getSymName() << "\n";

  func.walk([&](Operation *op) {
    auto tag = op->getAttrOfType<StringAttr>("tag");
    if (!tag)
      return;
    const FooState *state = solver.lookupState<FooState>(op);
    assert(state && !state->isUninitialized());
    os << tag.getValue() << " -> " << state->getValue() << "\n";
  });
}

namespace mlir {
namespace test {
void registerTestFooAnalysisPass() { PassRegistration<TestFooAnalysisPass>(); }
} // namespace test
} // namespace mlir
