//===- TestDataFlowFramework.cpp - Test data-flow analysis framework ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include <optional>

using namespace mlir;

namespace {
constexpr char kTagAttrName[] = "tag";
constexpr char kFooAttrName[] = "foo";
constexpr char kFooStateAttrName[] = "foo_state";
constexpr char kBarStateAttrName[] = "bar_state";

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

  /// Join the state with another. If either is uninitialized, take the
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

  static bool classof(const DataFlowAnalysis *a) {
    return a->getTypeID() == TypeID::get<FooAnalysis>();
  }

  LogicalResult initialize(Operation *top) override;
  LogicalResult visit(ProgramPoint *point) override;

private:
  void visitBlock(Block *block);
  void visitOperation(Operation *op);
};

/// This analysis state stores whether all previously observed `FooState`
/// values at tagged program points along the CFG leading to the current point
/// have been non-multiples of 4. Once the state becomes false at some point,
/// all later points reachable from it also remain false.
class BarState : public AnalysisState {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BarState)

  using AnalysisState::AnalysisState;

  bool isUninitialized() const { return !state; }

  void print(raw_ostream &os) const override {
    if (!state) {
      os << "none";
      return;
    }
    os << (*state ? "true" : "false");
  }

  ChangeResult join(const BarState &rhs) {
    if (rhs.isUninitialized())
      return ChangeResult::NoChange;
    return join(rhs.getValue());
  }

  ChangeResult join(bool value) {
    if (isUninitialized()) {
      state = value;
      return ChangeResult::Change;
    }
    bool newValue = *state && value;
    if (newValue == *state)
      return ChangeResult::NoChange;
    state = newValue;
    return ChangeResult::Change;
  }

  bool getValue() const { return *state; }

private:
  std::optional<bool> state;
};

/// This analysis is intended to be loaded after `FooAnalysis` has converged.
/// It records whether every observed `FooState` on or before a given tagged
/// program point has been non-divisible by 4. Because the state only ever
/// transitions from true to false, observing a transient divisible-by-4
/// `FooState` before `FooAnalysis` converges can permanently poison the
/// result.
class BarAnalysis : public DataFlowAnalysis {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BarAnalysis)

  using DataFlowAnalysis::DataFlowAnalysis;

  static bool classof(const DataFlowAnalysis *a) {
    return a->getTypeID() == TypeID::get<BarAnalysis>();
  }

  LogicalResult initialize(Operation *top) override;
  LogicalResult visit(ProgramPoint *point) override;

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

struct TestStagedAnalysesPass
    : public PassWrapper<TestStagedAnalysesPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestStagedAnalysesPass)

  StringRef getArgument() const override { return "test-staged-analyses"; }

  void runOnOperation() override;
};
} // namespace

LogicalResult FooAnalysis::initialize(Operation *top) {
  if (top->getNumRegions() != 1)
    return top->emitError("expected a single region top-level op");

  if (top->getRegion(0).getBlocks().empty())
    return top->emitError("expected at least one block in the region");

  // Initialize the top-level state.
  (void)getOrCreate<FooState>(getProgramPointBefore(&top->getRegion(0).front()))
      ->join(0);

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

LogicalResult FooAnalysis::visit(ProgramPoint *point) {
  if (!point->isBlockStart())
    visitOperation(point->getPrevOp());
  else
    visitBlock(point->getBlock());
  return success();
}

void FooAnalysis::visitBlock(Block *block) {
  if (block->isEntryBlock()) {
    // This is the initial state. Let the framework default-initialize it.
    return;
  }
  ProgramPoint *point = getProgramPointBefore(block);
  FooState *state = getOrCreate<FooState>(point);
  ChangeResult result = ChangeResult::NoChange;
  for (Block *pred : block->getPredecessors()) {
    // Join the state at the terminators of all predecessors.
    const FooState *predState = getOrCreateFor<FooState>(
        point, getProgramPointAfter(pred->getTerminator()));
    result |= state->join(*predState);
  }
  propagateIfChanged(state, result);
}

void FooAnalysis::visitOperation(Operation *op) {
  ProgramPoint *point = getProgramPointAfter(op);
  FooState *state = getOrCreate<FooState>(point);
  ChangeResult result = ChangeResult::NoChange;

  // Copy the state across the operation.
  const FooState *prevState;
  prevState = getOrCreateFor<FooState>(point, getProgramPointBefore(op));
  result |= state->set(*prevState);

  // Modify the state with the attribute, if specified.
  if (auto attr = op->getAttrOfType<IntegerAttr>(kFooAttrName)) {
    uint64_t value = attr.getUInt();
    result |= state->join(value);
  }
  propagateIfChanged(state, result);
}

LogicalResult BarAnalysis::initialize(Operation *top) {
  if (top->getNumRegions() != 1)
    return top->emitError("expected a single region top-level op");

  if (top->getRegion(0).getBlocks().empty())
    return top->emitError("expected at least one block in the region");

  // Seed the entry state to true before observing any `FooState`.
  (void)getOrCreate<BarState>(getProgramPointBefore(&top->getRegion(0).front()))
      ->join(true);

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

LogicalResult BarAnalysis::visit(ProgramPoint *point) {
  if (!point->isBlockStart())
    visitOperation(point->getPrevOp());
  else
    visitBlock(point->getBlock());
  return success();
}

void BarAnalysis::visitBlock(Block *block) {
  if (block->isEntryBlock())
    return;

  ProgramPoint *point = getProgramPointBefore(block);
  BarState *state = getOrCreate<BarState>(point);
  ChangeResult result = ChangeResult::NoChange;
  for (Block *pred : block->getPredecessors()) {
    const BarState *predState = getOrCreateFor<BarState>(
        point, getProgramPointAfter(pred->getTerminator()));
    result |= state->join(*predState);
  }
  propagateIfChanged(state, result);
}

void BarAnalysis::visitOperation(Operation *op) {
  ProgramPoint *point = getProgramPointAfter(op);
  BarState *state = getOrCreate<BarState>(point);
  ChangeResult result = ChangeResult::NoChange;

  const BarState *prevState =
      getOrCreateFor<BarState>(point, getProgramPointBefore(op));
  result |= state->join(*prevState);

  if (op->hasAttr(kTagAttrName)) {
    const FooState *fooState = getOrCreateFor<FooState>(point, point);
    if (fooState->isUninitialized())
      return;
    result |= state->join((fooState->getValue() & 0x3) != 0);
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
    auto tag = op->getAttrOfType<StringAttr>(kTagAttrName);
    if (!tag)
      return;
    const FooState *state =
        solver.lookupState<FooState>(solver.getProgramPointAfter(op));
    assert(state && !state->isUninitialized());
    os << tag.getValue() << " -> " << state->getValue() << "\n";
  });
}

void TestStagedAnalysesPass::runOnOperation() {
  func::FuncOp func = getOperation();
  Builder builder(func.getContext());

  DataFlowSolver solver;
  solver.load<FooAnalysis>();
  if (failed(solver.initializeAndRun(func)))
    return signalPassFailure();
  solver.load<BarAnalysis>();
  if (failed(solver.initializeAndRun(func, llvm::IsaPred<BarAnalysis>)))
    return signalPassFailure();

  func.walk([&](Operation *op) {
    if (!op->hasAttr(kTagAttrName))
      return;

    ProgramPoint *point = solver.getProgramPointAfter(op);
    const FooState *fooState = solver.lookupState<FooState>(point);
    const BarState *barState = solver.lookupState<BarState>(point);
    assert(fooState && !fooState->isUninitialized());
    assert(barState && !barState->isUninitialized());

    op->setAttr(kFooStateAttrName,
                builder.getI64IntegerAttr(fooState->getValue()));
    op->setAttr(kBarStateAttrName, builder.getBoolAttr(barState->getValue()));
  });
}

namespace mlir {
namespace test {
void registerTestFooAnalysisPass() { PassRegistration<TestFooAnalysisPass>(); }
void registerTestStagedAnalysesPass() {
  PassRegistration<TestStagedAnalysesPass>();
}
} // namespace test
} // namespace mlir
