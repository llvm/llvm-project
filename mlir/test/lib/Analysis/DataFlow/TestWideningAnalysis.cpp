//===- TestWideningAnalysis.cpp - Test merge-site widening hook ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Exercises DataFlowSolver::enableWidening across the four sparse/dense x
// forward/backward analysis variants. Each driver uses the same infinite-
// height `CounterValue` lattice (max-merge, +1 per transfer), so loops
// ratchet the counter without bound until widening caps it. The widen
// callback flips a `widened` bit. Emits the final lattice value for each
// "tag"-annotated op so FileCheck can distinguish tight-bounded vs widened
// outcomes.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::dataflow;

namespace {

/// Budget deliberately small so loop tests trip widening quickly. Acyclic
/// control flow stays well below it.
static constexpr unsigned kTestBudget = 4;

//===----------------------------------------------------------------------===//
// Counter value
//===----------------------------------------------------------------------===//

/// Infinite-height scalar counter. Merge = max on the count; the `widened`
/// bit saturates (once set, stays set and the count freezes). Without
/// widening, a loop body's transfer ratchets the count by +1 on every
/// revisit and never converges; with widening, the bit flips and the value
/// becomes idempotent.
struct CounterValue {
  int count = 0;
  bool widened = false;

  bool operator==(const CounterValue &rhs) const {
    return count == rhs.count && widened == rhs.widened;
  }

  static CounterValue getWidened() {
    return CounterValue{std::numeric_limits<int>::max(), true};
  }

  static CounterValue join(const CounterValue &lhs, const CounterValue &rhs) {
    if (lhs.widened || rhs.widened)
      return getWidened();
    return CounterValue{std::max(lhs.count, rhs.count), false};
  }

  /// For sparse backward we want the same max semantics as forward.
  static CounterValue meet(const CounterValue &lhs, const CounterValue &rhs) {
    return join(lhs, rhs);
  }

  void print(raw_ostream &os) const {
    if (widened)
      os << "widened";
    else
      os << "count=" << count;
  }
};

//===----------------------------------------------------------------------===//
// Sparse lattice
//===----------------------------------------------------------------------===//

class CounterSparseLattice : public Lattice<CounterValue> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CounterSparseLattice)
  using Lattice::Lattice;
};

/// Register a widen callback for the sparse lattice: force to `widened`.
static void enableSparseWidening(DataFlowSolver &solver) {
  solver.enableWidening<CounterSparseLattice>(
      kTestBudget, [](AnalysisState *state) -> ChangeResult {
        auto *lattice = static_cast<CounterSparseLattice *>(state);
        return lattice->join(CounterValue::getWidened());
      });
}

//===----------------------------------------------------------------------===//
// Dense lattice
//===----------------------------------------------------------------------===//

class CounterDenseLattice : public AbstractDenseLattice {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CounterDenseLattice)
  using AbstractDenseLattice::AbstractDenseLattice;

  ChangeResult join(const AbstractDenseLattice &rhs) override {
    const CounterValue &rv =
        static_cast<const CounterDenseLattice &>(rhs).value;
    return set(CounterValue::join(value, rv));
  }

  ChangeResult meet(const AbstractDenseLattice &rhs) override {
    return join(rhs);
  }

  ChangeResult set(const CounterValue &newValue) {
    if (value == newValue)
      return ChangeResult::NoChange;
    value = newValue;
    return ChangeResult::Change;
  }

  const CounterValue &getValue() const { return value; }

  void print(raw_ostream &os) const override { value.print(os); }

private:
  CounterValue value;
};

static void enableDenseWidening(DataFlowSolver &solver) {
  solver.enableWidening<CounterDenseLattice>(
      kTestBudget, [](AnalysisState *state) -> ChangeResult {
        auto *lattice = static_cast<CounterDenseLattice *>(state);
        return lattice->set(CounterValue::getWidened());
      });
}

//===----------------------------------------------------------------------===//
// Sparse forward driver
//===----------------------------------------------------------------------===//

class CounterSparseForwardAnalysis
    : public SparseForwardDataFlowAnalysis<CounterSparseLattice> {
public:
  explicit CounterSparseForwardAnalysis(DataFlowSolver &solver)
      : SparseForwardDataFlowAnalysis(solver) {
    enableSparseWidening(solver);
  }

  LogicalResult
  visitOperation(Operation *, ArrayRef<const CounterSparseLattice *> operands,
                 ArrayRef<CounterSparseLattice *> results) override {
    CounterValue out;
    for (const CounterSparseLattice *operand : operands)
      out = CounterValue::join(out, operand->getValue());
    if (!out.widened)
      out.count += 1;
    for (CounterSparseLattice *result : results)
      propagateIfChanged(result, result->join(out));
    return success();
  }

  void setToEntryState(CounterSparseLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(CounterValue{}));
  }
};

//===----------------------------------------------------------------------===//
// Sparse backward driver
//===----------------------------------------------------------------------===//

class CounterSparseBackwardAnalysis
    : public SparseBackwardDataFlowAnalysis<CounterSparseLattice> {
public:
  CounterSparseBackwardAnalysis(DataFlowSolver &solver,
                                SymbolTableCollection &symbols)
      : SparseBackwardDataFlowAnalysis(solver, symbols) {
    enableSparseWidening(solver);
  }

  LogicalResult
  visitOperation(Operation *op, ArrayRef<CounterSparseLattice *> operands,
                 ArrayRef<const CounterSparseLattice *> results) override {
    CounterValue in;
    for (const CounterSparseLattice *result : results)
      in = CounterValue::join(in, result->getValue());
    if (!in.widened)
      in.count += 1;
    // Route through the framework's `meet` helper so the widening hook fires.
    // `meet` only reads rhs, so a stack-allocated throwaway lattice with an
    // arbitrary anchor is sufficient.
    for (CounterSparseLattice *operand : operands) {
      CounterSparseLattice tmp(operand->getAnchor());
      (void)tmp.join(in);
      meet(operand, tmp);
    }
    return success();
  }

  void visitBranchOperand(OpOperand &) override {}
  void visitCallOperand(OpOperand &) override {}
  void visitNonControlFlowArguments(RegionSuccessor &,
                                    ArrayRef<BlockArgument>) override {}

  void setToExitState(CounterSparseLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(CounterValue{}));
  }
};

//===----------------------------------------------------------------------===//
// Dense forward driver
//===----------------------------------------------------------------------===//

class CounterDenseForwardAnalysis
    : public DenseForwardDataFlowAnalysis<CounterDenseLattice> {
public:
  explicit CounterDenseForwardAnalysis(DataFlowSolver &solver)
      : DenseForwardDataFlowAnalysis(solver) {
    enableDenseWidening(solver);
  }

  LogicalResult visitOperation(Operation *, const CounterDenseLattice &before,
                               CounterDenseLattice *after) override {
    CounterValue v = before.getValue();
    if (!v.widened)
      v.count += 1;
    // Route through the framework's `join` helper so the widening hook can
    // fire on the transfer update. `join` at merge sites is the only hook
    // the framework exposes for widening; for a max-monotone lattice,
    // joining-in the computed value is equivalent to assigning it.
    CounterDenseLattice tmp(after->getAnchor());
    (void)tmp.set(v);
    join(after, tmp);
    return success();
  }

  void setToEntryState(CounterDenseLattice *lattice) override {
    propagateIfChanged(lattice, lattice->set(CounterValue{}));
  }
};

//===----------------------------------------------------------------------===//
// Dense backward driver
//===----------------------------------------------------------------------===//

class CounterDenseBackwardAnalysis
    : public DenseBackwardDataFlowAnalysis<CounterDenseLattice> {
public:
  CounterDenseBackwardAnalysis(DataFlowSolver &solver,
                               SymbolTableCollection &symbols)
      : DenseBackwardDataFlowAnalysis(solver, symbols) {
    enableDenseWidening(solver);
  }

  LogicalResult visitOperation(Operation *, const CounterDenseLattice &after,
                               CounterDenseLattice *before) override {
    CounterValue v = after.getValue();
    if (!v.widened)
      v.count += 1;
    CounterDenseLattice tmp(before->getAnchor());
    (void)tmp.set(v);
    meet(before, tmp);
    return success();
  }

  void setToExitState(CounterDenseLattice *lattice) override {
    propagateIfChanged(lattice, lattice->set(CounterValue{}));
  }
};

//===----------------------------------------------------------------------===//
// Test pass
//===----------------------------------------------------------------------===//

enum class WideningVariant {
  SparseForward,
  SparseBackward,
  DenseForward,
  DenseBackward,
};

struct TestWideningAnalysisPass
    : public PassWrapper<TestWideningAnalysisPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestWideningAnalysisPass)

  TestWideningAnalysisPass() = default;
  TestWideningAnalysisPass(const TestWideningAnalysisPass &other)
      : PassWrapper(other) {
    variant = other.variant;
  }

  StringRef getArgument() const override { return "test-widening-analysis"; }
  StringRef getDescription() const override {
    return "Exercise DataFlowSolver::enableWidening across the four analysis "
           "variants; reflect the resulting counter value.";
  }

  Option<WideningVariant> variant{
      *this, "variant", llvm::cl::desc("Which analysis variant to run"),
      llvm::cl::init(WideningVariant::SparseForward),
      llvm::cl::values(clEnumValN(WideningVariant::SparseForward,
                                  "sparse-forward", "Sparse forward"),
                       clEnumValN(WideningVariant::SparseBackward,
                                  "sparse-backward", "Sparse backward"),
                       clEnumValN(WideningVariant::DenseForward,
                                  "dense-forward", "Dense forward"),
                       clEnumValN(WideningVariant::DenseBackward,
                                  "dense-backward", "Dense backward"))};

  void runOnOperation() override {
    Operation *top = getOperation();
    DataFlowSolver solver;
    loadBaselineAnalyses(solver);
    SymbolTableCollection symbols;

    switch (variant) {
    case WideningVariant::SparseForward:
      solver.load<CounterSparseForwardAnalysis>();
      break;
    case WideningVariant::SparseBackward:
      solver.load<CounterSparseBackwardAnalysis>(symbols);
      break;
    case WideningVariant::DenseForward:
      solver.load<CounterDenseForwardAnalysis>();
      break;
    case WideningVariant::DenseBackward:
      solver.load<CounterDenseBackwardAnalysis>(symbols);
      break;
    }

    if (failed(solver.initializeAndRun(top)))
      return signalPassFailure();

    raw_ostream &os = llvm::outs();
    top->walk([&](Operation *op) {
      auto tag = op->getAttrOfType<StringAttr>("tag");
      if (!tag)
        return;
      os << "tag=" << tag.getValue() << ": ";
      switch (variant) {
      case WideningVariant::SparseForward:
        printSparseResults(os, solver, op);
        break;
      case WideningVariant::SparseBackward:
        printSparseOperands(os, solver, op);
        break;
      case WideningVariant::DenseForward:
        printDenseAt(os, solver, solver.getProgramPointAfter(op));
        break;
      case WideningVariant::DenseBackward:
        printDenseAt(os, solver, solver.getProgramPointBefore(op));
        break;
      }
      os << "\n";
    });
  }

private:
  static void printSparseResults(raw_ostream &os, DataFlowSolver &solver,
                                 Operation *op) {
    os << "results=[";
    llvm::interleaveComma(op->getResults(), os, [&](Value v) {
      if (const auto *lat = solver.lookupState<CounterSparseLattice>(v))
        lat->getValue().print(os);
      else
        os << "<unset>";
    });
    os << "]";
  }

  static void printSparseOperands(raw_ostream &os, DataFlowSolver &solver,
                                  Operation *op) {
    os << "operands=[";
    llvm::interleaveComma(op->getOperands(), os, [&](Value v) {
      if (const auto *lat = solver.lookupState<CounterSparseLattice>(v))
        lat->getValue().print(os);
      else
        os << "<unset>";
    });
    os << "]";
  }

  static void printDenseAt(raw_ostream &os, DataFlowSolver &solver,
                           ProgramPoint *point) {
    if (const auto *lat = solver.lookupState<CounterDenseLattice>(point))
      lat->getValue().print(os);
    else
      os << "<unset>";
  }
};

} // end anonymous namespace

namespace mlir {
namespace test {
void registerTestWideningAnalysisPass() {
  PassRegistration<TestWideningAnalysisPass>();
}
} // namespace test
} // namespace mlir
