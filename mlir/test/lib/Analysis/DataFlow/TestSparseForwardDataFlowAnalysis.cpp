//===- TestForwardDataFlowAnalysis.cpp - Test dead code analysis ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::dataflow;

namespace {

class IntegerState {
public:
  IntegerState() : value(0) {}
  explicit IntegerState(int value) : value(value) {}
  ~IntegerState() = default;

  int get() const { return value; }

  bool operator==(const IntegerState &rhs) const { return value == rhs.value; }

  static IntegerState join(const IntegerState &lhs, const IntegerState &rhs) {
    return IntegerState{std::max(lhs.get(), rhs.get())};
  }

  void print(llvm::raw_ostream &os) const {
    os << "IntegerState(" << value << ")";
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const IntegerState &state) {
    state.print(os);
    return os;
  }

private:
  int value;
};

/// This lattice represents, for a given value, the set of memory resources that
/// this value, or anything derived from this value, is potentially written to.
struct IntegerLattice : public Lattice<IntegerState> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IntegerLattice)
  using Lattice::Lattice;
};

/// An analysis that, by going backwards along the dataflow graph, annotates
/// each value with all the memory resources it (or anything derived from it)
/// is eventually written to.
class IntegerLatticeAnalysis
    : public SparseForwardDataFlowAnalysis<IntegerLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const IntegerLattice *> operands,
                               ArrayRef<IntegerLattice *> results) override;

  void setToEntryState(IntegerLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(IntegerState()));
  }
};

LogicalResult IntegerLatticeAnalysis::visitOperation(
    Operation *op, ArrayRef<const IntegerLattice *> operands,
    ArrayRef<IntegerLattice *> results) {
  for (auto *operand : operands) {
    for (auto *result : results) {
      propagateIfChanged(result, result->join(*operand));
    }
  }
  return success();
}

} // end anonymous namespace

namespace {
struct TestIntegerLatticePass
    : public PassWrapper<TestIntegerLatticePass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestIntegerLatticePass)

  TestIntegerLatticePass() = default;
  TestIntegerLatticePass(const TestIntegerLatticePass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override { return "test-integer-lattice"; }

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = &getContext();

    DataFlowSolver solver;
    solver.load<DeadCodeAnalysis>();
    solver.load<SparseConstantPropagation>();
    solver.load<IntegerLatticeAnalysis>();
    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();

    // Walk the IR and attach operand and result lattices as attributes to each
    // operation.
    op->walk([&](Operation *op) {
      SmallVector<Attribute> operandAttrs;
      SmallVector<Attribute> resultAttrs;
      for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
        const IntegerLattice *lattice =
            solver.lookupState<IntegerLattice>(operand);
        assert(lattice && "expected a sparse lattice");
        operandAttrs.push_back(
            IntegerAttr::get(IndexType::get(ctx), lattice->getValue().get()));
      }
      for (auto [index, result] : llvm::enumerate(op->getResults())) {
        const IntegerLattice *lattice =
            solver.lookupState<IntegerLattice>(result);
        assert(lattice && "expected a sparse lattice");
        resultAttrs.push_back(
            IntegerAttr::get(IndexType::get(ctx), lattice->getValue().get()));
      }

      op->setAttr("test.operand_lattices", ArrayAttr::get(ctx, operandAttrs));
      op->setAttr("test.result_lattices", ArrayAttr::get(ctx, resultAttrs));
    });
  }
};
} // end anonymous namespace

namespace mlir {
namespace test {
void registerTestIntegerLatticePass() {
  PassRegistration<TestIntegerLatticePass>();
}
} // end namespace test
} // end namespace mlir
