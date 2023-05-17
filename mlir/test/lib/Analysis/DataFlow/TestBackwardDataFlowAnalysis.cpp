//===- TestBackwardDataFlowAnalysis.cpp - Test dead code analysis ---------===//
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

/// This lattice represents, for a given value, the set of memory resources that
/// this value, or anything derived from this value, is potentially written to.
struct WrittenTo : public AbstractSparseLattice {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(WrittenTo)
  using AbstractSparseLattice::AbstractSparseLattice;

  void print(raw_ostream &os) const override {
    os << "[";
    llvm::interleave(
        writes, os, [&](const StringAttr &a) { os << a.str(); }, " ");
    os << "]";
  }
  ChangeResult addWrites(const SetVector<StringAttr> &writes) {
    int sizeBefore = this->writes.size();
    this->writes.insert(writes.begin(), writes.end());
    int sizeAfter = this->writes.size();
    return sizeBefore == sizeAfter ? ChangeResult::NoChange
                                   : ChangeResult::Change;
  }
  ChangeResult meet(const AbstractSparseLattice &other) override {
    const auto *rhs = reinterpret_cast<const WrittenTo *>(&other);
    return addWrites(rhs->writes);
  }

  SetVector<StringAttr> writes;
};

/// An analysis that, by going backwards along the dataflow graph, annotates
/// each value with all the memory resources it (or anything derived from it)
/// is eventually written to.
class WrittenToAnalysis : public SparseBackwardDataFlowAnalysis<WrittenTo> {
public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  void visitOperation(Operation *op, ArrayRef<WrittenTo *> operands,
                      ArrayRef<const WrittenTo *> results) override;

  void visitBranchOperand(OpOperand &operand) override;

  void setToExitState(WrittenTo *lattice) override { lattice->writes.clear(); }
};

void WrittenToAnalysis::visitOperation(Operation *op,
                                       ArrayRef<WrittenTo *> operands,
                                       ArrayRef<const WrittenTo *> results) {
  if (auto store = dyn_cast<memref::StoreOp>(op)) {
    SetVector<StringAttr> newWrites;
    newWrites.insert(op->getAttrOfType<StringAttr>("tag_name"));
    propagateIfChanged(operands[0], operands[0]->addWrites(newWrites));
    return;
  } // By default, every result of an op depends on every operand.
    for (const WrittenTo *r : results) {
      for (WrittenTo *operand : operands) {
        meet(operand, *r);
      }
      addDependency(const_cast<WrittenTo *>(r), op);
    }
}

void WrittenToAnalysis::visitBranchOperand(OpOperand &operand) {
  // Mark branch operands as "brancharg%d", with %d the operand number.
  WrittenTo *lattice = getLatticeElement(operand.get());
  SetVector<StringAttr> newWrites;
  newWrites.insert(
      StringAttr::get(operand.getOwner()->getContext(),
                      "brancharg" + Twine(operand.getOperandNumber())));
  propagateIfChanged(lattice, lattice->addWrites(newWrites));
}

} // end anonymous namespace

namespace {
struct TestWrittenToPass
    : public PassWrapper<TestWrittenToPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestWrittenToPass)

  StringRef getArgument() const override { return "test-written-to"; }

  void runOnOperation() override {
    Operation *op = getOperation();

    SymbolTableCollection symbolTable;

    DataFlowSolver solver;
    solver.load<DeadCodeAnalysis>();
    solver.load<SparseConstantPropagation>();
    solver.load<WrittenToAnalysis>(symbolTable);
    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();

    raw_ostream &os = llvm::outs();
    op->walk([&](Operation *op) {
      auto tag = op->getAttrOfType<StringAttr>("tag");
      if (!tag)
        return;
      os << "test_tag: " << tag.getValue() << ":\n";
      for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
        const WrittenTo *writtenTo = solver.lookupState<WrittenTo>(operand);
        assert(writtenTo && "expected a sparse lattice");
        os << " operand #" << index << ": ";
        writtenTo->print(os);
        os << "\n";
      }
      for (auto [index, operand] : llvm::enumerate(op->getResults())) {
        const WrittenTo *writtenTo = solver.lookupState<WrittenTo>(operand);
        assert(writtenTo && "expected a sparse lattice");
        os << " result #" << index << ": ";
        writtenTo->print(os);
        os << "\n";
      }
    });
  }
};
} // end anonymous namespace

namespace mlir {
namespace test {
void registerTestWrittenToPass() { PassRegistration<TestWrittenToPass>(); }
} // end namespace test
} // end namespace mlir
