//===- TestDenseBackwardDataFlowAnalysis.cpp - Test pass ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test pass for backward dense dataflow analysis.
//
//===----------------------------------------------------------------------===//

#include "TestDenseDataFlowAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::dataflow;
using namespace mlir::dataflow::test;

namespace {

class NextAccess : public AbstractDenseLattice, public test::AccessLatticeBase {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NextAccess)

  using dataflow::AbstractDenseLattice::AbstractDenseLattice;

  ChangeResult meet(const AbstractDenseLattice &lattice) override {
    return AccessLatticeBase::merge(static_cast<test::AccessLatticeBase>(
        static_cast<const NextAccess &>(lattice)));
  }

  void print(raw_ostream &os) const override {
    return AccessLatticeBase::print(os);
  }
};

class NextAccessAnalysis : public DenseBackwardDataFlowAnalysis<NextAccess> {
public:
  using DenseBackwardDataFlowAnalysis::DenseBackwardDataFlowAnalysis;

  void visitOperation(Operation *op, const NextAccess &after,
                      NextAccess *before) override;

  // TODO: this isn't ideal for the analysis. When there is no next access, it
  // means "we don't know what the next access is" rather than "there is no next
  // access". But it's unclear how to differentiate the two cases...
  void setToExitState(NextAccess *lattice) override {
    propagateIfChanged(lattice, lattice->reset());
  }
};
} // namespace

void NextAccessAnalysis::visitOperation(Operation *op, const NextAccess &after,
                                        NextAccess *before) {
  auto memory = dyn_cast<MemoryEffectOpInterface>(op);
  // If we can't reason about the memory effects, conservatively assume we can't
  // say anything about the next access.
  if (!memory)
    return setToExitState(before);

  SmallVector<MemoryEffects::EffectInstance> effects;
  memory.getEffects(effects);
  ChangeResult result = before->meet(after);
  for (const MemoryEffects::EffectInstance &effect : effects) {
    Value value = effect.getValue();

    // Effects with unspecified value are treated conservatively and we cannot
    // assume anything about the next access.
    if (!value)
      return setToExitState(before);

    value = UnderlyingValueAnalysis::getMostUnderlyingValue(
        value, [&](Value value) {
          return getOrCreateFor<UnderlyingValueLattice>(op, value);
        });
    if (!value)
      return;

    result |= before->set(value, op);
  }
  propagateIfChanged(before, result);
}

namespace {
struct TestNextAccessPass
    : public PassWrapper<TestNextAccessPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestNextAccessPass)

  StringRef getArgument() const override { return "test-next-access"; }

  static constexpr llvm::StringLiteral kTagAttrName = "name";
  static constexpr llvm::StringLiteral kNextAccessAttrName = "next_access";

  void runOnOperation() override {
    Operation *op = getOperation();
    SymbolTableCollection symbolTable;

    DataFlowSolver solver;
    solver.load<DeadCodeAnalysis>();
    solver.load<NextAccessAnalysis>(symbolTable);
    solver.load<SparseConstantPropagation>();
    solver.load<UnderlyingValueAnalysis>();
    if (failed(solver.initializeAndRun(op))) {
      emitError(op->getLoc(), "dataflow solver failed");
      return signalPassFailure();
    }

    op->walk([&](Operation *op) {
      auto tag = op->getAttrOfType<StringAttr>(kTagAttrName);
      if (!tag)
        return;

      const NextAccess *nextAccess = solver.lookupState<NextAccess>(
          op->getNextNode() == nullptr ? ProgramPoint(op->getBlock())
                                       : op->getNextNode());
      if (!nextAccess) {
        op->setAttr(kNextAccessAttrName,
                    StringAttr::get(op->getContext(), "not computed"));
        return;
      }

      SmallVector<Attribute> attrs;
      for (Value operand : op->getOperands()) {
        Value value = UnderlyingValueAnalysis::getMostUnderlyingValue(
            operand, [&](Value value) {
              return solver.lookupState<UnderlyingValueLattice>(value);
            });
        std::optional<ArrayRef<Operation *>> nextAcc =
            nextAccess->getAdjacentAccess(value);
        if (!nextAcc) {
          attrs.push_back(StringAttr::get(op->getContext(), "unknown"));
          continue;
        }

        SmallVector<Attribute> innerAttrs;
        innerAttrs.reserve(nextAcc->size());
        for (Operation *nextAccOp : *nextAcc) {
          if (auto nextAccTag =
                  nextAccOp->getAttrOfType<StringAttr>(kTagAttrName)) {
            innerAttrs.push_back(nextAccTag);
            continue;
          }
          std::string repr;
          llvm::raw_string_ostream os(repr);
          nextAccOp->print(os);
          innerAttrs.push_back(StringAttr::get(op->getContext(), os.str()));
        }
        attrs.push_back(ArrayAttr::get(op->getContext(), innerAttrs));
      }

      op->setAttr(kNextAccessAttrName, ArrayAttr::get(op->getContext(), attrs));
    });
  }
};
} // namespace

namespace mlir::test {
void registerTestNextAccessPass() { PassRegistration<TestNextAccessPass>(); }
} // namespace mlir::test
