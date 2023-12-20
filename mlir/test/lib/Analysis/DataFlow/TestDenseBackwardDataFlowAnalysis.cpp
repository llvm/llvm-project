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
#include "TestDialect.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::dataflow;
using namespace mlir::dataflow::test;

namespace {

class NextAccess : public AbstractDenseLattice, public AccessLatticeBase {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NextAccess)

  using dataflow::AbstractDenseLattice::AbstractDenseLattice;

  ChangeResult meet(const AbstractDenseLattice &lattice) override {
    return AccessLatticeBase::merge(static_cast<AccessLatticeBase>(
        static_cast<const NextAccess &>(lattice)));
  }

  void print(raw_ostream &os) const override {
    return AccessLatticeBase::print(os);
  }
};

class NextAccessAnalysis : public DenseBackwardDataFlowAnalysis<NextAccess> {
public:
  NextAccessAnalysis(DataFlowSolver &solver, SymbolTableCollection &symbolTable,
                     bool assumeFuncReads = false)
      : DenseBackwardDataFlowAnalysis(solver, symbolTable),
        assumeFuncReads(assumeFuncReads) {}

  void visitOperation(Operation *op, const NextAccess &after,
                      NextAccess *before) override;

  void visitCallControlFlowTransfer(CallOpInterface call,
                                    CallControlFlowAction action,
                                    const NextAccess &after,
                                    NextAccess *before) override;

  void visitRegionBranchControlFlowTransfer(RegionBranchOpInterface branch,
                                            RegionBranchPoint regionFrom,
                                            RegionBranchPoint regionTo,
                                            const NextAccess &after,
                                            NextAccess *before) override;

  // TODO: this isn't ideal for the analysis. When there is no next access, it
  // means "we don't know what the next access is" rather than "there is no next
  // access". But it's unclear how to differentiate the two cases...
  void setToExitState(NextAccess *lattice) override {
    propagateIfChanged(lattice, lattice->setKnownToUnknown());
  }

  const bool assumeFuncReads;
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

  // First, check if all underlying values are already known. Otherwise, avoid
  // propagating and stay in the "undefined" state to avoid incorrectly
  // propagating values that may be overwritten later on as that could be
  // problematic for convergence based on monotonicity of lattice updates.
  SmallVector<Value> underlyingValues;
  underlyingValues.reserve(effects.size());
  for (const MemoryEffects::EffectInstance &effect : effects) {
    Value value = effect.getValue();

    // Effects with unspecified value are treated conservatively and we cannot
    // assume anything about the next access.
    if (!value)
      return setToExitState(before);

    // If cannot find the most underlying value, we cannot assume anything about
    // the next accesses.
    std::optional<Value> underlyingValue =
        UnderlyingValueAnalysis::getMostUnderlyingValue(
            value, [&](Value value) {
              return getOrCreateFor<UnderlyingValueLattice>(op, value);
            });

    // If the underlying value is not known yet, don't propagate.
    if (!underlyingValue)
      return;

    underlyingValues.push_back(*underlyingValue);
  }

  // Update the state if all underlying values are known.
  ChangeResult result = before->meet(after);
  for (const auto &[effect, value] : llvm::zip(effects, underlyingValues)) {
    // If the underlying value is known to be unknown, set to fixpoint.
    if (!value)
      return setToExitState(before);

    result |= before->set(value, op);
  }
  propagateIfChanged(before, result);
}

void NextAccessAnalysis::visitCallControlFlowTransfer(
    CallOpInterface call, CallControlFlowAction action, const NextAccess &after,
    NextAccess *before) {
  if (action == CallControlFlowAction::ExternalCallee && assumeFuncReads) {
    SmallVector<Value> underlyingValues;
    underlyingValues.reserve(call->getNumOperands());
    for (Value operand : call.getArgOperands()) {
      std::optional<Value> underlyingValue =
          UnderlyingValueAnalysis::getMostUnderlyingValue(
              operand, [&](Value value) {
                return getOrCreateFor<UnderlyingValueLattice>(
                    call.getOperation(), value);
              });
      if (!underlyingValue)
        return;
      underlyingValues.push_back(*underlyingValue);
    }

    ChangeResult result = before->meet(after);
    for (Value operand : underlyingValues) {
      result |= before->set(operand, call);
    }
    return propagateIfChanged(before, result);
  }
  auto testCallAndStore =
      dyn_cast<::test::TestCallAndStoreOp>(call.getOperation());
  if (testCallAndStore && ((action == CallControlFlowAction::EnterCallee &&
                            testCallAndStore.getStoreBeforeCall()) ||
                           (action == CallControlFlowAction::ExitCallee &&
                            !testCallAndStore.getStoreBeforeCall()))) {
    visitOperation(call, after, before);
  } else {
    AbstractDenseBackwardDataFlowAnalysis::visitCallControlFlowTransfer(
        call, action, after, before);
  }
}

void NextAccessAnalysis::visitRegionBranchControlFlowTransfer(
    RegionBranchOpInterface branch, RegionBranchPoint regionFrom,
    RegionBranchPoint regionTo, const NextAccess &after, NextAccess *before) {
  auto testStoreWithARegion =
      dyn_cast<::test::TestStoreWithARegion>(branch.getOperation());

  if (testStoreWithARegion &&
      ((regionTo.isParent() && !testStoreWithARegion.getStoreBeforeRegion()) ||
       (regionFrom.isParent() &&
        testStoreWithARegion.getStoreBeforeRegion()))) {
    visitOperation(branch, static_cast<const NextAccess &>(after),
                   static_cast<NextAccess *>(before));
  } else {
    propagateIfChanged(before, before->meet(after));
  }
}

namespace {
struct TestNextAccessPass
    : public PassWrapper<TestNextAccessPass, OperationPass<>> {
  TestNextAccessPass() = default;
  TestNextAccessPass(const TestNextAccessPass &other) : PassWrapper(other) {
    interprocedural = other.interprocedural;
    assumeFuncReads = other.assumeFuncReads;
  }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestNextAccessPass)

  StringRef getArgument() const override { return "test-next-access"; }

  Option<bool> interprocedural{
      *this, "interprocedural", llvm::cl::init(true),
      llvm::cl::desc("perform interprocedural analysis")};
  Option<bool> assumeFuncReads{
      *this, "assume-func-reads", llvm::cl::init(false),
      llvm::cl::desc(
          "assume external functions have read effect on all arguments")};

  static constexpr llvm::StringLiteral kTagAttrName = "name";
  static constexpr llvm::StringLiteral kNextAccessAttrName = "next_access";
  static constexpr llvm::StringLiteral kAtEntryPointAttrName =
      "next_at_entry_point";

  static Attribute makeNextAccessAttribute(Operation *op,
                                           const DataFlowSolver &solver,
                                           const NextAccess *nextAccess) {
    if (!nextAccess)
      return StringAttr::get(op->getContext(), "not computed");

    // Note that if the underlying value could not be computed or is unknown, we
    // conservatively treat the result also unknown.
    SmallVector<Attribute> attrs;
    for (Value operand : op->getOperands()) {
      std::optional<Value> underlyingValue =
          UnderlyingValueAnalysis::getMostUnderlyingValue(
              operand, [&](Value value) {
                return solver.lookupState<UnderlyingValueLattice>(value);
              });
      if (!underlyingValue) {
        attrs.push_back(StringAttr::get(op->getContext(), "unknown"));
        continue;
      }
      Value value = *underlyingValue;
      const AdjacentAccess *nextAcc = nextAccess->getAdjacentAccess(value);
      if (!nextAcc || !nextAcc->isKnown()) {
        attrs.push_back(StringAttr::get(op->getContext(), "unknown"));
        continue;
      }

      SmallVector<Attribute> innerAttrs;
      innerAttrs.reserve(nextAcc->get().size());
      for (Operation *nextAccOp : nextAcc->get()) {
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
    return ArrayAttr::get(op->getContext(), attrs);
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    SymbolTableCollection symbolTable;

    auto config = DataFlowConfig().setInterprocedural(interprocedural);
    DataFlowSolver solver(config);
    solver.load<DeadCodeAnalysis>();
    solver.load<NextAccessAnalysis>(symbolTable, assumeFuncReads);
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
      op->setAttr(kNextAccessAttrName,
                  makeNextAccessAttribute(op, solver, nextAccess));

      auto iface = dyn_cast<RegionBranchOpInterface>(op);
      if (!iface)
        return;

      SmallVector<Attribute> entryPointNextAccess;
      SmallVector<RegionSuccessor> regionSuccessors;
      iface.getSuccessorRegions(RegionBranchPoint::parent(), regionSuccessors);
      for (const RegionSuccessor &successor : regionSuccessors) {
        if (!successor.getSuccessor() || successor.getSuccessor()->empty())
          continue;
        Block &successorBlock = successor.getSuccessor()->front();
        ProgramPoint successorPoint = successorBlock.empty()
                                          ? ProgramPoint(&successorBlock)
                                          : &successorBlock.front();
        entryPointNextAccess.push_back(makeNextAccessAttribute(
            op, solver, solver.lookupState<NextAccess>(successorPoint)));
      }
      op->setAttr(kAtEntryPointAttrName,
                  ArrayAttr::get(op->getContext(), entryPointNextAccess));
    });
  }
};
} // namespace

namespace mlir::test {
void registerTestNextAccessPass() { PassRegistration<TestNextAccessPass>(); }
} // namespace mlir::test
