//===- TestDenseForwardDataFlowAnalysis.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of tests passes exercising dense forward data flow analysis.
//
//===----------------------------------------------------------------------===//

#include "TestDenseDataFlowAnalysis.h"
#include "TestDialect.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"
#include <optional>

using namespace mlir;
using namespace mlir::dataflow;
using namespace mlir::dataflow::test;

namespace {

/// This lattice represents, for a given memory resource, the potential last
/// operations that modified the resource.
class LastModification : public AbstractDenseLattice, public AccessLatticeBase {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LastModification)

  using AbstractDenseLattice::AbstractDenseLattice;

  /// Join the last modifications.
  ChangeResult join(const AbstractDenseLattice &lattice) override {
    return AccessLatticeBase::merge(static_cast<AccessLatticeBase>(
        static_cast<const LastModification &>(lattice)));
  }

  void print(raw_ostream &os) const override {
    return AccessLatticeBase::print(os);
  }
};

class LastModifiedAnalysis
    : public DenseForwardDataFlowAnalysis<LastModification> {
public:
  explicit LastModifiedAnalysis(DataFlowSolver &solver, bool assumeFuncWrites)
      : DenseForwardDataFlowAnalysis(solver),
        assumeFuncWrites(assumeFuncWrites) {}

  /// Visit an operation. If the operation has no memory effects, then the state
  /// is propagated with no change. If the operation allocates a resource, then
  /// its reaching definitions is set to empty. If the operation writes to a
  /// resource, then its reaching definition is set to the written value.
  void visitOperation(Operation *op, const LastModification &before,
                      LastModification *after) override;

  void visitCallControlFlowTransfer(CallOpInterface call,
                                    CallControlFlowAction action,
                                    const LastModification &before,
                                    LastModification *after) override;

  void visitRegionBranchControlFlowTransfer(RegionBranchOpInterface branch,
                                            std::optional<unsigned> regionFrom,
                                            std::optional<unsigned> regionTo,
                                            const LastModification &before,
                                            LastModification *after) override;

  /// At an entry point, the last modifications of all memory resources are
  /// unknown.
  void setToEntryState(LastModification *lattice) override {
    propagateIfChanged(lattice, lattice->reset());
  }

private:
  const bool assumeFuncWrites;
};
} // end anonymous namespace

void LastModifiedAnalysis::visitOperation(Operation *op,
                                          const LastModification &before,
                                          LastModification *after) {
  auto memory = dyn_cast<MemoryEffectOpInterface>(op);
  // If we can't reason about the memory effects, then conservatively assume we
  // can't deduce anything about the last modifications.
  if (!memory)
    return setToEntryState(after);

  SmallVector<MemoryEffects::EffectInstance> effects;
  memory.getEffects(effects);

  // First, check if all underlying values are already known. Otherwise, avoid
  // propagating and stay in the "undefined" state to avoid incorrectly
  // propagating values that may be overwritten later on as that could be
  // problematic for convergence based on monotonicity of lattice updates.
  SmallVector<Value> underlyingValues;
  underlyingValues.reserve(effects.size());
  for (const auto &effect : effects) {
    Value value = effect.getValue();

    // If we see an effect on anything other than a value, assume we can't
    // deduce anything about the last modifications.
    if (!value)
      return setToEntryState(after);

    // If we cannot find the underlying value, we shouldn't just propagate the
    // effects through, return the pessimistic state.
    std::optional<Value> underlyingValue =
        UnderlyingValueAnalysis::getMostUnderlyingValue(
            value, [&](Value value) {
              return getOrCreateFor<UnderlyingValueLattice>(op, value);
            });

    // If the underlying value is not yet known, don't propagate yet.
    if (!underlyingValue)
      return;

    underlyingValues.push_back(*underlyingValue);
  }

  // Update the state when all underlying values are known.
  ChangeResult result = after->join(before);
  for (const auto &[effect, value] : llvm::zip(effects, underlyingValues)) {
    // If the underlying value is known to be unknown, set to fixpoint state.
    if (!value)
      return setToEntryState(after);

    // Nothing to do for reads.
    if (isa<MemoryEffects::Read>(effect.getEffect()))
      continue;

    result |= after->set(value, op);
  }
  propagateIfChanged(after, result);
}

void LastModifiedAnalysis::visitCallControlFlowTransfer(
    CallOpInterface call, CallControlFlowAction action,
    const LastModification &before, LastModification *after) {
  if (action == CallControlFlowAction::ExternalCallee && assumeFuncWrites) {
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

    ChangeResult result = after->join(before);
    for (Value operand : underlyingValues)
      result |= after->set(operand, call);
    return propagateIfChanged(after, result);
  }
  auto testCallAndStore =
      dyn_cast<::test::TestCallAndStoreOp>(call.getOperation());
  if (testCallAndStore && ((action == CallControlFlowAction::EnterCallee &&
                            testCallAndStore.getStoreBeforeCall()) ||
                           (action == CallControlFlowAction::ExitCallee &&
                            !testCallAndStore.getStoreBeforeCall()))) {
    return visitOperation(call, before, after);
  }
  AbstractDenseForwardDataFlowAnalysis::visitCallControlFlowTransfer(
      call, action, before, after);
}

void LastModifiedAnalysis::visitRegionBranchControlFlowTransfer(
    RegionBranchOpInterface branch, std::optional<unsigned> regionFrom,
    std::optional<unsigned> regionTo, const LastModification &before,
    LastModification *after) {
  auto defaultHandling = [&]() {
    AbstractDenseForwardDataFlowAnalysis::visitRegionBranchControlFlowTransfer(
        branch, regionFrom, regionTo, before, after);
  };
  TypeSwitch<Operation *>(branch.getOperation())
      .Case<::test::TestStoreWithARegion, ::test::TestStoreWithALoopRegion>(
          [=](auto storeWithRegion) {
            if ((!regionTo && !storeWithRegion.getStoreBeforeRegion()) ||
                (!regionFrom && storeWithRegion.getStoreBeforeRegion()))
              visitOperation(branch, before, after);
            defaultHandling();
          })
      .Default([=](auto) { defaultHandling(); });
}

namespace {
struct TestLastModifiedPass
    : public PassWrapper<TestLastModifiedPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLastModifiedPass)

  TestLastModifiedPass() = default;
  TestLastModifiedPass(const TestLastModifiedPass &other) : PassWrapper(other) {
    interprocedural = other.interprocedural;
    assumeFuncWrites = other.assumeFuncWrites;
  }

  StringRef getArgument() const override { return "test-last-modified"; }

  Option<bool> interprocedural{
      *this, "interprocedural", llvm::cl::init(true),
      llvm::cl::desc("perform interprocedural analysis")};
  Option<bool> assumeFuncWrites{
      *this, "assume-func-writes", llvm::cl::init(false),
      llvm::cl::desc(
          "assume external functions have write effect on all arguments")};

  void runOnOperation() override {
    Operation *op = getOperation();

    DataFlowSolver solver(DataFlowConfig().setInterprocedural(interprocedural));
    solver.load<DeadCodeAnalysis>();
    solver.load<SparseConstantPropagation>();
    solver.load<LastModifiedAnalysis>(assumeFuncWrites);
    solver.load<UnderlyingValueAnalysis>();
    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();

    raw_ostream &os = llvm::errs();

    // Note that if the underlying value could not be computed or is unknown, we
    // conservatively treat the result also unknown.
    op->walk([&](Operation *op) {
      auto tag = op->getAttrOfType<StringAttr>("tag");
      if (!tag)
        return;
      os << "test_tag: " << tag.getValue() << ":\n";
      const LastModification *lastMods =
          solver.lookupState<LastModification>(op);
      assert(lastMods && "expected a dense lattice");
      for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
        os << " operand #" << index << "\n";
        std::optional<Value> underlyingValue =
            UnderlyingValueAnalysis::getMostUnderlyingValue(
                operand, [&](Value value) {
                  return solver.lookupState<UnderlyingValueLattice>(value);
                });
        if (!underlyingValue) {
          os << " - <unknown>\n";
          continue;
        }
        Value value = *underlyingValue;
        assert(value && "expected an underlying value");
        if (const AdjacentAccess *lastMod =
                lastMods->getAdjacentAccess(value)) {
          if (!lastMod->isKnown()) {
            os << " - <unknown>\n";
          } else {
            for (Operation *lastModifier : lastMod->get()) {
              if (auto tagName =
                      lastModifier->getAttrOfType<StringAttr>("tag_name")) {
                os << "  - " << tagName.getValue() << "\n";
              } else {
                os << "  - " << lastModifier->getName() << "\n";
              }
            }
          }
        } else {
          os << "  - <unknown>\n";
        }
      }
    });
  }
};
} // end anonymous namespace

namespace mlir {
namespace test {
void registerTestLastModifiedPass() {
  PassRegistration<TestLastModifiedPass>();
}
} // end namespace test
} // end namespace mlir
