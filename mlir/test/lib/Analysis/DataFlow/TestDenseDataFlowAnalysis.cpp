//===- TestDeadCodeAnalysis.cpp - Test dead code analysis -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::dataflow;

namespace {
/// This lattice represents a single underlying value for an SSA value.
class UnderlyingValue {
public:
  /// Create an underlying value state with a known underlying value.
  explicit UnderlyingValue(Optional<Value> underlyingValue = None)
      : underlyingValue(underlyingValue) {}

  /// Whether the state is uninitialized.
  bool isUninitialized() const { return !underlyingValue.has_value(); }

  /// Returns the underlying value.
  Value getUnderlyingValue() const {
    assert(!isUninitialized());
    return *underlyingValue;
  }

  /// Join two underlying values. If there are conflicting underlying values,
  /// go to the pessimistic value.
  static UnderlyingValue join(const UnderlyingValue &lhs,
                              const UnderlyingValue &rhs) {
    if (lhs.isUninitialized())
      return rhs;
    if (rhs.isUninitialized())
      return lhs;
    return lhs.underlyingValue == rhs.underlyingValue
               ? lhs
               : UnderlyingValue(Value{});
  }

  /// Compare underlying values.
  bool operator==(const UnderlyingValue &rhs) const {
    return underlyingValue == rhs.underlyingValue;
  }

  void print(raw_ostream &os) const { os << underlyingValue; }

private:
  Optional<Value> underlyingValue;
};

/// This lattice represents, for a given memory resource, the potential last
/// operations that modified the resource.
class LastModification : public AbstractDenseLattice {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LastModification)

  using AbstractDenseLattice::AbstractDenseLattice;

  /// Clear all modifications.
  ChangeResult reset() {
    if (lastMods.empty())
      return ChangeResult::NoChange;
    lastMods.clear();
    return ChangeResult::Change;
  }

  /// Join the last modifications.
  ChangeResult join(const AbstractDenseLattice &lattice) override {
    const auto &rhs = static_cast<const LastModification &>(lattice);
    ChangeResult result = ChangeResult::NoChange;
    for (const auto &mod : rhs.lastMods) {
      auto &lhsMod = lastMods[mod.first];
      if (lhsMod != mod.second) {
        lhsMod.insert(mod.second.begin(), mod.second.end());
        result |= ChangeResult::Change;
      }
    }
    return result;
  }

  /// Set the last modification of a value.
  ChangeResult set(Value value, Operation *op) {
    auto &lastMod = lastMods[value];
    ChangeResult result = ChangeResult::NoChange;
    if (lastMod.size() != 1 || *lastMod.begin() != op) {
      result = ChangeResult::Change;
      lastMod.clear();
      lastMod.insert(op);
    }
    return result;
  }

  /// Get the last modifications of a value. Returns none if the last
  /// modifications are not known.
  Optional<ArrayRef<Operation *>> getLastModifiers(Value value) const {
    auto it = lastMods.find(value);
    if (it == lastMods.end())
      return {};
    return it->second.getArrayRef();
  }

  void print(raw_ostream &os) const override {
    for (const auto &lastMod : lastMods) {
      os << lastMod.first << ":\n";
      for (Operation *op : lastMod.second)
        os << "  " << *op << "\n";
    }
  }

private:
  /// The potential last modifications of a memory resource. Use a set vector to
  /// keep the results deterministic.
  DenseMap<Value, SetVector<Operation *, SmallVector<Operation *, 2>,
                            SmallPtrSet<Operation *, 2>>>
      lastMods;
};

class LastModifiedAnalysis : public DenseDataFlowAnalysis<LastModification> {
public:
  using DenseDataFlowAnalysis::DenseDataFlowAnalysis;

  /// Visit an operation. If the operation has no memory effects, then the state
  /// is propagated with no change. If the operation allocates a resource, then
  /// its reaching definitions is set to empty. If the operation writes to a
  /// resource, then its reaching definition is set to the written value.
  void visitOperation(Operation *op, const LastModification &before,
                      LastModification *after) override;

  /// At an entry point, the last modifications of all memory resources are
  /// unknown.
  void setToEntryState(LastModification *lattice) override {
    propagateIfChanged(lattice, lattice->reset());
  }
};

/// Define the lattice class explicitly to provide a type ID.
struct UnderlyingValueLattice : public Lattice<UnderlyingValue> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnderlyingValueLattice)
  using Lattice::Lattice;
};

/// An analysis that uses forwarding of values along control-flow and callgraph
/// edges to determine single underlying values for block arguments. This
/// analysis exists so that the test analysis and pass can test the behaviour of
/// the dense data-flow analysis on the callgraph.
class UnderlyingValueAnalysis
    : public SparseDataFlowAnalysis<UnderlyingValueLattice> {
public:
  using SparseDataFlowAnalysis::SparseDataFlowAnalysis;

  /// The underlying value of the results of an operation are not known.
  void visitOperation(Operation *op,
                      ArrayRef<const UnderlyingValueLattice *> operands,
                      ArrayRef<UnderlyingValueLattice *> results) override {
    setAllToEntryStates(results);
  }

  /// At an entry point, the underlying value of a value is itself.
  void setToEntryState(UnderlyingValueLattice *lattice) override {
    propagateIfChanged(lattice,
                       lattice->join(UnderlyingValue{lattice->getPoint()}));
  }
};
} // end anonymous namespace

/// Look for the most underlying value of a value.
static Value getMostUnderlyingValue(
    Value value,
    function_ref<const UnderlyingValueLattice *(Value)> getUnderlyingValueFn) {
  const UnderlyingValueLattice *underlying;
  do {
    underlying = getUnderlyingValueFn(value);
    if (!underlying || underlying->getValue().isUninitialized())
      return {};
    Value underlyingValue = underlying->getValue().getUnderlyingValue();
    if (underlyingValue == value)
      break;
    value = underlyingValue;
  } while (true);
  return value;
}

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

  ChangeResult result = after->join(before);
  for (const auto &effect : effects) {
    Value value = effect.getValue();

    // If we see an effect on anything other than a value, assume we can't
    // deduce anything about the last modifications.
    if (!value)
      return setToEntryState(after);

    value = getMostUnderlyingValue(value, [&](Value value) {
      return getOrCreateFor<UnderlyingValueLattice>(op, value);
    });
    if (!value)
      return;

    // Nothing to do for reads.
    if (isa<MemoryEffects::Read>(effect.getEffect()))
      continue;

    result |= after->set(value, op);
  }
  propagateIfChanged(after, result);
}

namespace {
struct TestLastModifiedPass
    : public PassWrapper<TestLastModifiedPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLastModifiedPass)

  StringRef getArgument() const override { return "test-last-modified"; }

  void runOnOperation() override {
    Operation *op = getOperation();

    DataFlowSolver solver;
    solver.load<DeadCodeAnalysis>();
    solver.load<SparseConstantPropagation>();
    solver.load<LastModifiedAnalysis>();
    solver.load<UnderlyingValueAnalysis>();
    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();

    raw_ostream &os = llvm::errs();

    op->walk([&](Operation *op) {
      auto tag = op->getAttrOfType<StringAttr>("tag");
      if (!tag)
        return;
      os << "test_tag: " << tag.getValue() << ":\n";
      const LastModification *lastMods =
          solver.lookupState<LastModification>(op);
      assert(lastMods && "expected a dense lattice");
      for (auto &it : llvm::enumerate(op->getOperands())) {
        os << " operand #" << it.index() << "\n";
        Value value = getMostUnderlyingValue(it.value(), [&](Value value) {
          return solver.lookupState<UnderlyingValueLattice>(value);
        });
        assert(value && "expected an underlying value");
        if (Optional<ArrayRef<Operation *>> lastMod =
                lastMods->getLastModifiers(value)) {
          for (Operation *lastModifier : *lastMod) {
            if (auto tagName =
                    lastModifier->getAttrOfType<StringAttr>("tag_name")) {
              os << "  - " << tagName.getValue() << "\n";
            } else {
              os << "  - " << lastModifier->getName() << "\n";
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
