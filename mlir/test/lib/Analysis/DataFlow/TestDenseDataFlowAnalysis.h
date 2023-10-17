//===- TestDenseDataFlowAnalysis.h - Dataflow test utilities ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

namespace mlir {
namespace dataflow {
namespace test {

/// This lattice represents a single underlying value for an SSA value.
class UnderlyingValue {
public:
  /// Create an underlying value state with a known underlying value.
  explicit UnderlyingValue(std::optional<Value> underlyingValue = std::nullopt)
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
  std::optional<Value> underlyingValue;
};

/// This lattice represents, for a given memory resource, the potential last
/// operations that modified the resource.
class AccessLatticeBase {
public:
  /// Clear all modifications.
  ChangeResult reset() {
    if (adjAccesses.empty())
      return ChangeResult::NoChange;
    adjAccesses.clear();
    return ChangeResult::Change;
  }

  /// Join the last modifications.
  ChangeResult merge(const AccessLatticeBase &rhs) {
    ChangeResult result = ChangeResult::NoChange;
    for (const auto &mod : rhs.adjAccesses) {
      auto &lhsMod = adjAccesses[mod.first];
      if (lhsMod != mod.second) {
        lhsMod.insert(mod.second.begin(), mod.second.end());
        result |= ChangeResult::Change;
      }
    }
    return result;
  }

  /// Set the last modification of a value.
  ChangeResult set(Value value, Operation *op) {
    auto &lastMod = adjAccesses[value];
    ChangeResult result = ChangeResult::NoChange;
    if (lastMod.size() != 1 || *lastMod.begin() != op) {
      result = ChangeResult::Change;
      lastMod.clear();
      lastMod.insert(op);
    }
    return result;
  }

  /// Get the adjacent accesses to a value. Returns std::nullopt if they
  /// are not known.
  std::optional<ArrayRef<Operation *>> getAdjacentAccess(Value value) const {
    auto it = adjAccesses.find(value);
    if (it == adjAccesses.end())
      return {};
    return it->second.getArrayRef();
  }

  void print(raw_ostream &os) const {
    for (const auto &lastMod : adjAccesses) {
      os << lastMod.first << ":\n";
      for (Operation *op : lastMod.second)
        os << "  " << *op << "\n";
    }
  }

private:
  /// The potential adjacent accesses to a memory resource. Use a set vector to
  /// keep the results deterministic.
  DenseMap<Value, SetVector<Operation *, SmallVector<Operation *, 2>,
                            SmallPtrSet<Operation *, 2>>>
      adjAccesses;
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
    : public SparseForwardDataFlowAnalysis<UnderlyingValueLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

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

  /// Look for the most underlying value of a value.
  static Value
  getMostUnderlyingValue(Value value,
                         function_ref<const UnderlyingValueLattice *(Value)>
                             getUnderlyingValueFn) {
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
};

} // namespace test
} // namespace dataflow
} // namespace mlir
