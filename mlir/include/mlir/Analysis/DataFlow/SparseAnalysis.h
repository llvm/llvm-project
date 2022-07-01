//===- SparseAnalysis.h - Sparse data-flow analysis -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements sparse data-flow analysis using the data-flow analysis
// framework. The analysis is forward and conditional and uses the results of
// dead code analysis to prune dead code during the analysis.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_DATAFLOW_SPARSEANALYSIS_H
#define MLIR_ANALYSIS_DATAFLOW_SPARSEANALYSIS_H

#include "mlir/Analysis/DataFlowFramework.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir {
namespace dataflow {

//===----------------------------------------------------------------------===//
// AbstractSparseLattice
//===----------------------------------------------------------------------===//

/// This class represents an abstract lattice. A lattice contains information
/// about an SSA value and is what's propagated across the IR by sparse
/// data-flow analysis.
class AbstractSparseLattice : public AnalysisState {
public:
  /// Lattices can only be created for values.
  AbstractSparseLattice(Value value) : AnalysisState(value) {}

  /// Join the information contained in 'rhs' into this lattice. Returns
  /// if the value of the lattice changed.
  virtual ChangeResult join(const AbstractSparseLattice &rhs) = 0;

  /// Returns true if the lattice element is at fixpoint and further calls to
  /// `join` will not update the value of the element.
  virtual bool isAtFixpoint() const = 0;

  /// Mark the lattice element as having reached a pessimistic fixpoint. This
  /// means that the lattice may potentially have conflicting value states, and
  /// only the most conservative value should be relied on.
  virtual ChangeResult markPessimisticFixpoint() = 0;

  /// When the lattice gets updated, propagate an update to users of the value
  /// using its use-def chain to subscribed analyses.
  void onUpdate(DataFlowSolver *solver) const override;

  /// Subscribe an analysis to updates of the lattice. When the lattice changes,
  /// subscribed analyses are re-invoked on all users of the value. This is
  /// more efficient than relying on the dependency map.
  void useDefSubscribe(DataFlowAnalysis *analysis) {
    useDefSubscribers.insert(analysis);
  }

private:
  /// A set of analyses that should be updated when this lattice changes.
  SetVector<DataFlowAnalysis *, SmallVector<DataFlowAnalysis *, 4>,
            SmallPtrSet<DataFlowAnalysis *, 4>>
      useDefSubscribers;
};

//===----------------------------------------------------------------------===//
// Lattice
//===----------------------------------------------------------------------===//

/// This class represents a lattice holding a specific value of type `ValueT`.
/// Lattice values (`ValueT`) are required to adhere to the following:
///
///   * static ValueT join(const ValueT &lhs, const ValueT &rhs);
///     - This method conservatively joins the information held by `lhs`
///       and `rhs` into a new value. This method is required to be monotonic.
///   * bool operator==(const ValueT &rhs) const;
///
template <typename ValueT>
class Lattice : public AbstractSparseLattice {
public:
  using AbstractSparseLattice::AbstractSparseLattice;

  /// Get a lattice element with a known value.
  Lattice(const ValueT &knownValue = ValueT())
      : AbstractSparseLattice(Value()), knownValue(knownValue) {}

  /// Return the value held by this lattice. This requires that the value is
  /// initialized.
  ValueT &getValue() {
    assert(!isUninitialized() && "expected known lattice element");
    return *optimisticValue;
  }
  const ValueT &getValue() const {
    return const_cast<Lattice<ValueT> *>(this)->getValue();
  }

  /// Returns true if the value of this lattice hasn't yet been initialized.
  bool isUninitialized() const override { return !optimisticValue.hasValue(); }
  /// Force the initialization of the element by setting it to its pessimistic
  /// fixpoint.
  ChangeResult defaultInitialize() override {
    return markPessimisticFixpoint();
  }

  /// Returns true if the lattice has reached a fixpoint. A fixpoint is when
  /// the information optimistically assumed to be true is the same as the
  /// information known to be true.
  bool isAtFixpoint() const override { return optimisticValue == knownValue; }

  /// Join the information contained in the 'rhs' lattice into this
  /// lattice. Returns if the state of the current lattice changed.
  ChangeResult join(const AbstractSparseLattice &rhs) override {
    const Lattice<ValueT> &rhsLattice =
        static_cast<const Lattice<ValueT> &>(rhs);

    // If we are at a fixpoint, or rhs is uninitialized, there is nothing to do.
    if (isAtFixpoint() || rhsLattice.isUninitialized())
      return ChangeResult::NoChange;

    // Join the rhs value into this lattice.
    return join(rhsLattice.getValue());
  }

  /// Join the information contained in the 'rhs' value into this
  /// lattice. Returns if the state of the current lattice changed.
  ChangeResult join(const ValueT &rhs) {
    // If the current lattice is uninitialized, copy the rhs value.
    if (isUninitialized()) {
      optimisticValue = rhs;
      return ChangeResult::Change;
    }

    // Otherwise, join rhs with the current optimistic value.
    ValueT newValue = ValueT::join(*optimisticValue, rhs);
    assert(ValueT::join(newValue, *optimisticValue) == newValue &&
           "expected `join` to be monotonic");
    assert(ValueT::join(newValue, rhs) == newValue &&
           "expected `join` to be monotonic");

    // Update the current optimistic value if something changed.
    if (newValue == optimisticValue)
      return ChangeResult::NoChange;

    optimisticValue = newValue;
    return ChangeResult::Change;
  }

  /// Mark the lattice element as having reached a pessimistic fixpoint. This
  /// means that the lattice may potentially have conflicting value states,
  /// and only the conservatively known value state should be relied on.
  ChangeResult markPessimisticFixpoint() override {
    if (isAtFixpoint())
      return ChangeResult::NoChange;

    // For this fixed point, we take whatever we knew to be true and set that
    // to our optimistic value.
    optimisticValue = knownValue;
    return ChangeResult::Change;
  }

  /// Print the lattice element.
  void print(raw_ostream &os) const override {
    os << "[";
    knownValue.print(os);
    os << ", ";
    if (optimisticValue)
      optimisticValue->print(os);
    else
      os << "<NULL>";
    os << "]";
  }

private:
  /// The value that is conservatively known to be true.
  ValueT knownValue;
  /// The currently computed value that is optimistically assumed to be true,
  /// or None if the lattice element is uninitialized.
  Optional<ValueT> optimisticValue;
};

} // end namespace dataflow
} // end namespace mlir

#endif // MLIR_ANALYSIS_DATAFLOW_SPARSEANALYSIS_H
