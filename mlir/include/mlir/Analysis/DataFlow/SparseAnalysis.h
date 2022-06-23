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
#include "mlir/Interfaces/ControlFlowInterfaces.h"
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

//===----------------------------------------------------------------------===//
// AbstractSparseDataFlowAnalysis
//===----------------------------------------------------------------------===//

/// Base class for sparse (forward) data-flow analyses. A sparse analysis
/// implements a transfer function on operations from the lattices of the
/// operands to the lattices of the results. This analysis will propagate
/// lattices across control-flow edges and the callgraph using liveness
/// information.
class AbstractSparseDataFlowAnalysis : public DataFlowAnalysis {
public:
  /// Initialize the analysis by visiting every owner of an SSA value: all
  /// operations and blocks.
  LogicalResult initialize(Operation *top) override;

  /// Visit a program point. If this is a block and all control-flow
  /// predecessors or callsites are known, then the arguments lattices are
  /// propagated from them. If this is a call operation or an operation with
  /// region control-flow, then its result lattices are set accordingly.
  /// Otherwise, the operation transfer function is invoked.
  LogicalResult visit(ProgramPoint point) override;

protected:
  explicit AbstractSparseDataFlowAnalysis(DataFlowSolver &solver);

  /// The operation transfer function. Given the operand lattices, this
  /// function is expected to set the result lattices.
  virtual void
  visitOperationImpl(Operation *op,
                     ArrayRef<const AbstractSparseLattice *> operandLattices,
                     ArrayRef<AbstractSparseLattice *> resultLattices) = 0;

  /// Get the lattice element of a value.
  virtual AbstractSparseLattice *getLatticeElement(Value value) = 0;

  /// Get a read-only lattice element for a value and add it as a dependency to
  /// a program point.
  const AbstractSparseLattice *getLatticeElementFor(ProgramPoint point,
                                                    Value value);

  /// Mark the given lattice elements as having reached their pessimistic
  /// fixpoints and propagate an update if any changed.
  void markAllPessimisticFixpoint(ArrayRef<AbstractSparseLattice *> lattices);

  /// Join the lattice element and propagate and update if it changed.
  void join(AbstractSparseLattice *lhs, const AbstractSparseLattice &rhs);

private:
  /// Recursively initialize the analysis on nested operations and blocks.
  LogicalResult initializeRecursively(Operation *op);

  /// Visit an operation. If this is a call operation or an operation with
  /// region control-flow, then its result lattices are set accordingly.
  /// Otherwise, the operation transfer function is invoked.
  void visitOperation(Operation *op);

  /// Visit a block to compute the lattice values of its arguments. If this is
  /// an entry block, then the argument values are determined from the block's
  /// "predecessors" as set by `PredecessorState`. The predecessors can be
  /// region terminators or callable callsites. Otherwise, the values are
  /// determined from block predecessors.
  void visitBlock(Block *block);

  /// Visit a program point `point` with predecessors within a region branch
  /// operation `branch`, which can either be the entry block of one of the
  /// regions or the parent operation itself, and set either the argument or
  /// parent result lattices.
  void visitRegionSuccessors(ProgramPoint point, RegionBranchOpInterface branch,
                             Optional<unsigned> successorIndex,
                             ArrayRef<AbstractSparseLattice *> lattices);
};

//===----------------------------------------------------------------------===//
// SparseDataFlowAnalysis
//===----------------------------------------------------------------------===//

/// A sparse (forward) data-flow analysis for propagating SSA value lattices
/// across the IR by implementing transfer functions for operations.
///
/// `StateT` is expected to be a subclass of `AbstractSparseLattice`.
template <typename StateT>
class SparseDataFlowAnalysis : public AbstractSparseDataFlowAnalysis {
  static_assert(
      std::is_base_of<AbstractSparseLattice, StateT>::value,
      "analysis state class expected to subclass AbstractSparseLattice");

public:
  explicit SparseDataFlowAnalysis(DataFlowSolver &solver)
      : AbstractSparseDataFlowAnalysis(solver) {}

  /// Visit an operation with the lattices of its operands. This function is
  /// expected to set the lattices of the operation's results.
  virtual void visitOperation(Operation *op, ArrayRef<const StateT *> operands,
                              ArrayRef<StateT *> results) = 0;

protected:
  /// Get the lattice element for a value.
  StateT *getLatticeElement(Value value) override {
    return getOrCreate<StateT>(value);
  }

  /// Get the lattice element for a value and create a dependency on the
  /// provided program point.
  const StateT *getLatticeElementFor(ProgramPoint point, Value value) {
    return static_cast<const StateT *>(
        AbstractSparseDataFlowAnalysis::getLatticeElementFor(point, value));
  }

  /// Mark the lattice elements of a range of values as having reached their
  /// pessimistic fixpoint.
  void markAllPessimisticFixpoint(ArrayRef<StateT *> lattices) {
    AbstractSparseDataFlowAnalysis::markAllPessimisticFixpoint(
        {reinterpret_cast<AbstractSparseLattice *const *>(lattices.begin()),
         lattices.size()});
  }

private:
  /// Type-erased wrappers that convert the abstract lattice operands to derived
  /// lattices and invoke the virtual hooks operating on the derived lattices.
  void visitOperationImpl(
      Operation *op, ArrayRef<const AbstractSparseLattice *> operandLattices,
      ArrayRef<AbstractSparseLattice *> resultLattices) override {
    visitOperation(
        op,
        {reinterpret_cast<const StateT *const *>(operandLattices.begin()),
         operandLattices.size()},
        {reinterpret_cast<StateT *const *>(resultLattices.begin()),
         resultLattices.size()});
  }
};

} // end namespace dataflow
} // end namespace mlir

#endif // MLIR_ANALYSIS_DATAFLOW_SPARSEANALYSIS_H
