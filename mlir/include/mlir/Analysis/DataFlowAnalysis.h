//===- DataFlowAnalysis.h - General DataFlow Analysis Utilities -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file has several utilities and algorithms that perform abstract dataflow
// analysis over the IR. These allow for users to hook into various analysis
// propagation algorithms without needing to reinvent the traversal over the
// different types of control structures present within MLIR, such as regions,
// the callgraph, etc. A few of the main entry points are detailed below:
//
// FowardDataFlowAnalysis:
//  This class provides support for defining dataflow algorithms that are
//  forward, sparse, pessimistic (except along unreached backedges) and
//  context-insensitive for the interprocedural aspects.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_DATAFLOWANALYSIS_H
#define MLIR_ANALYSIS_DATAFLOWANALYSIS_H

#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/Allocator.h"

namespace mlir {
//===----------------------------------------------------------------------===//
// ChangeResult
//===----------------------------------------------------------------------===//

/// A result type used to indicate if a change happened. Boolean operations on
/// ChangeResult behave as though `Change` is truthy.
enum class ChangeResult {
  NoChange,
  Change,
};
inline ChangeResult operator|(ChangeResult lhs, ChangeResult rhs) {
  return lhs == ChangeResult::Change ? lhs : rhs;
}
inline ChangeResult &operator|=(ChangeResult &lhs, ChangeResult rhs) {
  lhs = lhs | rhs;
  return lhs;
}
inline ChangeResult operator&(ChangeResult lhs, ChangeResult rhs) {
  return lhs == ChangeResult::NoChange ? lhs : rhs;
}

//===----------------------------------------------------------------------===//
// AbstractLatticeElement
//===----------------------------------------------------------------------===//

namespace detail {
/// This class represents an abstract lattice. A lattice is what gets propagated
/// across the IR, and contains the information for a specific Value.
class AbstractLatticeElement {
public:
  virtual ~AbstractLatticeElement();

  /// Returns true if the value of this lattice is uninitialized, meaning that
  /// it hasn't yet been initialized.
  virtual bool isUninitialized() const = 0;

  /// Join the information contained in 'rhs' into this lattice. Returns
  /// if the value of the lattice changed.
  virtual ChangeResult join(const AbstractLatticeElement &rhs) = 0;

  /// Mark the lattice element as having reached a pessimistic fixpoint. This
  /// means that the lattice may potentially have conflicting value states, and
  /// only the most conservative value should be relied on.
  virtual ChangeResult markPessimisticFixpoint() = 0;

  /// Mark the lattice element as having reached an optimistic fixpoint. This
  /// means that we optimistically assume the current value is the true state.
  virtual void markOptimisticFixpoint() = 0;

  /// Returns true if the lattice has reached a fixpoint. A fixpoint is when the
  /// information optimistically assumed to be true is the same as the
  /// information known to be true.
  virtual bool isAtFixpoint() const = 0;
};
} // namespace detail

//===----------------------------------------------------------------------===//
// LatticeElement
//===----------------------------------------------------------------------===//

/// This class represents a lattice holding a specific value of type `ValueT`.
/// Lattice values (`ValueT`) are required to adhere to the following:
///   * static ValueT join(const ValueT &lhs, const ValueT &rhs);
///     - This method conservatively joins the information held by `lhs`
///       and `rhs` into a new value. This method is required to be monotonic.
///   * static ValueT getPessimisticValueState(MLIRContext *context);
///     - This method computes a pessimistic/conservative value state assuming
///       no information about the state of the IR.
///   * static ValueT getPessimisticValueState(Value value);
///     - This method computes a pessimistic/conservative value state for
///       `value` assuming only information present in the current IR.
///   * bool operator==(const ValueT &rhs) const;
///
template <typename ValueT>
class LatticeElement final : public detail::AbstractLatticeElement {
public:
  LatticeElement() = delete;
  LatticeElement(const ValueT &knownValue) : knownValue(knownValue) {}

  /// Return the value held by this lattice. This requires that the value is
  /// initialized.
  ValueT &getValue() {
    assert(!isUninitialized() && "expected known lattice element");
    return *optimisticValue;
  }
  const ValueT &getValue() const {
    assert(!isUninitialized() && "expected known lattice element");
    return *optimisticValue;
  }

  /// Returns true if the value of this lattice hasn't yet been initialized.
  bool isUninitialized() const final { return !optimisticValue; }

  /// Join the information contained in the 'rhs' lattice into this
  /// lattice. Returns if the state of the current lattice changed.
  ChangeResult join(const detail::AbstractLatticeElement &rhs) final {
    const LatticeElement<ValueT> &rhsLattice =
        static_cast<const LatticeElement<ValueT> &>(rhs);

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
  /// means that the lattice may potentially have conflicting value states, and
  /// only the conservatively known value state should be relied on.
  ChangeResult markPessimisticFixpoint() final {
    if (isAtFixpoint())
      return ChangeResult::NoChange;

    // For this fixed point, we take whatever we knew to be true and set that to
    // our optimistic value.
    optimisticValue = knownValue;
    return ChangeResult::Change;
  }

  /// Mark the lattice element as having reached an optimistic fixpoint. This
  /// means that we optimistically assume the current value is the true state.
  void markOptimisticFixpoint() final {
    assert(!isUninitialized() && "expected an initialized value");
    knownValue = *optimisticValue;
  }

  /// Returns true if the lattice has reached a fixpoint. A fixpoint is when the
  /// information optimistically assumed to be true is the same as the
  /// information known to be true.
  bool isAtFixpoint() const final { return optimisticValue == knownValue; }

private:
  /// The value that is conservatively known to be true.
  ValueT knownValue;
  /// The currently computed value that is optimistically assumed to be true, or
  /// None if the lattice element is uninitialized.
  Optional<ValueT> optimisticValue;
};

//===----------------------------------------------------------------------===//
// ForwardDataFlowAnalysisBase
//===----------------------------------------------------------------------===//

namespace detail {
/// This class is the non-templated virtual base class for the
/// ForwardDataFlowAnalysis. This class provides opaque hooks to the main
/// algorithm.
class ForwardDataFlowAnalysisBase {
public:
  virtual ~ForwardDataFlowAnalysisBase();

  /// Initialize and compute the analysis on operations rooted under the given
  /// top-level operation. Note that the top-level operation is not visited.
  void run(Operation *topLevelOp);

  /// Return the lattice element attached to the given value. If a lattice has
  /// not been added for the given value, a new 'uninitialized' value is
  /// inserted and returned.
  AbstractLatticeElement &getLatticeElement(Value value);

  /// Return the lattice element attached to the given value, or nullptr if no
  /// lattice for the value has yet been created.
  AbstractLatticeElement *lookupLatticeElement(Value value);

  /// Visit the given operation, and join any necessary analysis state
  /// into the lattices for the results and block arguments owned by this
  /// operation using the provided set of operand lattice elements (all pointer
  /// values are guaranteed to be non-null). Returns if any result or block
  /// argument value lattices changed during the visit. The lattice for a result
  /// or block argument value can be obtained and join'ed into by using
  /// `getLatticeElement`.
  virtual ChangeResult
  visitOperation(Operation *op,
                 ArrayRef<AbstractLatticeElement *> operands) = 0;

  /// Given a BranchOpInterface, and the current lattice elements that
  /// correspond to the branch operands (all pointer values are guaranteed to be
  /// non-null), try to compute a specific set of successors that would be
  /// selected for the branch. Returns failure if not computable, or if all of
  /// the successors would be chosen. If a subset of successors can be selected,
  /// `successors` is populated.
  virtual LogicalResult
  getSuccessorsForOperands(BranchOpInterface branch,
                           ArrayRef<AbstractLatticeElement *> operands,
                           SmallVectorImpl<Block *> &successors) = 0;

  /// Given a RegionBranchOpInterface, and the current lattice elements that
  /// correspond to the branch operands (all pointer values are guaranteed to be
  /// non-null), compute a specific set of region successors that would be
  /// selected.
  virtual void
  getSuccessorsForOperands(RegionBranchOpInterface branch,
                           Optional<unsigned> sourceIndex,
                           ArrayRef<AbstractLatticeElement *> operands,
                           SmallVectorImpl<RegionSuccessor> &successors) = 0;

  /// Given a operation with successor regions, one of those regions,
  /// and the lattice elements corresponding to the operation's
  /// arguments, compute the latice values for block arguments
  /// that are not accounted for by the branching control flow (ex. the
  /// bounds of loops).
  virtual ChangeResult
  visitNonControlFlowArguments(Operation *op, const RegionSuccessor &region,
                               ArrayRef<AbstractLatticeElement *> operands) = 0;

  /// Create a new uninitialized lattice element. An optional value is provided
  /// which, if valid, should be used to initialize the known conservative state
  /// of the lattice.
  virtual AbstractLatticeElement *createLatticeElement(Value value = {}) = 0;

private:
  /// A map from SSA value to lattice element.
  DenseMap<Value, AbstractLatticeElement *> latticeValues;
};
} // namespace detail

//===----------------------------------------------------------------------===//
// ForwardDataFlowAnalysis
//===----------------------------------------------------------------------===//

/// This class provides a general forward dataflow analysis driver
/// utilizing the lattice classes defined above, to enable the easy definition
/// of dataflow analysis algorithms. More specifically this driver is useful for
/// defining analyses that are forward, sparse, pessimistic (except along
/// unreached backedges) and context-insensitive for the interprocedural
/// aspects.
template <typename ValueT>
class ForwardDataFlowAnalysis : public detail::ForwardDataFlowAnalysisBase {
public:
  ForwardDataFlowAnalysis(MLIRContext *context) : context(context) {}

  /// Return the MLIR context used when constructing this analysis.
  MLIRContext *getContext() { return context; }

  /// Compute the analysis on operations rooted under the given top-level
  /// operation. Note that the top-level operation is not visited.
  void run(Operation *topLevelOp) {
    detail::ForwardDataFlowAnalysisBase::run(topLevelOp);
  }

  /// Return the lattice element attached to the given value, or nullptr if no
  /// lattice for the value has yet been created.
  LatticeElement<ValueT> *lookupLatticeElement(Value value) {
    return static_cast<LatticeElement<ValueT> *>(
        detail::ForwardDataFlowAnalysisBase::lookupLatticeElement(value));
  }

protected:
  /// Return the lattice element attached to the given value. If a lattice has
  /// not been added for the given value, a new 'uninitialized' value is
  /// inserted and returned.
  LatticeElement<ValueT> &getLatticeElement(Value value) {
    return static_cast<LatticeElement<ValueT> &>(
        detail::ForwardDataFlowAnalysisBase::getLatticeElement(value));
  }

  /// Mark all of the lattices for the given range of Values as having reached a
  /// pessimistic fixpoint.
  ChangeResult markAllPessimisticFixpoint(ValueRange values) {
    ChangeResult result = ChangeResult::NoChange;
    for (Value value : values)
      result |= getLatticeElement(value).markPessimisticFixpoint();
    return result;
  }

  /// Visit the given operation, and join any necessary analysis state
  /// into the lattices for the results and block arguments owned by this
  /// operation using the provided set of operand lattice elements (all pointer
  /// values are guaranteed to be non-null). Returns if any result or block
  /// argument value lattices changed during the visit. The lattice for a result
  /// or block argument value can be obtained by using
  /// `getLatticeElement`.
  virtual ChangeResult
  visitOperation(Operation *op,
                 ArrayRef<LatticeElement<ValueT> *> operands) = 0;

  /// Given a BranchOpInterface, and the current lattice elements that
  /// correspond to the branch operands (all pointer values are guaranteed to be
  /// non-null), try to compute a specific set of successors that would be
  /// selected for the branch. Returns failure if not computable, or if all of
  /// the successors would be chosen. If a subset of successors can be selected,
  /// `successors` is populated.
  virtual LogicalResult
  getSuccessorsForOperands(BranchOpInterface branch,
                           ArrayRef<LatticeElement<ValueT> *> operands,
                           SmallVectorImpl<Block *> &successors) {
    return failure();
  }

  /// Given a RegionBranchOpInterface, and the current lattice elements that
  /// correspond to the branch operands (all pointer values are guaranteed to be
  /// non-null), compute a specific set of region successors that would be
  /// selected.
  virtual void
  getSuccessorsForOperands(RegionBranchOpInterface branch,
                           Optional<unsigned> sourceIndex,
                           ArrayRef<LatticeElement<ValueT> *> operands,
                           SmallVectorImpl<RegionSuccessor> &successors) {
    SmallVector<Attribute> constantOperands(operands.size());
    branch.getSuccessorRegions(sourceIndex, constantOperands, successors);
  }

  /// Given a operation with successor regions, one of those regions,
  /// and the lattice elements corresponding to the operation's
  /// arguments, compute the latice values for block arguments
  /// that are not accounted for by the branching control flow (ex. the
  /// bounds of loops). By default, this method marks all such lattice elements
  /// as having reached a pessimistic fixpoint. The region in the
  /// RegionSuccessor and the operand latice elements are guaranteed to be
  /// non-null.
  virtual ChangeResult
  visitNonControlFlowArguments(Operation *op, const RegionSuccessor &successor,
                               ArrayRef<LatticeElement<ValueT> *> operands) {
    ChangeResult result = ChangeResult::NoChange;
    Region *region = successor.getSuccessor();
    ValueRange succArgs = successor.getSuccessorInputs();
    Block *block = &region->front();
    Block::BlockArgListType arguments = block->getArguments();
    if (arguments.size() != succArgs.size()) {
      unsigned firstArgIdx =
          succArgs.empty() ? 0
                           : succArgs[0].cast<BlockArgument>().getArgNumber();
      result |= markAllPessimisticFixpoint(arguments.take_front(firstArgIdx));
      result |= markAllPessimisticFixpoint(
          arguments.drop_front(firstArgIdx + succArgs.size()));
    }
    return result;
  }

private:
  /// Type-erased wrappers that convert the abstract lattice operands to derived
  /// lattices and invoke the virtual hooks operating on the derived lattices.
  ChangeResult
  visitOperation(Operation *op,
                 ArrayRef<detail::AbstractLatticeElement *> operands) final {
    LatticeElement<ValueT> *const *derivedOperandBase =
        reinterpret_cast<LatticeElement<ValueT> *const *>(operands.data());
    return visitOperation(
        op, llvm::makeArrayRef(derivedOperandBase, operands.size()));
  }
  LogicalResult
  getSuccessorsForOperands(BranchOpInterface branch,
                           ArrayRef<detail::AbstractLatticeElement *> operands,
                           SmallVectorImpl<Block *> &successors) final {
    LatticeElement<ValueT> *const *derivedOperandBase =
        reinterpret_cast<LatticeElement<ValueT> *const *>(operands.data());
    return getSuccessorsForOperands(
        branch, llvm::makeArrayRef(derivedOperandBase, operands.size()),
        successors);
  }
  void
  getSuccessorsForOperands(RegionBranchOpInterface branch,
                           Optional<unsigned> sourceIndex,
                           ArrayRef<detail::AbstractLatticeElement *> operands,
                           SmallVectorImpl<RegionSuccessor> &successors) final {
    LatticeElement<ValueT> *const *derivedOperandBase =
        reinterpret_cast<LatticeElement<ValueT> *const *>(operands.data());
    getSuccessorsForOperands(
        branch, sourceIndex,
        llvm::makeArrayRef(derivedOperandBase, operands.size()), successors);
  }
  ChangeResult visitNonControlFlowArguments(
      Operation *op, const RegionSuccessor &region,
      ArrayRef<detail::AbstractLatticeElement *> operands) final {
    LatticeElement<ValueT> *const *derivedOperandBase =
        reinterpret_cast<LatticeElement<ValueT> *const *>(operands.data());
    return visitNonControlFlowArguments(
        op, region, llvm::makeArrayRef(derivedOperandBase, operands.size()));
  }

  /// Create a new uninitialized lattice element. An optional value is provided,
  /// which if valid, should be used to initialize the known conservative state
  /// of the lattice.
  detail::AbstractLatticeElement *createLatticeElement(Value value) final {
    ValueT knownValue = value ? ValueT::getPessimisticValueState(value)
                              : ValueT::getPessimisticValueState(context);
    return new (allocator.Allocate()) LatticeElement<ValueT>(knownValue);
  }

  /// An allocator used for new lattice elements.
  llvm::SpecificBumpPtrAllocator<LatticeElement<ValueT>> allocator;

  /// The MLIRContext of this solver.
  MLIRContext *context;
};

} // namespace mlir

#endif // MLIR_ANALYSIS_DATAFLOWANALYSIS_H
