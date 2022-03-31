//===- ControlFlowInterfaces.h - ControlFlow Interfaces ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the branch interfaces defined in
// `ControlFlowInterfaces.td`.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_CONTROLFLOWINTERFACES_H
#define MLIR_INTERFACES_CONTROLFLOWINTERFACES_H

#include "mlir/IR/OpDefinition.h"

namespace mlir {
class BranchOpInterface;
class RegionBranchOpInterface;

//===----------------------------------------------------------------------===//
// BranchOpInterface
//===----------------------------------------------------------------------===//

namespace detail {
/// Return the `BlockArgument` corresponding to operand `operandIndex` in some
/// successor if `operandIndex` is within the range of `operands`, or None if
/// `operandIndex` isn't a successor operand index.
Optional<BlockArgument>
getBranchSuccessorArgument(Optional<OperandRange> operands,
                           unsigned operandIndex, Block *successor);

/// Verify that the given operands match those of the given successor block.
LogicalResult verifyBranchSuccessorOperands(Operation *op, unsigned succNo,
                                            Optional<OperandRange> operands);
} // namespace detail

//===----------------------------------------------------------------------===//
// RegionBranchOpInterface
//===----------------------------------------------------------------------===//

namespace detail {
/// Verify that types match along control flow edges described the given op.
LogicalResult verifyTypesAlongControlFlowEdges(Operation *op);
} //  namespace detail

/// This class represents a successor of a region. A region successor can either
/// be another region, or the parent operation. If the successor is a region,
/// this class represents the destination region, as well as a set of arguments
/// from that region that will be populated when control flows into the region.
/// If the successor is the parent operation, this class represents an optional
/// set of results that will be populated when control returns to the parent
/// operation.
///
/// This interface assumes that the values from the current region that are used
/// to populate the successor inputs are the operands of the return-like
/// terminator operations in the blocks within this region.
class RegionSuccessor {
public:
  /// Initialize a successor that branches to another region of the parent
  /// operation.
  RegionSuccessor(Region *region, Block::BlockArgListType regionInputs = {})
      : region(region), inputs(regionInputs) {}
  /// Initialize a successor that branches back to/out of the parent operation.
  RegionSuccessor(Optional<Operation::result_range> results = {})
      : inputs(results ? ValueRange(*results) : ValueRange()) {}

  /// Return the given region successor. Returns nullptr if the successor is the
  /// parent operation.
  Region *getSuccessor() const { return region; }

  /// Return true if the successor is the parent operation.
  bool isParent() const { return region == nullptr; }

  /// Return the inputs to the successor that are remapped by the exit values of
  /// the current region.
  ValueRange getSuccessorInputs() const { return inputs; }

private:
  Region *region{nullptr};
  ValueRange inputs;
};

/// This class represents upper and lower bounds on the number of times a region
/// of a `RegionBranchOpInterface` can be invoked. The lower bound is at least
/// zero, but the upper bound may not be known.
class InvocationBounds {
public:
  /// Create invocation bounds. The lower bound must be at least 0 and only the
  /// upper bound can be unknown.
  InvocationBounds(unsigned lb, Optional<unsigned> ub) : lower(lb), upper(ub) {
    assert((!ub || ub >= lb) && "upper bound cannot be less than lower bound");
  }

  /// Return the lower bound.
  unsigned getLowerBound() const { return lower; }

  /// Return the upper bound.
  Optional<unsigned> getUpperBound() const { return upper; }

  /// Returns the unknown invocation bounds, i.e., there is no information on
  /// how many times a region may be invoked.
  static InvocationBounds getUnknown() { return {0, llvm::None}; }

private:
  /// The minimum number of times the successor region will be invoked.
  unsigned lower;
  /// The maximum number of times the successor region will be invoked or `None`
  /// if an upper bound is not known.
  Optional<unsigned> upper;
};

/// Return `true` if `a` and `b` are in mutually exclusive regions as per
/// RegionBranchOpInterface.
bool insideMutuallyExclusiveRegions(Operation *a, Operation *b);

//===----------------------------------------------------------------------===//
// RegionBranchTerminatorOpInterface
//===----------------------------------------------------------------------===//

/// Returns true if the given operation is either annotated with the
/// `ReturnLike` trait or implements the `RegionBranchTerminatorOpInterface`.
bool isRegionReturnLike(Operation *operation);

/// Returns the mutable operands that are passed to the region with the given
/// `regionIndex`. If the operation does not implement the
/// `RegionBranchTerminatorOpInterface` and is not marked as `ReturnLike`, the
/// result will be `llvm::None`. In all other cases, the resulting
/// `OperandRange` represents all operands that are passed to the specified
/// successor region. If `regionIndex` is `llvm::None`, all operands that are
/// passed to the parent operation will be returned.
Optional<MutableOperandRange>
getMutableRegionBranchSuccessorOperands(Operation *operation,
                                        Optional<unsigned> regionIndex);

/// Returns the read only operands that are passed to the region with the given
/// `regionIndex`. See `getMutableRegionBranchSuccessorOperands` for more
/// information.
Optional<OperandRange>
getRegionBranchSuccessorOperands(Operation *operation,
                                 Optional<unsigned> regionIndex);

//===----------------------------------------------------------------------===//
// ControlFlow Traits
//===----------------------------------------------------------------------===//

namespace OpTrait {
/// This trait indicates that a terminator operation is "return-like". This
/// means that it exits its current region and forwards its operands as "exit"
/// values to the parent region. Operations with this trait are not permitted to
/// contain successors or produce results.
template <typename ConcreteType>
struct ReturnLike : public TraitBase<ConcreteType, ReturnLike> {
  static LogicalResult verifyTrait(Operation *op) {
    static_assert(ConcreteType::template hasTrait<IsTerminator>(),
                  "expected operation to be a terminator");
    static_assert(ConcreteType::template hasTrait<ZeroResult>(),
                  "expected operation to have zero results");
    static_assert(ConcreteType::template hasTrait<ZeroSuccessor>(),
                  "expected operation to have zero successors");
    return success();
  }
};
} // namespace OpTrait

} // namespace mlir

//===----------------------------------------------------------------------===//
// ControlFlow Interfaces
//===----------------------------------------------------------------------===//

/// Include the generated interface declarations.
#include "mlir/Interfaces/ControlFlowInterfaces.h.inc"

#endif // MLIR_INTERFACES_CONTROLFLOWINTERFACES_H
