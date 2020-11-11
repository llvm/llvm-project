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

// A constant value to represent unknown number of region invocations.
extern const int64_t kUnknownNumRegionInvocations;

namespace detail {
/// Verify that types match along control flow edges described the given op.
LogicalResult verifyTypesAlongControlFlowEdges(Operation *op);
} //  namespace detail

/// This class represents a successor of a region. A region successor can either
/// be another region, or the parent operation. If the successor is a region,
/// this class represents the destination region, as well as a set of arguments
/// from that region that will be populated by values from the current region.
/// If the successor is the parent operation, this class represents an optional
/// set of results that will be populated by values from the current region.
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
      : region(nullptr), inputs(results ? ValueRange(*results) : ValueRange()) {
  }

  /// Return the given region successor. Returns nullptr if the successor is the
  /// parent operation.
  Region *getSuccessor() const { return region; }

  /// Return true if the successor is the parent operation.
  bool isParent() const { return region == nullptr; }

  /// Return the inputs to the successor that are remapped by the exit values of
  /// the current region.
  ValueRange getSuccessorInputs() const { return inputs; }

private:
  Region *region;
  ValueRange inputs;
};

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

} // end namespace mlir

//===----------------------------------------------------------------------===//
// ControlFlow Interfaces
//===----------------------------------------------------------------------===//

/// Include the generated interface declarations.
#include "mlir/Interfaces/ControlFlowInterfaces.h.inc"

#endif // MLIR_INTERFACES_CONTROLFLOWINTERFACES_H
