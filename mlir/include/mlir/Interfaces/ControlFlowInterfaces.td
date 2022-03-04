//===-- ControlFlowInterfaces.td - ControlFlow Interfaces --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a set of interfaces that can be used to define information
// about control flow operations, e.g. branches.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_CONTROLFLOWINTERFACES
#define MLIR_INTERFACES_CONTROLFLOWINTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// BranchOpInterface
//===----------------------------------------------------------------------===//

def BranchOpInterface : OpInterface<"BranchOpInterface"> {
  let description = [{
    This interface provides information for branching terminator operations,
    i.e. terminator operations with successors.

    This interface is meant to model well-defined cases of control-flow of
    value propagation, where what occurs along control-flow edges is assumed to
    be side-effect free. For example, corresponding successor operands and
    successor block arguments may have different types. In such cases,
    `areTypesCompatible` can be implemented to compare types along control-flow
    edges. By default, type equality is used.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Returns a mutable range of operands that correspond to the arguments of
        successor at the given index. Returns None if the operands to the
        successor are non-materialized values, i.e. they are internal to the
        operation.
      }],
      "::mlir::Optional<::mlir::MutableOperandRange>", "getMutableSuccessorOperands",
      (ins "unsigned":$index)
    >,
    InterfaceMethod<[{
        Returns a range of operands that correspond to the arguments of
        successor at the given index. Returns None if the operands to the
        successor are non-materialized values, i.e. they are internal to the
        operation.
      }],
      "::mlir::Optional<::mlir::OperandRange>", "getSuccessorOperands",
      (ins "unsigned":$index), [{}], [{
        auto operands = $_op.getMutableSuccessorOperands(index);
        return operands ? ::mlir::Optional<::mlir::OperandRange>(*operands) : ::llvm::None;
      }]
    >,
    InterfaceMethod<[{
        Returns the `BlockArgument` corresponding to operand `operandIndex` in
        some successor, or None if `operandIndex` isn't a successor operand
        index.
      }],
      "::mlir::Optional<::mlir::BlockArgument>", "getSuccessorBlockArgument",
      (ins "unsigned":$operandIndex), [{
        ::mlir::Operation *opaqueOp = $_op;
        for (unsigned i = 0, e = opaqueOp->getNumSuccessors(); i != e; ++i) {
          if (::mlir::Optional<::mlir::BlockArgument> arg = ::mlir::detail::getBranchSuccessorArgument(
                $_op.getSuccessorOperands(i), operandIndex,
                opaqueOp->getSuccessor(i)))
            return arg;
        }
        return ::llvm::None;
      }]
    >,
    InterfaceMethod<[{
        Returns the successor that would be chosen with the given constant
        operands. Returns nullptr if a single successor could not be chosen.
      }],
      "::mlir::Block *", "getSuccessorForOperands",
      (ins "::mlir::ArrayRef<::mlir::Attribute>":$operands), [{}],
      /*defaultImplementation=*/[{ return nullptr; }]
    >,
    InterfaceMethod<[{
        This method is called to compare types along control-flow edges. By
        default, the types are checked as equal.
      }],
      "bool", "areTypesCompatible",
      (ins "::mlir::Type":$lhs, "::mlir::Type":$rhs), [{}],
       [{ return lhs == rhs; }]
    >,
  ];

  let verify = [{
    auto concreteOp = ::mlir::cast<ConcreteOp>($_op);
    for (unsigned i = 0, e = $_op->getNumSuccessors(); i != e; ++i) {
      ::mlir::Optional<OperandRange> operands = concreteOp.getSuccessorOperands(i);
      if (::mlir::failed(::mlir::detail::verifyBranchSuccessorOperands($_op, i, operands)))
        return ::mlir::failure();
    }
    return ::mlir::success();
  }];
}

//===----------------------------------------------------------------------===//
// RegionBranchOpInterface
//===----------------------------------------------------------------------===//

def RegionBranchOpInterface : OpInterface<"RegionBranchOpInterface"> {
  let description = [{
    This interface provides information for region operations that contain
    branching behavior between held regions, i.e. this interface allows for
    expressing control flow information for region holding operations.

    This interface is meant to model well-defined cases of control-flow of
    value propagation, where what occurs along control-flow edges is assumed to
    be side-effect free. For example, corresponding successor operands and
    successor block arguments may have different types. In such cases,
    `areTypesCompatible` can be implemented to compare types along control-flow
    edges. By default, type equality is used.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Returns the operands of this operation used as the entry arguments when
        entering the region at `index`, which was specified as a successor of
        this operation by `getSuccessorRegions`. These operands should
        correspond 1-1 with the successor inputs specified in
        `getSuccessorRegions`.
      }],
      "::mlir::OperandRange", "getSuccessorEntryOperands",
      (ins "unsigned":$index), [{}], /*defaultImplementation=*/[{
        auto operandEnd = this->getOperation()->operand_end();
        return ::mlir::OperandRange(operandEnd, operandEnd);
      }]
    >,
    InterfaceMethod<[{
        Returns the viable successors of a region at `index`, or the possible
        successors when branching from the parent op if `index` is None. These
        are the regions that may be selected during the flow of control. If
        `index` is None, `operands` is a set of optional attributes that
        either correspond to a constant value for each operand of this
        operation, or null if that operand is not a constant. If `index` is
        valid, `operands` corresponds to the entry values of the region at
        `index`. Only a region, i.e. a valid `index`, may use the parent
        operation as a successor. This method allows for describing which
        regions may be executed when entering an operation, and which regions
        are executed after having executed another region of the parent op. The
        successor region must be non-empty.
      }],
      "void", "getSuccessorRegions",
      (ins "::mlir::Optional<unsigned>":$index,
           "::mlir::ArrayRef<::mlir::Attribute>":$operands,
           "::mlir::SmallVectorImpl<::mlir::RegionSuccessor> &":$regions)
    >,
    InterfaceMethod<[{
        Populates `invocationBounds` with the minimum and maximum number of
        times this operation will invoke the attached regions (assuming the
        regions yield normally, i.e. do not abort or invoke an infinite loop).
        The minimum number of invocations is at least 0. If the maximum number
        of invocations cannot be statically determined, then it will not have a
        value (i.e., it is set to `llvm::None`).

        `operands` is a set of optional attributes that either correspond to
        constant values for each operand of this operation or null if that
        operand is not a constant.

        This method may be called speculatively on operations where the provided
        operands are not necessarily the same as the operation's current
        operands. This may occur in analyses that wish to determine "what would
        be the region invocations if these were the operands?"
      }],
      "void", "getRegionInvocationBounds",
      (ins "::mlir::ArrayRef<::mlir::Attribute>":$operands,
           "::llvm::SmallVectorImpl<::mlir::InvocationBounds> &"
             :$invocationBounds), [{}],
       [{ invocationBounds.append($_op->getNumRegions(),
                                  ::mlir::InvocationBounds::getUnknown()); }]
    >,
    InterfaceMethod<[{
        This method is called to compare types along control-flow edges. By
        default, the types are checked as equal.
      }],
      "bool", "areTypesCompatible",
      (ins "::mlir::Type":$lhs, "::mlir::Type":$rhs), [{}],
       [{ return lhs == rhs; }]
    >,
  ];

  let verify = [{
    static_assert(!ConcreteOp::template hasTrait<OpTrait::ZeroRegion>(),
                  "expected operation to have non-zero regions");
    return detail::verifyTypesAlongControlFlowEdges($_op);
  }];

  let extraClassDeclaration = [{
    /// Convenience helper in case none of the operands is known.
    void getSuccessorRegions(Optional<unsigned> index,
                             SmallVectorImpl<RegionSuccessor> &regions) {
       SmallVector<Attribute, 2> nullAttrs(getOperation()->getNumOperands());
       getSuccessorRegions(index, nullAttrs, regions);
    }
  }];
}

//===----------------------------------------------------------------------===//
// RegionBranchTerminatorOpInterface
//===----------------------------------------------------------------------===//

def RegionBranchTerminatorOpInterface :
  OpInterface<"RegionBranchTerminatorOpInterface"> {
  let description = [{
    This interface provides information for branching terminator operations
    in the presence of a parent RegionBranchOpInterface implementation. It
    specifies which operands are passed to which successor region.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Returns a mutable range of operands that are semantically "returned" by
        passing them to the region successor given by `index`.  If `index` is
        None, this function returns the operands that are passed as a result to
        the parent operation.
      }],
      "::mlir::MutableOperandRange", "getMutableSuccessorOperands",
      (ins "::mlir::Optional<unsigned>":$index)
    >,
    InterfaceMethod<[{
        Returns a range of operands that are semantically "returned" by passing
        them to the region successor given by `index`.  If `index` is None, this
        function returns the operands that are passed as a result to the parent
        operation.
      }],
      "::mlir::OperandRange", "getSuccessorOperands",
      (ins "::mlir::Optional<unsigned>":$index), [{}],
      /*defaultImplementation=*/[{
        return $_op.getMutableSuccessorOperands(index);
      }]
    >
  ];

  let verify = [{
    static_assert(ConcreteOp::template hasTrait<OpTrait::IsTerminator>(),
                  "expected operation to be a terminator");
    static_assert(ConcreteOp::template hasTrait<OpTrait::ZeroResult>(),
                  "expected operation to have zero results");
    static_assert(ConcreteOp::template hasTrait<OpTrait::ZeroSuccessor>(),
                  "expected operation to have zero successors");
    return success();
  }];
}

//===----------------------------------------------------------------------===//
// ControlFlow Traits
//===----------------------------------------------------------------------===//

// Op is "return-like".
def ReturnLike : NativeOpTrait<"ReturnLike">;

#endif // MLIR_INTERFACES_CONTROLFLOWINTERFACES
