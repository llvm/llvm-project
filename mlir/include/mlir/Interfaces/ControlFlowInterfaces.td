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
        Returns the operands that correspond to the arguments of the successor
        at the given index. It consists of a number of operands that are
        internally produced by the operation, followed by a range of operands
        that are forwarded. An example operation making use of produced
        operands would be:

        ```mlir
        invoke %function(%0)
            label ^success ^error(%1 : i32)

        ^error(%e: !error, %arg0: i32):
            ...
        ```

        The operand that would map to the `^error`s `%e` operand is produced
        by the `invoke` operation, while `%1` is a forwarded operand that maps
        to `%arg0` in the successor.

        Produced operands always map to the first few block arguments of the
        successor, followed by the forwarded operands. Mapping them in any
        other order is not supported by the interface.

        By having the forwarded operands last allows users of the interface
        to append more forwarded operands to the branch operation without
        interfering with other successor operands.
      }],
      "::mlir::SuccessorOperands", "getSuccessorOperands",
      (ins "unsigned":$index)
    >,
    InterfaceMethod<[{
        Returns the `BlockArgument` corresponding to operand `operandIndex` in
        some successor, or std::nullopt if `operandIndex` isn't a successor operand
        index.
      }],
      "::std::optional<::mlir::BlockArgument>", "getSuccessorBlockArgument",
      (ins "unsigned":$operandIndex), [{
        ::mlir::Operation *opaqueOp = $_op;
        for (unsigned i = 0, e = opaqueOp->getNumSuccessors(); i != e; ++i) {
          if (::std::optional<::mlir::BlockArgument> arg = ::mlir::detail::getBranchSuccessorArgument(
                $_op.getSuccessorOperands(i), operandIndex,
                opaqueOp->getSuccessor(i)))
            return arg;
        }
        return ::std::nullopt;
      }]
    >,
    InterfaceMethod<[{
        Returns the successor that would be chosen with the given constant
        operands. Returns nullptr if a single successor could not be chosen.
      }],
      "::mlir::Block *", "getSuccessorForOperands",
      (ins "::llvm::ArrayRef<::mlir::Attribute>":$operands), [{}],
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
      ::mlir::SuccessorOperands operands = concreteOp.getSuccessorOperands(i);
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
    This interface provides information for region operations that exhibit
    branching behavior between held regions. I.e., this interface allows for
    expressing control flow information for region holding operations.

    This interface is meant to model well-defined cases of control-flow and
    value propagation, where what occurs along control-flow edges is assumed to
    be side-effect free.

    A "region branch point" indicates a point from which a branch originates. It
    can indicate either a region of this op or `RegionBranchPoint::parent()`. In
    the latter case, the branch originates from outside of the op, i.e., when
    first executing this op.

    A "region successor" indicates the target of a branch. It can indicate
    either a region of this op or this op. In the former case, the region
    successor is a region pointer and a range of block arguments to which the
    "successor operands" are forwarded to. In the latter case, the control flow
    leaves this op and the region successor is a range of results of this op to
    which the successor operands are forwarded to.

    By default, successor operands and successor block arguments/successor
    results must have the same type. `areTypesCompatible` can be implemented to
    allow non-equal types.

    Example:

    ```
    %r = scf.for %iv = %lb to %ub step %step iter_args(%a = %b)
        -> tensor<5xf32> {
      ...
      scf.yield %c : tensor<5xf32>
    }
    ```

    `scf.for` has one region. The region has two region successors: the region
    itself and the `scf.for` op. %b is an entry successor operand. %c is a
    successor operand. %a is a successor block argument. %r is a successor
    result.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Returns the operands of this operation that are forwarded to the region
        successor's block arguments or this operation's results when branching
        to `point`. `point` is guaranteed to be among the successors that are
        returned by `getEntrySuccessorRegions`/`getSuccessorRegions(parent())`.

        Example: In the above example, this method returns the operand %b of the
        `scf.for` op, regardless of the value of `point`. I.e., this op always
        forwards the same operands, regardless of whether the loop has 0 or more
        iterations.
      }],
      "::mlir::OperandRange", "getEntrySuccessorOperands",
      (ins "::mlir::RegionBranchPoint":$point), [{}],
      /*defaultImplementation=*/[{
        auto operandEnd = this->getOperation()->operand_end();
        return ::mlir::OperandRange(operandEnd, operandEnd);
      }]
    >,
    InterfaceMethod<[{
        Returns the potential region successors when first executing the op.

        Unlike `getSuccessorRegions`, this method also passes along the
        constant operands of this op. Based on these, the implementation may
        filter out certain successors. By default, simply dispatches to
        `getSuccessorRegions`. `operands` contains an entry for every
        operand of this op, with a null attribute if the operand has no constant
        value.

        Note: The control flow does not necessarily have to enter any region of
        this op.

        Example: In the above example, this method may return two region
        region successors: the single region of the `scf.for` op and the
        `scf.for` operation (that implements this interface). If %lb, %ub, %step
        are constants and it can be determined the loop does not have any
        iterations, this method may choose to return only this operation.
        Similarly, if it can be determined that the loop has at least one
        iteration, this method may choose to return only the region of the loop.
      }],
      "void", "getEntrySuccessorRegions",
      (ins "::llvm::ArrayRef<::mlir::Attribute>":$operands,
           "::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &":$regions), [{}],
      /*defaultImplementation=*/[{
        $_op.getSuccessorRegions(mlir::RegionBranchPoint::parent(), regions);
      }]
    >,
    InterfaceMethod<[{
        Returns the potential region successors when branching from `point`.
        These are the regions that may be selected during the flow of control.

        When `point = RegionBranchPoint::parent()`, this method returns the
        region successors when entering the operation. Otherwise, this method
        returns the successor regions when branching from the region indicated
        by `point`.

        Example: In the above example, this method returns the region of the
        `scf.for` and this operation for either region branch point (`parent`
        and the region of the `scf.for`). An implementation may choose to filter
        out region successors when it is statically known (e.g., by examining
        the operands of this op) that those successors are not branched to.
      }],
      "void", "getSuccessorRegions",
      (ins "::mlir::RegionBranchPoint":$point,
           "::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &":$regions)
    >,
    InterfaceMethod<[{
        Populates `invocationBounds` with the minimum and maximum number of
        times this operation will invoke the attached regions (assuming the
        regions yield normally, i.e. do not abort or invoke an infinite loop).
        The minimum number of invocations is at least 0. If the maximum number
        of invocations cannot be statically determined, then it will be set to
        `InvocationBounds::getUnknown()`.

        This method also passes along the constant operands of this op.
        `operands` contains an entry for every operand of this op, with a null
        attribute if the operand has no constant value.

        This method may be called speculatively on operations where the provided
        operands are not necessarily the same as the operation's current
        operands. This may occur in analyses that wish to determine "what would
        be the region invocations if these were the operands?"
      }],
      "void", "getRegionInvocationBounds",
      (ins "::llvm::ArrayRef<::mlir::Attribute>":$operands,
           "::llvm::SmallVectorImpl<::mlir::InvocationBounds> &"
             :$invocationBounds), [{}],
      /*defaultImplementation=*/[{
        invocationBounds.append($_op->getNumRegions(),
                                ::mlir::InvocationBounds::getUnknown());
      }]
    >,
    InterfaceMethod<[{
        This method is called to compare types along control-flow edges. By
        default, the types are checked as equal.
      }],
      "bool", "areTypesCompatible",
      (ins "::mlir::Type":$lhs, "::mlir::Type":$rhs), [{}],
      /*defaultImplementation=*/[{ return lhs == rhs; }]
    >,
  ];

  let verify = [{
    static_assert(!ConcreteOp::template hasTrait<OpTrait::ZeroRegions>(),
                  "expected operation to have non-zero regions");
    return detail::verifyTypesAlongControlFlowEdges($_op);
  }];
  let verifyWithRegions = 1;

  let extraClassDeclaration = [{
    /// Return `true` if control flow originating from the given region may
    /// eventually branch back to the same region. (Maybe after passing through
    /// other regions.)
    bool isRepetitiveRegion(unsigned index);
  }];
}

//===----------------------------------------------------------------------===//
// RegionBranchTerminatorOpInterface
//===----------------------------------------------------------------------===//

def RegionBranchTerminatorOpInterface :
  OpInterface<"RegionBranchTerminatorOpInterface"> {
  let description = [{
    This interface provides information for branching terminator operations
    in the presence of a parent `RegionBranchOpInterface` implementation. It
    specifies which operands are passed to which successor region.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Returns a mutable range of operands that are semantically "returned" by
        passing them to the region successor indicated by `point`.
      }],
      "::mlir::MutableOperandRange", "getMutableSuccessorOperands",
      (ins "::mlir::RegionBranchPoint":$point)
    >,
    InterfaceMethod<[{
        Returns the potential region successors that are branched to after this
        terminator based on the given constant operands.

        This method also passes along the constant operands of this op.
        `operands` contains an entry for every operand of this op, with a null
        attribute if the operand has no constant value.

        The default implementation simply dispatches to the parent
        `RegionBranchOpInterface`'s `getSuccessorRegions` implementation.
      }],
      "void", "getSuccessorRegions",
      (ins "::llvm::ArrayRef<::mlir::Attribute>":$operands,
           "::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &":$regions), [{}],
      /*defaultImplementation=*/[{
        ::mlir::Operation *op = $_op;
        ::llvm::cast<::mlir::RegionBranchOpInterface>(op->getParentOp())
          .getSuccessorRegions(op->getParentRegion(), regions);
      }]
    >,
  ];

  let verify = [{
    static_assert(ConcreteOp::template hasTrait<OpTrait::IsTerminator>(),
                  "expected operation to be a terminator");
    static_assert(ConcreteOp::template hasTrait<OpTrait::ZeroResults>(),
                  "expected operation to have zero results");
    static_assert(ConcreteOp::template hasTrait<OpTrait::ZeroSuccessors>(),
                  "expected operation to have zero successors");
    return success();
  }];

  let extraClassDeclaration = [{
    // Returns a range of operands that are semantically "returned" by passing
    // them to the region successor given by `index`.  If `index` is None, this
    // function returns the operands that are passed as a result to the parent
    // operation.
    ::mlir::OperandRange getSuccessorOperands(::mlir::RegionBranchPoint point) {
      return getMutableSuccessorOperands(point);
    }
  }];
}

//===----------------------------------------------------------------------===//
// ControlFlow Traits
//===----------------------------------------------------------------------===//

// Op is "return-like".
def ReturnLike : TraitList<[
    DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface>,
    NativeOpTrait<
        /*name=*/"ReturnLike",
        /*traits=*/[],
        /*extraOpDeclaration=*/"",
        /*extraOpDefinition=*/[{
          ::mlir::MutableOperandRange $cppClass::getMutableSuccessorOperands(
            ::mlir::RegionBranchPoint point) {
            return ::mlir::MutableOperandRange(*this);
          }
        }]
    >
]>;

#endif // MLIR_INTERFACES_CONTROLFLOWINTERFACES
