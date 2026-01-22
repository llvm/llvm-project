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
    This interface provides information for region-holding operations that
    exhibit branching behavior between held regions. I.e., this interface allows
    for expressing control flow information for region holding operations.

    This interface is meant to model well-defined cases of control-flow and
    value propagation, where what occurs along control-flow edges is assumed to
    be side-effect free.

    A "region branch point" indicates a point from which a branch originates. It
    can indicate:
    1. A `RegionBranchTerminatorOpInterface` terminator in any of the
       immediately nested regions of this op.
    2. `RegionBranchPoint::parent()`: the branch originates from outside of the
       op, i.e., when first executing this op.

    When branching from a region branch point to a region successor, the
    "successor operands" to be forwarded from the region branch point can be
    specified with `getEntrySuccessorOperands` /
    `RegionBranchTerminatorOpInterface::getSuccessorOperands`.

    A "region successor" indicates the target of a branch. It can indicate:
    1. A region of this op.
    2. `RegionSuccessor::parent()`, i.e., the control flow leaves this op.

    The SSA values to which successor operands are forwarded are called
    "successor inputs".

    By default, successor operands and successor block arguments/successor
    results must have the same type. `areTypesCompatible` can be implemented to
    allow non-equal types.


    Note: This interface works in conjunction with
    `RegionBranchTerminatorOpInterface`. All immediately nested block
    terminators that model branching between regions must implement the
    `RegionBranchTerminatorOpInterface`. Otherwise, analyses/transformations
    may miss control flow edges and produce incorrect results. Not every block
    terminator is necessarily a region branch terminator: e.g., in the presence
    of unstructured control flow, a block terminator could indicate a branch to
    a different block within the same region.

    Example:

    ```
    %r = scf.for %iv = %lb to %ub step %step iter_args(%a = %b)
        -> tensor<5xf32> {
      ...
      scf.yield %c : tensor<5xf32>
    }
    ```

    `scf.for` has one region. There are two region branch points with two
    identical region successors:
    * parent => parent(%r), region0(%a)
    * `scf.yield` => parent(%r), region0(%a)

    `%a` and %r are successor inputs. `%b` is an entry successor operand. `%c`
    is a successor operand.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Returns the operands of this operation that are forwarded to the
        successor inputs when branching to `successor`. `successor` is
        guaranteed to be among the successors that are returned by
        `getEntrySuccessorRegions`/`getSuccessorRegions(parent())`.

        Example: In the above example, this method returns the operand %b of the
        `scf.for` op, regardless of the value of `successor`. I.e., this op always
        forwards the same operands, regardless of whether the loop has 0 or more
        iterations.
      }],
      "::mlir::OperandRange", "getEntrySuccessorOperands",
      (ins "::mlir::RegionSuccessor":$successor), [{}],
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
        Returns all potential region successors when branching from `point`.
        These are the regions that may be selected during the flow of control.

        When `point = RegionBranchPoint::parent()`, this method returns the
        region successors when entering the operation. Otherwise, this method
        returns the successor regions when branching from the region indicated
        by `point`.

        Example: In the above example, this method returns the region of the
        `scf.for` and `parent` for either region branch point. An implementation
        may choose to filter out region successors when it is statically known
        (e.g., by examining the operands of this op) that those successors are
        not branched to.
      }],
      "void", "getSuccessorRegions",
      (ins "::mlir::RegionBranchPoint":$point,
           "::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &":$regions)
    >,
    InterfaceMethod<[{
        Returns the potential region successors when branching from any
        terminator in `region`.
      }],
      "void", "getSuccessorRegions",
      (ins "::mlir::Region&":$region,
           "::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &":$regions),
      [{}],
      /*defaultImplementation=*/[{
        for (::mlir::Block &block : region) {
          if (block.empty())
            continue;
          if (auto terminator =
                  dyn_cast<RegionBranchTerminatorOpInterface>(block.back()))
            $_op.getSuccessorRegions(RegionBranchPoint(terminator),
                                     regions);
        }
      }]>,
    InterfaceMethod<[{
        Return all successor inputs for the given region successor. If the
        given region successor is a region, then the returned values are block
        arguments. Otherwise, if the given region successor is the "parent",
        the returned values are op results.
      }],
      "::mlir::ValueRange", "getSuccessorInputs",
      (ins "::mlir::RegionSuccessor":$successor),
      [{}],
      /*defaultImplementation=*/[{
        // Default implementation: No successor inputs.
        return ::mlir::ValueRange();
      }]>,
    InterfaceMethod<[{
        Returns the potential branching points (predecessors) for a given
        region successor.
      }],
      "void", "getPredecessors",
      (ins "::mlir::RegionSuccessor":$successor,
           "::llvm::SmallVectorImpl<::mlir::RegionBranchPoint> &":$predecessors),
      [{}],
      /*defaultImplementation=*/[{
        auto op = cast<RegionBranchOpInterface>($_op.getOperation());
        for (::mlir::RegionBranchPoint point : op.getAllRegionBranchPoints()) {
          ::llvm::SmallVector<::mlir::RegionSuccessor> successors;
          op.getSuccessorRegions(point, successors);
          bool isPred = llvm::any_of(successors, [&] (const auto &succ) {
            return succ.getSuccessor() == successor.getSuccessor() ||
                   (succ.isParent() && successor.isParent());
            });
          if (isPred)
            predecessors.push_back(point);
        }
      }]>,
    InterfaceMethod<[{
        Returns the potential values across all (predecessors) for a given successor
        input, modeled by its index (its position in the list of values).
      }],
      "void", "getPredecessorValues",
      (ins "::mlir::RegionSuccessor":$successor,
           "int":$index,
           "::llvm::SmallVectorImpl<::mlir::Value> &":$predecessorValues),
      [{}],
      /*defaultImplementation=*/[{
        ::llvm::SmallVector<::mlir::RegionBranchPoint> predecessors;
        $_op.getPredecessors(successor, predecessors);
        for (auto predecessor : predecessors) {
          if (predecessor.isParent()) {
            predecessorValues.push_back($_op.getEntrySuccessorOperands(successor)[index]);
            continue;
          }
          auto terminator = predecessor.getTerminatorPredecessorOrNull();
          predecessorValues.push_back(terminator.getSuccessorOperands(successor)[index]);
        }
      }]>,
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
    return detail::verifyRegionBranchOpInterface($_op);
  }];
  let verifyWithRegions = 1;

  let extraClassDeclaration = [{
    /// Return `true` if control flow originating from the given region may
    /// eventually branch back to the same region. (Maybe after passing through
    /// other regions.)
    bool isRepetitiveRegion(unsigned index);

    /// Return `true` if there is a loop in the region branching graph. Only
    /// reachable regions (starting from the entry regions) are considered.
    bool hasLoop();

    /// Return the successor operands from the source branch point to the
    /// destination region successor.
    ///
    /// If the branch point is the parent op, this function returns entry
    /// successor operands of this op. Otherwise, it returns successor operands
    /// of the respective terminator.
    ::mlir::OperandRange getSuccessorOperands(
        ::mlir::RegionBranchPoint src, ::mlir::RegionSuccessor dest);

    /// Build a mapping from successor operands to successor input. Each
    /// successor operand could be forwarded to multiple successor inputs.
    /// Operands that are not forwarded are not added to the map. Unless a
    /// specific region branch point is specified, this function takes into
    /// account all possible region branch points.
    void getSuccessorOperandInputMapping(
        ::mlir::RegionBranchSuccessorMapping &mapping,
        std::optional<::mlir::RegionBranchPoint> src = std::nullopt);

    /// Build a mapping from successor inputs to successor operands. This is
    /// the same as "getSuccessorOperandInputMapping", but inverted.
    void getSuccessorInputOperandMapping(
        ::mlir::RegionBranchInverseSuccessorMapping &mapping);

    /// Return all possible region branch points: the region branch op itself
    /// and all region branch terminators.
    ::llvm::SmallVector<::mlir::RegionBranchPoint> getAllRegionBranchPoints();
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
    acts as a marker for valid region branch points and specifies which
    operands are passed to which region successor.

    Note: If an operation does not implement the
    `RegionBranchTerminatorOpInterface`, then that op has no region successors.
    (However, there may be other block terminators in the same region that
    implement the `RegionBranchTerminatorOpInterface`, so the enclosing region
    may have region successors.)
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Returns a mutable range of operands that are semantically "returned" by
        passing them to the region successor indicated by `point`.
      }],
      "::mlir::MutableOperandRange", "getMutableSuccessorOperands",
      (ins "::mlir::RegionSuccessor":$point)
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
          .getSuccessorRegions(::llvm::cast<::mlir::RegionBranchTerminatorOpInterface>(op), regions);
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
    ::mlir::OperandRange getSuccessorOperands(::mlir::RegionSuccessor successor) {
      return getMutableSuccessorOperands(successor);
    }
  }];
}

def SelectLikeOpInterface : OpInterface<"SelectLikeOpInterface"> {
  let description = [{
    This interface provides information for select-like operations, i.e.,
    operations that forward specific operands to the output, depending on a
    binary condition.

    If the value of the condition is 1, then the `true` operand is returned,
    and the third operand is ignored, even if it was poison.

    If the value of the condition is 0, then the `false` operand is returned,
    and the second operand is ignored, even if it was poison.

    If the condition is poison, then poison is returned.

    Implementing operations can also accept shaped conditions, in which case
    the operation works element-wise.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Returns the operand that would be chosen for a false condition.
      }], "::mlir::Value", "getFalseValue", (ins)>,
    InterfaceMethod<[{
        Returns the operand that would be chosen for a true condition.
      }], "::mlir::Value", "getTrueValue", (ins)>,
    InterfaceMethod<[{
        Returns the condition operand.
      }], "::mlir::Value", "getCondition", (ins)>
  ];
}

//===----------------------------------------------------------------------===//
// WeightedBranchOpInterface
//===----------------------------------------------------------------------===//

def WeightedBranchOpInterface : OpInterface<"WeightedBranchOpInterface"> {
  let description = [{
    This interface provides weight information for branching terminator
    operations, i.e. terminator operations with successors.

    This interface provides methods for getting/setting integer weights of each
    branch. The probability of executing a branch is computed as the ratio
    between the branch's weight and the total sum of the weights (which cannot
    be zero). The weights are optional. If they are provided, then their number
    must match the number of successors of the operation.

    Note that the branch weight use an i32 representation but they are to be
    interpreted as unsigned integers.

    The default implementations of the methods expect the operation
    to have an attribute of type DenseI32ArrayAttr named branch_weights.
  }];
  let cppNamespace = "::mlir";

  let methods = [InterfaceMethod<
                     /*desc=*/"Returns the branch weights",
                     /*returnType=*/"::llvm::ArrayRef<int32_t>",
                     /*methodName=*/"getWeights",
                     /*args=*/(ins),
                     /*methodBody=*/[{}],
                     /*defaultImpl=*/[{
        auto op = cast<ConcreteOp>(this->getOperation());
        if (auto attr = op.getBranchWeightsAttr())
          return attr.asArrayRef();
        return {};
      }]>,
                 InterfaceMethod<
                     /*desc=*/"Sets the branch weights",
                     /*returnType=*/"void",
                     /*methodName=*/"setWeights",
                     /*args=*/(ins "::llvm::ArrayRef<int32_t>":$weights),
                     /*methodBody=*/[{}],
                     /*defaultImpl=*/[{
        auto op = cast<ConcreteOp>(this->getOperation());
        op.setBranchWeightsAttr(::mlir::DenseI32ArrayAttr::get(op->getContext(), weights));
      }]>,
  ];

  let verify = [{
    return ::mlir::detail::verifyBranchWeights($_op);
  }];
}

//===----------------------------------------------------------------------===//
// WeightedRegionBranchOpInterface
//===----------------------------------------------------------------------===//

// TODO: the probabilities of entering a particular region seem
// to correlate with the values returned by
// RegionBranchOpInterface::invocationBounds(), and we should probably
// verify that the values are consistent. In that case, should
// WeightedRegionBranchOpInterface extend RegionBranchOpInterface?
def WeightedRegionBranchOpInterface
    : OpInterface<"WeightedRegionBranchOpInterface"> {
  let description = [{
    This interface provides weight information for region operations
    that exhibit branching behavior between held regions.

    This interface provides methods for getting/setting integer weights of each
    branch. The probability of executing a region is computed as the ratio
    between the region branch's weight and the total sum of the weights (which
    cannot be zero). The weights are optional. If they are provided, then their
    number must match the number of regions held by the operation (including
    empty regions).

    The weights specify the probability of branching to a particular
    region when first executing the operation.
    For example, for loop-like operations with a single region
    the weight specifies the probability of entering the loop.

    Note that the branch weight use an i32 representation but they are to be
    interpreted as unsigned integers.

    The default implementations of the methods expect the operation
    to have an attribute of type DenseI32ArrayAttr named branch_weights.
  }];
  let cppNamespace = "::mlir";

  let methods = [InterfaceMethod<
                     /*desc=*/"Returns the region weights",
                     /*returnType=*/"::llvm::ArrayRef<int32_t>",
                     /*methodName=*/"getWeights",
                     /*args=*/(ins),
                     /*methodBody=*/[{}],
                     /*defaultImpl=*/[{
        auto op = cast<ConcreteOp>(this->getOperation());
        if (auto attr = op.getRegionWeightsAttr())
          return attr.asArrayRef();
        return {};
      }]>,
                 InterfaceMethod<
                     /*desc=*/"Sets the region weights",
                     /*returnType=*/"void",
                     /*methodName=*/"setWeights",
                     /*args=*/(ins "::llvm::ArrayRef<int32_t>":$weights),
                     /*methodBody=*/[{}],
                     /*defaultImpl=*/[{
        auto op = cast<ConcreteOp>(this->getOperation());
        op.setRegionWeightsAttr(::mlir::DenseI32ArrayAttr::get(op->getContext(), weights));
      }]>,
  ];

  let verify = [{
    return ::mlir::detail::verifyRegionBranchWeights($_op);
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
            ::mlir::RegionSuccessor successor) {
            return ::mlir::MutableOperandRange(*this);
          }
        }]
    >
]>;

#endif // MLIR_INTERFACES_CONTROLFLOWINTERFACES
