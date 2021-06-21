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
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Returns a mutable range of operands that correspond to the arguments of
        successor at the given index. Returns None if the operands to the
        successor are non-materialized values, i.e. they are internal to the
        operation.
      }],
      "Optional<MutableOperandRange>", "getMutableSuccessorOperands",
      (ins "unsigned":$index)
    >,
    InterfaceMethod<[{
        Returns a range of operands that correspond to the arguments of
        successor at the given index. Returns None if the operands to the
        successor are non-materialized values, i.e. they are internal to the
        operation.
      }],
      "Optional<OperandRange>", "getSuccessorOperands",
      (ins "unsigned":$index), [{}], [{
        ConcreteOp *op = static_cast<ConcreteOp *>(this);
        auto operands = op->getMutableSuccessorOperands(index);
        return operands ? Optional<OperandRange>(*operands) : llvm::None;
      }]
    >,
    InterfaceMethod<[{
        Returns the `BlockArgument` corresponding to operand `operandIndex` in
        some successor, or None if `operandIndex` isn't a successor operand
        index.
      }],
      "Optional<BlockArgument>", "getSuccessorBlockArgument",
      (ins "unsigned":$operandIndex), [{
        Operation *opaqueOp = $_op;
        for (unsigned i = 0, e = opaqueOp->getNumSuccessors(); i != e; ++i) {
          if (Optional<BlockArgument> arg = detail::getBranchSuccessorArgument(
                $_op.getSuccessorOperands(i), operandIndex,
                opaqueOp->getSuccessor(i)))
            return arg;
        }
        return llvm::None;
      }]
    >,
    InterfaceMethod<[{
        Returns the successor that would be chosen with the given constant
        operands. Returns nullptr if a single successor could not be chosen.
      }],
      "Block *", "getSuccessorForOperands",
      (ins "ArrayRef<Attribute>":$operands), [{}],
      /*defaultImplementation=*/[{ return nullptr; }]
    >
  ];

  let verify = [{
    auto concreteOp = cast<ConcreteOp>($_op);
    for (unsigned i = 0, e = $_op->getNumSuccessors(); i != e; ++i) {
      Optional<OperandRange> operands = concreteOp.getSuccessorOperands(i);
      if (failed(detail::verifyBranchSuccessorOperands($_op, i, operands)))
        return failure();
    }
    return success();
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
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Returns the operands of this operation used as the entry arguments when
        entering the region at `index`, which was specified as a successor of this
        operation by `getSuccessorRegions`. These operands should correspond 1-1
        with the successor inputs specified in `getSuccessorRegions`.
      }],
      "OperandRange", "getSuccessorEntryOperands",
      (ins "unsigned":$index), [{}], /*defaultImplementation=*/[{
        auto operandEnd = this->getOperation()->operand_end();
        return OperandRange(operandEnd, operandEnd);
      }]
    >,
    InterfaceMethod<[{
        Returns the viable successors of a region at `index`, or the possible
        successors when branching from the parent op if `index` is None. These
        are the regions that may be selected during the flow of control. If
        `index` is None, `operands` is a set of optional attributes that
        either correspond to a constant value for each operand of this
        operation, or null if that operand is not a constant. If `index` is
        valid, `operands` corresponds to the exit values of the region at
        `index`. Only a region, i.e. a valid `index`, may use the parent
        operation as a successor. This method allows for describing which
        regions may be executed when entering an operation, and which regions
        are executed after having executed another region of the parent op. The
        successor region must be non-empty.
      }],
      "void", "getSuccessorRegions",
      (ins "Optional<unsigned>":$index, "ArrayRef<Attribute>":$operands,
           "SmallVectorImpl<RegionSuccessor> &":$regions)
    >,
    InterfaceMethod<[{
        Populates countPerRegion with the number of times this operation will
        invoke the attached regions (assuming the regions yield normally, i.e.
        do not abort or invoke an infinite loop). If the number of region
        invocations is not known statically it will set the number of
        invocations to `kUnknownNumRegionInvocations`.

        `operands` is a set of optional attributes that either correspond to a
        constant values for each operand of this operation, or null if that
        operand is not a constant.
      }],
      "void", "getNumRegionInvocations",
      (ins "ArrayRef<Attribute>":$operands,
           "SmallVectorImpl<int64_t> &":$countPerRegion), [{}],
      /*defaultImplementation=*/[{
        unsigned numRegions = this->getOperation()->getNumRegions();
        assert(countPerRegion.empty());
        countPerRegion.resize(numRegions, kUnknownNumRegionInvocations);
      }]
    >
  ];

  let verify = [{
    static_assert(!ConcreteOp::template hasTrait<OpTrait::ZeroRegion>(),
                  "expected operation to have non-zero regions");
    return success();
  }];

  let extraClassDeclaration = [{
    /// Convenience helper in case none of the operands is known.
    void getSuccessorRegions(Optional<unsigned> index,
                             SmallVectorImpl<RegionSuccessor> &regions) {
       SmallVector<Attribute, 2> nullAttrs(getOperation()->getNumOperands());
       getSuccessorRegions(index, nullAttrs, regions);
    }

    /// Verify types along control flow edges described by this interface.
    static LogicalResult verifyTypes(Operation *op) {
      return detail::verifyTypesAlongControlFlowEdges(op);
    }
  }];
}

//===----------------------------------------------------------------------===//
// ControlFlow Traits
//===----------------------------------------------------------------------===//

// Op is "return-like".
def ReturnLike : NativeOpTrait<"ReturnLike">;

#endif // MLIR_INTERFACES_CONTROLFLOWINTERFACES
