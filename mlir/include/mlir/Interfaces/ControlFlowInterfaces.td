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
  let methods = [
    InterfaceMethod<[{
        Returns a set of values that correspond to the arguments to the
        successor at the given index. Returns None if the operands to the
        successor are non-materialized values, i.e. they are internal to the
        operation.
      }],
      "Optional<OperandRange>", "getSuccessorOperands", (ins "unsigned":$index)
    >,
    InterfaceMethod<[{
        Return true if this operation can erase an operand to a successor block.
      }],
      "bool", "canEraseSuccessorOperand"
    >,
    InterfaceMethod<[{
        Erase the operand at `operandIndex` from the `index`-th successor. This
        should only be called if `canEraseSuccessorOperand` returns true.
      }],
      "void", "eraseSuccessorOperand",
      (ins "unsigned":$index, "unsigned":$operandIndex), [{}],
      /*defaultImplementation=*/[{
        ConcreteOp *op = static_cast<ConcreteOp *>(this);
        Optional<OperandRange> operands = op->getSuccessorOperands(index);
        assert(operands && "unable to query operands for successor");
        detail::eraseBranchSuccessorOperand(*operands, operandIndex, *op);
      }]
    >,
    InterfaceMethod<[{
        Returns the `BlockArgument` corresponding to operand `operandIndex` in
        some successor, or None if `operandIndex` isn't a successor operand
        index.
      }],
      "Optional<BlockArgument>", "getSuccessorBlockArgument",
      (ins "unsigned":$operandIndex), [{
        Operation *opaqueOp = op;
        for (unsigned i = 0, e = opaqueOp->getNumSuccessors(); i != e; ++i) {
          if (Optional<BlockArgument> arg = detail::getBranchSuccessorArgument(
                op.getSuccessorOperands(i), operandIndex,
                opaqueOp->getSuccessor(i)))
            return arg;
        }
        return llvm::None;
      }]
    >
  ];

  let verify = [{
    auto concreteOp = cast<ConcreteOpType>($_op);
    for (unsigned i = 0, e = $_op->getNumSuccessors(); i != e; ++i) {
      Optional<OperandRange> operands = concreteOp.getSuccessorOperands(i);
      if (failed(detail::verifyBranchSuccessorOperands($_op, i, operands)))
        return failure();
    }
    return success();
  }];
}

#endif // MLIR_INTERFACES_CONTROLFLOWINTERFACES
