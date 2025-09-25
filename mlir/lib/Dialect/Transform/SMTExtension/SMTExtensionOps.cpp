//===- SMTExtensionOps.cpp - SMT extension for the Transform dialect ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/SMTExtension/SMTExtensionOps.h"
#include "mlir/Dialect/SMT/IR/SMTDialect.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"

using namespace mlir;

#define GET_OP_CLASSES
#include "mlir/Dialect/Transform/SMTExtension/SMTExtensionOps.cpp.inc"

//===----------------------------------------------------------------------===//
// ConstrainParamsOp
//===----------------------------------------------------------------------===//

void transform::smt::ConstrainParamsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getParamsMutable(), effects);
  producesHandle(getResults(), effects);
}

DiagnosedSilenceableFailure
transform::smt::ConstrainParamsOp::apply(transform::TransformRewriter &rewriter,
                                         transform::TransformResults &results,
                                         transform::TransformState &state) {
  // TODO: Proper operational semantics are to check the SMT problem in the body
  //       with a SMT solver with the arguments of the body constrained to the
  //       values passed into the op. Success or failure is then determined by
  //       the solver's result.
  //       One way to support this is to just promise the TransformOpInterface
  //       and allow for users to attach their own implementation, which would,
  //       e.g., translate the ops to SMTLIB and hand that over to the user's
  //       favourite solver. This requires changes to the dialect's verifier.
  return emitSilenceableFailure(getLoc())
         << "op does not have interpreted semantics yet";
}

LogicalResult transform::smt::ConstrainParamsOp::verify() {
  auto yieldTerminator =
      llvm::dyn_cast_if_present<mlir::smt::YieldOp>(getRegion().front().back());
  if (!yieldTerminator)
    return emitOpError() << "expected '"
                         << mlir::smt::YieldOp::getOperationName()
                         << "' as terminator";

  if (getOperands().size() != getBody().getNumArguments())
    return emitOpError(
        "must have the same number of block arguments as operands");

  for (auto [i, operandType, blockArgType] :
       llvm::zip_equal(llvm::seq<unsigned>(0, getBody().getNumArguments()),
                       getOperandTypes(), getBody().getArgumentTypes())) {
    if (isa<transform::AnyParamType>(operandType))
      continue; // No type checking as operand is of !transform.any_param type.
    auto paramOperandType = dyn_cast<transform::ParamType>(operandType);
    if (!paramOperandType)
      return emitOpError() << "operand type #" << i
                           << " is not a !transform.param";
    Type wrappedOperandType = paramOperandType.getType();

    if (isa<mlir::smt::IntType>(blockArgType)) {
      if (!isa<IntegerType>(paramOperandType.getType()))
        return emitOpError()
               << "the type of block arg #" << i
               << " is !smt.int though the corresponding operand type ("
               << operandType << ") is not wrapping an integer type";
    } else if (isa<mlir::smt::BoolType>(blockArgType)) {
      auto intOperandType = dyn_cast<IntegerType>(wrappedOperandType);
      if (!intOperandType || intOperandType.getWidth() != 1)
        return emitOpError()
               << "the type of block arg #" << i
               << " is !smt.bool though the corresponding operand type ("
               << operandType << ") is not wrapping i1 (i.e. bool)";
    } else if (auto bvBlockArgType =
                   dyn_cast<mlir::smt::BitVectorType>(blockArgType)) {
      auto intOperandType = dyn_cast<IntegerType>(wrappedOperandType);
      if (!intOperandType ||
          intOperandType.getWidth() != bvBlockArgType.getWidth())
        return emitOpError()
               << "the type of block arg #" << i << " is " << blockArgType
               << " though the corresponding operand type (" << operandType
               << ") is not wrapping an integer type of the same bitwidth";
    }
  }

  for (auto &op : getBody().getOps()) {
    if (!isa<mlir::smt::SMTDialect>(op.getDialect()))
      return emitOpError(
          "ops contained in region should belong to SMT-dialect");
  }

  if (getOperands().size() != getBody().getNumArguments())
    return emitOpError(
        "must have the same number of block arguments as operands");

  if (yieldTerminator->getNumOperands() != getNumResults())
    return yieldTerminator.emitOpError()
           << "expected terminator to have as many operands as the parent op "
              "has results";

  for (auto [i, termOperandType, resultType] : llvm::zip_equal(
           llvm::seq<unsigned>(0, yieldTerminator->getNumOperands()),
           yieldTerminator->getOperands().getType(), getResultTypes())) {
    if (isa<transform::AnyParamType>(resultType))
      continue; // No type checking as result is of !transform.any_param type.
    auto paramResultType = dyn_cast<transform::ParamType>(resultType);
    if (!paramResultType)
      return emitOpError() << "result type #" << i
                           << " is not a !transform.param";
    Type wrappedResultType = paramResultType.getType();

    if (isa<mlir::smt::IntType>(termOperandType)) {
      if (!isa<IntegerType>(wrappedResultType))
        return yieldTerminator.emitOpError()
               << "the type of terminator operand #" << i
               << " is !smt.int though the corresponding result type ("
               << resultType
               << ") of the parent op is not wrapping an integer type";
    } else if (isa<mlir::smt::BoolType>(termOperandType)) {
      auto intResultType = dyn_cast<IntegerType>(wrappedResultType);
      if (!intResultType || intResultType.getWidth() != 1)
        return yieldTerminator.emitOpError()
               << "the type of terminator operand #" << i
               << " is !smt.bool though the corresponding result type ("
               << resultType
               << ") of the parent op is not wrapping i1 (i.e. bool)";
    } else if (auto bvOperandType =
                   dyn_cast<mlir::smt::BitVectorType>(termOperandType)) {
      auto intResultType = dyn_cast<IntegerType>(wrappedResultType);
      if (!intResultType ||
          intResultType.getWidth() != bvOperandType.getWidth())
        return yieldTerminator.emitOpError()
               << "the type of terminator operand #" << i << " is "
               << termOperandType << " though the corresponding result type ("
               << resultType
               << ") is not wrapping an integer type of the same bitwidth";
    }
  }

  return success();
}
