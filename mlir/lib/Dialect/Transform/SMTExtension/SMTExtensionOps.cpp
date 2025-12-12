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
      dyn_cast<mlir::smt::YieldOp>(getRegion().front().back());
  if (!yieldTerminator)
    return emitOpError() << "expected '"
                         << mlir::smt::YieldOp::getOperationName()
                         << "' as terminator";

  auto checkTypes = [](size_t idx, Type smtType, StringRef smtDesc,
                       Type paramType, StringRef paramDesc,
                       auto *atOp) -> InFlightDiagnostic {
    if (!isa<mlir::smt::BoolType, mlir::smt::IntType, mlir::smt::BitVectorType>(
            smtType))
      return atOp->emitOpError() << "the type of " << smtDesc << " #" << idx
                                 << " is expected to be either a !smt.bool, a "
                                    "!smt.int, or a !smt.bv";

    assert(isa<TransformParamTypeInterface>(paramType) &&
           "ODS specifies params' type should implement param interface");
    if (isa<transform::AnyParamType>(paramType))
      return {}; // No further checks can be done.

    // NB: This cast must succeed as long as the only implementors of
    //     TransformParamTypeInterface are AnyParamType and ParamType.
    Type typeWrappedByParam = cast<ParamType>(paramType).getType();

    if (isa<mlir::smt::IntType>(smtType)) {
      if (!isa<IntegerType>(typeWrappedByParam))
        return atOp->emitOpError()
               << "the type of " << smtDesc << " #" << idx
               << " is !smt.int though the corresponding " << paramDesc
               << " type (" << paramType << ") is not wrapping an integer type";
    } else if (isa<mlir::smt::BoolType>(smtType)) {
      auto wrappedIntType = dyn_cast<IntegerType>(typeWrappedByParam);
      if (!wrappedIntType || wrappedIntType.getWidth() != 1)
        return atOp->emitOpError()
               << "the type of " << smtDesc << " #" << idx
               << " is !smt.bool though the corresponding " << paramDesc
               << " type (" << paramType << ") is not wrapping i1";
    } else if (auto bvSmtType = dyn_cast<mlir::smt::BitVectorType>(smtType)) {
      auto wrappedIntType = dyn_cast<IntegerType>(typeWrappedByParam);
      if (!wrappedIntType || wrappedIntType.getWidth() != bvSmtType.getWidth())
        return atOp->emitOpError()
               << "the type of " << smtDesc << " #" << idx << " is " << smtType
               << " though the corresponding " << paramDesc << " type ("
               << paramType
               << ") is not wrapping an integer type of the same bitwidth";
    }

    return {};
  };

  if (getOperands().size() != getBody().getNumArguments())
    return emitOpError(
        "must have the same number of block arguments as operands");

  for (auto [idx, operandType, blockArgType] :
       llvm::enumerate(getOperandTypes(), getBody().getArgumentTypes())) {
    InFlightDiagnostic typeCheckResult =
        checkTypes(idx, blockArgType, "block arg", operandType, "operand",
                   /*atOp=*/this);
    if (LogicalResult(typeCheckResult).failed())
      return typeCheckResult;
  }

  for (auto &op : getBody().getOps()) {
    if (!isa<mlir::smt::SMTDialect>(op.getDialect()))
      return emitOpError(
          "ops contained in region should belong to SMT-dialect");
  }

  if (yieldTerminator->getNumOperands() != getNumResults())
    return yieldTerminator.emitOpError()
           << "expected terminator to have as many operands as the parent op "
              "has results";

  for (auto [idx, termOperandType, resultType] : llvm::enumerate(
           yieldTerminator->getOperands().getType(), getResultTypes())) {
    InFlightDiagnostic typeCheckResult =
        checkTypes(idx, termOperandType, "terminator operand",
                   cast<transform::ParamType>(resultType), "result",
                   /*atOp=*/&yieldTerminator);
    if (LogicalResult(typeCheckResult).failed())
      return typeCheckResult;
  }

  return success();
}
