//===- SMTExtensionOps.cpp - SMT extension for the Transform dialect ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/SMTExtension/SMTExtensionOps.h"
#include "mlir/Dialect/SMT/IR/SMTDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/SMTExtension/SMTExtension.h"

using namespace mlir;

#define GET_OP_CLASSES
#include "mlir/Dialect/Transform/SMTExtension/SMTExtensionOps.cpp.inc"

//===----------------------------------------------------------------------===//
// ConstrainParamsOp
//===----------------------------------------------------------------------===//

void transform::smt::ConstrainParamsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getParamsMutable(), effects);
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
  return emitDefiniteFailure() << "op does not have interpreted semantics yet";
}

LogicalResult transform::smt::ConstrainParamsOp::verify() {
  if (getOperands().size() != getBody().getNumArguments())
    return emitOpError(
        "must have the same number of block arguments as operands");

  for (auto &op : getBody().getOps()) {
    if (!isa<mlir::smt::SMTDialect>(op.getDialect()))
      return emitOpError(
          "ops contained in region should belong to SMT-dialect");
  }

  return success();
}
