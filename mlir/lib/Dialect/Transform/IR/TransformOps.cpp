//===- TransformDialect.cpp - Transform dialect operations ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/IR/Builders.h"

#include "mlir/IR/OpImplementation.h"

using namespace mlir;

#define GET_OP_CLASSES
#include "mlir/Dialect/Transform/IR/TransformOps.cpp.inc"

LogicalResult transform::SequenceOp::apply(transform::TransformResults &results,
                                           transform::TransformState &state) {
  SmallVector<Operation *> targets;
  if (getRoot())
    llvm::append_range(targets, state.getPayloadOps(getRoot()));
  else
    targets.push_back(state.getTopLevel());

  // Map the entry block argument to the list of operations.
  auto scope = state.make_region_scope(*getBodyBlock()->getParent());
  if (failed(state.mapBlockArguments(getBodyBlock()->getArgument(0), targets)))
    return failure();

  // Apply the sequenced ops one by one.
  for (Operation &transform : getBodyBlock()->without_terminator())
    if (failed(state.applyTransform(cast<TransformOpInterface>(transform))))
      return failure();

  // Forward the operation mapping for values yielded from the sequence to the
  // values produced by the sequence op.
  for (const auto &pair :
       llvm::zip(getBodyBlock()->getTerminator()->getOperands(),
                 getOperation()->getOpResults())) {
    Value terminatorOperand = std::get<0>(pair);
    OpResult result = std::get<1>(pair);
    results.set(result, state.getPayloadOps(terminatorOperand));
  }

  return success();
}

LogicalResult transform::SequenceOp::verify() {
  if (getBodyBlock()->getNumArguments() != 1 ||
      !getBodyBlock()->getArgumentTypes()[0].isa<pdl::OperationType>()) {
    return emitOpError()
           << "expected the entry block to have one argument of type "
           << pdl::OperationType::get(getContext());
  }

  if (auto parent = getOperation()->getParentOfType<transform::SequenceOp>()) {
    if (!getRoot()) {
      InFlightDiagnostic diag =
          emitOpError()
          << "expected the root operation to be provided for a nested sequence";
      diag.attachNote(parent.getLoc()) << "nested in another sequence";
      return diag;
    }
  }

  for (Operation &child : *getBodyBlock()) {
    if (!isa<TransformOpInterface>(child) &&
        &child != &getBodyBlock()->back()) {
      InFlightDiagnostic diag =
          emitOpError()
          << "expected children ops to implement TransformOpInterface";
      diag.attachNote(child.getLoc()) << "op without interface";
      return diag;
    }

    for (OpResult result : child.getResults()) {
      if (llvm::hasNItemsOrLess(result.getUses(), 1))
        continue;
      InFlightDiagnostic diag = child.emitError()
                                << "result #" << result.getResultNumber()
                                << " has more than one use";
      for (OpOperand &use : result.getUses()) {
        diag.attachNote(use.getOwner()->getLoc())
            << "used here as operand #" << use.getOperandNumber();
      }
      return diag;
    }
  }

  if (getBodyBlock()->getTerminator()->getOperandTypes() !=
      getOperation()->getResultTypes()) {
    InFlightDiagnostic diag = emitOpError()
                              << "expects the types of the terminator operands "
                                 "to match the types of the result";
    diag.attachNote(getBodyBlock()->getTerminator()->getLoc()) << "terminator";
    return diag;
  }
  return success();
}
