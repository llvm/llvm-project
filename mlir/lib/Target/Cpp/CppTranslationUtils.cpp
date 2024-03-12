//===- CppTranslationUtils.cpp - Helpers used in C++ emitter ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines common helper functions used across the Cpp translation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_CPP_CPPTRANSLATIONUTILS_CPP
#define MLIR_TARGET_CPP_CPPTRANSLATIONUTILS_CPP

#include "mlir/Target/Cpp/CppTranslationUtils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

using namespace mlir;

bool shouldBeInlined(emitc::ExpressionOp expressionOp) {
  // Do not inline if expression is marked as such.
  if (expressionOp.getDoNotInline())
    return false;

  // Do not inline expressions with side effects to prevent side-effect
  // reordering.
  if (expressionOp.hasSideEffects())
    return false;

  // Do not inline expressions with multiple uses.
  Value result = expressionOp.getResult();
  if (!result.hasOneUse())
    return false;

  // Do not inline expressions used by other expressions, as any desired
  // expression folding was taken care of by transformations.
  Operation *user = *result.getUsers().begin();
  return !user->getParentOfType<emitc::ExpressionOp>();
}

// ! constant, binary and unart op are currently only used by emitc translation

LogicalResult printConstantOp(CppEmitter &emitter, Operation *operation,
                              Attribute value) {
  OpResult result = operation->getResult(0);

  // Only emit an assignment as the variable was already declared when printing
  // the FuncOp.
  if (emitter.shouldDeclareVariablesAtTop()) {
    // Skip the assignment if the emitc.constant has no value.
    if (auto oAttr = dyn_cast<emitc::OpaqueAttr>(value)) {
      if (oAttr.getValue().empty())
        return success();
    }

    if (failed(emitter.emitVariableAssignment(result)))
      return failure();
    return emitter.emitAttribute(operation->getLoc(), value);
  }

  // Emit a variable declaration for an emitc.constant op without value.
  if (auto oAttr = dyn_cast<emitc::OpaqueAttr>(value)) {
    if (oAttr.getValue().empty())
      // The semicolon gets printed by the emitOperation function.
      return emitter.emitVariableDeclaration(result,
                                             /*trailingSemicolon=*/false);
  }

  // Emit a variable declaration.
  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  return emitter.emitAttribute(operation->getLoc(), value);
}

LogicalResult printBinaryOperation(CppEmitter &emitter, Operation *operation,
                                   StringRef binaryOperator) {
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();

  if (failed(emitter.emitOperand(operation->getOperand(0))))
    return failure();

  os << " " << binaryOperator << " ";

  if (failed(emitter.emitOperand(operation->getOperand(1))))
    return failure();

  return success();
}

LogicalResult printUnaryOperation(CppEmitter &emitter, Operation *operation,
                                  StringRef unaryOperator) {
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();

  os << unaryOperator;

  if (failed(emitter.emitOperand(operation->getOperand(0))))
    return failure();

  return success();
}

LogicalResult printCallOperation(CppEmitter &emitter, Operation *callOp,
                                 StringRef callee) {
  if (failed(emitter.emitAssignPrefix(*callOp)))
    return failure();

  raw_ostream &os = emitter.ostream();
  os << callee << "(";
  if (failed(emitter.emitOperands(*callOp)))
    return failure();
  os << ")";
  return success();
}

LogicalResult printFunctionArgs(CppEmitter &emitter, Operation *functionOp,
                                ArrayRef<Type> arguments) {
  raw_indented_ostream &os = emitter.ostream();

  return (
      interleaveCommaWithError(arguments, os, [&](Type arg) -> LogicalResult {
        return emitter.emitType(functionOp->getLoc(), arg);
      }));
}

LogicalResult printFunctionArgs(CppEmitter &emitter, Operation *functionOp,
                                Region::BlockArgListType arguments) {
  raw_indented_ostream &os = emitter.ostream();

  return (interleaveCommaWithError(
      arguments, os, [&](BlockArgument arg) -> LogicalResult {
        return emitter.emitVariableDeclaration(
            functionOp->getLoc(), arg.getType(), emitter.getOrCreateName(arg));
      }));
}

LogicalResult printFunctionBody(CppEmitter &emitter, Operation *functionOp,
                                Region::BlockListType &blocks) {
  raw_indented_ostream &os = emitter.ostream();
  os.indent();

  if (emitter.shouldDeclareVariablesAtTop()) {
    // Declare all variables that hold op results including those from nested
    // regions.
    WalkResult result =
        functionOp->walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
          if (isa<emitc::LiteralOp>(op) ||
              isa<emitc::ExpressionOp>(op->getParentOp()) ||
              (isa<emitc::ExpressionOp>(op) &&
               shouldBeInlined(cast<emitc::ExpressionOp>(op))))
            return WalkResult::skip();
          for (OpResult result : op->getResults()) {
            if (failed(emitter.emitVariableDeclaration(
                    result, /*trailingSemicolon=*/true))) {
              return WalkResult(
                  op->emitError("unable to declare result variable for op"));
            }
          }
          return WalkResult::advance();
        });
    if (result.wasInterrupted())
      return failure();
  }

  // Create label names for basic blocks.
  for (Block &block : blocks) {
    emitter.getOrCreateName(block);
  }

  // Declare variables for basic block arguments.
  for (Block &block : llvm::drop_begin(blocks)) {
    for (BlockArgument &arg : block.getArguments()) {
      if (emitter.hasValueInScope(arg))
        return functionOp->emitOpError(" block argument #")
               << arg.getArgNumber() << " is out of scope";
      if (isa<emitc::ArrayType>(arg.getType()))
        return functionOp->emitOpError("cannot emit block argument #")
               << arg.getArgNumber() << " with array type";
      if (failed(
              emitter.emitType(block.getParentOp()->getLoc(), arg.getType()))) {
        return failure();
      }
      os << " " << emitter.getOrCreateName(arg) << ";\n";
    }
  }

  for (Block &block : blocks) {
    // Only print a label if the block has predecessors.
    if (!block.hasNoPredecessors()) {
      if (failed(emitter.emitLabel(block)))
        return failure();
    }
    for (Operation &op : block.getOperations()) {
      // When generating code for an emitc.if or cf.cond_br op no semicolon
      // needs to be printed after the closing brace.
      // When generating code for an emitc.for and emitc.verbatim op, printing a
      // trailing semicolon is handled within the printOperation function.
      bool trailingSemicolon =
          !isa<cf::CondBranchOp, emitc::DeclareFuncOp, emitc::ForOp,
               emitc::IfOp, emitc::LiteralOp, emitc::VerbatimOp>(op);

      if (failed(emitter.emitOperation(
              op, /*trailingSemicolon=*/trailingSemicolon)))
        return failure();
    }
  }

  os.unindent();

  return success();
}

#endif // MLIR_TARGET_CPP_CPPTRANSLATIONUTILS_CPP
