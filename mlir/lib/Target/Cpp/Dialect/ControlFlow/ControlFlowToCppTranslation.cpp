//===--- ControlFlowToCppTranslation.cpp - Translate CF dialect to Cpp ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements translation between the ControlFlow dialect and Cpp.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/Cpp/Dialect/ControlFlow/ControlFlowToCppTranslation.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Target/Cpp/CppTranslationInterface.h"
#include "mlir/Target/Cpp/CppTranslationUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

static LogicalResult printOperation(CppEmitter &emitter,
                                    cf::BranchOp branchOp) {
  raw_ostream &os = emitter.ostream();
  Block &successor = *branchOp.getSuccessor();

  for (auto pair :
       llvm::zip(branchOp.getOperands(), successor.getArguments())) {
    Value &operand = std::get<0>(pair);
    BlockArgument &argument = std::get<1>(pair);
    os << emitter.getOrCreateName(argument) << " = "
       << emitter.getOrCreateName(operand) << ";\n";
  }

  os << "goto ";
  if (!(emitter.hasBlockLabel(successor)))
    return branchOp.emitOpError("unable to find label for successor block");
  os << emitter.getOrCreateName(successor);
  return success();
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    cf::CondBranchOp condBranchOp) {
  raw_indented_ostream &os = emitter.ostream();
  Block &trueSuccessor = *condBranchOp.getTrueDest();
  Block &falseSuccessor = *condBranchOp.getFalseDest();

  os << "if (" << emitter.getOrCreateName(condBranchOp.getCondition())
     << ") {\n";

  os.indent();

  // If condition is true.
  for (auto pair : llvm::zip(condBranchOp.getTrueOperands(),
                             trueSuccessor.getArguments())) {
    Value &operand = std::get<0>(pair);
    BlockArgument &argument = std::get<1>(pair);
    os << emitter.getOrCreateName(argument) << " = "
       << emitter.getOrCreateName(operand) << ";\n";
  }

  os << "goto ";
  if (!(emitter.hasBlockLabel(trueSuccessor))) {
    return condBranchOp.emitOpError("unable to find label for successor block");
  }
  os << emitter.getOrCreateName(trueSuccessor) << ";\n";
  os.unindent() << "} else {\n";
  os.indent();
  // If condition is false.
  for (auto pair : llvm::zip(condBranchOp.getFalseOperands(),
                             falseSuccessor.getArguments())) {
    Value &operand = std::get<0>(pair);
    BlockArgument &argument = std::get<1>(pair);
    os << emitter.getOrCreateName(argument) << " = "
       << emitter.getOrCreateName(operand) << ";\n";
  }

  os << "goto ";
  if (!(emitter.hasBlockLabel(falseSuccessor))) {
    return condBranchOp.emitOpError()
           << "unable to find label for successor block";
  }
  os << emitter.getOrCreateName(falseSuccessor) << ";\n";
  os.unindent() << "}";
  return success();
}

namespace {
/// Implementation of the dialect interface that converts ControlFlow op to Cpp.
class ControlFlowDialectCppTranslationInterface
    : public CppTranslationDialectInterface {
public:
  using CppTranslationDialectInterface::CppTranslationDialectInterface;

  LogicalResult emitOperation(Operation *op, CppEmitter &cppEmitter,
                              bool trailingSemicolon) const final {
    LogicalResult status =
        llvm::TypeSwitch<Operation *, LogicalResult>(op)
            // CF ops.
            .Case<cf::BranchOp, cf::CondBranchOp>(
                [&](auto op) { return printOperation(cppEmitter, op); })
            .Default([&](Operation *) {
              return op->emitOpError("unable to find printer for op")
                     << op->getName();
            });

    if (failed(status))
      return failure();

    cppEmitter.ostream() << (trailingSemicolon ? ";\n" : "\n");

    return success();
  }
};
} // namespace

void mlir::registerControlFlowDialectCppTranslation(DialectRegistry &registry) {
  registry.insert<cf::ControlFlowDialect>();
  registry.addExtension(+[](MLIRContext *ctx, cf::ControlFlowDialect *dialect) {
    dialect->addInterfaces<ControlFlowDialectCppTranslationInterface>();
  });
}

void mlir::registerControlFlowDialectCppTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerControlFlowDialectCppTranslation(registry);
  context.appendDialectRegistry(registry);
}
