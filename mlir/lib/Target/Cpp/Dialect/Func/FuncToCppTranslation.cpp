//===------ FuncToCppTranslation.cpp - Translate Func dialect to Cpp ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements translation between the Func dialect and Cpp.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/Cpp/Dialect/Func/FuncToCppTranslation.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Target/Cpp/CppTranslationInterface.h"
#include "mlir/Target/Cpp/CppTranslationUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

static LogicalResult printOperation(CppEmitter &emitter, func::CallOp callOp) {
  Operation *operation = callOp.getOperation();
  StringRef callee = callOp.getCallee();

  return printCallOperation(emitter, operation, callee);
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    func::ReturnOp returnOp) {
  raw_ostream &os = emitter.ostream();
  os << "return";
  switch (returnOp.getNumOperands()) {
  case 0:
    return success();
  case 1:
    os << " ";
    if (failed(emitter.emitOperand(returnOp.getOperand(0))))
      return failure();
    return success();
  default:
    os << " std::make_tuple(";
    if (failed(emitter.emitOperandsAndAttributes(*returnOp.getOperation())))
      return failure();
    os << ")";
    return success();
  }
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    func::FuncOp functionOp) {
  // We need to declare variables at top if the function has multiple blocks.
  if (!emitter.shouldDeclareVariablesAtTop() &&
      functionOp.getBlocks().size() > 1) {
    return functionOp.emitOpError(
        "with multiple blocks needs variables declared at top");
  }

  if (llvm::any_of(functionOp.getResultTypes(),
                   [](Type type) { return isa<emitc::ArrayType>(type); })) {
    return functionOp.emitOpError() << "cannot emit array type as result type";
  }

  CppEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();
  if (failed(emitter.emitTypes(functionOp.getLoc(),
                               functionOp.getFunctionType().getResults())))
    return failure();
  os << " " << functionOp.getName();

  os << "(";
  Operation *operation = functionOp.getOperation();
  if (failed(printFunctionArgs(emitter, operation, functionOp.getArguments())))
    return failure();
  os << ") {\n";
  if (failed(printFunctionBody(emitter, operation, functionOp.getBlocks())))
    return failure();
  os << "}\n";

  return success();
}

namespace {
/// Implementation of the dialect interface that converts Func ops to Cpp.
class FuncDialectCppTranslationInterface
    : public CppTranslationDialectInterface {
public:
  using CppTranslationDialectInterface::CppTranslationDialectInterface;

  LogicalResult emitOperation(Operation *op, CppEmitter &cppEmitter,
                              bool trailingSemicolon) const final {
    LogicalResult status =
        llvm::TypeSwitch<Operation *, LogicalResult>(op)
            // Func ops.
            .Case<func::CallOp, func::FuncOp, func::ReturnOp>(
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

void mlir::registerFuncDialectCppTranslation(DialectRegistry &registry) {
  registry.insert<func::FuncDialect>();
  registry.addExtension(+[](MLIRContext *ctx, func::FuncDialect *dialect) {
    dialect->addInterfaces<FuncDialectCppTranslationInterface>();
  });
}

void mlir::registerFuncDialectCppTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerFuncDialectCppTranslation(registry);
  context.appendDialectRegistry(registry);
}
