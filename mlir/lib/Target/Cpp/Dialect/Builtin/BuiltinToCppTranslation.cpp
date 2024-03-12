//===-- BuiltinToCppTranslation.cpp - Translate Builtin dialect to Cpp ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR Builtin dialect and Cpp.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/Cpp/Dialect/Builtin/BuiltinToCppTranslation.h"

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/Cpp/CppTranslationInterface.h"
#include "mlir/Target/Cpp/CppTranslationUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

static LogicalResult printOperation(CppEmitter &emitter, ModuleOp moduleOp) {
  CppEmitter::Scope scope(emitter);

  for (Operation &op : moduleOp) {
    if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/false)))
      return failure();
  }
  return success();
}

namespace {
/// Implementation of the dialect interface that converts Builtin ops to Cpp.
class BuiltinDialectCppTranslationInterface
    : public CppTranslationDialectInterface {
public:
  using CppTranslationDialectInterface::CppTranslationDialectInterface;

  LogicalResult emitOperation(Operation *op, CppEmitter &cppEmitter,
                              bool trailingSemicolon) const final {
    LogicalResult status =
        llvm::TypeSwitch<Operation *, LogicalResult>(op)
            // Builtin ops.
            .Case<ModuleOp>(
                [&](auto op) { return printOperation(cppEmitter, op); })
            .Default([&](Operation *) {
              return op->emitOpError("unable to find printer for op: ")
                     << op->getName();
            });

    if (failed(status))
      return failure();

    cppEmitter.ostream() << (trailingSemicolon ? ";\n" : "\n");

    return success();
  }
};

} // namespace

void mlir::registerBuiltinDialectCppTranslation(DialectRegistry &registry) {
  registry.insert<BuiltinDialect>();
  registry.addExtension(+[](MLIRContext *ctx, BuiltinDialect *dialect) {
    dialect->addInterfaces<BuiltinDialectCppTranslationInterface>();
  });
}

void mlir::registerBuiltinDialectCppTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerBuiltinDialectCppTranslation(registry);
  context.appendDialectRegistry(registry);
}
