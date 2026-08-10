//===- BuiltinToLLVMIRTranslation.cpp - Translate builtin to LLVM IR ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR builtin dialect and LLVM
// IR.
//
//===----------------------------------------------------------------------===//
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"

using namespace mlir;

namespace {

class BuiltinDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const override {
    return success(isa<ModuleOp>(op));
  }
};

} // namespace

void mlir::registerBuiltinDialectTranslation(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, BuiltinDialect *dialect) {
    dialect->addInterfaces<BuiltinDialectLLVMIRTranslationInterface>();
  });
}

void mlir::registerBuiltinDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerBuiltinDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
