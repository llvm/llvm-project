//===- X86VectorToLLVMIRTranslation.cpp - Translate X86Vector to LLVM IR---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR X86Vector dialect and
// LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/X86Vector/X86VectorToLLVMIRTranslation.h"
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsX86.h"

using namespace mlir;
using namespace mlir::LLVM;

namespace {
/// Implementation of the dialect interface that converts operations belonging
/// to the X86Vector dialect to LLVM IR.
class X86VectorDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    Operation &opInst = *op;
#include "mlir/Dialect/X86Vector/X86VectorConversions.inc"

    return failure();
  }
};
} // namespace

void mlir::registerX86VectorDialectTranslation(DialectRegistry &registry) {
  registry.insert<x86vector::X86VectorDialect>();
  registry.addDialectInterface<x86vector::X86VectorDialect,
                               X86VectorDialectLLVMIRTranslationInterface>();
}

void mlir::registerX86VectorDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerX86VectorDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
