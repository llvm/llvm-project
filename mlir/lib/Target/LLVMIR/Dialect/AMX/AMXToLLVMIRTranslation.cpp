//===- AMXToLLVMIRTranslation.cpp - Translate AMX to LLVM IR --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the AMX dialect and LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/AMX/AMXToLLVMIRTranslation.h"
#include "mlir/Dialect/AMX/AMXDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsX86.h"

using namespace mlir;
using namespace mlir::LLVM;

namespace {
/// Implementation of the dialect interface that converts operations belonging
/// to the AMX dialect to LLVM IR.
class AMXDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    Operation &opInst = *op;
#include "mlir/Dialect/AMX/AMXConversions.inc"

    return failure();
  }
};
} // namespace

void mlir::registerAMXDialectTranslation(DialectRegistry &registry) {
  registry.insert<amx::AMXDialect>();
  registry.addDialectInterface<amx::AMXDialect,
                               AMXDialectLLVMIRTranslationInterface>();
}

void mlir::registerAMXDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerAMXDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
