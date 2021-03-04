//===- LLVMArmNeonToLLVMIRTranslation.cpp - LLVMArmNeon to LLVM IR --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR LLVMArmNeon dialect and
// LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/LLVMArmNeon/LLVMArmNeonToLLVMIRTranslation.h"
#include "mlir/Dialect/LLVMIR/LLVMArmNeonDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAArch64.h"

using namespace mlir;
using namespace mlir::LLVM;

namespace {
/// Implementation of the dialect interface that converts operations belonging
/// to the LLVMArmNeon dialect to LLVM IR.
class LLVMArmNeonDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    Operation &opInst = *op;
#include "mlir/Dialect/LLVMIR/LLVMArmNeonConversions.inc"

    return failure();
  }
};
} // end namespace

void mlir::registerLLVMArmNeonDialectTranslation(DialectRegistry &registry) {
  registry.insert<LLVM::LLVMArmNeonDialect>();
  registry.addDialectInterface<LLVM::LLVMArmNeonDialect,
                               LLVMArmNeonDialectLLVMIRTranslationInterface>();
}

void mlir::registerLLVMArmNeonDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerLLVMArmNeonDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
