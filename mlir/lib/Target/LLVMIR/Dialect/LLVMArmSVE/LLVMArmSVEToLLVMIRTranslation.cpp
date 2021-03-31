//===- LLVMArmSVEToLLVMIRTranslation.cpp - Translate LLVMArmSVE to LLVM IR-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR LLVMArmSVE dialect and
// LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/LLVMArmSVE/LLVMArmSVEToLLVMIRTranslation.h"
#include "mlir/Dialect/LLVMIR/LLVMArmSVEDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAArch64.h"

using namespace mlir;
using namespace mlir::LLVM;

namespace {
/// Implementation of the dialect interface that converts operations belonging
/// to the LLVMArmSVE dialect to LLVM IR.
class LLVMArmSVEDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    Operation &opInst = *op;
#include "mlir/Dialect/LLVMIR/LLVMArmSVEConversions.inc"

    return failure();
  }
};
} // end namespace

void mlir::registerLLVMArmSVEDialectTranslation(DialectRegistry &registry) {
  registry.insert<LLVM::LLVMArmSVEDialect>();
  registry.addDialectInterface<LLVM::LLVMArmSVEDialect,
                               LLVMArmSVEDialectLLVMIRTranslationInterface>();
}

void mlir::registerLLVMArmSVEDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerLLVMArmSVEDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
