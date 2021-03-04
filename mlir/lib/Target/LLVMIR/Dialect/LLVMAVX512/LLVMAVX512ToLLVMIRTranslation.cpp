//===- LLVMAVX512ToLLVMIRTranslation.cpp - Translate LLVMAVX512 to LLVM IR-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR LLVMAVX512 dialect and
// LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/LLVMAVX512/LLVMAVX512ToLLVMIRTranslation.h"
#include "mlir/Dialect/LLVMIR/LLVMAVX512Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsX86.h"

using namespace mlir;
using namespace mlir::LLVM;

namespace {
/// Implementation of the dialect interface that converts operations belonging
/// to the LLVMAVX512 dialect to LLVM IR.
class LLVMAVX512DialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    Operation &opInst = *op;
#include "mlir/Dialect/LLVMIR/LLVMAVX512Conversions.inc"

    return failure();
  }
};
} // end namespace

void mlir::registerLLVMAVX512DialectTranslation(DialectRegistry &registry) {
  registry.insert<LLVM::LLVMAVX512Dialect>();
  registry.addDialectInterface<LLVM::LLVMAVX512Dialect,
                               LLVMAVX512DialectLLVMIRTranslationInterface>();
}

void mlir::registerLLVMAVX512DialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerLLVMAVX512DialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
