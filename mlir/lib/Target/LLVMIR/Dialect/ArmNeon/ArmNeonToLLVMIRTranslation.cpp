//===- ArmNeonToLLVMIRTranslation.cpp - Translate ArmNeon to LLVM IR ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR ArmNeon dialect and
// LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/ArmNeon/ArmNeonToLLVMIRTranslation.h"
#include "mlir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAArch64.h"

using namespace mlir;
using namespace mlir::LLVM;

namespace {
/// Implementation of the dialect interface that converts operations belonging
/// to the ArmNeon dialect to LLVM IR.
class ArmNeonDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    Operation &opInst = *op;
#include "mlir/Dialect/ArmNeon/ArmNeonConversions.inc"

    return failure();
  }
};
} // namespace

void mlir::registerArmNeonDialectTranslation(DialectRegistry &registry) {
  registry.insert<arm_neon::ArmNeonDialect>();
  registry.addExtension(
      +[](MLIRContext *ctx, arm_neon::ArmNeonDialect *dialect) {
        dialect->addInterfaces<ArmNeonDialectLLVMIRTranslationInterface>();
      });
}

void mlir::registerArmNeonDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerArmNeonDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
