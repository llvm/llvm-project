//======- ArmSMEToLLVMIRTranslation.cpp - Translate ArmSME to LLVM IR -=======//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the ArmSME dialect and LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/ArmSME/ArmSMEToLLVMIRTranslation.h"
#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAArch64.h"

using namespace mlir;
using namespace mlir::LLVM;

namespace {
/// Implementation of the dialect interface that converts operations belonging
/// to the ArmSME dialect to LLVM IR.
class ArmSMEDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    Operation &opInst = *op;
#include "mlir/Dialect/ArmSME/IR/ArmSMEConversions.inc"

    return failure();
  }
};
} // namespace

void mlir::registerArmSMEDialectTranslation(DialectRegistry &registry) {
  registry.insert<arm_sme::ArmSMEDialect>();
  registry.addExtension(+[](MLIRContext *ctx, arm_sme::ArmSMEDialect *dialect) {
    dialect->addInterfaces<ArmSMEDialectLLVMIRTranslationInterface>();
  });
}

void mlir::registerArmSMEDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerArmSMEDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
