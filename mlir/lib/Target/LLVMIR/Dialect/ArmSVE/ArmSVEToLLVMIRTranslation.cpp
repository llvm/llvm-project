//======- ArmSVEToLLVMIRTranslation.cpp - Translate ArmSVE to LLVM IR -=======//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the ArmSVE dialect and LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/ArmSVE/ArmSVEToLLVMIRTranslation.h"
#include "mlir/Dialect/ArmSVE/ArmSVEDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAArch64.h"

using namespace mlir;
using namespace mlir::LLVM;

namespace {
/// Implementation of the dialect interface that converts operations belonging
/// to the ArmSVE dialect to LLVM IR.
class ArmSVEDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    Operation &opInst = *op;
#include "mlir/Dialect/ArmSVE/ArmSVEConversions.inc"

    return failure();
  }
};
} // namespace

void mlir::registerArmSVEDialectTranslation(DialectRegistry &registry) {
  registry.insert<arm_sve::ArmSVEDialect>();
  registry.addExtension(+[](MLIRContext *ctx, arm_sve::ArmSVEDialect *dialect) {
    dialect->addInterfaces<ArmSVEDialectLLVMIRTranslationInterface>();
  });
}

void mlir::registerArmSVEDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerArmSVEDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
