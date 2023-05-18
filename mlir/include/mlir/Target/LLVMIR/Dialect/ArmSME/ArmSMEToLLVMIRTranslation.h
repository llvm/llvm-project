//=======- ArmSMEToLLVMIRTranslation.h - ArmSME to LLVM IR --*- C++ -*-=======//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for ArmSME dialect to LLVM IR translation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVMIR_DIALECT_ARMSME_ARMSMETOLLVMIRTRANSLATION_H
#define MLIR_TARGET_LLVMIR_DIALECT_ARMSME_ARMSMETOLLVMIRTRANSLATION_H

namespace mlir {

class DialectRegistry;
class MLIRContext;

/// Register the ArmSME dialect and the translation from it to the LLVM IR in
/// the given registry;
void registerArmSMEDialectTranslation(DialectRegistry &registry);

/// Register the ArmSME dialect and the translation from it in the registry
/// associated with the given context.
void registerArmSMEDialectTranslation(MLIRContext &context);

} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_DIALECT_ARMSME_ARMSMETOLLVMIRTRANSLATION_H
