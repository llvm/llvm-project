//===- All.h - MLIR To LLVM IR Translation Registration ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to register the translations of all suitable
// dialects to LLVM IR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVMIR_DIALECT_ALL_H
#define MLIR_TARGET_LLVMIR_DIALECT_ALL_H

#include "mlir/Target/LLVMIR/Dialect/AMX/AMXToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ArmNeon/ArmNeonToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ArmSVE/ArmSVEToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenACC/OpenACCToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/X86Vector/X86VectorToLLVMIRTranslation.h"

namespace mlir {
class DialectRegistry;

/// Registers all dialects that can be translated to LLVM IR and the
/// corresponding translation interfaces.
static inline void registerAllToLLVMIRTranslations(DialectRegistry &registry) {
  registerArmNeonDialectTranslation(registry);
  registerAMXDialectTranslation(registry);
  registerArmSVEDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  registerNVVMDialectTranslation(registry);
  registerOpenACCDialectTranslation(registry);
  registerOpenMPDialectTranslation(registry);
  registerROCDLDialectTranslation(registry);
  registerX86VectorDialectTranslation(registry);
}
} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_DIALECT_ALL_H
