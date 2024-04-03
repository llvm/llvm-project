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
#include "mlir/Target/LLVMIR/Dialect/ArmSME/ArmSMEToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ArmSVE/ArmSVEToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMIRToLLVMTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/LLVMIRToNVVMTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenACC/OpenACCToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Ptr/LLVMIRToPtrTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Ptr/PtrToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/SPIRV/SPIRVToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/VCIX/VCIXToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/X86Vector/X86VectorToLLVMIRTranslation.h"

namespace mlir {
class DialectRegistry;

/// Registers all dialects that can be translated to LLVM IR and the
/// corresponding translation interfaces.
static inline void registerAllToLLVMIRTranslations(DialectRegistry &registry) {
  registerArmNeonDialectTranslation(registry);
  registerAMXDialectTranslation(registry);
  registerArmSMEDialectTranslation(registry);
  registerArmSVEDialectTranslation(registry);
  registerBuiltinDialectTranslation(registry);
  registerGPUDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  registerNVVMDialectTranslation(registry);
  registerOpenACCDialectTranslation(registry);
  registerOpenMPDialectTranslation(registry);
  registerPtrDialectTranslation(registry);
  registerROCDLDialectTranslation(registry);
  registerSPIRVDialectTranslation(registry);
  registerVCIXDialectTranslation(registry);
  registerX86VectorDialectTranslation(registry);

  // Extension required for translating GPU offloading Ops.
  gpu::registerOffloadingLLVMTranslationInterfaceExternalModels(registry);
}

/// Registers all the translations to LLVM IR required by GPU passes.
/// TODO: Remove this function when a safe dialect interface registration
/// mechanism is implemented, see D157703.
static inline void
registerAllGPUToLLVMIRTranslations(DialectRegistry &registry) {
  registerBuiltinDialectTranslation(registry);
  registerGPUDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  registerNVVMDialectTranslation(registry);
  registerPtrDialectTranslation(registry);
  registerROCDLDialectTranslation(registry);
  registerSPIRVDialectTranslation(registry);

  // Extension required for translating GPU offloading Ops.
  gpu::registerOffloadingLLVMTranslationInterfaceExternalModels(registry);
}

/// Registers all dialects that can be translated from LLVM IR and the
/// corresponding translation interfaces.
static inline void
registerAllFromLLVMIRTranslations(DialectRegistry &registry) {
  registerLLVMDialectImport(registry);
  registerPtrDialectImport(registry);
  registerNVVMDialectImport(registry);
}
} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_DIALECT_ALL_H
