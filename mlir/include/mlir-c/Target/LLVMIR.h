//===-- LLVMIR.h - C Interface for MLIR LLVMIR Target -------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface to target LLVMIR with MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_TARGET_LLVMIR_H
#define MLIR_C_TARGET_LLVMIR_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "llvm-c/Core.h"
#include "llvm-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Translate operation that satisfies LLVM dialect module requirements into an
/// LLVM IR module living in the given context. This translates operations from
/// any dilalect that has a registered implementation of
/// LLVMTranslationDialectInterface.
///
/// \returns the generated LLVM IR Module from the translated MLIR module, it is
/// owned by the caller.
MLIR_CAPI_EXPORTED LLVMModuleRef
mlirTranslateModuleToLLVMIR(MlirOperation module, LLVMContextRef context);

struct MlirTypeFromLLVMIRTranslator {
  void *ptr;
};

typedef struct MlirTypeFromLLVMIRTranslator MlirTypeFromLLVMIRTranslator;

/// Create an LLVM::TypeFromLLVMIRTranslator and transfer ownership to the
/// caller.
MLIR_CAPI_EXPORTED MlirTypeFromLLVMIRTranslator
mlirTypeFromLLVMIRTranslatorCreate(MlirContext ctx);

/// Takes an LLVM::TypeFromLLVMIRTranslator owned by the caller and destroys it.
/// It is the responsibility of the user to only pass an
/// LLVM::TypeFromLLVMIRTranslator class.
MLIR_CAPI_EXPORTED void
mlirTypeFromLLVMIRTranslatorDestroy(MlirTypeFromLLVMIRTranslator translator);

/// Translates the given LLVM IR type to the MLIR LLVM dialect.
MLIR_CAPI_EXPORTED MlirType mlirTypeFromLLVMIRTranslatorTranslateType(
    MlirTypeFromLLVMIRTranslator translator, LLVMTypeRef llvmType);

struct MlirTypeToLLVMIRTranslator {
  void *ptr;
};

typedef struct MlirTypeToLLVMIRTranslator MlirTypeToLLVMIRTranslator;

/// Create an LLVM::TypeToLLVMIRTranslator and transfer ownership to the
/// caller.
MLIR_CAPI_EXPORTED MlirTypeToLLVMIRTranslator
mlirTypeToLLVMIRTranslatorCreate(LLVMContextRef ctx);

/// Takes an LLVM::TypeToLLVMIRTranslator owned by the caller and destroys it.
/// It is the responsibility of the user to only pass an
/// LLVM::TypeToLLVMIRTranslator class.
MLIR_CAPI_EXPORTED void
mlirTypeToLLVMIRTranslatorDestroy(MlirTypeToLLVMIRTranslator translator);

/// Translates the given MLIR LLVM dialect to the LLVM IR type.
MLIR_CAPI_EXPORTED LLVMTypeRef mlirTypeToLLVMIRTranslatorTranslateType(
    MlirTypeToLLVMIRTranslator translator, MlirType mlirType);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_TARGET_LLVMIR_H
