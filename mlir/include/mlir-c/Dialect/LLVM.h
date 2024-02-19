//===-- mlir-c/Dialect/LLVM.h - C API for LLVM --------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_DIALECT_LLVM_H
#define MLIR_C_DIALECT_LLVM_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(LLVM, llvm);

/// Creates an llvm.ptr type.
MLIR_CAPI_EXPORTED MlirType mlirLLVMPointerTypeGet(MlirContext ctx,
                                                   unsigned addressSpace);

/// Creates an llmv.void type.
MLIR_CAPI_EXPORTED MlirType mlirLLVMVoidTypeGet(MlirContext ctx);

/// Creates an llvm.array type.
MLIR_CAPI_EXPORTED MlirType mlirLLVMArrayTypeGet(MlirType elementType,
                                                 unsigned numElements);

/// Creates an llvm.func type.
MLIR_CAPI_EXPORTED MlirType
mlirLLVMFunctionTypeGet(MlirType resultType, intptr_t nArgumentTypes,
                        MlirType const *argumentTypes, bool isVarArg);

/// Returns `true` if the type is an LLVM dialect struct type.
MLIR_CAPI_EXPORTED bool mlirTypeIsALLVMStructType(MlirType type);

/// Returns `true` if the type is a literal (unnamed) LLVM struct type.
MLIR_CAPI_EXPORTED bool mlirLLVMStructTypeIsLiteral(MlirType type);

/// Returns the number of fields in the struct. Asserts if the struct is opaque
/// or not yet initialized.
MLIR_CAPI_EXPORTED intptr_t mlirLLVMStructTypeGetNumElementTypes(MlirType type);

/// Returns the `positions`-th field of the struct. Asserts if the struct is
/// opaque, not yet initialized or if the position is out of range.
MLIR_CAPI_EXPORTED MlirType mlirLLVMStructTypeGetElementType(MlirType type,
                                                             intptr_t position);

/// Returns `true` if the struct is packed.
MLIR_CAPI_EXPORTED bool mlirLLVMStructTypeIsPacked(MlirType type);

/// Returns the identifier of the identified struct. Asserts that the struct is
/// identified, i.e., not literal.
MLIR_CAPI_EXPORTED MlirStringRef mlirLLVMStructTypeGetIdentifier(MlirType type);

/// Returns `true` is the struct is explicitly opaque (will not have a body) or
/// uninitialized (will eventually have a body).
MLIR_CAPI_EXPORTED bool mlirLLVMStructTypeIsOpaque(MlirType type);

/// Creates an LLVM literal (unnamed) struct type. This may assert if the fields
/// have types not compatible with the LLVM dialect. For a graceful failure, use
/// the checked version.
MLIR_CAPI_EXPORTED MlirType
mlirLLVMStructTypeLiteralGet(MlirContext ctx, intptr_t nFieldTypes,
                             MlirType const *fieldTypes, bool isPacked);

/// Creates an LLVM literal (unnamed) struct type if possible. Emits a
/// diagnostic at the given location and returns null otherwise.
MLIR_CAPI_EXPORTED MlirType
mlirLLVMStructTypeLiteralGetChecked(MlirLocation loc, intptr_t nFieldTypes,
                                    MlirType const *fieldTypes, bool isPacked);

/// Creates an LLVM identified struct type with no body. If a struct type with
/// this name already exists in the context, returns that type. Use
/// mlirLLVMStructTypeIdentifiedNewGet to create a fresh struct type,
/// potentially renaming it. The body should be set separatelty by calling
/// mlirLLVMStructTypeSetBody, if it isn't set already.
MLIR_CAPI_EXPORTED MlirType mlirLLVMStructTypeIdentifiedGet(MlirContext ctx,
                                                            MlirStringRef name);

/// Creates an LLVM identified struct type with no body and a name starting with
/// the given prefix. If a struct with the exact name as the given prefix
/// already exists, appends an unspecified suffix to the name so that the name
/// is unique in context.
MLIR_CAPI_EXPORTED MlirType mlirLLVMStructTypeIdentifiedNewGet(
    MlirContext ctx, MlirStringRef name, intptr_t nFieldTypes,
    MlirType const *fieldTypes, bool isPacked);

MLIR_CAPI_EXPORTED MlirType mlirLLVMStructTypeOpaqueGet(MlirContext ctx,
                                                        MlirStringRef name);

/// Sets the body of the identified struct if it hasn't been set yet. Returns
/// whether the operation was successful.
MLIR_CAPI_EXPORTED MlirLogicalResult
mlirLLVMStructTypeSetBody(MlirType structType, intptr_t nFieldTypes,
                          MlirType const *fieldTypes, bool isPacked);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_LLVM_H
