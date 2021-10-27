//===-- mlir-c/Interfaces.h - C API to Core MLIR IR interfaces ----*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface to MLIR interface classes. It is
// intended to contain interfaces defined in lib/Interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_DIALECT_H
#define MLIR_C_DIALECT_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Returns `true` if the given operation implements an interface identified by
/// its TypeID.
MLIR_CAPI_EXPORTED bool
mlirOperationImplementsInterface(MlirOperation operation,
                                 MlirTypeID interfaceTypeID);

/// Returns `true` if the operation identified by its canonical string name
/// implements the interface identified by its TypeID in the given context.
/// Note that interfaces may be attached to operations in some contexts and not
/// others.
MLIR_CAPI_EXPORTED bool
mlirOperationImplementsInterfaceStatic(MlirStringRef operationName,
                                       MlirContext context,
                                       MlirTypeID interfaceTypeID);

//===----------------------------------------------------------------------===//
// InferTypeOpInterface.
//===----------------------------------------------------------------------===//

/// Returns the interface TypeID of the InferTypeOpInterface.
MLIR_CAPI_EXPORTED MlirTypeID mlirInferTypeOpInterfaceTypeID();

/// These callbacks are used to return multiple types from functions while
/// transferring ownerhsip to the caller. The first argument is the number of
/// consecutive elements pointed to by the second argument. The third argument
/// is an opaque pointer forwarded to the callback by the caller.
typedef void (*MlirTypesCallback)(intptr_t, MlirType *, void *);

/// Infers the return types of the operation identified by its canonical given
/// the arguments that will be supplied to its generic builder. Calls `callback`
/// with the types of inferred arguments, potentially several times, on success.
/// Returns failure otherwise.
MLIR_CAPI_EXPORTED MlirLogicalResult mlirInferTypeOpInterfaceInferReturnTypes(
    MlirStringRef opName, MlirContext context, MlirLocation location,
    intptr_t nOperands, MlirValue *operands, MlirAttribute attributes,
    intptr_t nRegions, MlirRegion *regions, MlirTypesCallback callback,
    void *userData);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_H
