//===-- mlir-c/ExtensibleDialect.h - Extensible dialect APIs -----*- C -*-====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header provides APIs for extensible dialects.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_EXTENSIBLEDIALECT_H
#define MLIR_C_EXTENSIBLEDIALECT_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
/// Opaque type declarations (see mlir-c/IR.h for more details).
//===----------------------------------------------------------------------===//

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(MlirDynamicOpTrait, void);

#undef DEFINE_C_API_STRUCT

/// Attach a dynamic op trait to the given operation name.
/// Note that the operation name must be modeled by dynamic dialect and must be
/// registered.
/// The ownership of the trait will be transferred to the operation name
/// after this call.
MLIR_CAPI_EXPORTED bool
mlirDynamicOpTraitAttach(MlirDynamicOpTrait dynamicOpTrait,
                         MlirStringRef opName, MlirContext context);

/// Get the dynamic op trait that indicates the operation is a terminator.
MLIR_CAPI_EXPORTED MlirDynamicOpTrait
mlirDynamicOpTraitIsTerminatorCreate(void);

/// Get the dynamic op trait that indicates regions have no terminator.
MLIR_CAPI_EXPORTED MlirDynamicOpTrait
mlirDynamicOpTraitNoTerminatorCreate(void);

/// Destroy the dynamic op trait.
MLIR_CAPI_EXPORTED void
mlirDynamicOpTraitDestroy(MlirDynamicOpTrait dynamicOpTrait);

typedef struct {
  /// Optional constructor for the user data.
  /// Set to nullptr to disable it.
  void (*construct)(void *userData);
  /// Optional destructor for the user data.
  /// Set to nullptr to disable it.
  void (*destruct)(void *userData);
  /// The callback function to verify the operation.
  MlirLogicalResult (*verifyTrait)(MlirOperation op, void *userData);
  /// The callback function to verify the operation with access to regions.
  MlirLogicalResult (*verifyRegionTrait)(MlirOperation op, void *userData);
} MlirDynamicOpTraitCallbacks;

/// Create a custom dynamic op trait with the given type ID and callbacks.
MLIR_CAPI_EXPORTED MlirDynamicOpTrait mlirDynamicOpTraitCreate(
    MlirTypeID typeID, MlirDynamicOpTraitCallbacks callbacks, void *userData);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_EXTENSIBLEDIALECT_H
