//===-- mlir-c/ExtensibleDialect.h - Extensible dialect management ---*- C
//-*-====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header provides basic access to the MLIR JIT. This is minimalist and
// experimental at the moment.
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

/// Attach a dynamic op trait to the given operation name.
/// Note that the operation name must be modeled by dynamic dialect and must be
/// registered.
MLIR_CAPI_EXPORTED bool
mlirDynamicOpTraitAttach(MlirDynamicOpTrait dynamicOpTrait,
                         MlirStringRef opName, MlirContext context);

/// Get the dynamic op trait that indicates the operation is a terminator.
MLIR_CAPI_EXPORTED MlirDynamicOpTrait mlirDynamicOpTraitGetIsTerminator(void);

/// Get the dynamic op trait that indicates regions have no terminator.
MLIR_CAPI_EXPORTED MlirDynamicOpTrait mlirDynamicOpTraitGetNoTerminator(void);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_EXTENSIBLEDIALECT_H
