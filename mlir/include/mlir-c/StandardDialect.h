//===-- mlir-c/StandardDialect.h - C API for Standard dialect -----*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface for registering and accessing the
// Standard dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_STANDARDDIALECT_H
#define MLIR_C_STANDARDDIALECT_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Registers the Standard dialect with the given context. This allows the
 * dialect to be loaded dynamically if needed when parsing. */
MLIR_CAPI_EXPORTED void mlirContextRegisterStandardDialect(MlirContext context);

/** Loads the Standard dialect into the given context. The dialect does _not_
 * have to be registered in advance. */
MLIR_CAPI_EXPORTED MlirDialect
mlirContextLoadStandardDialect(MlirContext context);

/// Returns the namespace of the Standard dialect, suitable for loading it.
MLIR_CAPI_EXPORTED MlirStringRef mlirStandardDialectGetNamespace();

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_STANDARDDIALECT_H
