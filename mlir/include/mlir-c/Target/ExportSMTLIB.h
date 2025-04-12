//===- MLIR-c/ExportSMTLIB.h - C API for emitting SMTLIB ---------*- C -*-===//
//
// This header declares the C interface for emitting SMTLIB from a MLIR MLIR
// module.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_EXPORTSMTLIB_H
#define MLIR_C_EXPORTSMTLIB_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Emits SMTLIB for the specified module using the provided callback and user
/// data
MLIR_CAPI_EXPORTED MlirLogicalResult mlirExportSMTLIB(MlirModule,
                                                      MlirStringCallback,
                                                      void *userData);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_EXPORTSMTLIB_H
