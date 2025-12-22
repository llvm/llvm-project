//===-- mlir-c/Dialect/Complex.h - C API for Complex dialect ------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_DIALECT_COMPLEX_H
#define MLIR_C_DIALECT_COMPLEX_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Func, func);

/// Checks whether the given attribute is a complex attribute.
MLIR_CAPI_EXPORTED bool mlirAttributeIsAComplex(MlirAttribute attr);

/// Creates a complex attribute in the given context with the given
/// double real and imaginary values and double-precision FP semantics.
MLIR_CAPI_EXPORTED MlirAttribute mlirComplexAttrDoubleGet(MlirContext ctx,
                                                          MlirType type,
                                                          double real,
                                                          double imag);

/// Same as "mlirComplexAttrDoubleGet", but if the type is not valid for a
/// construction of a ComplexAttr, returns a null MlirAttribute.
MLIR_CAPI_EXPORTED MlirAttribute
mlirComplexAttrDoubleGetChecked(MlirLocation loc, MlirType type, double real,
                                double imag);

/// Returns the real value stored in the given complex attribute, interpreting
/// the value as double.
MLIR_CAPI_EXPORTED double mlirComplexAttrGetRealDouble(MlirAttribute attr);

/// Returns the imaginaryvalue stored in the given complex attribute,
/// interpreting the value as double.
MLIR_CAPI_EXPORTED double mlirComplexAttrGetImagDouble(MlirAttribute attr);

/// Returns the typeID of a Complex attribute.
MLIR_CAPI_EXPORTED MlirTypeID mlirComplexAttrGetTypeID(void);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_COMPLEX_H
