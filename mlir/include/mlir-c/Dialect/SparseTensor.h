//===-- mlir-c/Dialect/SparseTensor.h - C API for SparseTensor ----*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_DIALECT_SPARSETENSOR_H
#define MLIR_C_DIALECT_SPARSETENSOR_H

#include "mlir-c/AffineMap.h"
#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(SparseTensor, sparse_tensor);

/// Dimension level types (and properties) that define sparse tensors.
/// See the documentation in SparseTensorAttrDefs.td for their meaning.
///
/// These correspond to SparseTensorEncodingAttr::LevelType in the C++ API.
/// If updating, keep them in sync and update the static_assert in the impl
/// file.
typedef uint64_t MlirSparseTensorLevelType;

enum MlirSparseTensorLevelFormat {
  MLIR_SPARSE_TENSOR_LEVEL_DENSE = 0x000000010000,
  MLIR_SPARSE_TENSOR_LEVEL_BATCH = 0x000000020000,
  MLIR_SPARSE_TENSOR_LEVEL_COMPRESSED = 0x000000040000,
  MLIR_SPARSE_TENSOR_LEVEL_SINGLETON = 0x000000080000,
  MLIR_SPARSE_TENSOR_LEVEL_LOOSE_COMPRESSED = 0x000000100000,
  MLIR_SPARSE_TENSOR_LEVEL_N_OUT_OF_M = 0x000000200000,
};

enum MlirSparseTensorLevelPropertyNondefault {
  MLIR_SPARSE_PROPERTY_NON_UNIQUE = 0x0001,
  MLIR_SPARSE_PROPERTY_NON_ORDERED = 0x0002,
  MLIR_SPARSE_PROPERTY_SOA = 0x0004,
};

//===----------------------------------------------------------------------===//
// SparseTensorEncodingAttr
//===----------------------------------------------------------------------===//

/// Checks whether the given attribute is a `sparse_tensor.encoding` attribute.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsASparseTensorEncodingAttr(MlirAttribute attr);

/// Creates a `sparse_tensor.encoding` attribute with the given parameters.
MLIR_CAPI_EXPORTED MlirAttribute mlirSparseTensorEncodingAttrGet(
    MlirContext ctx, intptr_t lvlRank,
    MlirSparseTensorLevelType const *lvlTypes, MlirAffineMap dimToLvl,
    MlirAffineMap lvlTodim, int posWidth, int crdWidth,
    MlirAttribute explicitVal, MlirAttribute implicitVal);

/// Returns the level-rank of the `sparse_tensor.encoding` attribute.
MLIR_CAPI_EXPORTED intptr_t
mlirSparseTensorEncodingGetLvlRank(MlirAttribute attr);

/// Returns a specified level-type of the `sparse_tensor.encoding` attribute.
MLIR_CAPI_EXPORTED MlirSparseTensorLevelType
mlirSparseTensorEncodingAttrGetLvlType(MlirAttribute attr, intptr_t lvl);

/// Returns a specified level-format of the `sparse_tensor.encoding` attribute.
MLIR_CAPI_EXPORTED enum MlirSparseTensorLevelFormat
mlirSparseTensorEncodingAttrGetLvlFmt(MlirAttribute attr, intptr_t lvl);

/// Returns the dimension-to-level mapping of the `sparse_tensor.encoding`
/// attribute.
MLIR_CAPI_EXPORTED MlirAffineMap
mlirSparseTensorEncodingAttrGetDimToLvl(MlirAttribute attr);

/// Returns the level-to-dimension mapping of the `sparse_tensor.encoding`
/// attribute.
MLIR_CAPI_EXPORTED MlirAffineMap
mlirSparseTensorEncodingAttrGetLvlToDim(MlirAttribute attr);

/// Returns the position bitwidth of the `sparse_tensor.encoding` attribute.
MLIR_CAPI_EXPORTED int
mlirSparseTensorEncodingAttrGetPosWidth(MlirAttribute attr);

/// Returns the coordinate bitwidth of the `sparse_tensor.encoding` attribute.
MLIR_CAPI_EXPORTED int
mlirSparseTensorEncodingAttrGetCrdWidth(MlirAttribute attr);

/// Returns the explicit value of the `sparse_tensor.encoding` attribute.
MLIR_CAPI_EXPORTED MlirAttribute
mlirSparseTensorEncodingAttrGetExplicitVal(MlirAttribute attr);

/// Returns the implicit value of the `sparse_tensor.encoding` attribute.
MLIR_CAPI_EXPORTED MlirAttribute
mlirSparseTensorEncodingAttrGetImplicitVal(MlirAttribute attr);

MLIR_CAPI_EXPORTED unsigned
mlirSparseTensorEncodingAttrGetStructuredN(MlirSparseTensorLevelType lvlType);

MLIR_CAPI_EXPORTED unsigned
mlirSparseTensorEncodingAttrGetStructuredM(MlirSparseTensorLevelType lvlType);

MLIR_CAPI_EXPORTED MlirSparseTensorLevelType
mlirSparseTensorEncodingAttrBuildLvlType(
    enum MlirSparseTensorLevelFormat lvlFmt,
    const enum MlirSparseTensorLevelPropertyNondefault *properties,
    unsigned propSize, unsigned n, unsigned m);

#ifdef __cplusplus
}
#endif

#include "mlir/Dialect/SparseTensor/Transforms/Passes.capi.h.inc"

#endif // MLIR_C_DIALECT_SPARSETENSOR_H
