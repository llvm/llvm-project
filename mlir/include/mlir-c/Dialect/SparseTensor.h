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

enum MlirBaseSparseTensorLevelType {
  MLIR_SPARSE_TENSOR_LEVEL_DENSE = 65536,                   // 0x00_00_0001_0000
  MLIR_SPARSE_TENSOR_LEVEL_COMPRESSED = 131072,             // 0x00_00_0002_0000
  MLIR_SPARSE_TENSOR_LEVEL_COMPRESSED_NU = 131073,          // 0x00_00_0002_0001
  MLIR_SPARSE_TENSOR_LEVEL_COMPRESSED_NO = 131074,          // 0x00_00_0002_0002
  MLIR_SPARSE_TENSOR_LEVEL_COMPRESSED_NU_NO = 131075,       // 0x00_00_0002_0003
  MLIR_SPARSE_TENSOR_LEVEL_SINGLETON = 262144,              // 0x00_00_0004_0000
  MLIR_SPARSE_TENSOR_LEVEL_SINGLETON_NU = 262145,           // 0x00_00_0004_0001
  MLIR_SPARSE_TENSOR_LEVEL_SINGLETON_NO = 262146,           // 0x00_00_0004_0002
  MLIR_SPARSE_TENSOR_LEVEL_SINGLETON_NU_NO = 262147,        // 0x00_00_0004_0003
  MLIR_SPARSE_TENSOR_LEVEL_LOOSE_COMPRESSED = 524288,       // 0x00_00_0008_0000
  MLIR_SPARSE_TENSOR_LEVEL_LOOSE_COMPRESSED_NU = 524289,    // 0x00_00_0008_0001
  MLIR_SPARSE_TENSOR_LEVEL_LOOSE_COMPRESSED_NO = 524290,    // 0x00_00_0008_0002
  MLIR_SPARSE_TENSOR_LEVEL_LOOSE_COMPRESSED_NU_NO = 524291, // 0x00_00_0008_0003
  MLIR_SPARSE_TENSOR_LEVEL_N_OUT_OF_M = 1048576,            // 0x00_00_0010_0000
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
    MlirAffineMap lvlTodim, int posWidth, int crdWidth);

/// Returns the level-rank of the `sparse_tensor.encoding` attribute.
MLIR_CAPI_EXPORTED intptr_t
mlirSparseTensorEncodingGetLvlRank(MlirAttribute attr);

/// Returns a specified level-type of the `sparse_tensor.encoding` attribute.
MLIR_CAPI_EXPORTED MlirSparseTensorLevelType
mlirSparseTensorEncodingAttrGetLvlType(MlirAttribute attr, intptr_t lvl);

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

#ifdef __cplusplus
}
#endif

#include "mlir/Dialect/SparseTensor/Transforms/Passes.capi.h.inc"

#endif // MLIR_C_DIALECT_SPARSETENSOR_H
