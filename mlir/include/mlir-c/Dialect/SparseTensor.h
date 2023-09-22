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
/// These correspond to SparseTensorEncodingAttr::DimLevelType in the C++ API.
/// If updating, keep them in sync and update the static_assert in the impl
/// file.
enum MlirSparseTensorDimLevelType {
  MLIR_SPARSE_TENSOR_DIM_LEVEL_DENSE = 1,              // 0b00000000_00000001
  MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED = 2,         // 0b00000000_00000010
  MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_NU = 258,    // 0b00000001_00000010
  MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_NO = 514,    // 0b00000010_00000010
  MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_NU_NO = 770, // 0b00000011_00000010
  MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON = 4,          // 0b00000000_00000100
  MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON_NU = 260,     // 0b00000001_00000100
  MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON_NO = 516,     // 0b00000010_00000100
  MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON_NU_NO = 772,  // 0b00000011_00000100
  MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_WITH_HI = 1026,
  // 0b00000100_00000010
  MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_WITH_HI_NU = 1282,
  // 0b00000101_00000010
  MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_WITH_HI_NO = 1538,
  // 0b00000110_00000010
  MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_WITH_HI_NU_NO = 1794,
  // 0b00000111_00000010
  MLIR_SPARSE_TENSOR_DIM_LEVEL_TWO_OUT_OF_FOUR = 2050, // 0b00001000_00000010
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
    enum MlirSparseTensorDimLevelType const *lvlTypes, MlirAffineMap dimToLvl,
    int posWidth, int crdWidth);

/// Returns the level-rank of the `sparse_tensor.encoding` attribute.
MLIR_CAPI_EXPORTED intptr_t
mlirSparseTensorEncodingGetLvlRank(MlirAttribute attr);

/// Returns a specified level-type of the `sparse_tensor.encoding` attribute.
MLIR_CAPI_EXPORTED enum MlirSparseTensorDimLevelType
mlirSparseTensorEncodingAttrGetLvlType(MlirAttribute attr, intptr_t lvl);

/// Returns the dimension-to-level mapping of the `sparse_tensor.encoding`
/// attribute.
MLIR_CAPI_EXPORTED MlirAffineMap
mlirSparseTensorEncodingAttrGetDimToLvl(MlirAttribute attr);

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
