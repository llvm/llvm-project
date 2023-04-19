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
  MLIR_SPARSE_TENSOR_DIM_LEVEL_DENSE = 4,                     // 0b0001_00
  MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED = 8,                // 0b0010_00
  MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_NU = 9,             // 0b0010_01
  MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_NO = 10,            // 0b0010_10
  MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_NU_NO = 11,         // 0b0010_11
  MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON = 16,                // 0b0100_00
  MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON_NU = 17,             // 0b0100_01
  MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON_NO = 18,             // 0b0100_10
  MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON_NU_NO = 19,          // 0b0100_11
  MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_WITH_HI = 32,       // 0b1000_00
  MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_WITH_HI_NU = 33,    // 0b1000_01
  MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_WITH_HI_NO = 34,    // 0b1000_10
  MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_WITH_HI_NU_NO = 35, // 0b1000_11
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
    enum MlirSparseTensorDimLevelType const *dimLevelTypes,
    MlirAffineMap dimOrdering, MlirAffineMap higherOrdering, int posWidth,
    int crdWidth);

/// Returns the level-rank of the `sparse_tensor.encoding` attribute.
MLIR_CAPI_EXPORTED intptr_t
mlirSparseTensorEncodingGetLvlRank(MlirAttribute attr);

/// Returns a specified level-type of the `sparse_tensor.encoding` attribute.
MLIR_CAPI_EXPORTED enum MlirSparseTensorDimLevelType
mlirSparseTensorEncodingAttrGetDimLevelType(MlirAttribute attr, intptr_t lvl);

/// Returns the dimension-ordering of the `sparse_tensor.encoding` attribute.
MLIR_CAPI_EXPORTED MlirAffineMap
mlirSparseTensorEncodingAttrGetDimOrdering(MlirAttribute attr);

/// Returns the higher-ordering of the `sparse_tensor.encoding` attribute.
MLIR_CAPI_EXPORTED MlirAffineMap
mlirSparseTensorEncodingAttrGetHigherOrdering(MlirAttribute attr);

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
