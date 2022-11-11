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
  MLIR_SPARSE_TENSOR_DIM_LEVEL_DENSE = 4,             // 0b001_00
  MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED = 8,        // 0b010_00
  MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_NU = 9,     // 0b010_01
  MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_NO = 10,    // 0b010_10
  MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_NU_NO = 11, // 0b010_11
  MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON = 16,        // 0b100_00
  MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON_NU = 17,     // 0b100_01
  MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON_NO = 18,     // 0b100_10
  MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON_NU_NO = 19,  // 0b100_11
};

//===----------------------------------------------------------------------===//
// SparseTensorEncodingAttr
//===----------------------------------------------------------------------===//

/// Checks whether the given attribute is a sparse_tensor.encoding attribute.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsASparseTensorEncodingAttr(MlirAttribute attr);

/// Creates a sparse_tensor.encoding attribute with the given parameters.
MLIR_CAPI_EXPORTED MlirAttribute mlirSparseTensorEncodingAttrGet(
    MlirContext ctx, intptr_t numDimLevelTypes,
    enum MlirSparseTensorDimLevelType const *dimLevelTypes,
    MlirAffineMap dimOrdering, MlirAffineMap higherOrdering,
    int pointerBitWidth, int indexBitWidth);

/// Returns the number of dim level types in a sparse_tensor.encoding attribute.
MLIR_CAPI_EXPORTED intptr_t
mlirSparseTensorEncodingGetNumDimLevelTypes(MlirAttribute attr);

/// Returns a specified dim level type in a sparse_tensor.encoding attribute.
MLIR_CAPI_EXPORTED enum MlirSparseTensorDimLevelType
mlirSparseTensorEncodingAttrGetDimLevelType(MlirAttribute attr, intptr_t pos);

/// Returns the dimension ordering in a sparse_tensor.encoding attribute.
MLIR_CAPI_EXPORTED MlirAffineMap
mlirSparseTensorEncodingAttrGetDimOrdering(MlirAttribute attr);

/// Returns the higher ordering in a sparse_tensor.encoding attribute.
MLIR_CAPI_EXPORTED MlirAffineMap
mlirSparseTensorEncodingAttrGetHigherOrdering(MlirAttribute attr);

/// Returns the pointer bit width in a sparse_tensor.encoding attribute.
MLIR_CAPI_EXPORTED int
mlirSparseTensorEncodingAttrGetPointerBitWidth(MlirAttribute attr);

/// Returns the index bit width in a sparse_tensor.encoding attribute.
MLIR_CAPI_EXPORTED int
mlirSparseTensorEncodingAttrGetIndexBitWidth(MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#include "mlir/Dialect/SparseTensor/Transforms/Passes.capi.h.inc"

#endif // MLIR_C_DIALECT_SPARSETENSOR_H
