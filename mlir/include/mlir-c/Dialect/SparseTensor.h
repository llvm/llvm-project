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
#include "mlir-c/Registration.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(SparseTensor, sparse_tensor);

/// Dimension level types that define sparse tensors:
///   - MLIR_SPARSE_TENSOR_DIM_LEVEL_DENSE      - dimension is dense, every
///   entry is stored
///   - MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED - dimension is sparse,
///   only nonzeros are stored.
///   - MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON  - dimension contains single
///   coordinate, no siblings.
///
/// These correspond to SparseTensorEncodingAttr::DimLevelType in the C++ API.
/// If updating, keep them in sync and update the static_assert in the impl
/// file.
enum MlirSparseTensorDimLevelType {
  MLIR_SPARSE_TENSOR_DIM_LEVEL_DENSE,
  MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED,
  MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON,
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
    MlirAffineMap dimOrdering, int pointerBitWidth, int indexBitWidth);

/// Returns the number of dim level types in a sparse_tensor.encoding attribute.
MLIR_CAPI_EXPORTED intptr_t
mlirSparseTensorEncodingGetNumDimLevelTypes(MlirAttribute attr);

/// Returns a specified dim level type in a sparse_tensor.encoding attribute.
MLIR_CAPI_EXPORTED enum MlirSparseTensorDimLevelType
mlirSparseTensorEncodingAttrGetDimLevelType(MlirAttribute attr, intptr_t pos);

/// Returns the dimension ordering in a sparse_tensor.encoding attribute.
MLIR_CAPI_EXPORTED MlirAffineMap
mlirSparseTensorEncodingAttrGetDimOrdering(MlirAttribute attr);

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
