//===- Tensor.cpp - C API for SparseTensor dialect ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/SparseTensor.h"
#include "mlir-c/IR.h"
#include "mlir/CAPI/AffineMap.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Support/LLVM.h"

using namespace llvm;
using namespace mlir::sparse_tensor;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(SparseTensor, sparse_tensor,
                                      mlir::sparse_tensor::SparseTensorDialect)

// Ensure the C-API enums are int-castable to C++ equivalents.
static_assert(
    static_cast<int>(MLIR_SPARSE_TENSOR_DIM_LEVEL_DENSE) ==
            static_cast<int>(SparseTensorEncodingAttr::DimLevelType::Dense) &&
        static_cast<int>(MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED) ==
            static_cast<int>(
                SparseTensorEncodingAttr::DimLevelType::Compressed) &&
        static_cast<int>(MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_NU) ==
            static_cast<int>(
                SparseTensorEncodingAttr::DimLevelType::CompressedNu) &&
        static_cast<int>(MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_NO) ==
            static_cast<int>(
                SparseTensorEncodingAttr::DimLevelType::CompressedNo) &&
        static_cast<int>(MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_NU_NO) ==
            static_cast<int>(
                SparseTensorEncodingAttr::DimLevelType::CompressedNuNo) &&
        static_cast<int>(MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON) ==
            static_cast<int>(
                SparseTensorEncodingAttr::DimLevelType::Singleton) &&
        static_cast<int>(MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON_NU) ==
            static_cast<int>(
                SparseTensorEncodingAttr::DimLevelType::SingletonNu) &&
        static_cast<int>(MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON_NO) ==
            static_cast<int>(
                SparseTensorEncodingAttr::DimLevelType::SingletonNo) &&
        static_cast<int>(MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON_NU_NO) ==
            static_cast<int>(
                SparseTensorEncodingAttr::DimLevelType::SingletonNuNo),
    "MlirSparseTensorDimLevelType (C-API) and DimLevelType (C++) mismatch");

bool mlirAttributeIsASparseTensorEncodingAttr(MlirAttribute attr) {
  return unwrap(attr).isa<SparseTensorEncodingAttr>();
}

MlirAttribute mlirSparseTensorEncodingAttrGet(
    MlirContext ctx, intptr_t numDimLevelTypes,
    MlirSparseTensorDimLevelType const *dimLevelTypes,
    MlirAffineMap dimOrdering, MlirAffineMap higherOrdering,
    int pointerBitWidth, int indexBitWidth) {
  SmallVector<SparseTensorEncodingAttr::DimLevelType> cppDimLevelTypes;
  cppDimLevelTypes.resize(numDimLevelTypes);
  for (intptr_t i = 0; i < numDimLevelTypes; ++i)
    cppDimLevelTypes[i] =
        static_cast<SparseTensorEncodingAttr::DimLevelType>(dimLevelTypes[i]);
  return wrap(SparseTensorEncodingAttr::get(
      unwrap(ctx), cppDimLevelTypes, unwrap(dimOrdering),
      unwrap(higherOrdering), pointerBitWidth, indexBitWidth));
}

MlirAffineMap mlirSparseTensorEncodingAttrGetDimOrdering(MlirAttribute attr) {
  return wrap(unwrap(attr).cast<SparseTensorEncodingAttr>().getDimOrdering());
}

MlirAffineMap
mlirSparseTensorEncodingAttrGetHigherOrdering(MlirAttribute attr) {
  return wrap(
      unwrap(attr).cast<SparseTensorEncodingAttr>().getHigherOrdering());
}

intptr_t mlirSparseTensorEncodingGetNumDimLevelTypes(MlirAttribute attr) {
  return unwrap(attr).cast<SparseTensorEncodingAttr>().getDimLevelType().size();
}

MlirSparseTensorDimLevelType
mlirSparseTensorEncodingAttrGetDimLevelType(MlirAttribute attr, intptr_t pos) {
  return static_cast<MlirSparseTensorDimLevelType>(
      unwrap(attr).cast<SparseTensorEncodingAttr>().getDimLevelType()[pos]);
}

int mlirSparseTensorEncodingAttrGetPointerBitWidth(MlirAttribute attr) {
  return unwrap(attr).cast<SparseTensorEncodingAttr>().getPointerBitWidth();
}

int mlirSparseTensorEncodingAttrGetIndexBitWidth(MlirAttribute attr) {
  return unwrap(attr).cast<SparseTensorEncodingAttr>().getIndexBitWidth();
}
