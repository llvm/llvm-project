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
            static_cast<int>(DimLevelType::Dense) &&
        static_cast<int>(MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED) ==
            static_cast<int>(DimLevelType::Compressed) &&
        static_cast<int>(MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_NU) ==
            static_cast<int>(DimLevelType::CompressedNu) &&
        static_cast<int>(MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_NO) ==
            static_cast<int>(DimLevelType::CompressedNo) &&
        static_cast<int>(MLIR_SPARSE_TENSOR_DIM_LEVEL_COMPRESSED_NU_NO) ==
            static_cast<int>(DimLevelType::CompressedNuNo) &&
        static_cast<int>(MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON) ==
            static_cast<int>(DimLevelType::Singleton) &&
        static_cast<int>(MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON_NU) ==
            static_cast<int>(DimLevelType::SingletonNu) &&
        static_cast<int>(MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON_NO) ==
            static_cast<int>(DimLevelType::SingletonNo) &&
        static_cast<int>(MLIR_SPARSE_TENSOR_DIM_LEVEL_SINGLETON_NU_NO) ==
            static_cast<int>(DimLevelType::SingletonNuNo),
    "MlirSparseTensorDimLevelType (C-API) and DimLevelType (C++) mismatch");

bool mlirAttributeIsASparseTensorEncodingAttr(MlirAttribute attr) {
  return isa<SparseTensorEncodingAttr>(unwrap(attr));
}

MlirAttribute mlirSparseTensorEncodingAttrGet(
    MlirContext ctx, intptr_t lvlRank,
    MlirSparseTensorDimLevelType const *dimLevelTypes,
    MlirAffineMap dimOrdering, MlirAffineMap higherOrdering, int posWidth,
    int crdWidth) {
  SmallVector<DimLevelType> cppDimLevelTypes;
  cppDimLevelTypes.reserve(lvlRank);
  for (intptr_t l = 0; l < lvlRank; ++l)
    cppDimLevelTypes.push_back(static_cast<DimLevelType>(dimLevelTypes[l]));
  return wrap(SparseTensorEncodingAttr::get(
      unwrap(ctx), cppDimLevelTypes, unwrap(dimOrdering),
      unwrap(higherOrdering), posWidth, crdWidth));
}

MlirAffineMap mlirSparseTensorEncodingAttrGetDimOrdering(MlirAttribute attr) {
  return wrap(cast<SparseTensorEncodingAttr>(unwrap(attr)).getDimOrdering());
}

MlirAffineMap
mlirSparseTensorEncodingAttrGetHigherOrdering(MlirAttribute attr) {
  return wrap(cast<SparseTensorEncodingAttr>(unwrap(attr)).getHigherOrdering());
}

intptr_t mlirSparseTensorEncodingGetLvlRank(MlirAttribute attr) {
  return cast<SparseTensorEncodingAttr>(unwrap(attr)).getLvlRank();
}

MlirSparseTensorDimLevelType
mlirSparseTensorEncodingAttrGetDimLevelType(MlirAttribute attr, intptr_t lvl) {
  return static_cast<MlirSparseTensorDimLevelType>(
      cast<SparseTensorEncodingAttr>(unwrap(attr)).getLvlType(lvl));
}

int mlirSparseTensorEncodingAttrGetPosWidth(MlirAttribute attr) {
  return cast<SparseTensorEncodingAttr>(unwrap(attr)).getPosWidth();
}

int mlirSparseTensorEncodingAttrGetCrdWidth(MlirAttribute attr) {
  return cast<SparseTensorEncodingAttr>(unwrap(attr)).getCrdWidth();
}
