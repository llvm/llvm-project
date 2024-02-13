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
    static_cast<int>(MLIR_SPARSE_TENSOR_LEVEL_DENSE) ==
            static_cast<int>(LevelType::Dense) &&
        static_cast<int>(MLIR_SPARSE_TENSOR_LEVEL_COMPRESSED) ==
            static_cast<int>(LevelType::Compressed) &&
        static_cast<int>(MLIR_SPARSE_TENSOR_LEVEL_COMPRESSED_NU) ==
            static_cast<int>(LevelType::CompressedNu) &&
        static_cast<int>(MLIR_SPARSE_TENSOR_LEVEL_COMPRESSED_NO) ==
            static_cast<int>(LevelType::CompressedNo) &&
        static_cast<int>(MLIR_SPARSE_TENSOR_LEVEL_COMPRESSED_NU_NO) ==
            static_cast<int>(LevelType::CompressedNuNo) &&
        static_cast<int>(MLIR_SPARSE_TENSOR_LEVEL_SINGLETON) ==
            static_cast<int>(LevelType::Singleton) &&
        static_cast<int>(MLIR_SPARSE_TENSOR_LEVEL_SINGLETON_NU) ==
            static_cast<int>(LevelType::SingletonNu) &&
        static_cast<int>(MLIR_SPARSE_TENSOR_LEVEL_SINGLETON_NO) ==
            static_cast<int>(LevelType::SingletonNo) &&
        static_cast<int>(MLIR_SPARSE_TENSOR_LEVEL_SINGLETON_NU_NO) ==
            static_cast<int>(LevelType::SingletonNuNo) &&
        static_cast<int>(MLIR_SPARSE_TENSOR_LEVEL_LOOSE_COMPRESSED) ==
            static_cast<int>(LevelType::LooseCompressed) &&
        static_cast<int>(MLIR_SPARSE_TENSOR_LEVEL_LOOSE_COMPRESSED_NU) ==
            static_cast<int>(LevelType::LooseCompressedNu) &&
        static_cast<int>(MLIR_SPARSE_TENSOR_LEVEL_LOOSE_COMPRESSED_NO) ==
            static_cast<int>(LevelType::LooseCompressedNo) &&
        static_cast<int>(MLIR_SPARSE_TENSOR_LEVEL_LOOSE_COMPRESSED_NU_NO) ==
            static_cast<int>(LevelType::LooseCompressedNuNo) &&
        static_cast<int>(MLIR_SPARSE_TENSOR_LEVEL_N_OUT_OF_M) ==
            static_cast<int>(LevelType::NOutOfM),
    "MlirSparseTensorLevelType (C-API) and LevelType (C++) mismatch");

bool mlirAttributeIsASparseTensorEncodingAttr(MlirAttribute attr) {
  return isa<SparseTensorEncodingAttr>(unwrap(attr));
}

MlirAttribute
mlirSparseTensorEncodingAttrGet(MlirContext ctx, intptr_t lvlRank,
                                MlirSparseTensorLevelType const *lvlTypes,
                                MlirAffineMap dimToLvl, MlirAffineMap lvlToDim,
                                int posWidth, int crdWidth) {
  SmallVector<LevelType> cppLvlTypes;
  cppLvlTypes.reserve(lvlRank);
  for (intptr_t l = 0; l < lvlRank; ++l)
    cppLvlTypes.push_back(static_cast<LevelType>(lvlTypes[l]));
  return wrap(SparseTensorEncodingAttr::get(unwrap(ctx), cppLvlTypes,
                                            unwrap(dimToLvl), unwrap(lvlToDim),
                                            posWidth, crdWidth));
}

MlirAffineMap mlirSparseTensorEncodingAttrGetDimToLvl(MlirAttribute attr) {
  return wrap(cast<SparseTensorEncodingAttr>(unwrap(attr)).getDimToLvl());
}

MlirAffineMap mlirSparseTensorEncodingAttrGetLvlToDim(MlirAttribute attr) {
  return wrap(cast<SparseTensorEncodingAttr>(unwrap(attr)).getLvlToDim());
}

intptr_t mlirSparseTensorEncodingGetLvlRank(MlirAttribute attr) {
  return cast<SparseTensorEncodingAttr>(unwrap(attr)).getLvlRank();
}

MlirSparseTensorLevelType
mlirSparseTensorEncodingAttrGetLvlType(MlirAttribute attr, intptr_t lvl) {
  return static_cast<MlirSparseTensorLevelType>(
      cast<SparseTensorEncodingAttr>(unwrap(attr)).getLvlType(lvl));
}

int mlirSparseTensorEncodingAttrGetPosWidth(MlirAttribute attr) {
  return cast<SparseTensorEncodingAttr>(unwrap(attr)).getPosWidth();
}

int mlirSparseTensorEncodingAttrGetCrdWidth(MlirAttribute attr) {
  return cast<SparseTensorEncodingAttr>(unwrap(attr)).getCrdWidth();
}

MlirSparseTensorLevelType
mlirSparseTensorEncodingAttrBuildLvlType(MlirBaseSparseTensorLevelType lvlType,
                                         unsigned n, unsigned m) {
  LevelType lt = static_cast<LevelType>(lvlType);
  return static_cast<MlirSparseTensorLevelType>(*buildLevelType(
      *getLevelFormat(lt), isOrderedLT(lt), isUniqueLT(lt), n, m));
}

unsigned
mlirSparseTensorEncodingAttrGetStructuredN(MlirSparseTensorLevelType lvlType) {
  return getN(static_cast<LevelType>(lvlType));
}

unsigned
mlirSparseTensorEncodingAttrGetStructuredM(MlirSparseTensorLevelType lvlType) {
  return getM(static_cast<LevelType>(lvlType));
}
