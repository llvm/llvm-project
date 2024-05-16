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
            static_cast<int>(LevelFormat::Dense) &&
        static_cast<int>(MLIR_SPARSE_TENSOR_LEVEL_COMPRESSED) ==
            static_cast<int>(LevelFormat::Compressed) &&
        static_cast<int>(MLIR_SPARSE_TENSOR_LEVEL_SINGLETON) ==
            static_cast<int>(LevelFormat::Singleton) &&
        static_cast<int>(MLIR_SPARSE_TENSOR_LEVEL_LOOSE_COMPRESSED) ==
            static_cast<int>(LevelFormat::LooseCompressed) &&
        static_cast<int>(MLIR_SPARSE_TENSOR_LEVEL_N_OUT_OF_M) ==
            static_cast<int>(LevelFormat::NOutOfM),
    "MlirSparseTensorLevelFormat (C-API) and LevelFormat (C++) mismatch");

static_assert(static_cast<int>(MLIR_SPARSE_PROPERTY_NON_ORDERED) ==
                      static_cast<int>(LevelPropNonDefault::Nonordered) &&
                  static_cast<int>(MLIR_SPARSE_PROPERTY_NON_UNIQUE) ==
                      static_cast<int>(LevelPropNonDefault::Nonunique),
              "MlirSparseTensorLevelProperty (C-API) and "
              "LevelPropertyNondefault (C++) mismatch");

bool mlirAttributeIsASparseTensorEncodingAttr(MlirAttribute attr) {
  return isa<SparseTensorEncodingAttr>(unwrap(attr));
}

MlirAttribute mlirSparseTensorEncodingAttrGet(
    MlirContext ctx, intptr_t lvlRank,
    MlirSparseTensorLevelType const *lvlTypes, MlirAffineMap dimToLvl,
    MlirAffineMap lvlToDim, int posWidth, int crdWidth,
    MlirAttribute explicitVal, MlirAttribute implicitVal) {
  SmallVector<LevelType> cppLvlTypes;

  cppLvlTypes.reserve(lvlRank);
  for (intptr_t l = 0; l < lvlRank; ++l)
    cppLvlTypes.push_back(static_cast<LevelType>(lvlTypes[l]));

  return wrap(SparseTensorEncodingAttr::get(
      unwrap(ctx), cppLvlTypes, unwrap(dimToLvl), unwrap(lvlToDim), posWidth,
      crdWidth, unwrap(explicitVal), unwrap(implicitVal)));
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

enum MlirSparseTensorLevelFormat
mlirSparseTensorEncodingAttrGetLvlFmt(MlirAttribute attr, intptr_t lvl) {
  LevelType lt =
      static_cast<LevelType>(mlirSparseTensorEncodingAttrGetLvlType(attr, lvl));
  return static_cast<MlirSparseTensorLevelFormat>(lt.getLvlFmt());
}

int mlirSparseTensorEncodingAttrGetPosWidth(MlirAttribute attr) {
  return cast<SparseTensorEncodingAttr>(unwrap(attr)).getPosWidth();
}

int mlirSparseTensorEncodingAttrGetCrdWidth(MlirAttribute attr) {
  return cast<SparseTensorEncodingAttr>(unwrap(attr)).getCrdWidth();
}

MlirAttribute mlirSparseTensorEncodingAttrGetExplicitVal(MlirAttribute attr) {
  return wrap(cast<SparseTensorEncodingAttr>(unwrap(attr)).getExplicitVal());
}

MlirAttribute mlirSparseTensorEncodingAttrGetImplicitVal(MlirAttribute attr) {
  return wrap(cast<SparseTensorEncodingAttr>(unwrap(attr)).getImplicitVal());
}

MlirSparseTensorLevelType mlirSparseTensorEncodingAttrBuildLvlType(
    enum MlirSparseTensorLevelFormat lvlFmt,
    const enum MlirSparseTensorLevelPropertyNondefault *properties,
    unsigned size, unsigned n, unsigned m) {

  std::vector<LevelPropNonDefault> props;
  for (unsigned i = 0; i < size; i++)
    props.push_back(static_cast<LevelPropNonDefault>(properties[i]));

  return static_cast<MlirSparseTensorLevelType>(
      *buildLevelType(static_cast<LevelFormat>(lvlFmt), props, n, m));
}

unsigned
mlirSparseTensorEncodingAttrGetStructuredN(MlirSparseTensorLevelType lvlType) {
  return getN(static_cast<LevelType>(lvlType));
}

unsigned
mlirSparseTensorEncodingAttrGetStructuredM(MlirSparseTensorLevelType lvlType) {
  return getM(static_cast<LevelType>(lvlType));
}
