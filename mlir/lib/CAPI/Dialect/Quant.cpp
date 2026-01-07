//===- Quant.cpp - C Interface for Quant dialect --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/Quant.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"

using namespace mlir;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(quant, quant, quant::QuantDialect)

//===---------------------------------------------------------------------===//
// QuantizedType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAQuantizedType(MlirType type) {
  return isa<quant::QuantizedType>(unwrap(type));
}

unsigned mlirQuantizedTypeGetSignedFlag() {
  return quant::QuantizationFlags::Signed;
}

int64_t mlirQuantizedTypeGetDefaultMinimumForInteger(bool isSigned,
                                                     unsigned integralWidth) {
  return quant::QuantizedType::getDefaultMinimumForInteger(isSigned,
                                                           integralWidth);
}

int64_t mlirQuantizedTypeGetDefaultMaximumForInteger(bool isSigned,
                                                     unsigned integralWidth) {
  return quant::QuantizedType::getDefaultMaximumForInteger(isSigned,
                                                           integralWidth);
}

MlirType mlirQuantizedTypeGetExpressedType(MlirType type) {
  return wrap(cast<quant::QuantizedType>(unwrap(type)).getExpressedType());
}

unsigned mlirQuantizedTypeGetFlags(MlirType type) {
  return cast<quant::QuantizedType>(unwrap(type)).getFlags();
}

bool mlirQuantizedTypeIsSigned(MlirType type) {
  return cast<quant::QuantizedType>(unwrap(type)).isSigned();
}

MlirType mlirQuantizedTypeGetStorageType(MlirType type) {
  return wrap(cast<quant::QuantizedType>(unwrap(type)).getStorageType());
}

int64_t mlirQuantizedTypeGetStorageTypeMin(MlirType type) {
  return cast<quant::QuantizedType>(unwrap(type)).getStorageTypeMin();
}

int64_t mlirQuantizedTypeGetStorageTypeMax(MlirType type) {
  return cast<quant::QuantizedType>(unwrap(type)).getStorageTypeMax();
}

unsigned mlirQuantizedTypeGetStorageTypeIntegralWidth(MlirType type) {
  return cast<quant::QuantizedType>(unwrap(type)).getStorageTypeIntegralWidth();
}

bool mlirQuantizedTypeIsCompatibleExpressedType(MlirType type,
                                                MlirType candidate) {
  return cast<quant::QuantizedType>(unwrap(type))
      .isCompatibleExpressedType(unwrap(candidate));
}

MlirType mlirQuantizedTypeGetQuantizedElementType(MlirType type) {
  return wrap(quant::QuantizedType::getQuantizedElementType(unwrap(type)));
}

MlirType mlirQuantizedTypeCastFromStorageType(MlirType type,
                                              MlirType candidate) {
  return wrap(cast<quant::QuantizedType>(unwrap(type))
                  .castFromStorageType(unwrap(candidate)));
}

MlirType mlirQuantizedTypeCastToStorageType(MlirType type) {
  return wrap(quant::QuantizedType::castToStorageType(
      cast<quant::QuantizedType>(unwrap(type))));
}

MlirType mlirQuantizedTypeCastFromExpressedType(MlirType type,
                                                MlirType candidate) {
  return wrap(cast<quant::QuantizedType>(unwrap(type))
                  .castFromExpressedType(unwrap(candidate)));
}

MlirType mlirQuantizedTypeCastToExpressedType(MlirType type) {
  return wrap(quant::QuantizedType::castToExpressedType(unwrap(type)));
}

MlirType mlirQuantizedTypeCastExpressedToStorageType(MlirType type,
                                                     MlirType candidate) {
  return wrap(cast<quant::QuantizedType>(unwrap(type))
                  .castExpressedToStorageType(unwrap(candidate)));
}

//===---------------------------------------------------------------------===//
// AnyQuantizedType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAAnyQuantizedType(MlirType type) {
  return isa<quant::AnyQuantizedType>(unwrap(type));
}

MlirTypeID mlirAnyQuantizedTypeGetTypeID(void) {
  return wrap(quant::AnyQuantizedType::getTypeID());
}

MlirType mlirAnyQuantizedTypeGet(unsigned flags, MlirType storageType,
                                 MlirType expressedType, int64_t storageTypeMin,
                                 int64_t storageTypeMax) {
  return wrap(quant::AnyQuantizedType::get(flags, unwrap(storageType),
                                           unwrap(expressedType),
                                           storageTypeMin, storageTypeMax));
}

MlirStringRef mlirAnyQuantizedTypeGetName(void) {
  return wrap(quant::AnyQuantizedType::name);
}

//===---------------------------------------------------------------------===//
// UniformQuantizedType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAUniformQuantizedType(MlirType type) {
  return isa<quant::UniformQuantizedType>(unwrap(type));
}

MlirTypeID mlirUniformQuantizedTypeGetTypeID(void) {
  return wrap(quant::UniformQuantizedType::getTypeID());
}

MlirType mlirUniformQuantizedTypeGet(unsigned flags, MlirType storageType,
                                     MlirType expressedType, double scale,
                                     int64_t zeroPoint, int64_t storageTypeMin,
                                     int64_t storageTypeMax) {
  return wrap(quant::UniformQuantizedType::get(
      flags, unwrap(storageType), unwrap(expressedType), scale, zeroPoint,
      storageTypeMin, storageTypeMax));
}

MlirStringRef mlirUniformQuantizedTypeGetName(void) {
  return wrap(quant::UniformQuantizedType::name);
}

double mlirUniformQuantizedTypeGetScale(MlirType type) {
  return cast<quant::UniformQuantizedType>(unwrap(type)).getScale();
}

int64_t mlirUniformQuantizedTypeGetZeroPoint(MlirType type) {
  return cast<quant::UniformQuantizedType>(unwrap(type)).getZeroPoint();
}

bool mlirUniformQuantizedTypeIsFixedPoint(MlirType type) {
  return cast<quant::UniformQuantizedType>(unwrap(type)).isFixedPoint();
}

//===---------------------------------------------------------------------===//
// UniformQuantizedPerAxisType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAUniformQuantizedPerAxisType(MlirType type) {
  return isa<quant::UniformQuantizedPerAxisType>(unwrap(type));
}

MlirTypeID mlirUniformQuantizedPerAxisTypeGetTypeID(void) {
  return wrap(quant::UniformQuantizedPerAxisType::getTypeID());
}

MlirType mlirUniformQuantizedPerAxisTypeGet(
    unsigned flags, MlirType storageType, MlirType expressedType,
    intptr_t nDims, double *scales, int64_t *zeroPoints,
    int32_t quantizedDimension, int64_t storageTypeMin,
    int64_t storageTypeMax) {
  return wrap(quant::UniformQuantizedPerAxisType::get(
      flags, unwrap(storageType), unwrap(expressedType),
      llvm::ArrayRef(scales, nDims), llvm::ArrayRef(zeroPoints, nDims),
      quantizedDimension, storageTypeMin, storageTypeMax));
}

MlirStringRef mlirUniformQuantizedPerAxisTypeGetName(void) {
  return wrap(quant::UniformQuantizedPerAxisType::name);
}

intptr_t mlirUniformQuantizedPerAxisTypeGetNumDims(MlirType type) {
  return cast<quant::UniformQuantizedPerAxisType>(unwrap(type))
      .getScales()
      .size();
}

double mlirUniformQuantizedPerAxisTypeGetScale(MlirType type, intptr_t pos) {
  return cast<quant::UniformQuantizedPerAxisType>(unwrap(type))
      .getScales()[pos];
}

int64_t mlirUniformQuantizedPerAxisTypeGetZeroPoint(MlirType type,
                                                    intptr_t pos) {
  return cast<quant::UniformQuantizedPerAxisType>(unwrap(type))
      .getZeroPoints()[pos];
}

int32_t mlirUniformQuantizedPerAxisTypeGetQuantizedDimension(MlirType type) {
  return cast<quant::UniformQuantizedPerAxisType>(unwrap(type))
      .getQuantizedDimension();
}

bool mlirUniformQuantizedPerAxisTypeIsFixedPoint(MlirType type) {
  return cast<quant::UniformQuantizedPerAxisType>(unwrap(type)).isFixedPoint();
}

//===---------------------------------------------------------------------===//
// UniformQuantizedSubChannelType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAUniformQuantizedSubChannelType(MlirType type) {
  return isa<quant::UniformQuantizedSubChannelType>(unwrap(type));
}

MlirTypeID mlirUniformQuantizedSubChannelTypeGetTypeID(void) {
  return wrap(quant::UniformQuantizedSubChannelType::getTypeID());
}

MlirType mlirUniformQuantizedSubChannelTypeGet(
    unsigned flags, MlirType storageType, MlirType expressedType,
    MlirAttribute scalesAttr, MlirAttribute zeroPointsAttr, intptr_t nDims,
    int32_t *quantizedDimensions, int64_t *blockSizes, int64_t storageTypeMin,
    int64_t storageTypeMax) {
  auto scales = dyn_cast<mlir::DenseElementsAttr>(unwrap(scalesAttr));
  auto zeroPoints = dyn_cast<mlir::DenseElementsAttr>(unwrap(zeroPointsAttr));

  if (!scales || !zeroPoints) {
    return {};
  }

  return wrap(quant::UniformQuantizedSubChannelType::get(
      flags, unwrap(storageType), unwrap(expressedType), scales, zeroPoints,
      llvm::ArrayRef<int32_t>(quantizedDimensions, nDims),
      llvm::ArrayRef<int64_t>(blockSizes, nDims), storageTypeMin,
      storageTypeMax));
}

MlirStringRef mlirUniformQuantizedSubChannelTypeGetName(void) {
  return wrap(quant::UniformQuantizedSubChannelType::name);
}

intptr_t mlirUniformQuantizedSubChannelTypeGetNumBlockSizes(MlirType type) {
  return cast<quant::UniformQuantizedSubChannelType>(unwrap(type))
      .getBlockSizes()
      .size();
}

int32_t mlirUniformQuantizedSubChannelTypeGetQuantizedDimension(MlirType type,
                                                                intptr_t pos) {
  return cast<quant::UniformQuantizedSubChannelType>(unwrap(type))
      .getQuantizedDimensions()[pos];
}

int64_t mlirUniformQuantizedSubChannelTypeGetBlockSize(MlirType type,
                                                       intptr_t pos) {
  return cast<quant::UniformQuantizedSubChannelType>(unwrap(type))
      .getBlockSizes()[pos];
}

MlirAttribute mlirUniformQuantizedSubChannelTypeGetScales(MlirType type) {
  return wrap(
      cast<quant::UniformQuantizedSubChannelType>(unwrap(type)).getScales());
}

MlirAttribute mlirUniformQuantizedSubChannelTypeGetZeroPoints(MlirType type) {
  return wrap(cast<quant::UniformQuantizedSubChannelType>(unwrap(type))
                  .getZeroPoints());
}

//===---------------------------------------------------------------------===//
// CalibratedQuantizedType
//===---------------------------------------------------------------------===//

bool mlirTypeIsACalibratedQuantizedType(MlirType type) {
  return isa<quant::CalibratedQuantizedType>(unwrap(type));
}

MlirTypeID mlirCalibratedQuantizedTypeGetTypeID(void) {
  return wrap(quant::CalibratedQuantizedType::getTypeID());
}

MlirType mlirCalibratedQuantizedTypeGet(MlirType expressedType, double min,
                                        double max) {
  return wrap(
      quant::CalibratedQuantizedType::get(unwrap(expressedType), min, max));
}

MlirStringRef mlirCalibratedQuantizedTypeGetName(void) {
  return wrap(quant::CalibratedQuantizedType::name);
}

double mlirCalibratedQuantizedTypeGetMin(MlirType type) {
  return cast<quant::CalibratedQuantizedType>(unwrap(type)).getMin();
}

double mlirCalibratedQuantizedTypeGetMax(MlirType type) {
  return cast<quant::CalibratedQuantizedType>(unwrap(type)).getMax();
}
