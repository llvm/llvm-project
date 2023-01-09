//===- Quant.cpp - C Interface for Quant dialect --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/Quant.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/Quant/QuantTypes.h"

using namespace mlir;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(quant, quant, quant::QuantizationDialect)

//===---------------------------------------------------------------------===//
// QuantizedType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAQuantizedType(MlirType type) {
  return unwrap(type).isa<quant::QuantizedType>();
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
  return wrap(unwrap(type).cast<quant::QuantizedType>().getExpressedType());
}

unsigned mlirQuantizedTypeGetFlags(MlirType type) {
  return unwrap(type).cast<quant::QuantizedType>().getFlags();
}

bool mlirQuantizedTypeIsSigned(MlirType type) {
  return unwrap(type).cast<quant::QuantizedType>().isSigned();
}

MlirType mlirQuantizedTypeGetStorageType(MlirType type) {
  return wrap(unwrap(type).cast<quant::QuantizedType>().getStorageType());
}

int64_t mlirQuantizedTypeGetStorageTypeMin(MlirType type) {
  return unwrap(type).cast<quant::QuantizedType>().getStorageTypeMin();
}

int64_t mlirQuantizedTypeGetStorageTypeMax(MlirType type) {
  return unwrap(type).cast<quant::QuantizedType>().getStorageTypeMax();
}

unsigned mlirQuantizedTypeGetStorageTypeIntegralWidth(MlirType type) {
  return unwrap(type)
      .cast<quant::QuantizedType>()
      .getStorageTypeIntegralWidth();
}

bool mlirQuantizedTypeIsCompatibleExpressedType(MlirType type,
                                                MlirType candidate) {
  return unwrap(type).cast<quant::QuantizedType>().isCompatibleExpressedType(
      unwrap(candidate));
}

MlirType mlirQuantizedTypeGetQuantizedElementType(MlirType type) {
  return wrap(quant::QuantizedType::getQuantizedElementType(unwrap(type)));
}

MlirType mlirQuantizedTypeCastFromStorageType(MlirType type,
                                              MlirType candidate) {
  return wrap(unwrap(type).cast<quant::QuantizedType>().castFromStorageType(
      unwrap(candidate)));
}

MlirType mlirQuantizedTypeCastToStorageType(MlirType type) {
  return wrap(quant::QuantizedType::castToStorageType(
      unwrap(type).cast<quant::QuantizedType>()));
}

MlirType mlirQuantizedTypeCastFromExpressedType(MlirType type,
                                                MlirType candidate) {
  return wrap(unwrap(type).cast<quant::QuantizedType>().castFromExpressedType(
      unwrap(candidate)));
}

MlirType mlirQuantizedTypeCastToExpressedType(MlirType type) {
  return wrap(quant::QuantizedType::castToExpressedType(unwrap(type)));
}

MlirType mlirQuantizedTypeCastExpressedToStorageType(MlirType type,
                                                     MlirType candidate) {
  return wrap(
      unwrap(type).cast<quant::QuantizedType>().castExpressedToStorageType(
          unwrap(candidate)));
}

//===---------------------------------------------------------------------===//
// AnyQuantizedType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAAnyQuantizedType(MlirType type) {
  return unwrap(type).isa<quant::AnyQuantizedType>();
}

MlirType mlirAnyQuantizedTypeGet(unsigned flags, MlirType storageType,
                                 MlirType expressedType, int64_t storageTypeMin,
                                 int64_t storageTypeMax) {
  return wrap(quant::AnyQuantizedType::get(flags, unwrap(storageType),
                                           unwrap(expressedType),
                                           storageTypeMin, storageTypeMax));
}

//===---------------------------------------------------------------------===//
// UniformQuantizedType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAUniformQuantizedType(MlirType type) {
  return unwrap(type).isa<quant::UniformQuantizedType>();
}

MlirType mlirUniformQuantizedTypeGet(unsigned flags, MlirType storageType,
                                     MlirType expressedType, double scale,
                                     int64_t zeroPoint, int64_t storageTypeMin,
                                     int64_t storageTypeMax) {
  return wrap(quant::UniformQuantizedType::get(
      flags, unwrap(storageType), unwrap(expressedType), scale, zeroPoint,
      storageTypeMin, storageTypeMax));
}

double mlirUniformQuantizedTypeGetScale(MlirType type) {
  return unwrap(type).cast<quant::UniformQuantizedType>().getScale();
}

int64_t mlirUniformQuantizedTypeGetZeroPoint(MlirType type) {
  return unwrap(type).cast<quant::UniformQuantizedType>().getZeroPoint();
}

bool mlirUniformQuantizedTypeIsFixedPoint(MlirType type) {
  return unwrap(type).cast<quant::UniformQuantizedType>().isFixedPoint();
}

//===---------------------------------------------------------------------===//
// UniformQuantizedPerAxisType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAUniformQuantizedPerAxisType(MlirType type) {
  return unwrap(type).isa<quant::UniformQuantizedPerAxisType>();
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

intptr_t mlirUniformQuantizedPerAxisTypeGetNumDims(MlirType type) {
  return unwrap(type)
      .cast<quant::UniformQuantizedPerAxisType>()
      .getScales()
      .size();
}

double mlirUniformQuantizedPerAxisTypeGetScale(MlirType type, intptr_t pos) {
  return unwrap(type)
      .cast<quant::UniformQuantizedPerAxisType>()
      .getScales()[pos];
}

int64_t mlirUniformQuantizedPerAxisTypeGetZeroPoint(MlirType type,
                                                    intptr_t pos) {
  return unwrap(type)
      .cast<quant::UniformQuantizedPerAxisType>()
      .getZeroPoints()[pos];
}

int32_t mlirUniformQuantizedPerAxisTypeGetQuantizedDimension(MlirType type) {
  return unwrap(type)
      .cast<quant::UniformQuantizedPerAxisType>()
      .getQuantizedDimension();
}

bool mlirUniformQuantizedPerAxisTypeIsFixedPoint(MlirType type) {
  return unwrap(type).cast<quant::UniformQuantizedPerAxisType>().isFixedPoint();
}

//===---------------------------------------------------------------------===//
// CalibratedQuantizedType
//===---------------------------------------------------------------------===//

bool mlirTypeIsACalibratedQuantizedType(MlirType type) {
  return unwrap(type).isa<quant::CalibratedQuantizedType>();
}

MlirType mlirCalibratedQuantizedTypeGet(MlirType expressedType, double min,
                                        double max) {
  return wrap(
      quant::CalibratedQuantizedType::get(unwrap(expressedType), min, max));
}

double mlirCalibratedQuantizedTypeGetMin(MlirType type) {
  return unwrap(type).cast<quant::CalibratedQuantizedType>().getMin();
}

double mlirCalibratedQuantizedTypeGetMax(MlirType type) {
  return unwrap(type).cast<quant::CalibratedQuantizedType>().getMax();
}
