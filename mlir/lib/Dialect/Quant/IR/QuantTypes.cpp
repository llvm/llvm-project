//===- QuantOps.cpp - Quantization Type and Ops Implementation --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypeDetail.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::quant;
using namespace mlir::quant::detail;

namespace {

// Return the minimum scale representable in a given float type
double getMinScale(Type expressedType) {
  auto floatType = cast<FloatType>(expressedType);
  return APFloat::getSmallest(floatType.getFloatSemantics()).convertToDouble();
}

// Return the maximum scale representable in a given float type
double getMaxScale(Type expressedType) {
  auto floatType = cast<FloatType>(expressedType);
  return APFloat::getLargest(floatType.getFloatSemantics()).convertToDouble();
}

}  // namespace

unsigned QuantizedType::getFlags() const {
  return static_cast<ImplType *>(impl)->flags;
}

bool QuantizedType::classof(Type type) {
  return llvm::isa<QuantDialect>(type.getDialect());
}

LogicalResult
QuantizedType::verifyInvariants(function_ref<InFlightDiagnostic()> emitError,
                                unsigned flags, Type storageType,
                                Type expressedType, int64_t storageTypeMin,
                                int64_t storageTypeMax) {
  // Verify that the storage type is integral.
  // This restriction may be lifted at some point in favor of using bf16
  // or f16 as exact representations on hardware where that is advantageous.
  auto intStorageType = llvm::dyn_cast<IntegerType>(storageType);
  if (!intStorageType)
    return emitError() << "storage type must be integral";
  unsigned integralWidth = intStorageType.getWidth();

  // Verify storage width.
  if (integralWidth == 0 || integralWidth > MaxStorageBits)
    return emitError() << "illegal storage type size: " << integralWidth;

  // Verify storageTypeMin and storageTypeMax.
  bool isSigned =
      (flags & QuantizationFlags::Signed) == QuantizationFlags::Signed;
  int64_t defaultIntegerMin =
      getDefaultMinimumForInteger(isSigned, integralWidth);
  int64_t defaultIntegerMax =
      getDefaultMaximumForInteger(isSigned, integralWidth);
  if (storageTypeMax - storageTypeMin <= 0 ||
      storageTypeMin < defaultIntegerMin ||
      storageTypeMax > defaultIntegerMax) {
    return emitError() << "illegal storage min and storage max: ("
                       << storageTypeMin << ":" << storageTypeMax << ")";
  }
  return success();
}

Type QuantizedType::getStorageType() const {
  return static_cast<ImplType *>(impl)->storageType;
}

int64_t QuantizedType::getStorageTypeMin() const {
  return static_cast<ImplType *>(impl)->storageTypeMin;
}

int64_t QuantizedType::getStorageTypeMax() const {
  return static_cast<ImplType *>(impl)->storageTypeMax;
}

bool QuantizedType::hasStorageTypeBounds() const {
  unsigned int integralWidth = getStorageTypeIntegralWidth();
  bool isSignedInteger = isSigned();
  int64_t defaultIntegerMin =
      getDefaultMinimumForInteger(isSignedInteger, integralWidth);
  int64_t defaultIntegerMax =
      getDefaultMaximumForInteger(isSignedInteger, integralWidth);
  return defaultIntegerMin != getStorageTypeMin() ||
         defaultIntegerMax != getStorageTypeMax();
}

unsigned QuantizedType::getStorageTypeIntegralWidth() const {
  // NOTE: If ever supporting non-integral storage types, some other scheme
  // for determining the width will be needed.
  return static_cast<ImplType *>(impl)->storageType.getIntOrFloatBitWidth();
}

Type QuantizedType::getExpressedType() const {
  return static_cast<ImplType *>(impl)->expressedType;
}

bool QuantizedType::isCompatibleExpressedType(Type candidateExpressedType) {
  if (llvm::isa<ShapedType>(candidateExpressedType)) {
    return llvm::cast<ShapedType>(candidateExpressedType).getElementType() ==
           getExpressedType();
  }
  return candidateExpressedType == getExpressedType();
}

QuantizedType
QuantizedType::getQuantizedElementType(Type primitiveOrContainerType) {
  if (llvm::isa<ShapedType>(primitiveOrContainerType)) {
    Type elementType =
        llvm::cast<ShapedType>(primitiveOrContainerType).getElementType();
    return llvm::dyn_cast<QuantizedType>(elementType);
  }
  return llvm::dyn_cast<QuantizedType>(primitiveOrContainerType);
}

Type QuantizedType::castFromStorageType(Type candidateType) {
  if (candidateType == getStorageType()) {
    // i.e. i32 -> quant<"uniform[i8:f32]{1.0}">
    return *this;
  }
  if (llvm::isa<RankedTensorType>(candidateType)) {
    // i.e. tensor<4xi8> -> tensor<4x!quant<"uniform[i8:f32]{1.0}">>
    return RankedTensorType::get(
        llvm::cast<RankedTensorType>(candidateType).getShape(),
        getStorageType());
  }
  if (llvm::isa<UnrankedTensorType>(candidateType)) {
    // i.e. tensor<i8> -> tensor<!quant<"uniform[i8:f32]{1.0}">>
    return UnrankedTensorType::get(getStorageType());
  }
  if (llvm::isa<VectorType>(candidateType)) {
    // i.e. tensor<4xi8> -> tensor<4x!quant<"uniform[i8:f32]{1.0}">>
    return VectorType::get(llvm::cast<VectorType>(candidateType).getShape(),
                           getStorageType());
  }

  return nullptr;
}

Type QuantizedType::castToStorageType(Type quantizedType) {
  if (llvm::isa<QuantizedType>(quantizedType)) {
    // i.e. quant<"uniform[i8:f32]{1.0}"> -> i8
    return llvm::cast<QuantizedType>(quantizedType).getStorageType();
  }
  if (llvm::isa<ShapedType>(quantizedType)) {
    // i.e. tensor<4xi8> -> tensor<4x!quant<"uniform[i8:f32]{1.0}">>
    ShapedType sType = llvm::cast<ShapedType>(quantizedType);
    if (!llvm::isa<QuantizedType>(sType.getElementType())) {
      return nullptr;
    }
    Type storageType =
        llvm::cast<QuantizedType>(sType.getElementType()).getStorageType();
    if (llvm::isa<RankedTensorType>(quantizedType)) {
      return RankedTensorType::get(sType.getShape(), storageType);
    }
    if (llvm::isa<UnrankedTensorType>(quantizedType)) {
      return UnrankedTensorType::get(storageType);
    }
    if (llvm::isa<VectorType>(quantizedType)) {
      return VectorType::get(sType.getShape(), storageType);
    }
  }

  return nullptr;
}

Type QuantizedType::castFromExpressedType(Type candidateType) {
  if (candidateType == getExpressedType()) {
    // i.e. f32 -> quant<"uniform[i8:f32]{1.0}">
    return *this;
  }
  if (llvm::isa<ShapedType>(candidateType)) {
    ShapedType candidateShapedType = llvm::cast<ShapedType>(candidateType);
    if (candidateShapedType.getElementType() != getExpressedType()) {
      return nullptr;
    }

    if (llvm::isa<RankedTensorType>(candidateType)) {
      // i.e. tensor<4xf32> -> tensor<4x!quant<"uniform[i8:f32]{1.0}">>
      return RankedTensorType::get(candidateShapedType.getShape(), *this);
    }
    if (llvm::isa<UnrankedTensorType>(candidateType)) {
      // i.e. tensor<xf32> -> tensor<x!quant<"uniform[i8:f32]{1.0}">>
      return UnrankedTensorType::get(*this);
    }
    if (llvm::isa<VectorType>(candidateType)) {
      // i.e. tensor<4xf32> -> tensor<4x!quant<"uniform[i8:f32]{1.0}">>
      return VectorType::get(candidateShapedType.getShape(), *this);
    }
  }

  return nullptr;
}

Type QuantizedType::castToExpressedType(Type quantizedType) {
  if (llvm::isa<QuantizedType>(quantizedType)) {
    // i.e. quant<"uniform[i8:f32]{1.0}"> -> f32
    return llvm::cast<QuantizedType>(quantizedType).getExpressedType();
  }
  if (llvm::isa<ShapedType>(quantizedType)) {
    // i.e. tensor<4xi8> -> tensor<4x!quant<"uniform[i8:f32]{1.0}">>
    ShapedType sType = llvm::cast<ShapedType>(quantizedType);
    if (!llvm::isa<QuantizedType>(sType.getElementType())) {
      return nullptr;
    }
    Type expressedType =
        llvm::cast<QuantizedType>(sType.getElementType()).getExpressedType();
    if (llvm::isa<RankedTensorType>(quantizedType)) {
      return RankedTensorType::get(sType.getShape(), expressedType);
    }
    if (llvm::isa<UnrankedTensorType>(quantizedType)) {
      return UnrankedTensorType::get(expressedType);
    }
    if (llvm::isa<VectorType>(quantizedType)) {
      return VectorType::get(sType.getShape(), expressedType);
    }
  }

  return nullptr;
}

Type QuantizedType::castExpressedToStorageType(Type candidateType) {
  Type expressedQuantizedType = castFromExpressedType(candidateType);
  if (!expressedQuantizedType) {
    return nullptr;
  }
  return QuantizedType::castToStorageType(expressedQuantizedType);
}

AnyQuantizedType AnyQuantizedType::get(unsigned flags, Type storageType,
                                       Type expressedType,
                                       int64_t storageTypeMin,
                                       int64_t storageTypeMax) {
  return Base::get(storageType.getContext(), flags, storageType, expressedType,
                   storageTypeMin, storageTypeMax);
}

AnyQuantizedType
AnyQuantizedType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                             unsigned flags, Type storageType,
                             Type expressedType, int64_t storageTypeMin,
                             int64_t storageTypeMax) {
  return Base::getChecked(emitError, storageType.getContext(), flags,
                          storageType, expressedType, storageTypeMin,
                          storageTypeMax);
}

LogicalResult
AnyQuantizedType::verifyInvariants(function_ref<InFlightDiagnostic()> emitError,
                                   unsigned flags, Type storageType,
                                   Type expressedType, int64_t storageTypeMin,
                                   int64_t storageTypeMax) {
  if (failed(QuantizedType::verifyInvariants(emitError, flags, storageType,
                                             expressedType, storageTypeMin,
                                             storageTypeMax))) {
    return failure();
  }

  // Verify that the expressed type is floating point.
  // If this restriction is ever eliminated, the parser/printer must be
  // extended.
  if (expressedType && !llvm::isa<FloatType>(expressedType))
    return emitError() << "expressed type must be floating point";

  return success();
}

UniformQuantizedType UniformQuantizedType::get(unsigned flags, Type storageType,
                                               Type expressedType, double scale,
                                               int64_t zeroPoint,
                                               int64_t storageTypeMin,
                                               int64_t storageTypeMax) {
  return Base::get(storageType.getContext(), flags, storageType, expressedType,
                   scale, zeroPoint, storageTypeMin, storageTypeMax);
}

UniformQuantizedType UniformQuantizedType::getChecked(
    function_ref<InFlightDiagnostic()> emitError, unsigned flags,
    Type storageType, Type expressedType, double scale, int64_t zeroPoint,
    int64_t storageTypeMin, int64_t storageTypeMax) {
  return Base::getChecked(emitError, storageType.getContext(), flags,
                          storageType, expressedType, scale, zeroPoint,
                          storageTypeMin, storageTypeMax);
}

LogicalResult UniformQuantizedType::verifyInvariants(
    function_ref<InFlightDiagnostic()> emitError, unsigned flags,
    Type storageType, Type expressedType, double scale, int64_t zeroPoint,
    int64_t storageTypeMin, int64_t storageTypeMax) {
  if (failed(QuantizedType::verifyInvariants(emitError, flags, storageType,
                                             expressedType, storageTypeMin,
                                             storageTypeMax))) {
    return failure();
  }

  // Uniform quantization requires fully expressed parameters, including
  // expressed type.
  if (!expressedType)
    return emitError() << "uniform quantization requires expressed type";

  // Verify that the expressed type is floating point.
  // If this restriction is ever eliminated, the parser/printer must be
  // extended.
  if (!llvm::isa<FloatType>(expressedType))
    return emitError() << "expressed type must be floating point";

  // Verify scale.
  double minScale = getMinScale(expressedType);
  double maxScale = getMaxScale(expressedType);
  if (scale < minScale || scale > maxScale)
    return emitError() << "scale out of expressed type range [" << minScale
                       << ", " << maxScale << "]";

  return success();
}

double UniformQuantizedType::getScale() const { return getImpl()->scale; }

int64_t UniformQuantizedType::getZeroPoint() const {
  return getImpl()->zeroPoint;
}

UniformQuantizedPerAxisType UniformQuantizedPerAxisType::get(
    unsigned flags, Type storageType, Type expressedType,
    ArrayRef<double> scales, ArrayRef<int64_t> zeroPoints,
    int32_t quantizedDimension, int64_t storageTypeMin,
    int64_t storageTypeMax) {
  return Base::get(storageType.getContext(), flags, storageType, expressedType,
                   scales, zeroPoints, quantizedDimension, storageTypeMin,
                   storageTypeMax);
}

UniformQuantizedPerAxisType UniformQuantizedPerAxisType::getChecked(
    function_ref<InFlightDiagnostic()> emitError, unsigned flags,
    Type storageType, Type expressedType, ArrayRef<double> scales,
    ArrayRef<int64_t> zeroPoints, int32_t quantizedDimension,
    int64_t storageTypeMin, int64_t storageTypeMax) {
  return Base::getChecked(emitError, storageType.getContext(), flags,
                          storageType, expressedType, scales, zeroPoints,
                          quantizedDimension, storageTypeMin, storageTypeMax);
}

LogicalResult UniformQuantizedPerAxisType::verifyInvariants(
    function_ref<InFlightDiagnostic()> emitError, unsigned flags,
    Type storageType, Type expressedType, ArrayRef<double> scales,
    ArrayRef<int64_t> zeroPoints, int32_t quantizedDimension,
    int64_t storageTypeMin, int64_t storageTypeMax) {
  if (failed(QuantizedType::verifyInvariants(emitError, flags, storageType,
                                             expressedType, storageTypeMin,
                                             storageTypeMax))) {
    return failure();
  }

  // Uniform quantization requires fully expressed parameters, including
  // expressed type.
  if (!expressedType)
    return emitError() << "uniform quantization requires expressed type";

  // Verify that the expressed type is floating point.
  // If this restriction is ever eliminated, the parser/printer must be
  // extended.
  if (!llvm::isa<FloatType>(expressedType))
    return emitError() << "expressed type must be floating point";

  // Ensure that the number of scales and zeroPoints match.
  if (scales.size() != zeroPoints.size())
    return emitError() << "illegal number of scales and zeroPoints: "
                       << scales.size() << ", " << zeroPoints.size();

  // Verify scale.
  double minScale = getMinScale(expressedType);
  double maxScale = getMaxScale(expressedType);
  for (double scale : scales) {
    if (scale < minScale || scale > maxScale)
      return emitError() << "scale out of expressed type range [" << minScale
                         << ", " << maxScale << "]";
  }

  // Verify quantized dimension.
  if (quantizedDimension < 0)
    return emitError() << "illegal quantized dimension: " << quantizedDimension;

  return success();
}

ArrayRef<double> UniformQuantizedPerAxisType::getScales() const {
  return getImpl()->getScales();
}

ArrayRef<int64_t> UniformQuantizedPerAxisType::getZeroPoints() const {
  return getImpl()->getZeroPoints();
}

int32_t UniformQuantizedPerAxisType::getQuantizedDimension() const {
  return getImpl()->quantizedDimension;
}

CalibratedQuantizedType CalibratedQuantizedType::get(Type expressedType,
                                                     double min, double max) {
  return Base::get(expressedType.getContext(), expressedType, min, max);
}

CalibratedQuantizedType CalibratedQuantizedType::getChecked(
    function_ref<InFlightDiagnostic()> emitError, Type expressedType,
    double min, double max) {
  return Base::getChecked(emitError, expressedType.getContext(), expressedType,
                          min, max);
}

LogicalResult CalibratedQuantizedType::verifyInvariants(
    function_ref<InFlightDiagnostic()> emitError, Type expressedType,
    double min, double max) {
  // Verify that the expressed type is floating point.
  // If this restriction is ever eliminated, the parser/printer must be
  // extended.
  if (!llvm::isa<FloatType>(expressedType))
    return emitError() << "expressed type must be floating point";
  if (max <= min)
    return emitError() << "illegal min and max: (" << min << ":" << max << ")";

  return success();
}

double CalibratedQuantizedType::getMin() const { return getImpl()->min; }

double CalibratedQuantizedType::getMax() const { return getImpl()->max; }
