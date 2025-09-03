//===- DialectQuant.cpp - 'quant' dialect submodule -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <vector>

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/Dialect/Quant.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;
using namespace llvm;
using namespace mlir;
using namespace mlir::python::nanobind_adaptors;

static void populateDialectQuantSubmodule(const nb::module_ &m) {
  //===-------------------------------------------------------------------===//
  // QuantizedType
  //===-------------------------------------------------------------------===//

  auto quantizedType =
      mlir_type_subclass(m, "QuantizedType", mlirTypeIsAQuantizedType);
  quantizedType.def_staticmethod(
      "default_minimum_for_integer",
      [](bool isSigned, unsigned integralWidth) {
        return mlirQuantizedTypeGetDefaultMinimumForInteger(isSigned,
                                                            integralWidth);
      },
      "Default minimum value for the integer with the specified signedness and "
      "bit width.",
      nb::arg("is_signed"), nb::arg("integral_width"));
  quantizedType.def_staticmethod(
      "default_maximum_for_integer",
      [](bool isSigned, unsigned integralWidth) {
        return mlirQuantizedTypeGetDefaultMaximumForInteger(isSigned,
                                                            integralWidth);
      },
      "Default maximum value for the integer with the specified signedness and "
      "bit width.",
      nb::arg("is_signed"), nb::arg("integral_width"));
  quantizedType.def_property_readonly(
      "expressed_type",
      [](MlirType type) { return mlirQuantizedTypeGetExpressedType(type); },
      "Type expressed by this quantized type.");
  quantizedType.def_property_readonly(
      "flags", [](MlirType type) { return mlirQuantizedTypeGetFlags(type); },
      "Flags of this quantized type (named accessors should be preferred to "
      "this)");
  quantizedType.def_property_readonly(
      "is_signed",
      [](MlirType type) { return mlirQuantizedTypeIsSigned(type); },
      "Signedness of this quantized type.");
  quantizedType.def_property_readonly(
      "storage_type",
      [](MlirType type) { return mlirQuantizedTypeGetStorageType(type); },
      "Storage type backing this quantized type.");
  quantizedType.def_property_readonly(
      "storage_type_min",
      [](MlirType type) { return mlirQuantizedTypeGetStorageTypeMin(type); },
      "The minimum value held by the storage type of this quantized type.");
  quantizedType.def_property_readonly(
      "storage_type_max",
      [](MlirType type) { return mlirQuantizedTypeGetStorageTypeMax(type); },
      "The maximum value held by the storage type of this quantized type.");
  quantizedType.def_property_readonly(
      "storage_type_integral_width",
      [](MlirType type) {
        return mlirQuantizedTypeGetStorageTypeIntegralWidth(type);
      },
      "The bitwidth of the storage type of this quantized type.");
  quantizedType.def(
      "is_compatible_expressed_type",
      [](MlirType type, MlirType candidate) {
        return mlirQuantizedTypeIsCompatibleExpressedType(type, candidate);
      },
      "Checks whether the candidate type can be expressed by this quantized "
      "type.",
      nb::arg("candidate"));
  quantizedType.def_property_readonly(
      "quantized_element_type",
      [](MlirType type) {
        return mlirQuantizedTypeGetQuantizedElementType(type);
      },
      "Element type of this quantized type expressed as quantized type.");
  quantizedType.def(
      "cast_from_storage_type",
      [](MlirType type, MlirType candidate) {
        MlirType castResult =
            mlirQuantizedTypeCastFromStorageType(type, candidate);
        if (!mlirTypeIsNull(castResult))
          return castResult;
        throw nb::type_error("Invalid cast.");
      },
      "Casts from a type based on the storage type of this quantized type to a "
      "corresponding type based on the quantized type. Raises TypeError if the "
      "cast is not valid.",
      nb::arg("candidate"));
  quantizedType.def_staticmethod(
      "cast_to_storage_type",
      [](MlirType type) {
        MlirType castResult = mlirQuantizedTypeCastToStorageType(type);
        if (!mlirTypeIsNull(castResult))
          return castResult;
        throw nb::type_error("Invalid cast.");
      },
      "Casts from a type based on a quantized type to a corresponding type "
      "based on the storage type of this quantized type. Raises TypeError if "
      "the cast is not valid.",
      nb::arg("type"));
  quantizedType.def(
      "cast_from_expressed_type",
      [](MlirType type, MlirType candidate) {
        MlirType castResult =
            mlirQuantizedTypeCastFromExpressedType(type, candidate);
        if (!mlirTypeIsNull(castResult))
          return castResult;
        throw nb::type_error("Invalid cast.");
      },
      "Casts from a type based on the expressed type of this quantized type to "
      "a corresponding type based on the quantized type. Raises TypeError if "
      "the cast is not valid.",
      nb::arg("candidate"));
  quantizedType.def_staticmethod(
      "cast_to_expressed_type",
      [](MlirType type) {
        MlirType castResult = mlirQuantizedTypeCastToExpressedType(type);
        if (!mlirTypeIsNull(castResult))
          return castResult;
        throw nb::type_error("Invalid cast.");
      },
      "Casts from a type based on a quantized type to a corresponding type "
      "based on the expressed type of this quantized type. Raises TypeError if "
      "the cast is not valid.",
      nb::arg("type"));
  quantizedType.def(
      "cast_expressed_to_storage_type",
      [](MlirType type, MlirType candidate) {
        MlirType castResult =
            mlirQuantizedTypeCastExpressedToStorageType(type, candidate);
        if (!mlirTypeIsNull(castResult))
          return castResult;
        throw nb::type_error("Invalid cast.");
      },
      "Casts from a type based on the expressed type of this quantized type to "
      "a corresponding type based on the storage type. Raises TypeError if the "
      "cast is not valid.",
      nb::arg("candidate"));

  quantizedType.get_class().attr("FLAG_SIGNED") =
      mlirQuantizedTypeGetSignedFlag();

  //===-------------------------------------------------------------------===//
  // AnyQuantizedType
  //===-------------------------------------------------------------------===//

  auto anyQuantizedType =
      mlir_type_subclass(m, "AnyQuantizedType", mlirTypeIsAAnyQuantizedType,
                         quantizedType.get_class());
  anyQuantizedType.def_classmethod(
      "get",
      [](const nb::object &cls, unsigned flags, MlirType storageType,
         MlirType expressedType, int64_t storageTypeMin,
         int64_t storageTypeMax) {
        return cls(mlirAnyQuantizedTypeGet(flags, storageType, expressedType,
                                           storageTypeMin, storageTypeMax));
      },
      "Gets an instance of AnyQuantizedType in the same context as the "
      "provided storage type.",
      nb::arg("cls"), nb::arg("flags"), nb::arg("storage_type"),
      nb::arg("expressed_type"), nb::arg("storage_type_min"),
      nb::arg("storage_type_max"));

  //===-------------------------------------------------------------------===//
  // UniformQuantizedType
  //===-------------------------------------------------------------------===//

  auto uniformQuantizedType = mlir_type_subclass(
      m, "UniformQuantizedType", mlirTypeIsAUniformQuantizedType,
      quantizedType.get_class());
  uniformQuantizedType.def_classmethod(
      "get",
      [](const nb::object &cls, unsigned flags, MlirType storageType,
         MlirType expressedType, double scale, int64_t zeroPoint,
         int64_t storageTypeMin, int64_t storageTypeMax) {
        return cls(mlirUniformQuantizedTypeGet(flags, storageType,
                                               expressedType, scale, zeroPoint,
                                               storageTypeMin, storageTypeMax));
      },
      "Gets an instance of UniformQuantizedType in the same context as the "
      "provided storage type.",
      nb::arg("cls"), nb::arg("flags"), nb::arg("storage_type"),
      nb::arg("expressed_type"), nb::arg("scale"), nb::arg("zero_point"),
      nb::arg("storage_type_min"), nb::arg("storage_type_max"));
  uniformQuantizedType.def_property_readonly(
      "scale",
      [](MlirType type) { return mlirUniformQuantizedTypeGetScale(type); },
      "The scale designates the difference between the real values "
      "corresponding to consecutive quantized values differing by 1.");
  uniformQuantizedType.def_property_readonly(
      "zero_point",
      [](MlirType type) { return mlirUniformQuantizedTypeGetZeroPoint(type); },
      "The storage value corresponding to the real value 0 in the affine "
      "equation.");
  uniformQuantizedType.def_property_readonly(
      "is_fixed_point",
      [](MlirType type) { return mlirUniformQuantizedTypeIsFixedPoint(type); },
      "Fixed point values are real numbers divided by a scale.");

  //===-------------------------------------------------------------------===//
  // UniformQuantizedPerAxisType
  //===-------------------------------------------------------------------===//
  auto uniformQuantizedPerAxisType = mlir_type_subclass(
      m, "UniformQuantizedPerAxisType", mlirTypeIsAUniformQuantizedPerAxisType,
      quantizedType.get_class());
  uniformQuantizedPerAxisType.def_classmethod(
      "get",
      [](const nb::object &cls, unsigned flags, MlirType storageType,
         MlirType expressedType, std::vector<double> scales,
         std::vector<int64_t> zeroPoints, int32_t quantizedDimension,
         int64_t storageTypeMin, int64_t storageTypeMax) {
        if (scales.size() != zeroPoints.size())
          throw nb::value_error(
              "Mismatching number of scales and zero points.");
        auto nDims = static_cast<intptr_t>(scales.size());
        return cls(mlirUniformQuantizedPerAxisTypeGet(
            flags, storageType, expressedType, nDims, scales.data(),
            zeroPoints.data(), quantizedDimension, storageTypeMin,
            storageTypeMax));
      },
      "Gets an instance of UniformQuantizedPerAxisType in the same context as "
      "the provided storage type.",
      nb::arg("cls"), nb::arg("flags"), nb::arg("storage_type"),
      nb::arg("expressed_type"), nb::arg("scales"), nb::arg("zero_points"),
      nb::arg("quantized_dimension"), nb::arg("storage_type_min"),
      nb::arg("storage_type_max"));
  uniformQuantizedPerAxisType.def_property_readonly(
      "scales",
      [](MlirType type) {
        intptr_t nDim = mlirUniformQuantizedPerAxisTypeGetNumDims(type);
        std::vector<double> scales;
        scales.reserve(nDim);
        for (intptr_t i = 0; i < nDim; ++i) {
          double scale = mlirUniformQuantizedPerAxisTypeGetScale(type, i);
          scales.push_back(scale);
        }
        return scales;
      },
      "The scales designate the difference between the real values "
      "corresponding to consecutive quantized values differing by 1. The ith "
      "scale corresponds to the ith slice in the quantized_dimension.");
  uniformQuantizedPerAxisType.def_property_readonly(
      "zero_points",
      [](MlirType type) {
        intptr_t nDim = mlirUniformQuantizedPerAxisTypeGetNumDims(type);
        std::vector<int64_t> zeroPoints;
        zeroPoints.reserve(nDim);
        for (intptr_t i = 0; i < nDim; ++i) {
          int64_t zeroPoint =
              mlirUniformQuantizedPerAxisTypeGetZeroPoint(type, i);
          zeroPoints.push_back(zeroPoint);
        }
        return zeroPoints;
      },
      "the storage values corresponding to the real value 0 in the affine "
      "equation. The ith zero point corresponds to the ith slice in the "
      "quantized_dimension.");
  uniformQuantizedPerAxisType.def_property_readonly(
      "quantized_dimension",
      [](MlirType type) {
        return mlirUniformQuantizedPerAxisTypeGetQuantizedDimension(type);
      },
      "Specifies the dimension of the shape that the scales and zero points "
      "correspond to.");
  uniformQuantizedPerAxisType.def_property_readonly(
      "is_fixed_point",
      [](MlirType type) {
        return mlirUniformQuantizedPerAxisTypeIsFixedPoint(type);
      },
      "Fixed point values are real numbers divided by a scale.");

  //===-------------------------------------------------------------------===//
  // UniformQuantizedSubChannelType
  //===-------------------------------------------------------------------===//
  auto uniformQuantizedSubChannelType = mlir_type_subclass(
      m, "UniformQuantizedSubChannelType",
      mlirTypeIsAUniformQuantizedSubChannelType, quantizedType.get_class());
  uniformQuantizedSubChannelType.def_classmethod(
      "get",
      [](const nb::object &cls, unsigned flags, MlirType storageType,
         MlirType expressedType, MlirAttribute scales, MlirAttribute zeroPoints,
         std::vector<int32_t> quantizedDimensions,
         std::vector<int64_t> blockSizes, int64_t storageTypeMin,
         int64_t storageTypeMax) {
        return cls(mlirUniformQuantizedSubChannelTypeGet(
            flags, storageType, expressedType, scales, zeroPoints,
            static_cast<intptr_t>(blockSizes.size()),
            quantizedDimensions.data(), blockSizes.data(), storageTypeMin,
            storageTypeMax));
      },
      "Gets an instance of UniformQuantizedSubChannel in the same context as "
      "the provided storage type.",
      nb::arg("cls"), nb::arg("flags"), nb::arg("storage_type"),
      nb::arg("expressed_type"), nb::arg("scales"), nb::arg("zero_points"),
      nb::arg("quantized_dimensions"), nb::arg("block_sizes"),
      nb::arg("storage_type_min"), nb::arg("storage_type_max"));
  uniformQuantizedSubChannelType.def_property_readonly(
      "quantized_dimensions",
      [](MlirType type) {
        intptr_t nDim =
            mlirUniformQuantizedSubChannelTypeGetNumBlockSizes(type);
        std::vector<int32_t> quantizedDimensions;
        quantizedDimensions.reserve(nDim);
        for (intptr_t i = 0; i < nDim; ++i) {
          quantizedDimensions.push_back(
              mlirUniformQuantizedSubChannelTypeGetQuantizedDimension(type, i));
        }
        return quantizedDimensions;
      },
      "Gets the quantized dimensions. Each element in the returned list "
      "represents an axis of the quantized data tensor that has a specified "
      "block size. The order of elements corresponds to the order of block "
      "sizes returned by 'block_sizes' method. It means that the data tensor "
      "is quantized along the i-th dimension in the returned list using the "
      "i-th block size from block_sizes method.");
  uniformQuantizedSubChannelType.def_property_readonly(
      "block_sizes",
      [](MlirType type) {
        intptr_t nDim =
            mlirUniformQuantizedSubChannelTypeGetNumBlockSizes(type);
        std::vector<int64_t> blockSizes;
        blockSizes.reserve(nDim);
        for (intptr_t i = 0; i < nDim; ++i) {
          blockSizes.push_back(
              mlirUniformQuantizedSubChannelTypeGetBlockSize(type, i));
        }
        return blockSizes;
      },
      "Gets the block sizes for the quantized dimensions. The i-th element in "
      "the returned list corresponds to the block size for the i-th dimension "
      "in the list returned by quantized_dimensions method.");
  uniformQuantizedSubChannelType.def_property_readonly(
      "scales",
      [](MlirType type) -> MlirAttribute {
        return mlirUniformQuantizedSubChannelTypeGetScales(type);
      },
      "The scales of the quantized type.");
  uniformQuantizedSubChannelType.def_property_readonly(
      "zero_points",
      [](MlirType type) -> MlirAttribute {
        return mlirUniformQuantizedSubChannelTypeGetZeroPoints(type);
      },
      "The zero points of the quantized type.");

  //===-------------------------------------------------------------------===//
  // CalibratedQuantizedType
  //===-------------------------------------------------------------------===//

  auto calibratedQuantizedType = mlir_type_subclass(
      m, "CalibratedQuantizedType", mlirTypeIsACalibratedQuantizedType,
      quantizedType.get_class());
  calibratedQuantizedType.def_classmethod(
      "get",
      [](const nb::object &cls, MlirType expressedType, double min,
         double max) {
        return cls(mlirCalibratedQuantizedTypeGet(expressedType, min, max));
      },
      "Gets an instance of CalibratedQuantizedType in the same context as the "
      "provided expressed type.",
      nb::arg("cls"), nb::arg("expressed_type"), nb::arg("min"),
      nb::arg("max"));
  calibratedQuantizedType.def_property_readonly("min", [](MlirType type) {
    return mlirCalibratedQuantizedTypeGetMin(type);
  });
  calibratedQuantizedType.def_property_readonly("max", [](MlirType type) {
    return mlirCalibratedQuantizedTypeGetMax(type);
  });
}

NB_MODULE(_mlirDialectsQuant, m) {
  m.doc() = "MLIR Quantization dialect";

  populateDialectQuantSubmodule(m);
}
