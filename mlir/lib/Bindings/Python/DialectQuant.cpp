//===- DialectQuant.cpp - 'quant' dialect submodule -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <vector>

#include "mlir-c/Dialect/Quant.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

#include <mlir/Bindings/Python/IRAttributes.h>

namespace nb = nanobind;
using namespace llvm;
using namespace mlir::python::nanobind_adaptors;

namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {
namespace quant {
//===-------------------------------------------------------------------===//
// QuantizedType
//===-------------------------------------------------------------------===//

struct QuantizedType : PyConcreteType<QuantizedType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAQuantizedType;
  static constexpr const char *pyClassName = "QuantizedType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "default_minimum_for_integer",
        [](bool isSigned, unsigned integralWidth) {
          return mlirQuantizedTypeGetDefaultMinimumForInteger(isSigned,
                                                              integralWidth);
        },
        "Default minimum value for the integer with the specified signedness "
        "and "
        "bit width.",
        nb::arg("is_signed"), nb::arg("integral_width"));
    c.def_static(
        "default_maximum_for_integer",
        [](bool isSigned, unsigned integralWidth) {
          return mlirQuantizedTypeGetDefaultMaximumForInteger(isSigned,
                                                              integralWidth);
        },
        "Default maximum value for the integer with the specified signedness "
        "and "
        "bit width.",
        nb::arg("is_signed"), nb::arg("integral_width"));
    c.def_prop_ro(
        "expressed_type",
        [](QuantizedType &type) {
          return PyType(type.getContext(),
                        mlirQuantizedTypeGetExpressedType(type))
              .maybeDownCast();
        },
        "Type expressed by this quantized type.");
    c.def_prop_ro(
        "flags",
        [](const QuantizedType &type) {
          return mlirQuantizedTypeGetFlags(type);
        },
        "Flags of this quantized type (named accessors should be preferred to "
        "this)");
    c.def_prop_ro(
        "is_signed",
        [](const QuantizedType &type) {
          return mlirQuantizedTypeIsSigned(type);
        },
        "Signedness of this quantized type.");
    c.def_prop_ro(
        "storage_type",
        [](QuantizedType &type) {
          return PyType(type.getContext(),
                        mlirQuantizedTypeGetStorageType(type))
              .maybeDownCast();
        },
        "Storage type backing this quantized type.");
    c.def_prop_ro(
        "storage_type_min",
        [](const QuantizedType &type) {
          return mlirQuantizedTypeGetStorageTypeMin(type);
        },
        "The minimum value held by the storage type of this quantized type.");
    c.def_prop_ro(
        "storage_type_max",
        [](const QuantizedType &type) {
          return mlirQuantizedTypeGetStorageTypeMax(type);
        },
        "The maximum value held by the storage type of this quantized type.");
    c.def_prop_ro(
        "storage_type_integral_width",
        [](const QuantizedType &type) {
          return mlirQuantizedTypeGetStorageTypeIntegralWidth(type);
        },
        "The bitwidth of the storage type of this quantized type.");
    c.def(
        "is_compatible_expressed_type",
        [](const QuantizedType &type, const PyType &candidate) {
          return mlirQuantizedTypeIsCompatibleExpressedType(type, candidate);
        },
        "Checks whether the candidate type can be expressed by this quantized "
        "type.",
        nb::arg("candidate"));
    c.def_prop_ro(
        "quantized_element_type",
        [](QuantizedType &type) {
          return PyType(type.getContext(),
                        mlirQuantizedTypeGetQuantizedElementType(type))
              .maybeDownCast();
        },
        "Element type of this quantized type expressed as quantized type.");
    c.def(
        "cast_from_storage_type",
        [](QuantizedType &type, const PyType &candidate) {
          MlirType castResult =
              mlirQuantizedTypeCastFromStorageType(type, candidate);
          if (!mlirTypeIsNull(castResult))
            return QuantizedType(type.getContext(), castResult);
          throw nb::type_error("Invalid cast.");
        },
        "Casts from a type based on the storage type of this quantized type to "
        "a "
        "corresponding type based on the quantized type. Raises TypeError if "
        "the "
        "cast is not valid.",
        nb::arg("candidate"));
    c.def_static(
        "cast_to_storage_type",
        [](PyType &type) {
          MlirType castResult = mlirQuantizedTypeCastToStorageType(type);
          if (!mlirTypeIsNull(castResult))
            return PyType(type.getContext(), castResult).maybeDownCast();
          throw nb::type_error("Invalid cast.");
        },
        "Casts from a type based on a quantized type to a corresponding type "
        "based on the storage type of this quantized type. Raises TypeError if "
        "the cast is not valid.",
        nb::arg("type"));
    c.def(
        "cast_from_expressed_type",
        [](QuantizedType &type, const PyType &candidate) {
          MlirType castResult =
              mlirQuantizedTypeCastFromExpressedType(type, candidate);
          if (!mlirTypeIsNull(castResult))
            return PyType(type.getContext(), castResult).maybeDownCast();
          throw nb::type_error("Invalid cast.");
        },
        "Casts from a type based on the expressed type of this quantized type "
        "to "
        "a corresponding type based on the quantized type. Raises TypeError if "
        "the cast is not valid.",
        nb::arg("candidate"));
    c.def_static(
        "cast_to_expressed_type",
        [](PyType &type) {
          MlirType castResult = mlirQuantizedTypeCastToExpressedType(type);
          if (!mlirTypeIsNull(castResult))
            return PyType(type.getContext(), castResult).maybeDownCast();
          throw nb::type_error("Invalid cast.");
        },
        "Casts from a type based on a quantized type to a corresponding type "
        "based on the expressed type of this quantized type. Raises TypeError "
        "if "
        "the cast is not valid.",
        nb::arg("type"));
    c.def(
        "cast_expressed_to_storage_type",
        [](QuantizedType &type, const PyType &candidate) {
          MlirType castResult =
              mlirQuantizedTypeCastExpressedToStorageType(type, candidate);
          if (!mlirTypeIsNull(castResult))
            return PyType(type.getContext(), castResult).maybeDownCast();
          throw nb::type_error("Invalid cast.");
        },
        "Casts from a type based on the expressed type of this quantized type "
        "to "
        "a corresponding type based on the storage type. Raises TypeError if "
        "the "
        "cast is not valid.",
        nb::arg("candidate"));
  }
};

//===-------------------------------------------------------------------===//
// AnyQuantizedType
//===-------------------------------------------------------------------===//

struct AnyQuantizedType : PyConcreteType<AnyQuantizedType, QuantizedType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAAnyQuantizedType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirAnyQuantizedTypeGetTypeID;
  static constexpr const char *pyClassName = "AnyQuantizedType";
  static inline const MlirStringRef name = mlirAnyQuantizedTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](unsigned flags, const PyType &storageType,
           const PyType &expressedType, int64_t storageTypeMin,
           int64_t storageTypeMax, DefaultingPyMlirContext context) {
          return AnyQuantizedType(
              context->getRef(),
              mlirAnyQuantizedTypeGet(flags, storageType, expressedType,
                                      storageTypeMin, storageTypeMax));
        },
        "Gets an instance of AnyQuantizedType in the same context as the "
        "provided storage type.",
        nb::arg("flags"), nb::arg("storage_type"), nb::arg("expressed_type"),
        nb::arg("storage_type_min"), nb::arg("storage_type_max"),
        nb::arg("context") = nb::none());
  }
};

//===-------------------------------------------------------------------===//
// UniformQuantizedType
//===-------------------------------------------------------------------===//

struct UniformQuantizedType
    : PyConcreteType<UniformQuantizedType, QuantizedType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAUniformQuantizedType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirUniformQuantizedTypeGetTypeID;
  static constexpr const char *pyClassName = "UniformQuantizedType";
  static inline const MlirStringRef name = mlirUniformQuantizedTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](unsigned flags, const PyType &storageType,
           const PyType &expressedType, double scale, int64_t zeroPoint,
           int64_t storageTypeMin, int64_t storageTypeMax,
           DefaultingPyMlirContext context) {
          return UniformQuantizedType(
              context->getRef(),
              mlirUniformQuantizedTypeGet(flags, storageType, expressedType,
                                          scale, zeroPoint, storageTypeMin,
                                          storageTypeMax));
        },
        "Gets an instance of UniformQuantizedType in the same context as the "
        "provided storage type.",
        nb::arg("flags"), nb::arg("storage_type"), nb::arg("expressed_type"),
        nb::arg("scale"), nb::arg("zero_point"), nb::arg("storage_type_min"),
        nb::arg("storage_type_max"), nb::arg("context") = nb::none());
    c.def_prop_ro(
        "scale",
        [](const UniformQuantizedType &type) {
          return mlirUniformQuantizedTypeGetScale(type);
        },
        "The scale designates the difference between the real values "
        "corresponding to consecutive quantized values differing by 1.");
    c.def_prop_ro(
        "zero_point",
        [](const UniformQuantizedType &type) {
          return mlirUniformQuantizedTypeGetZeroPoint(type);
        },
        "The storage value corresponding to the real value 0 in the affine "
        "equation.");
    c.def_prop_ro(
        "is_fixed_point",
        [](const UniformQuantizedType &type) {
          return mlirUniformQuantizedTypeIsFixedPoint(type);
        },
        "Fixed point values are real numbers divided by a scale.");
  }
};

//===-------------------------------------------------------------------===//
// UniformQuantizedPerAxisType
//===-------------------------------------------------------------------===//

struct UniformQuantizedPerAxisType
    : PyConcreteType<UniformQuantizedPerAxisType, QuantizedType> {
  static constexpr IsAFunctionTy isaFunction =
      mlirTypeIsAUniformQuantizedPerAxisType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirUniformQuantizedPerAxisTypeGetTypeID;
  static constexpr const char *pyClassName = "UniformQuantizedPerAxisType";
  static inline const MlirStringRef name =
      mlirUniformQuantizedPerAxisTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](unsigned flags, const PyType &storageType,
           const PyType &expressedType, std::vector<double> scales,
           std::vector<int64_t> zeroPoints, int32_t quantizedDimension,
           int64_t storageTypeMin, int64_t storageTypeMax,
           DefaultingPyMlirContext context) {
          if (scales.size() != zeroPoints.size())
            throw nb::value_error(
                "Mismatching number of scales and zero points.");
          auto nDims = static_cast<intptr_t>(scales.size());
          return UniformQuantizedPerAxisType(
              context->getRef(),
              mlirUniformQuantizedPerAxisTypeGet(
                  flags, storageType, expressedType, nDims, scales.data(),
                  zeroPoints.data(), quantizedDimension, storageTypeMin,
                  storageTypeMax));
        },
        "Gets an instance of UniformQuantizedPerAxisType in the same context "
        "as "
        "the provided storage type.",
        nb::arg("flags"), nb::arg("storage_type"), nb::arg("expressed_type"),
        nb::arg("scales"), nb::arg("zero_points"),
        nb::arg("quantized_dimension"), nb::arg("storage_type_min"),
        nb::arg("storage_type_max"), nb::arg("context") = nb::none());
    c.def_prop_ro(
        "scales",
        [](const UniformQuantizedPerAxisType &type) {
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
    c.def_prop_ro(
        "zero_points",
        [](const UniformQuantizedPerAxisType &type) {
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
    c.def_prop_ro(
        "quantized_dimension",
        [](const UniformQuantizedPerAxisType &type) {
          return mlirUniformQuantizedPerAxisTypeGetQuantizedDimension(type);
        },
        "Specifies the dimension of the shape that the scales and zero points "
        "correspond to.");
    c.def_prop_ro(
        "is_fixed_point",
        [](const UniformQuantizedPerAxisType &type) {
          return mlirUniformQuantizedPerAxisTypeIsFixedPoint(type);
        },
        "Fixed point values are real numbers divided by a scale.");
  }
};

//===-------------------------------------------------------------------===//
// UniformQuantizedSubChannelType
//===-------------------------------------------------------------------===//

struct UniformQuantizedSubChannelType
    : PyConcreteType<UniformQuantizedSubChannelType, QuantizedType> {
  static constexpr IsAFunctionTy isaFunction =
      mlirTypeIsAUniformQuantizedSubChannelType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirUniformQuantizedSubChannelTypeGetTypeID;
  static constexpr const char *pyClassName = "UniformQuantizedSubChannelType";
  static inline const MlirStringRef name =
      mlirUniformQuantizedSubChannelTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](unsigned flags, const PyType &storageType,
           const PyType &expressedType, PyAttribute scales,
           PyAttribute zeroPoints, std::vector<int32_t> quantizedDimensions,
           std::vector<int64_t> blockSizes, int64_t storageTypeMin,
           int64_t storageTypeMax, DefaultingPyMlirContext context) {
          return UniformQuantizedSubChannelType(
              context->getRef(),
              mlirUniformQuantizedSubChannelTypeGet(
                  flags, storageType, expressedType, scales, zeroPoints,
                  static_cast<intptr_t>(blockSizes.size()),
                  quantizedDimensions.data(), blockSizes.data(), storageTypeMin,
                  storageTypeMax));
        },
        "Gets an instance of UniformQuantizedSubChannel in the same context as "
        "the provided storage type.",
        nb::arg("flags"), nb::arg("storage_type"), nb::arg("expressed_type"),
        nb::arg("scales"), nb::arg("zero_points"),
        nb::arg("quantized_dimensions"), nb::arg("block_sizes"),
        nb::arg("storage_type_min"), nb::arg("storage_type_max"),
        nb::arg("context") = nb::none());
    c.def_prop_ro(
        "quantized_dimensions",
        [](const UniformQuantizedSubChannelType &type) {
          intptr_t nDim =
              mlirUniformQuantizedSubChannelTypeGetNumBlockSizes(type);
          std::vector<int32_t> quantizedDimensions;
          quantizedDimensions.reserve(nDim);
          for (intptr_t i = 0; i < nDim; ++i) {
            quantizedDimensions.push_back(
                mlirUniformQuantizedSubChannelTypeGetQuantizedDimension(type,
                                                                        i));
          }
          return quantizedDimensions;
        },
        "Gets the quantized dimensions. Each element in the returned list "
        "represents an axis of the quantized data tensor that has a specified "
        "block size. The order of elements corresponds to the order of block "
        "sizes returned by 'block_sizes' method. It means that the data tensor "
        "is quantized along the i-th dimension in the returned list using the "
        "i-th block size from block_sizes method.");
    c.def_prop_ro(
        "block_sizes",
        [](const UniformQuantizedSubChannelType &type) {
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
        "Gets the block sizes for the quantized dimensions. The i-th element "
        "in "
        "the returned list corresponds to the block size for the i-th "
        "dimension "
        "in the list returned by quantized_dimensions method.");
    c.def_prop_ro(
        "scales",
        [](UniformQuantizedSubChannelType &type) {
          return PyDenseElementsAttribute(
              type.getContext(),
              mlirUniformQuantizedSubChannelTypeGetScales(type));
        },
        "The scales of the quantized type.");
    c.def_prop_ro(
        "zero_points",
        [](UniformQuantizedSubChannelType &type) {
          return PyDenseElementsAttribute(
              type.getContext(),
              mlirUniformQuantizedSubChannelTypeGetZeroPoints(type));
        },
        "The zero points of the quantized type.");
  }
};

//===-------------------------------------------------------------------===//
// CalibratedQuantizedType
//===-------------------------------------------------------------------===//

struct CalibratedQuantizedType
    : PyConcreteType<CalibratedQuantizedType, QuantizedType> {
  static constexpr IsAFunctionTy isaFunction =
      mlirTypeIsACalibratedQuantizedType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirCalibratedQuantizedTypeGetTypeID;
  static constexpr const char *pyClassName = "CalibratedQuantizedType";
  static inline const MlirStringRef name = mlirCalibratedQuantizedTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const PyType &expressedType, double min, double max,
           DefaultingPyMlirContext context) {
          return CalibratedQuantizedType(
              context->getRef(),
              mlirCalibratedQuantizedTypeGet(expressedType, min, max));
        },
        "Gets an instance of CalibratedQuantizedType in the same context as "
        "the "
        "provided expressed type.",
        nb::arg("expressed_type"), nb::arg("min"), nb::arg("max"),
        nb::arg("context") = nb::none());
    c.def_prop_ro("min", [](const PyType &type) {
      return mlirCalibratedQuantizedTypeGetMin(type);
    });
    c.def_prop_ro("max", [](const PyType &type) {
      return mlirCalibratedQuantizedTypeGetMax(type);
    });
  }
};

static void populateDialectQuantSubmodule(nb::module_ &m) {
  QuantizedType::bind(m);

  // Set the FLAG_SIGNED class attribute after binding QuantizedType
  auto quantizedTypeClass = m.attr("QuantizedType");
  quantizedTypeClass.attr("FLAG_SIGNED") = mlirQuantizedTypeGetSignedFlag();

  AnyQuantizedType::bind(m);
  UniformQuantizedType::bind(m);
  UniformQuantizedPerAxisType::bind(m);
  UniformQuantizedSubChannelType::bind(m);
  CalibratedQuantizedType::bind(m);
}
} // namespace quant
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

NB_MODULE(_mlirDialectsQuant, m) {
  m.doc() = "MLIR Quantization dialect";

  mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::quant::
      populateDialectQuantSubmodule(m);
}
