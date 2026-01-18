//===- IRTypes.cpp - Exports builtin and standard types -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// clang-format off
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/IRTypes.h"
// clang-format on

#include <optional>

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/NanobindUtils.h"

namespace nb = nanobind;
using namespace mlir;
using namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN;

using llvm::SmallVector;
using llvm::Twine;

namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {

int mlirTypeIsAIntegerOrFloat(MlirType type) {
  return mlirTypeIsAInteger(type) || mlirTypeIsABF16(type) ||
         mlirTypeIsAF16(type) || mlirTypeIsAF32(type) || mlirTypeIsAF64(type);
}

void PyIntegerType::bindDerived(ClassTy &c) {
  nb::enum_<Signedness>(c, "Signedness")
      .value("SIGNLESS", Signless)
      .value("SIGNED", Signed)
      .value("UNSIGNED", Unsigned)
      .export_values();

  c.def_static(
      "get_signless",
      [](unsigned width, DefaultingPyMlirContext context) {
        MlirType t = mlirIntegerTypeGet(context->get(), width);
        return PyIntegerType(context->getRef(), t);
      },
      nb::arg("width"), nb::arg("context") = nb::none(),
      "Create a signless integer type");
  c.def_static(
      "get_signed",
      [](unsigned width, DefaultingPyMlirContext context) {
        MlirType t = mlirIntegerTypeSignedGet(context->get(), width);
        return PyIntegerType(context->getRef(), t);
      },
      nb::arg("width"), nb::arg("context") = nb::none(),
      "Create a signed integer type");
  c.def_static(
      "get_unsigned",
      [](unsigned width, DefaultingPyMlirContext context) {
        MlirType t = mlirIntegerTypeUnsignedGet(context->get(), width);
        return PyIntegerType(context->getRef(), t);
      },
      nb::arg("width"), nb::arg("context") = nb::none(),
      "Create an unsigned integer type");
  c.def_static(
      "get",
      [](unsigned width, Signedness signedness,
         DefaultingPyMlirContext context) {
        MlirType t;
        switch (signedness) {
        case Signless:
          t = mlirIntegerTypeGet(context->get(), width);
          break;
        case Signed:
          t = mlirIntegerTypeSignedGet(context->get(), width);
          break;
        case Unsigned:
          t = mlirIntegerTypeUnsignedGet(context->get(), width);
          break;
        }
        return PyIntegerType(context->getRef(), t);
      },
      nb::arg("width"), nb::arg("signedness") = Signless,
      nb::arg("context") = nb::none(), "Create an integer type");
  c.def_prop_ro("signedness", [](PyIntegerType &self) -> Signedness {
    if (mlirIntegerTypeIsSignless(self))
      return Signless;
    if (mlirIntegerTypeIsSigned(self))
      return Signed;
    return Unsigned;
  });
  c.def_prop_ro(
      "width",
      [](PyIntegerType &self) { return mlirIntegerTypeGetWidth(self); },
      "Returns the width of the integer type");
  c.def_prop_ro(
      "is_signless",
      [](PyIntegerType &self) -> bool {
        return mlirIntegerTypeIsSignless(self);
      },
      "Returns whether this is a signless integer");
  c.def_prop_ro(
      "is_signed",
      [](PyIntegerType &self) -> bool { return mlirIntegerTypeIsSigned(self); },
      "Returns whether this is a signed integer");
  c.def_prop_ro(
      "is_unsigned",
      [](PyIntegerType &self) -> bool {
        return mlirIntegerTypeIsUnsigned(self);
      },
      "Returns whether this is an unsigned integer");
}

void PyIndexType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](DefaultingPyMlirContext context) {
        MlirType t = mlirIndexTypeGet(context->get());
        return PyIndexType(context->getRef(), t);
      },
      nb::arg("context") = nb::none(), "Create a index type.");
}

void PyFloatType::bindDerived(ClassTy &c) {
  c.def_prop_ro(
      "width", [](PyFloatType &self) { return mlirFloatTypeGetWidth(self); },
      "Returns the width of the floating-point type");
}

void PyFloat4E2M1FNType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](DefaultingPyMlirContext context) {
        MlirType t = mlirFloat4E2M1FNTypeGet(context->get());
        return PyFloat4E2M1FNType(context->getRef(), t);
      },
      nb::arg("context") = nb::none(), "Create a float4_e2m1fn type.");
}

void PyFloat6E2M3FNType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](DefaultingPyMlirContext context) {
        MlirType t = mlirFloat6E2M3FNTypeGet(context->get());
        return PyFloat6E2M3FNType(context->getRef(), t);
      },
      nb::arg("context") = nb::none(), "Create a float6_e2m3fn type.");
}

void PyFloat6E3M2FNType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](DefaultingPyMlirContext context) {
        MlirType t = mlirFloat6E3M2FNTypeGet(context->get());
        return PyFloat6E3M2FNType(context->getRef(), t);
      },
      nb::arg("context") = nb::none(), "Create a float6_e3m2fn type.");
}

void PyFloat8E4M3FNType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](DefaultingPyMlirContext context) {
        MlirType t = mlirFloat8E4M3FNTypeGet(context->get());
        return PyFloat8E4M3FNType(context->getRef(), t);
      },
      nb::arg("context") = nb::none(), "Create a float8_e4m3fn type.");
}

void PyFloat8E5M2Type::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](DefaultingPyMlirContext context) {
        MlirType t = mlirFloat8E5M2TypeGet(context->get());
        return PyFloat8E5M2Type(context->getRef(), t);
      },
      nb::arg("context") = nb::none(), "Create a float8_e5m2 type.");
}

void PyFloat8E4M3Type::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](DefaultingPyMlirContext context) {
        MlirType t = mlirFloat8E4M3TypeGet(context->get());
        return PyFloat8E4M3Type(context->getRef(), t);
      },
      nb::arg("context") = nb::none(), "Create a float8_e4m3 type.");
}

void PyFloat8E4M3FNUZType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](DefaultingPyMlirContext context) {
        MlirType t = mlirFloat8E4M3FNUZTypeGet(context->get());
        return PyFloat8E4M3FNUZType(context->getRef(), t);
      },
      nb::arg("context") = nb::none(), "Create a float8_e4m3fnuz type.");
}

void PyFloat8E4M3B11FNUZType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](DefaultingPyMlirContext context) {
        MlirType t = mlirFloat8E4M3B11FNUZTypeGet(context->get());
        return PyFloat8E4M3B11FNUZType(context->getRef(), t);
      },
      nb::arg("context") = nb::none(), "Create a float8_e4m3b11fnuz type.");
}

void PyFloat8E5M2FNUZType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](DefaultingPyMlirContext context) {
        MlirType t = mlirFloat8E5M2FNUZTypeGet(context->get());
        return PyFloat8E5M2FNUZType(context->getRef(), t);
      },
      nb::arg("context") = nb::none(), "Create a float8_e5m2fnuz type.");
}

void PyFloat8E3M4Type::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](DefaultingPyMlirContext context) {
        MlirType t = mlirFloat8E3M4TypeGet(context->get());
        return PyFloat8E3M4Type(context->getRef(), t);
      },
      nb::arg("context") = nb::none(), "Create a float8_e3m4 type.");
}

void PyFloat8E8M0FNUType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](DefaultingPyMlirContext context) {
        MlirType t = mlirFloat8E8M0FNUTypeGet(context->get());
        return PyFloat8E8M0FNUType(context->getRef(), t);
      },
      nb::arg("context") = nb::none(), "Create a float8_e8m0fnu type.");
}

void PyBF16Type::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](DefaultingPyMlirContext context) {
        MlirType t = mlirBF16TypeGet(context->get());
        return PyBF16Type(context->getRef(), t);
      },
      nb::arg("context") = nb::none(), "Create a bf16 type.");
}

void PyF16Type::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](DefaultingPyMlirContext context) {
        MlirType t = mlirF16TypeGet(context->get());
        return PyF16Type(context->getRef(), t);
      },
      nb::arg("context") = nb::none(), "Create a f16 type.");
}

void PyTF32Type::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](DefaultingPyMlirContext context) {
        MlirType t = mlirTF32TypeGet(context->get());
        return PyTF32Type(context->getRef(), t);
      },
      nb::arg("context") = nb::none(), "Create a tf32 type.");
}

void PyF32Type::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](DefaultingPyMlirContext context) {
        MlirType t = mlirF32TypeGet(context->get());
        return PyF32Type(context->getRef(), t);
      },
      nb::arg("context") = nb::none(), "Create a f32 type.");
}

void PyF64Type::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](DefaultingPyMlirContext context) {
        MlirType t = mlirF64TypeGet(context->get());
        return PyF64Type(context->getRef(), t);
      },
      nb::arg("context") = nb::none(), "Create a f64 type.");
}

void PyNoneType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](DefaultingPyMlirContext context) {
        MlirType t = mlirNoneTypeGet(context->get());
        return PyNoneType(context->getRef(), t);
      },
      nb::arg("context") = nb::none(), "Create a none type.");
}

void PyComplexType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](PyType &elementType) {
        // The element must be a floating point or integer scalar type.
        if (mlirTypeIsAIntegerOrFloat(elementType)) {
          MlirType t = mlirComplexTypeGet(elementType);
          return PyComplexType(elementType.getContext(), t);
        }
        throw nb::value_error(
            (Twine("invalid '") +
             nb::cast<std::string>(nb::repr(nb::cast(elementType))) +
             "' and expected floating point or integer type.")
                .str()
                .c_str());
      },
      "Create a complex type");
  c.def_prop_ro(
      "element_type",
      [](PyComplexType &self) -> nb::typed<nb::object, PyType> {
        return PyType(self.getContext(), mlirComplexTypeGetElementType(self))
            .maybeDownCast();
      },
      "Returns element type.");
}

// Shaped Type Interface - ShapedType
void PyShapedType::bindDerived(ClassTy &c) {
  c.def_prop_ro(
      "element_type",
      [](PyShapedType &self) -> nb::typed<nb::object, PyType> {
        return PyType(self.getContext(), mlirShapedTypeGetElementType(self))
            .maybeDownCast();
      },
      "Returns the element type of the shaped type.");
  c.def_prop_ro(
      "has_rank",
      [](PyShapedType &self) -> bool { return mlirShapedTypeHasRank(self); },
      "Returns whether the given shaped type is ranked.");
  c.def_prop_ro(
      "rank",
      [](PyShapedType &self) {
        self.requireHasRank();
        return mlirShapedTypeGetRank(self);
      },
      "Returns the rank of the given ranked shaped type.");
  c.def_prop_ro(
      "has_static_shape",
      [](PyShapedType &self) -> bool {
        return mlirShapedTypeHasStaticShape(self);
      },
      "Returns whether the given shaped type has a static shape.");
  c.def(
      "is_dynamic_dim",
      [](PyShapedType &self, intptr_t dim) -> bool {
        self.requireHasRank();
        return mlirShapedTypeIsDynamicDim(self, dim);
      },
      nb::arg("dim"),
      "Returns whether the dim-th dimension of the given shaped type is "
      "dynamic.");
  c.def(
      "is_static_dim",
      [](PyShapedType &self, intptr_t dim) -> bool {
        self.requireHasRank();
        return mlirShapedTypeIsStaticDim(self, dim);
      },
      nb::arg("dim"),
      "Returns whether the dim-th dimension of the given shaped type is "
      "static.");
  c.def(
      "get_dim_size",
      [](PyShapedType &self, intptr_t dim) {
        self.requireHasRank();
        return mlirShapedTypeGetDimSize(self, dim);
      },
      nb::arg("dim"),
      "Returns the dim-th dimension of the given ranked shaped type.");
  c.def_static(
      "is_dynamic_size",
      [](int64_t size) -> bool { return mlirShapedTypeIsDynamicSize(size); },
      nb::arg("dim_size"),
      "Returns whether the given dimension size indicates a dynamic "
      "dimension.");
  c.def_static(
      "is_static_size",
      [](int64_t size) -> bool { return mlirShapedTypeIsStaticSize(size); },
      nb::arg("dim_size"),
      "Returns whether the given dimension size indicates a static "
      "dimension.");
  c.def(
      "is_dynamic_stride_or_offset",
      [](PyShapedType &self, int64_t val) -> bool {
        self.requireHasRank();
        return mlirShapedTypeIsDynamicStrideOrOffset(val);
      },
      nb::arg("dim_size"),
      "Returns whether the given value is used as a placeholder for dynamic "
      "strides and offsets in shaped types.");
  c.def(
      "is_static_stride_or_offset",
      [](PyShapedType &self, int64_t val) -> bool {
        self.requireHasRank();
        return mlirShapedTypeIsStaticStrideOrOffset(val);
      },
      nb::arg("dim_size"),
      "Returns whether the given shaped type stride or offset value is "
      "statically-sized.");
  c.def_prop_ro(
      "shape",
      [](PyShapedType &self) {
        self.requireHasRank();

        std::vector<int64_t> shape;
        int64_t rank = mlirShapedTypeGetRank(self);
        shape.reserve(rank);
        for (int64_t i = 0; i < rank; ++i)
          shape.push_back(mlirShapedTypeGetDimSize(self, i));
        return shape;
      },
      "Returns the shape of the ranked shaped type as a list of integers.");
  c.def_static(
      "get_dynamic_size", []() { return mlirShapedTypeGetDynamicSize(); },
      "Returns the value used to indicate dynamic dimensions in shaped "
      "types.");
  c.def_static(
      "get_dynamic_stride_or_offset",
      []() { return mlirShapedTypeGetDynamicStrideOrOffset(); },
      "Returns the value used to indicate dynamic strides or offsets in "
      "shaped types.");
}

void PyShapedType::requireHasRank() {
  if (!mlirShapedTypeHasRank(*this)) {
    throw nb::value_error(
        "calling this method requires that the type has a rank.");
  }
}

const PyShapedType::IsAFunctionTy PyShapedType::isaFunction = mlirTypeIsAShaped;

void PyVectorType::bindDerived(ClassTy &c) {
  c.def_static("get", &PyVectorType::getChecked, nb::arg("shape"),
               nb::arg("element_type"), nb::kw_only(),
               nb::arg("scalable") = nb::none(),
               nb::arg("scalable_dims") = nb::none(),
               nb::arg("loc") = nb::none(), "Create a vector type")
      .def_static("get_unchecked", &PyVectorType::get, nb::arg("shape"),
                  nb::arg("element_type"), nb::kw_only(),
                  nb::arg("scalable") = nb::none(),
                  nb::arg("scalable_dims") = nb::none(),
                  nb::arg("context") = nb::none(), "Create a vector type")
      .def_prop_ro("scalable",
                   [](MlirType self) { return mlirVectorTypeIsScalable(self); })
      .def_prop_ro("scalable_dims", [](MlirType self) {
        std::vector<bool> scalableDims;
        size_t rank = static_cast<size_t>(mlirShapedTypeGetRank(self));
        scalableDims.reserve(rank);
        for (size_t i = 0; i < rank; ++i)
          scalableDims.push_back(mlirVectorTypeIsDimScalable(self, i));
        return scalableDims;
      });
}

PyVectorType
PyVectorType::getChecked(std::vector<int64_t> shape, PyType &elementType,
                         std::optional<nb::list> scalable,
                         std::optional<std::vector<int64_t>> scalableDims,
                         DefaultingPyLocation loc) {
  if (scalable && scalableDims) {
    throw nb::value_error("'scalable' and 'scalable_dims' kwargs "
                          "are mutually exclusive.");
  }

  PyMlirContext::ErrorCapture errors(loc->getContext());
  MlirType type;
  if (scalable) {
    if (scalable->size() != shape.size())
      throw nb::value_error("Expected len(scalable) == len(shape).");

    SmallVector<bool> scalableDimFlags = llvm::to_vector(llvm::map_range(
        *scalable, [](const nb::handle &h) { return nb::cast<bool>(h); }));
    type = mlirVectorTypeGetScalableChecked(
        loc, shape.size(), shape.data(), scalableDimFlags.data(), elementType);
  } else if (scalableDims) {
    SmallVector<bool> scalableDimFlags(shape.size(), false);
    for (int64_t dim : *scalableDims) {
      if (static_cast<size_t>(dim) >= scalableDimFlags.size() || dim < 0)
        throw nb::value_error("Scalable dimension index out of bounds.");
      scalableDimFlags[dim] = true;
    }
    type = mlirVectorTypeGetScalableChecked(
        loc, shape.size(), shape.data(), scalableDimFlags.data(), elementType);
  } else {
    type =
        mlirVectorTypeGetChecked(loc, shape.size(), shape.data(), elementType);
  }
  if (mlirTypeIsNull(type))
    throw MLIRError("Invalid type", errors.take());
  return PyVectorType(elementType.getContext(), type);
}

PyVectorType PyVectorType::get(std::vector<int64_t> shape, PyType &elementType,
                               std::optional<nb::list> scalable,
                               std::optional<std::vector<int64_t>> scalableDims,
                               DefaultingPyMlirContext context) {
  if (scalable && scalableDims) {
    throw nb::value_error("'scalable' and 'scalable_dims' kwargs "
                          "are mutually exclusive.");
  }

  PyMlirContext::ErrorCapture errors(context->getRef());
  MlirType type;
  if (scalable) {
    if (scalable->size() != shape.size())
      throw nb::value_error("Expected len(scalable) == len(shape).");

    SmallVector<bool> scalableDimFlags = llvm::to_vector(llvm::map_range(
        *scalable, [](const nb::handle &h) { return nb::cast<bool>(h); }));
    type = mlirVectorTypeGetScalable(shape.size(), shape.data(),
                                     scalableDimFlags.data(), elementType);
  } else if (scalableDims) {
    SmallVector<bool> scalableDimFlags(shape.size(), false);
    for (int64_t dim : *scalableDims) {
      if (static_cast<size_t>(dim) >= scalableDimFlags.size() || dim < 0)
        throw nb::value_error("Scalable dimension index out of bounds.");
      scalableDimFlags[dim] = true;
    }
    type = mlirVectorTypeGetScalable(shape.size(), shape.data(),
                                     scalableDimFlags.data(), elementType);
  } else {
    type = mlirVectorTypeGet(shape.size(), shape.data(), elementType);
  }
  if (mlirTypeIsNull(type))
    throw MLIRError("Invalid type", errors.take());
  return PyVectorType(elementType.getContext(), type);
}

void PyRankedTensorType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](std::vector<int64_t> shape, PyType &elementType,
         std::optional<PyAttribute> &encodingAttr, DefaultingPyLocation loc) {
        PyMlirContext::ErrorCapture errors(loc->getContext());
        MlirType t = mlirRankedTensorTypeGetChecked(
            loc, shape.size(), shape.data(), elementType,
            encodingAttr ? encodingAttr->get() : mlirAttributeGetNull());
        if (mlirTypeIsNull(t))
          throw MLIRError("Invalid type", errors.take());
        return PyRankedTensorType(elementType.getContext(), t);
      },
      nb::arg("shape"), nb::arg("element_type"),
      nb::arg("encoding") = nb::none(), nb::arg("loc") = nb::none(),
      "Create a ranked tensor type");
  c.def_static(
      "get_unchecked",
      [](std::vector<int64_t> shape, PyType &elementType,
         std::optional<PyAttribute> &encodingAttr,
         DefaultingPyMlirContext context) {
        PyMlirContext::ErrorCapture errors(context->getRef());
        MlirType t = mlirRankedTensorTypeGet(
            shape.size(), shape.data(), elementType,
            encodingAttr ? encodingAttr->get() : mlirAttributeGetNull());
        if (mlirTypeIsNull(t))
          throw MLIRError("Invalid type", errors.take());
        return PyRankedTensorType(elementType.getContext(), t);
      },
      nb::arg("shape"), nb::arg("element_type"),
      nb::arg("encoding") = nb::none(), nb::arg("context") = nb::none(),
      "Create a ranked tensor type");
  c.def_prop_ro(
      "encoding",
      [](PyRankedTensorType &self)
          -> std::optional<nb::typed<nb::object, PyAttribute>> {
        MlirAttribute encoding = mlirRankedTensorTypeGetEncoding(self.get());
        if (mlirAttributeIsNull(encoding))
          return std::nullopt;
        return PyAttribute(self.getContext(), encoding).maybeDownCast();
      });
}

void PyUnrankedTensorType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](PyType &elementType, DefaultingPyLocation loc) {
        PyMlirContext::ErrorCapture errors(loc->getContext());
        MlirType t = mlirUnrankedTensorTypeGetChecked(loc, elementType);
        if (mlirTypeIsNull(t))
          throw MLIRError("Invalid type", errors.take());
        return PyUnrankedTensorType(elementType.getContext(), t);
      },
      nb::arg("element_type"), nb::arg("loc") = nb::none(),
      "Create a unranked tensor type");
  c.def_static(
      "get_unchecked",
      [](PyType &elementType, DefaultingPyMlirContext context) {
        PyMlirContext::ErrorCapture errors(context->getRef());
        MlirType t = mlirUnrankedTensorTypeGet(elementType);
        if (mlirTypeIsNull(t))
          throw MLIRError("Invalid type", errors.take());
        return PyUnrankedTensorType(elementType.getContext(), t);
      },
      nb::arg("element_type"), nb::arg("context") = nb::none(),
      "Create a unranked tensor type");
}

void PyMemRefType::bindDerived(ClassTy &c) {
  c.def_static(
       "get",
       [](std::vector<int64_t> shape, PyType &elementType, PyAttribute *layout,
          PyAttribute *memorySpace, DefaultingPyLocation loc) {
         PyMlirContext::ErrorCapture errors(loc->getContext());
         MlirAttribute layoutAttr = layout ? *layout : mlirAttributeGetNull();
         MlirAttribute memSpaceAttr =
             memorySpace ? *memorySpace : mlirAttributeGetNull();
         MlirType t =
             mlirMemRefTypeGetChecked(loc, elementType, shape.size(),
                                      shape.data(), layoutAttr, memSpaceAttr);
         if (mlirTypeIsNull(t))
           throw MLIRError("Invalid type", errors.take());
         return PyMemRefType(elementType.getContext(), t);
       },
       nb::arg("shape"), nb::arg("element_type"),
       nb::arg("layout") = nb::none(), nb::arg("memory_space") = nb::none(),
       nb::arg("loc") = nb::none(), "Create a memref type")
      .def_static(
          "get_unchecked",
          [](std::vector<int64_t> shape, PyType &elementType,
             PyAttribute *layout, PyAttribute *memorySpace,
             DefaultingPyMlirContext context) {
            PyMlirContext::ErrorCapture errors(context->getRef());
            MlirAttribute layoutAttr =
                layout ? *layout : mlirAttributeGetNull();
            MlirAttribute memSpaceAttr =
                memorySpace ? *memorySpace : mlirAttributeGetNull();
            MlirType t =
                mlirMemRefTypeGet(elementType, shape.size(), shape.data(),
                                  layoutAttr, memSpaceAttr);
            if (mlirTypeIsNull(t))
              throw MLIRError("Invalid type", errors.take());
            return PyMemRefType(elementType.getContext(), t);
          },
          nb::arg("shape"), nb::arg("element_type"),
          nb::arg("layout") = nb::none(), nb::arg("memory_space") = nb::none(),
          nb::arg("context") = nb::none(), "Create a memref type")
      .def_prop_ro(
          "layout",
          [](PyMemRefType &self) -> nb::typed<nb::object, PyAttribute> {
            return PyAttribute(self.getContext(), mlirMemRefTypeGetLayout(self))
                .maybeDownCast();
          },
          "The layout of the MemRef type.")
      .def(
          "get_strides_and_offset",
          [](PyMemRefType &self) -> std::pair<std::vector<int64_t>, int64_t> {
            std::vector<int64_t> strides(mlirShapedTypeGetRank(self));
            int64_t offset;
            if (mlirLogicalResultIsFailure(mlirMemRefTypeGetStridesAndOffset(
                    self, strides.data(), &offset)))
              throw std::runtime_error(
                  "Failed to extract strides and offset from memref.");
            return {strides, offset};
          },
          "The strides and offset of the MemRef type.")
      .def_prop_ro(
          "affine_map",
          [](PyMemRefType &self) -> PyAffineMap {
            MlirAffineMap map = mlirMemRefTypeGetAffineMap(self);
            return PyAffineMap(self.getContext(), map);
          },
          "The layout of the MemRef type as an affine map.")
      .def_prop_ro(
          "memory_space",
          [](PyMemRefType &self)
              -> std::optional<nb::typed<nb::object, PyAttribute>> {
            MlirAttribute a = mlirMemRefTypeGetMemorySpace(self);
            if (mlirAttributeIsNull(a))
              return std::nullopt;
            return PyAttribute(self.getContext(), a).maybeDownCast();
          },
          "Returns the memory space of the given MemRef type.");
}

void PyUnrankedMemRefType::bindDerived(ClassTy &c) {
  c.def_static(
       "get",
       [](PyType &elementType, PyAttribute *memorySpace,
          DefaultingPyLocation loc) {
         PyMlirContext::ErrorCapture errors(loc->getContext());
         MlirAttribute memSpaceAttr = {};
         if (memorySpace)
           memSpaceAttr = *memorySpace;

         MlirType t =
             mlirUnrankedMemRefTypeGetChecked(loc, elementType, memSpaceAttr);
         if (mlirTypeIsNull(t))
           throw MLIRError("Invalid type", errors.take());
         return PyUnrankedMemRefType(elementType.getContext(), t);
       },
       nb::arg("element_type"), nb::arg("memory_space").none(),
       nb::arg("loc") = nb::none(), "Create a unranked memref type")
      .def_static(
          "get_unchecked",
          [](PyType &elementType, PyAttribute *memorySpace,
             DefaultingPyMlirContext context) {
            PyMlirContext::ErrorCapture errors(context->getRef());
            MlirAttribute memSpaceAttr = {};
            if (memorySpace)
              memSpaceAttr = *memorySpace;

            MlirType t = mlirUnrankedMemRefTypeGet(elementType, memSpaceAttr);
            if (mlirTypeIsNull(t))
              throw MLIRError("Invalid type", errors.take());
            return PyUnrankedMemRefType(elementType.getContext(), t);
          },
          nb::arg("element_type"), nb::arg("memory_space").none(),
          nb::arg("context") = nb::none(), "Create a unranked memref type")
      .def_prop_ro(
          "memory_space",
          [](PyUnrankedMemRefType &self)
              -> std::optional<nb::typed<nb::object, PyAttribute>> {
            MlirAttribute a = mlirUnrankedMemrefGetMemorySpace(self);
            if (mlirAttributeIsNull(a))
              return std::nullopt;
            return PyAttribute(self.getContext(), a).maybeDownCast();
          },
          "Returns the memory space of the given Unranked MemRef type.");
}

void PyTupleType::bindDerived(ClassTy &c) {
  c.def_static(
      "get_tuple",
      [](const std::vector<PyType> &elements, DefaultingPyMlirContext context) {
        std::vector<MlirType> mlirElements;
        mlirElements.reserve(elements.size());
        for (const auto &element : elements)
          mlirElements.push_back(element.get());
        MlirType t = mlirTupleTypeGet(context->get(), elements.size(),
                                      mlirElements.data());
        return PyTupleType(context->getRef(), t);
      },
      nb::arg("elements"), nb::arg("context") = nb::none(),
      "Create a tuple type");
  c.def_static(
      "get_tuple",
      [](std::vector<MlirType> elements, DefaultingPyMlirContext context) {
        MlirType t =
            mlirTupleTypeGet(context->get(), elements.size(), elements.data());
        return PyTupleType(context->getRef(), t);
      },
      nb::arg("elements"), nb::arg("context") = nb::none(),
      // clang-format off
        nb::sig("def get_tuple(elements: Sequence[Type], context: Context | None = None) -> TupleType"),
      // clang-format on
      "Create a tuple type");
  c.def(
      "get_type",
      [](PyTupleType &self, intptr_t pos) -> nb::typed<nb::object, PyType> {
        return PyType(self.getContext(), mlirTupleTypeGetType(self, pos))
            .maybeDownCast();
      },
      nb::arg("pos"), "Returns the pos-th type in the tuple type.");
  c.def_prop_ro(
      "num_types",
      [](PyTupleType &self) -> intptr_t {
        return mlirTupleTypeGetNumTypes(self);
      },
      "Returns the number of types contained in a tuple.");
}

void PyFunctionType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](std::vector<PyType> inputs, std::vector<PyType> results,
         DefaultingPyMlirContext context) {
        std::vector<MlirType> mlirInputs;
        mlirInputs.reserve(inputs.size());
        for (const auto &input : inputs)
          mlirInputs.push_back(input.get());
        std::vector<MlirType> mlirResults;
        mlirResults.reserve(results.size());
        for (const auto &result : results)
          mlirResults.push_back(result.get());

        MlirType t = mlirFunctionTypeGet(context->get(), inputs.size(),
                                         mlirInputs.data(), results.size(),
                                         mlirResults.data());
        return PyFunctionType(context->getRef(), t);
      },
      nb::arg("inputs"), nb::arg("results"), nb::arg("context") = nb::none(),
      "Gets a FunctionType from a list of input and result types");
  c.def_static(
      "get",
      [](std::vector<MlirType> inputs, std::vector<MlirType> results,
         DefaultingPyMlirContext context) {
        MlirType t =
            mlirFunctionTypeGet(context->get(), inputs.size(), inputs.data(),
                                results.size(), results.data());
        return PyFunctionType(context->getRef(), t);
      },
      nb::arg("inputs"), nb::arg("results"), nb::arg("context") = nb::none(),
      // clang-format off
        nb::sig("def get(inputs: Sequence[Type], results: Sequence[Type], context: Context | None = None) -> FunctionType"),
      // clang-format on
      "Gets a FunctionType from a list of input and result types");
  c.def_prop_ro(
      "inputs",
      [](PyFunctionType &self) {
        MlirType t = self;
        nb::list types;
        for (intptr_t i = 0, e = mlirFunctionTypeGetNumInputs(self); i < e;
             ++i) {
          types.append(mlirFunctionTypeGetInput(t, i));
        }
        return types;
      },
      "Returns the list of input types in the FunctionType.");
  c.def_prop_ro(
      "results",
      [](PyFunctionType &self) {
        nb::list types;
        for (intptr_t i = 0, e = mlirFunctionTypeGetNumResults(self); i < e;
             ++i) {
          types.append(mlirFunctionTypeGetResult(self, i));
        }
        return types;
      },
      "Returns the list of result types in the FunctionType.");
}

void PyOpaqueType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](const std::string &dialectNamespace, const std::string &typeData,
         DefaultingPyMlirContext context) {
        MlirType type =
            mlirOpaqueTypeGet(context->get(), toMlirStringRef(dialectNamespace),
                              toMlirStringRef(typeData));
        return PyOpaqueType(context->getRef(), type);
      },
      nb::arg("dialect_namespace"), nb::arg("buffer"),
      nb::arg("context") = nb::none(),
      "Create an unregistered (opaque) dialect type.");
  c.def_prop_ro(
      "dialect_namespace",
      [](PyOpaqueType &self) {
        MlirStringRef stringRef = mlirOpaqueTypeGetDialectNamespace(self);
        return nb::str(stringRef.data, stringRef.length);
      },
      "Returns the dialect namespace for the Opaque type as a string.");
  c.def_prop_ro(
      "data",
      [](PyOpaqueType &self) {
        MlirStringRef stringRef = mlirOpaqueTypeGetData(self);
        return nb::str(stringRef.data, stringRef.length);
      },
      "Returns the data for the Opaque type as a string.");
}

void populateIRTypes(nb::module_ &m) {
  PyIntegerType::bind(m);
  PyFloatType::bind(m);
  PyIndexType::bind(m);
  PyFloat4E2M1FNType::bind(m);
  PyFloat6E2M3FNType::bind(m);
  PyFloat6E3M2FNType::bind(m);
  PyFloat8E4M3FNType::bind(m);
  PyFloat8E5M2Type::bind(m);
  PyFloat8E4M3Type::bind(m);
  PyFloat8E4M3FNUZType::bind(m);
  PyFloat8E4M3B11FNUZType::bind(m);
  PyFloat8E5M2FNUZType::bind(m);
  PyFloat8E3M4Type::bind(m);
  PyFloat8E8M0FNUType::bind(m);
  PyBF16Type::bind(m);
  PyF16Type::bind(m);
  PyTF32Type::bind(m);
  PyF32Type::bind(m);
  PyF64Type::bind(m);
  PyNoneType::bind(m);
  PyComplexType::bind(m);
  PyShapedType::bind(m);
  PyVectorType::bind(m);
  PyRankedTensorType::bind(m);
  PyUnrankedTensorType::bind(m);
  PyMemRefType::bind(m);
  PyUnrankedMemRefType::bind(m);
  PyTupleType::bind(m);
  PyFunctionType::bind(m);
  PyOpaqueType::bind(m);
}
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir
