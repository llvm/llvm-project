//===- IRTypes.cpp - Exports builtin and standard types -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// clang-format off
#include "IRModule.h"
#include "mlir/Bindings/Python/IRTypes.h"
// clang-format on

#include <optional>

#include "IRModule.h"
#include "NanobindUtils.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;
using namespace mlir;
using namespace mlir::python;

using llvm::SmallVector;

namespace {

/// Checks whether the given type is an integer or float type.
static int mlirTypeIsAIntegerOrFloat(MlirType type) {
  return mlirTypeIsAInteger(type) || mlirTypeIsABF16(type) ||
         mlirTypeIsAF16(type) || mlirTypeIsAF32(type) || mlirTypeIsAF64(type);
}

static void populateIRTypesModule(const nanobind::module_ &m) {
  using namespace nanobind_adaptors;

  mlir_type_subclass integerType(m, "IntegerType", mlirTypeIsAInteger,
                                 mlirIntegerTypeGetTypeID, &m);
  integerType.def_classmethod(
      "get_signless",
      [](const nb::object &cls, unsigned width, MlirContext ctx) {
        return cls(mlirIntegerTypeGet(ctx, width));
      },
      nb::arg("cls"), nb::arg("width"), nb::arg("context") = nb::none(),
      "Create a signless integer type");
  integerType.def_classmethod(
      "get_signed",
      [](const nb::object &cls, unsigned width, MlirContext ctx) {
        return cls(mlirIntegerTypeSignedGet(ctx, width));
      },
      nb::arg("cls"), nb::arg("width"), nb::arg("context") = nb::none(),
      "Create a signed integer type");
  integerType.def_classmethod(
      "get_unsigned",
      [](const nb::object &cls, unsigned width, MlirContext ctx) {
        return cls(mlirIntegerTypeUnsignedGet(ctx, width));
      },
      nb::arg("cls"), nb::arg("width"), nb::arg("context") = nb::none(),
      "Create an unsigned integer type");
  integerType.def_property_readonly(
      "width", [](MlirType self) { return mlirIntegerTypeGetWidth(self); },
      "Returns the width of the integer type");
  integerType.def_property_readonly(
      "is_signless",
      [](MlirType self) { return mlirIntegerTypeIsSignless(self); },
      "Returns whether this is a signless integer");
  integerType.def_property_readonly(
      "is_signed", [](MlirType self) { return mlirIntegerTypeIsSigned(self); },
      "Returns whether this is a signed integer");
  integerType.def_property_readonly(
      "is_unsigned",
      [](MlirType self) { return mlirIntegerTypeIsUnsigned(self); },
      "Returns whether this is an unsigned integer");

  // IndexType
  mlir_type_subclass indexType(m, "IndexType", mlirTypeIsAIndex,
                               mlirIndexTypeGetTypeID, &m);

  indexType.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirIndexTypeGet(ctx));
      },
      nb::arg("cls"), nb::arg("context") = nb::none(), "Create a index type.");

  // FloatType (base class for specific float types)
  mlir_type_subclass floatType(m, "FloatType", mlirTypeIsAFloat, nullptr, &m);
  floatType.def_property_readonly(
      "width", [](MlirType self) { return mlirFloatTypeGetWidth(self); },
      "Returns the width of the floating-point type");

  // Float4E2M1FNType
  mlir_type_subclass float4E2M1FNType(
      m, "Float4E2M1FNType", mlirTypeIsAFloat4E2M1FN, floatType.get_class(),
      mlirFloat4E2M1FNTypeGetTypeID, &m);
  float4E2M1FNType.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirFloat4E2M1FNTypeGet(ctx));
      },
      nb::arg("cls"), nb::arg("context") = nb::none(),
      "Create a float4_e2m1fn type.");

  // Float6E2M3FNType
  mlir_type_subclass float6E2M3FNType(
      m, "Float6E2M3FNType", mlirTypeIsAFloat6E2M3FN, floatType.get_class(),
      mlirFloat6E2M3FNTypeGetTypeID, &m);
  float6E2M3FNType.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirFloat6E2M3FNTypeGet(ctx));
      },
      nb::arg("cls"), nb::arg("context") = nb::none(),
      "Create a float6_e2m3fn type.");

  // Float6E3M2FNType
  mlir_type_subclass float6E3M2FNType(
      m, "Float6E3M2FNType", mlirTypeIsAFloat6E3M2FN, floatType.get_class(),
      mlirFloat6E3M2FNTypeGetTypeID, &m);
  float6E3M2FNType.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirFloat6E3M2FNTypeGet(ctx));
      },
      nb::arg("cls"), nb::arg("context") = nb::none(),
      "Create a float6_e3m2fn type.");

  // Float8E4M3FNType
  mlir_type_subclass float8E4M3FNType(
      m, "Float8E4M3FNType", mlirTypeIsAFloat8E4M3FN, floatType.get_class(),
      mlirFloat8E4M3FNTypeGetTypeID, &m);
  float8E4M3FNType.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirFloat8E4M3FNTypeGet(ctx));
      },
      nb::arg("cls"), nb::arg("context") = nb::none(),
      "Create a float8_e4m3fn type.");

  // Float8E5M2Type
  mlir_type_subclass float8E5M2Type(m, "Float8E5M2Type", mlirTypeIsAFloat8E5M2,
                                    floatType.get_class(),
                                    mlirFloat8E5M2TypeGetTypeID, &m);
  float8E5M2Type.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirFloat8E5M2TypeGet(ctx));
      },
      nb::arg("cls"), nb::arg("context") = nb::none(),
      "Create a float8_e5m2 type.");

  // Float8E4M3Type
  mlir_type_subclass float8E4M3Type(m, "Float8E4M3Type", mlirTypeIsAFloat8E4M3,
                                    floatType.get_class(),
                                    mlirFloat8E4M3TypeGetTypeID, &m);
  float8E4M3Type.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirFloat8E4M3TypeGet(ctx));
      },
      nb::arg("cls"), nb::arg("context") = nb::none(),
      "Create a float8_e4m3 type.");

  // Float8E4M3FNUZType
  mlir_type_subclass float8E4M3FNUZType(
      m, "Float8E4M3FNUZType", mlirTypeIsAFloat8E4M3FNUZ, floatType.get_class(),
      mlirFloat8E4M3FNUZTypeGetTypeID, &m);
  float8E4M3FNUZType.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirFloat8E4M3FNUZTypeGet(ctx));
      },
      nb::arg("cls"), nb::arg("context") = nb::none(),
      "Create a float8_e4m3fnuz type.");

  // Float8E4M3B11FNUZType
  mlir_type_subclass float8E4M3B11FNUZType(
      m, "Float8E4M3B11FNUZType", mlirTypeIsAFloat8E4M3B11FNUZ,
      floatType.get_class(), mlirFloat8E4M3B11FNUZTypeGetTypeID, &m);
  float8E4M3B11FNUZType.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirFloat8E4M3B11FNUZTypeGet(ctx));
      },
      nb::arg("cls"), nb::arg("context") = nb::none(),
      "Create a float8_e4m3b11fnuz type.");

  // Float8E5M2FNUZType
  mlir_type_subclass float8E5M2FNUZType(
      m, "Float8E5M2FNUZType", mlirTypeIsAFloat8E5M2FNUZ, floatType.get_class(),
      mlirFloat8E5M2FNUZTypeGetTypeID, &m);
  float8E5M2FNUZType.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirFloat8E5M2FNUZTypeGet(ctx));
      },
      nb::arg("cls"), nb::arg("context") = nb::none(),
      "Create a float8_e5m2fnuz type.");

  // Float8E3M4Type
  mlir_type_subclass float8E3M4Type(m, "Float8E3M4Type", mlirTypeIsAFloat8E3M4,
                                    floatType.get_class(),
                                    mlirFloat8E3M4TypeGetTypeID, &m);
  float8E3M4Type.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirFloat8E3M4TypeGet(ctx));
      },
      nb::arg("cls"), nb::arg("context") = nb::none(),
      "Create a float8_e3m4 type.");

  // Float8E8M0FNUType
  mlir_type_subclass float8E8M0FNUType(
      m, "Float8E8M0FNUType", mlirTypeIsAFloat8E8M0FNU, floatType.get_class(),
      mlirFloat8E8M0FNUTypeGetTypeID, &m);
  float8E8M0FNUType.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirFloat8E8M0FNUTypeGet(ctx));
      },
      nb::arg("cls"), nb::arg("context") = nb::none(),
      "Create a float8_e8m0fnu type.");

  // BF16Type
  mlir_type_subclass bf16Type(m, "BF16Type", mlirTypeIsABF16,
                              floatType.get_class(), mlirBFloat16TypeGetTypeID,
                              &m);
  bf16Type.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirBF16TypeGet(ctx));
      },
      nb::arg("cls"), nb::arg("context") = nb::none(), "Create a bf16 type.");

  // F16Type
  mlir_type_subclass f16Type(m, "F16Type", mlirTypeIsAF16,
                             floatType.get_class(), mlirFloat16TypeGetTypeID,
                             &m);
  f16Type.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirF16TypeGet(ctx));
      },
      nb::arg("cls"), nb::arg("context") = nb::none(), "Create a f16 type.");

  // FloatTF32Type
  mlir_type_subclass tf32Type(m, "FloatTF32Type", mlirTypeIsATF32,
                              floatType.get_class(), mlirFloatTF32TypeGetTypeID,
                              &m);
  tf32Type.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirTF32TypeGet(ctx));
      },
      nb::arg("cls"), nb::arg("context") = nb::none(), "Create a tf32 type.");

  // F32Type
  mlir_type_subclass f32Type(m, "F32Type", mlirTypeIsAF32,
                             floatType.get_class(), mlirFloat32TypeGetTypeID,
                             &m);
  f32Type.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirF32TypeGet(ctx));
      },
      nb::arg("cls"), nb::arg("context") = nb::none(), "Create a f32 type.");

  // F64Type
  mlir_type_subclass f64Type(m, "F64Type", mlirTypeIsAF64,
                             floatType.get_class(), mlirFloat64TypeGetTypeID,
                             &m);
  f64Type.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirF64TypeGet(ctx));
      },
      nb::arg("cls"), nb::arg("context") = nb::none(), "Create a f64 type.");

  // NoneType
  mlir_type_subclass noneType(m, "NoneType", mlirTypeIsANone,
                              mlirNoneTypeGetTypeID, &m);
  noneType.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirNoneTypeGet(ctx));
      },
      nb::arg("cls"), nb::arg("context") = nb::none(), "Create a none type.");

  // ComplexType
  mlir_type_subclass complexType(m, "ComplexType", mlirTypeIsAComplex,
                                 mlirComplexTypeGetTypeID, &m);
  complexType.def_classmethod(
      "get",
      [](const nb::object &cls, MlirType elementType) {
        // The element must be a floating point or integer scalar type.
        if (mlirTypeIsAIntegerOrFloat(elementType)) {
          return cls(mlirComplexTypeGet(elementType));
        }
        throw nb::value_error("Invalid element type for ComplexType: expected "
                              "floating point or integer type.");
      },
      "Create a complex type");
  complexType.def_property_readonly(
      "element_type",
      [](MlirType self) { return mlirComplexTypeGetElementType(self); },
      "Returns element type.");

  // TupleType
  mlir_type_subclass tupleType(m, "TupleType", mlirTypeIsATuple,
                               mlirTupleTypeGetTypeID, &m);
  tupleType.def_classmethod(
      "get_tuple",
      [](const nb::object &cls, std::vector<MlirType> elements,
         MlirContext ctx) {
        return cls(mlirTupleTypeGet(ctx, elements.size(), elements.data()));
      },
      nb::arg("cls"), nb::arg("elements"), nb::arg("context") = nb::none(),
      "Create a tuple type");
  tupleType.def(
      "get_type",
      [](MlirType self, intptr_t pos) {
        return mlirTupleTypeGetType(self, pos);
      },
      nb::arg("pos"), "Returns the pos-th type in the tuple type.");
  tupleType.def_property_readonly(
      "num_types", [](MlirType self) { return mlirTupleTypeGetNumTypes(self); },
      "Returns the number of types contained in a tuple.");

  // FunctionType
  mlir_type_subclass functionType(m, "FunctionType", mlirTypeIsAFunction,
                                  mlirFunctionTypeGetTypeID, &m);
  functionType.def_classmethod(
      "get",
      [](const nb::object &cls, std::vector<MlirType> inputs,
         std::vector<MlirType> results, MlirContext ctx) {
        return cls(mlirFunctionTypeGet(ctx, inputs.size(), inputs.data(),
                                       results.size(), results.data()));
      },
      nb::arg("cls"), nb::arg("inputs"), nb::arg("results"),
      nb::arg("context") = nb::none(),
      "Gets a FunctionType from a list of input and result types");
  functionType.def_property_readonly(
      "inputs",
      [](MlirType self) {
        nb::list types;
        for (intptr_t i = 0, e = mlirFunctionTypeGetNumInputs(self); i < e;
             ++i) {
          types.append(mlirFunctionTypeGetInput(self, i));
        }
        return types;
      },
      "Returns the list of input types in the FunctionType.");
  functionType.def_property_readonly(
      "results",
      [](MlirType self) {
        nb::list types;
        for (intptr_t i = 0, e = mlirFunctionTypeGetNumResults(self); i < e;
             ++i) {
          types.append(mlirFunctionTypeGetResult(self, i));
        }
        return types;
      },
      "Returns the list of result types in the FunctionType.");

  // OpaqueType
  mlir_type_subclass opaqueType(m, "OpaqueType", mlirTypeIsAOpaque,
                                mlirOpaqueTypeGetTypeID, &m);
  opaqueType.def_classmethod(
      "get",
      [](const nb::object &cls, const std::string &dialectNamespace,
         const std::string &typeData, MlirContext ctx) {
        MlirStringRef dialectNs = mlirStringRefCreate(dialectNamespace.data(),
                                                      dialectNamespace.size());
        MlirStringRef data =
            mlirStringRefCreate(typeData.data(), typeData.size());
        return cls(mlirOpaqueTypeGet(ctx, dialectNs, data));
      },
      nb::arg("cls"), nb::arg("dialect_namespace"), nb::arg("buffer"),
      nb::arg("context") = nb::none(),
      "Create an unregistered (opaque) dialect type.");
  opaqueType.def_property_readonly(
      "dialect_namespace",
      [](MlirType self) {
        MlirStringRef stringRef = mlirOpaqueTypeGetDialectNamespace(self);
        return nb::str(stringRef.data, stringRef.length);
      },
      "Returns the dialect namespace for the Opaque type as a string.");
  opaqueType.def_property_readonly(
      "data",
      [](MlirType self) {
        MlirStringRef stringRef = mlirOpaqueTypeGetData(self);
        return nb::str(stringRef.data, stringRef.length);
      },
      "Returns the data for the Opaque type as a string.");
}

} // namespace

// Shaped Type Interface - ShapedType
void mlir::PyShapedType::bindDerived(ClassTy &c) {
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

void mlir::PyShapedType::requireHasRank() {
  if (!mlirShapedTypeHasRank(*this)) {
    throw nb::value_error(
        "calling this method requires that the type has a rank.");
  }
}

const mlir::PyShapedType::IsAFunctionTy mlir::PyShapedType::isaFunction =
    mlirTypeIsAShaped;

namespace {

/// Vector Type subclass - VectorType.
class PyVectorType : public PyConcreteType<PyVectorType, PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAVector;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirVectorTypeGetTypeID;
  static constexpr const char *pyClassName = "VectorType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
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
        .def_prop_ro(
            "scalable",
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

private:
  static PyVectorType
  getChecked(std::vector<int64_t> shape, PyType &elementType,
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
      type = mlirVectorTypeGetScalableChecked(loc, shape.size(), shape.data(),
                                              scalableDimFlags.data(),
                                              elementType);
    } else if (scalableDims) {
      SmallVector<bool> scalableDimFlags(shape.size(), false);
      for (int64_t dim : *scalableDims) {
        if (static_cast<size_t>(dim) >= scalableDimFlags.size() || dim < 0)
          throw nb::value_error("Scalable dimension index out of bounds.");
        scalableDimFlags[dim] = true;
      }
      type = mlirVectorTypeGetScalableChecked(loc, shape.size(), shape.data(),
                                              scalableDimFlags.data(),
                                              elementType);
    } else {
      type = mlirVectorTypeGetChecked(loc, shape.size(), shape.data(),
                                      elementType);
    }
    if (mlirTypeIsNull(type))
      throw MLIRError("Invalid type", errors.take());
    return PyVectorType(elementType.getContext(), type);
  }

  static PyVectorType get(std::vector<int64_t> shape, PyType &elementType,
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
};

/// Ranked Tensor Type subclass - RankedTensorType.
class PyRankedTensorType
    : public PyConcreteType<PyRankedTensorType, PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsARankedTensor;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirRankedTensorTypeGetTypeID;
  static constexpr const char *pyClassName = "RankedTensorType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
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
};

/// Unranked Tensor Type subclass - UnrankedTensorType.
class PyUnrankedTensorType
    : public PyConcreteType<PyUnrankedTensorType, PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAUnrankedTensor;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirUnrankedTensorTypeGetTypeID;
  static constexpr const char *pyClassName = "UnrankedTensorType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
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
};

/// Ranked MemRef Type subclass - MemRefType.
class PyMemRefType : public PyConcreteType<PyMemRefType, PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAMemRef;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirMemRefTypeGetTypeID;
  static constexpr const char *pyClassName = "MemRefType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
         "get",
         [](std::vector<int64_t> shape, PyType &elementType,
            PyAttribute *layout, PyAttribute *memorySpace,
            DefaultingPyLocation loc) {
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
            nb::arg("layout") = nb::none(),
            nb::arg("memory_space") = nb::none(),
            nb::arg("context") = nb::none(), "Create a memref type")
        .def_prop_ro(
            "layout",
            [](PyMemRefType &self) -> nb::typed<nb::object, PyAttribute> {
              return PyAttribute(self.getContext(),
                                 mlirMemRefTypeGetLayout(self))
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
};

/// Unranked MemRef Type subclass - UnrankedMemRefType.
class PyUnrankedMemRefType
    : public PyConcreteType<PyUnrankedMemRefType, PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAUnrankedMemRef;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirUnrankedMemRefTypeGetTypeID;
  static constexpr const char *pyClassName = "UnrankedMemRefType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
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
};

} // namespace

void mlir::python::populateIRTypes(nb::module_ &m) {
  // Populate types using mlir_type_subclass
  populateIRTypesModule(m);

  // Keep PyShapedType and its subclasses that weren't replaced
  PyShapedType::bind(m);
  PyVectorType::bind(m);
  PyRankedTensorType::bind(m);
  PyUnrankedTensorType::bind(m);
  PyMemRefType::bind(m);
  PyUnrankedMemRefType::bind(m);
}
