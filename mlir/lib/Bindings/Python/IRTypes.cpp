//===- IRTypes.cpp - Exports builtin and standard types -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IRModule.h"

#include "PybindUtils.h"

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include <optional>

namespace py = pybind11;
using namespace mlir;
using namespace mlir::python;

using llvm::SmallVector;
using llvm::Twine;

namespace {

/// Checks whether the given type is an integer or float type.
static int mlirTypeIsAIntegerOrFloat(MlirType type) {
  return mlirTypeIsAInteger(type) || mlirTypeIsABF16(type) ||
         mlirTypeIsAF16(type) || mlirTypeIsAF32(type) || mlirTypeIsAF64(type);
}

class PyIntegerType : public PyConcreteType<PyIntegerType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAInteger;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirIntegerTypeGetTypeID;
  static constexpr const char *pyClassName = "IntegerType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get_signless",
        [](unsigned width, DefaultingPyMlirContext context) {
          MlirType t = mlirIntegerTypeGet(context->get(), width);
          return PyIntegerType(context->getRef(), t);
        },
        py::arg("width"), py::arg("context") = py::none(),
        "Create a signless integer type");
    c.def_static(
        "get_signed",
        [](unsigned width, DefaultingPyMlirContext context) {
          MlirType t = mlirIntegerTypeSignedGet(context->get(), width);
          return PyIntegerType(context->getRef(), t);
        },
        py::arg("width"), py::arg("context") = py::none(),
        "Create a signed integer type");
    c.def_static(
        "get_unsigned",
        [](unsigned width, DefaultingPyMlirContext context) {
          MlirType t = mlirIntegerTypeUnsignedGet(context->get(), width);
          return PyIntegerType(context->getRef(), t);
        },
        py::arg("width"), py::arg("context") = py::none(),
        "Create an unsigned integer type");
    c.def_property_readonly(
        "width",
        [](PyIntegerType &self) { return mlirIntegerTypeGetWidth(self); },
        "Returns the width of the integer type");
    c.def_property_readonly(
        "is_signless",
        [](PyIntegerType &self) -> bool {
          return mlirIntegerTypeIsSignless(self);
        },
        "Returns whether this is a signless integer");
    c.def_property_readonly(
        "is_signed",
        [](PyIntegerType &self) -> bool {
          return mlirIntegerTypeIsSigned(self);
        },
        "Returns whether this is a signed integer");
    c.def_property_readonly(
        "is_unsigned",
        [](PyIntegerType &self) -> bool {
          return mlirIntegerTypeIsUnsigned(self);
        },
        "Returns whether this is an unsigned integer");
  }
};

/// Index Type subclass - IndexType.
class PyIndexType : public PyConcreteType<PyIndexType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAIndex;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirIndexTypeGetTypeID;
  static constexpr const char *pyClassName = "IndexType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          MlirType t = mlirIndexTypeGet(context->get());
          return PyIndexType(context->getRef(), t);
        },
        py::arg("context") = py::none(), "Create a index type.");
  }
};

/// Floating Point Type subclass - Float8E4M3FNType.
class PyFloat8E4M3FNType : public PyConcreteType<PyFloat8E4M3FNType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFloat8E4M3FN;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloat8E4M3FNTypeGetTypeID;
  static constexpr const char *pyClassName = "Float8E4M3FNType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          MlirType t = mlirFloat8E4M3FNTypeGet(context->get());
          return PyFloat8E4M3FNType(context->getRef(), t);
        },
        py::arg("context") = py::none(), "Create a float8_e4m3fn type.");
  }
};

/// Floating Point Type subclass - Float8M5E2Type.
class PyFloat8E5M2Type : public PyConcreteType<PyFloat8E5M2Type> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFloat8E5M2;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloat8E5M2TypeGetTypeID;
  static constexpr const char *pyClassName = "Float8E5M2Type";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          MlirType t = mlirFloat8E5M2TypeGet(context->get());
          return PyFloat8E5M2Type(context->getRef(), t);
        },
        py::arg("context") = py::none(), "Create a float8_e5m2 type.");
  }
};

/// Floating Point Type subclass - Float8E4M3FNUZ.
class PyFloat8E4M3FNUZType : public PyConcreteType<PyFloat8E4M3FNUZType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFloat8E4M3FNUZ;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloat8E4M3FNUZTypeGetTypeID;
  static constexpr const char *pyClassName = "Float8E4M3FNUZType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          MlirType t = mlirFloat8E4M3FNUZTypeGet(context->get());
          return PyFloat8E4M3FNUZType(context->getRef(), t);
        },
        py::arg("context") = py::none(), "Create a float8_e4m3fnuz type.");
  }
};

/// Floating Point Type subclass - Float8E4M3B11FNUZ.
class PyFloat8E4M3B11FNUZType : public PyConcreteType<PyFloat8E4M3B11FNUZType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFloat8E4M3B11FNUZ;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloat8E4M3B11FNUZTypeGetTypeID;
  static constexpr const char *pyClassName = "Float8E4M3B11FNUZType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          MlirType t = mlirFloat8E4M3B11FNUZTypeGet(context->get());
          return PyFloat8E4M3B11FNUZType(context->getRef(), t);
        },
        py::arg("context") = py::none(), "Create a float8_e4m3b11fnuz type.");
  }
};

/// Floating Point Type subclass - Float8E5M2FNUZ.
class PyFloat8E5M2FNUZType : public PyConcreteType<PyFloat8E5M2FNUZType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFloat8E5M2FNUZ;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloat8E5M2FNUZTypeGetTypeID;
  static constexpr const char *pyClassName = "Float8E5M2FNUZType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          MlirType t = mlirFloat8E5M2FNUZTypeGet(context->get());
          return PyFloat8E5M2FNUZType(context->getRef(), t);
        },
        py::arg("context") = py::none(), "Create a float8_e5m2fnuz type.");
  }
};

/// Floating Point Type subclass - BF16Type.
class PyBF16Type : public PyConcreteType<PyBF16Type> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsABF16;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirBFloat16TypeGetTypeID;
  static constexpr const char *pyClassName = "BF16Type";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          MlirType t = mlirBF16TypeGet(context->get());
          return PyBF16Type(context->getRef(), t);
        },
        py::arg("context") = py::none(), "Create a bf16 type.");
  }
};

/// Floating Point Type subclass - F16Type.
class PyF16Type : public PyConcreteType<PyF16Type> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAF16;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloat16TypeGetTypeID;
  static constexpr const char *pyClassName = "F16Type";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          MlirType t = mlirF16TypeGet(context->get());
          return PyF16Type(context->getRef(), t);
        },
        py::arg("context") = py::none(), "Create a f16 type.");
  }
};

/// Floating Point Type subclass - TF32Type.
class PyTF32Type : public PyConcreteType<PyTF32Type> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsATF32;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloatTF32TypeGetTypeID;
  static constexpr const char *pyClassName = "FloatTF32Type";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          MlirType t = mlirTF32TypeGet(context->get());
          return PyTF32Type(context->getRef(), t);
        },
        py::arg("context") = py::none(), "Create a tf32 type.");
  }
};

/// Floating Point Type subclass - F32Type.
class PyF32Type : public PyConcreteType<PyF32Type> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAF32;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloat32TypeGetTypeID;
  static constexpr const char *pyClassName = "F32Type";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          MlirType t = mlirF32TypeGet(context->get());
          return PyF32Type(context->getRef(), t);
        },
        py::arg("context") = py::none(), "Create a f32 type.");
  }
};

/// Floating Point Type subclass - F64Type.
class PyF64Type : public PyConcreteType<PyF64Type> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAF64;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloat64TypeGetTypeID;
  static constexpr const char *pyClassName = "F64Type";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          MlirType t = mlirF64TypeGet(context->get());
          return PyF64Type(context->getRef(), t);
        },
        py::arg("context") = py::none(), "Create a f64 type.");
  }
};

/// None Type subclass - NoneType.
class PyNoneType : public PyConcreteType<PyNoneType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsANone;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirNoneTypeGetTypeID;
  static constexpr const char *pyClassName = "NoneType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          MlirType t = mlirNoneTypeGet(context->get());
          return PyNoneType(context->getRef(), t);
        },
        py::arg("context") = py::none(), "Create a none type.");
  }
};

/// Complex Type subclass - ComplexType.
class PyComplexType : public PyConcreteType<PyComplexType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAComplex;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirComplexTypeGetTypeID;
  static constexpr const char *pyClassName = "ComplexType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyType &elementType) {
          // The element must be a floating point or integer scalar type.
          if (mlirTypeIsAIntegerOrFloat(elementType)) {
            MlirType t = mlirComplexTypeGet(elementType);
            return PyComplexType(elementType.getContext(), t);
          }
          throw py::value_error(
              (Twine("invalid '") +
               py::repr(py::cast(elementType)).cast<std::string>() +
               "' and expected floating point or integer type.")
                  .str());
        },
        "Create a complex type");
    c.def_property_readonly(
        "element_type",
        [](PyComplexType &self) { return mlirComplexTypeGetElementType(self); },
        "Returns element type.");
  }
};

class PyShapedType : public PyConcreteType<PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAShaped;
  static constexpr const char *pyClassName = "ShapedType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_property_readonly(
        "element_type",
        [](PyShapedType &self) { return mlirShapedTypeGetElementType(self); },
        "Returns the element type of the shaped type.");
    c.def_property_readonly(
        "has_rank",
        [](PyShapedType &self) -> bool { return mlirShapedTypeHasRank(self); },
        "Returns whether the given shaped type is ranked.");
    c.def_property_readonly(
        "rank",
        [](PyShapedType &self) {
          self.requireHasRank();
          return mlirShapedTypeGetRank(self);
        },
        "Returns the rank of the given ranked shaped type.");
    c.def_property_readonly(
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
        py::arg("dim"),
        "Returns whether the dim-th dimension of the given shaped type is "
        "dynamic.");
    c.def(
        "get_dim_size",
        [](PyShapedType &self, intptr_t dim) {
          self.requireHasRank();
          return mlirShapedTypeGetDimSize(self, dim);
        },
        py::arg("dim"),
        "Returns the dim-th dimension of the given ranked shaped type.");
    c.def_static(
        "is_dynamic_size",
        [](int64_t size) -> bool { return mlirShapedTypeIsDynamicSize(size); },
        py::arg("dim_size"),
        "Returns whether the given dimension size indicates a dynamic "
        "dimension.");
    c.def(
        "is_dynamic_stride_or_offset",
        [](PyShapedType &self, int64_t val) -> bool {
          self.requireHasRank();
          return mlirShapedTypeIsDynamicStrideOrOffset(val);
        },
        py::arg("dim_size"),
        "Returns whether the given value is used as a placeholder for dynamic "
        "strides and offsets in shaped types.");
    c.def_property_readonly(
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

private:
  void requireHasRank() {
    if (!mlirShapedTypeHasRank(*this)) {
      throw py::value_error(
          "calling this method requires that the type has a rank.");
    }
  }
};

/// Vector Type subclass - VectorType.
class PyVectorType : public PyConcreteType<PyVectorType, PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAVector;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirVectorTypeGetTypeID;
  static constexpr const char *pyClassName = "VectorType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](std::vector<int64_t> shape, PyType &elementType,
           DefaultingPyLocation loc) {
          PyMlirContext::ErrorCapture errors(loc->getContext());
          MlirType t = mlirVectorTypeGetChecked(loc, shape.size(), shape.data(),
                                                elementType);
          if (mlirTypeIsNull(t))
            throw MLIRError("Invalid type", errors.take());
          return PyVectorType(elementType.getContext(), t);
        },
        py::arg("shape"), py::arg("elementType"), py::arg("loc") = py::none(),
        "Create a vector type");
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
        py::arg("shape"), py::arg("element_type"),
        py::arg("encoding") = py::none(), py::arg("loc") = py::none(),
        "Create a ranked tensor type");
    c.def_property_readonly(
        "encoding",
        [](PyRankedTensorType &self) -> std::optional<MlirAttribute> {
          MlirAttribute encoding = mlirRankedTensorTypeGetEncoding(self.get());
          if (mlirAttributeIsNull(encoding))
            return std::nullopt;
          return encoding;
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
        py::arg("element_type"), py::arg("loc") = py::none(),
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
         py::arg("shape"), py::arg("element_type"),
         py::arg("layout") = py::none(), py::arg("memory_space") = py::none(),
         py::arg("loc") = py::none(), "Create a memref type")
        .def_property_readonly(
            "layout",
            [](PyMemRefType &self) -> MlirAttribute {
              return mlirMemRefTypeGetLayout(self);
            },
            "The layout of the MemRef type.")
        .def_property_readonly(
            "affine_map",
            [](PyMemRefType &self) -> PyAffineMap {
              MlirAffineMap map = mlirMemRefTypeGetAffineMap(self);
              return PyAffineMap(self.getContext(), map);
            },
            "The layout of the MemRef type as an affine map.")
        .def_property_readonly(
            "memory_space",
            [](PyMemRefType &self) -> std::optional<MlirAttribute> {
              MlirAttribute a = mlirMemRefTypeGetMemorySpace(self);
              if (mlirAttributeIsNull(a))
                return std::nullopt;
              return a;
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
         py::arg("element_type"), py::arg("memory_space"),
         py::arg("loc") = py::none(), "Create a unranked memref type")
        .def_property_readonly(
            "memory_space",
            [](PyUnrankedMemRefType &self) -> std::optional<MlirAttribute> {
              MlirAttribute a = mlirUnrankedMemrefGetMemorySpace(self);
              if (mlirAttributeIsNull(a))
                return std::nullopt;
              return a;
            },
            "Returns the memory space of the given Unranked MemRef type.");
  }
};

/// Tuple Type subclass - TupleType.
class PyTupleType : public PyConcreteType<PyTupleType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsATuple;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirTupleTypeGetTypeID;
  static constexpr const char *pyClassName = "TupleType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get_tuple",
        [](py::list elementList, DefaultingPyMlirContext context) {
          intptr_t num = py::len(elementList);
          // Mapping py::list to SmallVector.
          SmallVector<MlirType, 4> elements;
          for (auto element : elementList)
            elements.push_back(element.cast<PyType>());
          MlirType t = mlirTupleTypeGet(context->get(), num, elements.data());
          return PyTupleType(context->getRef(), t);
        },
        py::arg("elements"), py::arg("context") = py::none(),
        "Create a tuple type");
    c.def(
        "get_type",
        [](PyTupleType &self, intptr_t pos) {
          return mlirTupleTypeGetType(self, pos);
        },
        py::arg("pos"), "Returns the pos-th type in the tuple type.");
    c.def_property_readonly(
        "num_types",
        [](PyTupleType &self) -> intptr_t {
          return mlirTupleTypeGetNumTypes(self);
        },
        "Returns the number of types contained in a tuple.");
  }
};

/// Function type.
class PyFunctionType : public PyConcreteType<PyFunctionType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFunction;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFunctionTypeGetTypeID;
  static constexpr const char *pyClassName = "FunctionType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](std::vector<PyType> inputs, std::vector<PyType> results,
           DefaultingPyMlirContext context) {
          SmallVector<MlirType, 4> inputsRaw(inputs.begin(), inputs.end());
          SmallVector<MlirType, 4> resultsRaw(results.begin(), results.end());
          MlirType t = mlirFunctionTypeGet(context->get(), inputsRaw.size(),
                                           inputsRaw.data(), resultsRaw.size(),
                                           resultsRaw.data());
          return PyFunctionType(context->getRef(), t);
        },
        py::arg("inputs"), py::arg("results"), py::arg("context") = py::none(),
        "Gets a FunctionType from a list of input and result types");
    c.def_property_readonly(
        "inputs",
        [](PyFunctionType &self) {
          MlirType t = self;
          auto contextRef = self.getContext();
          py::list types;
          for (intptr_t i = 0, e = mlirFunctionTypeGetNumInputs(self); i < e;
               ++i) {
            types.append(mlirFunctionTypeGetInput(t, i));
          }
          return types;
        },
        "Returns the list of input types in the FunctionType.");
    c.def_property_readonly(
        "results",
        [](PyFunctionType &self) {
          auto contextRef = self.getContext();
          py::list types;
          for (intptr_t i = 0, e = mlirFunctionTypeGetNumResults(self); i < e;
               ++i) {
            types.append(mlirFunctionTypeGetResult(self, i));
          }
          return types;
        },
        "Returns the list of result types in the FunctionType.");
  }
};

static MlirStringRef toMlirStringRef(const std::string &s) {
  return mlirStringRefCreate(s.data(), s.size());
}

/// Opaque Type subclass - OpaqueType.
class PyOpaqueType : public PyConcreteType<PyOpaqueType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAOpaque;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirOpaqueTypeGetTypeID;
  static constexpr const char *pyClassName = "OpaqueType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](std::string dialectNamespace, std::string typeData,
           DefaultingPyMlirContext context) {
          MlirType type = mlirOpaqueTypeGet(context->get(),
                                            toMlirStringRef(dialectNamespace),
                                            toMlirStringRef(typeData));
          return PyOpaqueType(context->getRef(), type);
        },
        py::arg("dialect_namespace"), py::arg("buffer"),
        py::arg("context") = py::none(),
        "Create an unregistered (opaque) dialect type.");
    c.def_property_readonly(
        "dialect_namespace",
        [](PyOpaqueType &self) {
          MlirStringRef stringRef = mlirOpaqueTypeGetDialectNamespace(self);
          return py::str(stringRef.data, stringRef.length);
        },
        "Returns the dialect namespace for the Opaque type as a string.");
    c.def_property_readonly(
        "data",
        [](PyOpaqueType &self) {
          MlirStringRef stringRef = mlirOpaqueTypeGetData(self);
          return py::str(stringRef.data, stringRef.length);
        },
        "Returns the data for the Opaque type as a string.");
  }
};

} // namespace

void mlir::python::populateIRTypes(py::module &m) {
  PyIntegerType::bind(m);
  PyIndexType::bind(m);
  PyFloat8E4M3FNType::bind(m);
  PyFloat8E5M2Type::bind(m);
  PyFloat8E4M3FNUZType::bind(m);
  PyFloat8E4M3B11FNUZType::bind(m);
  PyFloat8E5M2FNUZType::bind(m);
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
