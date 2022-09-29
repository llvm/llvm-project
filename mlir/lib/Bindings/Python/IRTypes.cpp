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

/// Floating Point Type subclass - BF16Type.
class PyBF16Type : public PyConcreteType<PyBF16Type> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsABF16;
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

/// Floating Point Type subclass - F32Type.
class PyF32Type : public PyConcreteType<PyF32Type> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAF32;
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
          throw SetPyError(
              PyExc_ValueError,
              Twine("invalid '") +
                  py::repr(py::cast(elementType)).cast<std::string>() +
                  "' and expected floating point or integer type.");
        },
        "Create a complex type");
    c.def_property_readonly(
        "element_type",
        [](PyComplexType &self) -> PyType {
          MlirType t = mlirComplexTypeGetElementType(self);
          return PyType(self.getContext(), t);
        },
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
        [](PyShapedType &self) {
          MlirType t = mlirShapedTypeGetElementType(self);
          return PyType(self.getContext(), t);
        },
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
      throw SetPyError(
          PyExc_ValueError,
          "calling this method requires that the type has a rank.");
    }
  }
};

/// Vector Type subclass - VectorType.
class PyVectorType : public PyConcreteType<PyVectorType, PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAVector;
  static constexpr const char *pyClassName = "VectorType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](std::vector<int64_t> shape, PyType &elementType,
           DefaultingPyLocation loc) {
          MlirType t = mlirVectorTypeGetChecked(loc, shape.size(), shape.data(),
                                                elementType);
          // TODO: Rework error reporting once diagnostic engine is exposed
          // in C API.
          if (mlirTypeIsNull(t)) {
            throw SetPyError(
                PyExc_ValueError,
                Twine("invalid '") +
                    py::repr(py::cast(elementType)).cast<std::string>() +
                    "' and expected floating point or integer type.");
          }
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
  static constexpr const char *pyClassName = "RankedTensorType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](std::vector<int64_t> shape, PyType &elementType,
           llvm::Optional<PyAttribute> &encodingAttr,
           DefaultingPyLocation loc) {
          MlirType t = mlirRankedTensorTypeGetChecked(
              loc, shape.size(), shape.data(), elementType,
              encodingAttr ? encodingAttr->get() : mlirAttributeGetNull());
          // TODO: Rework error reporting once diagnostic engine is exposed
          // in C API.
          if (mlirTypeIsNull(t)) {
            throw SetPyError(
                PyExc_ValueError,
                Twine("invalid '") +
                    py::repr(py::cast(elementType)).cast<std::string>() +
                    "' and expected floating point, integer, vector or "
                    "complex "
                    "type.");
          }
          return PyRankedTensorType(elementType.getContext(), t);
        },
        py::arg("shape"), py::arg("element_type"),
        py::arg("encoding") = py::none(), py::arg("loc") = py::none(),
        "Create a ranked tensor type");
    c.def_property_readonly(
        "encoding",
        [](PyRankedTensorType &self) -> llvm::Optional<PyAttribute> {
          MlirAttribute encoding = mlirRankedTensorTypeGetEncoding(self.get());
          if (mlirAttributeIsNull(encoding))
            return llvm::None;
          return PyAttribute(self.getContext(), encoding);
        });
  }
};

/// Unranked Tensor Type subclass - UnrankedTensorType.
class PyUnrankedTensorType
    : public PyConcreteType<PyUnrankedTensorType, PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAUnrankedTensor;
  static constexpr const char *pyClassName = "UnrankedTensorType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyType &elementType, DefaultingPyLocation loc) {
          MlirType t = mlirUnrankedTensorTypeGetChecked(loc, elementType);
          // TODO: Rework error reporting once diagnostic engine is exposed
          // in C API.
          if (mlirTypeIsNull(t)) {
            throw SetPyError(
                PyExc_ValueError,
                Twine("invalid '") +
                    py::repr(py::cast(elementType)).cast<std::string>() +
                    "' and expected floating point, integer, vector or "
                    "complex "
                    "type.");
          }
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
  static constexpr const char *pyClassName = "MemRefType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
         "get",
         [](std::vector<int64_t> shape, PyType &elementType,
            PyAttribute *layout, PyAttribute *memorySpace,
            DefaultingPyLocation loc) {
           MlirAttribute layoutAttr = layout ? *layout : mlirAttributeGetNull();
           MlirAttribute memSpaceAttr =
               memorySpace ? *memorySpace : mlirAttributeGetNull();
           MlirType t =
               mlirMemRefTypeGetChecked(loc, elementType, shape.size(),
                                        shape.data(), layoutAttr, memSpaceAttr);
           // TODO: Rework error reporting once diagnostic engine is exposed
           // in C API.
           if (mlirTypeIsNull(t)) {
             throw SetPyError(
                 PyExc_ValueError,
                 Twine("invalid '") +
                     py::repr(py::cast(elementType)).cast<std::string>() +
                     "' and expected floating point, integer, vector or "
                     "complex "
                     "type.");
           }
           return PyMemRefType(elementType.getContext(), t);
         },
         py::arg("shape"), py::arg("element_type"),
         py::arg("layout") = py::none(), py::arg("memory_space") = py::none(),
         py::arg("loc") = py::none(), "Create a memref type")
        .def_property_readonly(
            "layout",
            [](PyMemRefType &self) -> PyAttribute {
              MlirAttribute layout = mlirMemRefTypeGetLayout(self);
              return PyAttribute(self.getContext(), layout);
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
            [](PyMemRefType &self) -> PyAttribute {
              MlirAttribute a = mlirMemRefTypeGetMemorySpace(self);
              return PyAttribute(self.getContext(), a);
            },
            "Returns the memory space of the given MemRef type.");
  }
};

/// Unranked MemRef Type subclass - UnrankedMemRefType.
class PyUnrankedMemRefType
    : public PyConcreteType<PyUnrankedMemRefType, PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAUnrankedMemRef;
  static constexpr const char *pyClassName = "UnrankedMemRefType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
         "get",
         [](PyType &elementType, PyAttribute *memorySpace,
            DefaultingPyLocation loc) {
           MlirAttribute memSpaceAttr = {};
           if (memorySpace)
             memSpaceAttr = *memorySpace;

           MlirType t =
               mlirUnrankedMemRefTypeGetChecked(loc, elementType, memSpaceAttr);
           // TODO: Rework error reporting once diagnostic engine is exposed
           // in C API.
           if (mlirTypeIsNull(t)) {
             throw SetPyError(
                 PyExc_ValueError,
                 Twine("invalid '") +
                     py::repr(py::cast(elementType)).cast<std::string>() +
                     "' and expected floating point, integer, vector or "
                     "complex "
                     "type.");
           }
           return PyUnrankedMemRefType(elementType.getContext(), t);
         },
         py::arg("element_type"), py::arg("memory_space"),
         py::arg("loc") = py::none(), "Create a unranked memref type")
        .def_property_readonly(
            "memory_space",
            [](PyUnrankedMemRefType &self) -> PyAttribute {
              MlirAttribute a = mlirMemRefTypeGetMemorySpace(self);
              return PyAttribute(self.getContext(), a);
            },
            "Returns the memory space of the given Unranked MemRef type.");
  }
};

/// Tuple Type subclass - TupleType.
class PyTupleType : public PyConcreteType<PyTupleType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsATuple;
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
        [](PyTupleType &self, intptr_t pos) -> PyType {
          MlirType t = mlirTupleTypeGetType(self, pos);
          return PyType(self.getContext(), t);
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
            types.append(PyType(contextRef, mlirFunctionTypeGetInput(t, i)));
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
            types.append(
                PyType(contextRef, mlirFunctionTypeGetResult(self, i)));
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
  PyBF16Type::bind(m);
  PyF16Type::bind(m);
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
