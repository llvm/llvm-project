//===- IRAttributes.cpp - Exports builtin and standard attributes ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <optional>
#include <string_view>
#include <utility>

#include "IRModule.h"

#include "PybindUtils.h"

#include "llvm/ADT/ScopeExit.h"

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace py = pybind11;
using namespace mlir;
using namespace mlir::python;

using llvm::SmallVector;

//------------------------------------------------------------------------------
// Docstrings (trivial, non-duplicated docstrings are included inline).
//------------------------------------------------------------------------------

static const char kDenseElementsAttrGetDocstring[] =
    R"(Gets a DenseElementsAttr from a Python buffer or array.

When `type` is not provided, then some limited type inferencing is done based
on the buffer format. Support presently exists for 8/16/32/64 signed and
unsigned integers and float16/float32/float64. DenseElementsAttrs of these
types can also be converted back to a corresponding buffer.

For conversions outside of these types, a `type=` must be explicitly provided
and the buffer contents must be bit-castable to the MLIR internal
representation:

  * Integer types (except for i1): the buffer must be byte aligned to the
    next byte boundary.
  * Floating point types: Must be bit-castable to the given floating point
    size.
  * i1 (bool): Bit packed into 8bit words where the bit pattern matches a
    row major ordering. An arbitrary Numpy `bool_` array can be bit packed to
    this specification with: `np.packbits(ary, axis=None, bitorder='little')`.

If a single element buffer is passed (or for i1, a single byte with value 0
or 255), then a splat will be created.

Args:
  array: The array or buffer to convert.
  signless: If inferring an appropriate MLIR type, use signless types for
    integers (defaults True).
  type: Skips inference of the MLIR element type and uses this instead. The
    storage size must be consistent with the actual contents of the buffer.
  shape: Overrides the shape of the buffer when constructing the MLIR
    shaped type. This is needed when the physical and logical shape differ (as
    for i1).
  context: Explicit context, if not from context manager.

Returns:
  DenseElementsAttr on success.

Raises:
  ValueError: If the type of the buffer or array cannot be matched to an MLIR
    type or if the buffer does not meet expectations.
)";

namespace {

static MlirStringRef toMlirStringRef(const std::string &s) {
  return mlirStringRefCreate(s.data(), s.size());
}

class PyAffineMapAttribute : public PyConcreteAttribute<PyAffineMapAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAAffineMap;
  static constexpr const char *pyClassName = "AffineMapAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirAffineMapAttrGetTypeID;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyAffineMap &affineMap) {
          MlirAttribute attr = mlirAffineMapAttrGet(affineMap.get());
          return PyAffineMapAttribute(affineMap.getContext(), attr);
        },
        py::arg("affine_map"), "Gets an attribute wrapping an AffineMap.");
  }
};

template <typename T>
static T pyTryCast(py::handle object) {
  try {
    return object.cast<T>();
  } catch (py::cast_error &err) {
    std::string msg =
        std::string(
            "Invalid attribute when attempting to create an ArrayAttribute (") +
        err.what() + ")";
    throw py::cast_error(msg);
  } catch (py::reference_cast_error &err) {
    std::string msg = std::string("Invalid attribute (None?) when attempting "
                                  "to create an ArrayAttribute (") +
                      err.what() + ")";
    throw py::cast_error(msg);
  }
}

/// A python-wrapped dense array attribute with an element type and a derived
/// implementation class.
template <typename EltTy, typename DerivedT>
class PyDenseArrayAttribute : public PyConcreteAttribute<DerivedT> {
public:
  using PyConcreteAttribute<DerivedT>::PyConcreteAttribute;

  /// Iterator over the integer elements of a dense array.
  class PyDenseArrayIterator {
  public:
    PyDenseArrayIterator(PyAttribute attr) : attr(std::move(attr)) {}

    /// Return a copy of the iterator.
    PyDenseArrayIterator dunderIter() { return *this; }

    /// Return the next element.
    EltTy dunderNext() {
      // Throw if the index has reached the end.
      if (nextIndex >= mlirDenseArrayGetNumElements(attr.get()))
        throw py::stop_iteration();
      return DerivedT::getElement(attr.get(), nextIndex++);
    }

    /// Bind the iterator class.
    static void bind(py::module &m) {
      py::class_<PyDenseArrayIterator>(m, DerivedT::pyIteratorName,
                                       py::module_local())
          .def("__iter__", &PyDenseArrayIterator::dunderIter)
          .def("__next__", &PyDenseArrayIterator::dunderNext);
    }

  private:
    /// The referenced dense array attribute.
    PyAttribute attr;
    /// The next index to read.
    int nextIndex = 0;
  };

  /// Get the element at the given index.
  EltTy getItem(intptr_t i) { return DerivedT::getElement(*this, i); }

  /// Bind the attribute class.
  static void bindDerived(typename PyConcreteAttribute<DerivedT>::ClassTy &c) {
    // Bind the constructor.
    c.def_static(
        "get",
        [](const std::vector<EltTy> &values, DefaultingPyMlirContext ctx) {
          return getAttribute(values, ctx->getRef());
        },
        py::arg("values"), py::arg("context") = py::none(),
        "Gets a uniqued dense array attribute");
    // Bind the array methods.
    c.def("__getitem__", [](DerivedT &arr, intptr_t i) {
      if (i >= mlirDenseArrayGetNumElements(arr))
        throw py::index_error("DenseArray index out of range");
      return arr.getItem(i);
    });
    c.def("__len__", [](const DerivedT &arr) {
      return mlirDenseArrayGetNumElements(arr);
    });
    c.def("__iter__",
          [](const DerivedT &arr) { return PyDenseArrayIterator(arr); });
    c.def("__add__", [](DerivedT &arr, const py::list &extras) {
      std::vector<EltTy> values;
      intptr_t numOldElements = mlirDenseArrayGetNumElements(arr);
      values.reserve(numOldElements + py::len(extras));
      for (intptr_t i = 0; i < numOldElements; ++i)
        values.push_back(arr.getItem(i));
      for (py::handle attr : extras)
        values.push_back(pyTryCast<EltTy>(attr));
      return getAttribute(values, arr.getContext());
    });
  }

private:
  static DerivedT getAttribute(const std::vector<EltTy> &values,
                               PyMlirContextRef ctx) {
    if constexpr (std::is_same_v<EltTy, bool>) {
      std::vector<int> intValues(values.begin(), values.end());
      MlirAttribute attr = DerivedT::getAttribute(ctx->get(), intValues.size(),
                                                  intValues.data());
      return DerivedT(ctx, attr);
    } else {
      MlirAttribute attr =
          DerivedT::getAttribute(ctx->get(), values.size(), values.data());
      return DerivedT(ctx, attr);
    }
  }
};

/// Instantiate the python dense array classes.
struct PyDenseBoolArrayAttribute
    : public PyDenseArrayAttribute<bool, PyDenseBoolArrayAttribute> {
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsADenseBoolArray;
  static constexpr auto getAttribute = mlirDenseBoolArrayGet;
  static constexpr auto getElement = mlirDenseBoolArrayGetElement;
  static constexpr const char *pyClassName = "DenseBoolArrayAttr";
  static constexpr const char *pyIteratorName = "DenseBoolArrayIterator";
  using PyDenseArrayAttribute::PyDenseArrayAttribute;
};
struct PyDenseI8ArrayAttribute
    : public PyDenseArrayAttribute<int8_t, PyDenseI8ArrayAttribute> {
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsADenseI8Array;
  static constexpr auto getAttribute = mlirDenseI8ArrayGet;
  static constexpr auto getElement = mlirDenseI8ArrayGetElement;
  static constexpr const char *pyClassName = "DenseI8ArrayAttr";
  static constexpr const char *pyIteratorName = "DenseI8ArrayIterator";
  using PyDenseArrayAttribute::PyDenseArrayAttribute;
};
struct PyDenseI16ArrayAttribute
    : public PyDenseArrayAttribute<int16_t, PyDenseI16ArrayAttribute> {
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsADenseI16Array;
  static constexpr auto getAttribute = mlirDenseI16ArrayGet;
  static constexpr auto getElement = mlirDenseI16ArrayGetElement;
  static constexpr const char *pyClassName = "DenseI16ArrayAttr";
  static constexpr const char *pyIteratorName = "DenseI16ArrayIterator";
  using PyDenseArrayAttribute::PyDenseArrayAttribute;
};
struct PyDenseI32ArrayAttribute
    : public PyDenseArrayAttribute<int32_t, PyDenseI32ArrayAttribute> {
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsADenseI32Array;
  static constexpr auto getAttribute = mlirDenseI32ArrayGet;
  static constexpr auto getElement = mlirDenseI32ArrayGetElement;
  static constexpr const char *pyClassName = "DenseI32ArrayAttr";
  static constexpr const char *pyIteratorName = "DenseI32ArrayIterator";
  using PyDenseArrayAttribute::PyDenseArrayAttribute;
};
struct PyDenseI64ArrayAttribute
    : public PyDenseArrayAttribute<int64_t, PyDenseI64ArrayAttribute> {
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsADenseI64Array;
  static constexpr auto getAttribute = mlirDenseI64ArrayGet;
  static constexpr auto getElement = mlirDenseI64ArrayGetElement;
  static constexpr const char *pyClassName = "DenseI64ArrayAttr";
  static constexpr const char *pyIteratorName = "DenseI64ArrayIterator";
  using PyDenseArrayAttribute::PyDenseArrayAttribute;
};
struct PyDenseF32ArrayAttribute
    : public PyDenseArrayAttribute<float, PyDenseF32ArrayAttribute> {
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsADenseF32Array;
  static constexpr auto getAttribute = mlirDenseF32ArrayGet;
  static constexpr auto getElement = mlirDenseF32ArrayGetElement;
  static constexpr const char *pyClassName = "DenseF32ArrayAttr";
  static constexpr const char *pyIteratorName = "DenseF32ArrayIterator";
  using PyDenseArrayAttribute::PyDenseArrayAttribute;
};
struct PyDenseF64ArrayAttribute
    : public PyDenseArrayAttribute<double, PyDenseF64ArrayAttribute> {
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsADenseF64Array;
  static constexpr auto getAttribute = mlirDenseF64ArrayGet;
  static constexpr auto getElement = mlirDenseF64ArrayGetElement;
  static constexpr const char *pyClassName = "DenseF64ArrayAttr";
  static constexpr const char *pyIteratorName = "DenseF64ArrayIterator";
  using PyDenseArrayAttribute::PyDenseArrayAttribute;
};

class PyArrayAttribute : public PyConcreteAttribute<PyArrayAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAArray;
  static constexpr const char *pyClassName = "ArrayAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirArrayAttrGetTypeID;

  class PyArrayAttributeIterator {
  public:
    PyArrayAttributeIterator(PyAttribute attr) : attr(std::move(attr)) {}

    PyArrayAttributeIterator &dunderIter() { return *this; }

    MlirAttribute dunderNext() {
      // TODO: Throw is an inefficient way to stop iteration.
      if (nextIndex >= mlirArrayAttrGetNumElements(attr.get()))
        throw py::stop_iteration();
      return mlirArrayAttrGetElement(attr.get(), nextIndex++);
    }

    static void bind(py::module &m) {
      py::class_<PyArrayAttributeIterator>(m, "ArrayAttributeIterator",
                                           py::module_local())
          .def("__iter__", &PyArrayAttributeIterator::dunderIter)
          .def("__next__", &PyArrayAttributeIterator::dunderNext);
    }

  private:
    PyAttribute attr;
    int nextIndex = 0;
  };

  MlirAttribute getItem(intptr_t i) {
    return mlirArrayAttrGetElement(*this, i);
  }

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](py::list attributes, DefaultingPyMlirContext context) {
          SmallVector<MlirAttribute> mlirAttributes;
          mlirAttributes.reserve(py::len(attributes));
          for (auto attribute : attributes) {
            mlirAttributes.push_back(pyTryCast<PyAttribute>(attribute));
          }
          MlirAttribute attr = mlirArrayAttrGet(
              context->get(), mlirAttributes.size(), mlirAttributes.data());
          return PyArrayAttribute(context->getRef(), attr);
        },
        py::arg("attributes"), py::arg("context") = py::none(),
        "Gets a uniqued Array attribute");
    c.def("__getitem__",
          [](PyArrayAttribute &arr, intptr_t i) {
            if (i >= mlirArrayAttrGetNumElements(arr))
              throw py::index_error("ArrayAttribute index out of range");
            return arr.getItem(i);
          })
        .def("__len__",
             [](const PyArrayAttribute &arr) {
               return mlirArrayAttrGetNumElements(arr);
             })
        .def("__iter__", [](const PyArrayAttribute &arr) {
          return PyArrayAttributeIterator(arr);
        });
    c.def("__add__", [](PyArrayAttribute arr, py::list extras) {
      std::vector<MlirAttribute> attributes;
      intptr_t numOldElements = mlirArrayAttrGetNumElements(arr);
      attributes.reserve(numOldElements + py::len(extras));
      for (intptr_t i = 0; i < numOldElements; ++i)
        attributes.push_back(arr.getItem(i));
      for (py::handle attr : extras)
        attributes.push_back(pyTryCast<PyAttribute>(attr));
      MlirAttribute arrayAttr = mlirArrayAttrGet(
          arr.getContext()->get(), attributes.size(), attributes.data());
      return PyArrayAttribute(arr.getContext(), arrayAttr);
    });
  }
};

/// Float Point Attribute subclass - FloatAttr.
class PyFloatAttribute : public PyConcreteAttribute<PyFloatAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAFloat;
  static constexpr const char *pyClassName = "FloatAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloatAttrGetTypeID;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyType &type, double value, DefaultingPyLocation loc) {
          PyMlirContext::ErrorCapture errors(loc->getContext());
          MlirAttribute attr = mlirFloatAttrDoubleGetChecked(loc, type, value);
          if (mlirAttributeIsNull(attr))
            throw MLIRError("Invalid attribute", errors.take());
          return PyFloatAttribute(type.getContext(), attr);
        },
        py::arg("type"), py::arg("value"), py::arg("loc") = py::none(),
        "Gets an uniqued float point attribute associated to a type");
    c.def_static(
        "get_f32",
        [](double value, DefaultingPyMlirContext context) {
          MlirAttribute attr = mlirFloatAttrDoubleGet(
              context->get(), mlirF32TypeGet(context->get()), value);
          return PyFloatAttribute(context->getRef(), attr);
        },
        py::arg("value"), py::arg("context") = py::none(),
        "Gets an uniqued float point attribute associated to a f32 type");
    c.def_static(
        "get_f64",
        [](double value, DefaultingPyMlirContext context) {
          MlirAttribute attr = mlirFloatAttrDoubleGet(
              context->get(), mlirF64TypeGet(context->get()), value);
          return PyFloatAttribute(context->getRef(), attr);
        },
        py::arg("value"), py::arg("context") = py::none(),
        "Gets an uniqued float point attribute associated to a f64 type");
    c.def_property_readonly("value", mlirFloatAttrGetValueDouble,
                            "Returns the value of the float attribute");
    c.def("__float__", mlirFloatAttrGetValueDouble,
          "Converts the value of the float attribute to a Python float");
  }
};

/// Integer Attribute subclass - IntegerAttr.
class PyIntegerAttribute : public PyConcreteAttribute<PyIntegerAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAInteger;
  static constexpr const char *pyClassName = "IntegerAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyType &type, int64_t value) {
          MlirAttribute attr = mlirIntegerAttrGet(type, value);
          return PyIntegerAttribute(type.getContext(), attr);
        },
        py::arg("type"), py::arg("value"),
        "Gets an uniqued integer attribute associated to a type");
    c.def_property_readonly("value", toPyInt,
                            "Returns the value of the integer attribute");
    c.def("__int__", toPyInt,
          "Converts the value of the integer attribute to a Python int");
    c.def_property_readonly_static("static_typeid",
                                   [](py::object & /*class*/) -> MlirTypeID {
                                     return mlirIntegerAttrGetTypeID();
                                   });
  }

private:
  static py::int_ toPyInt(PyIntegerAttribute &self) {
    MlirType type = mlirAttributeGetType(self);
    if (mlirTypeIsAIndex(type) || mlirIntegerTypeIsSignless(type))
      return mlirIntegerAttrGetValueInt(self);
    if (mlirIntegerTypeIsSigned(type))
      return mlirIntegerAttrGetValueSInt(self);
    return mlirIntegerAttrGetValueUInt(self);
  }
};

/// Bool Attribute subclass - BoolAttr.
class PyBoolAttribute : public PyConcreteAttribute<PyBoolAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsABool;
  static constexpr const char *pyClassName = "BoolAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](bool value, DefaultingPyMlirContext context) {
          MlirAttribute attr = mlirBoolAttrGet(context->get(), value);
          return PyBoolAttribute(context->getRef(), attr);
        },
        py::arg("value"), py::arg("context") = py::none(),
        "Gets an uniqued bool attribute");
    c.def_property_readonly("value", mlirBoolAttrGetValue,
                            "Returns the value of the bool attribute");
    c.def("__bool__", mlirBoolAttrGetValue,
          "Converts the value of the bool attribute to a Python bool");
  }
};

class PySymbolRefAttribute : public PyConcreteAttribute<PySymbolRefAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsASymbolRef;
  static constexpr const char *pyClassName = "SymbolRefAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static MlirAttribute fromList(const std::vector<std::string> &symbols,
                                PyMlirContext &context) {
    if (symbols.empty())
      throw std::runtime_error("SymbolRefAttr must be composed of at least "
                               "one symbol.");
    MlirStringRef rootSymbol = toMlirStringRef(symbols[0]);
    SmallVector<MlirAttribute, 3> referenceAttrs;
    for (size_t i = 1; i < symbols.size(); ++i) {
      referenceAttrs.push_back(
          mlirFlatSymbolRefAttrGet(context.get(), toMlirStringRef(symbols[i])));
    }
    return mlirSymbolRefAttrGet(context.get(), rootSymbol,
                                referenceAttrs.size(), referenceAttrs.data());
  }

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const std::vector<std::string> &symbols,
           DefaultingPyMlirContext context) {
          return PySymbolRefAttribute::fromList(symbols, context.resolve());
        },
        py::arg("symbols"), py::arg("context") = py::none(),
        "Gets a uniqued SymbolRef attribute from a list of symbol names");
    c.def_property_readonly(
        "value",
        [](PySymbolRefAttribute &self) {
          std::vector<std::string> symbols = {
              unwrap(mlirSymbolRefAttrGetRootReference(self)).str()};
          for (int i = 0; i < mlirSymbolRefAttrGetNumNestedReferences(self);
               ++i)
            symbols.push_back(
                unwrap(mlirSymbolRefAttrGetRootReference(
                           mlirSymbolRefAttrGetNestedReference(self, i)))
                    .str());
          return symbols;
        },
        "Returns the value of the SymbolRef attribute as a list[str]");
  }
};

class PyFlatSymbolRefAttribute
    : public PyConcreteAttribute<PyFlatSymbolRefAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAFlatSymbolRef;
  static constexpr const char *pyClassName = "FlatSymbolRefAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](std::string value, DefaultingPyMlirContext context) {
          MlirAttribute attr =
              mlirFlatSymbolRefAttrGet(context->get(), toMlirStringRef(value));
          return PyFlatSymbolRefAttribute(context->getRef(), attr);
        },
        py::arg("value"), py::arg("context") = py::none(),
        "Gets a uniqued FlatSymbolRef attribute");
    c.def_property_readonly(
        "value",
        [](PyFlatSymbolRefAttribute &self) {
          MlirStringRef stringRef = mlirFlatSymbolRefAttrGetValue(self);
          return py::str(stringRef.data, stringRef.length);
        },
        "Returns the value of the FlatSymbolRef attribute as a string");
  }
};

class PyOpaqueAttribute : public PyConcreteAttribute<PyOpaqueAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAOpaque;
  static constexpr const char *pyClassName = "OpaqueAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirOpaqueAttrGetTypeID;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](std::string dialectNamespace, py::buffer buffer, PyType &type,
           DefaultingPyMlirContext context) {
          const py::buffer_info bufferInfo = buffer.request();
          intptr_t bufferSize = bufferInfo.size;
          MlirAttribute attr = mlirOpaqueAttrGet(
              context->get(), toMlirStringRef(dialectNamespace), bufferSize,
              static_cast<char *>(bufferInfo.ptr), type);
          return PyOpaqueAttribute(context->getRef(), attr);
        },
        py::arg("dialect_namespace"), py::arg("buffer"), py::arg("type"),
        py::arg("context") = py::none(), "Gets an Opaque attribute.");
    c.def_property_readonly(
        "dialect_namespace",
        [](PyOpaqueAttribute &self) {
          MlirStringRef stringRef = mlirOpaqueAttrGetDialectNamespace(self);
          return py::str(stringRef.data, stringRef.length);
        },
        "Returns the dialect namespace for the Opaque attribute as a string");
    c.def_property_readonly(
        "data",
        [](PyOpaqueAttribute &self) {
          MlirStringRef stringRef = mlirOpaqueAttrGetData(self);
          return py::bytes(stringRef.data, stringRef.length);
        },
        "Returns the data for the Opaqued attributes as `bytes`");
  }
};

class PyStringAttribute : public PyConcreteAttribute<PyStringAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAString;
  static constexpr const char *pyClassName = "StringAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirStringAttrGetTypeID;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](std::string value, DefaultingPyMlirContext context) {
          MlirAttribute attr =
              mlirStringAttrGet(context->get(), toMlirStringRef(value));
          return PyStringAttribute(context->getRef(), attr);
        },
        py::arg("value"), py::arg("context") = py::none(),
        "Gets a uniqued string attribute");
    c.def_static(
        "get_typed",
        [](PyType &type, std::string value) {
          MlirAttribute attr =
              mlirStringAttrTypedGet(type, toMlirStringRef(value));
          return PyStringAttribute(type.getContext(), attr);
        },
        py::arg("type"), py::arg("value"),
        "Gets a uniqued string attribute associated to a type");
    c.def_property_readonly(
        "value",
        [](PyStringAttribute &self) {
          MlirStringRef stringRef = mlirStringAttrGetValue(self);
          return py::str(stringRef.data, stringRef.length);
        },
        "Returns the value of the string attribute");
    c.def_property_readonly(
        "value_bytes",
        [](PyStringAttribute &self) {
          MlirStringRef stringRef = mlirStringAttrGetValue(self);
          return py::bytes(stringRef.data, stringRef.length);
        },
        "Returns the value of the string attribute as `bytes`");
  }
};

// TODO: Support construction of string elements.
class PyDenseElementsAttribute
    : public PyConcreteAttribute<PyDenseElementsAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsADenseElements;
  static constexpr const char *pyClassName = "DenseElementsAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static PyDenseElementsAttribute
  getFromBuffer(py::buffer array, bool signless,
                std::optional<PyType> explicitType,
                std::optional<std::vector<int64_t>> explicitShape,
                DefaultingPyMlirContext contextWrapper) {
    // Request a contiguous view. In exotic cases, this will cause a copy.
    int flags = PyBUF_ND;
    if (!explicitType) {
      flags |= PyBUF_FORMAT;
    }
    Py_buffer view;
    if (PyObject_GetBuffer(array.ptr(), &view, flags) != 0) {
      throw py::error_already_set();
    }
    auto freeBuffer = llvm::make_scope_exit([&]() { PyBuffer_Release(&view); });
    SmallVector<int64_t> shape;
    if (explicitShape) {
      shape.append(explicitShape->begin(), explicitShape->end());
    } else {
      shape.append(view.shape, view.shape + view.ndim);
    }

    MlirAttribute encodingAttr = mlirAttributeGetNull();
    MlirContext context = contextWrapper->get();

    // Detect format codes that are suitable for bulk loading. This includes
    // all byte aligned integer and floating point types up to 8 bytes.
    // Notably, this excludes, bool (which needs to be bit-packed) and
    // other exotics which do not have a direct representation in the buffer
    // protocol (i.e. complex, etc).
    std::optional<MlirType> bulkLoadElementType;
    if (explicitType) {
      bulkLoadElementType = *explicitType;
    } else {
      std::string_view format(view.format);
      if (format == "f") {
        // f32
        assert(view.itemsize == 4 && "mismatched array itemsize");
        bulkLoadElementType = mlirF32TypeGet(context);
      } else if (format == "d") {
        // f64
        assert(view.itemsize == 8 && "mismatched array itemsize");
        bulkLoadElementType = mlirF64TypeGet(context);
      } else if (format == "e") {
        // f16
        assert(view.itemsize == 2 && "mismatched array itemsize");
        bulkLoadElementType = mlirF16TypeGet(context);
      } else if (isSignedIntegerFormat(format)) {
        if (view.itemsize == 4) {
          // i32
          bulkLoadElementType = signless
                                    ? mlirIntegerTypeGet(context, 32)
                                    : mlirIntegerTypeSignedGet(context, 32);
        } else if (view.itemsize == 8) {
          // i64
          bulkLoadElementType = signless
                                    ? mlirIntegerTypeGet(context, 64)
                                    : mlirIntegerTypeSignedGet(context, 64);
        } else if (view.itemsize == 1) {
          // i8
          bulkLoadElementType = signless ? mlirIntegerTypeGet(context, 8)
                                         : mlirIntegerTypeSignedGet(context, 8);
        } else if (view.itemsize == 2) {
          // i16
          bulkLoadElementType = signless
                                    ? mlirIntegerTypeGet(context, 16)
                                    : mlirIntegerTypeSignedGet(context, 16);
        }
      } else if (isUnsignedIntegerFormat(format)) {
        if (view.itemsize == 4) {
          // unsigned i32
          bulkLoadElementType = signless
                                    ? mlirIntegerTypeGet(context, 32)
                                    : mlirIntegerTypeUnsignedGet(context, 32);
        } else if (view.itemsize == 8) {
          // unsigned i64
          bulkLoadElementType = signless
                                    ? mlirIntegerTypeGet(context, 64)
                                    : mlirIntegerTypeUnsignedGet(context, 64);
        } else if (view.itemsize == 1) {
          // i8
          bulkLoadElementType = signless
                                    ? mlirIntegerTypeGet(context, 8)
                                    : mlirIntegerTypeUnsignedGet(context, 8);
        } else if (view.itemsize == 2) {
          // i16
          bulkLoadElementType = signless
                                    ? mlirIntegerTypeGet(context, 16)
                                    : mlirIntegerTypeUnsignedGet(context, 16);
        }
      }
      if (!bulkLoadElementType) {
        throw std::invalid_argument(
            std::string("unimplemented array format conversion from format: ") +
            std::string(format));
      }
    }

    MlirType shapedType;
    if (mlirTypeIsAShaped(*bulkLoadElementType)) {
      if (explicitShape) {
        throw std::invalid_argument("Shape can only be specified explicitly "
                                    "when the type is not a shaped type.");
      }
      shapedType = *bulkLoadElementType;
    } else {
      shapedType = mlirRankedTensorTypeGet(shape.size(), shape.data(),
                                           *bulkLoadElementType, encodingAttr);
    }
    size_t rawBufferSize = view.len;
    MlirAttribute attr =
        mlirDenseElementsAttrRawBufferGet(shapedType, rawBufferSize, view.buf);
    if (mlirAttributeIsNull(attr)) {
      throw std::invalid_argument(
          "DenseElementsAttr could not be constructed from the given buffer. "
          "This may mean that the Python buffer layout does not match that "
          "MLIR expected layout and is a bug.");
    }
    return PyDenseElementsAttribute(contextWrapper->getRef(), attr);
  }

  static PyDenseElementsAttribute getSplat(const PyType &shapedType,
                                           PyAttribute &elementAttr) {
    auto contextWrapper =
        PyMlirContext::forContext(mlirTypeGetContext(shapedType));
    if (!mlirAttributeIsAInteger(elementAttr) &&
        !mlirAttributeIsAFloat(elementAttr)) {
      std::string message = "Illegal element type for DenseElementsAttr: ";
      message.append(py::repr(py::cast(elementAttr)));
      throw py::value_error(message);
    }
    if (!mlirTypeIsAShaped(shapedType) ||
        !mlirShapedTypeHasStaticShape(shapedType)) {
      std::string message =
          "Expected a static ShapedType for the shaped_type parameter: ";
      message.append(py::repr(py::cast(shapedType)));
      throw py::value_error(message);
    }
    MlirType shapedElementType = mlirShapedTypeGetElementType(shapedType);
    MlirType attrType = mlirAttributeGetType(elementAttr);
    if (!mlirTypeEqual(shapedElementType, attrType)) {
      std::string message =
          "Shaped element type and attribute type must be equal: shaped=";
      message.append(py::repr(py::cast(shapedType)));
      message.append(", element=");
      message.append(py::repr(py::cast(elementAttr)));
      throw py::value_error(message);
    }

    MlirAttribute elements =
        mlirDenseElementsAttrSplatGet(shapedType, elementAttr);
    return PyDenseElementsAttribute(contextWrapper->getRef(), elements);
  }

  intptr_t dunderLen() { return mlirElementsAttrGetNumElements(*this); }

  py::buffer_info accessBuffer() {
    MlirType shapedType = mlirAttributeGetType(*this);
    MlirType elementType = mlirShapedTypeGetElementType(shapedType);
    std::string format;

    if (mlirTypeIsAF32(elementType)) {
      // f32
      return bufferInfo<float>(shapedType);
    }
    if (mlirTypeIsAF64(elementType)) {
      // f64
      return bufferInfo<double>(shapedType);
    }
    if (mlirTypeIsAF16(elementType)) {
      // f16
      return bufferInfo<uint16_t>(shapedType, "e");
    }
    if (mlirTypeIsAIndex(elementType)) {
      // Same as IndexType::kInternalStorageBitWidth
      return bufferInfo<int64_t>(shapedType);
    }
    if (mlirTypeIsAInteger(elementType) &&
        mlirIntegerTypeGetWidth(elementType) == 32) {
      if (mlirIntegerTypeIsSignless(elementType) ||
          mlirIntegerTypeIsSigned(elementType)) {
        // i32
        return bufferInfo<int32_t>(shapedType);
      }
      if (mlirIntegerTypeIsUnsigned(elementType)) {
        // unsigned i32
        return bufferInfo<uint32_t>(shapedType);
      }
    } else if (mlirTypeIsAInteger(elementType) &&
               mlirIntegerTypeGetWidth(elementType) == 64) {
      if (mlirIntegerTypeIsSignless(elementType) ||
          mlirIntegerTypeIsSigned(elementType)) {
        // i64
        return bufferInfo<int64_t>(shapedType);
      }
      if (mlirIntegerTypeIsUnsigned(elementType)) {
        // unsigned i64
        return bufferInfo<uint64_t>(shapedType);
      }
    } else if (mlirTypeIsAInteger(elementType) &&
               mlirIntegerTypeGetWidth(elementType) == 8) {
      if (mlirIntegerTypeIsSignless(elementType) ||
          mlirIntegerTypeIsSigned(elementType)) {
        // i8
        return bufferInfo<int8_t>(shapedType);
      }
      if (mlirIntegerTypeIsUnsigned(elementType)) {
        // unsigned i8
        return bufferInfo<uint8_t>(shapedType);
      }
    } else if (mlirTypeIsAInteger(elementType) &&
               mlirIntegerTypeGetWidth(elementType) == 16) {
      if (mlirIntegerTypeIsSignless(elementType) ||
          mlirIntegerTypeIsSigned(elementType)) {
        // i16
        return bufferInfo<int16_t>(shapedType);
      }
      if (mlirIntegerTypeIsUnsigned(elementType)) {
        // unsigned i16
        return bufferInfo<uint16_t>(shapedType);
      }
    }

    // TODO: Currently crashes the program.
    // Reported as https://github.com/pybind/pybind11/issues/3336
    throw std::invalid_argument(
        "unsupported data type for conversion to Python buffer");
  }

  static void bindDerived(ClassTy &c) {
    c.def("__len__", &PyDenseElementsAttribute::dunderLen)
        .def_static("get", PyDenseElementsAttribute::getFromBuffer,
                    py::arg("array"), py::arg("signless") = true,
                    py::arg("type") = py::none(), py::arg("shape") = py::none(),
                    py::arg("context") = py::none(),
                    kDenseElementsAttrGetDocstring)
        .def_static("get_splat", PyDenseElementsAttribute::getSplat,
                    py::arg("shaped_type"), py::arg("element_attr"),
                    "Gets a DenseElementsAttr where all values are the same")
        .def_property_readonly("is_splat",
                               [](PyDenseElementsAttribute &self) -> bool {
                                 return mlirDenseElementsAttrIsSplat(self);
                               })
        .def("get_splat_value",
             [](PyDenseElementsAttribute &self) {
               if (!mlirDenseElementsAttrIsSplat(self))
                 throw py::value_error(
                     "get_splat_value called on a non-splat attribute");
               return mlirDenseElementsAttrGetSplatValue(self);
             })
        .def_buffer(&PyDenseElementsAttribute::accessBuffer);
  }

private:
  static bool isUnsignedIntegerFormat(std::string_view format) {
    if (format.empty())
      return false;
    char code = format[0];
    return code == 'I' || code == 'B' || code == 'H' || code == 'L' ||
           code == 'Q';
  }

  static bool isSignedIntegerFormat(std::string_view format) {
    if (format.empty())
      return false;
    char code = format[0];
    return code == 'i' || code == 'b' || code == 'h' || code == 'l' ||
           code == 'q';
  }

  template <typename Type>
  py::buffer_info bufferInfo(MlirType shapedType,
                             const char *explicitFormat = nullptr) {
    intptr_t rank = mlirShapedTypeGetRank(shapedType);
    // Prepare the data for the buffer_info.
    // Buffer is configured for read-only access below.
    Type *data = static_cast<Type *>(
        const_cast<void *>(mlirDenseElementsAttrGetRawData(*this)));
    // Prepare the shape for the buffer_info.
    SmallVector<intptr_t, 4> shape;
    for (intptr_t i = 0; i < rank; ++i)
      shape.push_back(mlirShapedTypeGetDimSize(shapedType, i));
    // Prepare the strides for the buffer_info.
    SmallVector<intptr_t, 4> strides;
    if (mlirDenseElementsAttrIsSplat(*this)) {
      // Splats are special, only the single value is stored.
      strides.assign(rank, 0);
    } else {
      for (intptr_t i = 1; i < rank; ++i) {
        intptr_t strideFactor = 1;
        for (intptr_t j = i; j < rank; ++j)
          strideFactor *= mlirShapedTypeGetDimSize(shapedType, j);
        strides.push_back(sizeof(Type) * strideFactor);
      }
      strides.push_back(sizeof(Type));
    }
    std::string format;
    if (explicitFormat) {
      format = explicitFormat;
    } else {
      format = py::format_descriptor<Type>::format();
    }
    return py::buffer_info(data, sizeof(Type), format, rank, shape, strides,
                           /*readonly=*/true);
  }
}; // namespace

/// Refinement of the PyDenseElementsAttribute for attributes containing integer
/// (and boolean) values. Supports element access.
class PyDenseIntElementsAttribute
    : public PyConcreteAttribute<PyDenseIntElementsAttribute,
                                 PyDenseElementsAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsADenseIntElements;
  static constexpr const char *pyClassName = "DenseIntElementsAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  /// Returns the element at the given linear position. Asserts if the index is
  /// out of range.
  py::int_ dunderGetItem(intptr_t pos) {
    if (pos < 0 || pos >= dunderLen()) {
      throw py::index_error("attempt to access out of bounds element");
    }

    MlirType type = mlirAttributeGetType(*this);
    type = mlirShapedTypeGetElementType(type);
    assert(mlirTypeIsAInteger(type) &&
           "expected integer element type in dense int elements attribute");
    // Dispatch element extraction to an appropriate C function based on the
    // elemental type of the attribute. py::int_ is implicitly constructible
    // from any C++ integral type and handles bitwidth correctly.
    // TODO: consider caching the type properties in the constructor to avoid
    // querying them on each element access.
    unsigned width = mlirIntegerTypeGetWidth(type);
    bool isUnsigned = mlirIntegerTypeIsUnsigned(type);
    if (isUnsigned) {
      if (width == 1) {
        return mlirDenseElementsAttrGetBoolValue(*this, pos);
      }
      if (width == 8) {
        return mlirDenseElementsAttrGetUInt8Value(*this, pos);
      }
      if (width == 16) {
        return mlirDenseElementsAttrGetUInt16Value(*this, pos);
      }
      if (width == 32) {
        return mlirDenseElementsAttrGetUInt32Value(*this, pos);
      }
      if (width == 64) {
        return mlirDenseElementsAttrGetUInt64Value(*this, pos);
      }
    } else {
      if (width == 1) {
        return mlirDenseElementsAttrGetBoolValue(*this, pos);
      }
      if (width == 8) {
        return mlirDenseElementsAttrGetInt8Value(*this, pos);
      }
      if (width == 16) {
        return mlirDenseElementsAttrGetInt16Value(*this, pos);
      }
      if (width == 32) {
        return mlirDenseElementsAttrGetInt32Value(*this, pos);
      }
      if (width == 64) {
        return mlirDenseElementsAttrGetInt64Value(*this, pos);
      }
    }
    throw py::type_error("Unsupported integer type");
  }

  static void bindDerived(ClassTy &c) {
    c.def("__getitem__", &PyDenseIntElementsAttribute::dunderGetItem);
  }
};

class PyDictAttribute : public PyConcreteAttribute<PyDictAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsADictionary;
  static constexpr const char *pyClassName = "DictAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirDictionaryAttrGetTypeID;

  intptr_t dunderLen() { return mlirDictionaryAttrGetNumElements(*this); }

  bool dunderContains(const std::string &name) {
    return !mlirAttributeIsNull(
        mlirDictionaryAttrGetElementByName(*this, toMlirStringRef(name)));
  }

  static void bindDerived(ClassTy &c) {
    c.def("__contains__", &PyDictAttribute::dunderContains);
    c.def("__len__", &PyDictAttribute::dunderLen);
    c.def_static(
        "get",
        [](py::dict attributes, DefaultingPyMlirContext context) {
          SmallVector<MlirNamedAttribute> mlirNamedAttributes;
          mlirNamedAttributes.reserve(attributes.size());
          for (auto &it : attributes) {
            auto &mlirAttr = it.second.cast<PyAttribute &>();
            auto name = it.first.cast<std::string>();
            mlirNamedAttributes.push_back(mlirNamedAttributeGet(
                mlirIdentifierGet(mlirAttributeGetContext(mlirAttr),
                                  toMlirStringRef(name)),
                mlirAttr));
          }
          MlirAttribute attr =
              mlirDictionaryAttrGet(context->get(), mlirNamedAttributes.size(),
                                    mlirNamedAttributes.data());
          return PyDictAttribute(context->getRef(), attr);
        },
        py::arg("value") = py::dict(), py::arg("context") = py::none(),
        "Gets an uniqued dict attribute");
    c.def("__getitem__", [](PyDictAttribute &self, const std::string &name) {
      MlirAttribute attr =
          mlirDictionaryAttrGetElementByName(self, toMlirStringRef(name));
      if (mlirAttributeIsNull(attr))
        throw py::key_error("attempt to access a non-existent attribute");
      return attr;
    });
    c.def("__getitem__", [](PyDictAttribute &self, intptr_t index) {
      if (index < 0 || index >= self.dunderLen()) {
        throw py::index_error("attempt to access out of bounds attribute");
      }
      MlirNamedAttribute namedAttr = mlirDictionaryAttrGetElement(self, index);
      return PyNamedAttribute(
          namedAttr.attribute,
          std::string(mlirIdentifierStr(namedAttr.name).data));
    });
  }
};

/// Refinement of PyDenseElementsAttribute for attributes containing
/// floating-point values. Supports element access.
class PyDenseFPElementsAttribute
    : public PyConcreteAttribute<PyDenseFPElementsAttribute,
                                 PyDenseElementsAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsADenseFPElements;
  static constexpr const char *pyClassName = "DenseFPElementsAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  py::float_ dunderGetItem(intptr_t pos) {
    if (pos < 0 || pos >= dunderLen()) {
      throw py::index_error("attempt to access out of bounds element");
    }

    MlirType type = mlirAttributeGetType(*this);
    type = mlirShapedTypeGetElementType(type);
    // Dispatch element extraction to an appropriate C function based on the
    // elemental type of the attribute. py::float_ is implicitly constructible
    // from float and double.
    // TODO: consider caching the type properties in the constructor to avoid
    // querying them on each element access.
    if (mlirTypeIsAF32(type)) {
      return mlirDenseElementsAttrGetFloatValue(*this, pos);
    }
    if (mlirTypeIsAF64(type)) {
      return mlirDenseElementsAttrGetDoubleValue(*this, pos);
    }
    throw py::type_error("Unsupported floating-point type");
  }

  static void bindDerived(ClassTy &c) {
    c.def("__getitem__", &PyDenseFPElementsAttribute::dunderGetItem);
  }
};

class PyTypeAttribute : public PyConcreteAttribute<PyTypeAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAType;
  static constexpr const char *pyClassName = "TypeAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirTypeAttrGetTypeID;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyType value, DefaultingPyMlirContext context) {
          MlirAttribute attr = mlirTypeAttrGet(value.get());
          return PyTypeAttribute(context->getRef(), attr);
        },
        py::arg("value"), py::arg("context") = py::none(),
        "Gets a uniqued Type attribute");
    c.def_property_readonly("value", [](PyTypeAttribute &self) {
      return mlirTypeAttrGetValue(self.get());
    });
  }
};

/// Unit Attribute subclass. Unit attributes don't have values.
class PyUnitAttribute : public PyConcreteAttribute<PyUnitAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAUnit;
  static constexpr const char *pyClassName = "UnitAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirUnitAttrGetTypeID;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          return PyUnitAttribute(context->getRef(),
                                 mlirUnitAttrGet(context->get()));
        },
        py::arg("context") = py::none(), "Create a Unit attribute.");
  }
};

/// Strided layout attribute subclass.
class PyStridedLayoutAttribute
    : public PyConcreteAttribute<PyStridedLayoutAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAStridedLayout;
  static constexpr const char *pyClassName = "StridedLayoutAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirStridedLayoutAttrGetTypeID;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int64_t offset, const std::vector<int64_t> strides,
           DefaultingPyMlirContext ctx) {
          MlirAttribute attr = mlirStridedLayoutAttrGet(
              ctx->get(), offset, strides.size(), strides.data());
          return PyStridedLayoutAttribute(ctx->getRef(), attr);
        },
        py::arg("offset"), py::arg("strides"), py::arg("context") = py::none(),
        "Gets a strided layout attribute.");
    c.def_static(
        "get_fully_dynamic",
        [](int64_t rank, DefaultingPyMlirContext ctx) {
          auto dynamic = mlirShapedTypeGetDynamicStrideOrOffset();
          std::vector<int64_t> strides(rank);
          std::fill(strides.begin(), strides.end(), dynamic);
          MlirAttribute attr = mlirStridedLayoutAttrGet(
              ctx->get(), dynamic, strides.size(), strides.data());
          return PyStridedLayoutAttribute(ctx->getRef(), attr);
        },
        py::arg("rank"), py::arg("context") = py::none(),
        "Gets a strided layout attribute with dynamic offset and strides of a "
        "given rank.");
    c.def_property_readonly(
        "offset",
        [](PyStridedLayoutAttribute &self) {
          return mlirStridedLayoutAttrGetOffset(self);
        },
        "Returns the value of the float point attribute");
    c.def_property_readonly(
        "strides",
        [](PyStridedLayoutAttribute &self) {
          intptr_t size = mlirStridedLayoutAttrGetNumStrides(self);
          std::vector<int64_t> strides(size);
          for (intptr_t i = 0; i < size; i++) {
            strides[i] = mlirStridedLayoutAttrGetStride(self, i);
          }
          return strides;
        },
        "Returns the value of the float point attribute");
  }
};

py::object denseArrayAttributeCaster(PyAttribute &pyAttribute) {
  if (PyDenseBoolArrayAttribute::isaFunction(pyAttribute))
    return py::cast(PyDenseBoolArrayAttribute(pyAttribute));
  if (PyDenseI8ArrayAttribute::isaFunction(pyAttribute))
    return py::cast(PyDenseI8ArrayAttribute(pyAttribute));
  if (PyDenseI16ArrayAttribute::isaFunction(pyAttribute))
    return py::cast(PyDenseI16ArrayAttribute(pyAttribute));
  if (PyDenseI32ArrayAttribute::isaFunction(pyAttribute))
    return py::cast(PyDenseI32ArrayAttribute(pyAttribute));
  if (PyDenseI64ArrayAttribute::isaFunction(pyAttribute))
    return py::cast(PyDenseI64ArrayAttribute(pyAttribute));
  if (PyDenseF32ArrayAttribute::isaFunction(pyAttribute))
    return py::cast(PyDenseF32ArrayAttribute(pyAttribute));
  if (PyDenseF64ArrayAttribute::isaFunction(pyAttribute))
    return py::cast(PyDenseF64ArrayAttribute(pyAttribute));
  std::string msg =
      std::string("Can't cast unknown element type DenseArrayAttr (") +
      std::string(py::repr(py::cast(pyAttribute))) + ")";
  throw py::cast_error(msg);
}

py::object denseIntOrFPElementsAttributeCaster(PyAttribute &pyAttribute) {
  if (PyDenseFPElementsAttribute::isaFunction(pyAttribute))
    return py::cast(PyDenseFPElementsAttribute(pyAttribute));
  if (PyDenseIntElementsAttribute::isaFunction(pyAttribute))
    return py::cast(PyDenseIntElementsAttribute(pyAttribute));
  std::string msg =
      std::string(
          "Can't cast unknown element type DenseIntOrFPElementsAttr (") +
      std::string(py::repr(py::cast(pyAttribute))) + ")";
  throw py::cast_error(msg);
}

py::object integerOrBoolAttributeCaster(PyAttribute &pyAttribute) {
  if (PyBoolAttribute::isaFunction(pyAttribute))
    return py::cast(PyBoolAttribute(pyAttribute));
  if (PyIntegerAttribute::isaFunction(pyAttribute))
    return py::cast(PyIntegerAttribute(pyAttribute));
  std::string msg =
      std::string("Can't cast unknown element type DenseArrayAttr (") +
      std::string(py::repr(py::cast(pyAttribute))) + ")";
  throw py::cast_error(msg);
}

py::object symbolRefOrFlatSymbolRefAttributeCaster(PyAttribute &pyAttribute) {
  if (PyFlatSymbolRefAttribute::isaFunction(pyAttribute))
    return py::cast(PyFlatSymbolRefAttribute(pyAttribute));
  if (PySymbolRefAttribute::isaFunction(pyAttribute))
    return py::cast(PySymbolRefAttribute(pyAttribute));
  std::string msg = std::string("Can't cast unknown SymbolRef attribute (") +
                    std::string(py::repr(py::cast(pyAttribute))) + ")";
  throw py::cast_error(msg);
}

} // namespace

void mlir::python::populateIRAttributes(py::module &m) {
  PyAffineMapAttribute::bind(m);

  PyDenseBoolArrayAttribute::bind(m);
  PyDenseBoolArrayAttribute::PyDenseArrayIterator::bind(m);
  PyDenseI8ArrayAttribute::bind(m);
  PyDenseI8ArrayAttribute::PyDenseArrayIterator::bind(m);
  PyDenseI16ArrayAttribute::bind(m);
  PyDenseI16ArrayAttribute::PyDenseArrayIterator::bind(m);
  PyDenseI32ArrayAttribute::bind(m);
  PyDenseI32ArrayAttribute::PyDenseArrayIterator::bind(m);
  PyDenseI64ArrayAttribute::bind(m);
  PyDenseI64ArrayAttribute::PyDenseArrayIterator::bind(m);
  PyDenseF32ArrayAttribute::bind(m);
  PyDenseF32ArrayAttribute::PyDenseArrayIterator::bind(m);
  PyDenseF64ArrayAttribute::bind(m);
  PyDenseF64ArrayAttribute::PyDenseArrayIterator::bind(m);
  PyGlobals::get().registerTypeCaster(
      mlirDenseArrayAttrGetTypeID(),
      pybind11::cpp_function(denseArrayAttributeCaster));

  PyArrayAttribute::bind(m);
  PyArrayAttribute::PyArrayAttributeIterator::bind(m);
  PyBoolAttribute::bind(m);
  PyDenseElementsAttribute::bind(m);
  PyDenseFPElementsAttribute::bind(m);
  PyDenseIntElementsAttribute::bind(m);
  PyGlobals::get().registerTypeCaster(
      mlirDenseIntOrFPElementsAttrGetTypeID(),
      pybind11::cpp_function(denseIntOrFPElementsAttributeCaster));

  PyDictAttribute::bind(m);
  PySymbolRefAttribute::bind(m);
  PyGlobals::get().registerTypeCaster(
      mlirSymbolRefAttrGetTypeID(),
      pybind11::cpp_function(symbolRefOrFlatSymbolRefAttributeCaster));

  PyFlatSymbolRefAttribute::bind(m);
  PyOpaqueAttribute::bind(m);
  PyFloatAttribute::bind(m);
  PyIntegerAttribute::bind(m);
  PyStringAttribute::bind(m);
  PyTypeAttribute::bind(m);
  PyGlobals::get().registerTypeCaster(
      mlirIntegerAttrGetTypeID(),
      pybind11::cpp_function(integerOrBoolAttributeCaster));
  PyUnitAttribute::bind(m);

  PyStridedLayoutAttribute::bind(m);
}
