//===- IRAttributes.cpp - Exports builtin and standard attributes ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "IRModule.h"
#include "NanobindUtils.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/raw_ostream.h"

namespace nb = nanobind;
using namespace nanobind::literals;
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

static const char kDenseElementsAttrGetFromListDocstring[] =
    R"(Gets a DenseElementsAttr from a Python list of attributes.

Note that it can be expensive to construct attributes individually.
For a large number of elements, consider using a Python buffer or array instead.

Args:
  attrs: A list of attributes.
  type: The desired shape and type of the resulting DenseElementsAttr.
    If not provided, the element type is determined based on the type
    of the 0th attribute and the shape is `[len(attrs)]`.
  context: Explicit context, if not from context manager.

Returns:
  DenseElementsAttr on success.

Raises:
  ValueError: If the type of the attributes does not match the type
    specified by `shaped_type`.
)";

static const char kDenseResourceElementsAttrGetFromBufferDocstring[] =
    R"(Gets a DenseResourceElementsAttr from a Python buffer or array.

This function does minimal validation or massaging of the data, and it is
up to the caller to ensure that the buffer meets the characteristics
implied by the shape.

The backing buffer and any user objects will be retained for the lifetime
of the resource blob. This is typically bounded to the context but the
resource can have a shorter lifespan depending on how it is used in
subsequent processing.

Args:
  buffer: The array or buffer to convert.
  name: Name to provide to the resource (may be changed upon collision).
  type: The explicit ShapedType to construct the attribute with.
  context: Explicit context, if not from context manager.

Returns:
  DenseResourceElementsAttr on success.

Raises:
  ValueError: If the type of the buffer or array cannot be matched to an MLIR
    type or if the buffer does not meet expectations.
)";

namespace {

struct nb_buffer_info {
  void *ptr = nullptr;
  ssize_t itemsize = 0;
  ssize_t size = 0;
  const char *format = nullptr;
  ssize_t ndim = 0;
  SmallVector<ssize_t, 4> shape;
  SmallVector<ssize_t, 4> strides;
  bool readonly = false;

  nb_buffer_info(
      void *ptr, ssize_t itemsize, const char *format, ssize_t ndim,
      SmallVector<ssize_t, 4> shape_in, SmallVector<ssize_t, 4> strides_in,
      bool readonly = false,
      std::unique_ptr<Py_buffer, void (*)(Py_buffer *)> owned_view_in =
          std::unique_ptr<Py_buffer, void (*)(Py_buffer *)>(nullptr, nullptr))
      : ptr(ptr), itemsize(itemsize), format(format), ndim(ndim),
        shape(std::move(shape_in)), strides(std::move(strides_in)),
        readonly(readonly), owned_view(std::move(owned_view_in)) {
    size = 1;
    for (ssize_t i = 0; i < ndim; ++i) {
      size *= shape[i];
    }
  }

  explicit nb_buffer_info(Py_buffer *view)
      : nb_buffer_info(view->buf, view->itemsize, view->format, view->ndim,
                       {view->shape, view->shape + view->ndim},
                       // TODO(phawkins): check for null strides
                       {view->strides, view->strides + view->ndim},
                       view->readonly != 0,
                       std::unique_ptr<Py_buffer, void (*)(Py_buffer *)>(
                           view, PyBuffer_Release)) {}

  nb_buffer_info(const nb_buffer_info &) = delete;
  nb_buffer_info(nb_buffer_info &&) = default;
  nb_buffer_info &operator=(const nb_buffer_info &) = delete;
  nb_buffer_info &operator=(nb_buffer_info &&) = default;

private:
  std::unique_ptr<Py_buffer, void (*)(Py_buffer *)> owned_view;
};

class nb_buffer : public nb::object {
  NB_OBJECT_DEFAULT(nb_buffer, object, "buffer", PyObject_CheckBuffer);

  nb_buffer_info request() const {
    int flags = PyBUF_STRIDES | PyBUF_FORMAT;
    auto *view = new Py_buffer();
    if (PyObject_GetBuffer(ptr(), view, flags) != 0) {
      delete view;
      throw nb::python_error();
    }
    return nb_buffer_info(view);
  }
};

template <typename T>
struct nb_format_descriptor {};

template <>
struct nb_format_descriptor<bool> {
  static const char *format() { return "?"; }
};
template <>
struct nb_format_descriptor<int8_t> {
  static const char *format() { return "b"; }
};
template <>
struct nb_format_descriptor<uint8_t> {
  static const char *format() { return "B"; }
};
template <>
struct nb_format_descriptor<int16_t> {
  static const char *format() { return "h"; }
};
template <>
struct nb_format_descriptor<uint16_t> {
  static const char *format() { return "H"; }
};
template <>
struct nb_format_descriptor<int32_t> {
  static const char *format() { return "i"; }
};
template <>
struct nb_format_descriptor<uint32_t> {
  static const char *format() { return "I"; }
};
template <>
struct nb_format_descriptor<int64_t> {
  static const char *format() { return "q"; }
};
template <>
struct nb_format_descriptor<uint64_t> {
  static const char *format() { return "Q"; }
};
template <>
struct nb_format_descriptor<float> {
  static const char *format() { return "f"; }
};
template <>
struct nb_format_descriptor<double> {
  static const char *format() { return "d"; }
};

static MlirStringRef toMlirStringRef(const std::string &s) {
  return mlirStringRefCreate(s.data(), s.size());
}

static MlirStringRef toMlirStringRef(const nb::bytes &s) {
  return mlirStringRefCreate(static_cast<const char *>(s.data()), s.size());
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
        nb::arg("affine_map"), "Gets an attribute wrapping an AffineMap.");
    c.def_prop_ro("value", mlirAffineMapAttrGetValue,
                  "Returns the value of the AffineMap attribute");
  }
};

class PyIntegerSetAttribute
    : public PyConcreteAttribute<PyIntegerSetAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAIntegerSet;
  static constexpr const char *pyClassName = "IntegerSetAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirIntegerSetAttrGetTypeID;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyIntegerSet &integerSet) {
          MlirAttribute attr = mlirIntegerSetAttrGet(integerSet.get());
          return PyIntegerSetAttribute(integerSet.getContext(), attr);
        },
        nb::arg("integer_set"), "Gets an attribute wrapping an IntegerSet.");
  }
};

template <typename T>
static T pyTryCast(nb::handle object) {
  try {
    return nb::cast<T>(object);
  } catch (nb::cast_error &err) {
    std::string msg = std::string("Invalid attribute when attempting to "
                                  "create an ArrayAttribute (") +
                      err.what() + ")";
    throw std::runtime_error(msg.c_str());
  } catch (std::runtime_error &err) {
    std::string msg = std::string("Invalid attribute (None?) when attempting "
                                  "to create an ArrayAttribute (") +
                      err.what() + ")";
    throw std::runtime_error(msg.c_str());
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
        throw nb::stop_iteration();
      return DerivedT::getElement(attr.get(), nextIndex++);
    }

    /// Bind the iterator class.
    static void bind(nb::module_ &m) {
      nb::class_<PyDenseArrayIterator>(m, DerivedT::pyIteratorName)
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
    if constexpr (std::is_same_v<EltTy, bool>) {
      c.def_static(
          "get",
          [](const nb::sequence &py_values, DefaultingPyMlirContext ctx) {
            std::vector<bool> values;
            for (nb::handle py_value : py_values) {
              int is_true = PyObject_IsTrue(py_value.ptr());
              if (is_true < 0) {
                throw nb::python_error();
              }
              values.push_back(is_true);
            }
            return getAttribute(values, ctx->getRef());
          },
          nb::arg("values"), nb::arg("context").none() = nb::none(),
          "Gets a uniqued dense array attribute");
    } else {
      c.def_static(
          "get",
          [](const std::vector<EltTy> &values, DefaultingPyMlirContext ctx) {
            return getAttribute(values, ctx->getRef());
          },
          nb::arg("values"), nb::arg("context").none() = nb::none(),
          "Gets a uniqued dense array attribute");
    }
    // Bind the array methods.
    c.def("__getitem__", [](DerivedT &arr, intptr_t i) {
      if (i >= mlirDenseArrayGetNumElements(arr))
        throw nb::index_error("DenseArray index out of range");
      return arr.getItem(i);
    });
    c.def("__len__", [](const DerivedT &arr) {
      return mlirDenseArrayGetNumElements(arr);
    });
    c.def("__iter__",
          [](const DerivedT &arr) { return PyDenseArrayIterator(arr); });
    c.def("__add__", [](DerivedT &arr, const nb::list &extras) {
      std::vector<EltTy> values;
      intptr_t numOldElements = mlirDenseArrayGetNumElements(arr);
      values.reserve(numOldElements + nb::len(extras));
      for (intptr_t i = 0; i < numOldElements; ++i)
        values.push_back(arr.getItem(i));
      for (nb::handle attr : extras)
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
        throw nb::stop_iteration();
      return mlirArrayAttrGetElement(attr.get(), nextIndex++);
    }

    static void bind(nb::module_ &m) {
      nb::class_<PyArrayAttributeIterator>(m, "ArrayAttributeIterator")
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
        [](const nb::list &attributes, DefaultingPyMlirContext context) {
          SmallVector<MlirAttribute> mlirAttributes;
          mlirAttributes.reserve(nb::len(attributes));
          for (auto attribute : attributes) {
            mlirAttributes.push_back(pyTryCast<PyAttribute>(attribute));
          }
          MlirAttribute attr = mlirArrayAttrGet(
              context->get(), mlirAttributes.size(), mlirAttributes.data());
          return PyArrayAttribute(context->getRef(), attr);
        },
        nb::arg("attributes"), nb::arg("context").none() = nb::none(),
        "Gets a uniqued Array attribute");
    c.def("__getitem__",
          [](PyArrayAttribute &arr, intptr_t i) {
            if (i >= mlirArrayAttrGetNumElements(arr))
              throw nb::index_error("ArrayAttribute index out of range");
            return arr.getItem(i);
          })
        .def("__len__",
             [](const PyArrayAttribute &arr) {
               return mlirArrayAttrGetNumElements(arr);
             })
        .def("__iter__", [](const PyArrayAttribute &arr) {
          return PyArrayAttributeIterator(arr);
        });
    c.def("__add__", [](PyArrayAttribute arr, const nb::list &extras) {
      std::vector<MlirAttribute> attributes;
      intptr_t numOldElements = mlirArrayAttrGetNumElements(arr);
      attributes.reserve(numOldElements + nb::len(extras));
      for (intptr_t i = 0; i < numOldElements; ++i)
        attributes.push_back(arr.getItem(i));
      for (nb::handle attr : extras)
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
        nb::arg("type"), nb::arg("value"), nb::arg("loc").none() = nb::none(),
        "Gets an uniqued float point attribute associated to a type");
    c.def_static(
        "get_f32",
        [](double value, DefaultingPyMlirContext context) {
          MlirAttribute attr = mlirFloatAttrDoubleGet(
              context->get(), mlirF32TypeGet(context->get()), value);
          return PyFloatAttribute(context->getRef(), attr);
        },
        nb::arg("value"), nb::arg("context").none() = nb::none(),
        "Gets an uniqued float point attribute associated to a f32 type");
    c.def_static(
        "get_f64",
        [](double value, DefaultingPyMlirContext context) {
          MlirAttribute attr = mlirFloatAttrDoubleGet(
              context->get(), mlirF64TypeGet(context->get()), value);
          return PyFloatAttribute(context->getRef(), attr);
        },
        nb::arg("value"), nb::arg("context").none() = nb::none(),
        "Gets an uniqued float point attribute associated to a f64 type");
    c.def_prop_ro("value", mlirFloatAttrGetValueDouble,
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
        nb::arg("type"), nb::arg("value"),
        "Gets an uniqued integer attribute associated to a type");
    c.def_prop_ro("value", toPyInt,
                  "Returns the value of the integer attribute");
    c.def("__int__", toPyInt,
          "Converts the value of the integer attribute to a Python int");
    c.def_prop_ro_static("static_typeid",
                         [](nb::object & /*class*/) -> MlirTypeID {
                           return mlirIntegerAttrGetTypeID();
                         });
  }

private:
  static int64_t toPyInt(PyIntegerAttribute &self) {
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
        nb::arg("value"), nb::arg("context").none() = nb::none(),
        "Gets an uniqued bool attribute");
    c.def_prop_ro("value", mlirBoolAttrGetValue,
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
        nb::arg("symbols"), nb::arg("context").none() = nb::none(),
        "Gets a uniqued SymbolRef attribute from a list of symbol names");
    c.def_prop_ro(
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
        [](const std::string &value, DefaultingPyMlirContext context) {
          MlirAttribute attr =
              mlirFlatSymbolRefAttrGet(context->get(), toMlirStringRef(value));
          return PyFlatSymbolRefAttribute(context->getRef(), attr);
        },
        nb::arg("value"), nb::arg("context").none() = nb::none(),
        "Gets a uniqued FlatSymbolRef attribute");
    c.def_prop_ro(
        "value",
        [](PyFlatSymbolRefAttribute &self) {
          MlirStringRef stringRef = mlirFlatSymbolRefAttrGetValue(self);
          return nb::str(stringRef.data, stringRef.length);
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
        [](const std::string &dialectNamespace, const nb_buffer &buffer,
           PyType &type, DefaultingPyMlirContext context) {
          const nb_buffer_info bufferInfo = buffer.request();
          intptr_t bufferSize = bufferInfo.size;
          MlirAttribute attr = mlirOpaqueAttrGet(
              context->get(), toMlirStringRef(dialectNamespace), bufferSize,
              static_cast<char *>(bufferInfo.ptr), type);
          return PyOpaqueAttribute(context->getRef(), attr);
        },
        nb::arg("dialect_namespace"), nb::arg("buffer"), nb::arg("type"),
        nb::arg("context").none() = nb::none(), "Gets an Opaque attribute.");
    c.def_prop_ro(
        "dialect_namespace",
        [](PyOpaqueAttribute &self) {
          MlirStringRef stringRef = mlirOpaqueAttrGetDialectNamespace(self);
          return nb::str(stringRef.data, stringRef.length);
        },
        "Returns the dialect namespace for the Opaque attribute as a string");
    c.def_prop_ro(
        "data",
        [](PyOpaqueAttribute &self) {
          MlirStringRef stringRef = mlirOpaqueAttrGetData(self);
          return nb::bytes(stringRef.data, stringRef.length);
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
        [](const std::string &value, DefaultingPyMlirContext context) {
          MlirAttribute attr =
              mlirStringAttrGet(context->get(), toMlirStringRef(value));
          return PyStringAttribute(context->getRef(), attr);
        },
        nb::arg("value"), nb::arg("context").none() = nb::none(),
        "Gets a uniqued string attribute");
    c.def_static(
        "get",
        [](const nb::bytes &value, DefaultingPyMlirContext context) {
          MlirAttribute attr =
              mlirStringAttrGet(context->get(), toMlirStringRef(value));
          return PyStringAttribute(context->getRef(), attr);
        },
        nb::arg("value"), nb::arg("context").none() = nb::none(),
        "Gets a uniqued string attribute");
    c.def_static(
        "get_typed",
        [](PyType &type, const std::string &value) {
          MlirAttribute attr =
              mlirStringAttrTypedGet(type, toMlirStringRef(value));
          return PyStringAttribute(type.getContext(), attr);
        },
        nb::arg("type"), nb::arg("value"),
        "Gets a uniqued string attribute associated to a type");
    c.def_prop_ro(
        "value",
        [](PyStringAttribute &self) {
          MlirStringRef stringRef = mlirStringAttrGetValue(self);
          return nb::str(stringRef.data, stringRef.length);
        },
        "Returns the value of the string attribute");
    c.def_prop_ro(
        "value_bytes",
        [](PyStringAttribute &self) {
          MlirStringRef stringRef = mlirStringAttrGetValue(self);
          return nb::bytes(stringRef.data, stringRef.length);
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
  getFromList(const nb::list &attributes, std::optional<PyType> explicitType,
              DefaultingPyMlirContext contextWrapper) {
    const size_t numAttributes = nb::len(attributes);
    if (numAttributes == 0)
      throw nb::value_error("Attributes list must be non-empty.");

    MlirType shapedType;
    if (explicitType) {
      if ((!mlirTypeIsAShaped(*explicitType) ||
           !mlirShapedTypeHasStaticShape(*explicitType))) {

        std::string message;
        llvm::raw_string_ostream os(message);
        os << "Expected a static ShapedType for the shaped_type parameter: "
           << nb::cast<std::string>(nb::repr(nb::cast(*explicitType)));
        throw nb::value_error(message.c_str());
      }
      shapedType = *explicitType;
    } else {
      SmallVector<int64_t> shape = {static_cast<int64_t>(numAttributes)};
      shapedType = mlirRankedTensorTypeGet(
          shape.size(), shape.data(),
          mlirAttributeGetType(pyTryCast<PyAttribute>(attributes[0])),
          mlirAttributeGetNull());
    }

    SmallVector<MlirAttribute> mlirAttributes;
    mlirAttributes.reserve(numAttributes);
    for (const nb::handle &attribute : attributes) {
      MlirAttribute mlirAttribute = pyTryCast<PyAttribute>(attribute);
      MlirType attrType = mlirAttributeGetType(mlirAttribute);
      mlirAttributes.push_back(mlirAttribute);

      if (!mlirTypeEqual(mlirShapedTypeGetElementType(shapedType), attrType)) {
        std::string message;
        llvm::raw_string_ostream os(message);
        os << "All attributes must be of the same type and match "
           << "the type parameter: expected="
           << nb::cast<std::string>(nb::repr(nb::cast(shapedType)))
           << ", but got="
           << nb::cast<std::string>(nb::repr(nb::cast(attrType)));
        throw nb::value_error(message.c_str());
      }
    }

    MlirAttribute elements = mlirDenseElementsAttrGet(
        shapedType, mlirAttributes.size(), mlirAttributes.data());

    return PyDenseElementsAttribute(contextWrapper->getRef(), elements);
  }

  static PyDenseElementsAttribute
  getFromBuffer(const nb_buffer &array, bool signless,
                const std::optional<PyType> &explicitType,
                std::optional<std::vector<int64_t>> explicitShape,
                DefaultingPyMlirContext contextWrapper) {
    // Request a contiguous view. In exotic cases, this will cause a copy.
    int flags = PyBUF_ND;
    if (!explicitType) {
      flags |= PyBUF_FORMAT;
    }
    Py_buffer view;
    if (PyObject_GetBuffer(array.ptr(), &view, flags) != 0) {
      throw nb::python_error();
    }
    auto freeBuffer = llvm::make_scope_exit([&]() { PyBuffer_Release(&view); });

    MlirContext context = contextWrapper->get();
    MlirAttribute attr = getAttributeFromBuffer(
        view, signless, explicitType, std::move(explicitShape), context);
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
      message.append(nb::cast<std::string>(nb::repr(nb::cast(elementAttr))));
      throw nb::value_error(message.c_str());
    }
    if (!mlirTypeIsAShaped(shapedType) ||
        !mlirShapedTypeHasStaticShape(shapedType)) {
      std::string message =
          "Expected a static ShapedType for the shaped_type parameter: ";
      message.append(nb::cast<std::string>(nb::repr(nb::cast(shapedType))));
      throw nb::value_error(message.c_str());
    }
    MlirType shapedElementType = mlirShapedTypeGetElementType(shapedType);
    MlirType attrType = mlirAttributeGetType(elementAttr);
    if (!mlirTypeEqual(shapedElementType, attrType)) {
      std::string message =
          "Shaped element type and attribute type must be equal: shaped=";
      message.append(nb::cast<std::string>(nb::repr(nb::cast(shapedType))));
      message.append(", element=");
      message.append(nb::cast<std::string>(nb::repr(nb::cast(elementAttr))));
      throw nb::value_error(message.c_str());
    }

    MlirAttribute elements =
        mlirDenseElementsAttrSplatGet(shapedType, elementAttr);
    return PyDenseElementsAttribute(contextWrapper->getRef(), elements);
  }

  intptr_t dunderLen() { return mlirElementsAttrGetNumElements(*this); }

  std::unique_ptr<nb_buffer_info> accessBuffer() {
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
    } else if (mlirTypeIsAInteger(elementType) &&
               mlirIntegerTypeGetWidth(elementType) == 1) {
      // i1 / bool
      // We can not send the buffer directly back to Python, because the i1
      // values are bitpacked within MLIR. We call numpy's unpackbits function
      // to convert the bytes.
      return getBooleanBufferFromBitpackedAttribute();
    }

    // TODO: Currently crashes the program.
    // Reported as https://github.com/pybind/pybind11/issues/3336
    throw std::invalid_argument(
        "unsupported data type for conversion to Python buffer");
  }

  static void bindDerived(ClassTy &c) {
#if PY_VERSION_HEX < 0x03090000
    PyTypeObject *tp = reinterpret_cast<PyTypeObject *>(c.ptr());
    tp->tp_as_buffer->bf_getbuffer = PyDenseElementsAttribute::bf_getbuffer;
    tp->tp_as_buffer->bf_releasebuffer =
        PyDenseElementsAttribute::bf_releasebuffer;
#endif
    c.def("__len__", &PyDenseElementsAttribute::dunderLen)
        .def_static("get", PyDenseElementsAttribute::getFromBuffer,
                    nb::arg("array"), nb::arg("signless") = true,
                    nb::arg("type").none() = nb::none(),
                    nb::arg("shape").none() = nb::none(),
                    nb::arg("context").none() = nb::none(),
                    kDenseElementsAttrGetDocstring)
        .def_static("get", PyDenseElementsAttribute::getFromList,
                    nb::arg("attrs"), nb::arg("type").none() = nb::none(),
                    nb::arg("context").none() = nb::none(),
                    kDenseElementsAttrGetFromListDocstring)
        .def_static("get_splat", PyDenseElementsAttribute::getSplat,
                    nb::arg("shaped_type"), nb::arg("element_attr"),
                    "Gets a DenseElementsAttr where all values are the same")
        .def_prop_ro("is_splat",
                     [](PyDenseElementsAttribute &self) -> bool {
                       return mlirDenseElementsAttrIsSplat(self);
                     })
        .def("get_splat_value", [](PyDenseElementsAttribute &self) {
          if (!mlirDenseElementsAttrIsSplat(self))
            throw nb::value_error(
                "get_splat_value called on a non-splat attribute");
          return mlirDenseElementsAttrGetSplatValue(self);
        });
  }

  static PyType_Slot slots[];

private:
  static int bf_getbuffer(PyObject *exporter, Py_buffer *view, int flags);
  static void bf_releasebuffer(PyObject *, Py_buffer *buffer);

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

  static MlirType
  getShapedType(std::optional<MlirType> bulkLoadElementType,
                std::optional<std::vector<int64_t>> explicitShape,
                Py_buffer &view) {
    SmallVector<int64_t> shape;
    if (explicitShape) {
      shape.append(explicitShape->begin(), explicitShape->end());
    } else {
      shape.append(view.shape, view.shape + view.ndim);
    }

    if (mlirTypeIsAShaped(*bulkLoadElementType)) {
      if (explicitShape) {
        throw std::invalid_argument("Shape can only be specified explicitly "
                                    "when the type is not a shaped type.");
      }
      return *bulkLoadElementType;
    }
    MlirAttribute encodingAttr = mlirAttributeGetNull();
    return mlirRankedTensorTypeGet(shape.size(), shape.data(),
                                   *bulkLoadElementType, encodingAttr);
  }

  static MlirAttribute getAttributeFromBuffer(
      Py_buffer &view, bool signless, std::optional<PyType> explicitType,
      const std::optional<std::vector<int64_t>> &explicitShape,
      MlirContext &context) {
    // Detect format codes that are suitable for bulk loading. This includes
    // all byte aligned integer and floating point types up to 8 bytes.
    // Notably, this excludes exotics types which do not have a direct
    // representation in the buffer protocol (i.e. complex, etc).
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
      } else if (format == "?") {
        // i1
        // The i1 type needs to be bit-packed, so we will handle it seperately
        return getBitpackedAttributeFromBooleanBuffer(view, explicitShape,
                                                      context);
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

    MlirType type = getShapedType(bulkLoadElementType, explicitShape, view);
    return mlirDenseElementsAttrRawBufferGet(type, view.len, view.buf);
  }

  // There is a complication for boolean numpy arrays, as numpy represents
  // them as 8 bits (1 byte) per boolean, whereas MLIR bitpacks them into 8
  // booleans per byte.
  static MlirAttribute getBitpackedAttributeFromBooleanBuffer(
      Py_buffer &view, std::optional<std::vector<int64_t>> explicitShape,
      MlirContext &context) {
    if (llvm::endianness::native != llvm::endianness::little) {
      // Given we have no good way of testing the behavior on big-endian
      // systems we will throw
      throw nb::type_error("Constructing a bit-packed MLIR attribute is "
                           "unsupported on big-endian systems");
    }
    nb::ndarray<uint8_t, nb::numpy, nb::ndim<1>, nb::c_contig> unpackedArray(
        /*data=*/static_cast<uint8_t *>(view.buf),
        /*shape=*/{static_cast<size_t>(view.len)});

    nb::module_ numpy = nb::module_::import_("numpy");
    nb::object packbitsFunc = numpy.attr("packbits");
    nb::object packedBooleans =
        packbitsFunc(nb::cast(unpackedArray), "bitorder"_a = "little");
    nb_buffer_info pythonBuffer = nb::cast<nb_buffer>(packedBooleans).request();

    MlirType bitpackedType = getShapedType(mlirIntegerTypeGet(context, 1),
                                           std::move(explicitShape), view);
    assert(pythonBuffer.itemsize == 1 && "Packbits must return uint8");
    // Notice that `mlirDenseElementsAttrRawBufferGet` copies the memory of
    // packedBooleans, hence the MlirAttribute will remain valid even when
    // packedBooleans get reclaimed by the end of the function.
    return mlirDenseElementsAttrRawBufferGet(bitpackedType, pythonBuffer.size,
                                             pythonBuffer.ptr);
  }

  // This does the opposite transformation of
  // `getBitpackedAttributeFromBooleanBuffer`
  std::unique_ptr<nb_buffer_info> getBooleanBufferFromBitpackedAttribute() {
    if (llvm::endianness::native != llvm::endianness::little) {
      // Given we have no good way of testing the behavior on big-endian
      // systems we will throw
      throw nb::type_error("Constructing a numpy array from a MLIR attribute "
                           "is unsupported on big-endian systems");
    }

    int64_t numBooleans = mlirElementsAttrGetNumElements(*this);
    int64_t numBitpackedBytes = llvm::divideCeil(numBooleans, 8);
    uint8_t *bitpackedData = static_cast<uint8_t *>(
        const_cast<void *>(mlirDenseElementsAttrGetRawData(*this)));
    nb::ndarray<uint8_t, nb::numpy, nb::ndim<1>, nb::c_contig> packedArray(
        /*data=*/bitpackedData,
        /*shape=*/{static_cast<size_t>(numBitpackedBytes)});

    nb::module_ numpy = nb::module_::import_("numpy");
    nb::object unpackbitsFunc = numpy.attr("unpackbits");
    nb::object equalFunc = numpy.attr("equal");
    nb::object reshapeFunc = numpy.attr("reshape");
    nb::object unpackedBooleans =
        unpackbitsFunc(nb::cast(packedArray), "bitorder"_a = "little");

    // Unpackbits operates on bytes and gives back a flat 0 / 1 integer array.
    // We need to:
    //   1. Slice away the padded bits
    //   2. Make the boolean array have the correct shape
    //   3. Convert the array to a boolean array
    unpackedBooleans = unpackedBooleans[nb::slice(
        nb::int_(0), nb::int_(numBooleans), nb::int_(1))];
    unpackedBooleans = equalFunc(unpackedBooleans, 1);

    MlirType shapedType = mlirAttributeGetType(*this);
    intptr_t rank = mlirShapedTypeGetRank(shapedType);
    std::vector<intptr_t> shape(rank);
    for (intptr_t i = 0; i < rank; ++i) {
      shape[i] = mlirShapedTypeGetDimSize(shapedType, i);
    }
    unpackedBooleans = reshapeFunc(unpackedBooleans, shape);

    // Make sure the returned nb::buffer_view claims ownership of the data in
    // `pythonBuffer` so it remains valid when Python reads it
    nb_buffer pythonBuffer = nb::cast<nb_buffer>(unpackedBooleans);
    return std::make_unique<nb_buffer_info>(pythonBuffer.request());
  }

  template <typename Type>
  std::unique_ptr<nb_buffer_info>
  bufferInfo(MlirType shapedType, const char *explicitFormat = nullptr) {
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
    const char *format;
    if (explicitFormat) {
      format = explicitFormat;
    } else {
      format = nb_format_descriptor<Type>::format();
    }
    return std::make_unique<nb_buffer_info>(
        data, sizeof(Type), format, rank, std::move(shape), std::move(strides),
        /*readonly=*/true);
  }
}; // namespace

PyType_Slot PyDenseElementsAttribute::slots[] = {
// Python 3.8 doesn't allow setting the buffer protocol slots from a type spec.
#if PY_VERSION_HEX >= 0x03090000
    {Py_bf_getbuffer,
     reinterpret_cast<void *>(PyDenseElementsAttribute::bf_getbuffer)},
    {Py_bf_releasebuffer,
     reinterpret_cast<void *>(PyDenseElementsAttribute::bf_releasebuffer)},
#endif
    {0, nullptr},
};

/*static*/ int PyDenseElementsAttribute::bf_getbuffer(PyObject *obj,
                                                      Py_buffer *view,
                                                      int flags) {
  view->obj = nullptr;
  std::unique_ptr<nb_buffer_info> info;
  try {
    auto *attr = nb::cast<PyDenseElementsAttribute *>(nb::handle(obj));
    info = attr->accessBuffer();
  } catch (nb::python_error &e) {
    e.restore();
    nb::chain_error(PyExc_BufferError, "Error converting attribute to buffer");
    return -1;
  }
  view->obj = obj;
  view->ndim = 1;
  view->buf = info->ptr;
  view->itemsize = info->itemsize;
  view->len = info->itemsize;
  for (auto s : info->shape) {
    view->len *= s;
  }
  view->readonly = info->readonly;
  if ((flags & PyBUF_FORMAT) == PyBUF_FORMAT) {
    view->format = const_cast<char *>(info->format);
  }
  if ((flags & PyBUF_STRIDES) == PyBUF_STRIDES) {
    view->ndim = static_cast<int>(info->ndim);
    view->strides = info->strides.data();
    view->shape = info->shape.data();
  }
  view->suboffsets = nullptr;
  view->internal = info.release();
  Py_INCREF(obj);
  return 0;
}

/*static*/ void PyDenseElementsAttribute::bf_releasebuffer(PyObject *,
                                                           Py_buffer *view) {
  delete reinterpret_cast<nb_buffer_info *>(view->internal);
}

/// Refinement of the PyDenseElementsAttribute for attributes containing
/// integer (and boolean) values. Supports element access.
class PyDenseIntElementsAttribute
    : public PyConcreteAttribute<PyDenseIntElementsAttribute,
                                 PyDenseElementsAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsADenseIntElements;
  static constexpr const char *pyClassName = "DenseIntElementsAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  /// Returns the element at the given linear position. Asserts if the index
  /// is out of range.
  nb::object dunderGetItem(intptr_t pos) {
    if (pos < 0 || pos >= dunderLen()) {
      throw nb::index_error("attempt to access out of bounds element");
    }

    MlirType type = mlirAttributeGetType(*this);
    type = mlirShapedTypeGetElementType(type);
    // Index type can also appear as a DenseIntElementsAttr and therefore can be
    // casted to integer.
    assert(mlirTypeIsAInteger(type) ||
           mlirTypeIsAIndex(type) && "expected integer/index element type in "
                                     "dense int elements attribute");
    // Dispatch element extraction to an appropriate C function based on the
    // elemental type of the attribute. nb::int_ is implicitly constructible
    // from any C++ integral type and handles bitwidth correctly.
    // TODO: consider caching the type properties in the constructor to avoid
    // querying them on each element access.
    if (mlirTypeIsAIndex(type)) {
      return nb::int_(mlirDenseElementsAttrGetIndexValue(*this, pos));
    }
    unsigned width = mlirIntegerTypeGetWidth(type);
    bool isUnsigned = mlirIntegerTypeIsUnsigned(type);
    if (isUnsigned) {
      if (width == 1) {
        return nb::int_(int(mlirDenseElementsAttrGetBoolValue(*this, pos)));
      }
      if (width == 8) {
        return nb::int_(mlirDenseElementsAttrGetUInt8Value(*this, pos));
      }
      if (width == 16) {
        return nb::int_(mlirDenseElementsAttrGetUInt16Value(*this, pos));
      }
      if (width == 32) {
        return nb::int_(mlirDenseElementsAttrGetUInt32Value(*this, pos));
      }
      if (width == 64) {
        return nb::int_(mlirDenseElementsAttrGetUInt64Value(*this, pos));
      }
    } else {
      if (width == 1) {
        return nb::int_(int(mlirDenseElementsAttrGetBoolValue(*this, pos)));
      }
      if (width == 8) {
        return nb::int_(mlirDenseElementsAttrGetInt8Value(*this, pos));
      }
      if (width == 16) {
        return nb::int_(mlirDenseElementsAttrGetInt16Value(*this, pos));
      }
      if (width == 32) {
        return nb::int_(mlirDenseElementsAttrGetInt32Value(*this, pos));
      }
      if (width == 64) {
        return nb::int_(mlirDenseElementsAttrGetInt64Value(*this, pos));
      }
    }
    throw nb::type_error("Unsupported integer type");
  }

  static void bindDerived(ClassTy &c) {
    c.def("__getitem__", &PyDenseIntElementsAttribute::dunderGetItem);
  }
};

// Check if the python version is less than 3.13. Py_IsFinalizing is a part
// of stable ABI since 3.13 and before it was available as _Py_IsFinalizing.
#if PY_VERSION_HEX < 0x030d0000
#define Py_IsFinalizing _Py_IsFinalizing
#endif

class PyDenseResourceElementsAttribute
    : public PyConcreteAttribute<PyDenseResourceElementsAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction =
      mlirAttributeIsADenseResourceElements;
  static constexpr const char *pyClassName = "DenseResourceElementsAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static PyDenseResourceElementsAttribute
  getFromBuffer(const nb_buffer &buffer, const std::string &name,
                const PyType &type, std::optional<size_t> alignment,
                bool isMutable, DefaultingPyMlirContext contextWrapper) {
    if (!mlirTypeIsAShaped(type)) {
      throw std::invalid_argument(
          "Constructing a DenseResourceElementsAttr requires a ShapedType.");
    }

    // Do not request any conversions as we must ensure to use caller
    // managed memory.
    int flags = PyBUF_STRIDES;
    std::unique_ptr<Py_buffer> view = std::make_unique<Py_buffer>();
    if (PyObject_GetBuffer(buffer.ptr(), view.get(), flags) != 0) {
      throw nb::python_error();
    }

    // This scope releaser will only release if we haven't yet transferred
    // ownership.
    auto freeBuffer = llvm::make_scope_exit([&]() {
      if (view)
        PyBuffer_Release(view.get());
    });

    if (!PyBuffer_IsContiguous(view.get(), 'A')) {
      throw std::invalid_argument("Contiguous buffer is required.");
    }

    // Infer alignment to be the stride of one element if not explicit.
    size_t inferredAlignment;
    if (alignment)
      inferredAlignment = *alignment;
    else
      inferredAlignment = view->strides[view->ndim - 1];

    // The userData is a Py_buffer* that the deleter owns.
    auto deleter = [](void *userData, const void *data, size_t size,
                      size_t align) {
      if (Py_IsFinalizing())
        return;
      assert(Py_IsInitialized() && "expected interpreter to be initialized");
      Py_buffer *ownedView = static_cast<Py_buffer *>(userData);
      nb::gil_scoped_acquire gil;
      PyBuffer_Release(ownedView);
      delete ownedView;
    };

    size_t rawBufferSize = view->len;
    MlirAttribute attr = mlirUnmanagedDenseResourceElementsAttrGet(
        type, toMlirStringRef(name), view->buf, rawBufferSize,
        inferredAlignment, isMutable, deleter, static_cast<void *>(view.get()));
    if (mlirAttributeIsNull(attr)) {
      throw std::invalid_argument(
          "DenseResourceElementsAttr could not be constructed from the given "
          "buffer. "
          "This may mean that the Python buffer layout does not match that "
          "MLIR expected layout and is a bug.");
    }
    view.release();
    return PyDenseResourceElementsAttribute(contextWrapper->getRef(), attr);
  }

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get_from_buffer", PyDenseResourceElementsAttribute::getFromBuffer,
        nb::arg("array"), nb::arg("name"), nb::arg("type"),
        nb::arg("alignment").none() = nb::none(), nb::arg("is_mutable") = false,
        nb::arg("context").none() = nb::none(),
        kDenseResourceElementsAttrGetFromBufferDocstring);
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
        [](const nb::dict &attributes, DefaultingPyMlirContext context) {
          SmallVector<MlirNamedAttribute> mlirNamedAttributes;
          mlirNamedAttributes.reserve(attributes.size());
          for (std::pair<nb::handle, nb::handle> it : attributes) {
            auto &mlirAttr = nb::cast<PyAttribute &>(it.second);
            auto name = nb::cast<std::string>(it.first);
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
        nb::arg("value") = nb::dict(), nb::arg("context").none() = nb::none(),
        "Gets an uniqued dict attribute");
    c.def("__getitem__", [](PyDictAttribute &self, const std::string &name) {
      MlirAttribute attr =
          mlirDictionaryAttrGetElementByName(self, toMlirStringRef(name));
      if (mlirAttributeIsNull(attr))
        throw nb::key_error("attempt to access a non-existent attribute");
      return attr;
    });
    c.def("__getitem__", [](PyDictAttribute &self, intptr_t index) {
      if (index < 0 || index >= self.dunderLen()) {
        throw nb::index_error("attempt to access out of bounds attribute");
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

  nb::float_ dunderGetItem(intptr_t pos) {
    if (pos < 0 || pos >= dunderLen()) {
      throw nb::index_error("attempt to access out of bounds element");
    }

    MlirType type = mlirAttributeGetType(*this);
    type = mlirShapedTypeGetElementType(type);
    // Dispatch element extraction to an appropriate C function based on the
    // elemental type of the attribute. nb::float_ is implicitly constructible
    // from float and double.
    // TODO: consider caching the type properties in the constructor to avoid
    // querying them on each element access.
    if (mlirTypeIsAF32(type)) {
      return nb::float_(mlirDenseElementsAttrGetFloatValue(*this, pos));
    }
    if (mlirTypeIsAF64(type)) {
      return nb::float_(mlirDenseElementsAttrGetDoubleValue(*this, pos));
    }
    throw nb::type_error("Unsupported floating-point type");
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
        [](const PyType &value, DefaultingPyMlirContext context) {
          MlirAttribute attr = mlirTypeAttrGet(value.get());
          return PyTypeAttribute(context->getRef(), attr);
        },
        nb::arg("value"), nb::arg("context").none() = nb::none(),
        "Gets a uniqued Type attribute");
    c.def_prop_ro("value", [](PyTypeAttribute &self) {
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
        nb::arg("context").none() = nb::none(), "Create a Unit attribute.");
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
        [](int64_t offset, const std::vector<int64_t> &strides,
           DefaultingPyMlirContext ctx) {
          MlirAttribute attr = mlirStridedLayoutAttrGet(
              ctx->get(), offset, strides.size(), strides.data());
          return PyStridedLayoutAttribute(ctx->getRef(), attr);
        },
        nb::arg("offset"), nb::arg("strides"),
        nb::arg("context").none() = nb::none(),
        "Gets a strided layout attribute.");
    c.def_static(
        "get_fully_dynamic",
        [](int64_t rank, DefaultingPyMlirContext ctx) {
          auto dynamic = mlirShapedTypeGetDynamicStrideOrOffset();
          std::vector<int64_t> strides(rank);
          llvm::fill(strides, dynamic);
          MlirAttribute attr = mlirStridedLayoutAttrGet(
              ctx->get(), dynamic, strides.size(), strides.data());
          return PyStridedLayoutAttribute(ctx->getRef(), attr);
        },
        nb::arg("rank"), nb::arg("context").none() = nb::none(),
        "Gets a strided layout attribute with dynamic offset and strides of "
        "a "
        "given rank.");
    c.def_prop_ro(
        "offset",
        [](PyStridedLayoutAttribute &self) {
          return mlirStridedLayoutAttrGetOffset(self);
        },
        "Returns the value of the float point attribute");
    c.def_prop_ro(
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

nb::object denseArrayAttributeCaster(PyAttribute &pyAttribute) {
  if (PyDenseBoolArrayAttribute::isaFunction(pyAttribute))
    return nb::cast(PyDenseBoolArrayAttribute(pyAttribute));
  if (PyDenseI8ArrayAttribute::isaFunction(pyAttribute))
    return nb::cast(PyDenseI8ArrayAttribute(pyAttribute));
  if (PyDenseI16ArrayAttribute::isaFunction(pyAttribute))
    return nb::cast(PyDenseI16ArrayAttribute(pyAttribute));
  if (PyDenseI32ArrayAttribute::isaFunction(pyAttribute))
    return nb::cast(PyDenseI32ArrayAttribute(pyAttribute));
  if (PyDenseI64ArrayAttribute::isaFunction(pyAttribute))
    return nb::cast(PyDenseI64ArrayAttribute(pyAttribute));
  if (PyDenseF32ArrayAttribute::isaFunction(pyAttribute))
    return nb::cast(PyDenseF32ArrayAttribute(pyAttribute));
  if (PyDenseF64ArrayAttribute::isaFunction(pyAttribute))
    return nb::cast(PyDenseF64ArrayAttribute(pyAttribute));
  std::string msg =
      std::string("Can't cast unknown element type DenseArrayAttr (") +
      nb::cast<std::string>(nb::repr(nb::cast(pyAttribute))) + ")";
  throw nb::type_error(msg.c_str());
}

nb::object denseIntOrFPElementsAttributeCaster(PyAttribute &pyAttribute) {
  if (PyDenseFPElementsAttribute::isaFunction(pyAttribute))
    return nb::cast(PyDenseFPElementsAttribute(pyAttribute));
  if (PyDenseIntElementsAttribute::isaFunction(pyAttribute))
    return nb::cast(PyDenseIntElementsAttribute(pyAttribute));
  std::string msg =
      std::string(
          "Can't cast unknown element type DenseIntOrFPElementsAttr (") +
      nb::cast<std::string>(nb::repr(nb::cast(pyAttribute))) + ")";
  throw nb::type_error(msg.c_str());
}

nb::object integerOrBoolAttributeCaster(PyAttribute &pyAttribute) {
  if (PyBoolAttribute::isaFunction(pyAttribute))
    return nb::cast(PyBoolAttribute(pyAttribute));
  if (PyIntegerAttribute::isaFunction(pyAttribute))
    return nb::cast(PyIntegerAttribute(pyAttribute));
  std::string msg =
      std::string("Can't cast unknown element type DenseArrayAttr (") +
      nb::cast<std::string>(nb::repr(nb::cast(pyAttribute))) + ")";
  throw nb::type_error(msg.c_str());
}

nb::object symbolRefOrFlatSymbolRefAttributeCaster(PyAttribute &pyAttribute) {
  if (PyFlatSymbolRefAttribute::isaFunction(pyAttribute))
    return nb::cast(PyFlatSymbolRefAttribute(pyAttribute));
  if (PySymbolRefAttribute::isaFunction(pyAttribute))
    return nb::cast(PySymbolRefAttribute(pyAttribute));
  std::string msg = std::string("Can't cast unknown SymbolRef attribute (") +
                    nb::cast<std::string>(nb::repr(nb::cast(pyAttribute))) +
                    ")";
  throw nb::type_error(msg.c_str());
}

} // namespace

void mlir::python::populateIRAttributes(nb::module_ &m) {
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
      nb::cast<nb::callable>(nb::cpp_function(denseArrayAttributeCaster)));

  PyArrayAttribute::bind(m);
  PyArrayAttribute::PyArrayAttributeIterator::bind(m);
  PyBoolAttribute::bind(m);
  PyDenseElementsAttribute::bind(m, PyDenseElementsAttribute::slots);
  PyDenseFPElementsAttribute::bind(m);
  PyDenseIntElementsAttribute::bind(m);
  PyGlobals::get().registerTypeCaster(
      mlirDenseIntOrFPElementsAttrGetTypeID(),
      nb::cast<nb::callable>(
          nb::cpp_function(denseIntOrFPElementsAttributeCaster)));
  PyDenseResourceElementsAttribute::bind(m);

  PyDictAttribute::bind(m);
  PySymbolRefAttribute::bind(m);
  PyGlobals::get().registerTypeCaster(
      mlirSymbolRefAttrGetTypeID(),
      nb::cast<nb::callable>(
          nb::cpp_function(symbolRefOrFlatSymbolRefAttributeCaster)));

  PyFlatSymbolRefAttribute::bind(m);
  PyOpaqueAttribute::bind(m);
  PyFloatAttribute::bind(m);
  PyIntegerAttribute::bind(m);
  PyIntegerSetAttribute::bind(m);
  PyStringAttribute::bind(m);
  PyTypeAttribute::bind(m);
  PyGlobals::get().registerTypeCaster(
      mlirIntegerAttrGetTypeID(),
      nb::cast<nb::callable>(nb::cpp_function(integerOrBoolAttributeCaster)));
  PyUnitAttribute::bind(m);

  PyStridedLayoutAttribute::bind(m);
}
