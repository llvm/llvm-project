//===- IRAttributes.h - Exports builtin and standard attributes -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_BINDINGS_PYTHON_IRATTRIBUTES_H
#define MLIR_BINDINGS_PYTHON_IRATTRIBUTES_H

#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/Bindings/Python/NanobindUtils.h"

namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {

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
          std::unique_ptr<Py_buffer, void (*)(Py_buffer *)>(nullptr, nullptr));

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

class MLIR_PYTHON_API_EXPORTED nb_buffer : public nanobind::object {
  NB_OBJECT_DEFAULT(nb_buffer, object, "Buffer", PyObject_CheckBuffer);

  nb_buffer_info request() const;
};

template <typename T>
struct nb_format_descriptor {};

class MLIR_PYTHON_API_EXPORTED PyAffineMapAttribute
    : public PyConcreteAttribute<PyAffineMapAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAAffineMap;
  static constexpr const char *pyClassName = "AffineMapAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirAffineMapAttrGetTypeID;
  static inline const MlirStringRef name = mlirAffineMapAttrGetName();

  static void bindDerived(ClassTy &c);
};

class MLIR_PYTHON_API_EXPORTED PyIntegerSetAttribute
    : public PyConcreteAttribute<PyIntegerSetAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAIntegerSet;
  static constexpr const char *pyClassName = "IntegerSetAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirIntegerSetAttrGetTypeID;
  static inline const MlirStringRef name = mlirIntegerSetAttrGetName();

  static void bindDerived(ClassTy &c);
};

template <typename T>
static T pyTryCast(nanobind::handle object) {
  try {
    return nanobind::cast<T>(object);
  } catch (nanobind::cast_error &err) {
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
class MLIR_PYTHON_API_EXPORTED PyDenseArrayAttribute
    : public PyConcreteAttribute<DerivedT> {
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
        throw nanobind::stop_iteration();
      return DerivedT::getElement(attr.get(), nextIndex++);
    }

    /// Bind the iterator class.
    static void bind(nanobind::module_ &m) {
      nanobind::class_<PyDenseArrayIterator>(m, DerivedT::pyIteratorName)
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
          [](const nanobind::sequence &py_values, DefaultingPyMlirContext ctx) {
            std::vector<bool> values;
            for (nanobind::handle py_value : py_values) {
              int is_true = PyObject_IsTrue(py_value.ptr());
              if (is_true < 0) {
                throw nanobind::python_error();
              }
              values.push_back(is_true);
            }
            return getAttribute(values, ctx->getRef());
          },
          nanobind::arg("values"), nanobind::arg("context") = nanobind::none(),
          "Gets a uniqued dense array attribute");
    } else {
      c.def_static(
          "get",
          [](const std::vector<EltTy> &values, DefaultingPyMlirContext ctx) {
            return getAttribute(values, ctx->getRef());
          },
          nanobind::arg("values"), nanobind::arg("context") = nanobind::none(),
          "Gets a uniqued dense array attribute");
    }
    // Bind the array methods.
    c.def("__getitem__", [](DerivedT &arr, intptr_t i) {
      if (i >= mlirDenseArrayGetNumElements(arr))
        throw nanobind::index_error("DenseArray index out of range");
      return arr.getItem(i);
    });
    c.def("__len__", [](const DerivedT &arr) {
      return mlirDenseArrayGetNumElements(arr);
    });
    c.def("__iter__",
          [](const DerivedT &arr) { return PyDenseArrayIterator(arr); });
    c.def("__add__", [](DerivedT &arr, const nanobind::list &extras) {
      std::vector<EltTy> values;
      intptr_t numOldElements = mlirDenseArrayGetNumElements(arr);
      values.reserve(numOldElements + nanobind::len(extras));
      for (intptr_t i = 0; i < numOldElements; ++i)
        values.push_back(arr.getItem(i));
      for (nanobind::handle attr : extras)
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

class MLIR_PYTHON_API_EXPORTED PyArrayAttribute
    : public PyConcreteAttribute<PyArrayAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAArray;
  static constexpr const char *pyClassName = "ArrayAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirArrayAttrGetTypeID;
  static inline const MlirStringRef name = mlirArrayAttrGetName();

  class PyArrayAttributeIterator {
  public:
    PyArrayAttributeIterator(PyAttribute attr) : attr(std::move(attr)) {}

    PyArrayAttributeIterator &dunderIter() { return *this; }

    nanobind::typed<nanobind::object, PyAttribute> dunderNext();

    static void bind(nanobind::module_ &m);

  private:
    PyAttribute attr;
    int nextIndex = 0;
  };

  MlirAttribute getItem(intptr_t i) const;

  static void bindDerived(ClassTy &c);
};

/// Float Point Attribute subclass - FloatAttr.
class MLIR_PYTHON_API_EXPORTED PyFloatAttribute
    : public PyConcreteAttribute<PyFloatAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAFloat;
  static constexpr const char *pyClassName = "FloatAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloatAttrGetTypeID;
  static inline const MlirStringRef name = mlirFloatAttrGetName();

  static void bindDerived(ClassTy &c);
};

/// Integer Attribute subclass - IntegerAttr.
class MLIR_PYTHON_API_EXPORTED PyIntegerAttribute
    : public PyConcreteAttribute<PyIntegerAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAInteger;
  static constexpr const char *pyClassName = "IntegerAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static inline const MlirStringRef name = mlirIntegerAttrGetName();

  static void bindDerived(ClassTy &c);

private:
  static int64_t toPyInt(PyIntegerAttribute &self);
};

/// Bool Attribute subclass - BoolAttr.
class MLIR_PYTHON_API_EXPORTED PyBoolAttribute
    : public PyConcreteAttribute<PyBoolAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsABool;
  static constexpr const char *pyClassName = "BoolAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c);
};

class MLIR_PYTHON_API_EXPORTED PySymbolRefAttribute
    : public PyConcreteAttribute<PySymbolRefAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsASymbolRef;
  static constexpr const char *pyClassName = "SymbolRefAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static inline const MlirStringRef name = mlirSymbolRefAttrGetName();

  static PySymbolRefAttribute fromList(const std::vector<std::string> &symbols,
                                       PyMlirContext &context);

  static void bindDerived(ClassTy &c);
};

class MLIR_PYTHON_API_EXPORTED PyFlatSymbolRefAttribute
    : public PyConcreteAttribute<PyFlatSymbolRefAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAFlatSymbolRef;
  static constexpr const char *pyClassName = "FlatSymbolRefAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static inline const MlirStringRef name = mlirFlatSymbolRefAttrGetName();

  static void bindDerived(ClassTy &c);
};

class MLIR_PYTHON_API_EXPORTED PyOpaqueAttribute
    : public PyConcreteAttribute<PyOpaqueAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAOpaque;
  static constexpr const char *pyClassName = "OpaqueAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirOpaqueAttrGetTypeID;
  static inline const MlirStringRef name = mlirOpaqueAttrGetName();

  static void bindDerived(ClassTy &c);
};

// TODO: Support construction of string elements.
class MLIR_PYTHON_API_EXPORTED PyDenseElementsAttribute
    : public PyConcreteAttribute<PyDenseElementsAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsADenseElements;
  static constexpr const char *pyClassName = "DenseElementsAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static PyDenseElementsAttribute
  getFromList(const nanobind::list &attributes,
              std::optional<PyType> explicitType,
              DefaultingPyMlirContext contextWrapper);

  static PyDenseElementsAttribute
  getFromBuffer(const nb_buffer &array, bool signless,
                const std::optional<PyType> &explicitType,
                std::optional<std::vector<int64_t>> explicitShape,
                DefaultingPyMlirContext contextWrapper);

  static PyDenseElementsAttribute getSplat(const PyType &shapedType,
                                           PyAttribute &elementAttr);

  intptr_t dunderLen() const;

  std::unique_ptr<nb_buffer_info> accessBuffer();

  static void bindDerived(ClassTy &c);

  static PyType_Slot slots[];

private:
  static int bf_getbuffer(PyObject *exporter, Py_buffer *view, int flags);
  static void bf_releasebuffer(PyObject *, Py_buffer *buffer);

  static bool isUnsignedIntegerFormat(std::string_view format);

  static bool isSignedIntegerFormat(std::string_view format);

  static MlirType
  getShapedType(std::optional<MlirType> bulkLoadElementType,
                std::optional<std::vector<int64_t>> explicitShape,
                Py_buffer &view);

  static MlirAttribute getAttributeFromBuffer(
      Py_buffer &view, bool signless, std::optional<PyType> explicitType,
      const std::optional<std::vector<int64_t>> &explicitShape,
      MlirContext &context);

  // There is a complication for boolean numpy arrays, as numpy represents
  // them as 8 bits (1 byte) per boolean, whereas MLIR bitpacks them into 8
  // booleans per byte.
  static MlirAttribute getBitpackedAttributeFromBooleanBuffer(
      Py_buffer &view, std::optional<std::vector<int64_t>> explicitShape,
      MlirContext &context);

  // This does the opposite transformation of
  // `getBitpackedAttributeFromBooleanBuffer`
  std::unique_ptr<nb_buffer_info>
  getBooleanBufferFromBitpackedAttribute() const;

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
};

/// Refinement of the PyDenseElementsAttribute for attributes containing
/// integer (and boolean) values. Supports element access.
class MLIR_PYTHON_API_EXPORTED PyDenseIntElementsAttribute
    : public PyConcreteAttribute<PyDenseIntElementsAttribute,
                                 PyDenseElementsAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsADenseIntElements;
  static constexpr const char *pyClassName = "DenseIntElementsAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  /// Returns the element at the given linear position. Asserts if the index
  /// is out of range.
  nanobind::int_ dunderGetItem(intptr_t pos) const;

  static void bindDerived(ClassTy &c);
};

class MLIR_PYTHON_API_EXPORTED PyDenseResourceElementsAttribute
    : public PyConcreteAttribute<PyDenseResourceElementsAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction =
      mlirAttributeIsADenseResourceElements;
  static constexpr const char *pyClassName = "DenseResourceElementsAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static inline const MlirStringRef name =
      mlirDenseResourceElementsAttrGetName();

  static PyDenseResourceElementsAttribute
  getFromBuffer(const nb_buffer &buffer, const std::string &name,
                const PyType &type, std::optional<size_t> alignment,
                bool isMutable, DefaultingPyMlirContext contextWrapper);

  static void bindDerived(ClassTy &c);
};

class MLIR_PYTHON_API_EXPORTED PyDictAttribute
    : public PyConcreteAttribute<PyDictAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsADictionary;
  static constexpr const char *pyClassName = "DictAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirDictionaryAttrGetTypeID;
  static inline const MlirStringRef name = mlirDictionaryAttrGetName();

  intptr_t dunderLen() const;

  bool dunderContains(const std::string &name) const;

  static void bindDerived(ClassTy &c);
};

/// Refinement of PyDenseElementsAttribute for attributes containing
/// floating-point values. Supports element access.
class MLIR_PYTHON_API_EXPORTED PyDenseFPElementsAttribute
    : public PyConcreteAttribute<PyDenseFPElementsAttribute,
                                 PyDenseElementsAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsADenseFPElements;
  static constexpr const char *pyClassName = "DenseFPElementsAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  nanobind::float_ dunderGetItem(intptr_t pos) const;

  static void bindDerived(ClassTy &c);
};

class MLIR_PYTHON_API_EXPORTED PyTypeAttribute
    : public PyConcreteAttribute<PyTypeAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAType;
  static constexpr const char *pyClassName = "TypeAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirTypeAttrGetTypeID;
  static inline const MlirStringRef name = mlirTypeAttrGetName();

  static void bindDerived(ClassTy &c);
};

/// Unit Attribute subclass. Unit attributes don't have values.
class MLIR_PYTHON_API_EXPORTED PyUnitAttribute
    : public PyConcreteAttribute<PyUnitAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAUnit;
  static constexpr const char *pyClassName = "UnitAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirUnitAttrGetTypeID;
  static inline const MlirStringRef name = mlirUnitAttrGetName();

  static void bindDerived(ClassTy &c);
};

/// Strided layout attribute subclass.
class MLIR_PYTHON_API_EXPORTED PyStridedLayoutAttribute
    : public PyConcreteAttribute<PyStridedLayoutAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAStridedLayout;
  static constexpr const char *pyClassName = "StridedLayoutAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirStridedLayoutAttrGetTypeID;
  static inline const MlirStringRef name = mlirStridedLayoutAttrGetName();

  static void bindDerived(ClassTy &c);
};

MLIR_PYTHON_API_EXPORTED void populateIRAttributes(nanobind::module_ &m);
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

#endif
