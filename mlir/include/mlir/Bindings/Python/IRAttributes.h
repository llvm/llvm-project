//===- IRAttributes.h - Attribute Interfaces
//----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_BINDINGS_PYTHON_IRATTRIBUTES_H
#define MLIR_BINDINGS_PYTHON_IRATTRIBUTES_H

#include "mlir/Bindings/Python/IRModule.h"

namespace mlir::python {

class PyAffineMapAttribute : public PyConcreteAttribute<PyAffineMapAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAAffineMap;
  static constexpr const char *pyClassName = "AffineMapAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirAffineMapAttrGetTypeID;

  static void bindDerived(ClassTy &c);
};

class PyIntegerSetAttribute
    : public PyConcreteAttribute<PyIntegerSetAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAIntegerSet;
  static constexpr const char *pyClassName = "IntegerSetAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirIntegerSetAttrGetTypeID;

  static void bindDerived(ClassTy &c);
};

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
    PyDenseArrayIterator dunderIter();

    /// Return the next element.
    EltTy dunderNext();

    /// Bind the iterator class.
    static void bind(nanobind::module_ &m);

  private:
    /// The referenced dense array attribute.
    PyAttribute attr;
    /// The next index to read.
    int nextIndex = 0;
  };

  /// Get the element at the given index.
  EltTy getItem(intptr_t i);

  /// Bind the attribute class.
  static void bindDerived(typename PyConcreteAttribute<DerivedT>::ClassTy &c);

private:
  static DerivedT getAttribute(const std::vector<EltTy> &values,
                               PyMlirContextRef ctx);
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

    PyArrayAttributeIterator &dunderIter();

    MlirAttribute dunderNext();

    static void bind(nanobind::module_ &m);

  private:
    PyAttribute attr;
    int nextIndex = 0;
  };

  MlirAttribute getItem(intptr_t i);

  static void bindDerived(ClassTy &c);
};

/// Float Point Attribute subclass - FloatAttr.
class PyFloatAttribute : public PyConcreteAttribute<PyFloatAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAFloat;
  static constexpr const char *pyClassName = "FloatAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFloatAttrGetTypeID;

  static void bindDerived(ClassTy &c);
};

/// Integer Attribute subclass - IntegerAttr.
class PyIntegerAttribute : public PyConcreteAttribute<PyIntegerAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAInteger;
  static constexpr const char *pyClassName = "IntegerAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c);

private:
  static int64_t toPyInt(PyIntegerAttribute &self);
};

/// Bool Attribute subclass - BoolAttr.
class PyBoolAttribute : public PyConcreteAttribute<PyBoolAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsABool;
  static constexpr const char *pyClassName = "BoolAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c);
};

class PySymbolRefAttribute : public PyConcreteAttribute<PySymbolRefAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsASymbolRef;
  static constexpr const char *pyClassName = "SymbolRefAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static MlirAttribute fromList(const std::vector<std::string> &symbols,
                                PyMlirContext &context);

  static void bindDerived(ClassTy &c);
};

class PyFlatSymbolRefAttribute
    : public PyConcreteAttribute<PyFlatSymbolRefAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAFlatSymbolRef;
  static constexpr const char *pyClassName = "FlatSymbolRefAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c);
};

class PyOpaqueAttribute : public PyConcreteAttribute<PyOpaqueAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAOpaque;
  static constexpr const char *pyClassName = "OpaqueAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirOpaqueAttrGetTypeID;

  static void bindDerived(ClassTy &c);
};

class PyStringAttribute : public PyConcreteAttribute<PyStringAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAString;
  static constexpr const char *pyClassName = "StringAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirStringAttrGetTypeID;

  static void bindDerived(ClassTy &c);
};

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

class nb_buffer : public nanobind::object {
  NB_OBJECT_DEFAULT(nb_buffer, object, "buffer", PyObject_CheckBuffer);

  nb_buffer_info request() const;
};

// TODO: Support construction of string elements.
class PyDenseElementsAttribute
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

  intptr_t dunderLen();

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
  std::unique_ptr<nb_buffer_info> getBooleanBufferFromBitpackedAttribute();

  template <typename Type>
  std::unique_ptr<nb_buffer_info>
  bufferInfo(MlirType shapedType, const char *explicitFormat = nullptr);
}; // namespace

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
                bool isMutable, DefaultingPyMlirContext contextWrapper);

  static void bindDerived(ClassTy &c);
};

class PyDictAttribute : public PyConcreteAttribute<PyDictAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsADictionary;
  static constexpr const char *pyClassName = "DictAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirDictionaryAttrGetTypeID;

  intptr_t dunderLen();

  bool dunderContains(const std::string &name);

  static void bindDerived(ClassTy &c);
};

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
  nanobind::object dunderGetItem(intptr_t pos);

  static void bindDerived(ClassTy &c);
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

  nanobind::float_ dunderGetItem(intptr_t pos);

  static void bindDerived(ClassTy &c);
};

class PyTypeAttribute : public PyConcreteAttribute<PyTypeAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAType;
  static constexpr const char *pyClassName = "TypeAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirTypeAttrGetTypeID;

  static void bindDerived(ClassTy &c);
};

/// Unit Attribute subclass. Unit attributes don't have values.
class PyUnitAttribute : public PyConcreteAttribute<PyUnitAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAUnit;
  static constexpr const char *pyClassName = "UnitAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirUnitAttrGetTypeID;

  static void bindDerived(ClassTy &c);
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

  static void bindDerived(ClassTy &c);
};
} // namespace mlir::python

#endif // MLIR_BINDINGS_PYTHON_IRATTRIBUTES_H
