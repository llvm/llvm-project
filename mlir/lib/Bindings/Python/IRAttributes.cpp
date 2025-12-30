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

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir/Bindings/Python/IRAttributes.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/Bindings/Python/NanobindUtils.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/raw_ostream.h"

namespace nb = nanobind;
using namespace nanobind::literals;
using namespace mlir;
using namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN;

using llvm::SmallVector;

//------------------------------------------------------------------------------
// Docstrings (trivial, non-duplicated docstrings are included inline).
//------------------------------------------------------------------------------

namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {

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

nanobind::typed<nanobind::object, PyAttribute>
PyArrayAttribute::PyArrayAttributeIterator::dunderNext() {
  // TODO: Throw is an inefficient way to stop iteration.
  if (nextIndex >= mlirArrayAttrGetNumElements(attr.get()))
    throw nanobind::stop_iteration();
  return PyAttribute(this->attr.getContext(),
                     mlirArrayAttrGetElement(attr.get(), nextIndex++))
      .maybeDownCast();
}

MlirAttribute PyArrayAttribute::getItem(intptr_t i) const {
  return mlirArrayAttrGetElement(*this, i);
}

int64_t PyIntegerAttribute::toPyInt(PyIntegerAttribute &self) {
  MlirType type = mlirAttributeGetType(self);
  if (mlirTypeIsAIndex(type) || mlirIntegerTypeIsSignless(type))
    return mlirIntegerAttrGetValueInt(self);
  if (mlirIntegerTypeIsSigned(type))
    return mlirIntegerAttrGetValueSInt(self);
  return mlirIntegerAttrGetValueUInt(self);
}

PySymbolRefAttribute
PySymbolRefAttribute::fromList(const std::vector<std::string> &symbols,
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
  return PySymbolRefAttribute(context.getRef(),
                              mlirSymbolRefAttrGet(context.get(), rootSymbol,
                                                   referenceAttrs.size(),
                                                   referenceAttrs.data()));
}

PyDenseElementsAttribute
PyDenseElementsAttribute::getFromList(const nanobind::list &attributes,
                                      std::optional<PyType> explicitType,
                                      DefaultingPyMlirContext contextWrapper) {
  const size_t numAttributes = nanobind::len(attributes);
  if (numAttributes == 0)
    throw nanobind::value_error("Attributes list must be non-empty.");

  MlirType shapedType;
  if (explicitType) {
    if ((!mlirTypeIsAShaped(*explicitType) ||
         !mlirShapedTypeHasStaticShape(*explicitType))) {

      std::string message;
      llvm::raw_string_ostream os(message);
      os << "Expected a static ShapedType for the shaped_type parameter: "
         << nanobind::cast<std::string>(
                nanobind::repr(nanobind::cast(*explicitType)));
      throw nanobind::value_error(message.c_str());
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
  for (const nanobind::handle &attribute : attributes) {
    MlirAttribute mlirAttribute = pyTryCast<PyAttribute>(attribute);
    MlirType attrType = mlirAttributeGetType(mlirAttribute);
    mlirAttributes.push_back(mlirAttribute);

    if (!mlirTypeEqual(mlirShapedTypeGetElementType(shapedType), attrType)) {
      std::string message;
      llvm::raw_string_ostream os(message);
      os << "All attributes must be of the same type and match "
         << "the type parameter: expected="
         << nanobind::cast<std::string>(
                nanobind::repr(nanobind::cast(shapedType)))
         << ", but got="
         << nanobind::cast<std::string>(
                nanobind::repr(nanobind::cast(attrType)));
      throw nanobind::value_error(message.c_str());
    }
  }

  MlirAttribute elements = mlirDenseElementsAttrGet(
      shapedType, mlirAttributes.size(), mlirAttributes.data());

  return PyDenseElementsAttribute(contextWrapper->getRef(), elements);
}
PyDenseElementsAttribute PyDenseElementsAttribute::getFromBuffer(
    const nb_buffer &array, bool signless,
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
    throw nanobind::python_error();
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

PyDenseElementsAttribute
PyDenseElementsAttribute::getSplat(const PyType &shapedType,
                                   PyAttribute &elementAttr) {
  auto contextWrapper =
      PyMlirContext::forContext(mlirTypeGetContext(shapedType));
  if (!mlirAttributeIsAInteger(elementAttr) &&
      !mlirAttributeIsAFloat(elementAttr)) {
    std::string message = "Illegal element type for DenseElementsAttr: ";
    message.append(nanobind::cast<std::string>(
        nanobind::repr(nanobind::cast(elementAttr))));
    throw nanobind::value_error(message.c_str());
  }
  if (!mlirTypeIsAShaped(shapedType) ||
      !mlirShapedTypeHasStaticShape(shapedType)) {
    std::string message =
        "Expected a static ShapedType for the shaped_type parameter: ";
    message.append(nanobind::cast<std::string>(
        nanobind::repr(nanobind::cast(shapedType))));
    throw nanobind::value_error(message.c_str());
  }
  MlirType shapedElementType = mlirShapedTypeGetElementType(shapedType);
  MlirType attrType = mlirAttributeGetType(elementAttr);
  if (!mlirTypeEqual(shapedElementType, attrType)) {
    std::string message =
        "Shaped element type and attribute type must be equal: shaped=";
    message.append(nanobind::cast<std::string>(
        nanobind::repr(nanobind::cast(shapedType))));
    message.append(", element=");
    message.append(nanobind::cast<std::string>(
        nanobind::repr(nanobind::cast(elementAttr))));
    throw nanobind::value_error(message.c_str());
  }

  MlirAttribute elements =
      mlirDenseElementsAttrSplatGet(shapedType, elementAttr);
  return PyDenseElementsAttribute(contextWrapper->getRef(), elements);
}

std::unique_ptr<nb_buffer_info> PyDenseElementsAttribute::accessBuffer() {
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

// Check if the python version is less than 3.13. Py_IsFinalizing is a part
// of stable ABI since 3.13 and before it was available as _Py_IsFinalizing.
#if PY_VERSION_HEX < 0x030d0000
#define Py_IsFinalizing _Py_IsFinalizing
#endif

bool PyDenseElementsAttribute::isUnsignedIntegerFormat(
    std::string_view format) {
  if (format.empty())
    return false;
  char code = format[0];
  return code == 'I' || code == 'B' || code == 'H' || code == 'L' ||
         code == 'Q';
}

bool PyDenseElementsAttribute::isSignedIntegerFormat(std::string_view format) {
  if (format.empty())
    return false;
  char code = format[0];
  return code == 'i' || code == 'b' || code == 'h' || code == 'l' ||
         code == 'q';
}

MlirType PyDenseElementsAttribute::getShapedType(
    std::optional<MlirType> bulkLoadElementType,
    std::optional<std::vector<int64_t>> explicitShape, Py_buffer &view) {
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

MlirAttribute PyDenseElementsAttribute::getAttributeFromBuffer(
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
      // The i1 type needs to be bit-packed, so we will handle it separately
      return getBitpackedAttributeFromBooleanBuffer(view, explicitShape,
                                                    context);
    } else if (isSignedIntegerFormat(format)) {
      if (view.itemsize == 4) {
        // i32
        bulkLoadElementType = signless ? mlirIntegerTypeGet(context, 32)
                                       : mlirIntegerTypeSignedGet(context, 32);
      } else if (view.itemsize == 8) {
        // i64
        bulkLoadElementType = signless ? mlirIntegerTypeGet(context, 64)
                                       : mlirIntegerTypeSignedGet(context, 64);
      } else if (view.itemsize == 1) {
        // i8
        bulkLoadElementType = signless ? mlirIntegerTypeGet(context, 8)
                                       : mlirIntegerTypeSignedGet(context, 8);
      } else if (view.itemsize == 2) {
        // i16
        bulkLoadElementType = signless ? mlirIntegerTypeGet(context, 16)
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
        bulkLoadElementType = signless ? mlirIntegerTypeGet(context, 8)
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

MlirAttribute PyDenseElementsAttribute::getBitpackedAttributeFromBooleanBuffer(
    Py_buffer &view, std::optional<std::vector<int64_t>> explicitShape,
    MlirContext &context) {
  if (llvm::endianness::native != llvm::endianness::little) {
    // Given we have no good way of testing the behavior on big-endian
    // systems we will throw
    throw nanobind::type_error("Constructing a bit-packed MLIR attribute is "
                               "unsupported on big-endian systems");
  }
  nanobind::ndarray<uint8_t, nanobind::numpy, nanobind::ndim<1>,
                    nanobind::c_contig>
      unpackedArray(
          /*data=*/static_cast<uint8_t *>(view.buf),
          /*shape=*/{static_cast<size_t>(view.len)});

  nanobind::module_ numpy = nanobind::module_::import_("numpy");
  nanobind::object packbitsFunc = numpy.attr("packbits");
  nanobind::object packedBooleans =
      packbitsFunc(nanobind::cast(unpackedArray), "bitorder"_a = "little");
  nb_buffer_info pythonBuffer =
      nanobind::cast<nb_buffer>(packedBooleans).request();

  MlirType bitpackedType = getShapedType(mlirIntegerTypeGet(context, 1),
                                         std::move(explicitShape), view);
  assert(pythonBuffer.itemsize == 1 && "Packbits must return uint8");
  // Notice that `mlirDenseElementsAttrRawBufferGet` copies the memory of
  // packedBooleans, hence the MlirAttribute will remain valid even when
  // packedBooleans get reclaimed by the end of the function.
  return mlirDenseElementsAttrRawBufferGet(bitpackedType, pythonBuffer.size,
                                           pythonBuffer.ptr);
}

std::unique_ptr<nb_buffer_info>
PyDenseElementsAttribute::getBooleanBufferFromBitpackedAttribute() {
  if (llvm::endianness::native != llvm::endianness::little) {
    // Given we have no good way of testing the behavior on big-endian
    // systems we will throw
    throw nanobind::type_error(
        "Constructing a numpy array from a MLIR attribute "
        "is unsupported on big-endian systems");
  }

  int64_t numBooleans = mlirElementsAttrGetNumElements(*this);
  int64_t numBitpackedBytes = llvm::divideCeil(numBooleans, 8);
  uint8_t *bitpackedData = static_cast<uint8_t *>(
      const_cast<void *>(mlirDenseElementsAttrGetRawData(*this)));
  nanobind::ndarray<uint8_t, nanobind::numpy, nanobind::ndim<1>,
                    nanobind::c_contig>
      packedArray(
          /*data=*/bitpackedData,
          /*shape=*/{static_cast<size_t>(numBitpackedBytes)});

  nanobind::module_ numpy = nanobind::module_::import_("numpy");
  nanobind::object unpackbitsFunc = numpy.attr("unpackbits");
  nanobind::object equalFunc = numpy.attr("equal");
  nanobind::object reshapeFunc = numpy.attr("reshape");
  nanobind::object unpackedBooleans =
      unpackbitsFunc(nanobind::cast(packedArray), "bitorder"_a = "little");

  // Unpackbits operates on bytes and gives back a flat 0 / 1 integer array.
  // We need to:
  //   1. Slice away the padded bits
  //   2. Make the boolean array have the correct shape
  //   3. Convert the array to a boolean array
  unpackedBooleans = unpackedBooleans[nanobind::slice(
      nanobind::int_(0), nanobind::int_(numBooleans), nanobind::int_(1))];
  unpackedBooleans = equalFunc(unpackedBooleans, 1);

  MlirType shapedType = mlirAttributeGetType(*this);
  intptr_t rank = mlirShapedTypeGetRank(shapedType);
  std::vector<intptr_t> shape(rank);
  for (intptr_t i = 0; i < rank; ++i) {
    shape[i] = mlirShapedTypeGetDimSize(shapedType, i);
  }
  unpackedBooleans = reshapeFunc(unpackedBooleans, shape);

  // Make sure the returned nanobind::buffer_view claims ownership of the data
  // in `pythonBuffer` so it remains valid when Python reads it
  nb_buffer pythonBuffer = nanobind::cast<nb_buffer>(unpackedBooleans);
  return std::make_unique<nb_buffer_info>(pythonBuffer.request());
}

nanobind::int_ PyDenseIntElementsAttribute::dunderGetItem(intptr_t pos) {
  if (pos < 0 || pos >= dunderLen()) {
    throw nanobind::index_error("attempt to access out of bounds element");
  }

  MlirType type = mlirAttributeGetType(*this);
  type = mlirShapedTypeGetElementType(type);
  // Index type can also appear as a DenseIntElementsAttr and therefore can be
  // casted to integer.
  assert(mlirTypeIsAInteger(type) ||
         mlirTypeIsAIndex(type) && "expected integer/index element type in "
                                   "dense int elements attribute");
  // Dispatch element extraction to an appropriate C function based on the
  // elemental type of the attribute. nanobind::int_ is implicitly
  // constructible from any C++ integral type and handles bitwidth correctly.
  // TODO: consider caching the type properties in the constructor to avoid
  // querying them on each element access.
  if (mlirTypeIsAIndex(type)) {
    return nanobind::int_(mlirDenseElementsAttrGetIndexValue(*this, pos));
  }
  unsigned width = mlirIntegerTypeGetWidth(type);
  bool isUnsigned = mlirIntegerTypeIsUnsigned(type);
  if (isUnsigned) {
    if (width == 1) {
      return nanobind::int_(int(mlirDenseElementsAttrGetBoolValue(*this, pos)));
    }
    if (width == 8) {
      return nanobind::int_(mlirDenseElementsAttrGetUInt8Value(*this, pos));
    }
    if (width == 16) {
      return nanobind::int_(mlirDenseElementsAttrGetUInt16Value(*this, pos));
    }
    if (width == 32) {
      return nanobind::int_(mlirDenseElementsAttrGetUInt32Value(*this, pos));
    }
    if (width == 64) {
      return nanobind::int_(mlirDenseElementsAttrGetUInt64Value(*this, pos));
    }
  } else {
    if (width == 1) {
      return nanobind::int_(int(mlirDenseElementsAttrGetBoolValue(*this, pos)));
    }
    if (width == 8) {
      return nanobind::int_(mlirDenseElementsAttrGetInt8Value(*this, pos));
    }
    if (width == 16) {
      return nanobind::int_(mlirDenseElementsAttrGetInt16Value(*this, pos));
    }
    if (width == 32) {
      return nanobind::int_(mlirDenseElementsAttrGetInt32Value(*this, pos));
    }
    if (width == 64) {
      return nanobind::int_(mlirDenseElementsAttrGetInt64Value(*this, pos));
    }
  }
  throw nanobind::type_error("Unsupported integer type");
}

PyDenseResourceElementsAttribute
PyDenseResourceElementsAttribute::getFromBuffer(
    const nb_buffer &buffer, const std::string &name, const PyType &type,
    std::optional<size_t> alignment, bool isMutable,
    DefaultingPyMlirContext contextWrapper) {
  if (!mlirTypeIsAShaped(type)) {
    throw std::invalid_argument(
        "Constructing a DenseResourceElementsAttr requires a ShapedType.");
  }

  // Do not request any conversions as we must ensure to use caller
  // managed memory.
  int flags = PyBUF_STRIDES;
  std::unique_ptr<Py_buffer> view = std::make_unique<Py_buffer>();
  if (PyObject_GetBuffer(buffer.ptr(), view.get(), flags) != 0) {
    throw nanobind::python_error();
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
    nanobind::gil_scoped_acquire gil;
    PyBuffer_Release(ownedView);
    delete ownedView;
  };

  size_t rawBufferSize = view->len;
  MlirAttribute attr = mlirUnmanagedDenseResourceElementsAttrGet(
      type, toMlirStringRef(name), view->buf, rawBufferSize, inferredAlignment,
      isMutable, deleter, static_cast<void *>(view.get()));
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

bool PyDictAttribute::dunderContains(const std::string &name) const {
  return !mlirAttributeIsNull(
      mlirDictionaryAttrGetElementByName(*this, toMlirStringRef(name)));
}

nanobind::float_ PyDenseFPElementsAttribute::dunderGetItem(intptr_t pos) {
  if (pos < 0 || pos >= dunderLen()) {
    throw nanobind::index_error("attempt to access out of bounds element");
  }

  MlirType type = mlirAttributeGetType(*this);
  type = mlirShapedTypeGetElementType(type);
  // Dispatch element extraction to an appropriate C function based on the
  // elemental type of the attribute. nanobind::float_ is implicitly
  // constructible from float and double.
  // TODO: consider caching the type properties in the constructor to avoid
  // querying them on each element access.
  if (mlirTypeIsAF32(type)) {
    return nanobind::float_(mlirDenseElementsAttrGetFloatValue(*this, pos));
  }
  if (mlirTypeIsAF64(type)) {
    return nanobind::float_(mlirDenseElementsAttrGetDoubleValue(*this, pos));
  }
  throw nanobind::type_error("Unsupported floating-point type");
}

/*static*/ int PyDenseElementsAttribute::bf_getbuffer(PyObject *obj,
                                                      Py_buffer *view,
                                                      int flags) {
  view->obj = nullptr;
  std::unique_ptr<nb_buffer_info> info;
  try {
    auto *attr =
        nanobind::cast<PyDenseElementsAttribute *>(nanobind::handle(obj));
    info = attr->accessBuffer();
  } catch (nanobind::python_error &e) {
    e.restore();
    nanobind::chain_error(PyExc_BufferError,
                          "Error converting attribute to buffer");
    return -1;
  } catch (std::exception &e) {
    nanobind::chain_error(PyExc_BufferError,
                          "Error converting attribute to buffer: %s", e.what());
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

nanobind::object denseArrayAttributeCaster(PyAttribute &pyAttribute) {
  if (PyDenseBoolArrayAttribute::isaFunction(pyAttribute))
    return nanobind::cast(PyDenseBoolArrayAttribute(pyAttribute));
  if (PyDenseI8ArrayAttribute::isaFunction(pyAttribute))
    return nanobind::cast(PyDenseI8ArrayAttribute(pyAttribute));
  if (PyDenseI16ArrayAttribute::isaFunction(pyAttribute))
    return nanobind::cast(PyDenseI16ArrayAttribute(pyAttribute));
  if (PyDenseI32ArrayAttribute::isaFunction(pyAttribute))
    return nanobind::cast(PyDenseI32ArrayAttribute(pyAttribute));
  if (PyDenseI64ArrayAttribute::isaFunction(pyAttribute))
    return nanobind::cast(PyDenseI64ArrayAttribute(pyAttribute));
  if (PyDenseF32ArrayAttribute::isaFunction(pyAttribute))
    return nanobind::cast(PyDenseF32ArrayAttribute(pyAttribute));
  if (PyDenseF64ArrayAttribute::isaFunction(pyAttribute))
    return nanobind::cast(PyDenseF64ArrayAttribute(pyAttribute));
  std::string msg =
      std::string("Can't cast unknown element type DenseArrayAttr (") +
      nanobind::cast<std::string>(nanobind::repr(nanobind::cast(pyAttribute))) +
      ")";
  throw nanobind::type_error(msg.c_str());
}

nanobind::object denseIntOrFPElementsAttributeCaster(PyAttribute &pyAttribute) {
  if (PyDenseFPElementsAttribute::isaFunction(pyAttribute))
    return nanobind::cast(PyDenseFPElementsAttribute(pyAttribute));
  if (PyDenseIntElementsAttribute::isaFunction(pyAttribute))
    return nanobind::cast(PyDenseIntElementsAttribute(pyAttribute));
  std::string msg =
      std::string(
          "Can't cast unknown element type DenseIntOrFPElementsAttr (") +
      nanobind::cast<std::string>(nanobind::repr(nanobind::cast(pyAttribute))) +
      ")";
  throw nanobind::type_error(msg.c_str());
}

nanobind::object integerOrBoolAttributeCaster(PyAttribute &pyAttribute) {
  if (PyBoolAttribute::isaFunction(pyAttribute))
    return nanobind::cast(PyBoolAttribute(pyAttribute));
  if (PyIntegerAttribute::isaFunction(pyAttribute))
    return nanobind::cast(PyIntegerAttribute(pyAttribute));
  std::string msg =
      std::string("Can't cast unknown attribute type Attr (") +
      nanobind::cast<std::string>(nanobind::repr(nanobind::cast(pyAttribute))) +
      ")";
  throw nanobind::type_error(msg.c_str());
}

nanobind::object
symbolRefOrFlatSymbolRefAttributeCaster(PyAttribute &pyAttribute) {
  if (PyFlatSymbolRefAttribute::isaFunction(pyAttribute))
    return nanobind::cast(PyFlatSymbolRefAttribute(pyAttribute));
  if (PySymbolRefAttribute::isaFunction(pyAttribute))
    return nanobind::cast(PySymbolRefAttribute(pyAttribute));
  std::string msg =
      std::string("Can't cast unknown SymbolRef attribute (") +
      nanobind::cast<std::string>(nanobind::repr(nanobind::cast(pyAttribute))) +
      ")";
  throw nanobind::type_error(msg.c_str());
}
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir
