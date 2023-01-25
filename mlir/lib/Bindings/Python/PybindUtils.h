//===- PybindUtils.h - Utilities for interop with pybind11 ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_BINDINGS_PYTHON_PYBINDUTILS_H
#define MLIR_BINDINGS_PYTHON_PYBINDUTILS_H

#include "mlir-c/Support.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/DataTypes.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace mlir {
namespace python {

// Sets a python error, ready to be thrown to return control back to the
// python runtime.
// Correct usage:
//   throw SetPyError(PyExc_ValueError, "Foobar'd");
pybind11::error_already_set SetPyError(PyObject *excClass,
                                       const llvm::Twine &message);

/// CRTP template for special wrapper types that are allowed to be passed in as
/// 'None' function arguments and can be resolved by some global mechanic if
/// so. Such types will raise an error if this global resolution fails, and
/// it is actually illegal for them to ever be unresolved. From a user
/// perspective, they behave like a smart ptr to the underlying type (i.e.
/// 'get' method and operator-> overloaded).
///
/// Derived types must provide a method, which is called when an environmental
/// resolution is required. It must raise an exception if resolution fails:
///   static ReferrentTy &resolve()
///
/// They must also provide a parameter description that will be used in
/// error messages about mismatched types:
///   static constexpr const char kTypeDescription[] = "<Description>";

template <typename DerivedTy, typename T>
class Defaulting {
public:
  using ReferrentTy = T;
  /// Type casters require the type to be default constructible, but using
  /// such an instance is illegal.
  Defaulting() = default;
  Defaulting(ReferrentTy &referrent) : referrent(&referrent) {}

  ReferrentTy *get() const { return referrent; }
  ReferrentTy *operator->() { return referrent; }

private:
  ReferrentTy *referrent = nullptr;
};

} // namespace python
} // namespace mlir

namespace pybind11 {
namespace detail {

template <typename DefaultingTy>
struct MlirDefaultingCaster {
  PYBIND11_TYPE_CASTER(DefaultingTy, _(DefaultingTy::kTypeDescription));

  bool load(pybind11::handle src, bool) {
    if (src.is_none()) {
      // Note that we do want an exception to propagate from here as it will be
      // the most informative.
      value = DefaultingTy{DefaultingTy::resolve()};
      return true;
    }

    // Unlike many casters that chain, these casters are expected to always
    // succeed, so instead of doing an isinstance check followed by a cast,
    // just cast in one step and handle the exception. Returning false (vs
    // letting the exception propagate) causes higher level signature parsing
    // code to produce nice error messages (other than "Cannot cast...").
    try {
      value = DefaultingTy{
          pybind11::cast<typename DefaultingTy::ReferrentTy &>(src)};
      return true;
    } catch (std::exception &) {
      return false;
    }
  }

  static handle cast(DefaultingTy src, return_value_policy policy,
                     handle parent) {
    return pybind11::cast(src, policy);
  }
};
} // namespace detail
} // namespace pybind11

//------------------------------------------------------------------------------
// Conversion utilities.
//------------------------------------------------------------------------------

namespace mlir {

/// Accumulates into a python string from a method that accepts an
/// MlirStringCallback.
struct PyPrintAccumulator {
  pybind11::list parts;

  void *getUserData() { return this; }

  MlirStringCallback getCallback() {
    return [](MlirStringRef part, void *userData) {
      PyPrintAccumulator *printAccum =
          static_cast<PyPrintAccumulator *>(userData);
      pybind11::str pyPart(part.data,
                           part.length); // Decodes as UTF-8 by default.
      printAccum->parts.append(std::move(pyPart));
    };
  }

  pybind11::str join() {
    pybind11::str delim("", 0);
    return delim.attr("join")(parts);
  }
};

/// Accumulates int a python file-like object, either writing text (default)
/// or binary.
class PyFileAccumulator {
public:
  PyFileAccumulator(const pybind11::object &fileObject, bool binary)
      : pyWriteFunction(fileObject.attr("write")), binary(binary) {}

  void *getUserData() { return this; }

  MlirStringCallback getCallback() {
    return [](MlirStringRef part, void *userData) {
      pybind11::gil_scoped_acquire acquire;
      PyFileAccumulator *accum = static_cast<PyFileAccumulator *>(userData);
      if (accum->binary) {
        // Note: Still has to copy and not avoidable with this API.
        pybind11::bytes pyBytes(part.data, part.length);
        accum->pyWriteFunction(pyBytes);
      } else {
        pybind11::str pyStr(part.data,
                            part.length); // Decodes as UTF-8 by default.
        accum->pyWriteFunction(pyStr);
      }
    };
  }

private:
  pybind11::object pyWriteFunction;
  bool binary;
};

/// Accumulates into a python string from a method that is expected to make
/// one (no more, no less) call to the callback (asserts internally on
/// violation).
struct PySinglePartStringAccumulator {
  void *getUserData() { return this; }

  MlirStringCallback getCallback() {
    return [](MlirStringRef part, void *userData) {
      PySinglePartStringAccumulator *accum =
          static_cast<PySinglePartStringAccumulator *>(userData);
      assert(!accum->invoked &&
             "PySinglePartStringAccumulator called back multiple times");
      accum->invoked = true;
      accum->value = pybind11::str(part.data, part.length);
    };
  }

  pybind11::str takeValue() {
    assert(invoked && "PySinglePartStringAccumulator not called back");
    return std::move(value);
  }

private:
  pybind11::str value;
  bool invoked = false;
};

/// A CRTP base class for pseudo-containers willing to support Python-type
/// slicing access on top of indexed access. Calling ::bind on this class
/// will define `__len__` as well as `__getitem__` with integer and slice
/// arguments.
///
/// This is intended for pseudo-containers that can refer to arbitrary slices of
/// underlying storage indexed by a single integer. Indexing those with an
/// integer produces an instance of ElementTy. Indexing those with a slice
/// produces a new instance of Derived, which can be sliced further.
///
/// A derived class must provide the following:
///   - a `static const char *pyClassName ` field containing the name of the
///     Python class to bind;
///   - an instance method `intptr_t getRawNumElements()` that returns the
///   number
///     of elements in the backing container (NOT that of the slice);
///   - an instance method `ElementTy getRawElement(intptr_t)` that returns a
///     single element at the given linear index (NOT slice index);
///   - an instance method `Derived slice(intptr_t, intptr_t, intptr_t)` that
///     constructs a new instance of the derived pseudo-container with the
///     given slice parameters (to be forwarded to the Sliceable constructor).
///
/// The getRawNumElements() and getRawElement(intptr_t) callbacks must not
/// throw.
///
/// A derived class may additionally define:
///   - a `static void bindDerived(ClassTy &)` method to bind additional methods
///     the python class.
template <typename Derived, typename ElementTy>
class Sliceable {
protected:
  using ClassTy = pybind11::class_<Derived>;

  /// Transforms `index` into a legal value to access the underlying sequence.
  /// Returns <0 on failure.
  intptr_t wrapIndex(intptr_t index) {
    if (index < 0)
      index = length + index;
    if (index < 0 || index >= length)
      return -1;
    return index;
  }

  /// Computes the linear index given the current slice properties.
  intptr_t linearizeIndex(intptr_t index) {
    intptr_t linearIndex = index * step + startIndex;
    assert(linearIndex >= 0 &&
           linearIndex < static_cast<Derived *>(this)->getRawNumElements() &&
           "linear index out of bounds, the slice is ill-formed");
    return linearIndex;
  }

  /// Returns the element at the given slice index. Supports negative indices
  /// by taking elements in inverse order. Returns a nullptr object if out
  /// of bounds.
  pybind11::object getItem(intptr_t index) {
    // Negative indices mean we count from the end.
    index = wrapIndex(index);
    if (index < 0) {
      PyErr_SetString(PyExc_IndexError, "index out of range");
      return {};
    }

    return pybind11::cast(
        static_cast<Derived *>(this)->getRawElement(linearizeIndex(index)));
  }

  /// Returns a new instance of the pseudo-container restricted to the given
  /// slice. Returns a nullptr object on failure.
  pybind11::object getItemSlice(PyObject *slice) {
    ssize_t start, stop, extraStep, sliceLength;
    if (PySlice_GetIndicesEx(slice, length, &start, &stop, &extraStep,
                             &sliceLength) != 0) {
      PyErr_SetString(PyExc_IndexError, "index out of range");
      return {};
    }
    return pybind11::cast(static_cast<Derived *>(this)->slice(
        startIndex + start * step, sliceLength, step * extraStep));
  }

public:
  explicit Sliceable(intptr_t startIndex, intptr_t length, intptr_t step)
      : startIndex(startIndex), length(length), step(step) {
    assert(length >= 0 && "expected non-negative slice length");
  }

  /// Returns the `index`-th element in the slice, supports negative indices.
  /// Throws if the index is out of bounds.
  ElementTy getElement(intptr_t index) {
    // Negative indices mean we count from the end.
    index = wrapIndex(index);
    if (index < 0) {
      throw pybind11::index_error("index out of range");
    }

    return static_cast<Derived *>(this)->getRawElement(linearizeIndex(index));
  }

  /// Returns the size of slice.
  intptr_t size() { return length; }

  /// Returns a new vector (mapped to Python list) containing elements from two
  /// slices. The new vector is necessary because slices may not be contiguous
  /// or even come from the same original sequence.
  std::vector<ElementTy> dunderAdd(Derived &other) {
    std::vector<ElementTy> elements;
    elements.reserve(length + other.length);
    for (intptr_t i = 0; i < length; ++i) {
      elements.push_back(static_cast<Derived *>(this)->getElement(i));
    }
    for (intptr_t i = 0; i < other.length; ++i) {
      elements.push_back(static_cast<Derived *>(&other)->getElement(i));
    }
    return elements;
  }

  /// Binds the indexing and length methods in the Python class.
  static void bind(pybind11::module &m) {
    auto clazz = pybind11::class_<Derived>(m, Derived::pyClassName,
                                           pybind11::module_local())
                     .def("__add__", &Sliceable::dunderAdd);
    Derived::bindDerived(clazz);

    // Manually implement the sequence protocol via the C API. We do this
    // because it is approx 4x faster than via pybind11, largely because that
    // formulation requires a C++ exception to be thrown to detect end of
    // sequence.
    // Since we are in a C-context, any C++ exception that happens here
    // will terminate the program. There is nothing in this implementation
    // that should throw in a non-terminal way, so we forgo further
    // exception marshalling.
    // See: https://github.com/pybind/pybind11/issues/2842
    auto heap_type = reinterpret_cast<PyHeapTypeObject *>(clazz.ptr());
    assert(heap_type->ht_type.tp_flags & Py_TPFLAGS_HEAPTYPE &&
           "must be heap type");
    heap_type->as_sequence.sq_length = +[](PyObject *rawSelf) -> Py_ssize_t {
      auto self = pybind11::cast<Derived *>(rawSelf);
      return self->length;
    };
    // sq_item is called as part of the sequence protocol for iteration,
    // list construction, etc.
    heap_type->as_sequence.sq_item =
        +[](PyObject *rawSelf, Py_ssize_t index) -> PyObject * {
      auto self = pybind11::cast<Derived *>(rawSelf);
      return self->getItem(index).release().ptr();
    };
    // mp_subscript is used for both slices and integer lookups.
    heap_type->as_mapping.mp_subscript =
        +[](PyObject *rawSelf, PyObject *rawSubscript) -> PyObject * {
      auto self = pybind11::cast<Derived *>(rawSelf);
      Py_ssize_t index = PyNumber_AsSsize_t(rawSubscript, PyExc_IndexError);
      if (!PyErr_Occurred()) {
        // Integer indexing.
        return self->getItem(index).release().ptr();
      }
      PyErr_Clear();

      // Assume slice-based indexing.
      if (PySlice_Check(rawSubscript)) {
        return self->getItemSlice(rawSubscript).release().ptr();
      }

      PyErr_SetString(PyExc_ValueError, "expected integer or slice");
      return nullptr;
    };
  }

  /// Hook for derived classes willing to bind more methods.
  static void bindDerived(ClassTy &) {}

private:
  intptr_t startIndex;
  intptr_t length;
  intptr_t step;
};

} // namespace mlir

#endif // MLIR_BINDINGS_PYTHON_PYBINDUTILS_H
