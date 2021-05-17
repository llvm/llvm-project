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
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/Twine.h"

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

template <typename T>
struct type_caster<llvm::Optional<T>> : optional_caster<llvm::Optional<T>> {};
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
  PyFileAccumulator(pybind11::object fileObject, bool binary)
      : pyWriteFunction(fileObject.attr("write")), binary(binary) {}

  void *getUserData() { return this; }

  MlirStringCallback getCallback() {
    return [](MlirStringRef part, void *userData) {
      pybind11::gil_scoped_acquire();
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
///   - an instance method `intptr_t getNumElements()` that returns the number
///     of elements in the backing container (NOT that of the slice);
///   - an instance method `ElementTy getElement(intptr_t)` that returns a
///     single element at the given index.
///   - an instance method `Derived slice(intptr_t, intptr_t, intptr_t)` that
///     constructs a new instance of the derived pseudo-container with the
///     given slice parameters (to be forwarded to the Sliceable constructor).
///
/// A derived class may additionally define:
///   - a `static void bindDerived(ClassTy &)` method to bind additional methods
///     the python class.
template <typename Derived, typename ElementTy>
class Sliceable {
protected:
  using ClassTy = pybind11::class_<Derived>;

  intptr_t wrapIndex(intptr_t index) {
    if (index < 0)
      index = length + index;
    if (index < 0 || index >= length) {
      throw python::SetPyError(PyExc_IndexError,
                               "attempt to access out of bounds");
    }
    return index;
  }

public:
  explicit Sliceable(intptr_t startIndex, intptr_t length, intptr_t step)
      : startIndex(startIndex), length(length), step(step) {
    assert(length >= 0 && "expected non-negative slice length");
  }

  /// Returns the length of the slice.
  intptr_t dunderLen() const { return length; }

  /// Returns the element at the given slice index. Supports negative indices
  /// by taking elements in inverse order. Throws if the index is out of bounds.
  ElementTy dunderGetItem(intptr_t index) {
    // Negative indices mean we count from the end.
    index = wrapIndex(index);

    // Compute the linear index given the current slice properties.
    int linearIndex = index * step + startIndex;
    assert(linearIndex >= 0 &&
           linearIndex < static_cast<Derived *>(this)->getNumElements() &&
           "linear index out of bounds, the slice is ill-formed");
    return static_cast<Derived *>(this)->getElement(linearIndex);
  }

  /// Returns a new instance of the pseudo-container restricted to the given
  /// slice.
  Derived dunderGetItemSlice(pybind11::slice slice) {
    ssize_t start, stop, extraStep, sliceLength;
    if (!slice.compute(dunderLen(), &start, &stop, &extraStep, &sliceLength)) {
      throw python::SetPyError(PyExc_IndexError,
                               "attempt to access out of bounds");
    }
    return static_cast<Derived *>(this)->slice(startIndex + start * step,
                                               sliceLength, step * extraStep);
  }

  /// Binds the indexing and length methods in the Python class.
  static void bind(pybind11::module &m) {
    auto clazz = pybind11::class_<Derived>(m, Derived::pyClassName)
                     .def("__len__", &Sliceable::dunderLen)
                     .def("__getitem__", &Sliceable::dunderGetItem)
                     .def("__getitem__", &Sliceable::dunderGetItemSlice);
    Derived::bindDerived(clazz);
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
