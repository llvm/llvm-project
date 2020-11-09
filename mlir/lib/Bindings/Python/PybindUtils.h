//===- PybindUtils.h - Utilities for interop with pybind11 ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_BINDINGS_PYTHON_PYBINDUTILS_H
#define MLIR_BINDINGS_PYTHON_PYBINDUTILS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/Twine.h"

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

  ReferrentTy *get() { return referrent; }
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
    } catch (std::exception &e) {
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

#endif // MLIR_BINDINGS_PYTHON_PYBINDUTILS_H
