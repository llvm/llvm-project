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

#include "llvm/ADT/Twine.h"

namespace mlir {
namespace python {

// Sets a python error, ready to be thrown to return control back to the
// python runtime.
// Correct usage:
//   throw SetPyError(PyExc_ValueError, "Foobar'd");
pybind11::error_already_set SetPyError(PyObject *excClass,
                                       const llvm::Twine &message);

} // namespace python
} // namespace mlir

#endif // MLIR_BINDINGS_PYTHON_PYBINDUTILS_H
