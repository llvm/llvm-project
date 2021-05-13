//===- Dialects.h - Declaration for dialect submodule factories -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_BINDINGS_PYTHON_DIALECTS_H
#define MLIR_BINDINGS_PYTHON_DIALECTS_H

#include <pybind11/pybind11.h>

namespace mlir {
namespace python {

void populateDialectLinalgSubmodule(pybind11::module m);
void populateDialectSparseTensorSubmodule(pybind11::module m,
                                          const pybind11::module &irModule);

} // namespace python
} // namespace mlir

#endif // MLIR_BINDINGS_PYTHON_DIALECTS_H
