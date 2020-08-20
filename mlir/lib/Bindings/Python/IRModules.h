//===- IRModules.h - IR Submodules of pybind module -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_BINDINGS_PYTHON_IRMODULES_H
#define MLIR_BINDINGS_PYTHON_IRMODULES_H

#include <pybind11/pybind11.h>

#include "mlir-c/IR.h"

namespace mlir {
namespace python {

class PyMlirContext;
class PyMlirModule;

/// Wrapper around MlirContext.
class PyMlirContext {
public:
  PyMlirContext() { context = mlirContextCreate(); }
  ~PyMlirContext() { mlirContextDestroy(context); }
  /// Parses the module from asm.
  PyMlirModule parse(const std::string &module);

  MlirContext context;
};

/// Wrapper around MlirModule.
class PyMlirModule {
public:
  PyMlirModule(MlirModule module) : module(module) {}
  PyMlirModule(PyMlirModule &) = delete;
  PyMlirModule(PyMlirModule &&other) {
    module = other.module;
    other.module.ptr = nullptr;
  }
  ~PyMlirModule() {
    if (module.ptr)
      mlirModuleDestroy(module);
  }
  /// Dumps the module.
  void dump();

  MlirModule module;
};

void populateIRSubmodule(pybind11::module &m);

} // namespace python
} // namespace mlir

#endif // MLIR_BINDINGS_PYTHON_IRMODULES_H
