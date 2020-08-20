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
class PyModule;

/// Wrapper around MlirContext.
class PyMlirContext {
public:
  PyMlirContext() { context = mlirContextCreate(); }
  ~PyMlirContext() { mlirContextDestroy(context); }

  MlirContext context;
};

/// Wrapper around MlirModule.
class PyModule {
public:
  PyModule(MlirModule module) : module(module) {}
  PyModule(PyModule &) = delete;
  PyModule(PyModule &&other) {
    module = other.module;
    other.module.ptr = nullptr;
  }
  ~PyModule() {
    if (module.ptr)
      mlirModuleDestroy(module);
  }

  MlirModule module;
};

/// Wrapper around the generic MlirType.
/// The lifetime of a type is bound by the PyContext that created it.
class PyType {
public:
  PyType(MlirType type) : type(type) {}
  bool operator==(const PyType &other);

  MlirType type;
};

void populateIRSubmodule(pybind11::module &m);

} // namespace python
} // namespace mlir

#endif // MLIR_BINDINGS_PYTHON_IRMODULES_H
