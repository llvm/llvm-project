//===- IRModules.cpp - IR Submodules of pybind module ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IRModules.h"

//------------------------------------------------------------------------------
// Context Wrapper Class.
//------------------------------------------------------------------------------

PyMlirModule PyMlirContext::parse(const std::string &module) {
  auto moduleRef = mlirModuleCreateParse(context, module.c_str());
  return PyMlirModule(moduleRef);
}

//------------------------------------------------------------------------------
// Module Wrapper Class.
//------------------------------------------------------------------------------

void PyMlirModule::dump() { mlirOperationDump(mlirModuleGetOperation(module)); }

//------------------------------------------------------------------------------
// Populates the pybind11 IR submodule.
//------------------------------------------------------------------------------

void populateIRSubmodule(py::module &m) {
  py::class_<PyMlirContext>(m, "MlirContext")
      .def(py::init<>())
      .def("parse", &PyMlirContext::parse, py::keep_alive<0, 1>());

  py::class_<PyMlirModule>(m, "MlirModule").def("dump", &PyMlirModule::dump);
}
