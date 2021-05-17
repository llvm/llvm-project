//===- MainModule.cpp - Main pybind module --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <tuple>

#include "PybindUtils.h"

#include "Dialects.h"
#include "ExecutionEngine.h"
#include "Globals.h"
#include "IRModule.h"
#include "Pass.h"

namespace py = pybind11;
using namespace mlir;
using namespace mlir::python;

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------

PYBIND11_MODULE(_mlir, m) {
  m.doc() = "MLIR Python Native Extension";

  py::class_<PyGlobals>(m, "_Globals")
      .def_property("dialect_search_modules",
                    &PyGlobals::getDialectSearchPrefixes,
                    &PyGlobals::setDialectSearchPrefixes)
      .def("append_dialect_search_prefix",
           [](PyGlobals &self, std::string moduleName) {
             self.getDialectSearchPrefixes().push_back(std::move(moduleName));
             self.clearImportCache();
           })
      .def("_register_dialect_impl", &PyGlobals::registerDialectImpl,
           "Testing hook for directly registering a dialect")
      .def("_register_operation_impl", &PyGlobals::registerOperationImpl,
           "Testing hook for directly registering an operation");

  // Aside from making the globals accessible to python, having python manage
  // it is necessary to make sure it is destroyed (and releases its python
  // resources) properly.
  m.attr("globals") =
      py::cast(new PyGlobals, py::return_value_policy::take_ownership);

  // Registration decorators.
  m.def(
      "register_dialect",
      [](py::object pyClass) {
        std::string dialectNamespace =
            pyClass.attr("DIALECT_NAMESPACE").cast<std::string>();
        PyGlobals::get().registerDialectImpl(dialectNamespace, pyClass);
        return pyClass;
      },
      "Class decorator for registering a custom Dialect wrapper");
  m.def(
      "register_operation",
      [](py::object dialectClass) -> py::cpp_function {
        return py::cpp_function(
            [dialectClass](py::object opClass) -> py::object {
              std::string operationName =
                  opClass.attr("OPERATION_NAME").cast<std::string>();
              auto rawSubclass = PyOpView::createRawSubclass(opClass);
              PyGlobals::get().registerOperationImpl(operationName, opClass,
                                                     rawSubclass);

              // Dict-stuff the new opClass by name onto the dialect class.
              py::object opClassName = opClass.attr("__name__");
              dialectClass.attr(opClassName) = opClass;

              // Now create a special "Raw" subclass that passes through
              // construction to the OpView parent (bypasses the intermediate
              // child's __init__).
              opClass.attr("_Raw") = rawSubclass;
              return opClass;
            });
      },
      "Class decorator for registering a custom Operation wrapper");

  // Define and populate IR submodule.
  auto irModule = m.def_submodule("ir", "MLIR IR Bindings");
  populateIRCore(irModule);
  populateIRAffine(irModule);
  populateIRAttributes(irModule);
  populateIRTypes(irModule);

  // Define and populate PassManager submodule.
  auto passModule =
      m.def_submodule("passmanager", "MLIR Pass Management Bindings");
  populatePassManagerSubmodule(passModule);

  // Define and populate ExecutionEngine submodule.
  auto executionEngineModule =
      m.def_submodule("execution_engine", "MLIR JIT Execution Engine");
  populateExecutionEngineSubmodule(executionEngineModule);

  // Define and populate dialect submodules.
  auto dialectsModule = m.def_submodule("dialects");
  auto linalgModule = dialectsModule.def_submodule("linalg");
  populateDialectLinalgSubmodule(linalgModule);
  populateDialectSparseTensorSubmodule(
      dialectsModule.def_submodule("sparse_tensor"), irModule);
}
