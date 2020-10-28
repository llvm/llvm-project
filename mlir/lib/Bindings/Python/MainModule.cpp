//===- MainModule.cpp - Main pybind module --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <tuple>

#include "PybindUtils.h"

#include "Globals.h"
#include "IRModules.h"

namespace py = pybind11;
using namespace mlir;
using namespace mlir::python;

// -----------------------------------------------------------------------------
// PyGlobals
// -----------------------------------------------------------------------------

PyGlobals *PyGlobals::instance = nullptr;

PyGlobals::PyGlobals() {
  assert(!instance && "PyGlobals already constructed");
  instance = this;
}

PyGlobals::~PyGlobals() { instance = nullptr; }

void PyGlobals::loadDialectModule(const std::string &dialectNamespace) {
  if (loadedDialectModules.contains(dialectNamespace))
    return;
  // Since re-entrancy is possible, make a copy of the search prefixes.
  std::vector<std::string> localSearchPrefixes = dialectSearchPrefixes;
  py::object loaded;
  for (std::string moduleName : localSearchPrefixes) {
    moduleName.push_back('.');
    moduleName.append(dialectNamespace);

    try {
      loaded = py::module::import(moduleName.c_str());
    } catch (py::error_already_set &e) {
      if (e.matches(PyExc_ModuleNotFoundError)) {
        continue;
      } else {
        throw;
      }
    }
    break;
  }

  // Note: Iterator cannot be shared from prior to loading, since re-entrancy
  // may have occurred, which may do anything.
  loadedDialectModules.insert(dialectNamespace);
}

void PyGlobals::registerDialectImpl(const std::string &dialectNamespace,
                                    py::object pyClass) {
  py::object &found = dialectClassMap[dialectNamespace];
  if (found) {
    throw SetPyError(PyExc_RuntimeError, llvm::Twine("Dialect namespace '") +
                                             dialectNamespace +
                                             "' is already registered.");
  }
  found = std::move(pyClass);
}

void PyGlobals::registerOperationImpl(const std::string &operationName,
                                      py::object pyClass, py::object rawClass) {
  py::object &found = operationClassMap[operationName];
  if (found) {
    throw SetPyError(PyExc_RuntimeError, llvm::Twine("Operation '") +
                                             operationName +
                                             "' is already registered.");
  }
  found = std::move(pyClass);
  rawOperationClassMap[operationName] = std::move(rawClass);
}

llvm::Optional<py::object>
PyGlobals::lookupDialectClass(const std::string &dialectNamespace) {
  loadDialectModule(dialectNamespace);
  // Fast match against the class map first (common case).
  const auto foundIt = dialectClassMap.find(dialectNamespace);
  if (foundIt != dialectClassMap.end()) {
    if (foundIt->second.is_none())
      return llvm::None;
    assert(foundIt->second && "py::object is defined");
    return foundIt->second;
  }

  // Not found and loading did not yield a registration. Negative cache.
  dialectClassMap[dialectNamespace] = py::none();
  return llvm::None;
}

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
  populateIRSubmodule(irModule);
}
