//===- MainModule.cpp - Main pybind module --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Globals.h"
#include "IRModule.h"
#include "NanobindUtils.h"
#include "Pass.h"
#include "Rewrite.h"
#include "mlir/Bindings/Python/Nanobind.h"

namespace nb = nanobind;
using namespace mlir;
using namespace nb::literals;
using namespace mlir::python;

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------

NB_MODULE(_mlir, m) {
  m.doc() = "MLIR Python Native Extension";

  nb::class_<PyGlobals>(m, "_Globals")
      .def_prop_rw("dialect_search_modules",
                   &PyGlobals::getDialectSearchPrefixes,
                   &PyGlobals::setDialectSearchPrefixes)
      .def("append_dialect_search_prefix", &PyGlobals::addDialectSearchPrefix,
           "module_name"_a)
      .def(
          "_check_dialect_module_loaded",
          [](PyGlobals &self, const std::string &dialectNamespace) {
            return self.loadDialectModule(dialectNamespace);
          },
          "dialect_namespace"_a)
      .def("_register_dialect_impl", &PyGlobals::registerDialectImpl,
           "dialect_namespace"_a, "dialect_class"_a,
           "Testing hook for directly registering a dialect")
      .def("_register_operation_impl", &PyGlobals::registerOperationImpl,
           "operation_name"_a, "operation_class"_a, nb::kw_only(),
           "replace"_a = false,
           "Testing hook for directly registering an operation")
      .def("loc_tracebacks_enabled",
           [](PyGlobals &self) {
             return self.getTracebackLoc().locTracebacksEnabled();
           })
      .def("set_loc_tracebacks_enabled",
           [](PyGlobals &self, bool enabled) {
             self.getTracebackLoc().setLocTracebacksEnabled(enabled);
           })
      .def("set_loc_tracebacks_frame_limit",
           [](PyGlobals &self, int n) {
             self.getTracebackLoc().setLocTracebackFramesLimit(n);
           })
      .def("register_traceback_file_inclusion",
           [](PyGlobals &self, const std::string &filename) {
             self.getTracebackLoc().registerTracebackFileInclusion(filename);
           })
      .def("register_traceback_file_exclusion",
           [](PyGlobals &self, const std::string &filename) {
             self.getTracebackLoc().registerTracebackFileExclusion(filename);
           });

  // Aside from making the globals accessible to python, having python manage
  // it is necessary to make sure it is destroyed (and releases its python
  // resources) properly.
  m.attr("globals") = nb::cast(new PyGlobals, nb::rv_policy::take_ownership);

  // Registration decorators.
  m.def(
      "register_dialect",
      [](nb::type_object pyClass) {
        std::string dialectNamespace =
            nanobind::cast<std::string>(pyClass.attr("DIALECT_NAMESPACE"));
        PyGlobals::get().registerDialectImpl(dialectNamespace, pyClass);
        return pyClass;
      },
      "dialect_class"_a,
      "Class decorator for registering a custom Dialect wrapper");
  m.def(
      "register_operation",
      [](const nb::type_object &dialectClass, bool replace) -> nb::object {
        return nb::cpp_function(
            [dialectClass,
             replace](nb::type_object opClass) -> nb::type_object {
              std::string operationName =
                  nanobind::cast<std::string>(opClass.attr("OPERATION_NAME"));
              PyGlobals::get().registerOperationImpl(operationName, opClass,
                                                     replace);
              // Dict-stuff the new opClass by name onto the dialect class.
              nb::object opClassName = opClass.attr("__name__");
              dialectClass.attr(opClassName) = opClass;
              return opClass;
            });
      },
      "dialect_class"_a, nb::kw_only(), "replace"_a = false,
      "Produce a class decorator for registering an Operation class as part of "
      "a dialect");
  m.def(
      MLIR_PYTHON_CAPI_TYPE_CASTER_REGISTER_ATTR,
      [](MlirTypeID mlirTypeID, bool replace) -> nb::object {
        return nb::cpp_function([mlirTypeID, replace](
                                    nb::callable typeCaster) -> nb::object {
          PyGlobals::get().registerTypeCaster(mlirTypeID, typeCaster, replace);
          return typeCaster;
        });
      },
      "typeid"_a, nb::kw_only(), "replace"_a = false,
      "Register a type caster for casting MLIR types to custom user types.");
  m.def(
      MLIR_PYTHON_CAPI_VALUE_CASTER_REGISTER_ATTR,
      [](MlirTypeID mlirTypeID, bool replace) -> nb::object {
        return nb::cpp_function(
            [mlirTypeID, replace](nb::callable valueCaster) -> nb::object {
              PyGlobals::get().registerValueCaster(mlirTypeID, valueCaster,
                                                   replace);
              return valueCaster;
            });
      },
      "typeid"_a, nb::kw_only(), "replace"_a = false,
      "Register a value caster for casting MLIR values to custom user values.");

  // Define and populate IR submodule.
  auto irModule = m.def_submodule("ir", "MLIR IR Bindings");
  populateIRCore(irModule);
  populateIRAffine(irModule);
  populateIRAttributes(irModule);
  populateIRInterfaces(irModule);
  populateIRTypes(irModule);

  auto rewriteModule = m.def_submodule("rewrite", "MLIR Rewrite Bindings");
  populateRewriteSubmodule(rewriteModule);

  // Define and populate PassManager submodule.
  auto passModule =
      m.def_submodule("passmanager", "MLIR Pass Management Bindings");
  populatePassManagerSubmodule(passModule);
}
