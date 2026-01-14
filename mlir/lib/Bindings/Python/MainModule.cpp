//===- MainModule.cpp - Main pybind module --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Pass.h"
#include "Rewrite.h"
#include "mlir/Bindings/Python/Globals.h"
#include "mlir/Bindings/Python/IRAttributes.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/IRTypes.h"
#include "mlir/Bindings/Python/Nanobind.h"

namespace nb = nanobind;
using namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN;

namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {
void populateIRAffine(nb::module_ &m);
void populateIRAttributes(nb::module_ &m);
void populateIRInterfaces(nb::module_ &m);
void populateIRTypes(nb::module_ &m);
void populateIRCore(nb::module_ &m);
void populateRoot(nb::module_ &m);
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------
NB_MODULE(_mlir, m) {
  // disable leak warnings which tend to be false positives.
  nb::set_leak_warnings(false);

  m.doc() = "MLIR Python Native Extension";
  populateRoot(m);
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
  auto passManagerModule =
      m.def_submodule("passmanager", "MLIR Pass Management Bindings");
  populatePassManagerSubmodule(passManagerModule);
  nanobind::register_exception_translator(
      [](const std::exception_ptr &p, void *payload) {
        // We can't define exceptions with custom fields through pybind, so
        // instead the exception class is defined in python and imported here.
        try {
          if (p)
            std::rethrow_exception(p);
        } catch (const MLIRError &e) {
          nanobind::object obj =
              nanobind::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
                  .attr("MLIRError")(e.message, e.errorDiagnostics);
          PyErr_SetObject(PyExc_Exception, obj.ptr());
        }
      });
}
