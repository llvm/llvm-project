//===- IRModules.cpp - IR Submodules of pybind module ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IRModules.h"
#include "PybindUtils.h"

namespace py = pybind11;
using namespace mlir::python;

//------------------------------------------------------------------------------
// Docstrings (trivial, non-duplicated docstrings are included inline).
//------------------------------------------------------------------------------

static const char kContextParseDocstring[] =
    R"(Parses a module's assembly format from a string.

Returns a new MlirModule or raises a ValueError if the parsing fails.
)";

static const char kOperationStrDunderDocstring[] =
    R"(Prints the assembly form of the operation with default options.

If more advanced control over the assembly formatting or I/O options is needed,
use the dedicated print method, which supports keyword arguments to customize
behavior.
)";

static const char kDumpDocstring[] =
    R"(Dumps a debug representation of the object to stderr.)";

//------------------------------------------------------------------------------
// Conversion utilities.
//------------------------------------------------------------------------------

namespace {

/// Accumulates into a python string from a method that accepts an
/// MlirPrintCallback.
struct PyPrintAccumulator {
  py::list parts;

  void *getUserData() { return this; }

  MlirPrintCallback getCallback() {
    return [](const char *part, intptr_t size, void *userData) {
      PyPrintAccumulator *printAccum =
          static_cast<PyPrintAccumulator *>(userData);
      py::str pyPart(part, size); // Decodes as UTF-8 by default.
      printAccum->parts.append(std::move(pyPart));
    };
  }

  py::str join() {
    py::str delim("", 0);
    return delim.attr("join")(parts);
  }
};

} // namespace

//------------------------------------------------------------------------------
// Context Wrapper Class.
//------------------------------------------------------------------------------

PyMlirModule PyMlirContext::parse(const std::string &module) {
  auto moduleRef = mlirModuleCreateParse(context, module.c_str());
  if (!moduleRef.ptr) {
    throw SetPyError(PyExc_ValueError,
                     "Unable to parse module assembly (see diagnostics)");
  }
  return PyMlirModule(moduleRef);
}

//------------------------------------------------------------------------------
// Module Wrapper Class.
//------------------------------------------------------------------------------

void PyMlirModule::dump() { mlirOperationDump(mlirModuleGetOperation(module)); }

//------------------------------------------------------------------------------
// Populates the pybind11 IR submodule.
//------------------------------------------------------------------------------

void mlir::python::populateIRSubmodule(py::module &m) {
  py::class_<PyMlirContext>(m, "MlirContext")
      .def(py::init<>())
      .def("parse", &PyMlirContext::parse, py::keep_alive<0, 1>(),
           kContextParseDocstring);

  py::class_<PyMlirModule>(m, "MlirModule")
      .def("dump", &PyMlirModule::dump, kDumpDocstring)
      .def(
          "__str__",
          [](PyMlirModule &self) {
            auto operation = mlirModuleGetOperation(self.module);
            PyPrintAccumulator printAccum;
            mlirOperationPrint(operation, printAccum.getCallback(),
                               printAccum.getUserData());
            return printAccum.join();
          },
          kOperationStrDunderDocstring);
}
