//===- Pass.cpp - Pass Management -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Pass.h"

#include "IRModules.h"
#include "mlir-c/Pass.h"

namespace py = pybind11;
using namespace mlir;
using namespace mlir::python;

namespace {

/// Owning Wrapper around a PassManager.
class PyPassManager {
public:
  PyPassManager(MlirPassManager passManager) : passManager(passManager) {}
  ~PyPassManager() { mlirPassManagerDestroy(passManager); }
  MlirPassManager get() { return passManager; }

private:
  MlirPassManager passManager;
};

} // anonymous namespace

/// Create the `mlir.passmanager` here.
void mlir::python::populatePassManagerSubmodule(py::module &m) {
  //----------------------------------------------------------------------------
  // Mapping of the top-level PassManager
  //----------------------------------------------------------------------------
  py::class_<PyPassManager>(m, "PassManager")
      .def(py::init<>([](DefaultingPyMlirContext context) {
             MlirPassManager passManager =
                 mlirPassManagerCreate(context->get());
             return new PyPassManager(passManager);
           }),
           py::arg("context") = py::none(),
           "Create a new PassManager for the current (or provided) Context.")
      .def_static(
          "parse",
          [](const std::string pipeline, DefaultingPyMlirContext context) {
            MlirPassManager passManager = mlirPassManagerCreate(context->get());
            MlirLogicalResult status = mlirParsePassPipeline(
                mlirPassManagerGetAsOpPassManager(passManager),
                mlirStringRefCreate(pipeline.data(), pipeline.size()));
            if (mlirLogicalResultIsFailure(status))
              throw SetPyError(PyExc_ValueError,
                               llvm::Twine("invalid pass pipeline '") +
                                   pipeline + "'.");
            return new PyPassManager(passManager);
          },
          py::arg("pipeline"), py::arg("context") = py::none(),
          "Parse a textual pass-pipeline and return a top-level PassManager "
          "that can be applied on a Module. Throw a ValueError if the pipeline "
          "can't be parsed")
      .def(
          "__str__",
          [](PyPassManager &self) {
            MlirPassManager passManager = self.get();
            PyPrintAccumulator printAccum;
            mlirPrintPassPipeline(
                mlirPassManagerGetAsOpPassManager(passManager),
                printAccum.getCallback(), printAccum.getUserData());
            return printAccum.join();
          },
          "Print the textual representation for this PassManager, suitable to "
          "be passed to `parse` for round-tripping.");
}
