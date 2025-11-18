//===- Pass.cpp - Pass Management -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Pass.h"

#include "Globals.h"
#include "IRModule.h"
#include "mlir-c/Pass.h"
// clang-format off
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir-c/Bindings/Python/Interop.h" // This is expected after nanobind.
// clang-format on

namespace nb = nanobind;
using namespace nb::literals;
using namespace mlir;
using namespace mlir::python;

namespace {

/// Owning Wrapper around a PassManager.
class PyPassManager {
public:
  PyPassManager(MlirPassManager passManager) : passManager(passManager) {}
  PyPassManager(PyPassManager &&other) noexcept
      : passManager(other.passManager) {
    other.passManager.ptr = nullptr;
  }
  ~PyPassManager() {
    if (!mlirPassManagerIsNull(passManager))
      mlirPassManagerDestroy(passManager);
  }
  MlirPassManager get() { return passManager; }

  void release() { passManager.ptr = nullptr; }
  nb::object getCapsule() {
    return nb::steal<nb::object>(mlirPythonPassManagerToCapsule(get()));
  }

  static nb::object createFromCapsule(const nb::object &capsule) {
    MlirPassManager rawPm = mlirPythonCapsuleToPassManager(capsule.ptr());
    if (mlirPassManagerIsNull(rawPm))
      throw nb::python_error();
    return nb::cast(PyPassManager(rawPm), nb::rv_policy::move);
  }

private:
  MlirPassManager passManager;
};

} // namespace

/// Create the `mlir.passmanager` here.
void mlir::python::populatePassManagerSubmodule(nb::module_ &m) {
  //----------------------------------------------------------------------------
  // Mapping of enumerated types
  //----------------------------------------------------------------------------
  nb::enum_<MlirPassDisplayMode>(m, "PassDisplayMode")
      .value("LIST", MLIR_PASS_DISPLAY_MODE_LIST)
      .value("PIPELINE", MLIR_PASS_DISPLAY_MODE_PIPELINE);

  //----------------------------------------------------------------------------
  // Mapping of MlirExternalPass
  //----------------------------------------------------------------------------
  nb::class_<MlirExternalPass>(m, "ExternalPass")
      .def("signal_pass_failure",
           [](MlirExternalPass pass) { mlirExternalPassSignalFailure(pass); });

  //----------------------------------------------------------------------------
  // Mapping of the top-level PassManager
  //----------------------------------------------------------------------------
  nb::class_<PyPassManager>(m, "PassManager")
      .def(
          "__init__",
          [](PyPassManager &self, const std::string &anchorOp,
             DefaultingPyMlirContext context) {
            MlirPassManager passManager = mlirPassManagerCreateOnOperation(
                context->get(),
                mlirStringRefCreate(anchorOp.data(), anchorOp.size()));
            new (&self) PyPassManager(passManager);
          },
          "anchor_op"_a = nb::str("any"), "context"_a = nb::none(),
          // clang-format off
          nb::sig("def __init__(self, anchor_op: str = 'any', context: " MAKE_MLIR_PYTHON_QUALNAME("ir.Context") " | None = None) -> None"),
          // clang-format on
          "Create a new PassManager for the current (or provided) Context.")
      .def_prop_ro(MLIR_PYTHON_CAPI_PTR_ATTR, &PyPassManager::getCapsule)
      .def(MLIR_PYTHON_CAPI_FACTORY_ATTR, &PyPassManager::createFromCapsule)
      .def("_testing_release", &PyPassManager::release,
           "Releases (leaks) the backing pass manager (testing)")
      .def(
          "enable_ir_printing",
          [](PyPassManager &passManager, bool printBeforeAll,
             bool printAfterAll, bool printModuleScope, bool printAfterChange,
             bool printAfterFailure, std::optional<int64_t> largeElementsLimit,
             std::optional<int64_t> largeResourceLimit, bool enableDebugInfo,
             bool printGenericOpForm,
             std::optional<std::string> optionalTreePrintingPath) {
            MlirOpPrintingFlags flags = mlirOpPrintingFlagsCreate();
            if (largeElementsLimit) {
              mlirOpPrintingFlagsElideLargeElementsAttrs(flags,
                                                         *largeElementsLimit);
              mlirOpPrintingFlagsElideLargeResourceString(flags,
                                                          *largeElementsLimit);
            }
            if (largeResourceLimit)
              mlirOpPrintingFlagsElideLargeResourceString(flags,
                                                          *largeResourceLimit);
            if (enableDebugInfo)
              mlirOpPrintingFlagsEnableDebugInfo(flags, /*enable=*/true,
                                                 /*prettyForm=*/false);
            if (printGenericOpForm)
              mlirOpPrintingFlagsPrintGenericOpForm(flags);
            std::string treePrintingPath = "";
            if (optionalTreePrintingPath.has_value())
              treePrintingPath = optionalTreePrintingPath.value();
            mlirPassManagerEnableIRPrinting(
                passManager.get(), printBeforeAll, printAfterAll,
                printModuleScope, printAfterChange, printAfterFailure, flags,
                mlirStringRefCreate(treePrintingPath.data(),
                                    treePrintingPath.size()));
            mlirOpPrintingFlagsDestroy(flags);
          },
          "print_before_all"_a = false, "print_after_all"_a = true,
          "print_module_scope"_a = false, "print_after_change"_a = false,
          "print_after_failure"_a = false,
          "large_elements_limit"_a = nb::none(),
          "large_resource_limit"_a = nb::none(), "enable_debug_info"_a = false,
          "print_generic_op_form"_a = false,
          "tree_printing_dir_path"_a = nb::none(),
          "Enable IR printing, default as mlir-print-ir-after-all.")
      .def(
          "enable_verifier",
          [](PyPassManager &passManager, bool enable) {
            mlirPassManagerEnableVerifier(passManager.get(), enable);
          },
          "enable"_a, "Enable / disable verify-each.")
      .def(
          "enable_timing",
          [](PyPassManager &passManager) {
            mlirPassManagerEnableTiming(passManager.get());
          },
          "Enable pass timing.")
      .def(
          "enable_statistics",
          [](PyPassManager &passManager, MlirPassDisplayMode displayMode) {
            mlirPassManagerEnableStatistics(passManager.get(), displayMode);
          },
          "displayMode"_a =
              MlirPassDisplayMode::MLIR_PASS_DISPLAY_MODE_PIPELINE,
          "Enable pass statistics.")
      .def_static(
          "parse",
          [](const std::string &pipeline, DefaultingPyMlirContext context) {
            MlirPassManager passManager = mlirPassManagerCreate(context->get());
            PyPrintAccumulator errorMsg;
            MlirLogicalResult status = mlirParsePassPipeline(
                mlirPassManagerGetAsOpPassManager(passManager),
                mlirStringRefCreate(pipeline.data(), pipeline.size()),
                errorMsg.getCallback(), errorMsg.getUserData());
            if (mlirLogicalResultIsFailure(status))
              throw nb::value_error(errorMsg.join().c_str());
            return new PyPassManager(passManager);
          },
          "pipeline"_a, "context"_a = nb::none(),
          // clang-format off
          nb::sig("def parse(pipeline: str, context: " MAKE_MLIR_PYTHON_QUALNAME("ir.Context") " | None = None) -> PassManager"),
          // clang-format on
          "Parse a textual pass-pipeline and return a top-level PassManager "
          "that can be applied on a Module. Throw a ValueError if the pipeline "
          "can't be parsed")
      .def(
          "add",
          [](PyPassManager &passManager, const std::string &pipeline) {
            PyPrintAccumulator errorMsg;
            MlirLogicalResult status = mlirOpPassManagerAddPipeline(
                mlirPassManagerGetAsOpPassManager(passManager.get()),
                mlirStringRefCreate(pipeline.data(), pipeline.size()),
                errorMsg.getCallback(), errorMsg.getUserData());
            if (mlirLogicalResultIsFailure(status))
              throw nb::value_error(errorMsg.join().c_str());
          },
          "pipeline"_a,
          "Add textual pipeline elements to the pass manager. Throws a "
          "ValueError if the pipeline can't be parsed.")
      .def(
          "add",
          [](PyPassManager &passManager, const nb::callable &run,
             std::optional<std::string> &name, const std::string &argument,
             const std::string &description, const std::string &opName) {
            if (!name.has_value()) {
              name = nb::cast<std::string>(
                  nb::borrow<nb::str>(run.attr("__name__")));
            }
            MlirTypeID passID = PyGlobals::get().allocateTypeID();
            MlirExternalPassCallbacks callbacks;
            callbacks.construct = [](void *obj) {
              (void)nb::handle(static_cast<PyObject *>(obj)).inc_ref();
            };
            callbacks.destruct = [](void *obj) {
              (void)nb::handle(static_cast<PyObject *>(obj)).dec_ref();
            };
            callbacks.initialize = nullptr;
            callbacks.clone = [](void *) -> void * {
              throw std::runtime_error("Cloning Python passes not supported");
            };
            callbacks.run = [](MlirOperation op, MlirExternalPass pass,
                               void *userData) {
              nb::handle(static_cast<PyObject *>(userData))(op, pass);
            };
            auto externalPass = mlirCreateExternalPass(
                passID, mlirStringRefCreate(name->data(), name->length()),
                mlirStringRefCreate(argument.data(), argument.length()),
                mlirStringRefCreate(description.data(), description.length()),
                mlirStringRefCreate(opName.data(), opName.size()),
                /*nDependentDialects*/ 0, /*dependentDialects*/ nullptr,
                callbacks, /*userData*/ run.ptr());
            mlirPassManagerAddOwnedPass(passManager.get(), externalPass);
          },
          "run"_a, "name"_a.none() = nb::none(), "argument"_a.none() = "",
          "description"_a.none() = "", "op_name"_a.none() = "",
          "Add a python-defined pass to the pass manager.")
      .def(
          "run",
          [](PyPassManager &passManager, PyOperationBase &op) {
            // Actually run the pass manager.
            PyMlirContext::ErrorCapture errors(op.getOperation().getContext());
            MlirLogicalResult status = mlirPassManagerRunOnOp(
                passManager.get(), op.getOperation().get());
            if (mlirLogicalResultIsFailure(status))
              throw MLIRError("Failure while executing pass pipeline",
                              errors.take());
          },
          "operation"_a,
          // clang-format off
          nb::sig("def run(self, operation: " MAKE_MLIR_PYTHON_QUALNAME("ir._OperationBase") ") -> None"),
          // clang-format on
          "Run the pass manager on the provided operation, raising an "
          "MLIRError on failure.")
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
