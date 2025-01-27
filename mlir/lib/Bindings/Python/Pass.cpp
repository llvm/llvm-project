//===- Pass.cpp - Pass Management -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Pass.h"

#include "IRModule.h"
#include "mlir-c/Pass.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir-c/Bindings/Python/Interop.h" // This is expected after nanobind.

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

  static nb::object createFromCapsule(nb::object capsule) {
    MlirPassManager rawPm = mlirPythonCapsuleToPassManager(capsule.ptr());
    if (mlirPassManagerIsNull(rawPm))
      throw nb::python_error();
    return nb::cast(PyPassManager(rawPm), nb::rv_policy::move);
  }

private:
  MlirPassManager passManager;
};

} // namespace

class PythonPass {
public:
  explicit PythonPass(py::object passObj) : passObj(std::move(passObj)) {}

  void *construct() {}
  void *destruct() {}

  MlirLogicalResult *initialize(MlirContext ctx) {}
  void *clone() {}

  void run(MlirOperation op, MlirExternalPass pass) {}

  py::object passObj;
};

template <typename T, typename R>
void *void_cast(R (T::*f)()) {
  union {
    R (T::*pf)();
    void *p;
  };
  pf = f;
  return p;
}

template <typename classT, typename memberT>
union u_ptm_cast {
  memberT pmember;
  void *pvoid;
};

MlirExternalPassCallbacks makeTestExternalPassCallbacks() {
  return (MlirExternalPassCallbacks){
      reinterpret_cast<decltype(MlirExternalPassCallbacks::construct)>(
          void_cast(&PythonPass::construct)),
      reinterpret_cast<decltype(MlirExternalPassCallbacks::destruct)>(
          void_cast(&PythonPass::destruct)),
      nullptr,
      reinterpret_cast<decltype(MlirExternalPassCallbacks::clone)>(
          void_cast(&PythonPass::clone)),
      reinterpret_cast<decltype(MlirExternalPassCallbacks::run)>(
          u_ptm_cast<PythonPass,
                     void (PythonPass::*)(MlirOperation, MlirExternalPass)>{
              &PythonPass::run}
              .pvoid),

  };
}

/// Create the `mlir.passmanager` here.
void mlir::python::populatePassManagerSubmodule(nb::module_ &m) {
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
          "anchor_op"_a = nb::str("any"), "context"_a.none() = nb::none(),
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
             bool enableDebugInfo, bool printGenericOpForm,
             std::optional<std::string> optionalTreePrintingPath) {
            MlirOpPrintingFlags flags = mlirOpPrintingFlagsCreate();
            if (largeElementsLimit)
              mlirOpPrintingFlagsElideLargeElementsAttrs(flags,
                                                         *largeElementsLimit);
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
          "large_elements_limit"_a.none() = nb::none(),
          "enable_debug_info"_a = false, "print_generic_op_form"_a = false,
          "tree_printing_dir_path"_a.none() = nb::none(),
          "Enable IR printing, default as mlir-print-ir-after-all.")
      .def(
          "enable_verifier",
          [](PyPassManager &passManager, bool enable) {
            mlirPassManagerEnableVerifier(passManager.get(), enable);
          },
          "enable"_a, "Enable / disable verify-each.")
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
          "pipeline"_a, "context"_a.none() = nb::none(),
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
      .def_static(
          "create_external_pass",
          [](py::object &passObj) {
            PythonPass pass = PythonPass(passObj);

            MlirTypeIDAllocator typeIDAllocator = mlirTypeIDAllocatorCreate();
            MlirTypeID passID =
                mlirTypeIDAllocatorAllocateTypeID(typeIDAllocator);
            MlirStringRef name =
                mlirStringRefCreateFromCString("TestExternalPass");
            MlirStringRef description = mlirStringRefCreateFromCString("");
            MlirStringRef emptyOpName = mlirStringRefCreateFromCString("");
            MlirStringRef argument =
                mlirStringRefCreateFromCString("test-external-pass");

            auto cbs = makeTestExternalPassCallbacks();

            MlirPass externalPass =
                mlirCreateExternalPass(passID, name, argument, description,
                                       emptyOpName, 0, NULL, cbs, &pass);
          })
      .def(
          "run",
          [](PyPassManager &passManager, PyOperationBase &op,
             bool invalidateOps) {
            if (invalidateOps) {
              op.getOperation().getContext()->clearOperationsInside(op);
            }
            // Actually run the pass manager.
            PyMlirContext::ErrorCapture errors(op.getOperation().getContext());
            MlirLogicalResult status = mlirPassManagerRunOnOp(
                passManager.get(), op.getOperation().get());
            if (mlirLogicalResultIsFailure(status))
              throw MLIRError("Failure while executing pass pipeline",
                              errors.take());
          },
          "operation"_a, "invalidate_ops"_a = true,
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

  py::class_<PythonPass>(m, "PythonPass", py::module_local())
      .def(py::init<>(
          [](py::object pass) { return PythonPass(std::move(pass)); }));
}
