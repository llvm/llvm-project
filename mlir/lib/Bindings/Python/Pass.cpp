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
// clang-format off
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir-c/Bindings/Python/Interop.h" // This is expected after nanobind.
// clang-format on
#include "nanobind/trampoline.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace mlir;
using namespace mlir::python;

namespace {

// A base class for defining passes in Python
// Users are expected to subclass this and implement the `run` method, e.g.
// ```
// class MyPass(mlir.passmanager.Pass):
//   def __init__(self):
//     super().__init__("MyPass", ..)
//     # other init stuff..
//   def run(self, operation):
//     # do something with operation..
//     pass
// ```
class PyPassBase {
public:
  PyPassBase(std::string name, std::string argument, std::string description,
             std::string opName)
      : name(std::move(name)), argument(std::move(argument)),
        description(std::move(description)), opName(std::move(opName)) {
    callbacks.construct = [](void *obj) {};
    callbacks.destruct = [](void *obj) {
      nb::handle(static_cast<PyObject *>(obj)).dec_ref();
    };
    callbacks.run = [](MlirOperation op, MlirExternalPass, void *obj) {
      auto handle = nb::handle(static_cast<PyObject *>(obj));
      nb::cast<PyPassBase *>(handle)->run(op);
    };
    callbacks.clone = [](void *obj) -> void * {
      nb::object copy = nb::module_::import_("copy");
      nb::object deepcopy = copy.attr("deepcopy");
      return deepcopy(obj).release().ptr();
    };
    callbacks.initialize = nullptr;
  }

  // this method should be overridden by subclasses in Python.
  virtual void run(MlirOperation op) = 0;

  virtual ~PyPassBase() = default;

  // Make an MlirPass instance on-the-fly that wraps this object.
  // Note that passmanager will take the ownership of the returned
  // object and release it when appropriate.
  MlirPass make() {
    auto *obj = nb::find(this).release().ptr();
    return mlirCreateExternalPass(
        mlirTypeIDCreate(this), mlirStringRefCreate(name.data(), name.length()),
        mlirStringRefCreate(argument.data(), argument.length()),
        mlirStringRefCreate(description.data(), description.length()),
        mlirStringRefCreate(opName.data(), opName.size()), 0, nullptr,
        callbacks, obj);
  }

  const std::string &getName() const { return name; }
  const std::string &getArgument() const { return argument; }
  const std::string &getDescription() const { return description; }
  const std::string &getOpName() const { return opName; }

private:
  MlirExternalPassCallbacks callbacks;

  std::string name;
  std::string argument;
  std::string description;
  std::string opName;
};

// A trampoline class upon PyPassBase.
// Refer to
// https://nanobind.readthedocs.io/en/latest/classes.html#overriding-virtual-functions-in-python
class PyPass : PyPassBase {
public:
  NB_TRAMPOLINE(PyPassBase, 1);

  void run(MlirOperation op) override { NB_OVERRIDE_PURE(run, op); }
};

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

void mlir::python::populatePassSubmodule(nanobind::module_ &m) {
  //----------------------------------------------------------------------------
  // Mapping of the Python-defined Pass interface
  //----------------------------------------------------------------------------
  nb::class_<PyPassBase, PyPass>(m, "Pass")
      .def(nb::init<std::string, std::string, std::string, std::string>(),
           "name"_a, nb::kw_only(), "argument"_a = "", "description"_a = "",
           "op_name"_a = "", "Create a new Pass.")
      .def("run", &PyPassBase::run, "operation"_a,
           "Run the pass on the provided operation.")
      .def_prop_ro("name",
                   [](const PyPassBase &self) { return self.getName(); })
      .def_prop_ro("argument",
                   [](const PyPassBase &self) { return self.getArgument(); })
      .def_prop_ro("description",
                   [](const PyPassBase &self) { return self.getDescription(); })
      .def_prop_ro("op_name",
                   [](const PyPassBase &self) { return self.getOpName(); });
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
          "large_elements_limit"_a.none() = nb::none(),
          "large_resource_limit"_a.none() = nb::none(),
          "enable_debug_info"_a = false, "print_generic_op_form"_a = false,
          "tree_printing_dir_path"_a.none() = nb::none(),
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
      .def(
          "add",
          [](PyPassManager &passManager, PyPassBase &pass) {
            mlirPassManagerAddOwnedPass(passManager.get(), pass.make());
          },
          "pass"_a, "Add a python-defined pass to the pass manager.")
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
}
