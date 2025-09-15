//===- ExecutionEngineModule.cpp - Python module for execution engine -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/ExecutionEngine.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;
using namespace mlir;
using namespace mlir::python;

namespace {

/// Owning Wrapper around an ExecutionEngine.
class PyExecutionEngine {
public:
  PyExecutionEngine(MlirExecutionEngine executionEngine)
      : executionEngine(executionEngine) {}
  PyExecutionEngine(PyExecutionEngine &&other) noexcept
      : executionEngine(other.executionEngine) {
    other.executionEngine.ptr = nullptr;
  }
  ~PyExecutionEngine() {
    if (!mlirExecutionEngineIsNull(executionEngine))
      mlirExecutionEngineDestroy(executionEngine);
  }
  MlirExecutionEngine get() { return executionEngine; }

  void release() {
    executionEngine.ptr = nullptr;
    referencedObjects.clear();
  }
  nb::object getCapsule() {
    return nb::steal<nb::object>(mlirPythonExecutionEngineToCapsule(get()));
  }

  // Add an object to the list of referenced objects whose lifetime must exceed
  // those of the ExecutionEngine.
  void addReferencedObject(const nb::object &obj) {
    referencedObjects.push_back(obj);
  }

  static nb::object createFromCapsule(const nb::object &capsule) {
    MlirExecutionEngine rawPm =
        mlirPythonCapsuleToExecutionEngine(capsule.ptr());
    if (mlirExecutionEngineIsNull(rawPm))
      throw nb::python_error();
    return nb::cast(PyExecutionEngine(rawPm), nb::rv_policy::move);
  }

private:
  MlirExecutionEngine executionEngine;
  // We support Python ctypes closures as callbacks. Keep a list of the objects
  // so that they don't get garbage collected. (The ExecutionEngine itself
  // just holds raw pointers with no lifetime semantics).
  std::vector<nb::object> referencedObjects;
};

} // namespace

/// Create the `mlir.execution_engine` module here.
NB_MODULE(_mlirExecutionEngine, m) {
  m.doc() = "MLIR Execution Engine";

  //----------------------------------------------------------------------------
  // Mapping of the top-level PassManager
  //----------------------------------------------------------------------------
  nb::class_<PyExecutionEngine>(m, "ExecutionEngine")
      .def(
          "__init__",
          [](PyExecutionEngine &self, MlirModule module, int optLevel,
             const std::vector<std::string> &sharedLibPaths,
             bool enableObjectDump) {
            llvm::SmallVector<MlirStringRef, 4> libPaths;
            for (const std::string &path : sharedLibPaths)
              libPaths.push_back({path.c_str(), path.length()});
            MlirExecutionEngine executionEngine =
                mlirExecutionEngineCreate(module, optLevel, libPaths.size(),
                                          libPaths.data(), enableObjectDump);
            if (mlirExecutionEngineIsNull(executionEngine))
              throw std::runtime_error(
                  "Failure while creating the ExecutionEngine.");
            new (&self) PyExecutionEngine(executionEngine);
          },
          nb::arg("module"), nb::arg("opt_level") = 2,
          nb::arg("shared_libs") = nb::list(),
          nb::arg("enable_object_dump") = true,
          "Create a new ExecutionEngine instance for the given Module. The "
          "module must contain only dialects that can be translated to LLVM. "
          "Perform transformations and code generation at the optimization "
          "level `opt_level` if specified, or otherwise at the default "
          "level of two (-O2). Load a list of libraries specified in "
          "`shared_libs`.")
      .def_prop_ro(MLIR_PYTHON_CAPI_PTR_ATTR, &PyExecutionEngine::getCapsule)
      .def("_testing_release", &PyExecutionEngine::release,
           "Releases (leaks) the backing ExecutionEngine (for testing purpose)")
      .def(MLIR_PYTHON_CAPI_FACTORY_ATTR, &PyExecutionEngine::createFromCapsule)
      .def(
          "raw_lookup",
          [](PyExecutionEngine &executionEngine, const std::string &func) {
            auto *res = mlirExecutionEngineLookupPacked(
                executionEngine.get(),
                mlirStringRefCreate(func.c_str(), func.size()));
            return reinterpret_cast<uintptr_t>(res);
          },
          nb::arg("func_name"),
          "Lookup function `func` in the ExecutionEngine.")
      .def(
          "raw_register_runtime",
          [](PyExecutionEngine &executionEngine, const std::string &name,
             const nb::object &callbackObj) {
            executionEngine.addReferencedObject(callbackObj);
            uintptr_t rawSym =
                nb::cast<uintptr_t>(nb::getattr(callbackObj, "value"));
            mlirExecutionEngineRegisterSymbol(
                executionEngine.get(),
                mlirStringRefCreate(name.c_str(), name.size()),
                reinterpret_cast<void *>(rawSym));
          },
          nb::arg("name"), nb::arg("callback"),
          "Register `callback` as the runtime symbol `name`.")
      .def(
          "initialize",
          [](PyExecutionEngine &executionEngine) {
            mlirExecutionEngineInitialize(executionEngine.get());
          },
          "Initialize the ExecutionEngine. Global constructors specified by "
          "`llvm.mlir.global_ctors` will be run. One common scenario is that "
          "kernel binary compiled from `gpu.module` gets loaded during "
          "initialization. Make sure all symbols are resolvable before "
          "initialization by calling `register_runtime` or including "
          "shared libraries.")
      .def(
          "dump_to_object_file",
          [](PyExecutionEngine &executionEngine, const std::string &fileName) {
            mlirExecutionEngineDumpToObjectFile(
                executionEngine.get(),
                mlirStringRefCreate(fileName.c_str(), fileName.size()));
          },
          nb::arg("file_name"), "Dump ExecutionEngine to an object file.");
}
