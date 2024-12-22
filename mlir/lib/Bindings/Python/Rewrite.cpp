//===- Rewrite.cpp - Rewrite ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Rewrite.h"

#include "IRModule.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/Rewrite.h"
#include "mlir/Config/mlir-config.h"

namespace py = pybind11;
using namespace mlir;
using namespace py::literals;
using namespace mlir::python;

namespace {

#if MLIR_ENABLE_PDL_IN_PATTERNMATCH
/// Owning Wrapper around a PDLPatternModule.
class PyPDLPatternModule {
public:
  PyPDLPatternModule(MlirPDLPatternModule module) : module(module) {}
  PyPDLPatternModule(PyPDLPatternModule &&other) noexcept
      : module(other.module) {
    other.module.ptr = nullptr;
  }
  ~PyPDLPatternModule() {
    if (module.ptr != nullptr)
      mlirPDLPatternModuleDestroy(module);
  }
  MlirPDLPatternModule get() { return module; }

private:
  MlirPDLPatternModule module;
};
#endif // MLIR_ENABLE_PDL_IN_PATTERNMATCH

/// Owning Wrapper around a FrozenRewritePatternSet.
class PyFrozenRewritePatternSet {
public:
  PyFrozenRewritePatternSet(MlirFrozenRewritePatternSet set) : set(set) {}
  PyFrozenRewritePatternSet(PyFrozenRewritePatternSet &&other) noexcept
      : set(other.set) {
    other.set.ptr = nullptr;
  }
  ~PyFrozenRewritePatternSet() {
    if (set.ptr != nullptr)
      mlirFrozenRewritePatternSetDestroy(set);
  }
  MlirFrozenRewritePatternSet get() { return set; }

  pybind11::object getCapsule() {
    return py::reinterpret_steal<py::object>(
        mlirPythonFrozenRewritePatternSetToCapsule(get()));
  }

  static pybind11::object createFromCapsule(pybind11::object capsule) {
    MlirFrozenRewritePatternSet rawPm =
        mlirPythonCapsuleToFrozenRewritePatternSet(capsule.ptr());
    if (rawPm.ptr == nullptr)
      throw py::error_already_set();
    return py::cast(PyFrozenRewritePatternSet(rawPm),
                    py::return_value_policy::move);
  }

private:
  MlirFrozenRewritePatternSet set;
};

} // namespace

/// Create the `mlir.rewrite` here.
void mlir::python::populateRewriteSubmodule(py::module &m) {
  //----------------------------------------------------------------------------
  // Mapping of the top-level PassManager
  //----------------------------------------------------------------------------
#if MLIR_ENABLE_PDL_IN_PATTERNMATCH
  py::class_<PyPDLPatternModule>(m, "PDLModule", py::module_local())
      .def(py::init<>([](MlirModule module) {
             return mlirPDLPatternModuleFromModule(module);
           }),
           "module"_a, "Create a PDL module from the given module.")
      .def("freeze", [](PyPDLPatternModule &self) {
        return new PyFrozenRewritePatternSet(mlirFreezeRewritePattern(
            mlirRewritePatternSetFromPDLPatternModule(self.get())));
      });
#endif // MLIR_ENABLE_PDL_IN_PATTERNMATCg
  py::class_<PyFrozenRewritePatternSet>(m, "FrozenRewritePatternSet",
                                        py::module_local())
      .def_property_readonly(MLIR_PYTHON_CAPI_PTR_ATTR,
                             &PyFrozenRewritePatternSet::getCapsule)
      .def(MLIR_PYTHON_CAPI_FACTORY_ATTR,
           &PyFrozenRewritePatternSet::createFromCapsule);
  m.def(
      "apply_patterns_and_fold_greedily",
      [](MlirModule module, MlirFrozenRewritePatternSet set) {
        auto status = mlirApplyPatternsAndFoldGreedily(module, set, {});
        if (mlirLogicalResultIsFailure(status))
          // FIXME: Not sure this is the right error to throw here.
          throw py::value_error("pattern application failed to converge");
      },
      "module"_a, "set"_a,
      "Applys the given patterns to the given module greedily while folding "
      "results.");
}
