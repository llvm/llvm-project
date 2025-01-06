//===- Rewrite.cpp - Rewrite ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Rewrite.h"

#include "IRModule.h"
#include "mlir-c/Rewrite.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir-c/Bindings/Python/Interop.h" // This is expected after nanobind.
#include "mlir/Config/mlir-config.h"

namespace nb = nanobind;
using namespace mlir;
using namespace nb::literals;
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

  nb::object getCapsule() {
    return nb::steal<nb::object>(
        mlirPythonFrozenRewritePatternSetToCapsule(get()));
  }

  static nb::object createFromCapsule(nb::object capsule) {
    MlirFrozenRewritePatternSet rawPm =
        mlirPythonCapsuleToFrozenRewritePatternSet(capsule.ptr());
    if (rawPm.ptr == nullptr)
      throw nb::python_error();
    return nb::cast(PyFrozenRewritePatternSet(rawPm), nb::rv_policy::move);
  }

private:
  MlirFrozenRewritePatternSet set;
};

} // namespace

/// Create the `mlir.rewrite` here.
void mlir::python::populateRewriteSubmodule(nb::module_ &m) {
  //----------------------------------------------------------------------------
  // Mapping of the top-level PassManager
  //----------------------------------------------------------------------------
#if MLIR_ENABLE_PDL_IN_PATTERNMATCH
  nb::class_<PyPDLPatternModule>(m, "PDLModule")
      .def(
          "__init__",
          [](PyPDLPatternModule &self, MlirModule module) {
            new (&self)
                PyPDLPatternModule(mlirPDLPatternModuleFromModule(module));
          },
          "module"_a, "Create a PDL module from the given module.")
      .def("freeze", [](PyPDLPatternModule &self) {
        return new PyFrozenRewritePatternSet(mlirFreezeRewritePattern(
            mlirRewritePatternSetFromPDLPatternModule(self.get())));
      });
#endif // MLIR_ENABLE_PDL_IN_PATTERNMATCH
  nb::class_<PyFrozenRewritePatternSet>(m, "FrozenRewritePatternSet")
      .def_prop_ro(MLIR_PYTHON_CAPI_PTR_ATTR,
                   &PyFrozenRewritePatternSet::getCapsule)
      .def(MLIR_PYTHON_CAPI_FACTORY_ATTR,
           &PyFrozenRewritePatternSet::createFromCapsule);
  m.def(
      "apply_patterns_and_fold_greedily",
      [](MlirModule module, MlirFrozenRewritePatternSet set) {
        auto status = mlirApplyPatternsAndFoldGreedily(module, set, {});
        if (mlirLogicalResultIsFailure(status))
          // FIXME: Not sure this is the right error to throw here.
          throw nb::value_error("pattern application failed to converge");
      },
      "module"_a, "set"_a,
      "Applys the given patterns to the given module greedily while folding "
      "results.");
}
