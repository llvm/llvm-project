//===- Rewrite.cpp - Rewrite ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Rewrite.h"

#include "IRModule.h"
#include "mlir-c/IR.h"
#include "mlir-c/Rewrite.h"
#include "mlir-c/Support.h"
// clang-format off
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir-c/Bindings/Python/Interop.h" // This is expected after nanobind.
// clang-format on
#include "mlir/Config/mlir-config.h"
#include "nanobind/nanobind.h"

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

  static nb::object fromPDLValue(MlirPDLValue value) {
    if (mlirPDLValueIsValue(value))
      return nb::cast(mlirPDLValueAsValue(value));
    if (mlirPDLValueIsOperation(value))
      return nb::cast(mlirPDLValueAsOperation(value));
    if (mlirPDLValueIsAttribute(value))
      return nb::cast(mlirPDLValueAsAttribute(value));
    if (mlirPDLValueIsType(value))
      return nb::cast(mlirPDLValueAsType(value));

    throw std::runtime_error("unsupported PDL value type");
  }

  void registerRewriteFunction(const std::string &name,
                               const nb::callable &fn) {
    mlirPDLPatternModuleRegisterRewriteFunction(
        get(), mlirStringRefCreate(name.data(), name.size()),
        [](MlirPatternRewriter rewriter, MlirPDLResultList results,
           size_t nValues, MlirPDLValue *values,
           void *userData) -> MlirLogicalResult {
          auto f = nb::handle(static_cast<PyObject *>(userData));
          std::vector<nb::object> args;
          args.reserve(nValues);
          for (size_t i = 0; i < nValues; ++i) {
            args.push_back(fromPDLValue(values[i]));
          }
          return nb::cast<bool>(f(rewriter, results, args))
                     ? mlirLogicalResultSuccess()
                     : mlirLogicalResultFailure();
        },
        fn.ptr());
  }

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
  nb::class_<MlirPatternRewriter>(m, "PatternRewriter");
  //----------------------------------------------------------------------------
  // Mapping of the PDLResultList and PDLModule
  //----------------------------------------------------------------------------
#if MLIR_ENABLE_PDL_IN_PATTERNMATCH
  nb::class_<MlirPDLResultList>(m, "PDLResultList")
      .def("push_back",
           [](MlirPDLResultList results, const PyValue &value) {
             mlirPDLResultListPushBackValue(results, value);
           })
      .def("push_back",
           [](MlirPDLResultList results, const PyOperation &op) {
             mlirPDLResultListPushBackOperation(results, op);
           })
      .def("push_back",
           [](MlirPDLResultList results, const PyType &type) {
             mlirPDLResultListPushBackType(results, type);
           })
      .def("push_back", [](MlirPDLResultList results, const PyAttribute &attr) {
        mlirPDLResultListPushBackAttribute(results, attr);
      });
  nb::class_<PyPDLPatternModule>(m, "PDLModule")
      .def(
          "__init__",
          [](PyPDLPatternModule &self, MlirModule module) {
            new (&self)
                PyPDLPatternModule(mlirPDLPatternModuleFromModule(module));
          },
          "module"_a, "Create a PDL module from the given module.")
      .def(
          "freeze",
          [](PyPDLPatternModule &self) {
            return new PyFrozenRewritePatternSet(mlirFreezeRewritePattern(
                mlirRewritePatternSetFromPDLPatternModule(self.get())));
          },
          nb::keep_alive<0, 1>())
      .def(
          "register_rewrite_function",
          [](PyPDLPatternModule &self, const std::string &name,
             const nb::callable &fn) {
            self.registerRewriteFunction(name, fn);
          },
          nb::keep_alive<1, 3>());
#endif // MLIR_ENABLE_PDL_IN_PATTERNMATCH
  nb::class_<PyFrozenRewritePatternSet>(m, "FrozenRewritePatternSet")
      .def_prop_ro(MLIR_PYTHON_CAPI_PTR_ATTR,
                   &PyFrozenRewritePatternSet::getCapsule)
      .def(MLIR_PYTHON_CAPI_FACTORY_ATTR,
           &PyFrozenRewritePatternSet::createFromCapsule);
  m.def(
       "apply_patterns_and_fold_greedily",
       [](PyModule &module, MlirFrozenRewritePatternSet set) {
         auto status = mlirApplyPatternsAndFoldGreedily(module.get(), set, {});
         if (mlirLogicalResultIsFailure(status))
           throw std::runtime_error("pattern application failed to converge");
       },
       "module"_a, "set"_a,
       "Applys the given patterns to the given module greedily while folding "
       "results.")
      .def(
          "apply_patterns_and_fold_greedily",
          [](PyOperationBase &op, MlirFrozenRewritePatternSet set) {
            auto status = mlirApplyPatternsAndFoldGreedilyWithOp(
                op.getOperation(), set, {});
            if (mlirLogicalResultIsFailure(status))
              throw std::runtime_error(
                  "pattern application failed to converge");
          },
          "op"_a, "set"_a,
          "Applys the given patterns to the given op greedily while folding "
          "results.");
}
