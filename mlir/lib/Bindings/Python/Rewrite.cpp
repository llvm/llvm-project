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

class PyPatternRewriter {
public:
  PyPatternRewriter(MlirPatternRewriter rewriter)
      : base(mlirPatternRewriterAsBase(rewriter)),
        ctx(PyMlirContext::forContext(mlirRewriterBaseGetContext(base))) {}

  PyInsertionPoint getInsertionPoint() const {
    MlirBlock block = mlirRewriterBaseGetInsertionBlock(base);
    MlirOperation op = mlirRewriterBaseGetOperationAfterInsertion(base);

    if (mlirOperationIsNull(op)) {
      MlirOperation owner = mlirBlockGetParentOperation(block);
      auto parent = PyOperation::forOperation(ctx, owner);
      return PyInsertionPoint(PyBlock(parent, block));
    }

    return PyInsertionPoint(PyOperation::forOperation(ctx, op));
  }

private:
  MlirRewriterBase base;
  PyMlirContextRef ctx;
};

#if MLIR_ENABLE_PDL_IN_PATTERNMATCH
static nb::object objectFromPDLValue(MlirPDLValue value) {
  if (MlirValue v = mlirPDLValueAsValue(value); !mlirValueIsNull(v))
    return nb::cast(v);
  if (MlirOperation v = mlirPDLValueAsOperation(value); !mlirOperationIsNull(v))
    return nb::cast(v);
  if (MlirAttribute v = mlirPDLValueAsAttribute(value); !mlirAttributeIsNull(v))
    return nb::cast(v);
  if (MlirType v = mlirPDLValueAsType(value); !mlirTypeIsNull(v))
    return nb::cast(v);

  throw std::runtime_error("unsupported PDL value type");
}

static std::vector<nb::object> objectsFromPDLValues(size_t nValues,
                                                    MlirPDLValue *values) {
  std::vector<nb::object> args;
  args.reserve(nValues);
  for (size_t i = 0; i < nValues; ++i)
    args.push_back(objectFromPDLValue(values[i]));
  return args;
}

// Convert the Python object to a boolean.
// If it evaluates to False, treat it as success;
// otherwise, treat it as failure.
// Note that None is considered success.
static MlirLogicalResult logicalResultFromObject(const nb::object &obj) {
  if (obj.is_none())
    return mlirLogicalResultSuccess();

  return nb::cast<bool>(obj) ? mlirLogicalResultFailure()
                             : mlirLogicalResultSuccess();
}

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

  void registerRewriteFunction(const std::string &name,
                               const nb::callable &fn) {
    mlirPDLPatternModuleRegisterRewriteFunction(
        get(), mlirStringRefCreate(name.data(), name.size()),
        [](MlirPatternRewriter rewriter, MlirPDLResultList results,
           size_t nValues, MlirPDLValue *values,
           void *userData) -> MlirLogicalResult {
          nb::handle f = nb::handle(static_cast<PyObject *>(userData));
          return logicalResultFromObject(
              f(PyPatternRewriter(rewriter), results,
                objectsFromPDLValues(nValues, values)));
        },
        fn.ptr());
  }

  void registerConstraintFunction(const std::string &name,
                                  const nb::callable &fn) {
    mlirPDLPatternModuleRegisterConstraintFunction(
        get(), mlirStringRefCreate(name.data(), name.size()),
        [](MlirPatternRewriter rewriter, MlirPDLResultList results,
           size_t nValues, MlirPDLValue *values,
           void *userData) -> MlirLogicalResult {
          nb::handle f = nb::handle(static_cast<PyObject *>(userData));
          return logicalResultFromObject(
              f(PyPatternRewriter(rewriter), results,
                objectsFromPDLValues(nValues, values)));
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

  static nb::object createFromCapsule(const nb::object &capsule) {
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
  nb::class_<PyPatternRewriter>(m, "PatternRewriter")
      .def_prop_ro("ip", &PyPatternRewriter::getInsertionPoint,
                   "The current insertion point of the PatternRewriter.");
  //----------------------------------------------------------------------------
  // Mapping of the PDLResultList and PDLModule
  //----------------------------------------------------------------------------
#if MLIR_ENABLE_PDL_IN_PATTERNMATCH
  nb::class_<MlirPDLResultList>(m, "PDLResultList")
      .def(
          "append",
          [](MlirPDLResultList results, const PyValue &value) {
            mlirPDLResultListPushBackValue(results, value);
          },
          // clang-format off
          nb::sig("def append(self, value: " MAKE_MLIR_PYTHON_QUALNAME("ir.Value") ")")
          // clang-format on
          )
      .def(
          "append",
          [](MlirPDLResultList results, const PyOperation &op) {
            mlirPDLResultListPushBackOperation(results, op);
          },
          // clang-format off
          nb::sig("def append(self, op: " MAKE_MLIR_PYTHON_QUALNAME("ir.Operation") ")")
          // clang-format on
          )
      .def(
          "append",
          [](MlirPDLResultList results, const PyType &type) {
            mlirPDLResultListPushBackType(results, type);
          },
          // clang-format off
          nb::sig("def append(self, type: " MAKE_MLIR_PYTHON_QUALNAME("ir.Type") ")")
          // clang-format on
          )
      .def(
          "append",
          [](MlirPDLResultList results, const PyAttribute &attr) {
            mlirPDLResultListPushBackAttribute(results, attr);
          },
          // clang-format off
          nb::sig("def append(self, attr: " MAKE_MLIR_PYTHON_QUALNAME("ir.Attribute") ")")
          // clang-format on
      );
  nb::class_<PyPDLPatternModule>(m, "PDLModule")
      .def(
          "__init__",
          [](PyPDLPatternModule &self, MlirModule module) {
            new (&self)
                PyPDLPatternModule(mlirPDLPatternModuleFromModule(module));
          },
          // clang-format off
          nb::sig("def __init__(self, module: " MAKE_MLIR_PYTHON_QUALNAME("ir.Module") ") -> None"),
          // clang-format on
          "module"_a, "Create a PDL module from the given module.")
      .def(
          "__init__",
          [](PyPDLPatternModule &self, PyModule &module) {
            new (&self) PyPDLPatternModule(
                mlirPDLPatternModuleFromModule(module.get()));
          },
          // clang-format off
          nb::sig("def __init__(self, module: " MAKE_MLIR_PYTHON_QUALNAME("ir.Module") ") -> None"),
          // clang-format on
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
          nb::keep_alive<1, 3>())
      .def(
          "register_constraint_function",
          [](PyPDLPatternModule &self, const std::string &name,
             const nb::callable &fn) {
            self.registerConstraintFunction(name, fn);
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
       [](PyModule &module, PyFrozenRewritePatternSet &set) {
         auto status =
             mlirApplyPatternsAndFoldGreedily(module.get(), set.get(), {});
         if (mlirLogicalResultIsFailure(status))
           throw std::runtime_error("pattern application failed to converge");
       },
       "module"_a, "set"_a,
       // clang-format off
       nb::sig("def apply_patterns_and_fold_greedily(module: " MAKE_MLIR_PYTHON_QUALNAME("ir.Module") ", set: FrozenRewritePatternSet) -> None"),
       // clang-format on
       "Applys the given patterns to the given module greedily while folding "
       "results.")
      .def(
          "apply_patterns_and_fold_greedily",
          [](PyModule &module, MlirFrozenRewritePatternSet set) {
            auto status =
                mlirApplyPatternsAndFoldGreedily(module.get(), set, {});
            if (mlirLogicalResultIsFailure(status))
              throw std::runtime_error(
                  "pattern application failed to converge");
          },
          "module"_a, "set"_a,
          // clang-format off
          nb::sig("def apply_patterns_and_fold_greedily(module: " MAKE_MLIR_PYTHON_QUALNAME("ir.Module") ", set: FrozenRewritePatternSet) -> None"),
          // clang-format on
          "Applys the given patterns to the given module greedily while "
          "folding "
          "results.")
      .def(
          "apply_patterns_and_fold_greedily",
          [](PyOperationBase &op, PyFrozenRewritePatternSet &set) {
            auto status = mlirApplyPatternsAndFoldGreedilyWithOp(
                op.getOperation(), set.get(), {});
            if (mlirLogicalResultIsFailure(status))
              throw std::runtime_error(
                  "pattern application failed to converge");
          },
          "op"_a, "set"_a,
          // clang-format off
          nb::sig("def apply_patterns_and_fold_greedily(op: " MAKE_MLIR_PYTHON_QUALNAME("ir._OperationBase") ", set: FrozenRewritePatternSet) -> None"),
          // clang-format on
          "Applys the given patterns to the given op greedily while folding "
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
          // clang-format off
          nb::sig("def apply_patterns_and_fold_greedily(op: " MAKE_MLIR_PYTHON_QUALNAME("ir._OperationBase") ", set: FrozenRewritePatternSet) -> None"),
          // clang-format on
          "Applys the given patterns to the given op greedily while folding "
          "results.");
}
