//===- Rewrite.cpp - Rewrite ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Rewrite.h"

#include "mlir-c/IR.h"
#include "mlir-c/Rewrite.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/IRCore.h"
// clang-format off
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir-c/Bindings/Python/Interop.h" // This is expected after nanobind.
// clang-format on
#include "mlir/Config/mlir-config.h"
#include "nanobind/nanobind.h"

namespace nb = nanobind;
using namespace mlir;
using namespace nb::literals;
using namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN;

namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {

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

  void replaceOp(MlirOperation op, MlirOperation newOp) {
    mlirRewriterBaseReplaceOpWithOperation(base, op, newOp);
  }

  void replaceOp(MlirOperation op, const std::vector<MlirValue> &values) {
    mlirRewriterBaseReplaceOpWithValues(base, op, values.size(), values.data());
  }

  void eraseOp(const PyOperation &op) { mlirRewriterBaseEraseOp(base, op); }

private:
  MlirRewriterBase base;
  PyMlirContextRef ctx;
};

struct PyMlirPDLResultList : MlirPDLResultList {};

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
              f(PyPatternRewriter(rewriter), PyMlirPDLResultList{results.ptr},
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
              f(PyPatternRewriter(rewriter), PyMlirPDLResultList{results.ptr},
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

class PyRewritePatternSet {
public:
  PyRewritePatternSet(MlirContext ctx)
      : set(mlirRewritePatternSetCreate(ctx)), ctx(ctx) {}
  ~PyRewritePatternSet() {
    if (set.ptr)
      mlirRewritePatternSetDestroy(set);
  }

  void add(MlirStringRef rootName, unsigned benefit,
           const nb::callable &matchAndRewrite) {
    MlirRewritePatternCallbacks callbacks;
    callbacks.construct = [](void *userData) {
      nb::handle(static_cast<PyObject *>(userData)).inc_ref();
    };
    callbacks.destruct = [](void *userData) {
      nb::handle(static_cast<PyObject *>(userData)).dec_ref();
    };
    callbacks.matchAndRewrite = [](MlirRewritePattern, MlirOperation op,
                                   MlirPatternRewriter rewriter,
                                   void *userData) -> MlirLogicalResult {
      nb::handle f(static_cast<PyObject *>(userData));

      PyMlirContextRef ctx =
          PyMlirContext::forContext(mlirOperationGetContext(op));
      nb::object opView = PyOperation::forOperation(ctx, op)->createOpView();

      nb::object res = f(opView, PyPatternRewriter(rewriter));
      return logicalResultFromObject(res);
    };
    MlirRewritePattern pattern = mlirOpRewritePatternCreate(
        rootName, benefit, ctx, callbacks, matchAndRewrite.ptr(),
        /* nGeneratedNames */ 0,
        /* generatedNames */ nullptr);
    mlirRewritePatternSetAdd(set, pattern);
  }

  PyFrozenRewritePatternSet freeze() {
    MlirRewritePatternSet s = set;
    set.ptr = nullptr;
    return mlirFreezeRewritePattern(s);
  }

private:
  MlirRewritePatternSet set;
  MlirContext ctx;
};

enum class PyGreedyRewriteStrictness : std::underlying_type_t<
    MlirGreedyRewriteStrictness> {
  ANY_OP = MLIR_GREEDY_REWRITE_STRICTNESS_ANY_OP,
  EXISTING_AND_NEW_OPS = MLIR_GREEDY_REWRITE_STRICTNESS_EXISTING_AND_NEW_OPS,
  EXISTING_OPS = MLIR_GREEDY_REWRITE_STRICTNESS_EXISTING_OPS,
};

enum class PyGreedySimplifyRegionLevel : std::underlying_type_t<
    MlirGreedySimplifyRegionLevel> {
  DISABLED = MLIR_GREEDY_SIMPLIFY_REGION_LEVEL_DISABLED,
  NORMAL = MLIR_GREEDY_SIMPLIFY_REGION_LEVEL_NORMAL,
  AGGRESSIVE = MLIR_GREEDY_SIMPLIFY_REGION_LEVEL_AGGRESSIVE
};

/// Owning Wrapper around a GreedyRewriteDriverConfig.
class PyGreedyRewriteConfig {
public:
  PyGreedyRewriteConfig()
      : config(mlirGreedyRewriteDriverConfigCreate().ptr,
               PyGreedyRewriteConfig::customDeleter) {}
  PyGreedyRewriteConfig(PyGreedyRewriteConfig &&other) noexcept
      : config(std::move(other.config)) {}
  PyGreedyRewriteConfig(const PyGreedyRewriteConfig &other) noexcept
      : config(other.config) {}

  MlirGreedyRewriteDriverConfig get() {
    return MlirGreedyRewriteDriverConfig{config.get()};
  }

  void setMaxIterations(int64_t maxIterations) {
    mlirGreedyRewriteDriverConfigSetMaxIterations(get(), maxIterations);
  }

  void setMaxNumRewrites(int64_t maxNumRewrites) {
    mlirGreedyRewriteDriverConfigSetMaxNumRewrites(get(), maxNumRewrites);
  }

  void setUseTopDownTraversal(bool useTopDownTraversal) {
    mlirGreedyRewriteDriverConfigSetUseTopDownTraversal(get(),
                                                        useTopDownTraversal);
  }

  void enableFolding(bool enable) {
    mlirGreedyRewriteDriverConfigEnableFolding(get(), enable);
  }

  void setStrictness(PyGreedyRewriteStrictness strictness) {
    mlirGreedyRewriteDriverConfigSetStrictness(
        get(), static_cast<MlirGreedyRewriteStrictness>(strictness));
  }

  void setRegionSimplificationLevel(PyGreedySimplifyRegionLevel level) {
    mlirGreedyRewriteDriverConfigSetRegionSimplificationLevel(
        get(), static_cast<MlirGreedySimplifyRegionLevel>(level));
  }

  void enableConstantCSE(bool enable) {
    mlirGreedyRewriteDriverConfigEnableConstantCSE(get(), enable);
  }

  int64_t getMaxIterations() {
    return mlirGreedyRewriteDriverConfigGetMaxIterations(get());
  }

  int64_t getMaxNumRewrites() {
    return mlirGreedyRewriteDriverConfigGetMaxNumRewrites(get());
  }

  bool getUseTopDownTraversal() {
    return mlirGreedyRewriteDriverConfigGetUseTopDownTraversal(get());
  }

  bool isFoldingEnabled() {
    return mlirGreedyRewriteDriverConfigIsFoldingEnabled(get());
  }

  PyGreedyRewriteStrictness getStrictness() {
    return static_cast<PyGreedyRewriteStrictness>(
        mlirGreedyRewriteDriverConfigGetStrictness(get()));
  }

  PyGreedySimplifyRegionLevel getRegionSimplificationLevel() {
    return static_cast<PyGreedySimplifyRegionLevel>(
        mlirGreedyRewriteDriverConfigGetRegionSimplificationLevel(get()));
  }

  bool isConstantCSEEnabled() {
    return mlirGreedyRewriteDriverConfigIsConstantCSEEnabled(get());
  }

private:
  std::shared_ptr<void> config;
  static void customDeleter(void *c) {
    mlirGreedyRewriteDriverConfigDestroy(MlirGreedyRewriteDriverConfig{c});
  }
};

/// Create the `mlir.rewrite` here.
void populateRewriteSubmodule(nb::module_ &m) {
  // Enum definitions
  nb::enum_<PyGreedyRewriteStrictness>(m, "GreedyRewriteStrictness")
      .value("ANY_OP", PyGreedyRewriteStrictness::ANY_OP)
      .value("EXISTING_AND_NEW_OPS",
             PyGreedyRewriteStrictness::EXISTING_AND_NEW_OPS)
      .value("EXISTING_OPS", PyGreedyRewriteStrictness::EXISTING_OPS);

  nb::enum_<PyGreedySimplifyRegionLevel>(m, "GreedySimplifyRegionLevel")
      .value("DISABLED", PyGreedySimplifyRegionLevel::DISABLED)
      .value("NORMAL", PyGreedySimplifyRegionLevel::NORMAL)
      .value("AGGRESSIVE", PyGreedySimplifyRegionLevel::AGGRESSIVE);
  //----------------------------------------------------------------------------
  // Mapping of the PatternRewriter
  //----------------------------------------------------------------------------
  nb::class_<PyPatternRewriter>(m, "PatternRewriter")
      .def_prop_ro("ip", &PyPatternRewriter::getInsertionPoint,
                   "The current insertion point of the PatternRewriter.")
      .def(
          "replace_op",
          [](PyPatternRewriter &self, PyOperationBase &op,
             PyOperationBase &newOp) {
            self.replaceOp(op.getOperation(), newOp.getOperation());
          },
          "Replace an operation with a new operation.", nb::arg("op"),
          nb::arg("new_op"))
      .def(
          "replace_op",
          [](PyPatternRewriter &self, PyOperationBase &op,
             const std::vector<PyValue> &values) {
            std::vector<MlirValue> values_(values.size());
            std::copy(values.begin(), values.end(), values_.begin());
            self.replaceOp(op.getOperation(), values_);
          },
          "Replace an operation with a list of values.", nb::arg("op"),
          nb::arg("values"))
      .def("erase_op", &PyPatternRewriter::eraseOp, "Erase an operation.",
           nb::arg("op"));

  //----------------------------------------------------------------------------
  // Mapping of the RewritePatternSet
  //----------------------------------------------------------------------------
  nb::class_<PyRewritePatternSet>(m, "RewritePatternSet")
      .def(
          "__init__",
          [](PyRewritePatternSet &self, DefaultingPyMlirContext context) {
            new (&self) PyRewritePatternSet(context.get()->get());
          },
          "context"_a = nb::none())
      .def(
          "add",
          [](PyRewritePatternSet &self, nb::handle root, const nb::callable &fn,
             unsigned benefit) {
            std::string opName;
            if (root.is_type()) {
              opName = nb::cast<std::string>(root.attr("OPERATION_NAME"));
            } else if (nb::isinstance<nb::str>(root)) {
              opName = nb::cast<std::string>(root);
            } else {
              throw nb::type_error(
                  "the root argument must be a type or a string");
            }
            self.add(mlirStringRefCreate(opName.data(), opName.size()), benefit,
                     fn);
          },
          "root"_a, "fn"_a, "benefit"_a = 1,
          // clang-format off
          nb::sig("def add(self, root: type | str, fn: typing.Callable[[" MAKE_MLIR_PYTHON_QUALNAME("ir.Operation") ", PatternRewriter], typing.Any], benefit: int = 1) -> None"),
          // clang-format on
          R"(
            Add a new rewrite pattern on the specified root operation, using the provided callable
            for matching and rewriting, and assign it the given benefit.

            Args:
              root: The root operation to which this pattern applies.
                    This may be either an OpView subclass (e.g., ``arith.AddIOp``) or
                    an operation name string (e.g., ``"arith.addi"``).
              fn: The callable to use for matching and rewriting,
                  which takes an operation and a pattern rewriter as arguments.
                  The match is considered successful iff the callable returns
                  a value where ``bool(value)`` is ``False`` (e.g. ``None``).
                  If possible, the operation is cast to its corresponding OpView subclass
                  before being passed to the callable.
              benefit: The benefit of the pattern, defaulting to 1.)")
      .def("freeze", &PyRewritePatternSet::freeze,
           "Freeze the pattern set into a frozen one.");

  //----------------------------------------------------------------------------
  // Mapping of the PDLResultList and PDLModule
  //----------------------------------------------------------------------------
#if MLIR_ENABLE_PDL_IN_PATTERNMATCH
  nb::class_<PyMlirPDLResultList>(m, "PDLResultList")
      .def("append",
           [](PyMlirPDLResultList results, const PyValue &value) {
             mlirPDLResultListPushBackValue(results, value);
           })
      .def("append",
           [](PyMlirPDLResultList results, const PyOperation &op) {
             mlirPDLResultListPushBackOperation(results, op);
           })
      .def("append",
           [](PyMlirPDLResultList results, const PyType &type) {
             mlirPDLResultListPushBackType(results, type);
           })
      .def("append", [](PyMlirPDLResultList results, const PyAttribute &attr) {
        mlirPDLResultListPushBackAttribute(results, attr);
      });
  nb::class_<PyPDLPatternModule>(m, "PDLModule")
      .def(
          "__init__",
          [](PyPDLPatternModule &self, PyModule &module) {
            new (&self) PyPDLPatternModule(
                mlirPDLPatternModuleFromModule(module.get()));
          },
          "module"_a, "Create a PDL module from the given module.")
      .def(
          "__init__",
          [](PyPDLPatternModule &self, PyModule &module) {
            new (&self) PyPDLPatternModule(
                mlirPDLPatternModuleFromModule(module.get()));
          },
          "module"_a, "Create a PDL module from the given module.")
      .def(
          "freeze",
          [](PyPDLPatternModule &self) {
            return PyFrozenRewritePatternSet(mlirFreezeRewritePattern(
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

  nb::class_<PyGreedyRewriteConfig>(m, "GreedyRewriteConfig")
      .def(nb::init<>(), "Create a greedy rewrite driver config with defaults")
      .def_prop_rw("max_iterations", &PyGreedyRewriteConfig::getMaxIterations,
                   &PyGreedyRewriteConfig::setMaxIterations,
                   "Maximum number of iterations")
      .def_prop_rw("max_num_rewrites",
                   &PyGreedyRewriteConfig::getMaxNumRewrites,
                   &PyGreedyRewriteConfig::setMaxNumRewrites,
                   "Maximum number of rewrites per iteration")
      .def_prop_rw("use_top_down_traversal",
                   &PyGreedyRewriteConfig::getUseTopDownTraversal,
                   &PyGreedyRewriteConfig::setUseTopDownTraversal,
                   "Whether to use top-down traversal")
      .def_prop_rw("enable_folding", &PyGreedyRewriteConfig::isFoldingEnabled,
                   &PyGreedyRewriteConfig::enableFolding,
                   "Enable or disable folding")
      .def_prop_rw("strictness", &PyGreedyRewriteConfig::getStrictness,
                   &PyGreedyRewriteConfig::setStrictness,
                   "Rewrite strictness level")
      .def_prop_rw("region_simplification_level",
                   &PyGreedyRewriteConfig::getRegionSimplificationLevel,
                   &PyGreedyRewriteConfig::setRegionSimplificationLevel,
                   "Region simplification level")
      .def_prop_rw("enable_constant_cse",
                   &PyGreedyRewriteConfig::isConstantCSEEnabled,
                   &PyGreedyRewriteConfig::enableConstantCSE,
                   "Enable or disable constant CSE");

  nb::class_<PyFrozenRewritePatternSet>(m, "FrozenRewritePatternSet")
      .def_prop_ro(MLIR_PYTHON_CAPI_PTR_ATTR,
                   &PyFrozenRewritePatternSet::getCapsule)
      .def(MLIR_PYTHON_CAPI_FACTORY_ATTR,
           &PyFrozenRewritePatternSet::createFromCapsule);
  m.def(
       "apply_patterns_and_fold_greedily",
       [](PyModule &module, PyFrozenRewritePatternSet &set,
          std::optional<PyGreedyRewriteConfig> config) {
         MlirLogicalResult status = mlirApplyPatternsAndFoldGreedily(
             module.get(), set.get(),
             config.has_value() ? config->get()
                                : mlirGreedyRewriteDriverConfigCreate());
         if (mlirLogicalResultIsFailure(status))
           throw std::runtime_error("pattern application failed to converge");
       },
       "module"_a, "set"_a, "config"_a = nb::none(),
       "Applys the given patterns to the given module greedily while folding "
       "results.")
      .def(
          "apply_patterns_and_fold_greedily",
          [](PyOperationBase &op, PyFrozenRewritePatternSet &set,
             std::optional<PyGreedyRewriteConfig> config) {
            MlirLogicalResult status = mlirApplyPatternsAndFoldGreedilyWithOp(
                op.getOperation(), set.get(),
                config.has_value() ? config->get()
                                   : mlirGreedyRewriteDriverConfigCreate());
            if (mlirLogicalResultIsFailure(status))
              throw std::runtime_error(
                  "pattern application failed to converge");
          },
          "op"_a, "set"_a, "config"_a = nb::none(),
          "Applys the given patterns to the given op greedily while folding "
          "results.")
      .def(
          "walk_and_apply_patterns",
          [](PyOperationBase &op, PyFrozenRewritePatternSet &set) {
            mlirWalkAndApplyPatterns(op.getOperation(), set.get());
          },
          "op"_a, "set"_a,
          "Applies the given patterns to the given op by a fast walk-based "
          "driver.");
}
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir
