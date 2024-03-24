//===- MainModule.cpp - Main pybind module --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PybindUtils.h"

#include "Globals.h"
#include "IRModule.h"
#include "Pass.h"

#include "mlir-c/Dialect/AMDGPU.h"
#include "mlir-c/Dialect/Arith.h"
#include "mlir-c/Dialect/Async.h"
#include "mlir-c/Dialect/ControlFlow.h"
#include "mlir-c/Dialect/Func.h"
#include "mlir-c/Dialect/GPU.h"
#include "mlir-c/Dialect/LLVM.h"
#include "mlir-c/Dialect/Linalg.h"
#include "mlir-c/Dialect/MLProgram.h"
#include "mlir-c/Dialect/Math.h"
#include "mlir-c/Dialect/MemRef.h"
#include "mlir-c/Dialect/NVGPU.h"
#include "mlir-c/Dialect/NVVM.h"
#include "mlir-c/Dialect/OpenMP.h"
#include "mlir-c/Dialect/PDL.h"
#include "mlir-c/Dialect/Quant.h"
#include "mlir-c/Dialect/ROCDL.h"
#include "mlir-c/Dialect/SCF.h"
#include "mlir-c/Dialect/Shape.h"
#include "mlir-c/Dialect/SparseTensor.h"
#include "mlir-c/Dialect/Tensor.h"
#include "mlir-c/Dialect/Transform.h"
#include "mlir-c/Dialect/Vector.h"

#include "mlir-c/Dialect/RemainingDialects.h"

namespace py = pybind11;
using namespace mlir;
using namespace py::literals;
using namespace mlir::python;

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------

PYBIND11_MODULE(_mlir, m) {
  m.doc() = "MLIR Python Native Extension";

  py::class_<PyGlobals>(m, "_Globals", py::module_local())
      .def_property("dialect_search_modules",
                    &PyGlobals::getDialectSearchPrefixes,
                    &PyGlobals::setDialectSearchPrefixes)
      .def(
          "append_dialect_search_prefix",
          [](PyGlobals &self, std::string moduleName) {
            self.getDialectSearchPrefixes().push_back(std::move(moduleName));
          },
          "module_name"_a)
      .def(
          "_check_dialect_module_loaded",
          [](PyGlobals &self, const std::string &dialectNamespace) {
            return self.loadDialectModule(dialectNamespace);
          },
          "dialect_namespace"_a)
      .def("_register_dialect_impl", &PyGlobals::registerDialectImpl,
           "dialect_namespace"_a, "dialect_class"_a,
           "Testing hook for directly registering a dialect")
      .def("_register_operation_impl", &PyGlobals::registerOperationImpl,
           "operation_name"_a, "operation_class"_a, py::kw_only(),
           "replace"_a = false,
           "Testing hook for directly registering an operation");

  // Aside from making the globals accessible to python, having python manage
  // it is necessary to make sure it is destroyed (and releases its python
  // resources) properly.
  m.attr("globals") =
      py::cast(new PyGlobals, py::return_value_policy::take_ownership);

  // Registration decorators.
  m.def(
      "register_dialect",
      [](py::object pyClass) {
        std::string dialectNamespace =
            pyClass.attr("DIALECT_NAMESPACE").cast<std::string>();
        PyGlobals::get().registerDialectImpl(dialectNamespace, pyClass);
        return pyClass;
      },
      "dialect_class"_a,
      "Class decorator for registering a custom Dialect wrapper");
  m.def(
      "add_dialect_to_dialect_registry",
      [](MlirDialectRegistry registry, const std::string &dialectNamespace) {

#define MLIR_DEFINE_CAPI_DIALECT_REGISTRATION_(NAMESPACE)                      \
  if (dialectNamespace == #NAMESPACE) {                                        \
    mlirDialectHandleInsertDialect(mlirGetDialectHandle__##NAMESPACE##__(),    \
                                   registry);                                  \
    return;                                                                    \
  }

#define FORALL_DIALECTS(_)                                                     \
  _(acc)                                                                       \
  _(affine)                                                                    \
  _(amdgpu)                                                                    \
  _(amx)                                                                       \
  _(arith)                                                                     \
  _(arm_neon)                                                                  \
  _(arm_sme)                                                                   \
  _(arm_sve)                                                                   \
  _(async)                                                                     \
  _(bufferization)                                                             \
  _(cf)                                                                        \
  _(complex)                                                                   \
  _(emitc)                                                                     \
  _(func)                                                                      \
  _(gpu)                                                                       \
  _(index)                                                                     \
  _(irdl)                                                                      \
  _(linalg)                                                                    \
  _(llvm)                                                                      \
  _(math)                                                                      \
  _(memref)                                                                    \
  _(mesh)                                                                      \
  _(ml_program)                                                                \
  _(nvgpu)                                                                     \
  _(nvvm)                                                                      \
  _(omp)                                                                       \
  _(pdl)                                                                       \
  _(quant)                                                                     \
  _(rocdl)                                                                     \
  _(scf)                                                                       \
  _(shape)                                                                     \
  _(spirv)                                                                     \
  _(tensor)                                                                    \
  _(tosa)                                                                      \
  _(ub)                                                                        \
  _(vector)                                                                    \
  _(x86vector)
        FORALL_DIALECTS(MLIR_DEFINE_CAPI_DIALECT_REGISTRATION_)

#undef MLIR_DEFINE_CAPI_DIALECT_REGISTRATION_
#undef FORALL_DIALECTS
        throw std::runtime_error("unknown dialect namespace: " +
                                 dialectNamespace);
      },
      "dialect_registry"_a, "dialect_namespace"_a);
  m.def(
      "register_operation",
      [](const py::object &dialectClass, bool replace) -> py::cpp_function {
        return py::cpp_function(
            [dialectClass, replace](py::object opClass) -> py::object {
              std::string operationName =
                  opClass.attr("OPERATION_NAME").cast<std::string>();
              PyGlobals::get().registerOperationImpl(operationName, opClass,
                                                     replace);

              // Dict-stuff the new opClass by name onto the dialect class.
              py::object opClassName = opClass.attr("__name__");
              dialectClass.attr(opClassName) = opClass;
              return opClass;
            });
      },
      "dialect_class"_a, py::kw_only(), "replace"_a = false,
      "Produce a class decorator for registering an Operation class as part of "
      "a dialect");
  m.def(
      MLIR_PYTHON_CAPI_TYPE_CASTER_REGISTER_ATTR,
      [](MlirTypeID mlirTypeID, bool replace) -> py::cpp_function {
        return py::cpp_function([mlirTypeID,
                                 replace](py::object typeCaster) -> py::object {
          PyGlobals::get().registerTypeCaster(mlirTypeID, typeCaster, replace);
          return typeCaster;
        });
      },
      "typeid"_a, py::kw_only(), "replace"_a = false,
      "Register a type caster for casting MLIR types to custom user types.");
  m.def(
      MLIR_PYTHON_CAPI_VALUE_CASTER_REGISTER_ATTR,
      [](MlirTypeID mlirTypeID, bool replace) -> py::cpp_function {
        return py::cpp_function(
            [mlirTypeID, replace](py::object valueCaster) -> py::object {
              PyGlobals::get().registerValueCaster(mlirTypeID, valueCaster,
                                                   replace);
              return valueCaster;
            });
      },
      "typeid"_a, py::kw_only(), "replace"_a = false,
      "Register a value caster for casting MLIR values to custom user values.");

  // Define and populate IR submodule.
  auto irModule = m.def_submodule("ir", "MLIR IR Bindings");
  populateIRCore(irModule);
  populateIRAffine(irModule);
  populateIRAttributes(irModule);
  populateIRInterfaces(irModule);
  populateIRTypes(irModule);

  // Define and populate PassManager submodule.
  auto passModule =
      m.def_submodule("passmanager", "MLIR Pass Management Bindings");
  populatePassManagerSubmodule(passModule);
}
