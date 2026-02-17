//===--- DialectAMDGPU.cpp - Pybind module for AMDGPU dialect API support -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/AMDGPU.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "nanobind/nanobind.h"

namespace nb = nanobind;
using namespace llvm;
using namespace mlir::python::nanobind_adaptors;

namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {
namespace amdgpu {
struct TDMBaseType : PyConcreteType<TDMBaseType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAAMDGPUTDMBaseType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirAMDGPUTDMBaseTypeGetTypeID;
  static constexpr const char *pyClassName = "TDMBaseType";
  static inline const MlirStringRef name = mlirAMDGPUTDMBaseTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const PyType &elementType, DefaultingPyMlirContext context) {
          return TDMBaseType(
              context->getRef(),
              mlirAMDGPUTDMBaseTypeGet(context.get()->get(), elementType));
        },
        "Gets an instance of TDMBaseType in the same context",
        nb::arg("element_type"), nb::arg("context").none() = nb::none());
  }
};

struct TDMDescriptorType : PyConcreteType<TDMDescriptorType> {
  static constexpr IsAFunctionTy isaFunction =
      mlirTypeIsAAMDGPUTDMDescriptorType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirAMDGPUTDMDescriptorTypeGetTypeID;
  static constexpr const char *pyClassName = "TDMDescriptorType";
  static inline const MlirStringRef name = mlirAMDGPUTDMDescriptorTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          return TDMDescriptorType(
              context->getRef(),
              mlirAMDGPUTDMDescriptorTypeGet(context.get()->get()));
        },
        "Gets an instance of TDMDescriptorType in the same context",
        nb::arg("context").none() = nb::none());
  }
};

struct TDMGatherBaseType : PyConcreteType<TDMGatherBaseType> {
  static constexpr IsAFunctionTy isaFunction =
      mlirTypeIsAAMDGPUTDMGatherBaseType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirAMDGPUTDMGatherBaseTypeGetTypeID;
  static constexpr const char *pyClassName = "TDMGatherBaseType";
  static inline const MlirStringRef name = mlirAMDGPUTDMGatherBaseTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const PyType &elementType, const PyType &indexType,
           DefaultingPyMlirContext context) {
          return TDMGatherBaseType(
              context->getRef(),
              mlirAMDGPUTDMGatherBaseTypeGet(context.get()->get(), elementType,
                                             indexType));
        },
        "Gets an instance of TDMGatherBaseType in the same context",
        nb::arg("element_type"), nb::arg("index_type"),
        nb::arg("context").none() = nb::none());
  }
};

static void populateDialectAMDGPUSubmodule(nb::module_ &m) {
  TDMBaseType::bind(m);
  TDMDescriptorType::bind(m);
  TDMGatherBaseType::bind(m);
}
} // namespace amdgpu
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

NB_MODULE(_mlirDialectsAMDGPU, m) {
  m.doc() = "MLIR AMDGPU dialect.";

  mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::amdgpu::
      populateDialectAMDGPUSubmodule(m);
}
