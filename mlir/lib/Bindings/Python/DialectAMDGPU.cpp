//===--- DialectAMDGPU.cpp - Pybind module for AMDGPU dialect API support -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/AMDGPU.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "nanobind/nanobind.h"

namespace nb = nanobind;
using namespace llvm;
using namespace mlir;
using namespace mlir::python;
using namespace mlir::python::nanobind_adaptors;

static void populateDialectAMDGPUSubmodule(const nb::module_ &m) {
  auto amdgpuTDMBaseType =
      mlir_type_subclass(m, "TDMBaseType", mlirTypeIsAAMDGPUTDMBaseType,
                         mlirAMDGPUTDMBaseTypeGetTypeID);

  amdgpuTDMBaseType.def_classmethod(
      "get",
      [](const nb::object &cls, MlirType elementType, MlirContext ctx) {
        return cls(mlirAMDGPUTDMBaseTypeGet(ctx, elementType));
      },
      "Gets an instance of TDMBaseType in the same context", nb::arg("cls"),
      nb::arg("element_type"), nb::arg("ctx") = nb::none());

  auto amdgpuTDMDescriptorType = mlir_type_subclass(
      m, "TDMDescriptorType", mlirTypeIsAAMDGPUTDMDescriptorType,
      mlirAMDGPUTDMDescriptorTypeGetTypeID);

  amdgpuTDMDescriptorType.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirAMDGPUTDMDescriptorTypeGet(ctx));
      },
      "Gets an instance of TDMDescriptorType in the same context",
      nb::arg("cls"), nb::arg("ctx") = nb::none());

  auto amdgpuTDMGatherBaseType = mlir_type_subclass(
      m, "TDMGatherBaseType", mlirTypeIsAAMDGPUTDMGatherBaseType,
      mlirAMDGPUTDMGatherBaseTypeGetTypeID);

  amdgpuTDMGatherBaseType.def_classmethod(
      "get",
      [](const nb::object &cls, MlirType elementType, MlirType indexType,
         MlirContext ctx) {
        return cls(mlirAMDGPUTDMGatherBaseTypeGet(ctx, elementType, indexType));
      },
      "Gets an instance of TDMGatherBaseType in the same context",
      nb::arg("cls"), nb::arg("element_type"), nb::arg("index_type"),
      nb::arg("ctx") = nb::none());
};

NB_MODULE(_mlirDialectsAMDGPU, m) {
  m.doc() = "MLIR AMDGPU dialect.";

  populateDialectAMDGPUSubmodule(m);
}
