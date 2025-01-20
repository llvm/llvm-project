//===--- DialectNVGPU.cpp - Pybind module for NVGPU dialect API support ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/NVGPU.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/Bindings/Python/Nanobind.h"

namespace nb = nanobind;
using namespace llvm;
using namespace mlir;
using namespace mlir::python;
using namespace mlir::python::nanobind_adaptors;

static void populateDialectNVGPUSubmodule(const nb::module_ &m) {
  auto nvgpuTensorMapDescriptorType = mlir_type_subclass(
      m, "TensorMapDescriptorType", mlirTypeIsANVGPUTensorMapDescriptorType);

  nvgpuTensorMapDescriptorType.def_classmethod(
      "get",
      [](nb::object cls, MlirType tensorMemrefType, int swizzle, int l2promo,
         int oobFill, int interleave, MlirContext ctx) {
        return cls(mlirNVGPUTensorMapDescriptorTypeGet(
            ctx, tensorMemrefType, swizzle, l2promo, oobFill, interleave));
      },
      "Gets an instance of TensorMapDescriptorType in the same context",
      nb::arg("cls"), nb::arg("tensor_type"), nb::arg("swizzle"),
      nb::arg("l2promo"), nb::arg("oob_fill"), nb::arg("interleave"),
      nb::arg("ctx").none() = nb::none());
}

NB_MODULE(_mlirDialectsNVGPU, m) {
  m.doc() = "MLIR NVGPU dialect.";

  populateDialectNVGPUSubmodule(m);
}
