//===--- DialectNVGPU.cpp - Pybind module for NVGPU dialect API support ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/NVGPU.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace llvm;
using namespace mlir;
using namespace mlir::python;
using namespace mlir::python::adaptors;

static void populateDialectNVGPUSubmodule(const pybind11::module &m) {
  auto nvgpuTensorMapDescriptorType = mlir_type_subclass(
      m, "TensorMapDescriptorType", mlirTypeIsANVGPUTensorMapDescriptorType);

  nvgpuTensorMapDescriptorType.def_classmethod(
      "get",
      [](py::object cls, MlirType tensorMemrefType, int swizzle, int l2promo,
         int oobFill, int interleave, MlirContext ctx) {
        return cls(mlirNVGPUTensorMapDescriptorTypeGet(
            ctx, tensorMemrefType, swizzle, l2promo, oobFill, interleave));
      },
      "Gets an instance of TensorMapDescriptorType in the same context",
      py::arg("cls"), py::arg("tensor_type"), py::arg("swizzle"),
      py::arg("l2promo"), py::arg("oob_fill"), py::arg("interleave"),
      py::arg("ctx") = py::none());
}

PYBIND11_MODULE(_mlirDialectsNVGPU, m) {
  m.doc() = "MLIR NVGPU dialect.";

  populateDialectNVGPUSubmodule(m);
}
