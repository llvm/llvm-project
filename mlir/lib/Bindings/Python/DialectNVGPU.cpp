//===--- DialectNVGPU.cpp - Pybind module for NVGPU dialect API support ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/NVGPU.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;
using namespace llvm;
using namespace mlir::python::nanobind_adaptors;

namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {
namespace nvgpu {
struct TensorMapDescriptorType : PyConcreteType<TensorMapDescriptorType> {
  static constexpr IsAFunctionTy isaFunction =
      mlirTypeIsANVGPUTensorMapDescriptorType;
  static constexpr const char *pyClassName = "TensorMapDescriptorType";
  static inline const MlirStringRef name =
      mlirNVGPUTensorMapDescriptorTypeGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const PyType &tensorMemrefType, int swizzle, int l2promo,
           int oobFill, int interleave, DefaultingPyMlirContext context) {
          return TensorMapDescriptorType(
              context->getRef(), mlirNVGPUTensorMapDescriptorTypeGet(
                                     context.get()->get(), tensorMemrefType,
                                     swizzle, l2promo, oobFill, interleave));
        },
        "Gets an instance of TensorMapDescriptorType in the same context",
        nb::arg("tensor_type"), nb::arg("swizzle"), nb::arg("l2promo"),
        nb::arg("oob_fill"), nb::arg("interleave"),
        nb::arg("context").none() = nb::none());
  }
};
} // namespace nvgpu
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

NB_MODULE(_mlirDialectsNVGPU, m) {
  m.doc() = "MLIR NVGPU dialect.";

  mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::nvgpu::TensorMapDescriptorType::
      bind(m);
}
