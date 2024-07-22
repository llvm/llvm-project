//===- NVGPU.cpp - C Interface for NVGPU dialect ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/NVGPU.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::nvgpu;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(NVGPU, nvgpu, mlir::nvgpu::NVGPUDialect)

bool mlirTypeIsANVGPUTensorMapDescriptorType(MlirType type) {
  return isa<nvgpu::TensorMapDescriptorType>(unwrap(type));
}

MlirType mlirNVGPUTensorMapDescriptorTypeGet(MlirContext ctx,
                                             MlirType tensorMemrefType,
                                             int swizzle, int l2promo,
                                             int oobFill, int interleave) {
  return wrap(nvgpu::TensorMapDescriptorType::get(
      unwrap(ctx), cast<MemRefType>(unwrap(tensorMemrefType)),
      TensorMapSwizzleKind(swizzle), TensorMapL2PromoKind(l2promo),
      TensorMapOOBKind(oobFill), TensorMapInterleaveKind(interleave)));
}
