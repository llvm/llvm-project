//===- NVGPUDialect.h - MLIR Dialect for NVGPU ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Target dialect for NVGPU in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_NVGPU_NVGPUDIALECT_H_
#define MLIR_DIALECT_NVGPU_NVGPUDIALECT_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace nvgpu {

/// Device-side token storage type. There is only one type of device-side token.
class DeviceAsyncTokenType
    : public Type::TypeBase<DeviceAsyncTokenType, Type, TypeStorage> {
public:
  // Used for generic hooks in TypeBase.
  using Base::Base;
};

} // namespace nvgpu
} // namespace mlir

#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/NVGPU/IR/NVGPU.h.inc"

#endif // MLIR_DIALECT_NVGPU_NVGPUDIALECT_H_
