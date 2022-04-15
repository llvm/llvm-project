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

#include "mlir/Dialect/NVGPU/NVGPUDialect.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/NVGPU/NVGPU.h.inc"

#endif // MLIR_DIALECT_NVGPU_NVGPUDIALECT_H_
