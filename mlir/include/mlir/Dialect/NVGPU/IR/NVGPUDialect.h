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

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/NVGPU/IR/NVGPUEnums.h.inc"

constexpr int kWarpSize = 32;

/// M size of wgmma.mma_async instruction
constexpr int kWgmmaSizeM = 64;
/// Maximum tensor dimension that TMA supports
constexpr int kMaxTMATensorDimension = 5;

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/NVGPU/IR/NVGPUAttrDefs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/NVGPU/IR/NVGPUTypes.h.inc"

#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/NVGPU/IR/NVGPU.h.inc"

#endif // MLIR_DIALECT_NVGPU_NVGPUDIALECT_H_
