//===- AMDGPUDialect.h - MLIR Dialect for AMDGPU ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares a dialect for MLIR wrappers around AMDGPU-specific
// intrinssics and for other AMD GPU-specific functionality.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AMDGPU_AMDGPUDIALECT_H_
#define MLIR_DIALECT_AMDGPU_AMDGPUDIALECT_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/AMDGPU/AMDGPUDialect.h.inc"

#include "mlir/Dialect/AMDGPU/AMDGPUEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/AMDGPU/AMDGPUAttributes.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/AMDGPU/AMDGPU.h.inc"

#endif // MLIR_DIALECT_AMDGPU_AMDGPUDIALECT_H_
