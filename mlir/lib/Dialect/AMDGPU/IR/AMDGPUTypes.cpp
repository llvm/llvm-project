//===- AMDGPUTypes.cpp - MLIR AMDGPU dialect types ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AMDGPU dialect types.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/AMDGPU/IR/AMDGPUTypes.cpp.inc"

void mlir::amdgpu::AMDGPUDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/AMDGPU/IR/AMDGPUTypes.cpp.inc"
      >();
}
