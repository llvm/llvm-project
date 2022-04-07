//===- NVGPUDialect.cpp - MLIR NVGPU ops implementation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the NVGPU dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/NVGPU/NVGPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

#include "mlir/Dialect/NVGPU/NVGPUDialect.cpp.inc"

void nvgpu::NVGPUDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/NVGPU/NVGPU.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/NVGPU/NVGPU.cpp.inc"
