//===- GPUPasses.cpp - C API for GPU Dialect Passes ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/CAPI/Pass.h"
#include "mlir/Dialect/GPU/Transforms/NVVM/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/GPU/Transforms/ROCDL/Passes.h"
#include "mlir/Dialect/GPU/Transforms/SPIRV/Passes.h"
#include "mlir/Dialect/GPU/Transforms/XeVM/Passes.h"
#include "mlir/Pass/Pass.h"

// Must include the declarations as they carry important visibility attributes.
#include "mlir/Dialect/GPU/Transforms/NVVM/Passes.capi.h.inc"
#include "mlir/Dialect/GPU/Transforms/Passes.capi.h.inc"
#include "mlir/Dialect/GPU/Transforms/ROCDL/Passes.capi.h.inc"
#include "mlir/Dialect/GPU/Transforms/SPIRV/Passes.capi.h.inc"
#include "mlir/Dialect/GPU/Transforms/XeVM/Passes.capi.h.inc"

using namespace mlir;

#ifdef __cplusplus
extern "C" {
#endif

#include "mlir/Dialect/GPU/Transforms/NVVM/Passes.capi.cpp.inc"
#include "mlir/Dialect/GPU/Transforms/Passes.capi.cpp.inc"
#include "mlir/Dialect/GPU/Transforms/ROCDL/Passes.capi.cpp.inc"
#include "mlir/Dialect/GPU/Transforms/SPIRV/Passes.capi.cpp.inc"
#include "mlir/Dialect/GPU/Transforms/XeVM/Passes.capi.cpp.inc"

#ifdef __cplusplus
}
#endif
