//===- GPUSPIRVPasses.cpp - C API for GPU SPIRV passes --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/CAPI/Pass.h"
#include "mlir/Dialect/GPU/Transforms/SPIRV/Passes.h"
#include "mlir/Pass/Pass.h"

// Must include the declarations as they carry important visibility attributes.
#include "mlir/Dialect/GPU/Transforms/SPIRV/Passes.capi.h.inc"

using namespace mlir;

#ifdef __cplusplus
extern "C" {
#endif

#include "mlir/Dialect/GPU/Transforms/SPIRV/Passes.capi.cpp.inc"

#ifdef __cplusplus
}
#endif
