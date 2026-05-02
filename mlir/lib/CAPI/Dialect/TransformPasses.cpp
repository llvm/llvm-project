//===- TransformPasses.cpp - C API for Transform Dialect Passes -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/CAPI/Pass.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"

// Must include the declarations as they carry important visibility attributes.
#include "mlir/Dialect/Transform/Transforms/Passes.capi.h.inc"
using namespace mlir;
using namespace mlir::transform;

#ifdef __cplusplus
extern "C" {
#endif

#include "mlir/Dialect/Transform/Transforms/Passes.capi.cpp.inc"

#ifdef __cplusplus
}
#endif
