//===- SPIRVWebGPUTransforms.h - WebGPU-specific Transforms -*- C++ -----*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines SPIR-V transforms used when targetting WebGPU.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_TRANSFORMS_SPIRV_WEBGPU_TRANSFORMS_H
#define MLIR_DIALECT_SPIRV_TRANSFORMS_SPIRV_WEBGPU_TRANSFORMS_H

#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace spirv {

/// Appends to a pattern list additional patterns to expand extended
/// multiplication ops into regular arithmetic ops. Extended multiplication ops
/// are not supported by the WebGPU Shading Language (WGSL).
void populateSPIRVExpandExtendedMultiplicationPatterns(
    RewritePatternSet &patterns);

} // namespace spirv
} // namespace mlir

#endif // MLIR_DIALECT_SPIRV_TRANSFORMS_SPIRV_WEBGPU_TRANSFORMS_H
