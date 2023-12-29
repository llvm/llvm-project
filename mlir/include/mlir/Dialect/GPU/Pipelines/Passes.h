//===- Passes.h - GPU NVVM pipeline entry points --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GPU_PIPELINES_PASSES_H_
#define MLIR_DIALECT_GPU_PIPELINES_PASSES_H_

namespace mlir {
namespace gpu {
void registerGPUToNVVMPipeline();
} // namespace gpu
} // namespace mlir

#endif
