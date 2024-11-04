//===- Passes.h - GPU NVVM pipeline entry points --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GPU_PIPELINES_PASSES_H_
#define MLIR_DIALECT_GPU_PIPELINES_PASSES_H_

#include "mlir/Pass/PassOptions.h"

namespace mlir {
namespace gpu {

/// Options for the gpu to nvvm pipeline.
struct GPUToNVVMPipelineOptions
    : public PassPipelineOptions<GPUToNVVMPipelineOptions> {
  PassOptions::Option<int64_t> indexBitWidth{
      *this, "index-bitwidth",
      llvm::cl::desc("Bitwidth of the index type for the host (warning this "
                     "should be 64 until the GPU layering is fixed)"),
      llvm::cl::init(64)};
  PassOptions::Option<std::string> cubinTriple{
      *this, "cubin-triple",
      llvm::cl::desc("Triple to use to serialize to cubin."),
      llvm::cl::init("nvptx64-nvidia-cuda")};
  PassOptions::Option<std::string> cubinChip{
      *this, "cubin-chip", llvm::cl::desc("Chip to use to serialize to cubin."),
      llvm::cl::init("sm_50")};
  PassOptions::Option<std::string> cubinFeatures{
      *this, "cubin-features",
      llvm::cl::desc("Features to use to serialize to cubin."),
      llvm::cl::init("+ptx60")};
  PassOptions::Option<std::string> cubinFormat{
      *this, "cubin-format",
      llvm::cl::desc("Compilation format to use to serialize to cubin."),
      llvm::cl::init("fatbin")};
  PassOptions::Option<int> optLevel{
      *this, "opt-level",
      llvm::cl::desc("Optimization level for NVVM compilation"),
      llvm::cl::init(2)};
  PassOptions::Option<bool> kernelUseBarePtrCallConv{
      *this, "kernel-bare-ptr-calling-convention",
      llvm::cl::desc(
          "Whether to use the bareptr calling convention on the kernel "
          "(warning this should be false until the GPU layering is fixed)"),
      llvm::cl::init(false)};
  PassOptions::Option<bool> hostUseBarePtrCallConv{
      *this, "host-bare-ptr-calling-convention",
      llvm::cl::desc(
          "Whether to use the bareptr calling convention on the host (warning "
          "this should be false until the GPU layering is fixed)"),
      llvm::cl::init(false)};
};

//===----------------------------------------------------------------------===//
// Building and Registering.
//===----------------------------------------------------------------------===//

/// Adds the GPU to NVVM pipeline to the given pass manager. Transforms main
/// dialects into NVVM targets. Begins with GPU code regions, then handles host
/// code.
void buildLowerToNVVMPassPipeline(OpPassManager &pm,
                                  const GPUToNVVMPipelineOptions &options);

/// Register all pipeleines for the `gpu` dialect.
void registerGPUToNVVMPipeline();

} // namespace gpu
} // namespace mlir

#endif
