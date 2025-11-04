//===- Passes.h - GPU pipeline entry points--------------------------------===//
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
  PassOptions::Option<std::string> cmdOptions{
      *this, "ptxas-cmd-options",
      llvm::cl::desc(
          "Command line options to pass to the downstream compiler."),
      llvm::cl::init("")};
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

// Options for the gpu to xevm pipeline.
struct GPUToXeVMPipelineOptions
    : public PassPipelineOptions<GPUToXeVMPipelineOptions> {
  PassOptions::Option<std::string> xegpuOpLevel{
      *this, "xegpu-op-level",
      llvm::cl::desc("Granularity of XeGPU operations to target: workgroup | "
                     "subgroup | lane"),
      llvm::cl::init("workgroup")};
  // General lowering controls.
  PassOptions::Option<bool> use64bitIndex{
      *this, "use-64bit-index",
      llvm::cl::desc("Bitwidth of the index type (host & device)"),
      llvm::cl::init(true)};
  PassOptions::Option<bool> kernelBarePtrCallConv{
      *this, "kernel-bare-ptr-calling-convention",
      llvm::cl::desc("Use bare pointer calling convention for device kernels"),
      llvm::cl::init(false)};
  PassOptions::Option<bool> hostBarePtrCallConv{
      *this, "host-bare-ptr-calling-convention",
      llvm::cl::desc("Use bare pointer calling convention for host launches"),
      llvm::cl::init(false)};
  PassOptions::Option<std::string> binaryFormat{
      *this, "binary-format",
      llvm::cl::desc("Final GPU binary emission format (e.g. fatbin)"),
      llvm::cl::init("fatbin")};
  // Options mirroring xevm-attach-target (GpuXeVMAttachTarget).
  PassOptions::Option<std::string> xevmModuleMatcher{
      *this, "xevm-module-matcher",
      llvm::cl::desc("Regex to match gpu.module names for XeVM target attach"),
      llvm::cl::init("")};
  PassOptions::Option<std::string> zebinTriple{
      *this, "zebin-triple", llvm::cl::desc("Target triple for XeVM codegen"),
      llvm::cl::init("spirv64-unknown-unknown")};
  PassOptions::Option<std::string> zebinChip{
      *this, "zebin-chip", llvm::cl::desc("Target chip (e.g. pvc, bmg)"),
      llvm::cl::init("bmg")};
  PassOptions::Option<unsigned> optLevel{
      *this, "opt-level",
      llvm::cl::desc("Optimization level for attached target/codegen"),
      llvm::cl::init(2)};
  PassOptions::Option<std::string> cmdOptions{
      *this, "igc-cmd-options",
      llvm::cl::desc("Additional downstream compiler command line options"),
      llvm::cl::init("")};
};

//===----------------------------------------------------------------------===//
// Building and Registering.
//===----------------------------------------------------------------------===//

/// Adds the GPU to NVVM pipeline to the given pass manager. Transforms main
/// dialects into NVVM targets. Begins with GPU code regions, then handles host
/// code.
void buildLowerToNVVMPassPipeline(OpPassManager &pm,
                                  const GPUToNVVMPipelineOptions &options);

/// Adds the GPU to XeVM pipeline to the given pass manager. Transforms main
/// dialects into XeVM targets. Begins with GPU code regions, then handles host
/// code.
void buildLowerToXeVMPassPipeline(OpPassManager &pm,
                                  const GPUToXeVMPipelineOptions &options);

/// Register all pipelines for the `gpu` dialect.
void registerGPUToNVVMPipeline();
void registerGPUToXeVMPipeline();

} // namespace gpu
} // namespace mlir

#endif
