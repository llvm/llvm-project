//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GPU_PASSES_H_
#define MLIR_DIALECT_GPU_PASSES_H_

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Pass/Pass.h"

namespace llvm {
class TargetMachine;
class LLVMContext;
class Module;
} // namespace llvm

namespace mlir {
/// Replaces `gpu.launch` with `gpu.launch_func` by moving the region into
/// a separate kernel function.
std::unique_ptr<OperationPass<ModuleOp>> createGpuKernelOutliningPass();

/// Rewrites a function region so that GPU ops execute asynchronously.
std::unique_ptr<OperationPass<FuncOp>> createGpuAsyncRegionPass();

/// Collect a set of patterns to rewrite all-reduce ops within the GPU dialect.
void populateGpuAllReducePatterns(MLIRContext *context,
                                  OwningRewritePatternList &patterns);

/// Collect all patterns to rewrite ops within the GPU dialect.
inline void populateGpuRewritePatterns(MLIRContext *context,
                                       OwningRewritePatternList &patterns) {
  populateGpuAllReducePatterns(context, patterns);
}

namespace gpu {
/// Returns the default annotation name for GPU binary blobs.
std::string getDefaultGpuBinaryAnnotation();

/// Base pass class to serialize kernel functions through LLVM into
/// user-specified IR and add the resulting blob as module attribute.
class SerializeToBlobPass : public OperationPass<gpu::GPUModuleOp> {
public:
  SerializeToBlobPass(TypeID passID);
  SerializeToBlobPass(const SerializeToBlobPass &other);

  void runOnOperation() final;

protected:
  void getDependentDialects(DialectRegistry &registry) const override;

private:
  /// Creates the LLVM target machine to generate the ISA.
  std::unique_ptr<llvm::TargetMachine> createTargetMachine();

  /// Translates the 'getOperation()' result to an LLVM module.
  virtual std::unique_ptr<llvm::Module>
  translateToLLVMIR(llvm::LLVMContext &llvmContext);

  /// Serializes the target ISA to binary form.
  virtual std::unique_ptr<std::vector<char>>
  serializeISA(const std::string &isa) = 0;

protected:
  Option<std::string> triple{*this, "triple",
                             ::llvm::cl::desc("Target triple")};
  Option<std::string> chip{*this, "chip",
                           ::llvm::cl::desc("Target architecture")};
  Option<std::string> features{*this, "features",
                               ::llvm::cl::desc("Target features")};
  Option<std::string> gpuBinaryAnnotation{
      *this, "gpu-binary-annotation",
      llvm::cl::desc("Annotation attribute string for GPU binary"),
      llvm::cl::init(getDefaultGpuBinaryAnnotation())};
};
} // namespace gpu

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Register pass to serialize GPU kernel functions to a CUBIN binary
/// annotation.
void registerGpuSerializeToCubinPass();

/// Register pass to serialize GPU kernel functions to a HSAco binary
/// annotation.
void registerGpuSerializeToHsacoPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/GPU/Passes.h.inc"

} // namespace mlir

#endif // MLIR_DIALECT_GPU_PASSES_H_
