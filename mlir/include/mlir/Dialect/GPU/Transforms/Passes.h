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

#ifndef MLIR_DIALECT_GPU_TRANSFORMS_PASSES_H_
#define MLIR_DIALECT_GPU_TRANSFORMS_PASSES_H_

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Pass/Pass.h"

namespace llvm {
class TargetMachine;
class LLVMContext;
class Module;
} // namespace llvm

namespace mlir {
class TypeConverter;
class ConversionTarget;
namespace func {
class FuncOp;
} // namespace func

#define GEN_PASS_DECL
#include "mlir/Dialect/GPU/Transforms/Passes.h.inc"

/// Pass that moves ops which are likely an index computation into gpu.launch
/// body.
std::unique_ptr<Pass> createGpuLauchSinkIndexComputationsPass();

/// Replaces `gpu.launch` with `gpu.launch_func` by moving the region into
/// a separate kernel function.
std::unique_ptr<OperationPass<ModuleOp>>
createGpuKernelOutliningPass(StringRef dataLayoutStr = StringRef());

/// Rewrites a function region so that GPU ops execute asynchronously.
std::unique_ptr<OperationPass<func::FuncOp>> createGpuAsyncRegionPass();

/// Maps the parallel loops found in the given function to workgroups. The first
/// loop encountered will be mapped to the global workgroup and the second loop
/// encountered to the local workgroup. Within each mapping, the first three
/// dimensions are mapped to x/y/z hardware ids and all following dimensions are
/// mapped to sequential loops.
std::unique_ptr<OperationPass<func::FuncOp>> createGpuMapParallelLoopsPass();

/// Collect a set of patterns to rewrite all-reduce ops within the GPU dialect.
void populateGpuAllReducePatterns(RewritePatternSet &patterns);

/// Collect all patterns to rewrite ops within the GPU dialect.
inline void populateGpuRewritePatterns(RewritePatternSet &patterns) {
  populateGpuAllReducePatterns(patterns);
}

namespace gpu {
/// A function that maps a MemorySpace enum to a target-specific integer value.
using MemorySpaceMapping =
    std::function<unsigned(gpu::AddressSpace gpuAddressSpace)>;

/// Populates type conversion rules for lowering memory space attributes to
/// numeric values.
void populateMemorySpaceAttributeTypeConversions(
    TypeConverter &typeConverter, const MemorySpaceMapping &mapping);

/// Populates patterns to lower memory space attributes to numeric values.
void populateMemorySpaceLoweringPatterns(TypeConverter &typeConverter,
                                         RewritePatternSet &patterns);

/// Populates legality rules for lowering memory space attriutes to numeric
/// values.
void populateLowerMemorySpaceOpLegality(ConversionTarget &target);

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

  /// Hook allowing the application of optimizations before codegen
  /// By default, does nothing
  virtual LogicalResult optimizeLlvm(llvm::Module &llvmModule,
                                     llvm::TargetMachine &targetMachine);

  /// Translates the 'getOperation()' result to an LLVM module.
  virtual std::unique_ptr<llvm::Module>
  translateToLLVMIR(llvm::LLVMContext &llvmContext);

private:
  /// Creates the LLVM target machine to generate the ISA.
  std::unique_ptr<llvm::TargetMachine> createTargetMachine();

  /// Translates the module to ISA
  Optional<std::string> translateToISA(llvm::Module &llvmModule,
                                       llvm::TargetMachine &targetMachine);

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

/// Create an instance of the GPU kernel function to CUBIN binary serialization
/// pass.
std::unique_ptr<Pass> createGpuSerializeToCubinPass(StringRef triple,
                                                    StringRef chip,
                                                    StringRef features);

/// Create an instance of the GPU kernel function to HSAco binary serialization
/// pass.
std::unique_ptr<Pass> createGpuSerializeToHsacoPass(StringRef triple,
                                                    StringRef arch,
                                                    StringRef features,
                                                    int optLevel);

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/GPU/Transforms/Passes.h.inc"

} // namespace mlir

#endif // MLIR_DIALECT_GPU_TRANSFORMS_PASSES_H_
