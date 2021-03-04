//===- mlir-cuda-runner.cpp - MLIR CUDA Execution Driver-------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that executes an MLIR file on the GPU by
// translating MLIR to NVVM/LVVM IR before JIT-compiling and executing the
// latter.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"

#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/ExecutionEngine/JitRunner.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

#include "cuda.h"

using namespace mlir;

static void emitCudaError(const llvm::Twine &expr, const char *buffer,
                          CUresult result, Location loc) {
  const char *error;
  cuGetErrorString(result, &error);
  emitError(loc, expr.concat(" failed with error code ")
                     .concat(llvm::Twine{error})
                     .concat("[")
                     .concat(buffer)
                     .concat("]"));
}

#define RETURN_ON_CUDA_ERROR(expr)                                             \
  do {                                                                         \
    if (auto status = (expr)) {                                                \
      emitCudaError(#expr, jitErrorBuffer, status, loc);                       \
      return {};                                                               \
    }                                                                          \
  } while (false)

OwnedBlob compilePtxToCubin(const std::string ptx, Location loc,
                            StringRef name) {
  char jitErrorBuffer[4096] = {0};

  RETURN_ON_CUDA_ERROR(cuInit(0));

  // Linking requires a device context.
  CUdevice device;
  RETURN_ON_CUDA_ERROR(cuDeviceGet(&device, 0));
  CUcontext context;
  RETURN_ON_CUDA_ERROR(cuCtxCreate(&context, 0, device));
  CUlinkState linkState;

  CUjit_option jitOptions[] = {CU_JIT_ERROR_LOG_BUFFER,
                               CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
  void *jitOptionsVals[] = {jitErrorBuffer,
                            reinterpret_cast<void *>(sizeof(jitErrorBuffer))};

  RETURN_ON_CUDA_ERROR(cuLinkCreate(2,              /* number of jit options */
                                    jitOptions,     /* jit options */
                                    jitOptionsVals, /* jit option values */
                                    &linkState));

  RETURN_ON_CUDA_ERROR(
      cuLinkAddData(linkState, CUjitInputType::CU_JIT_INPUT_PTX,
                    const_cast<void *>(static_cast<const void *>(ptx.c_str())),
                    ptx.length(), name.str().data(), /* kernel name */
                    0,                               /* number of jit options */
                    nullptr,                         /* jit options */
                    nullptr                          /* jit option values */
                    ));

  void *cubinData;
  size_t cubinSize;
  RETURN_ON_CUDA_ERROR(cuLinkComplete(linkState, &cubinData, &cubinSize));

  char *cubinAsChar = static_cast<char *>(cubinData);
  OwnedBlob result =
      std::make_unique<std::vector<char>>(cubinAsChar, cubinAsChar + cubinSize);

  // This will also destroy the cubin data.
  RETURN_ON_CUDA_ERROR(cuLinkDestroy(linkState));
  RETURN_ON_CUDA_ERROR(cuCtxDestroy(context));

  return result;
}

struct GpuToCubinPipelineOptions
    : public mlir::PassPipelineOptions<GpuToCubinPipelineOptions> {
  Option<std::string> gpuBinaryAnnotation{
      *this, "gpu-binary-annotation",
      llvm::cl::desc("Annotation attribute string for GPU binary")};
};

// Register cuda-runner specific passes.
static void registerCudaRunnerPasses() {
  PassPipelineRegistration<GpuToCubinPipelineOptions> registerGpuToCubin(
      "gpu-to-cubin", "Generate CUBIN from gpu.launch regions",
      [&](OpPassManager &pm, const GpuToCubinPipelineOptions &options) {
        pm.addPass(createGpuKernelOutliningPass());
        auto &kernelPm = pm.nest<gpu::GPUModuleOp>();
        kernelPm.addPass(createStripDebugInfoPass());
        kernelPm.addPass(createLowerGpuOpsToNVVMOpsPass());
        kernelPm.addPass(createConvertGPUKernelToBlobPass(
            translateModuleToLLVMIR, compilePtxToCubin, "nvptx64-nvidia-cuda",
            "sm_35", "+ptx60", options.gpuBinaryAnnotation));
      });
  registerGPUPasses();
  registerGpuToLLVMConversionPassPass();
  registerAsyncPasses();
  registerConvertAsyncToLLVMPass();
  registerConvertStandardToLLVMPass();
}

static LogicalResult runMLIRPasses(ModuleOp module,
                                   PassPipelineCLParser &passPipeline) {
  PassManager pm(module.getContext(), PassManager::Nesting::Implicit);
  applyPassManagerCLOptions(pm);

  auto errorHandler = [&](const Twine &msg) {
    emitError(UnknownLoc::get(module.getContext())) << msg;
    return failure();
  };

  // Build the provided pipeline.
  if (failed(passPipeline.addToPipeline(pm, errorHandler)))
    return failure();

  // Run the pipeline.
  return pm.run(module);
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Initialize LLVM NVPTX backend.
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();

  mlir::initializeLLVMPasses();

  registerCudaRunnerPasses();
  PassPipelineCLParser passPipeline("", "Compiler passes to run");
  registerPassManagerCLOptions();

  auto mlirTransformer = [&](ModuleOp module) {
    return runMLIRPasses(module, passPipeline);
  };

  mlir::JitRunnerConfig jitRunnerConfig;
  jitRunnerConfig.mlirTransformer = mlirTransformer;

  mlir::DialectRegistry registry;
  registry.insert<mlir::LLVM::LLVMDialect, mlir::NVVM::NVVMDialect,
                  mlir::async::AsyncDialect, mlir::gpu::GPUDialect,
                  mlir::StandardOpsDialect>();
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);

  return mlir::JitRunnerMain(argc, argv, registry, jitRunnerConfig);
}
