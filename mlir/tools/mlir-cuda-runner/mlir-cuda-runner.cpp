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
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/ExecutionEngine/JitRunner.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/NVVMIR.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

#include "cuda.h"

using namespace mlir;

inline void emit_cuda_error(const llvm::Twine &message, const char *buffer,
                            CUresult error, Location loc) {
  emitError(loc, message.concat(" failed with error code ")
                     .concat(llvm::Twine{error})
                     .concat("[")
                     .concat(buffer)
                     .concat("]"));
}

#define RETURN_ON_CUDA_ERROR(expr, msg)                                        \
  {                                                                            \
    auto _cuda_error = (expr);                                                 \
    if (_cuda_error != CUDA_SUCCESS) {                                         \
      emit_cuda_error(msg, jitErrorBuffer, _cuda_error, loc);                  \
      return {};                                                               \
    }                                                                          \
  }

OwnedBlob compilePtxToCubin(const std::string ptx, Location loc,
                            StringRef name) {
  char jitErrorBuffer[4096] = {0};

  RETURN_ON_CUDA_ERROR(cuInit(0), "cuInit");

  // Linking requires a device context.
  CUdevice device;
  RETURN_ON_CUDA_ERROR(cuDeviceGet(&device, 0), "cuDeviceGet");
  CUcontext context;
  RETURN_ON_CUDA_ERROR(cuCtxCreate(&context, 0, device), "cuCtxCreate");
  CUlinkState linkState;

  CUjit_option jitOptions[] = {CU_JIT_ERROR_LOG_BUFFER,
                               CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
  void *jitOptionsVals[] = {jitErrorBuffer,
                            reinterpret_cast<void *>(sizeof(jitErrorBuffer))};

  RETURN_ON_CUDA_ERROR(cuLinkCreate(2,              /* number of jit options */
                                    jitOptions,     /* jit options */
                                    jitOptionsVals, /* jit option values */
                                    &linkState),
                       "cuLinkCreate");

  RETURN_ON_CUDA_ERROR(
      cuLinkAddData(linkState, CUjitInputType::CU_JIT_INPUT_PTX,
                    const_cast<void *>(static_cast<const void *>(ptx.c_str())),
                    ptx.length(), name.data(), /* kernel name */
                    0,                         /* number of jit options */
                    nullptr,                   /* jit options */
                    nullptr                    /* jit option values */
                    ),
      "cuLinkAddData");

  void *cubinData;
  size_t cubinSize;
  RETURN_ON_CUDA_ERROR(cuLinkComplete(linkState, &cubinData, &cubinSize),
                       "cuLinkComplete");

  char *cubinAsChar = static_cast<char *>(cubinData);
  OwnedBlob result =
      std::make_unique<std::vector<char>>(cubinAsChar, cubinAsChar + cubinSize);

  // This will also destroy the cubin data.
  RETURN_ON_CUDA_ERROR(cuLinkDestroy(linkState), "cuLinkDestroy");

  return result;
}

static LogicalResult runMLIRPasses(ModuleOp m) {
  PassManager pm(m.getContext());
  applyPassManagerCLOptions(pm);

  const char gpuBinaryAnnotation[] = "nvvm.cubin";
  pm.addPass(createGpuKernelOutliningPass());
  auto &kernelPm = pm.nest<gpu::GPUModuleOp>();
  kernelPm.addPass(createStripDebugInfoPass());
  kernelPm.addPass(createLowerGpuOpsToNVVMOpsPass());
  kernelPm.addPass(createConvertGPUKernelToBlobPass(
      translateModuleToNVVMIR, compilePtxToCubin, "nvptx64-nvidia-cuda",
      "sm_35", "+ptx60", gpuBinaryAnnotation));
  pm.addPass(createLowerToLLVMPass());
  pm.addPass(
      createConvertGpuLaunchFuncToGpuRuntimeCallsPass(gpuBinaryAnnotation));

  return pm.run(m);
}

int main(int argc, char **argv) {
  registerPassManagerCLOptions();
  mlir::registerAllDialects();
  llvm::InitLLVM y(argc, argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Initialize LLVM NVPTX backend.
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();

  mlir::initializeLLVMPasses();
  return mlir::JitRunnerMain(argc, argv, &runMLIRPasses);
}
