//===- LowerGPUToCUBIN.cpp - Convert GPU kernel to CUBIN blob -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that serializes a gpu module into CUBIN blob and
// adds that blob as a string attribute of the module.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "llvm/Support/Debug.h"

#if MLIR_GPU_TO_CUBIN_PASS_ENABLE
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Threading.h"

#include <cuda.h>

using namespace mlir;

static void emitCudaError(const llvm::Twine &expr, const char *buffer,
                          CUresult result, Location loc) {
  const char *error = nullptr;
  cuGetErrorString(result, &error);
  emitError(loc,
            expr.concat(error ? " failed with error code " + llvm::Twine{error}
                              : llvm::Twine(" failed with unknown error "))
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

namespace {
class SerializeToCubinPass
    : public PassWrapper<SerializeToCubinPass, gpu::SerializeToBlobPass> {
  static llvm::once_flag initializeBackendOnce;

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SerializeToCubinPass)

  SerializeToCubinPass(StringRef triple = "nvptx64-nvidia-cuda",
                       StringRef chip = "sm_35", StringRef features = "+ptx60",
                       int optLevel = 2, bool dumpPtx = false);

  StringRef getArgument() const override { return "gpu-to-cubin"; }
  StringRef getDescription() const override {
    return "Lower GPU kernel function to CUBIN binary annotations";
  }

private:
  // Serializes PTX to CUBIN.
  std::unique_ptr<std::vector<char>>
  serializeISA(const std::string &isa) override;
};
} // namespace

// Sets the 'option' to 'value' unless it already has a value.
static void maybeSetOption(Pass::Option<std::string> &option, StringRef value) {
  if (!option.hasValue())
    option = value.str();
}

llvm::once_flag SerializeToCubinPass::initializeBackendOnce;

SerializeToCubinPass::SerializeToCubinPass(StringRef triple, StringRef chip,
                                           StringRef features, int optLevel,
                                           bool dumpPtx) {
  // No matter how this pass is constructed, ensure that the NVPTX backend
  // is initialized exactly once.
  llvm::call_once(initializeBackendOnce, []() {
    // Initialize LLVM NVPTX backend.
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
  });

  maybeSetOption(this->triple, triple);
  maybeSetOption(this->chip, chip);
  maybeSetOption(this->features, features);
  this->dumpPtx = dumpPtx;
  if (this->optLevel.getNumOccurrences() == 0)
    this->optLevel.setValue(optLevel);
}

std::unique_ptr<std::vector<char>>
SerializeToCubinPass::serializeISA(const std::string &isa) {
  Location loc = getOperation().getLoc();
  char jitErrorBuffer[4096] = {0};

  RETURN_ON_CUDA_ERROR(cuInit(0));

  // Linking requires a device context.
  CUdevice device;
  RETURN_ON_CUDA_ERROR(cuDeviceGet(&device, 0));
  CUcontext context;
  // Use the primary context.
  RETURN_ON_CUDA_ERROR(cuDevicePrimaryCtxRetain(&context, device));
  // Push the primary context so that the next CUDA operations
  // actually use it.
  RETURN_ON_CUDA_ERROR(cuCtxPushCurrent(context));
  CUlinkState linkState;

  CUjit_option jitOptions[] = {CU_JIT_ERROR_LOG_BUFFER,
                               CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
  void *jitOptionsVals[] = {jitErrorBuffer,
                            reinterpret_cast<void *>(sizeof(jitErrorBuffer))};

  RETURN_ON_CUDA_ERROR(cuLinkCreate(2,              /* number of jit options */
                                    jitOptions,     /* jit options */
                                    jitOptionsVals, /* jit option values */
                                    &linkState));

  auto kernelName = getOperation().getName().str();
  if (dumpPtx) {
    llvm::dbgs() << " Kernel Name : [" << kernelName << "]\n";
    llvm::dbgs() << isa << "\n";
  }
  RETURN_ON_CUDA_ERROR(cuLinkAddData(
      linkState, CUjitInputType::CU_JIT_INPUT_PTX,
      const_cast<void *>(static_cast<const void *>(isa.c_str())), isa.length(),
      kernelName.c_str(), 0, /* number of jit options */
      nullptr,               /* jit options */
      nullptr                /* jit option values */
      ));

  void *cubinData;
  size_t cubinSize;
  RETURN_ON_CUDA_ERROR(cuLinkComplete(linkState, &cubinData, &cubinSize));

  char *cubinAsChar = static_cast<char *>(cubinData);
  auto result =
      std::make_unique<std::vector<char>>(cubinAsChar, cubinAsChar + cubinSize);

  // This will also destroy the cubin data.
  RETURN_ON_CUDA_ERROR(cuLinkDestroy(linkState));
  // Pop and release the primary context.
  CUcontext poppedContext;
  RETURN_ON_CUDA_ERROR(cuCtxPopCurrent(&poppedContext));
  RETURN_ON_CUDA_ERROR(cuDevicePrimaryCtxRelease(device));

  return result;
}

// Register pass to serialize GPU kernel functions to a CUBIN binary annotation.
void mlir::registerGpuSerializeToCubinPass() {
  PassRegistration<SerializeToCubinPass> registerSerializeToCubin(
      [] { return std::make_unique<SerializeToCubinPass>(); });
}

std::unique_ptr<Pass> mlir::createGpuSerializeToCubinPass(StringRef triple,
                                                          StringRef arch,
                                                          StringRef features,
                                                          int optLevel,
                                                          bool dumpPtx) {
  return std::make_unique<SerializeToCubinPass>(triple, arch, features,
                                                optLevel, dumpPtx);
}

#else  // MLIR_GPU_TO_CUBIN_PASS_ENABLE
void mlir::registerGpuSerializeToCubinPass() {}
#endif // MLIR_GPU_TO_CUBIN_PASS_ENABLE
