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

#if MLIR_GPU_TO_CUBIN_PASS_ENABLE
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/TargetSelect.h"

#include <cuda.h>
#include <nvPTXCompiler.h>

using namespace mlir;

static void emitNvptxError(const llvm::Twine &expr,
                           nvPTXCompilerHandle compiler,
                           nvPTXCompileResult result, Location loc) {
  const char *error;
  auto GetErrMsg = [](nvPTXCompileResult result) -> const char * {
    switch (result) {
    case NVPTXCOMPILE_SUCCESS:
      return "Success";
    case NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE:
      return "Invalid compiler handle";
    case NVPTXCOMPILE_ERROR_INVALID_INPUT:
      return "Invalid input";
    case NVPTXCOMPILE_ERROR_COMPILATION_FAILURE:
      return "Compilation failure";
    case NVPTXCOMPILE_ERROR_INTERNAL:
      return "Internal error";
    case NVPTXCOMPILE_ERROR_OUT_OF_MEMORY:
      return "Out of memory";
    case NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE:
      return "Invocation incomplete";
    case NVPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION:
      return "Unsupported PTX version";
    }
  };
  size_t errorSize;
  auto status = nvPTXCompilerGetErrorLogSize(compiler, &errorSize);
  std::string error_log;
  if (status == NVPTXCOMPILE_SUCCESS) {
    error_log.resize(errorSize);
    status = nvPTXCompilerGetErrorLog(compiler, error_log.data());
    if (status != NVPTXCOMPILE_SUCCESS)
      error_log = "<failed to retrieve compiler error log>";
  }
  emitError(loc, expr.concat(" failed with error code ")
                     .concat(llvm::Twine{GetErrMsg(result)})
                     .concat("[")
                     .concat(error_log)
                     .concat("]"));
}

#define RETURN_ON_CUDA_ERROR(expr)                                             \
  do {                                                                         \
    if (auto status = (expr)) {                                                \
      emitNvptxError(#expr, compiler, status, loc);                            \
      return {};                                                               \
    }                                                                          \
  } while (false)

#define RETURN_ON_NVPTX_ERROR(expr)                                            \
  do {                                                                         \
    nvPTXCompileResult result = (expr);                                        \
    if (result != NVPTXCOMPILE_SUCCESS) {                                      \
      emitNvptxError(#expr, compiler, result, loc);                            \
      return {};                                                               \
    }                                                                          \
  } while (false)

namespace {
class SerializeToCubinPass
    : public PassWrapper<SerializeToCubinPass, gpu::SerializeToBlobPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SerializeToCubinPass)

  SerializeToCubinPass(StringRef triple = "nvptx64-nvidia-cuda",
                       StringRef chip = "sm_35", StringRef features = "+ptx60");

  StringRef getArgument() const override { return "gpu-to-cubin"; }
  StringRef getDescription() const override {
    return "Lower GPU kernel function to CUBIN binary annotations";
  }

private:
  void getDependentDialects(DialectRegistry &registry) const override;

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

SerializeToCubinPass::SerializeToCubinPass(StringRef triple, StringRef chip,
                                           StringRef features) {
  maybeSetOption(this->triple, triple);
  maybeSetOption(this->chip, chip);
  maybeSetOption(this->features, features);
}

void SerializeToCubinPass::getDependentDialects(
    DialectRegistry &registry) const {
  registerNVVMDialectTranslation(registry);
  gpu::SerializeToBlobPass::getDependentDialects(registry);
}

std::unique_ptr<std::vector<char>>
SerializeToCubinPass::serializeISA(const std::string &isa) {
  Location loc = getOperation().getLoc();
  char jitErrorBuffer[4096] = {0};
  nvPTXCompilerHandle compiler;
  nvPTXCompilerCreate(&compiler, isa.length(), isa.c_str());

  nvPTXCompilerCompile(compiler, 0, nullptr);

  size_t cubinSize;
  nvPTXCompilerGetCompiledProgramSize(compiler, &cubinSize);

  auto result = std::make_unique<std::vector<char>>(cubinSize);
  nvPTXCompilerGetCompiledProgram(compiler, result->data());
  nvPTXCompilerDestroy(&compiler);

  return result;
}

// Register pass to serialize GPU kernel functions to a CUBIN binary annotation.
void mlir::registerGpuSerializeToCubinPass() {
  PassRegistration<SerializeToCubinPass> registerSerializeToCubin([] {
    // Initialize LLVM NVPTX backend.
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();

    return std::make_unique<SerializeToCubinPass>();
  });
}

std::unique_ptr<Pass> mlir::createGpuSerializeToCubinPass(StringRef triple,
                                                          StringRef arch,
                                                          StringRef features) {
  return std::make_unique<SerializeToCubinPass>(triple, arch, features);
}

#else  // MLIR_GPU_TO_CUBIN_PASS_ENABLE
void mlir::registerGpuSerializeToCubinPass() {}
#endif // MLIR_GPU_TO_CUBIN_PASS_ENABLE
