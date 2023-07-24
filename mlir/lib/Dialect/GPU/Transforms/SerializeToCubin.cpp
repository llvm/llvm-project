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
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

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
  const char *error;
  cuGetErrorString(result, &error);
  emitError(loc, expr.concat(" failed with error code ")
                     .concat(llvm::Twine{error})
                     .concat("[")
                     .concat(buffer)
                     .concat("]"));
}

static constexpr char kPtxasCompilerName[] = "ptxas";

/// Compiles the given generated PTX code with the given ptxas compiler.
static FailureOr<std::string>
compileWithPtxas(StringRef smCapability, StringRef ptxasParams,
                 StringRef ptxSource, bool dumpPtx, std::string *message) {
  // Step 0. Find ptxas compiler
  std::optional<std::string> ptxasCompiler =
      llvm::sys::Process::FindInEnvPath("PATH", kPtxasCompilerName);
  if (!ptxasCompiler.has_value())
    return failure();

  // Step 1. Create temporary files: ptx source file, log file and cubin file
  llvm::SmallString<64> ptxSourceFile, stdinFile, stdoutFile, stderrFile;
  llvm::sys::fs::createTemporaryFile("mlir-ptx", "", ptxSourceFile);
  llvm::sys::fs::createTemporaryFile("ptxas-stdin", "", stdinFile);
  llvm::sys::fs::createTemporaryFile("ptxas-stdout", "", stdoutFile);
  llvm::sys::fs::createTemporaryFile("ptxas-stderr", "", stderrFile);
  std::string cubinFile = std::string(ptxSourceFile) + ".cubin";
  llvm::FileRemover stdinRemover(stdinFile.c_str());
  llvm::FileRemover stdoutRemover(stdoutFile.c_str());
  llvm::FileRemover stderrRemover(stderrFile.c_str());
  llvm::FileRemover binRemover(cubinFile.c_str());
  llvm::FileRemover srcRemover(ptxSourceFile.c_str());

  // Step 2. Write the generated PTX into a file, so we can pass it  to ptxas
  // compiler
  std::error_code ec;
  llvm::raw_fd_ostream fPtxSource(ptxSourceFile, ec);
  fPtxSource << ptxSource;
  fPtxSource.close();
  if (fPtxSource.has_error()) {
    *message = std::string(
        "Could not write the generated ptx into a temporary file\n");
    return failure();
  }

  // Step 3. Build the ptxas command  line
  std::vector<StringRef> argVector{StringRef("ptxas"), StringRef("-arch"),
                                   smCapability,       StringRef(ptxSourceFile),
                                   StringRef("-o"),    StringRef(cubinFile)};
#ifdef _WIN32
  auto tokenize = llvm::cl::TokenizeWindowsCommandLine;
#else
  auto tokenize = llvm::cl::TokenizeGNUCommandLine;
#endif // _WIN32
  llvm::BumpPtrAllocator scratchAllocator;
  llvm::StringSaver stringSaver(scratchAllocator);
  SmallVector<const char *> rawArgs;
  tokenize(ptxasParams, stringSaver, rawArgs, /*MarkEOLs=*/false);
  for (const auto *rawArg : rawArgs)
    argVector.emplace_back(rawArg);

  std::optional<StringRef> redirects[] = {
      stdinFile.str(),
      stdoutFile.str(),
      stderrFile.str(),
  };

  // Step 4. Invoke ptxas
  if (llvm::sys::ExecuteAndWait(ptxasCompiler.value(),
                                llvm::ArrayRef<llvm::StringRef>(argVector),
                                /*Env=*/std::nullopt,
                                /*Redirects=*/redirects,
                                /*SecondsToWait=*/0,
                                /*MemoryLimit=*/0,
                                /*ErrMsg=*/message)) {
    if (message->empty()) {
      llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> maybeErrorlog =
          llvm::MemoryBuffer::getFile(stderrFile);
      *message = std::string("Invoking ptxas is failed, see the file: ");
      if (maybeErrorlog)
        *message += maybeErrorlog->get()->getBuffer().str();
    }
    stderrRemover.releaseFile();
    return failure();
  }

  // Step 5. The output of ptxas if  verbose flag is set. This is useful
  // because it shows local memory usage, register usage, and etc.
  if (dumpPtx) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> maybeFlog =
        llvm::MemoryBuffer::getFile(stderrFile);
    if (maybeFlog) {
      llvm::WithColor::note() << maybeFlog->get()->getBuffer().str();
    }
  }

  // Step 6. Read the cubin file, and return. It will eventually be written
  // into executable.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> maybeFcubin =
      llvm::MemoryBuffer::getFile(cubinFile);
  if (!maybeFcubin) {
    *message = std::string("Could not read cubin file \n");
    return failure();
  }

  return std::string(maybeFcubin->get()->getBuffer());
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
                       int optLevel = 2, bool dumpPtx = false,
                       bool usePtxas = true, StringRef ptxasParams = {});

  StringRef getArgument() const override { return "gpu-to-cubin"; }
  StringRef getDescription() const override {
    return "Lower GPU kernel function to CUBIN binary "
           "annotations";
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

llvm::once_flag SerializeToCubinPass::initializeBackendOnce;

SerializeToCubinPass::SerializeToCubinPass(StringRef triple, StringRef chip,
                                           StringRef features, int optLevel,
                                           bool dumpPtx, bool usePtxas,
                                           StringRef ptxasParams) {
  // No matter how this pass is constructed, ensure that
  // the NVPTX backend is initialized exactly once.
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
  maybeSetOption(this->ptxasParams, ptxasParams);
  this->dumpPtx = dumpPtx;
  this->usePtxas = usePtxas;
  if (this->optLevel.getNumOccurrences() == 0)
    this->optLevel.setValue(optLevel);
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

  RETURN_ON_CUDA_ERROR(cuInit(0));

  // Linking requires a device
  // context.
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

  auto kernelName = getOperation().getName().str();
  if (dumpPtx) {
    llvm::errs() << "// Kernel Name : [" << kernelName << "]\n";
    llvm::errs() << isa << "\n";
  }

  if (usePtxas) {
    // Try to compile it with ptxas first.
    std::string message;
    FailureOr<std::string> maybeCubinImage =
        compileWithPtxas(this->chip, ptxasParams, isa, dumpPtx, &message);
    if (succeeded(maybeCubinImage)) {
      return std::make_unique<std::vector<char>>(
          maybeCubinImage.value().begin(), maybeCubinImage.value().end());
    }
    emitError(loc) << message;
    return {};
  }

  // Fallback to JIT compilation if ptxas fails.
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

  // This will also destroy the cubin  data.
  RETURN_ON_CUDA_ERROR(cuLinkDestroy(linkState));
  RETURN_ON_CUDA_ERROR(cuCtxDestroy(context));

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

std::unique_ptr<Pass> mlir::createGpuSerializeToCubinPass(
    const gpu::SerializationToCubinOptions &options) {
  return std::make_unique<SerializeToCubinPass>(
      options.triple, options.chip, options.features, options.optLevel,
      options.dumpPtx, options.usePtxas, options.ptxasParams);
}

#else  // MLIR_GPU_TO_CUBIN_PASS_ENABLE
void mlir::registerGpuSerializeToCubinPass() {}
#endif // MLIR_GPU_TO_CUBIN_PASS_ENABLE
