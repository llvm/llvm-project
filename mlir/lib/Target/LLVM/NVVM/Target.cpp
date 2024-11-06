//===- Target.cpp - MLIR LLVM NVVM target compilation -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This files defines NVVM target related functions including registration
// calls for the `#nvvm.target` compilation attribute.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVM/NVVM/Target.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Target/LLVM/NVVM/Utils.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/Config/llvm-config.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

using namespace mlir;
using namespace mlir::NVVM;

#ifndef __DEFAULT_CUDATOOLKIT_PATH__
#define __DEFAULT_CUDATOOLKIT_PATH__ ""
#endif

namespace {
// Implementation of the `TargetAttrInterface` model.
class NVVMTargetAttrImpl
    : public gpu::TargetAttrInterface::FallbackModel<NVVMTargetAttrImpl> {
public:
  std::optional<SmallVector<char, 0>>
  serializeToObject(Attribute attribute, Operation *module,
                    const gpu::TargetOptions &options) const;

  Attribute createObject(Attribute attribute, Operation *module,
                         const SmallVector<char, 0> &object,
                         const gpu::TargetOptions &options) const;
};
} // namespace

// Register the NVVM dialect, the NVVM translation & the target interface.
void mlir::NVVM::registerNVVMTargetInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, NVVM::NVVMDialect *dialect) {
    NVVMTargetAttr::attachInterface<NVVMTargetAttrImpl>(*ctx);
  });
}

void mlir::NVVM::registerNVVMTargetInterfaceExternalModels(
    MLIRContext &context) {
  DialectRegistry registry;
  registerNVVMTargetInterfaceExternalModels(registry);
  context.appendDialectRegistry(registry);
}

// Search for the CUDA toolkit path.
StringRef mlir::NVVM::getCUDAToolkitPath() {
  if (const char *var = std::getenv("CUDA_ROOT"))
    return var;
  if (const char *var = std::getenv("CUDA_HOME"))
    return var;
  if (const char *var = std::getenv("CUDA_PATH"))
    return var;
  return __DEFAULT_CUDATOOLKIT_PATH__;
}

SerializeGPUModuleBase::SerializeGPUModuleBase(
    Operation &module, NVVMTargetAttr target,
    const gpu::TargetOptions &targetOptions)
    : ModuleToObject(module, target.getTriple(), target.getChip(),
                     target.getFeatures(), target.getO()),
      target(target), toolkitPath(targetOptions.getToolkitPath()),
      fileList(targetOptions.getLinkFiles()) {

  // If `targetOptions` have an empty toolkitPath use `getCUDAToolkitPath`
  if (toolkitPath.empty())
    toolkitPath = getCUDAToolkitPath();

  // Append the files in the target attribute.
  if (ArrayAttr files = target.getLink())
    for (Attribute attr : files.getValue())
      if (auto file = dyn_cast<StringAttr>(attr))
        fileList.push_back(file.str());

  // Append libdevice to the files to be loaded.
  (void)appendStandardLibs();
}

void SerializeGPUModuleBase::init() {
  static llvm::once_flag initializeBackendOnce;
  llvm::call_once(initializeBackendOnce, []() {
  // If the `NVPTX` LLVM target was built, initialize it.
#if LLVM_HAS_NVPTX_TARGET
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
#endif
  });
}

NVVMTargetAttr SerializeGPUModuleBase::getTarget() const { return target; }

StringRef SerializeGPUModuleBase::getToolkitPath() const { return toolkitPath; }

ArrayRef<std::string> SerializeGPUModuleBase::getFileList() const {
  return fileList;
}

// Try to append `libdevice` from a CUDA toolkit installation.
LogicalResult SerializeGPUModuleBase::appendStandardLibs() {
  StringRef pathRef = getToolkitPath();
  if (!pathRef.empty()) {
    SmallVector<char, 256> path;
    path.insert(path.begin(), pathRef.begin(), pathRef.end());
    pathRef = StringRef(path.data(), path.size());
    if (!llvm::sys::fs::is_directory(pathRef)) {
      getOperation().emitError() << "CUDA path: " << pathRef
                                 << " does not exist or is not a directory.\n";
      return failure();
    }
    llvm::sys::path::append(path, "nvvm", "libdevice", "libdevice.10.bc");
    pathRef = StringRef(path.data(), path.size());
    if (!llvm::sys::fs::is_regular_file(pathRef)) {
      getOperation().emitError() << "LibDevice path: " << pathRef
                                 << " does not exist or is not a file.\n";
      return failure();
    }
    fileList.push_back(pathRef.str());
  }
  return success();
}

std::optional<SmallVector<std::unique_ptr<llvm::Module>>>
SerializeGPUModuleBase::loadBitcodeFiles(llvm::Module &module) {
  SmallVector<std::unique_ptr<llvm::Module>> bcFiles;
  if (failed(loadBitcodeFilesFromList(module.getContext(), fileList, bcFiles,
                                      true)))
    return std::nullopt;
  return std::move(bcFiles);
}

namespace {
class NVPTXSerializer : public SerializeGPUModuleBase {
public:
  NVPTXSerializer(Operation &module, NVVMTargetAttr target,
                  const gpu::TargetOptions &targetOptions);

  /// Returns the GPU module op being serialized.
  gpu::GPUModuleOp getOperation();

  /// Compiles PTX to cubin using `ptxas`.
  std::optional<SmallVector<char, 0>>
  compileToBinary(const std::string &ptxCode);

  /// Compiles PTX to cubin using the `nvptxcompiler` library.
  std::optional<SmallVector<char, 0>>
  compileToBinaryNVPTX(const std::string &ptxCode);

  /// Serializes the LLVM module to an object format, depending on the
  /// compilation target selected in target options.
  std::optional<SmallVector<char, 0>>
  moduleToObject(llvm::Module &llvmModule) override;

private:
  using TmpFile = std::pair<llvm::SmallString<128>, llvm::FileRemover>;

  /// Creates a temp file.
  std::optional<TmpFile> createTemp(StringRef name, StringRef suffix);

  /// Finds the `tool` path, where `tool` is the name of the binary to search,
  /// i.e. `ptxas` or `fatbinary`. The search order is:
  /// 1. The toolkit path in `targetOptions`.
  /// 2. In the system PATH.
  /// 3. The path from `getCUDAToolkitPath()`.
  std::optional<std::string> findTool(StringRef tool);

  /// Target options.
  gpu::TargetOptions targetOptions;
};
} // namespace

NVPTXSerializer::NVPTXSerializer(Operation &module, NVVMTargetAttr target,
                                 const gpu::TargetOptions &targetOptions)
    : SerializeGPUModuleBase(module, target, targetOptions),
      targetOptions(targetOptions) {}

std::optional<NVPTXSerializer::TmpFile>
NVPTXSerializer::createTemp(StringRef name, StringRef suffix) {
  llvm::SmallString<128> filename;
  std::error_code ec =
      llvm::sys::fs::createTemporaryFile(name, suffix, filename);
  if (ec) {
    getOperation().emitError() << "Couldn't create the temp file: `" << filename
                               << "`, error message: " << ec.message();
    return std::nullopt;
  }
  return TmpFile(filename, llvm::FileRemover(filename.c_str()));
}

gpu::GPUModuleOp NVPTXSerializer::getOperation() {
  return dyn_cast<gpu::GPUModuleOp>(&SerializeGPUModuleBase::getOperation());
}

std::optional<std::string> NVPTXSerializer::findTool(StringRef tool) {
  // Find the `tool` path.
  // 1. Check the toolkit path given in the command line.
  StringRef pathRef = targetOptions.getToolkitPath();
  SmallVector<char, 256> path;
  if (!pathRef.empty()) {
    path.insert(path.begin(), pathRef.begin(), pathRef.end());
    llvm::sys::path::append(path, "bin", tool);
    if (llvm::sys::fs::can_execute(path))
      return StringRef(path.data(), path.size()).str();
  }

  // 2. Check PATH.
  if (std::optional<std::string> toolPath =
          llvm::sys::Process::FindInEnvPath("PATH", tool))
    return *toolPath;

  // 3. Check `getCUDAToolkitPath()`.
  pathRef = getCUDAToolkitPath();
  path.clear();
  if (!pathRef.empty()) {
    path.insert(path.begin(), pathRef.begin(), pathRef.end());
    llvm::sys::path::append(path, "bin", tool);
    if (llvm::sys::fs::can_execute(path))
      return StringRef(path.data(), path.size()).str();
  }
  getOperation().emitError()
      << "Couldn't find the `" << tool
      << "` binary. Please specify the toolkit "
         "path, add the compiler to $PATH, or set one of the environment "
         "variables in `NVVM::getCUDAToolkitPath()`.";
  return std::nullopt;
}

// TODO: clean this method & have a generic tool driver or never emit binaries
// with this mechanism and let another stage take care of it.
std::optional<SmallVector<char, 0>>
NVPTXSerializer::compileToBinary(const std::string &ptxCode) {
  // Determine if the serializer should create a fatbinary with the PTX embeded
  // or a simple CUBIN binary.
  const bool createFatbin =
      targetOptions.getCompilationTarget() == gpu::CompilationTarget::Fatbin;

  // Find the `ptxas` & `fatbinary` tools.
  std::optional<std::string> ptxasCompiler = findTool("ptxas");
  if (!ptxasCompiler)
    return std::nullopt;
  std::optional<std::string> fatbinaryTool;
  if (createFatbin) {
    fatbinaryTool = findTool("fatbinary");
    if (!fatbinaryTool)
      return std::nullopt;
  }
  Location loc = getOperation().getLoc();

  // Base name for all temp files: mlir-<module name>-<target triple>-<chip>.
  std::string basename =
      llvm::formatv("mlir-{0}-{1}-{2}", getOperation().getNameAttr().getValue(),
                    getTarget().getTriple(), getTarget().getChip());

  // Create temp files:
  std::optional<TmpFile> ptxFile = createTemp(basename, "ptx");
  if (!ptxFile)
    return std::nullopt;
  std::optional<TmpFile> logFile = createTemp(basename, "log");
  if (!logFile)
    return std::nullopt;
  std::optional<TmpFile> binaryFile = createTemp(basename, "bin");
  if (!binaryFile)
    return std::nullopt;
  TmpFile cubinFile;
  if (createFatbin) {
    Twine cubinFilename = ptxFile->first + ".cubin";
    cubinFile = TmpFile(cubinFilename.str(), llvm::FileRemover(cubinFilename));
  } else {
    cubinFile.first = binaryFile->first;
  }

  std::error_code ec;
  // Dump the PTX to a temp file.
  {
    llvm::raw_fd_ostream ptxStream(ptxFile->first, ec);
    if (ec) {
      emitError(loc) << "Couldn't open the file: `" << ptxFile->first
                     << "`, error message: " << ec.message();
      return std::nullopt;
    }
    ptxStream << ptxCode;
    if (ptxStream.has_error()) {
      emitError(loc) << "An error occurred while writing the PTX to: `"
                     << ptxFile->first << "`.";
      return std::nullopt;
    }
    ptxStream.flush();
  }

  // Command redirects.
  std::optional<StringRef> redirects[] = {
      std::nullopt,
      logFile->first,
      logFile->first,
  };

  // Get any extra args passed in `targetOptions`.
  std::pair<llvm::BumpPtrAllocator, SmallVector<const char *>> cmdOpts =
      targetOptions.tokenizeCmdOptions();

  // Create ptxas args.
  std::string optLevel = std::to_string(this->optLevel);
  SmallVector<StringRef, 12> ptxasArgs(
      {StringRef("ptxas"), StringRef("-arch"), getTarget().getChip(),
       StringRef(ptxFile->first), StringRef("-o"), StringRef(cubinFile.first),
       "--opt-level", optLevel});

  bool useFatbin32 = false;
  for (const auto *cArg : cmdOpts.second) {
    // All `cmdOpts` are for `ptxas` except `-32` which passes `-32` to
    // `fatbinary`, indicating a 32-bit target. By default a 64-bit target is
    // assumed.
    if (StringRef arg(cArg); arg != "-32")
      ptxasArgs.push_back(arg);
    else
      useFatbin32 = true;
  }

  // Create the `fatbinary` args.
  StringRef chip = getTarget().getChip();
  // Remove the arch prefix to obtain the compute capability.
  chip.consume_front("sm_"), chip.consume_front("compute_");
  // Embed the cubin object.
  std::string cubinArg =
      llvm::formatv("--image3=kind=elf,sm={0},file={1}", chip, cubinFile.first)
          .str();
  // Embed the PTX file so the driver can JIT if needed.
  std::string ptxArg =
      llvm::formatv("--image3=kind=ptx,sm={0},file={1}", chip, ptxFile->first)
          .str();
  SmallVector<StringRef, 6> fatbinArgs({StringRef("fatbinary"),
                                        useFatbin32 ? "-32" : "-64", cubinArg,
                                        ptxArg, "--create", binaryFile->first});

  // Dump tool invocation commands.
#define DEBUG_TYPE "serialize-to-binary"
  LLVM_DEBUG({
    llvm::dbgs() << "Tool invocation for module: "
                 << getOperation().getNameAttr() << "\n";
    llvm::interleave(ptxasArgs, llvm::dbgs(), " ");
    llvm::dbgs() << "\n";
    if (createFatbin) {
      llvm::interleave(fatbinArgs, llvm::dbgs(), " ");
      llvm::dbgs() << "\n";
    }
  });
#undef DEBUG_TYPE

  // Helper function for printing tool error logs.
  std::string message;
  auto emitLogError =
      [&](StringRef toolName) -> std::optional<SmallVector<char, 0>> {
    if (message.empty()) {
      llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> toolStderr =
          llvm::MemoryBuffer::getFile(logFile->first);
      if (toolStderr)
        emitError(loc) << toolName << " invocation failed. Log:\n"
                       << toolStderr->get()->getBuffer();
      else
        emitError(loc) << toolName << " invocation failed.";
      return std::nullopt;
    }
    emitError(loc) << toolName
                   << " invocation failed, error message: " << message;
    return std::nullopt;
  };

  // Invoke PTXAS.
  if (llvm::sys::ExecuteAndWait(ptxasCompiler.value(), ptxasArgs,
                                /*Env=*/std::nullopt,
                                /*Redirects=*/redirects,
                                /*SecondsToWait=*/0,
                                /*MemoryLimit=*/0,
                                /*ErrMsg=*/&message))
    return emitLogError("`ptxas`");
#define DEBUG_TYPE "dump-sass"
  LLVM_DEBUG({
    std::optional<std::string> nvdisasm = findTool("nvdisasm");
    SmallVector<StringRef> nvdisasmArgs(
        {StringRef("nvdisasm"), StringRef(cubinFile.first)});
    if (llvm::sys::ExecuteAndWait(nvdisasm.value(), nvdisasmArgs,
                                  /*Env=*/std::nullopt,
                                  /*Redirects=*/redirects,
                                  /*SecondsToWait=*/0,
                                  /*MemoryLimit=*/0,
                                  /*ErrMsg=*/&message))
      return emitLogError("`nvdisasm`");
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> logBuffer =
        llvm::MemoryBuffer::getFile(logFile->first);
    if (logBuffer && !(*logBuffer)->getBuffer().empty()) {
      llvm::dbgs() << "Output:\n" << (*logBuffer)->getBuffer() << "\n";
      llvm::dbgs().flush();
    }
  });
#undef DEBUG_TYPE

  // Invoke `fatbin`.
  message.clear();
  if (createFatbin && llvm::sys::ExecuteAndWait(*fatbinaryTool, fatbinArgs,
                                                /*Env=*/std::nullopt,
                                                /*Redirects=*/redirects,
                                                /*SecondsToWait=*/0,
                                                /*MemoryLimit=*/0,
                                                /*ErrMsg=*/&message))
    return emitLogError("`fatbinary`");

// Dump the output of the tools, helpful if the verbose flag was passed.
#define DEBUG_TYPE "serialize-to-binary"
  LLVM_DEBUG({
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> logBuffer =
        llvm::MemoryBuffer::getFile(logFile->first);
    if (logBuffer && !(*logBuffer)->getBuffer().empty()) {
      llvm::dbgs() << "Output:\n" << (*logBuffer)->getBuffer() << "\n";
      llvm::dbgs().flush();
    }
  });
#undef DEBUG_TYPE

  // Read the fatbin.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> binaryBuffer =
      llvm::MemoryBuffer::getFile(binaryFile->first);
  if (!binaryBuffer) {
    emitError(loc) << "Couldn't open the file: `" << binaryFile->first
                   << "`, error message: " << binaryBuffer.getError().message();
    return std::nullopt;
  }
  StringRef fatbin = (*binaryBuffer)->getBuffer();
  return SmallVector<char, 0>(fatbin.begin(), fatbin.end());
}

#if MLIR_ENABLE_NVPTXCOMPILER
#include "nvPTXCompiler.h"

#define RETURN_ON_NVPTXCOMPILER_ERROR(expr)                                    \
  do {                                                                         \
    if (auto status = (expr)) {                                                \
      emitError(loc) << llvm::Twine(#expr).concat(" failed with error code ")  \
                     << status;                                                \
      return std::nullopt;                                                     \
    }                                                                          \
  } while (false)

std::optional<SmallVector<char, 0>>
NVPTXSerializer::compileToBinaryNVPTX(const std::string &ptxCode) {
  Location loc = getOperation().getLoc();
  nvPTXCompilerHandle compiler = nullptr;
  nvPTXCompileResult status;
  size_t logSize;

  // Create the options.
  std::string optLevel = std::to_string(this->optLevel);
  std::pair<llvm::BumpPtrAllocator, SmallVector<const char *>> cmdOpts =
      targetOptions.tokenizeCmdOptions();
  cmdOpts.second.append(
      {"-arch", getTarget().getChip().data(), "--opt-level", optLevel.c_str()});

  // Create the compiler handle.
  RETURN_ON_NVPTXCOMPILER_ERROR(
      nvPTXCompilerCreate(&compiler, ptxCode.size(), ptxCode.c_str()));

  // Try to compile the binary.
  status = nvPTXCompilerCompile(compiler, cmdOpts.second.size(),
                                cmdOpts.second.data());

  // Check if compilation failed.
  if (status != NVPTXCOMPILE_SUCCESS) {
    RETURN_ON_NVPTXCOMPILER_ERROR(
        nvPTXCompilerGetErrorLogSize(compiler, &logSize));
    if (logSize != 0) {
      SmallVector<char> log(logSize + 1, 0);
      RETURN_ON_NVPTXCOMPILER_ERROR(
          nvPTXCompilerGetErrorLog(compiler, log.data()));
      emitError(loc) << "NVPTX compiler invocation failed, error log: "
                     << log.data();
    } else
      emitError(loc) << "NVPTX compiler invocation failed with error code: "
                     << status;
    return std::nullopt;
  }

  // Retrieve the binary.
  size_t elfSize;
  RETURN_ON_NVPTXCOMPILER_ERROR(
      nvPTXCompilerGetCompiledProgramSize(compiler, &elfSize));
  SmallVector<char, 0> binary(elfSize, 0);
  RETURN_ON_NVPTXCOMPILER_ERROR(
      nvPTXCompilerGetCompiledProgram(compiler, (void *)binary.data()));

// Dump the log of the compiler, helpful if the verbose flag was passed.
#define DEBUG_TYPE "serialize-to-binary"
  LLVM_DEBUG({
    RETURN_ON_NVPTXCOMPILER_ERROR(
        nvPTXCompilerGetInfoLogSize(compiler, &logSize));
    if (logSize != 0) {
      SmallVector<char> log(logSize + 1, 0);
      RETURN_ON_NVPTXCOMPILER_ERROR(
          nvPTXCompilerGetInfoLog(compiler, log.data()));
      llvm::dbgs() << "NVPTX compiler invocation for module: "
                   << getOperation().getNameAttr() << "\n";
      llvm::dbgs() << "Arguments: ";
      llvm::interleave(cmdOpts.second, llvm::dbgs(), " ");
      llvm::dbgs() << "\nOutput\n" << log.data() << "\n";
      llvm::dbgs().flush();
    }
  });
#undef DEBUG_TYPE
  RETURN_ON_NVPTXCOMPILER_ERROR(nvPTXCompilerDestroy(&compiler));
  return binary;
}
#endif // MLIR_ENABLE_NVPTXCOMPILER

std::optional<SmallVector<char, 0>>
NVPTXSerializer::moduleToObject(llvm::Module &llvmModule) {
  // Return LLVM IR if the compilation target is `offload`.
#define DEBUG_TYPE "serialize-to-llvm"
  LLVM_DEBUG({
    llvm::dbgs() << "LLVM IR for module: " << getOperation().getNameAttr()
                 << "\n";
    llvm::dbgs() << llvmModule << "\n";
    llvm::dbgs().flush();
  });
#undef DEBUG_TYPE
  if (targetOptions.getCompilationTarget() == gpu::CompilationTarget::Offload)
    return SerializeGPUModuleBase::moduleToObject(llvmModule);

#if !LLVM_HAS_NVPTX_TARGET
  getOperation()->emitError(
      "The `NVPTX` target was not built. Please enable it when building LLVM.");
  return std::nullopt;
#endif // LLVM_HAS_NVPTX_TARGET

  // Emit PTX code.
  std::optional<llvm::TargetMachine *> targetMachine =
      getOrCreateTargetMachine();
  if (!targetMachine) {
    getOperation().emitError() << "Target Machine unavailable for triple "
                               << triple << ", can't optimize with LLVM\n";
    return std::nullopt;
  }
  std::optional<std::string> serializedISA =
      translateToISA(llvmModule, **targetMachine);
  if (!serializedISA) {
    getOperation().emitError() << "Failed translating the module to ISA.";
    return std::nullopt;
  }
#define DEBUG_TYPE "serialize-to-isa"
  LLVM_DEBUG({
    llvm::dbgs() << "PTX for module: " << getOperation().getNameAttr() << "\n";
    llvm::dbgs() << *serializedISA << "\n";
    llvm::dbgs().flush();
  });
#undef DEBUG_TYPE

  // Return PTX if the compilation target is `assembly`.
  if (targetOptions.getCompilationTarget() ==
      gpu::CompilationTarget::Assembly) {
    // Make sure to include the null terminator.
    StringRef bin(serializedISA->c_str(), serializedISA->size() + 1);
    return SmallVector<char, 0>(bin.begin(), bin.end());
  }

  // Compile to binary.
#if MLIR_ENABLE_NVPTXCOMPILER
  return compileToBinaryNVPTX(*serializedISA);
#else
  return compileToBinary(*serializedISA);
#endif // MLIR_ENABLE_NVPTXCOMPILER
}

std::optional<SmallVector<char, 0>>
NVVMTargetAttrImpl::serializeToObject(Attribute attribute, Operation *module,
                                      const gpu::TargetOptions &options) const {
  assert(module && "The module must be non null.");
  if (!module)
    return std::nullopt;
  if (!mlir::isa<gpu::GPUModuleOp>(module)) {
    module->emitError("Module must be a GPU module.");
    return std::nullopt;
  }
  NVPTXSerializer serializer(*module, cast<NVVMTargetAttr>(attribute), options);
  serializer.init();
  return serializer.run();
}

Attribute
NVVMTargetAttrImpl::createObject(Attribute attribute, Operation *module,
                                 const SmallVector<char, 0> &object,
                                 const gpu::TargetOptions &options) const {
  auto target = cast<NVVMTargetAttr>(attribute);
  gpu::CompilationTarget format = options.getCompilationTarget();
  DictionaryAttr objectProps;
  Builder builder(attribute.getContext());
  if (format == gpu::CompilationTarget::Assembly)
    objectProps = builder.getDictionaryAttr(
        {builder.getNamedAttr("O", builder.getI32IntegerAttr(target.getO()))});
  return builder.getAttr<gpu::ObjectAttr>(
      attribute, format,
      builder.getStringAttr(StringRef(object.data(), object.size())),
      objectProps, /*kernels=*/nullptr);
}
