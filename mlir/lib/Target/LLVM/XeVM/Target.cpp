//===- Target.cpp - MLIR LLVM XeVM target compilation -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This files defines XeVM target related functions including registration
// calls for the `#xevm.target` compilation attribute.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVM/XeVM/Target.h"

#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/XeVMDialect.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Target/LLVM/XeVM/Utils.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/XeVM/XeVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Target/TargetMachine.h"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Config/Targets.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>

using namespace mlir;
using namespace mlir::xevm;

namespace {
// XeVM implementation of the gpu:TargetAttrInterface.
class XeVMTargetAttrImpl
    : public gpu::TargetAttrInterface::FallbackModel<XeVMTargetAttrImpl> {
public:
  std::optional<SmallVector<char, 0>>
  serializeToObject(Attribute attribute, Operation *module,
                    const gpu::TargetOptions &options) const;

  Attribute createObject(Attribute attribute, Operation *module,
                         const SmallVector<char, 0> &object,
                         const gpu::TargetOptions &options) const;
};
} // namespace

void mlir::xevm::registerXeVMTargetInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, XeVMDialect *dialect) {
    XeVMTargetAttr::attachInterface<XeVMTargetAttrImpl>(*ctx);
  });
}

void mlir::xevm::registerXeVMTargetInterfaceExternalModels(
    MLIRContext &context) {
  DialectRegistry registry;
  registerXeVMTargetInterfaceExternalModels(registry);
  context.appendDialectRegistry(registry);
}

SerializeGPUModuleBase::SerializeGPUModuleBase(
    Operation &module, XeVMTargetAttr xeTarget,
    const gpu::TargetOptions &targetOptions)
    : ModuleToObject(module, xeTarget.getTriple(), "", {}, xeTarget.getO()),
      xeTarget(xeTarget), librariesToLink(targetOptions.getLibrariesToLink()),
      targetOptions(targetOptions) {
  if (xeTarget.getLinkFiles())
    librariesToLink.append(xeTarget.getLinkFiles().begin(),
                           xeTarget.getLinkFiles().end());
}

XeVMTargetAttr SerializeGPUModuleBase::getTarget() const { return xeTarget; }

std::optional<SmallVector<std::unique_ptr<llvm::Module>>>
SerializeGPUModuleBase::loadBitcodeFiles(llvm::Module &module) {
  if (librariesToLink.empty())
    return SmallVector<std::unique_ptr<llvm::Module>>();
  SmallVector<std::unique_ptr<llvm::Module>> bcFiles;
  if (failed(loadBitcodeFilesFromList(module.getContext(), librariesToLink,
                                      bcFiles)))
    return std::nullopt;
  return std::move(bcFiles);
}

gpu::GPUModuleOp SerializeGPUModuleBase::getGPUModuleOp() {
  return dyn_cast<gpu::GPUModuleOp>(&SerializeGPUModuleBase::getOperation());
}

// There is 1 way to finalize IL to native code: IGC
// There are 2 ways to access IGC: AOT (ocloc) and JIT (L0 runtime).
// - L0 runtime consumes IL and is external to MLIR codebase (rt wrappers).
// - `ocloc` tool can be "queried" from within MLIR.
std::optional<SmallVector<char, 0>>
SerializeGPUModuleBase::compileToBinary(const std::string &asmStr,
                                        StringRef inputFormat) {
  using TmpFile = std::pair<llvm::SmallString<128>, llvm::FileRemover>;
  // Find the `ocloc` tool.
  std::optional<std::string> oclocCompiler = findTool("ocloc");
  if (!oclocCompiler)
    return std::nullopt;
  Location loc = getGPUModuleOp().getLoc();
  std::string basename = llvm::formatv(
      "mlir-{0}-{1}-{2}", getGPUModuleOp().getNameAttr().getValue(),
      getTarget().getTriple(), getTarget().getChip());

  auto createTemp = [&](StringRef name,
                        StringRef suffix) -> std::optional<TmpFile> {
    llvm::SmallString<128> filePath;
    if (auto ec = llvm::sys::fs::createTemporaryFile(name, suffix, filePath)) {
      getGPUModuleOp().emitError()
          << "Couldn't create the temp file: `" << filePath
          << "`, error message: " << ec.message();
      return std::nullopt;
    }
    return TmpFile(filePath, llvm::FileRemover(filePath.c_str()));
  };
  // Create temp file
  std::optional<TmpFile> asmFile = createTemp(basename, "asm");
  std::optional<TmpFile> binFile = createTemp(basename, "");
  std::optional<TmpFile> logFile = createTemp(basename, "log");
  if (!logFile || !asmFile || !binFile)
    return std::nullopt;
  // Dump the assembly to a temp file
  std::error_code ec;
  {
    llvm::raw_fd_ostream asmStream(asmFile->first, ec);
    if (ec) {
      emitError(loc) << "Couldn't open the file: `" << asmFile->first
                     << "`, error message: " << ec.message();
      return std::nullopt;
    }
    asmStream << asmStr;
    if (asmStream.has_error()) {
      emitError(loc) << "An error occurred while writing the assembly to: `"
                     << asmFile->first << "`.";
      return std::nullopt;
    }
    asmStream.flush();
  }
  // Set cmd options
  std::pair<llvm::BumpPtrAllocator, SmallVector<const char *>> cmdOpts =
      targetOptions.tokenizeCmdOptions();
  // Example: --gpu-module-to-binary="opts='opt1 opt2'"
  const std::string cmdOptsStr = "\"" + llvm::join(cmdOpts.second, " ") + "\"";
  SmallVector<StringRef, 12> oclocArgs(
      {"ocloc", "compile", "-file", asmFile->first, inputFormat, "-device",
       getTarget().getChip(), "-output", binFile->first, "-output_no_suffix",
       "-options", cmdOptsStr});

// Dump tool invocation commands.
#define DEBUG_TYPE "serialize-to-binary"
  LLVM_DEBUG({
    llvm::dbgs() << "Tool invocation for module: "
                 << getGPUModuleOp().getNameAttr() << "\n";
    llvm::interleave(oclocArgs, llvm::dbgs(), " ");
    llvm::dbgs() << "\n";
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
  std::optional<StringRef> redirects[] = {
      std::nullopt,
      logFile->first,
      logFile->first,
  };
  // Invoke ocloc.
  if (llvm::sys::ExecuteAndWait(oclocCompiler.value(), oclocArgs, std::nullopt,
                                redirects, 0, 0, &message))
    return emitLogError("`ocloc`");
  binFile->first.append(".bin");
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> binaryBuffer =
      llvm::MemoryBuffer::getFile(binFile->first);
  if (!binaryBuffer) {
    emitError(loc) << "Couldn't open the file: `" << binFile->first
                   << "`, error message: " << binaryBuffer.getError().message();
    return std::nullopt;
  }
  StringRef bin = (*binaryBuffer)->getBuffer();
  return SmallVector<char, 0>(bin.begin(), bin.end());
}

std::optional<std::string> SerializeGPUModuleBase::findTool(StringRef tool) {
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

  getGPUModuleOp().emitError()
      << "Couldn't find the `" << tool
      << "` binary. Please specify the toolkit "
         "path via GpuModuleToBinaryPass or add the compiler to $PATH`.";
  return std::nullopt;
}

namespace {
class SPIRVSerializer : public SerializeGPUModuleBase {
public:
  SPIRVSerializer(Operation &module, XeVMTargetAttr xeTarget,
                  const gpu::TargetOptions &targetOptions)
      : SerializeGPUModuleBase(module, xeTarget, targetOptions) {}

  static void init();

  /// Serializes the LLVM module to an object format, depending on the
  /// compilation target selected in target options.
  std::optional<SmallVector<char, 0>>
  moduleToObject(llvm::Module &llvmModule) override;

private:
  /// Translates the LLVM module to SPIR-V binary using LLVM's
  /// SPIR-V target.
  std::optional<std::string>
  translateToSPIRVBinary(llvm::Module &llvmModule,
                         llvm::TargetMachine &targetMachine);
};
} // namespace

void SPIRVSerializer::init() {
  static llvm::once_flag initializeBackendOnce;
  llvm::call_once(initializeBackendOnce, []() {
#if LLVM_HAS_SPIRV_TARGET
    LLVMInitializeSPIRVTarget();
    LLVMInitializeSPIRVTargetInfo();
    LLVMInitializeSPIRVTargetMC();
    LLVMInitializeSPIRVAsmPrinter();
#endif
  });
}

std::optional<SmallVector<char, 0>>
SPIRVSerializer::moduleToObject(llvm::Module &llvmModule) {
#define DEBUG_TYPE "serialize-to-llvm"
  LLVM_DEBUG({
    llvm::dbgs() << "LLVM IR for module: " << getGPUModuleOp().getNameAttr()
                 << "\n";
    llvm::dbgs() << llvmModule << "\n";
    llvm::dbgs().flush();
  });
#undef DEBUG_TYPE

  // Return LLVM IR if the compilation target is `offload`.
  if (targetOptions.getCompilationTarget() == gpu::CompilationTarget::Offload)
    return SerializeGPUModuleBase::moduleToObject(llvmModule);

#if !LLVM_HAS_SPIRV_TARGET
  getGPUModuleOp()->emitError("The `SPIRV` target was not built. Please enable "
                              "it when building LLVM.");
  return std::nullopt;
#endif // LLVM_HAS_SPIRV_TARGET

  std::optional<llvm::TargetMachine *> targetMachine =
      getOrCreateTargetMachine();
  if (!targetMachine) {
    getGPUModuleOp().emitError() << "Target Machine unavailable for triple "
                                 << triple << ", can't optimize with LLVM\n";
    return std::nullopt;
  }

  // Return SPIRV if the compilation target is `assembly`.
  if (targetOptions.getCompilationTarget() ==
      gpu::CompilationTarget::Assembly) {
    std::optional<std::string> serializedISA =
        translateToISA(llvmModule, **targetMachine);
    if (!serializedISA) {
      getGPUModuleOp().emitError() << "Failed translating the module to ISA."
                                   << triple << ", can't compile with LLVM\n";
      return std::nullopt;
    }

#define DEBUG_TYPE "serialize-to-isa"
    LLVM_DEBUG({
      llvm::dbgs() << "SPIR-V for module: " << getGPUModuleOp().getNameAttr()
                   << "\n";
      llvm::dbgs() << *serializedISA << "\n";
      llvm::dbgs().flush();
    });
#undef DEBUG_TYPE

    // Make sure to include the null terminator.
    StringRef bin(serializedISA->c_str(), serializedISA->size() + 1);
    return SmallVector<char, 0>(bin.begin(), bin.end());
  }

  // Level zero runtime is set up to accept SPIR-V binary
  // translateToSPIRVBinary translates the LLVM module to SPIR-V binary
  // using LLVM's SPIRV target.
  // compileToBinary can be used in the future if level zero runtime
  // implementation switches to native XeVM binary format.
  std::optional<std::string> serializedSPIRVBinary =
      translateToSPIRVBinary(llvmModule, **targetMachine);
  if (!serializedSPIRVBinary) {
    getGPUModuleOp().emitError() << "Failed translating the module to Binary.";
    return std::nullopt;
  }
  if (serializedSPIRVBinary->size() % 4) {
    getGPUModuleOp().emitError() << "SPIRV code size must be a multiple of 4.";
    return std::nullopt;
  }
  StringRef bin(serializedSPIRVBinary->c_str(), serializedSPIRVBinary->size());
  return SmallVector<char, 0>(bin.begin(), bin.end());
}

std::optional<std::string>
SPIRVSerializer::translateToSPIRVBinary(llvm::Module &llvmModule,
                                        llvm::TargetMachine &targetMachine) {
  std::string targetISA;
  llvm::raw_string_ostream stream(targetISA);

  { // Drop pstream after this to prevent the ISA from being stuck buffering
    llvm::buffer_ostream pstream(stream);
    llvm::legacy::PassManager codegenPasses;
    if (targetMachine.addPassesToEmitFile(codegenPasses, pstream, nullptr,
                                          llvm::CodeGenFileType::ObjectFile))
      return std::nullopt;

    codegenPasses.run(llvmModule);
  }
  return targetISA;
}

std::optional<SmallVector<char, 0>>
XeVMTargetAttrImpl::serializeToObject(Attribute attribute, Operation *module,
                                      const gpu::TargetOptions &options) const {
  if (!module)
    return std::nullopt;
  auto gpuMod = dyn_cast<gpu::GPUModuleOp>(module);
  if (!gpuMod) {
    module->emitError("expected to be a gpu.module op");
    return std::nullopt;
  }
  auto xeTarget = cast<XeVMTargetAttr>(attribute);
  if (xeTarget.getTriple().starts_with("spirv")) {
    gpuMod.walk([&](LLVM::LLVMFuncOp funcOp) {
      if (funcOp->hasAttr(gpu::GPUDialect::getKernelFuncAttrName())) {
        funcOp.setIntelReqdSubGroupSize(16);
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    SPIRVSerializer serializer(*module, cast<XeVMTargetAttr>(attribute),
                               options);
    serializer.init();

#if !LLVM_HAS_SPIRV_TARGET
    module->emitError("Cannot run `TargetRegistry::lookupTarget()` for SPIRV "
                      "without having the target built.");
#endif

    return serializer.run();
  }
  module->emitError("Unsupported XeVM target triple: ") << xeTarget.getTriple();
  return std::nullopt;
}

Attribute
XeVMTargetAttrImpl::createObject(Attribute attribute, Operation *module,
                                 const SmallVector<char, 0> &object,
                                 const gpu::TargetOptions &options) const {
  Builder builder(attribute.getContext());
  gpu::CompilationTarget format = options.getCompilationTarget();
  auto xeTarget = cast<XeVMTargetAttr>(attribute);
  SmallVector<NamedAttribute, 2> properties;
  if (format == gpu::CompilationTarget::Assembly)
    properties.push_back(
        builder.getNamedAttr("O", builder.getI32IntegerAttr(xeTarget.getO())));

  DictionaryAttr objectProps;
  if (!properties.empty())
    objectProps = builder.getDictionaryAttr(properties);

  return builder.getAttr<gpu::ObjectAttr>(
      attribute, format,
      builder.getStringAttr(StringRef(object.data(), object.size())),
      objectProps, /*kernels=*/nullptr);
}
