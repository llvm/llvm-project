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

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Config/Targets.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#if MLIR_XEVM_OCLOC_AVAILABLE
#include <ocloc_api.h>
#endif // MLIR_XEVM_OCLOC_AVAILABLE

#include <cstdint>
#include <cstdlib>

using namespace mlir;
using namespace mlir::xevm;

namespace {
// XeVM implementation of the gpu:TargetAttrInterface.
class XeVMTargetAttrImpl
    : public gpu::TargetAttrInterface::FallbackModel<XeVMTargetAttrImpl> {
public:
  std::optional<mlir::gpu::SerializedObject>
  serializeToObject(Attribute attribute, Operation *module,
                    const gpu::TargetOptions &options) const;

  Attribute createObject(Attribute attribute, Operation *module,
                         const mlir::gpu::SerializedObject &object,
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
#if MLIR_XEVM_OCLOC_AVAILABLE
FailureOr<SmallVector<char, 0>>
SerializeGPUModuleBase::compileToBinary(StringRef asmStr,
                                        StringRef inputFormat) {
  Location loc = getGPUModuleOp().getLoc();
  std::string asmFname = llvm::formatv(
      "mlir-{0}-{1}-{2}.asm", getGPUModuleOp().getNameAttr().getValue(),
      getTarget().getTriple(), getTarget().getChip());
  // Set cmd options
  std::pair<llvm::BumpPtrAllocator, SmallVector<const char *>> cmdOpts =
      targetOptions.tokenizeCmdOptions();
  // Example: --gpu-module-to-binary="opts='opt1 opt2'"
  const std::string cmdOptsStr = "\"" + llvm::join(cmdOpts.second, " ") + "\"";
  std::vector<std::string> oclocArgs = {"ocloc",
                                        "compile",
                                        "-file",
                                        asmFname,
                                        inputFormat.str(),
                                        "-device",
                                        getTarget().getChip().str(),
                                        "-options",
                                        cmdOptsStr};

// Dump tool invocation commands.
#define DEBUG_TYPE "serialize-to-binary"
  LLVM_DEBUG({
    llvm::dbgs() << "Tool invocation for module: "
                 << getGPUModuleOp().getNameAttr() << "\n";
    llvm::interleave(oclocArgs, llvm::dbgs(), " ");
    llvm::dbgs() << "\n";
  });
#undef DEBUG_TYPE

  std::vector<const char *> argv;
  for (const auto &str : oclocArgs)
    argv.push_back(str.c_str());

  uint32_t numSources = 1;
  const uint8_t *dataSources[1] = {
      reinterpret_cast<const uint8_t *>(asmStr.data())};
  const uint64_t lenSources[1] = {asmStr.size()};
  const char *nameSources[1] = {asmFname.c_str()};

  uint32_t outputs_num = 0;
  uint8_t **outputs = nullptr;
  uint64_t *output_length = nullptr;
  char **output_names = nullptr;
  auto _ = llvm::scope_exit([&]() {
    oclocFreeOutput(&outputs_num, &outputs, &output_length, &output_names);
  });

  int err = oclocInvoke(static_cast<uint32_t>(argv.size()), argv.data(),
                        numSources, dataSources, lenSources, nameSources, 0,
                        nullptr, nullptr, nullptr, &outputs_num, &outputs,
                        &output_length, &output_names);

  if (err != OCLOC_SUCCESS) {
    emitError(loc) << "`oclocInvoke` failed or produced no output, error: "
                   << err;
    for (uint32_t i = 0; i < outputs_num; ++i) {
      if (llvm::StringRef(output_names[i]).ends_with(".log")) {
        emitError(loc) << "Compiler log:\n";
        emitError(loc) << llvm::StringRef(reinterpret_cast<char *>(outputs[i]),
                                          output_length[i])
                       << "\n";
      }
    }
    return failure();
  }

  SmallVector<char, 0> binStr;
  for (uint32_t i = 0; i < outputs_num; ++i) {
    if (llvm::StringRef(output_names[i]).ends_with(".bin")) {
      char *outBegin = reinterpret_cast<char *>(outputs[i]);
      char *outEnd = outBegin + output_length[i];
      binStr.assign(outBegin, outEnd);
      break;
    }
  }
  if (binStr.empty())
    return emitError(loc) << "`oclocInvoke` did not produce `.bin` output";

  return binStr;
}
#else  // MLIR_XEVM_OCLOC_AVAILABLE
FailureOr<SmallVector<char, 0>>
SerializeGPUModuleBase::compileToBinary(StringRef asmStr,
                                        StringRef inputFormat) {
  return getGPUModuleOp().emitError()
         << "Native binary cannot be AOT compiled without ocloc.";
}
#endif // MLIR_XEVM_OCLOC_AVAILABLE

namespace {
class SPIRVSerializer : public SerializeGPUModuleBase {
public:
  SPIRVSerializer(Operation &module, XeVMTargetAttr xeTarget,
                  const gpu::TargetOptions &targetOptions)
      : SerializeGPUModuleBase(module, xeTarget, targetOptions) {}

  static void init();

  /// Serializes the LLVM module to an object format, depending on the
  /// compilation target selected in target options.
  FailureOr<SmallVector<char, 0>>
  moduleToObject(llvm::Module &llvmModule) override;

  /// Runs the serialization pipeline, returning `std::nullopt` on error.
  std::optional<SmallVector<char, 0>> run() override;

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

#if LLVM_HAS_SPIRV_TARGET
static const std::vector<std::string> getDefaultSPIRVExtensions() {
  return {
      "SPV_EXT_relaxed_printf_string_address_space",
      "SPV_INTEL_cache_controls",
      "SPV_INTEL_variable_length_array",
  };
}

namespace llvm {
class Module;

extern "C" bool
SPIRVTranslateModule(Module *M, std::string &SpirvObj, std::string &ErrMsg,
                     const std::vector<std::string> &AllowExtNames,
                     const std::vector<std::string> &Opts);
} // namespace llvm
#endif

FailureOr<SmallVector<char, 0>>
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
  return getGPUModuleOp()->emitError(
      "The `SPIRV` target was not built. Please enable "
      "it when building LLVM.");
#else
  std::string serializedSPIRVBinary;
  std::string ErrMsg;
  std::vector<std::string> Opts;
  Opts.push_back(triple.str());
  Opts.push_back(std::to_string(optLevel));

  // Translate the LLVM module to SPIR-V binary using LLVM's SPIR-V Backend API.
  bool success =
      SPIRVTranslateModule(&llvmModule, serializedSPIRVBinary, ErrMsg,
                           getDefaultSPIRVExtensions(), Opts);

  if (!success)
    return getGPUModuleOp().emitError()
           << "Failed translating the module to Binary."
           << "Error message: " << ErrMsg;

  if (serializedSPIRVBinary.size() % 4)
    return getGPUModuleOp().emitError()
           << "SPIRV code size must be a multiple of 4.";

  StringRef spirvBin(serializedSPIRVBinary.c_str(),
                     serializedSPIRVBinary.size());

  // Return SPIRV binary if the compilation target is `assembly`. Optimization
  // and SPIR-V extensions are enabled for SPIR-V binary output in both paths
  // (assembly and binary) as of now. SPIR-V binary
  // is generated directly using the SPIR-V backends `SPIRVTranslateModule` API.
  if (targetOptions.getCompilationTarget() ==
      gpu::CompilationTarget::Assembly) {
#define DEBUG_TYPE "serialize-to-isa"
    LLVM_DEBUG({
      llvm::dbgs() << "SPIR-V for module: " << getGPUModuleOp().getNameAttr()
                   << "\n";
      llvm::dbgs() << serializedSPIRVBinary << "\n";
      llvm::dbgs().flush();
    });
#undef DEBUG_TYPE
    return SmallVector<char, 0>(spirvBin.begin(), spirvBin.end());
  }

  // Return native binary. Compile the SPIR-V binary to native binary for Intel
  // GPUs using `ocloc` compiler (Intel's OpenCL Offline Compiler).

  return compileToBinary(spirvBin, "-spirv_input");
#endif // LLVM_HAS_SPIRV_TARGET
}

std::optional<SmallVector<char, 0>> SPIRVSerializer::run() {
  // Translate the module to LLVM IR.
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule = translateToLLVMIR(llvmContext);
  if (!llvmModule) {
    getOperation().emitError() << "Failed creating the llvm::Module.";
    return std::nullopt;
  }
  setDataLayoutAndTriple(*llvmModule);

  if (initialLlvmIRCallback)
    initialLlvmIRCallback(*llvmModule);

  // Link bitcode files.
  handleModulePreLink(*llvmModule);
  {
    auto libs = loadBitcodeFiles(*llvmModule);
    if (!libs)
      return std::nullopt;
    if (!libs->empty())
      if (failed(linkFiles(*llvmModule, std::move(*libs))))
        return std::nullopt;
    handleModulePostLink(*llvmModule);
  }

  if (linkedLlvmIRCallback)
    linkedLlvmIRCallback(*llvmModule);

  // Return the serialized object.
  return moduleToObject(*llvmModule);
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

std::optional<mlir::gpu::SerializedObject>
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

    std::optional<SmallVector<char, 0>> binary = serializer.run();
    if (!binary)
      return std::nullopt;
    return gpu::SerializedObject{std::move(*binary)};
  }
  module->emitError("Unsupported XeVM target triple: ") << xeTarget.getTriple();
  return std::nullopt;
}

Attribute
XeVMTargetAttrImpl::createObject(Attribute attribute, Operation *module,
                                 const mlir::gpu::SerializedObject &object,
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
      builder.getStringAttr(
          StringRef(object.getObject().data(), object.getObject().size())),
      objectProps, /*kernels=*/nullptr);
}
