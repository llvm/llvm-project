//===- Target.cpp - MLIR LLVM ROCDL target compilation ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This files defines ROCDL target related functions including registration
// calls for the `#rocdl.target` compilation attribute.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVM/ROCDL/Target.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVM/ROCDL/Utils.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/IR/Constants.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/TargetParser.h"

#include <cstdlib>

using namespace mlir;
using namespace mlir::ROCDL;

#ifndef __DEFAULT_ROCM_PATH__
#define __DEFAULT_ROCM_PATH__ ""
#endif

namespace {
// Implementation of the `TargetAttrInterface` model.
class ROCDLTargetAttrImpl
    : public gpu::TargetAttrInterface::FallbackModel<ROCDLTargetAttrImpl> {
public:
  std::optional<SmallVector<char, 0>>
  serializeToObject(Attribute attribute, Operation *module,
                    const gpu::TargetOptions &options) const;

  Attribute createObject(Attribute attribute,
                         const SmallVector<char, 0> &object,
                         const gpu::TargetOptions &options) const;
};
} // namespace

// Register the ROCDL dialect, the ROCDL translation and the target interface.
void mlir::ROCDL::registerROCDLTargetInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, ROCDL::ROCDLDialect *dialect) {
    ROCDLTargetAttr::attachInterface<ROCDLTargetAttrImpl>(*ctx);
  });
}

void mlir::ROCDL::registerROCDLTargetInterfaceExternalModels(
    MLIRContext &context) {
  DialectRegistry registry;
  registerROCDLTargetInterfaceExternalModels(registry);
  context.appendDialectRegistry(registry);
}

// Search for the ROCM path.
StringRef mlir::ROCDL::getROCMPath() {
  if (const char *var = std::getenv("ROCM_PATH"))
    return var;
  if (const char *var = std::getenv("ROCM_ROOT"))
    return var;
  if (const char *var = std::getenv("ROCM_HOME"))
    return var;
  return __DEFAULT_ROCM_PATH__;
}

SerializeGPUModuleBase::SerializeGPUModuleBase(
    Operation &module, ROCDLTargetAttr target,
    const gpu::TargetOptions &targetOptions)
    : ModuleToObject(module, target.getTriple(), target.getChip(),
                     target.getFeatures(), target.getO()),
      target(target), toolkitPath(targetOptions.getToolkitPath()),
      fileList(targetOptions.getLinkFiles()) {

  // If `targetOptions` has an empty toolkitPath use `getROCMPath`
  if (toolkitPath.empty())
    toolkitPath = getROCMPath();

  // Append the files in the target attribute.
  if (ArrayAttr files = target.getLink())
    for (Attribute attr : files.getValue())
      if (auto file = dyn_cast<StringAttr>(attr))
        fileList.push_back(file.str());

  // Append standard ROCm device bitcode libraries to the files to be loaded.
  (void)appendStandardLibs();
}

void SerializeGPUModuleBase::init() {
  static llvm::once_flag initializeBackendOnce;
  llvm::call_once(initializeBackendOnce, []() {
  // If the `AMDGPU` LLVM target was built, initialize it.
#if MLIR_ROCM_CONVERSIONS_ENABLED == 1
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUAsmParser();
    LLVMInitializeAMDGPUAsmPrinter();
#endif
  });
}

ROCDLTargetAttr SerializeGPUModuleBase::getTarget() const { return target; }

StringRef SerializeGPUModuleBase::getToolkitPath() const { return toolkitPath; }

ArrayRef<std::string> SerializeGPUModuleBase::getFileList() const {
  return fileList;
}

LogicalResult SerializeGPUModuleBase::appendStandardLibs() {
  StringRef pathRef = getToolkitPath();
  if (!pathRef.empty()) {
    SmallVector<char, 256> path;
    path.insert(path.begin(), pathRef.begin(), pathRef.end());
    llvm::sys::path::append(path, "amdgcn", "bitcode");
    pathRef = StringRef(path.data(), path.size());
    if (!llvm::sys::fs::is_directory(pathRef)) {
      getOperation().emitRemark() << "ROCm amdgcn bitcode path: " << pathRef
                                  << " does not exist or is not a directory.";
      return failure();
    }
    StringRef isaVersion =
        llvm::AMDGPU::getArchNameAMDGCN(llvm::AMDGPU::parseArchAMDGCN(chip));
    isaVersion.consume_front("gfx");
    return getCommonBitcodeLibs(fileList, path, isaVersion);
  }
  return success();
}

std::optional<SmallVector<std::unique_ptr<llvm::Module>>>
SerializeGPUModuleBase::loadBitcodeFiles(llvm::Module &module,
                                         llvm::TargetMachine &targetMachine) {
  SmallVector<std::unique_ptr<llvm::Module>> bcFiles;
  if (failed(loadBitcodeFilesFromList(module.getContext(), targetMachine,
                                      fileList, bcFiles, true)))
    return std::nullopt;
  return std::move(bcFiles);
}

LogicalResult
SerializeGPUModuleBase::handleBitcodeFile(llvm::Module &module,
                                          llvm::TargetMachine &targetMachine) {
  // Some ROCM builds don't strip this like they should
  if (auto *openclVersion = module.getNamedMetadata("opencl.ocl.version"))
    module.eraseNamedMetadata(openclVersion);
  // Stop spamming us with clang version numbers
  if (auto *ident = module.getNamedMetadata("llvm.ident"))
    module.eraseNamedMetadata(ident);
  return success();
}

void SerializeGPUModuleBase::handleModulePreLink(
    llvm::Module &module, llvm::TargetMachine &targetMachine) {
  addControlVariables(module, target.hasWave64(), target.hasDaz(),
                      target.hasFiniteOnly(), target.hasUnsafeMath(),
                      target.hasFastMath(), target.hasCorrectSqrt(),
                      target.getAbi());
}

// Get the paths of ROCm device libraries.
LogicalResult SerializeGPUModuleBase::getCommonBitcodeLibs(
    llvm::SmallVector<std::string> &libs, SmallVector<char, 256> &libPath,
    StringRef isaVersion) {
  auto addLib = [&](StringRef path) -> bool {
    if (!llvm::sys::fs::is_regular_file(path)) {
      getOperation().emitRemark() << "Bitcode library path: " << path
                                  << " does not exist or is not a file.\n";
      return true;
    }
    libs.push_back(path.str());
    return false;
  };
  auto getLibPath = [&libPath](Twine lib) {
    auto baseSize = libPath.size();
    llvm::sys::path::append(libPath, lib + ".bc");
    std::string path(StringRef(libPath.data(), libPath.size()).str());
    libPath.truncate(baseSize);
    return path;
  };

  // Add ROCm device libraries. Fail if any of the libraries is not found.
  if (addLib(getLibPath("ocml")) || addLib(getLibPath("ockl")) ||
      addLib(getLibPath("hip")) || addLib(getLibPath("opencl")) ||
      addLib(getLibPath("oclc_isa_version_" + isaVersion)))
    return failure();
  return success();
}

void SerializeGPUModuleBase::addControlVariables(
    llvm::Module &module, bool wave64, bool daz, bool finiteOnly,
    bool unsafeMath, bool fastMath, bool correctSqrt, StringRef abiVer) {
  llvm::Type *i8Ty = llvm::Type::getInt8Ty(module.getContext());
  auto addControlVariable = [i8Ty, &module](StringRef name, bool enable) {
    llvm::GlobalVariable *controlVariable = new llvm::GlobalVariable(
        module, i8Ty, true, llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage,
        llvm::ConstantInt::get(i8Ty, enable), name, nullptr,
        llvm::GlobalValue::ThreadLocalMode::NotThreadLocal, 4);
    controlVariable->setVisibility(
        llvm::GlobalValue::VisibilityTypes::ProtectedVisibility);
    controlVariable->setAlignment(llvm::MaybeAlign(1));
    controlVariable->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Local);
  };
  addControlVariable("__oclc_finite_only_opt", finiteOnly || fastMath);
  addControlVariable("__oclc_unsafe_math_opt", unsafeMath || fastMath);
  addControlVariable("__oclc_daz_opt", daz || fastMath);
  addControlVariable("__oclc_correctly_rounded_sqrt32",
                     correctSqrt && !fastMath);
  addControlVariable("__oclc_wavefrontsize64", wave64);

  llvm::Type *i32Ty = llvm::Type::getInt32Ty(module.getContext());
  int abi = 400;
  abiVer.getAsInteger(0, abi);
  llvm::GlobalVariable *abiVersion = new llvm::GlobalVariable(
      module, i32Ty, true, llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage,
      llvm::ConstantInt::get(i32Ty, abi), "__oclc_ABI_version", nullptr,
      llvm::GlobalValue::ThreadLocalMode::NotThreadLocal, 4);
  abiVersion->setVisibility(
      llvm::GlobalValue::VisibilityTypes::ProtectedVisibility);
  abiVersion->setAlignment(llvm::MaybeAlign(4));
  abiVersion->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Local);
}

std::optional<SmallVector<char, 0>>
SerializeGPUModuleBase::assembleIsa(StringRef isa) {
  auto loc = getOperation().getLoc();

  StringRef targetTriple = this->triple;

  SmallVector<char, 0> result;
  llvm::raw_svector_ostream os(result);

  llvm::Triple triple(llvm::Triple::normalize(targetTriple));
  std::string error;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple.normalize(), error);
  if (!target) {
    emitError(loc, Twine("failed to lookup target: ") + error);
    return std::nullopt;
  }

  llvm::SourceMgr srcMgr;
  srcMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(isa), SMLoc());

  const llvm::MCTargetOptions mcOptions;
  std::unique_ptr<llvm::MCRegisterInfo> mri(
      target->createMCRegInfo(targetTriple));
  std::unique_ptr<llvm::MCAsmInfo> mai(
      target->createMCAsmInfo(*mri, targetTriple, mcOptions));
  mai->setRelaxELFRelocations(true);
  std::unique_ptr<llvm::MCSubtargetInfo> sti(
      target->createMCSubtargetInfo(targetTriple, chip, features));

  llvm::MCContext ctx(triple, mai.get(), mri.get(), sti.get(), &srcMgr,
                      &mcOptions);
  std::unique_ptr<llvm::MCObjectFileInfo> mofi(target->createMCObjectFileInfo(
      ctx, /*PIC=*/false, /*LargeCodeModel=*/false));
  ctx.setObjectFileInfo(mofi.get());

  SmallString<128> cwd;
  if (!llvm::sys::fs::current_path(cwd))
    ctx.setCompilationDir(cwd);

  std::unique_ptr<llvm::MCStreamer> mcStreamer;
  std::unique_ptr<llvm::MCInstrInfo> mcii(target->createMCInstrInfo());

  llvm::MCCodeEmitter *ce = target->createMCCodeEmitter(*mcii, ctx);
  llvm::MCAsmBackend *mab = target->createMCAsmBackend(*sti, *mri, mcOptions);
  mcStreamer.reset(target->createMCObjectStreamer(
      triple, ctx, std::unique_ptr<llvm::MCAsmBackend>(mab),
      mab->createObjectWriter(os), std::unique_ptr<llvm::MCCodeEmitter>(ce),
      *sti, mcOptions.MCRelaxAll, mcOptions.MCIncrementalLinkerCompatible,
      /*DWARFMustBeAtTheEnd*/ false));
  mcStreamer->setUseAssemblerInfoForParsing(true);

  std::unique_ptr<llvm::MCAsmParser> parser(
      createMCAsmParser(srcMgr, ctx, *mcStreamer, *mai));
  std::unique_ptr<llvm::MCTargetAsmParser> tap(
      target->createMCAsmParser(*sti, *parser, *mcii, mcOptions));

  if (!tap) {
    emitError(loc, "assembler initialization error");
    return {};
  }

  parser->setTargetParser(*tap);
  parser->Run(false);

  return result;
}

#if MLIR_ROCM_CONVERSIONS_ENABLED == 1
namespace {
class AMDGPUSerializer : public SerializeGPUModuleBase {
public:
  AMDGPUSerializer(Operation &module, ROCDLTargetAttr target,
                   const gpu::TargetOptions &targetOptions);

  gpu::GPUModuleOp getOperation();

  // Compile to HSA.
  std::optional<SmallVector<char, 0>>
  compileToBinary(const std::string &serializedISA);

  std::optional<SmallVector<char, 0>>
  moduleToObject(llvm::Module &llvmModule,
                 llvm::TargetMachine &targetMachine) override;

private:
  // Target options.
  gpu::TargetOptions targetOptions;
};
} // namespace

AMDGPUSerializer::AMDGPUSerializer(Operation &module, ROCDLTargetAttr target,
                                   const gpu::TargetOptions &targetOptions)
    : SerializeGPUModuleBase(module, target, targetOptions),
      targetOptions(targetOptions) {}

gpu::GPUModuleOp AMDGPUSerializer::getOperation() {
  return dyn_cast<gpu::GPUModuleOp>(&SerializeGPUModuleBase::getOperation());
}

std::optional<SmallVector<char, 0>>
AMDGPUSerializer::compileToBinary(const std::string &serializedISA) {
  // Assemble the ISA.
  std::optional<SmallVector<char, 0>> isaBinary = assembleIsa(serializedISA);

  if (!isaBinary) {
    getOperation().emitError() << "Failed during ISA assembling.";
    return std::nullopt;
  }

  // Save the ISA binary to a temp file.
  int tempIsaBinaryFd = -1;
  SmallString<128> tempIsaBinaryFilename;
  if (llvm::sys::fs::createTemporaryFile("kernel%%", "o", tempIsaBinaryFd,
                                         tempIsaBinaryFilename)) {
    getOperation().emitError()
        << "Failed to create a temporary file for dumping the ISA binary.";
    return std::nullopt;
  }
  llvm::FileRemover cleanupIsaBinary(tempIsaBinaryFilename);
  {
    llvm::raw_fd_ostream tempIsaBinaryOs(tempIsaBinaryFd, true);
    tempIsaBinaryOs << StringRef(isaBinary->data(), isaBinary->size());
    tempIsaBinaryOs.flush();
  }

  // Create a temp file for HSA code object.
  int tempHsacoFD = -1;
  SmallString<128> tempHsacoFilename;
  if (llvm::sys::fs::createTemporaryFile("kernel", "hsaco", tempHsacoFD,
                                         tempHsacoFilename)) {
    getOperation().emitError()
        << "Failed to create a temporary file for the HSA code object.";
    return std::nullopt;
  }
  llvm::FileRemover cleanupHsaco(tempHsacoFilename);

  llvm::SmallString<128> lldPath(toolkitPath);
  llvm::sys::path::append(lldPath, "llvm", "bin", "ld.lld");
  int lldResult = llvm::sys::ExecuteAndWait(
      lldPath,
      {"ld.lld", "-shared", tempIsaBinaryFilename, "-o", tempHsacoFilename});
  if (lldResult != 0) {
    getOperation().emitError() << "lld invocation failed.";
    return std::nullopt;
  }

  // Load the HSA code object.
  auto hsacoFile = openInputFile(tempHsacoFilename);
  if (!hsacoFile) {
    getOperation().emitError()
        << "Failed to read the HSA code object from the temp file.";
    return std::nullopt;
  }

  StringRef buffer = hsacoFile->getBuffer();

  return SmallVector<char, 0>(buffer.begin(), buffer.end());
}

std::optional<SmallVector<char, 0>>
AMDGPUSerializer::moduleToObject(llvm::Module &llvmModule,
                                 llvm::TargetMachine &targetMachine) {
  // Return LLVM IR if the compilation target is offload.
#define DEBUG_TYPE "serialize-to-llvm"
  LLVM_DEBUG({
    llvm::dbgs() << "LLVM IR for module: " << getOperation().getNameAttr()
                 << "\n"
                 << llvmModule << "\n";
  });
#undef DEBUG_TYPE
  if (targetOptions.getCompilationTarget() == gpu::CompilationTarget::Offload)
    return SerializeGPUModuleBase::moduleToObject(llvmModule, targetMachine);

  // Translate the Module to ISA.
  std::optional<std::string> serializedISA =
      translateToISA(llvmModule, targetMachine);
  if (!serializedISA) {
    getOperation().emitError() << "Failed translating the module to ISA.";
    return std::nullopt;
  }
#define DEBUG_TYPE "serialize-to-isa"
  LLVM_DEBUG({
    llvm::dbgs() << "ISA for module: " << getOperation().getNameAttr() << "\n"
                 << *serializedISA << "\n";
  });
#undef DEBUG_TYPE
  // Return ISA assembly code if the compilation target is assembly.
  if (targetOptions.getCompilationTarget() == gpu::CompilationTarget::Assembly)
    return SmallVector<char, 0>(serializedISA->begin(), serializedISA->end());

  // Compile to binary.
  return compileToBinary(*serializedISA);
}
#endif // MLIR_ROCM_CONVERSIONS_ENABLED

std::optional<SmallVector<char, 0>> ROCDLTargetAttrImpl::serializeToObject(
    Attribute attribute, Operation *module,
    const gpu::TargetOptions &options) const {
  assert(module && "The module must be non null.");
  if (!module)
    return std::nullopt;
  if (!mlir::isa<gpu::GPUModuleOp>(module)) {
    module->emitError("Module must be a GPU module.");
    return std::nullopt;
  }
#if MLIR_ROCM_CONVERSIONS_ENABLED == 1
  AMDGPUSerializer serializer(*module, cast<ROCDLTargetAttr>(attribute),
                              options);
  serializer.init();
  return serializer.run();
#else
  module->emitError("The `AMDGPU` target was not built. Please enable it when "
                    "building LLVM.");
  return std::nullopt;
#endif // MLIR_ROCM_CONVERSIONS_ENABLED == 1
}

Attribute
ROCDLTargetAttrImpl::createObject(Attribute attribute,
                                  const SmallVector<char, 0> &object,
                                  const gpu::TargetOptions &options) const {
  gpu::CompilationTarget format = options.getCompilationTarget();
  Builder builder(attribute.getContext());
  return builder.getAttr<gpu::ObjectAttr>(
      attribute,
      format > gpu::CompilationTarget::Binary ? gpu::CompilationTarget::Binary
                                              : format,
      builder.getStringAttr(StringRef(object.data(), object.size())), nullptr);
}
