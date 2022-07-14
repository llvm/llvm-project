//===- LowerGPUToHSACO.cpp - Convert GPU kernel to HSACO blob -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that serializes a gpu module into HSAco blob and
// adds that blob as a string attribute of the module.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"

#if MLIR_GPU_TO_HSACO_PASS_ENABLE
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"

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

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/WithColor.h"

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

#include "llvm/Transforms/IPO/Internalize.h"

#include <mutex>

using namespace mlir;

namespace {
class SerializeToHsacoPass
    : public PassWrapper<SerializeToHsacoPass, gpu::SerializeToBlobPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SerializeToHsacoPass)

  SerializeToHsacoPass(StringRef triple, StringRef arch, StringRef features,
                       int optLevel);
  SerializeToHsacoPass(const SerializeToHsacoPass &other);
  StringRef getArgument() const override { return "gpu-to-hsaco"; }
  StringRef getDescription() const override {
    return "Lower GPU kernel function to HSACO binary annotations";
  }

protected:
  Option<int> optLevel{
      *this, "opt-level",
      llvm::cl::desc("Optimization level for HSACO compilation"),
      llvm::cl::init(2)};

  Option<std::string> rocmPath{*this, "rocm-path",
                               llvm::cl::desc("Path to ROCm install")};

  // Overload to allow linking in device libs
  std::unique_ptr<llvm::Module>
  translateToLLVMIR(llvm::LLVMContext &llvmContext) override;

  /// Adds LLVM optimization passes
  LogicalResult optimizeLlvm(llvm::Module &llvmModule,
                             llvm::TargetMachine &targetMachine) override;

private:
  void getDependentDialects(DialectRegistry &registry) const override;

  // Loads LLVM bitcode libraries
  Optional<SmallVector<std::unique_ptr<llvm::Module>, 3>>
  loadLibraries(SmallVectorImpl<char> &path,
                SmallVectorImpl<StringRef> &libraries,
                llvm::LLVMContext &context);

  // Serializes ROCDL to HSACO.
  std::unique_ptr<std::vector<char>>
  serializeISA(const std::string &isa) override;

  std::unique_ptr<SmallVectorImpl<char>> assembleIsa(const std::string &isa);
  std::unique_ptr<std::vector<char>>
  createHsaco(const SmallVectorImpl<char> &isaBinary);

  std::string getRocmPath();
};
} // namespace

SerializeToHsacoPass::SerializeToHsacoPass(const SerializeToHsacoPass &other)
    : PassWrapper<SerializeToHsacoPass, gpu::SerializeToBlobPass>(other) {}

/// Get a user-specified path to ROCm
// Tries, in order, the --rocm-path option, the ROCM_PATH environment variable
// and a compile-time default
std::string SerializeToHsacoPass::getRocmPath() {
  if (rocmPath.getNumOccurrences() > 0)
    return rocmPath.getValue();

  return __DEFAULT_ROCM_PATH__;
}

// Sets the 'option' to 'value' unless it already has a value.
static void maybeSetOption(Pass::Option<std::string> &option,
                           function_ref<std::string()> getValue) {
  if (!option.hasValue())
    option = getValue();
}

SerializeToHsacoPass::SerializeToHsacoPass(StringRef triple, StringRef arch,
                                           StringRef features, int optLevel) {
  maybeSetOption(this->triple, [&triple] { return triple.str(); });
  maybeSetOption(this->chip, [&arch] { return arch.str(); });
  maybeSetOption(this->features, [&features] { return features.str(); });
  if (this->optLevel.getNumOccurrences() == 0)
    this->optLevel.setValue(optLevel);
}

void SerializeToHsacoPass::getDependentDialects(
    DialectRegistry &registry) const {
  registerROCDLDialectTranslation(registry);
  gpu::SerializeToBlobPass::getDependentDialects(registry);
}

Optional<SmallVector<std::unique_ptr<llvm::Module>, 3>>
SerializeToHsacoPass::loadLibraries(SmallVectorImpl<char> &path,
                                    SmallVectorImpl<StringRef> &libraries,
                                    llvm::LLVMContext &context) {
  SmallVector<std::unique_ptr<llvm::Module>, 3> ret;
  size_t dirLength = path.size();

  if (!llvm::sys::fs::is_directory(path)) {
    getOperation().emitRemark() << "Bitcode path: " << path
                                << " does not exist or is not a directory\n";
    return llvm::None;
  }

  for (const StringRef file : libraries) {
    llvm::SMDiagnostic error;
    llvm::sys::path::append(path, file);
    llvm::StringRef pathRef(path.data(), path.size());
    std::unique_ptr<llvm::Module> library =
        llvm::getLazyIRFileModule(pathRef, error, context);
    path.truncate(dirLength);
    if (!library) {
      getOperation().emitError() << "Failed to load library " << file
                                 << " from " << path << error.getMessage();
      return llvm::None;
    }
    // Some ROCM builds don't strip this like they should
    if (auto *openclVersion = library->getNamedMetadata("opencl.ocl.version"))
      library->eraseNamedMetadata(openclVersion);
    // Stop spamming us with clang version numbers
    if (auto *ident = library->getNamedMetadata("llvm.ident"))
      library->eraseNamedMetadata(ident);
    ret.push_back(std::move(library));
  }

  return ret;
}

std::unique_ptr<llvm::Module>
SerializeToHsacoPass::translateToLLVMIR(llvm::LLVMContext &llvmContext) {
  // MLIR -> LLVM translation
  std::unique_ptr<llvm::Module> ret =
      gpu::SerializeToBlobPass::translateToLLVMIR(llvmContext);

  if (!ret) {
    getOperation().emitOpError("Module lowering failed");
    return ret;
  }
  // Walk the LLVM module in order to determine if we need to link in device
  // libs
  bool needOpenCl = false;
  bool needOckl = false;
  bool needOcml = false;
  for (llvm::Function &f : ret->functions()) {
    if (f.hasExternalLinkage() && f.hasName() && !f.hasExactDefinition()) {
      StringRef funcName = f.getName();
      if ("printf" == funcName)
        needOpenCl = true;
      if (funcName.startswith("__ockl_"))
        needOckl = true;
      if (funcName.startswith("__ocml_"))
        needOcml = true;
    }
  }

  if (needOpenCl)
    needOcml = needOckl = true;

  // No libraries needed (the typical case)
  if (!(needOpenCl || needOcml || needOckl))
    return ret;

  // Define one of the control constants the ROCm device libraries expect to be
  // present These constants can either be defined in the module or can be
  // imported by linking in bitcode that defines the constant. To simplify our
  // logic, we define the constants into the module we are compiling
  auto addControlConstant = [&module = *ret](StringRef name, uint32_t value,
                                             uint32_t bitwidth) {
    using llvm::GlobalVariable;
    if (module.getNamedGlobal(name)) {
      return;
    }
    llvm::IntegerType *type =
        llvm::IntegerType::getIntNTy(module.getContext(), bitwidth);
    auto *initializer = llvm::ConstantInt::get(type, value, /*isSigned=*/false);
    auto *constant = new GlobalVariable(
        module, type,
        /*isConstant=*/true, GlobalVariable::LinkageTypes::LinkOnceODRLinkage,
        initializer, name,
        /*before=*/nullptr,
        /*threadLocalMode=*/GlobalVariable::ThreadLocalMode::NotThreadLocal,
        /*addressSpace=*/4);
    constant->setUnnamedAddr(GlobalVariable::UnnamedAddr::Local);
    constant->setVisibility(
        GlobalVariable::VisibilityTypes::ProtectedVisibility);
    constant->setAlignment(llvm::MaybeAlign(bitwidth / 8));
  };

  // Set up control variables in the module instead of linking in tiny bitcode
  if (needOcml) {
    // TODO(kdrewnia): Enable math optimizations once we have support for
    // `-ffast-math`-like options
    addControlConstant("__oclc_finite_only_opt", 0, 8);
    addControlConstant("__oclc_daz_opt", 0, 8);
    addControlConstant("__oclc_correctly_rounded_sqrt32", 1, 8);
    addControlConstant("__oclc_unsafe_math_opt", 0, 8);
  }
  if (needOcml || needOckl) {
    addControlConstant("__oclc_wavefrontsize64", 1, 8);
    StringRef chipSet = this->chip.getValue();
    if (chipSet.startswith("gfx"))
      chipSet = chipSet.substr(3);
    uint32_t minor =
        llvm::APInt(32, chipSet.substr(chipSet.size() - 2), 16).getZExtValue();
    uint32_t major = llvm::APInt(32, chipSet.substr(0, chipSet.size() - 2), 10)
                         .getZExtValue();
    uint32_t isaNumber = minor + 1000 * major;
    addControlConstant("__oclc_ISA_version", isaNumber, 32);

    // This constant must always match the default code object ABI version
    // of the AMDGPU backend.
    addControlConstant("__oclc_ABI_version", 400, 32);
  }

  // Determine libraries we need to link - order matters due to dependencies
  llvm::SmallVector<StringRef, 4> libraries;
  if (needOpenCl)
    libraries.push_back("opencl.bc");
  if (needOcml)
    libraries.push_back("ocml.bc");
  if (needOckl)
    libraries.push_back("ockl.bc");

  Optional<SmallVector<std::unique_ptr<llvm::Module>, 3>> mbModules;
  std::string theRocmPath = getRocmPath();
  llvm::SmallString<32> bitcodePath(theRocmPath);
  llvm::sys::path::append(bitcodePath, "amdgcn", "bitcode");
  mbModules = loadLibraries(bitcodePath, libraries, llvmContext);

  if (!mbModules) {
    getOperation()
            .emitWarning("Could not load required device libraries")
            .attachNote()
        << "This will probably cause link-time or run-time failures";
    return ret; // We can still abort here
  }

  llvm::Linker linker(*ret);
  for (std::unique_ptr<llvm::Module> &libModule : *mbModules) {
    // This bitcode linking code is substantially similar to what is used in
    // hip-clang It imports the library functions into the module, allowing LLVM
    // optimization passes (which must run after linking) to optimize across the
    // libraries and the module's code. We also only import symbols if they are
    // referenced by the module or a previous library since there will be no
    // other source of references to those symbols in this compilation and since
    // we don't want to bloat the resulting code object.
    bool err = linker.linkInModule(
        std::move(libModule), llvm::Linker::Flags::LinkOnlyNeeded,
        [](llvm::Module &m, const StringSet<> &gvs) {
          llvm::internalizeModule(m, [&gvs](const llvm::GlobalValue &gv) {
            return !gv.hasName() || (gvs.count(gv.getName()) == 0);
          });
        });
    // True is linker failure
    if (err) {
      getOperation().emitError(
          "Unrecoverable failure during device library linking.");
      // We have no guaranties about the state of `ret`, so bail
      return nullptr;
    }
  }

  return ret;
}

LogicalResult
SerializeToHsacoPass::optimizeLlvm(llvm::Module &llvmModule,
                                   llvm::TargetMachine &targetMachine) {
  int optLevel = this->optLevel.getValue();
  if (optLevel < 0 || optLevel > 3)
    return getOperation().emitError()
           << "Invalid HSA optimization level" << optLevel << "\n";

  targetMachine.setOptLevel(static_cast<llvm::CodeGenOpt::Level>(optLevel));

  auto transformer =
      makeOptimizingTransformer(optLevel, /*sizeLevel=*/0, &targetMachine);
  auto error = transformer(&llvmModule);
  if (error) {
    InFlightDiagnostic mlirError = getOperation()->emitError();
    llvm::handleAllErrors(
        std::move(error), [&mlirError](const llvm::ErrorInfoBase &ei) {
          mlirError << "Could not optimize LLVM IR: " << ei.message() << "\n";
        });
    return mlirError;
  }
  return success();
}

std::unique_ptr<SmallVectorImpl<char>>
SerializeToHsacoPass::assembleIsa(const std::string &isa) {
  auto loc = getOperation().getLoc();

  SmallVector<char, 0> result;
  llvm::raw_svector_ostream os(result);

  llvm::Triple triple(llvm::Triple::normalize(this->triple));
  std::string error;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple.normalize(), error);
  if (!target) {
    emitError(loc, Twine("failed to lookup target: ") + error);
    return {};
  }

  llvm::SourceMgr srcMgr;
  srcMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(isa), SMLoc());

  const llvm::MCTargetOptions mcOptions;
  std::unique_ptr<llvm::MCRegisterInfo> mri(
      target->createMCRegInfo(this->triple));
  std::unique_ptr<llvm::MCAsmInfo> mai(
      target->createMCAsmInfo(*mri, this->triple, mcOptions));
  mai->setRelaxELFRelocations(true);
  std::unique_ptr<llvm::MCSubtargetInfo> sti(
      target->createMCSubtargetInfo(this->triple, this->chip, this->features));

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

  return std::make_unique<SmallVector<char, 0>>(std::move(result));
}

std::unique_ptr<std::vector<char>>
SerializeToHsacoPass::createHsaco(const SmallVectorImpl<char> &isaBinary) {
  auto loc = getOperation().getLoc();

  // Save the ISA binary to a temp file.
  int tempIsaBinaryFd = -1;
  SmallString<128> tempIsaBinaryFilename;
  if (llvm::sys::fs::createTemporaryFile("kernel", "o", tempIsaBinaryFd,
                                         tempIsaBinaryFilename)) {
    emitError(loc, "temporary file for ISA binary creation error");
    return {};
  }
  llvm::FileRemover cleanupIsaBinary(tempIsaBinaryFilename);
  llvm::raw_fd_ostream tempIsaBinaryOs(tempIsaBinaryFd, true);
  tempIsaBinaryOs << StringRef(isaBinary.data(), isaBinary.size());
  tempIsaBinaryOs.close();

  // Create a temp file for HSA code object.
  int tempHsacoFD = -1;
  SmallString<128> tempHsacoFilename;
  if (llvm::sys::fs::createTemporaryFile("kernel", "hsaco", tempHsacoFD,
                                         tempHsacoFilename)) {
    emitError(loc, "temporary file for HSA code object creation error");
    return {};
  }
  llvm::FileRemover cleanupHsaco(tempHsacoFilename);

  std::string theRocmPath = getRocmPath();
  llvm::SmallString<32> lldPath(theRocmPath);
  llvm::sys::path::append(lldPath, "llvm", "bin", "ld.lld");
  int lldResult = llvm::sys::ExecuteAndWait(
      lldPath,
      {"ld.lld", "-shared", tempIsaBinaryFilename, "-o", tempHsacoFilename});
  if (lldResult != 0) {
    emitError(loc, "lld invocation error");
    return {};
  }

  // Load the HSA code object.
  auto hsacoFile = openInputFile(tempHsacoFilename);
  if (!hsacoFile) {
    emitError(loc, "read HSA code object from temp file error");
    return {};
  }

  StringRef buffer = hsacoFile->getBuffer();
  return std::make_unique<std::vector<char>>(buffer.begin(), buffer.end());
}

std::unique_ptr<std::vector<char>>
SerializeToHsacoPass::serializeISA(const std::string &isa) {
  auto isaBinary = assembleIsa(isa);
  if (!isaBinary)
    return {};
  return createHsaco(*isaBinary);
}

// Register pass to serialize GPU kernel functions to a HSACO binary annotation.
void mlir::registerGpuSerializeToHsacoPass() {
  PassRegistration<SerializeToHsacoPass> registerSerializeToHSACO([] {
    // Initialize LLVM AMDGPU backend.
    LLVMInitializeAMDGPUAsmParser();
    LLVMInitializeAMDGPUAsmPrinter();
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();

    return std::make_unique<SerializeToHsacoPass>("amdgcn-amd-amdhsa", "", "",
                                                  2);
  });
}

/// Create an instance of the GPU kernel function to HSAco binary serialization
/// pass.
std::unique_ptr<Pass> mlir::createGpuSerializeToHsacoPass(StringRef triple,
                                                          StringRef arch,
                                                          StringRef features,
                                                          int optLevel) {
  return std::make_unique<SerializeToHsacoPass>(triple, arch, features,
                                                optLevel);
}

#else  // MLIR_GPU_TO_HSACO_PASS_ENABLE
void mlir::registerGpuSerializeToHsacoPass() {}
#endif // MLIR_GPU_TO_HSACO_PASS_ENABLE
