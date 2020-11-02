//===- mlir-rocm-runner.cpp - MLIR ROCM Execution Driver-------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that executes an MLIR file on the GPU by
// translating MLIR to ROCDL/LLVM IR before JIT-compiling and executing the
// latter.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"

#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/ExecutionEngine/JitRunner.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/ROCDLIR.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"

// MC headers.
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/AsmLexer.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptionsCommandFlags.h"

// lld headers.
#include "lld/Common/Driver.h"

// HIP headers.
#include "hip/hip_version.h"

#include <mutex>

using namespace mlir;
using namespace llvm;

using Blob = SmallVector<char, 0>;

static cl::opt<std::string> tripleName("triple", cl::desc("target triple"),
                                       cl::value_desc("triple string"),
                                       cl::init("amdgcn-amd-amdhsa"));

static cl::opt<std::string> targetChip("target", cl::desc("target chip"),
                                       cl::value_desc("AMDGPU ISA version"),
                                       cl::init(""));

static cl::opt<std::string> features("feature", cl::desc("target features"),
                                     cl::value_desc("AMDGPU target features"),
                                     cl::init(""));

static constexpr const char kRunnerProgram[] = "mlir-rocm-runner";
static constexpr const char kRocmAgentEnumerator[] = "rocm_agent_enumerator";
static constexpr const char kDefaultTargetChip[] = "gfx900";

static LogicalResult assembleIsa(const std::string isa, StringRef name,
                                 Blob &result) {
  raw_svector_ostream os(result);

  std::string error;
  Triple theTriple(Triple::normalize(tripleName));
  const Target *theTarget =
      TargetRegistry::lookupTarget(theTriple.normalize(), error);
  if (!theTarget) {
    WithColor::error(errs(), name) << error;
    return failure();
  }

  SourceMgr srcMgr;
  srcMgr.AddNewSourceBuffer(MemoryBuffer::getMemBuffer(isa), SMLoc());

  const MCTargetOptions mcOptions;
  std::unique_ptr<MCRegisterInfo> mri(theTarget->createMCRegInfo(tripleName));
  std::unique_ptr<MCAsmInfo> mai(
      theTarget->createMCAsmInfo(*mri, tripleName, mcOptions));
  mai->setRelaxELFRelocations(true);

  MCObjectFileInfo mofi;
  MCContext ctx(mai.get(), mri.get(), &mofi, &srcMgr, &mcOptions);
  mofi.InitMCObjectFileInfo(theTriple, false, ctx, false);

  SmallString<128> cwd;
  if (!sys::fs::current_path(cwd))
    ctx.setCompilationDir(cwd);

  std::unique_ptr<MCStreamer> mcStreamer;
  std::unique_ptr<MCInstrInfo> mcii(theTarget->createMCInstrInfo());
  std::unique_ptr<MCSubtargetInfo> sti(
      theTarget->createMCSubtargetInfo(tripleName, targetChip, features));

  MCCodeEmitter *ce = theTarget->createMCCodeEmitter(*mcii, *mri, ctx);
  MCAsmBackend *mab = theTarget->createMCAsmBackend(*sti, *mri, mcOptions);
  mcStreamer.reset(theTarget->createMCObjectStreamer(
      theTriple, ctx, std::unique_ptr<MCAsmBackend>(mab),
      mab->createObjectWriter(os), std::unique_ptr<MCCodeEmitter>(ce), *sti,
      mcOptions.MCRelaxAll, mcOptions.MCIncrementalLinkerCompatible,
      /*DWARFMustBeAtTheEnd*/ false));
  mcStreamer->setUseAssemblerInfoForParsing(true);

  std::unique_ptr<MCAsmParser> parser(
      createMCAsmParser(srcMgr, ctx, *mcStreamer, *mai));
  std::unique_ptr<MCTargetAsmParser> tap(
      theTarget->createMCAsmParser(*sti, *parser, *mcii, mcOptions));

  if (!tap) {
    WithColor::error(errs(), name) << "assembler initialization error.\n";
    return failure();
  }

  parser->setTargetParser(*tap);
  parser->Run(false);

  return success();
}

static std::mutex mutex;
static LogicalResult createHsaco(const Blob &isaBlob, StringRef name,
                                 Blob &hsacoBlob) {
  // Save the ISA binary to a temp file.
  int tempIsaBinaryFd = -1;
  SmallString<128> tempIsaBinaryFilename;
  std::error_code ec = sys::fs::createTemporaryFile(
      "kernel", "o", tempIsaBinaryFd, tempIsaBinaryFilename);
  if (ec) {
    WithColor::error(errs(), name)
        << "temporary file for ISA binary creation error.\n";
    return failure();
  }
  FileRemover cleanupIsaBinary(tempIsaBinaryFilename);
  raw_fd_ostream tempIsaBinaryOs(tempIsaBinaryFd, true);
  tempIsaBinaryOs << isaBlob;
  tempIsaBinaryOs.close();

  // Create a temp file for HSA code object.
  int tempHsacoFD = -1;
  SmallString<128> tempHsacoFilename;
  ec = sys::fs::createTemporaryFile("kernel", "hsaco", tempHsacoFD,
                                    tempHsacoFilename);
  if (ec) {
    WithColor::error(errs(), name)
        << "temporary file for HSA code object creation error.\n";
    return failure();
  }
  FileRemover cleanupHsaco(tempHsacoFilename);

  const std::lock_guard<std::mutex> lock(mutex);
  // Invoke lld. Expect a true return value from lld.
  bool ret = lld::elf::link({"ld.lld", "-shared", tempIsaBinaryFilename.c_str(),
                             "-o", tempHsacoFilename.c_str()},
                            /*canEarlyExit=*/false, llvm::outs(), llvm::errs());
  if (!ret) {
    WithColor::error(errs(), name) << "lld invocation error.\n";
    return failure();
  }

  // Load the HSA code object.
  auto hsacoFile = mlir::openInputFile(tempHsacoFilename);
  if (!hsacoFile) {
    WithColor::error(errs(), name)
        << "read HSA code object from temp file error.\n";
    return failure();
  }
  hsacoBlob.assign(hsacoFile->getBuffer().begin(),
                   hsacoFile->getBuffer().end());

  return success();
}

static std::unique_ptr<llvm::Module>
compileModuleToROCDLIR(Operation *m, llvm::LLVMContext &llvmContext,
                       StringRef name) {
  auto llvmModule = translateModuleToROCDLIR(m, llvmContext, name);
  // TODO: Link with ROCm-Device-Libs in case needed (ex: the Module
  // depends on math functions).
  return llvmModule;
}

static OwnedBlob compileISAToHsaco(const std::string isa, Location loc,
                                   StringRef name) {
  // ISA -> ISA in binary form via MC.
  // Use lld to create HSA code object.
  Blob isaBlob;
  Blob hsacoBlob;

  if (succeeded(assembleIsa(isa, name, isaBlob)) &&
      succeeded(createHsaco(isaBlob, name, hsacoBlob)))
    return std::make_unique<std::vector<char>>(hsacoBlob.begin(),
                                               hsacoBlob.end());

  WithColor::error(errs(), name) << "producing HSA code object error.\n";
  return {};
}

static void configTargetChip() {
  // Set targetChip to default value first.
  targetChip = kDefaultTargetChip;

  // Locate rocm_agent_enumerator.
  llvm::ErrorOr<std::string> rocmAgentEnumerator = llvm::sys::findProgramByName(
      kRocmAgentEnumerator, {__ROCM_PATH__ "/bin"});
  std::error_code ec;
  if ((ec = rocmAgentEnumerator.getError())) {
    WithColor::warning(errs(), kRunnerProgram)
        << kRocmAgentEnumerator << " couldn't be located under "
        << __ROCM_PATH__ << ", set target as " << kDefaultTargetChip << "\n";
    return;
  }

  // Prepare temp file to hold the outputs.
  int tempFd = -1;
  SmallString<128> tempFilename;
  ec = sys::fs::createTemporaryFile("rocm_agent", "txt", tempFd, tempFilename);
  if (ec) {
    WithColor::warning(errs(), kRunnerProgram)
        << "temporary file for " << kRocmAgentEnumerator
        << " creation error, set target as " << kDefaultTargetChip << "\n";
    return;
  }
  FileRemover cleanup(tempFilename);

  // Invoke rocm_agent_enumerator.
  std::string errorMessage;
  SmallVector<StringRef, 2> args{"-t", "GPU"};
  Optional<StringRef> redirects[3] = {{""}, tempFilename.str(), {""}};
  int result =
      llvm::sys::ExecuteAndWait(rocmAgentEnumerator.get(), args, llvm::None,
                                redirects, 0, 0, &errorMessage);
  if (result) {
    WithColor::warning(errs(), kRunnerProgram)
        << kRocmAgentEnumerator << " invocation error: " << errorMessage
        << ", set target as " << kDefaultTargetChip << "\n";
    return;
  }

  // Load and parse the result.
  auto gfxIsaList = mlir::openInputFile(tempFilename);
  if (!gfxIsaList) {
    WithColor::error(errs(), kRunnerProgram)
        << "read ROCm agent list temp file error, set target as "
        << kDefaultTargetChip << "\n";
    return;
  }
  for (line_iterator lines(*gfxIsaList); !lines.is_at_end(); ++lines) {
    // Skip the line with content "gfx000".
    if (*lines == "gfx000")
      continue;
    // Use the first ISA version found.
    targetChip = lines->str();
    break;
  }
}

static void configTargetFeatures() {
  if (features.size() > 0)
    features += ",";
  // After ROCm 3.5, adopt HSA code object V3.
  if (HIP_VERSION_MAJOR >= 3 && HIP_VERSION_MINOR >= 5)
    features += "+code-object-v3";
  else
    features += "-code-object-v3";
}

static LogicalResult runMLIRPasses(ModuleOp m) {
  PassManager pm(m.getContext());
  applyPassManagerCLOptions(pm);

  // Configure target chip ISA version if it has not been specified.
  if (!targetChip.size())
    configTargetChip();

  // Configure target features per ROCm / HIP version.
  configTargetFeatures();

  const char gpuBinaryAnnotation[] = "rocdl.hsaco";
  pm.addPass(createLowerToCFGPass());
  pm.addPass(createGpuKernelOutliningPass());
  auto &kernelPm = pm.nest<gpu::GPUModuleOp>();
  kernelPm.addPass(createStripDebugInfoPass());
  kernelPm.addPass(createLowerGpuOpsToROCDLOpsPass());
  kernelPm.addPass(createConvertGPUKernelToBlobPass(
      compileModuleToROCDLIR, compileISAToHsaco, tripleName, targetChip,
      features, gpuBinaryAnnotation));
  pm.addPass(createGpuToLLVMConversionPass(gpuBinaryAnnotation));

  return pm.run(m);
}

int main(int argc, char **argv) {
  registerPassManagerCLOptions();
  llvm::InitLLVM y(argc, argv);
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();

  // Initialize LLVM AMDGPU backend.
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUAsmPrinter();

  mlir::initializeLLVMPasses();

  mlir::JitRunnerConfig jitRunnerConfig;
  jitRunnerConfig.mlirTransformer = runMLIRPasses;

  return mlir::JitRunnerMain(argc, argv, jitRunnerConfig);
}
