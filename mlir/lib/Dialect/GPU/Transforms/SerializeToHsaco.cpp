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
#include "mlir/Dialect/GPU/Passes.h"

#if MLIR_GPU_TO_HSACO_PASS_ENABLE
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"

#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Target/TargetOptions.h"

#include "lld/Common/Driver.h"

#include "hip/hip_version.h"

#include <mutex>

using namespace mlir;

namespace {
class SerializeToHsacoPass
    : public PassWrapper<SerializeToHsacoPass, gpu::SerializeToBlobPass> {
public:
  SerializeToHsacoPass();

private:
  void getDependentDialects(DialectRegistry &registry) const override;

  // Serializes ROCDL to HSACO.
  std::unique_ptr<std::vector<char>>
  serializeISA(const std::string &isa) override;

  std::unique_ptr<SmallVectorImpl<char>> assembleIsa(const std::string &isa);
  std::unique_ptr<std::vector<char>>
  createHsaco(const SmallVectorImpl<char> &isaBinary);
};
} // namespace

static std::string getDefaultChip() {
  const char kDefaultChip[] = "gfx900";

  // Locate rocm_agent_enumerator.
  const char kRocmAgentEnumerator[] = "rocm_agent_enumerator";
  llvm::ErrorOr<std::string> rocmAgentEnumerator = llvm::sys::findProgramByName(
      kRocmAgentEnumerator, {__ROCM_PATH__ "/bin"});
  if (!rocmAgentEnumerator) {
    llvm::WithColor::warning(llvm::errs())
        << kRocmAgentEnumerator << "couldn't be located under " << __ROCM_PATH__
        << "/bin\n";
    return kDefaultChip;
  }

  // Prepare temp file to hold the outputs.
  int tempFd = -1;
  SmallString<128> tempFilename;
  if (llvm::sys::fs::createTemporaryFile("rocm_agent", "txt", tempFd,
                                         tempFilename)) {
    llvm::WithColor::warning(llvm::errs())
        << "temporary file for " << kRocmAgentEnumerator << " creation error\n";
    return kDefaultChip;
  }
  llvm::FileRemover cleanup(tempFilename);

  // Invoke rocm_agent_enumerator.
  std::string errorMessage;
  SmallVector<StringRef, 2> args{"-t", "GPU"};
  Optional<StringRef> redirects[3] = {{""}, tempFilename.str(), {""}};
  int result =
      llvm::sys::ExecuteAndWait(rocmAgentEnumerator.get(), args, llvm::None,
                                redirects, 0, 0, &errorMessage);
  if (result) {
    llvm::WithColor::warning(llvm::errs())
        << kRocmAgentEnumerator << " invocation error: " << errorMessage
        << "\n";
    return kDefaultChip;
  }

  // Load and parse the result.
  auto gfxIsaList = openInputFile(tempFilename);
  if (!gfxIsaList) {
    llvm::WithColor::error(llvm::errs())
        << "read ROCm agent list temp file error\n";
    return kDefaultChip;
  }
  for (llvm::line_iterator lines(*gfxIsaList); !lines.is_at_end(); ++lines) {
    // Skip the line with content "gfx000".
    if (*lines == "gfx000")
      continue;
    // Use the first ISA version found.
    return lines->str();
  }

  return kDefaultChip;
}

// Sets the 'option' to 'value' unless it already has a value.
static void maybeSetOption(Pass::Option<std::string> &option,
                           function_ref<std::string()> getValue) {
  if (!option.hasValue())
    option = getValue();
}

SerializeToHsacoPass::SerializeToHsacoPass() {
  maybeSetOption(this->triple, [] { return "amdgcn-amd-amdhsa"; });
  maybeSetOption(this->chip, [] {
    static auto chip = getDefaultChip();
    return chip;
  });
}

void SerializeToHsacoPass::getDependentDialects(
    DialectRegistry &registry) const {
  registerROCDLDialectTranslation(registry);
  gpu::SerializeToBlobPass::getDependentDialects(registry);
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
  srcMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(isa),
                            llvm::SMLoc());

  const llvm::MCTargetOptions mcOptions;
  std::unique_ptr<llvm::MCRegisterInfo> mri(
      target->createMCRegInfo(this->triple));
  std::unique_ptr<llvm::MCAsmInfo> mai(
      target->createMCAsmInfo(*mri, this->triple, mcOptions));
  mai->setRelaxELFRelocations(true);

  llvm::MCContext ctx(triple, mai.get(), mri.get(), &srcMgr, &mcOptions);
  std::unique_ptr<llvm::MCObjectFileInfo> mofi(target->createMCObjectFileInfo(
      ctx, /*PIC=*/false, /*LargeCodeModel=*/false));
  ctx.setObjectFileInfo(mofi.get());

  SmallString<128> cwd;
  if (!llvm::sys::fs::current_path(cwd))
    ctx.setCompilationDir(cwd);

  std::unique_ptr<llvm::MCStreamer> mcStreamer;
  std::unique_ptr<llvm::MCInstrInfo> mcii(target->createMCInstrInfo());
  std::unique_ptr<llvm::MCSubtargetInfo> sti(
      target->createMCSubtargetInfo(this->triple, this->chip, this->features));

  llvm::MCCodeEmitter *ce = target->createMCCodeEmitter(*mcii, *mri, ctx);
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

  {
    static std::mutex mutex;
    const std::lock_guard<std::mutex> lock(mutex);
    // Invoke lld. Expect a true return value from lld.
    if (!lld::elf::link({"ld.lld", "-shared", tempIsaBinaryFilename.c_str(),
                         "-o", tempHsacoFilename.c_str()},
                        /*canEarlyExit=*/false, llvm::outs(), llvm::errs())) {
      emitError(loc, "lld invocation error");
      return {};
    }
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
  PassRegistration<SerializeToHsacoPass> registerSerializeToHSACO(
      "gpu-to-hsaco", "Lower GPU kernel function to HSACO binary annotations",
      [] {
        // Initialize LLVM AMDGPU backend.
        LLVMInitializeAMDGPUAsmParser();
        LLVMInitializeAMDGPUAsmPrinter();
        LLVMInitializeAMDGPUTarget();
        LLVMInitializeAMDGPUTargetInfo();
        LLVMInitializeAMDGPUTargetMC();

        return std::make_unique<SerializeToHsacoPass>();
      });
}
#else  // MLIR_GPU_TO_HSACO_PASS_ENABLE
void mlir::registerGpuSerializeToHsacoPass() {}
#endif // MLIR_GPU_TO_HSACO_PASS_ENABLE
