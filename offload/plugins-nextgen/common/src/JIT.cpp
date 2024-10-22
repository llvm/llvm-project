//===- JIT.cpp - Target independent JIT infrastructure --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "JIT.h"

#include "Shared/Debug.h"
#include "Shared/Utils.h"

#include "PluginInterface.h"
#include "omptarget.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LLVMRemarkStreamer.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/IRObjectFile.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/SubtargetFeature.h"

#include <mutex>
#include <shared_mutex>
#include <system_error>

using namespace llvm;
using namespace llvm::object;
using namespace omp;
using namespace omp::target;

namespace {

bool isImageBitcode(const __tgt_device_image &Image) {
  StringRef Binary(reinterpret_cast<const char *>(Image.ImageStart),
                   utils::getPtrDiff(Image.ImageEnd, Image.ImageStart));

  return identify_magic(Binary) == file_magic::bitcode;
}

Expected<std::unique_ptr<Module>>
createModuleFromMemoryBuffer(std::unique_ptr<MemoryBuffer> &MB,
                             LLVMContext &Context) {
  SMDiagnostic Err;
  auto Mod = parseIR(*MB, Err, Context);
  if (!Mod)
    return make_error<StringError>("Failed to create module",
                                   inconvertibleErrorCode());
  return std::move(Mod);
}
Expected<std::unique_ptr<Module>>
createModuleFromImage(const __tgt_device_image &Image, LLVMContext &Context) {
  StringRef Data((const char *)Image.ImageStart,
                 utils::getPtrDiff(Image.ImageEnd, Image.ImageStart));
  std::unique_ptr<MemoryBuffer> MB = MemoryBuffer::getMemBuffer(
      Data, /*BufferName=*/"", /*RequiresNullTerminator=*/false);
  return createModuleFromMemoryBuffer(MB, Context);
}

OptimizationLevel getOptLevel(unsigned OptLevel) {
  switch (OptLevel) {
  case 0:
    return OptimizationLevel::O0;
  case 1:
    return OptimizationLevel::O1;
  case 2:
    return OptimizationLevel::O2;
  case 3:
    return OptimizationLevel::O3;
  }
  llvm_unreachable("Invalid optimization level");
}

Expected<std::unique_ptr<TargetMachine>>
createTargetMachine(Module &M, std::string CPU, unsigned OptLevel) {
  Triple TT(M.getTargetTriple());
  std::optional<CodeGenOptLevel> CGOptLevelOrNone =
      CodeGenOpt::getLevel(OptLevel);
  assert(CGOptLevelOrNone && "Invalid optimization level");
  CodeGenOptLevel CGOptLevel = *CGOptLevelOrNone;

  std::string Msg;
  const Target *T = TargetRegistry::lookupTarget(M.getTargetTriple(), Msg);
  if (!T)
    return make_error<StringError>(Msg, inconvertibleErrorCode());

  SubtargetFeatures Features;
  Features.getDefaultSubtargetFeatures(TT);

  std::optional<Reloc::Model> RelocModel;
  if (M.getModuleFlag("PIC Level"))
    RelocModel =
        M.getPICLevel() == PICLevel::NotPIC ? Reloc::Static : Reloc::PIC_;

  std::optional<CodeModel::Model> CodeModel = M.getCodeModel();

  TargetOptions Options = codegen::InitTargetOptionsFromCodeGenFlags(TT);

  std::unique_ptr<TargetMachine> TM(
      T->createTargetMachine(M.getTargetTriple(), CPU, Features.getString(),
                             Options, RelocModel, CodeModel, CGOptLevel));
  if (!TM)
    return make_error<StringError>("Failed to create target machine",
                                   inconvertibleErrorCode());
  return std::move(TM);
}

} // namespace

JITEngine::JITEngine(Triple::ArchType TA) : TT(Triple::getArchTypeName(TA)) {
  codegen::RegisterCodeGenFlags();
#ifdef LIBOMPTARGET_JIT_NVPTX
  if (TT.isNVPTX()) {
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
  }
#endif
#ifdef LIBOMPTARGET_JIT_AMDGPU
  if (TT.isAMDGPU()) {
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUAsmPrinter();
  }
#endif
}

void JITEngine::opt(TargetMachine *TM, TargetLibraryInfoImpl *TLII, Module &M,
                    unsigned OptLevel) {
  PipelineTuningOptions PTO;
  std::optional<PGOOptions> PGOOpt;

  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  ModulePassManager MPM;

  PassBuilder PB(TM, PTO, PGOOpt, nullptr);

  FAM.registerPass([&] { return TargetLibraryAnalysis(*TLII); });

  // Register all the basic analyses with the managers.
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  MPM.addPass(PB.buildPerModuleDefaultPipeline(getOptLevel(OptLevel)));
  MPM.run(M, MAM);
}

void JITEngine::codegen(TargetMachine *TM, TargetLibraryInfoImpl *TLII,
                        Module &M, raw_pwrite_stream &OS) {
  legacy::PassManager PM;
  PM.add(new TargetLibraryInfoWrapperPass(*TLII));
  MachineModuleInfoWrapperPass *MMIWP = new MachineModuleInfoWrapperPass(
      reinterpret_cast<LLVMTargetMachine *>(TM));
  TM->addPassesToEmitFile(PM, OS, nullptr,
                          TT.isNVPTX() ? CodeGenFileType::AssemblyFile
                                       : CodeGenFileType::ObjectFile,
                          /*DisableVerify=*/false, MMIWP);

  PM.run(M);
}

Expected<std::unique_ptr<MemoryBuffer>>
JITEngine::backend(Module &M, const std::string &ComputeUnitKind,
                   unsigned OptLevel) {

  auto RemarksFileOrErr = setupLLVMOptimizationRemarks(
      M.getContext(), /*RemarksFilename=*/"", /*RemarksPasses=*/"",
      /*RemarksFormat=*/"", /*RemarksWithHotness=*/false);
  if (Error E = RemarksFileOrErr.takeError())
    return std::move(E);
  if (*RemarksFileOrErr)
    (*RemarksFileOrErr)->keep();

  auto TMOrErr = createTargetMachine(M, ComputeUnitKind, OptLevel);
  if (!TMOrErr)
    return TMOrErr.takeError();

  std::unique_ptr<TargetMachine> TM = std::move(*TMOrErr);
  TargetLibraryInfoImpl TLII(TT);

  if (PreOptIRModuleFileName.isPresent()) {
    std::error_code EC;
    raw_fd_stream FD(PreOptIRModuleFileName.get(), EC);
    if (EC)
      return createStringError(
          EC, "Could not open %s to write the pre-opt IR module\n",
          PreOptIRModuleFileName.get().c_str());
    M.print(FD, nullptr);
  }

  if (!JITSkipOpt)
    opt(TM.get(), &TLII, M, OptLevel);

  if (PostOptIRModuleFileName.isPresent()) {
    std::error_code EC;
    raw_fd_stream FD(PostOptIRModuleFileName.get(), EC);
    if (EC)
      return createStringError(
          EC, "Could not open %s to write the post-opt IR module\n",
          PreOptIRModuleFileName.get().c_str());
    M.print(FD, nullptr);
  }

  // Prepare the output buffer and stream for codegen.
  SmallVector<char> CGOutputBuffer;
  raw_svector_ostream OS(CGOutputBuffer);

  codegen(TM.get(), &TLII, M, OS);

  return MemoryBuffer::getMemBufferCopy(OS.str());
}

Expected<std::unique_ptr<MemoryBuffer>>
JITEngine::getOrCreateObjFile(const __tgt_device_image &Image, LLVMContext &Ctx,
                              const std::string &ComputeUnitKind) {

  // Check if the user replaces the module at runtime with a finished object.
  if (ReplacementObjectFileName.isPresent()) {
    auto MBOrErr =
        MemoryBuffer::getFileOrSTDIN(ReplacementObjectFileName.get());
    if (!MBOrErr)
      return createStringError(MBOrErr.getError(),
                               "Could not read replacement obj from %s\n",
                               ReplacementModuleFileName.get().c_str());
    return std::move(*MBOrErr);
  }

  Module *Mod = nullptr;
  // Check if the user replaces the module at runtime or we read it from the
  // image.
  // TODO: Allow the user to specify images per device (Arch + ComputeUnitKind).
  if (!ReplacementModuleFileName.isPresent()) {
    auto ModOrErr = createModuleFromImage(Image, Ctx);
    if (!ModOrErr)
      return ModOrErr.takeError();
    Mod = ModOrErr->release();
  } else {
    auto MBOrErr =
        MemoryBuffer::getFileOrSTDIN(ReplacementModuleFileName.get());
    if (!MBOrErr)
      return createStringError(MBOrErr.getError(),
                               "Could not read replacement module from %s\n",
                               ReplacementModuleFileName.get().c_str());
    auto ModOrErr = createModuleFromMemoryBuffer(MBOrErr.get(), Ctx);
    if (!ModOrErr)
      return ModOrErr.takeError();
    Mod = ModOrErr->release();
  }

  return backend(*Mod, ComputeUnitKind, JITOptLevel);
}

Expected<const __tgt_device_image *>
JITEngine::compile(const __tgt_device_image &Image,
                   const std::string &ComputeUnitKind,
                   PostProcessingFn PostProcessing) {
  std::lock_guard<std::mutex> Lock(ComputeUnitMapMutex);

  // Check if we JITed this image for the given compute unit kind before.
  ComputeUnitInfo &CUI = ComputeUnitMap[ComputeUnitKind];
  if (__tgt_device_image *JITedImage = CUI.TgtImageMap.lookup(&Image))
    return JITedImage;

  auto ObjMBOrErr = getOrCreateObjFile(Image, CUI.Context, ComputeUnitKind);
  if (!ObjMBOrErr)
    return ObjMBOrErr.takeError();

  auto ImageMBOrErr = PostProcessing(std::move(*ObjMBOrErr));
  if (!ImageMBOrErr)
    return ImageMBOrErr.takeError();

  CUI.JITImages.push_back(std::move(*ImageMBOrErr));
  __tgt_device_image *&JITedImage = CUI.TgtImageMap[&Image];
  JITedImage = new __tgt_device_image();
  *JITedImage = Image;

  auto &ImageMB = CUI.JITImages.back();

  JITedImage->ImageStart = const_cast<char *>(ImageMB->getBufferStart());
  JITedImage->ImageEnd = const_cast<char *>(ImageMB->getBufferEnd());

  return JITedImage;
}

Expected<const __tgt_device_image *>
JITEngine::process(const __tgt_device_image &Image,
                   target::plugin::GenericDeviceTy &Device) {
  const std::string &ComputeUnitKind = Device.getComputeUnitKind();

  PostProcessingFn PostProcessing = [&Device](std::unique_ptr<MemoryBuffer> MB)
      -> Expected<std::unique_ptr<MemoryBuffer>> {
    return Device.doJITPostProcessing(std::move(MB));
  };

  if (isImageBitcode(Image))
    return compile(Image, ComputeUnitKind, PostProcessing);

  return &Image;
}
