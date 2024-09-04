//===-- SPIRV.cpp - SPIR-V Backend API ------------------------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/CommandFlags.h"
// #include "llvm/CodeGen/LinkAllAsmWriterComponents.h"
// #include "llvm/CodeGen/LinkAllCodegenComponents.h"
// #include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
// #include "llvm/IR/AutoUpgrade.h"
#include "llvm/IR/DataLayout.h"
// #include "llvm/IR/DiagnosticInfo.h"
// #include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
// #include "llvm/IR/LLVMRemarkStreamer.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
// #include "llvm/IRReader/IRReader.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCTargetOptionsCommandFlags.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
// #include "llvm/Remarks/HotnessThresholdParser.h"
#include "llvm/Support/CommandLine.h"
// #include "llvm/Support/Debug.h"
// #include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/InitLLVM.h"
// #include "llvm/Support/PluginLoader.h"
// #include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
// #include "llvm/Support/TimeProfiler.h"
// #include "llvm/Support/ToolOutputFile.h"
// #include "llvm/Support/WithColor.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
// #include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include "llvm/TargetParser/Triple.h"
// #include "llvm/Transforms/Utils/Cloning.h"
// #include <memory>
#include <optional>
// #include <ostream>
#include <string>
#include <utility>

using namespace llvm;

namespace {
void parseSPIRVCommandLineOptions(const std::vector<std::string> &Options,
                                  raw_ostream *Errs) {
  static constexpr const char *Origin = "SPIRVTranslateModule";
  if (!Options.empty()) {
    std::vector<const char *> Argv(1, Origin);
    for (const auto& Arg : Options)
      Argv.push_back(Arg.c_str());
    cl::ParseCommandLineOptions(Argv.size(), Argv.data(), Origin, Errs);
  }
}

std::once_flag InitOnceFlag;
void InitializeSPIRVTarget() {
  std::call_once(InitOnceFlag, []() {
    LLVMInitializeSPIRVTargetInfo();
    LLVMInitializeSPIRVTarget();
    LLVMInitializeSPIRVTargetMC();
    LLVMInitializeSPIRVAsmPrinter();
  });
}
} // namespace

extern "C" LLVM_EXTERNAL_VISIBILITY bool
SPIRVTranslateModule(Module *M, std::string &SpirvObj, std::string &ErrMsg,
                     const std::vector<std::string> &Opts) {
  // Fallbacks for a Triple, MArch, Opt-level values.
  static const std::string DefaultTriple = "spirv64-unknown-unknown";
  static const std::string DefaultMArch = "";
  static const llvm::CodeGenOptLevel OLevel = llvm::CodeGenOptLevel::None;

  // Parse Opts as if it'd be command line argument.
  std::string Errors;
  raw_string_ostream ErrorStream(Errors);
  parseSPIRVCommandLineOptions(Opts, &ErrorStream);
  if (!Errors.empty()) {
    ErrMsg = Errors;
    return false;
  }

  // SPIR-V-specific target initialization.
  InitializeSPIRVTarget();

  Triple TargetTriple(M->getTargetTriple());
  if (TargetTriple.getTriple().empty()) {
    TargetTriple.setTriple(DefaultTriple);
    M->setTargetTriple(DefaultTriple);
  }
  const Target *TheTarget =
      TargetRegistry::lookupTarget(DefaultMArch, TargetTriple, ErrMsg);
  if (!TheTarget)
    return false;

  // A call to codegen::InitTargetOptionsFromCodeGenFlags(TargetTriple)
  // hits the following assertion: llvm/lib/CodeGen/CommandFlags.cpp:78:
  // llvm::FPOpFusion::FPOpFusionMode llvm::codegen::getFuseFPOps(): Assertion
  // `FuseFPOpsView && "RegisterCodeGenFlags not created."' failed.
  TargetOptions Options;
  std::optional<Reloc::Model> RM;
  std::optional<CodeModel::Model> CM;
  std::unique_ptr<TargetMachine> Target =
      std::unique_ptr<TargetMachine>(TheTarget->createTargetMachine(
          TargetTriple.getTriple(), "", "", Options, RM, CM, OLevel));
  if (!Target) {
    ErrMsg = "Could not allocate target machine!";
    return false;
  }

  if (M->getCodeModel())
    Target->setCodeModel(*M->getCodeModel());

  std::string DLStr = M->getDataLayoutStr();
  Expected<DataLayout> MaybeDL = DataLayout::parse(
      DLStr.empty() ? Target->createDataLayout().getStringRepresentation()
                    : DLStr);
  if (!MaybeDL) {
    ErrMsg = toString(MaybeDL.takeError());
    return false;
  }
  M->setDataLayout(MaybeDL.get());

  TargetLibraryInfoImpl TLII(Triple(M->getTargetTriple()));
  legacy::PassManager PM;
  PM.add(new TargetLibraryInfoWrapperPass(TLII));
  LLVMTargetMachine &LLVMTM = static_cast<LLVMTargetMachine &>(*Target);
  MachineModuleInfoWrapperPass *MMIWP =
      new MachineModuleInfoWrapperPass(&LLVMTM);
  const_cast<TargetLoweringObjectFile *>(LLVMTM.getObjFileLowering())
      ->Initialize(MMIWP->getMMI().getContext(), *Target);

  SmallString<4096> OutBuffer;
  raw_svector_ostream OutStream(OutBuffer);
  if (Target->addPassesToEmitFile(PM, OutStream, nullptr,
                                  CodeGenFileType::ObjectFile)) {
    ErrMsg = "Target machine cannot emit a file of this type";
    return false;
  }

  PM.run(*M);
  SpirvObj = OutBuffer.str();

  return true;
}
