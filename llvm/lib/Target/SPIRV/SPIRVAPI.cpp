//===-- SPIRVAPI.cpp - SPIR-V Backend API ---------------------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SPIRVCommandLine.h"
#include "SPIRVSubtarget.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCTargetOptionsCommandFlags.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include "llvm/TargetParser/Triple.h"
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;

namespace {

// Mimic limited number of command line flags from llc to provide a better
// user experience when passing options into the translate API call.
static cl::opt<char> SpvOptLevel(" O", cl::Hidden, cl::Prefix, cl::init('0'));
static cl::opt<std::string> SpvTargetTriple(" mtriple", cl::Hidden,
                                            cl::init(""));

// Utility to accept options in a command line style.
void parseSPIRVCommandLineOptions(const std::vector<std::string> &Options,
                                  raw_ostream *Errs) {
  static constexpr const char *Origin = "SPIRVTranslateModule";
  if (!Options.empty()) {
    std::vector<const char *> Argv(1, Origin);
    for (const auto &Arg : Options)
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

namespace llvm {

// The goal of this function is to facilitate integration of SPIRV Backend into
// tools and libraries by means of exposing an API call that translate LLVM
// module to SPIR-V and write results into a string as binary SPIR-V output,
// providing diagnostics on fail and means of configuring translation in a style
// of command line options.
extern "C" LLVM_EXTERNAL_VISIBILITY bool
SPIRVTranslateModule(Module *M, std::string &SpirvObj, std::string &ErrMsg,
                     const std::vector<std::string> &AllowExtNames,
                     const std::vector<std::string> &Opts) {
  // Fallbacks for option values.
  static const std::string DefaultTriple = "spirv64-unknown-unknown";
  static const std::string DefaultMArch = "";

  // Parse Opts as if it'd be command line arguments.
  std::string Errors;
  raw_string_ostream ErrorStream(Errors);
  parseSPIRVCommandLineOptions(Opts, &ErrorStream);
  if (!Errors.empty()) {
    ErrMsg = Errors;
    return false;
  }

  llvm::CodeGenOptLevel OLevel;
  if (auto Level = CodeGenOpt::parseLevel(SpvOptLevel)) {
    OLevel = *Level;
  } else {
    ErrMsg = "Invalid optimization level!";
    return false;
  }

  // Overrides/ammends `-spirv-ext` command line switch (if present) by the
  // explicit list of allowed SPIR-V extensions.
  std::set<SPIRV::Extension::Extension> AllowedExtIds;
  StringRef UnknownExt =
      SPIRVExtensionsParser::checkExtensions(AllowExtNames, AllowedExtIds);
  if (!UnknownExt.empty()) {
    ErrMsg = "Unknown SPIR-V extension: " + UnknownExt.str();
    return false;
  }
  SPIRVSubtarget::addExtensionsToClOpt(AllowedExtIds);

  // SPIR-V-specific target initialization.
  InitializeSPIRVTarget();

  Triple TargetTriple(SpvTargetTriple.empty()
                          ? M->getTargetTriple()
                          : Triple::normalize(SpvTargetTriple));
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
  MachineModuleInfoWrapperPass *MMIWP =
      new MachineModuleInfoWrapperPass(Target.get());
  const_cast<TargetLoweringObjectFile *>(Target->getObjFileLowering())
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

} // namespace llvm
