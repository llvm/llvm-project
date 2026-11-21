//===-- SPIRVAPI.cpp - SPIR-V Backend API ---------------------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SPIRVCommandLine.h"
#include "SPIRVSubtarget.h"
#include "SPIRVTargetMachine.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/RunCodeGen.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include "llvm/TargetParser/Triple.h"
#include <optional>
#include <string>
#include <vector>

using namespace llvm;

namespace {

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
// providing diagnostics on fail and means of configuring translation.
extern "C" LLVM_EXTERNAL_VISIBILITY bool
SPIRVTranslate(Module *M, std::string &SpirvObj, std::string &ErrMsg,
               const std::vector<std::string> &ExtNames,
               llvm::CodeGenOptLevel OLevel, Triple TargetTriple) {
  // Fallbacks for option values.
  static const std::string DefaultTriple = "spirv64-unknown-unknown";
  static const std::string DefaultMArch = "";

  ExtensionSet AllowedExtIds;
  if (Error Err =
          SPIRVExtensionsParser::checkExtensions(ExtNames, AllowedExtIds)) {
    ErrMsg = toString(std::move(Err));
    return false;
  }

  // SPIR-V-specific target initialization.
  InitializeSPIRVTarget();

  if (TargetTriple.getTriple().empty()) {
    TargetTriple.setTriple(DefaultTriple);
    M->setTargetTriple(TargetTriple);
  }
  const Target *TheTarget =
      TargetRegistry::lookupTarget(DefaultMArch, TargetTriple, ErrMsg);
  if (!TheTarget)
    return false;

  // A call to codegen::getTargetOptions(TargetTriple) hits an
  // assertion in one of the codegen flag getters in
  // llvm/lib/CodeGen/CommandFlags.cpp:
  // `...View && "RegisterCodeGenFlags not created."' failed.
  TargetOptions Options;
  std::optional<Reloc::Model> RM;
  std::optional<CodeModel::Model> CM;
  std::unique_ptr<TargetMachine> Target(TheTarget->createTargetMachine(
      TargetTriple, "", "", Options, RM, CM, OLevel));
  if (!Target) {
    ErrMsg = "Could not allocate target machine!";
    return false;
  }

  // Set available extensions.
  SPIRVTargetMachine *STM = static_cast<SPIRVTargetMachine *>(Target.get());
  const_cast<SPIRVSubtarget *>(STM->getSubtargetImpl())
      ->initAvailableExtensions(AllowedExtIds);

  if (M->getCodeModel())
    Target->setCodeModel(*M->getCodeModel());

  std::string DLStr = M->getDataLayoutStr();
  Expected<DataLayout> MaybeDL = DataLayout::parse(
      DLStr.empty() ? TargetTriple.computeDataLayout() : DLStr);
  if (!MaybeDL) {
    ErrMsg = toString(MaybeDL.takeError());
    return false;
  }
  M->setDataLayout(MaybeDL.get());

  SmallString<4096> OutBuffer;
  raw_svector_ostream OutStream(OutBuffer);
  if (Error Err = runCodeGenPipeline(*Target, *M, OutStream, nullptr,
                                     CodeGenFileType::ObjectFile)) {
    ErrMsg = toString(std::move(Err));
    return false;
  }
  SpirvObj = OutBuffer.str();

  return true;
}

// TODO: Remove this wrapper after existing clients switch into a newer
// implementation of SPIRVTranslate().
extern "C" LLVM_EXTERNAL_VISIBILITY bool
SPIRVTranslateModule(Module *M, std::string &SpirvObj, std::string &ErrMsg,
                     const std::vector<std::string> &ExtNames,
                     const std::vector<std::string> &Opts) {
  // optional: Opts[0] is a string representation of Triple,
  // take Module triple otherwise
  Triple TargetTriple = Opts.empty() || Opts[0].empty()
                            ? M->getTargetTriple()
                            : Triple(Triple::normalize(Opts[0]));
  // optional: Opts[1] is a string representation of CodeGenOptLevel,
  // no optimization otherwise
  llvm::CodeGenOptLevel OLevel = CodeGenOptLevel::None;
  if (Opts.size() > 1 && !Opts[1].empty()) {
    if (auto Level = CodeGenOpt::parseLevel(Opts[1][0])) {
      OLevel = *Level;
    } else {
      ErrMsg = "Invalid optimization level!";
      return false;
    }
  }
  return SPIRVTranslate(M, SpirvObj, ErrMsg, ExtNames, OLevel,
                        std::move(TargetTriple));
}

} // namespace llvm
