//===- DXILValidateMetadata.cpp - Pass to validate DXIL metadata ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DXILValidateMetadata.h"
#include "DXILTranslateMetadata.h"
#include "DirectX.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/DXILMetadataAnalysis.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {

/// A simple Wrapper DiagnosticInfo that generates Module-level diagnostic
/// for the ValidateMetadata pass
class DiagnosticInfoValidateMD : public DiagnosticInfo {
private:
  const Twine &Msg;
  const Module &Mod;

public:
  /// \p M is the module for which the diagnostic is being emitted. \p Msg is
  /// the message to show. Note that this class does not copy this message, so
  /// this reference must be valid for the whole life time of the diagnostic.
  DiagnosticInfoValidateMD(const Module &M,
                           const Twine &Msg LLVM_LIFETIME_BOUND,
                           DiagnosticSeverity Severity = DS_Error)
      : DiagnosticInfo(DK_Unsupported, Severity), Msg(Msg), Mod(M) {}

  void print(DiagnosticPrinter &DP) const override {
    DP << Mod.getName() << ": " << Msg << '\n';
  }
};

static bool reportError(Module &M, Twine Message,
                        DiagnosticSeverity Severity = DS_Error) {
  M.getContext().diagnose(DiagnosticInfoValidateMD(M, Message, Severity));
  return true;
}

} // namespace

static void validateInstructionMetadata(Module &M) {
  llvm::errs() << "hello from new pass!\n";
}

static void validateGlobalMetadata(Module &M,
                                   const dxil::ModuleMetadataInfo &MMDI) {
  if (MMDI.ShaderProfile != Triple::EnvironmentType::Library) {
    if (1 < MMDI.EntryPropertyVec.size())
      reportError(M, "Non-library shader: One and only one entry expected");

    for (const dxil::EntryProperties &EntryProp : MMDI.EntryPropertyVec)
      if (EntryProp.ShaderStage != MMDI.ShaderProfile)
        reportError(
            M,
            "Shader stage '" +
                Twine(Twine(Triple::getEnvironmentTypeName(
                          EntryProp.ShaderStage)) +
                      "' for entry '" + Twine(EntryProp.Entry->getName()) +
                      "' different from specified target profile '" +
                      Twine(Triple::getEnvironmentTypeName(MMDI.ShaderProfile) +
                            "'")));
  }
}

PreservedAnalyses DXILValidateMetadata::run(Module &M,
                                            ModuleAnalysisManager &MAM) {

  const dxil::ModuleMetadataInfo MMDI = MAM.getResult<DXILMetadataAnalysis>(M);
  validateGlobalMetadata(M, MMDI);
  validateInstructionMetadata(M);

  return PreservedAnalyses::all();
}

namespace {
class DXILValidateMetadataLegacy : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  explicit DXILValidateMetadataLegacy() : ModulePass(ID) {}

  StringRef getPassName() const override { return "DXIL Validate Metadata"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DXILMetadataAnalysisWrapperPass>();
    AU.addRequired<DXILTranslateMetadataLegacy>();
    AU.setPreservesAll();
  }

  bool runOnModule(Module &M) override {
    dxil::ModuleMetadataInfo MMDI =
        getAnalysis<DXILMetadataAnalysisWrapperPass>().getModuleMetadata();
    validateGlobalMetadata(M, MMDI);
    return true;
  }
};

} // namespace

char DXILValidateMetadataLegacy::ID = 0;

ModulePass *llvm::createDXILValidateMetadataLegacyPass() {
  return new DXILValidateMetadataLegacy();
}

INITIALIZE_PASS_BEGIN(DXILValidateMetadataLegacy, "dxil-validate-metadata",
                      "DXIL Validate Metadata", false, false)
INITIALIZE_PASS_DEPENDENCY(DXILMetadataAnalysisWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DXILTranslateMetadataLegacy)
INITIALIZE_PASS_END(DXILValidateMetadataLegacy, "dxil-validate-metadata",
                    "DXIL validate Metadata", false, false)
