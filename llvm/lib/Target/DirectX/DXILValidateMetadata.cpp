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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/DXILMetadataAnalysis.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {

/// A simple wrapper of DiagnosticInfo that generates module-level diagnostic
/// for the DXILValidateMetadata pass
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

static bool reportLoopError(Module &M, Twine Message,
                            DiagnosticSeverity Severity = DS_Error) {
  return reportError(M, Twine("Invalid \"llvm.loop\" metadata: ") + Message,
                     Severity);
}

} // namespace

static void validateLoopMetadata(Module &M, MDNode *LoopMD) {
  // DXIL only accepts the following loop hints:
  //   llvm.loop.unroll.disable, llvm.loop.unroll.full, llvm.loop.unroll.count
  std::array<StringLiteral, 3> ValidHintNames = {"llvm.loop.unroll.count",
                                                 "llvm.loop.unroll.disable",
                                                 "llvm.loop.unroll.full"};

  // llvm.loop metadata must have its first operand be a self-reference, so we
  // require at least 1 operand.
  //
  // It only makes sense to specify up to 1 of the hints on a branch, so we can
  // have at most 2 operands.

  if (LoopMD->getNumOperands() != 1 && LoopMD->getNumOperands() != 2) {
    reportLoopError(M, "Requires exactly 1 or 2 operands");
    return;
  }

  if (LoopMD != LoopMD->getOperand(0)) {
    reportLoopError(M, "First operand must be a self-reference");
    return;
  }

  // A node only containing a self-reference is a valid use to denote a loop
  if (LoopMD->getNumOperands() == 1)
    return;

  LoopMD = dyn_cast<MDNode>(LoopMD->getOperand(1));
  if (!LoopMD) {
    reportLoopError(M, "Second operand must be a metadata node");
    return;
  }

  if (LoopMD->getNumOperands() != 1 && LoopMD->getNumOperands() != 2) {
    reportLoopError(M, "Requires exactly 1 or 2 operands");
    return;
  }

  // It is valid to have a chain of self-referential loop metadata nodes so if
  // we have another self-reference, recurse.
  //
  // Eg:
  // !0 = !{!0, !1}
  // !1 = !{!1, !2}
  // !2 = !{"llvm.loop.unroll.disable"}
  if (LoopMD == LoopMD->getOperand(0))
    return validateLoopMetadata(M, LoopMD);

  // Otherwise, we are at our base hint metadata node
  auto *HintStr = dyn_cast<MDString>(LoopMD->getOperand(0));
  if (!HintStr || !llvm::is_contained(ValidHintNames, HintStr->getString())) {
    reportLoopError(M,
                    "First operand must be a valid \"llvm.loop.unroll\" hint");
    return;
  }

  // Ensure count node is a constant integer value
  auto ValidCountNode = [](MDNode *HintMD) -> bool {
    if (HintMD->getNumOperands() == 2)
      if (auto *CountMD = dyn_cast<ConstantAsMetadata>(HintMD->getOperand(1)))
        if (isa<ConstantInt>(CountMD->getValue()))
          return true;
    return false;
  };

  if (HintStr->getString() == "llvm.loop.unroll.count") {
    if (!ValidCountNode(LoopMD)) {
      reportLoopError(M, "Second operand of \"llvm.loop.unroll.count\" "
                         "must be a constant integer");
      return;
    }
  } else if (LoopMD->getNumOperands() != 1) {
    reportLoopError(M, "Can't have a second operand");
    return;
  }
}

static void validateInstructionMetadata(Module &M) {
  unsigned char MDLoopKind = M.getContext().getMDKindID("llvm.loop");

  for (Function &F : M)
    for (BasicBlock &BB : F)
      for (Instruction &I : BB) {
        if (isa<BranchInst>(I))
          if (MDNode *LoopMD = I.getMetadata(MDLoopKind))
            validateLoopMetadata(M, LoopMD);
      }
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
    validateInstructionMetadata(M);
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
