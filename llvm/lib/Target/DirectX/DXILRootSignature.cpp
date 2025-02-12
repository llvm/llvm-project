//===- DXILRootSignature.cpp - DXIL Root Signature helper objects ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains helper objects and APIs for working with DXIL
///       Root Signatures.
///
//===----------------------------------------------------------------------===//
#include "DXILRootSignature.h"
#include "DirectX.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/DXILMetadataAnalysis.h"
#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Error.h"
#include <optional>

using namespace llvm;
using namespace llvm::dxil;

static bool reportError(LLVMContext *Ctx, Twine Message,
                        DiagnosticSeverity Severity = DS_Error) {
  Ctx->diagnose(DiagnosticInfoGeneric(Message, Severity));
  return true;
}

static bool parseRootFlags(LLVMContext *Ctx, ModuleRootSignature *MRS,
                           MDNode *RootFlagNode) {

  if (RootFlagNode->getNumOperands() != 2)
    return reportError(Ctx, "Invalid format for RootFlag Element");

  auto *Flag = mdconst::extract<ConstantInt>(RootFlagNode->getOperand(1));
  MRS->Flags = Flag->getZExtValue();

  return false;
}

static bool parseRootSignatureElement(LLVMContext *Ctx,
                                      ModuleRootSignature *MRS,
                                      MDNode *Element) {
  MDString *ElementText = cast<MDString>(Element->getOperand(0));
  if (ElementText == nullptr)
    return reportError(Ctx, "Invalid format for Root Element");

  RootSignatureElementKind ElementKind =
      StringSwitch<RootSignatureElementKind>(ElementText->getString())
          .Case("RootFlags", RootSignatureElementKind::RootFlags)
          .Default(RootSignatureElementKind::None);

  switch (ElementKind) {

  case RootSignatureElementKind::RootFlags:
    return parseRootFlags(Ctx, MRS, Element);
  case RootSignatureElementKind::None:
    return reportError(Ctx,
                       "Invalid Root Element: " + ElementText->getString());
  }

  llvm_unreachable("Root signature element kind not expected.");
}

static bool parse(LLVMContext *Ctx, ModuleRootSignature *MRS, NamedMDNode *Root,
                  const Function *EntryFunction) {
  bool HasError = false;

  /** Root Signature are specified as following in the metadata:

      !dx.rootsignatures = !{!2} ; list of function/root signature pairs
      !2 = !{ ptr @main, !3 } ; function, root signature
      !3 = !{ !4, !5, !6, !7 } ; list of root signature elements

      So for each MDNode inside dx.rootsignatures NamedMDNode
      (the Root parameter of this function), the parsing process needs
      to loop through each of its operands and process the function,
      signature pair.
   */

  for (const MDNode *Node : Root->operands()) {
    if (Node->getNumOperands() != 2) {
      HasError = reportError(
          Ctx, "Invalid format for Root Signature Definition. Pairs "
               "of function, root signature expected.");
      continue;
    }

    const MDOperand &FunctionPointerMdNode = Node->getOperand(0);
    if (FunctionPointerMdNode == nullptr) {
      // Function was pruned during compilation.
      continue;
    }

    ValueAsMetadata *VAM =
        llvm::dyn_cast<ValueAsMetadata>(FunctionPointerMdNode.get());
    if (VAM == nullptr) {
      HasError =
          reportError(Ctx, "First element of root signature is not a value");
      continue;
    }

    Function *F = dyn_cast<Function>(VAM->getValue());
    if (F == nullptr) {
      HasError =
          reportError(Ctx, "First element of root signature is not a function");
      continue;
    }

    if (F != EntryFunction)
      continue;

    // Get the Root Signature Description from the function signature pair.
    MDNode *RS = dyn_cast<MDNode>(Node->getOperand(1).get());

    if (RS == nullptr) {
      reportError(Ctx, "Missing Root Element List Metadata node.");
      continue;
    }

    // Loop through the Root Elements of the root signature.
    for (const auto &Operand : RS->operands()) {
      MDNode *Element = dyn_cast<MDNode>(Operand);
      if (Element == nullptr)
        return reportError(Ctx, "Missing Root Element Metadata Node.");

      HasError = HasError || parseRootSignatureElement(Ctx, MRS, Element);
    }
  }
  return HasError;
}

static bool validate(LLVMContext *Ctx, ModuleRootSignature *MRS) {
  if (!dxbc::RootSignatureValidations::isValidRootFlag(MRS->Flags)) {
    return reportError(Ctx, "Invalid Root Signature flag value");
  }
  return false;
}

static const Function *getEntryFunction(Module &M, ModuleMetadataInfo MMI) {

  LLVMContext *Ctx = &M.getContext();
  if (MMI.EntryPropertyVec.size() != 1) {
    reportError(Ctx, "More than one entry function defined.");
    return nullptr;
  }
  return MMI.EntryPropertyVec[0].Entry;
}

std::optional<ModuleRootSignature>
ModuleRootSignature::analyzeModule(Module &M, const Function *F) {

  LLVMContext *Ctx = &M.getContext();

  ModuleRootSignature MRS;

  NamedMDNode *RootSignatureNode = M.getNamedMetadata("dx.rootsignatures");
  if (RootSignatureNode == nullptr || parse(Ctx, &MRS, RootSignatureNode, F) ||
      validate(Ctx, &MRS))
    return std::nullopt;

  return MRS;
}

AnalysisKey RootSignatureAnalysis::Key;

std::optional<ModuleRootSignature>
RootSignatureAnalysis::run(Module &M, ModuleAnalysisManager &AM) {
  ModuleMetadataInfo MMI = AM.getResult<DXILMetadataAnalysis>(M);
  if (MMI.ShaderProfile == Triple::Library)
    return std::nullopt;
  return ModuleRootSignature::analyzeModule(M, getEntryFunction(M, MMI));
}

//===----------------------------------------------------------------------===//
bool RootSignatureAnalysisWrapper::runOnModule(Module &M) {
  dxil::ModuleMetadataInfo &MMI =
      getAnalysis<DXILMetadataAnalysisWrapperPass>().getModuleMetadata();
  if (MMI.ShaderProfile == Triple::Library)
    return false;
  MRS = ModuleRootSignature::analyzeModule(M, getEntryFunction(M, MMI));
  return false;
}

void RootSignatureAnalysisWrapper::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<DXILMetadataAnalysisWrapperPass>();
}

char RootSignatureAnalysisWrapper::ID = 0;

INITIALIZE_PASS_BEGIN(RootSignatureAnalysisWrapper,
                      "dx-root-signature-analysis",
                      "DXIL Root Signature Analysis", true, true)
INITIALIZE_PASS_DEPENDENCY(DXILMetadataAnalysisWrapperPass)
INITIALIZE_PASS_END(RootSignatureAnalysisWrapper, "dx-root-signature-analysis",
                    "DXIL Root Signature Analysis", true, true)
