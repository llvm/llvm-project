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
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Error.h"
#include <optional>

using namespace llvm;
using namespace llvm::dxil;

LLVMContext *Ctx;

static bool reportError(Twine Message, DiagnosticSeverity Severity = DS_Error) {
  Ctx->diagnose(DiagnosticInfoGeneric(Message, Severity));
  return true;
}

static bool parseRootFlags(ModuleRootSignature *MRS, MDNode *RootFlagNode) {

  if (RootFlagNode->getNumOperands() != 2)
    return reportError("Invalid format for RootFlag Element");

  auto *Flag = mdconst::extract<ConstantInt>(RootFlagNode->getOperand(1));
  MRS->Flags = Flag->getZExtValue();

  return false;
}

static bool parseRootSignatureElement(ModuleRootSignature *MRS,
                                      MDNode *Element) {
  MDString *ElementText = cast<MDString>(Element->getOperand(0));
  if (ElementText == nullptr)
    return reportError("Invalid format for Root Element");

  RootSignatureElementKind ElementKind =
      StringSwitch<RootSignatureElementKind>(ElementText->getString())
          .Case("RootFlags", RootSignatureElementKind::RootFlags)
          .Case("RootConstants", RootSignatureElementKind::RootConstants)
          .Case("RootCBV", RootSignatureElementKind::RootDescriptor)
          .Case("RootSRV", RootSignatureElementKind::RootDescriptor)
          .Case("RootUAV", RootSignatureElementKind::RootDescriptor)
          .Case("Sampler", RootSignatureElementKind::RootDescriptor)
          .Case("DescriptorTable", RootSignatureElementKind::DescriptorTable)
          .Case("StaticSampler", RootSignatureElementKind::StaticSampler)
          .Default(RootSignatureElementKind::None);

  switch (ElementKind) {

  case RootSignatureElementKind::RootFlags:
    return parseRootFlags(MRS, Element);
  case RootSignatureElementKind::RootConstants:
  case RootSignatureElementKind::RootDescriptor:
  case RootSignatureElementKind::DescriptorTable:
  case RootSignatureElementKind::StaticSampler:
  case RootSignatureElementKind::None:
    return reportError("Invalid Root Element: " + ElementText->getString());
  }

  return true;
}

static bool parse(ModuleRootSignature *MRS, NamedMDNode *Root,
                  const Function *EF) {
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
    if (Node->getNumOperands() != 2)
      return reportError("Invalid format for Root Signature Definition. Pairs "
                         "of function, root signature expected.");

    ValueAsMetadata *VAM =
        llvm::dyn_cast<ValueAsMetadata>(Node->getOperand(0).get());
    if (VAM == nullptr)
      return reportError("First element of root signature is not a value");

    Function *F = dyn_cast<Function>(VAM->getValue());
    if (F == nullptr)
      return reportError("First element of root signature is not a function");

    if (F != EF)
      continue;

    // Get the Root Signature Description from the function signature pair.
    MDNode *RS = dyn_cast<MDNode>(Node->getOperand(1).get());

    if (RS == nullptr)
      return reportError("Missing Root Element List Metadata node.");

    // Loop through the Root Elements of the root signature.
    for (unsigned int Eid = 0; Eid < RS->getNumOperands(); Eid++) {
      MDNode *Element = dyn_cast<MDNode>(RS->getOperand(Eid));
      if (Element == nullptr)
        return reportError("Missing Root Element Metadata Node.");

      HasError = HasError || parseRootSignatureElement(MRS, Element);
    }
  }
  return HasError;
}

static bool validate(ModuleRootSignature *MRS) {
  if (dxbc::RootSignatureValidations::validateRootFlag(MRS->Flags)) {
    return reportError("Invalid Root Signature flag value");
  }
  return false;
}

std::optional<ModuleRootSignature>
ModuleRootSignature::analyzeModule(Module &M, const Function *F) {
  ModuleRootSignature MRS;
  Ctx = &M.getContext();

  NamedMDNode *RootSignatureNode = M.getNamedMetadata("dx.rootsignatures");
  if (RootSignatureNode == nullptr || parse(&MRS, RootSignatureNode, F) ||
      validate(&MRS))
    return std::nullopt;

  return MRS;
}

AnalysisKey RootSignatureAnalysis::Key;

std::optional<ModuleRootSignature>
RootSignatureAnalysis::run(Module &M, ModuleAnalysisManager &AM) {
  auto MMI = AM.getResult<DXILMetadataAnalysis>(M);

  if (MMI.ShaderProfile == Triple::Library)
    return std::nullopt;

  assert(MMI.EntryPropertyVec.size() == 1);

  const Function *EntryFunction = MMI.EntryPropertyVec[0].Entry;
  return ModuleRootSignature::analyzeModule(M, EntryFunction);
}

//===----------------------------------------------------------------------===//
bool RootSignatureAnalysisWrapper::runOnModule(Module &M) {

  dxil::ModuleMetadataInfo &MMI =
      getAnalysis<DXILMetadataAnalysisWrapperPass>().getModuleMetadata();

  if (MMI.ShaderProfile == Triple::Library)
    return false;
  assert(MMI.EntryPropertyVec.size() == 1);

  const Function *EntryFunction = MMI.EntryPropertyVec[0].Entry;
  MRS = ModuleRootSignature::analyzeModule(M, EntryFunction);
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
