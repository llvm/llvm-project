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
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include <cstdint>

using namespace llvm;
using namespace llvm::dxil;

static bool reportError(Twine Message) {
  report_fatal_error(Message, false);
  return true;
}

static bool parseRootFlags(ModuleRootSignature *MRS, MDNode *RootFlagNode) {

  if (RootFlagNode->getNumOperands() != 2)
    return reportError("Invalid format for RootFlag Element");

  auto *Flag = mdconst::extract<ConstantInt>(RootFlagNode->getOperand(1));
  uint32_t Value = Flag->getZExtValue();

  // Root Element validation, as specified:
  // https://github.com/llvm/wg-hlsl/blob/main/proposals/0002-root-signature-in-clang.md#validations-during-dxil-generation
  if ((Value & ~0x80000fff) != 0)
    return reportError("Invalid flag value for RootFlag");

  MRS->Flags = Value;
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

  case RootSignatureElementKind::RootFlags: {
    return parseRootFlags(MRS, Element);
    break;
  }

  case RootSignatureElementKind::RootConstants:
  case RootSignatureElementKind::RootDescriptor:
  case RootSignatureElementKind::DescriptorTable:
  case RootSignatureElementKind::StaticSampler:
  case RootSignatureElementKind::None:
    return reportError("Invalid Root Element: " + ElementText->getString());
    break;
  }

  return true;
}

bool ModuleRootSignature::parse(NamedMDNode *Root) {
  bool HasError = false;

  /** Root Signature are specified as following in the metadata:

      !dx.rootsignatures = !{!2} ; list of function/root signature pairs
      !2 = !{ ptr @main, !3 } ; function, root signature
      !3 = !{ !4, !5, !6, !7 } ; list of root signature elements

      So for each MDNode inside dx.rootsignatures NamedMDNode
      (the Root parameter of this function), the parsing process needs
      to loop through each of it's operand and process the pairs function
      signature pair.
   */

  for (const MDNode *Node : Root->operands()) {

    if (Node->getNumOperands() != 2)
      return reportError("Invalid format for Root Signature Definition. Pairs "
                         "of function, root signature expected.");

    // Get the Root Signature Description from the function signature pair.
    MDNode *RS = dyn_cast<MDNode>(Node->getOperand(1).get());

    if (RS == nullptr)
      return reportError("Missing Root Signature Metadata node.");

    // Loop through the Root Elements of the root signature.
    for (unsigned int Eid = 0; Eid < RS->getNumOperands(); Eid++) {

      MDNode *Element = dyn_cast<MDNode>(RS->getOperand(Eid));
      if (Element == nullptr)
        return reportError("Missing Root Element Metadata Node.");

      HasError = HasError || parseRootSignatureElement(this, Element);
    }
  }
  return HasError;
}

ModuleRootSignature ModuleRootSignature::analyzeModule(Module &M) {
  ModuleRootSignature MRS;

  NamedMDNode *RootSignatureNode = M.getNamedMetadata("dx.rootsignatures");
  if (RootSignatureNode) {
    if (MRS.parse(RootSignatureNode))
      llvm_unreachable("Invalid Root Signature Metadata.");
  }

  return MRS;
}

AnalysisKey RootSignatureAnalysis::Key;

ModuleRootSignature RootSignatureAnalysis::run(Module &M,
                                               ModuleAnalysisManager &AM) {
  return ModuleRootSignature::analyzeModule(M);
}

//===----------------------------------------------------------------------===//
bool RootSignatureAnalysisWrapper::runOnModule(Module &M) {

  this->MRS = MRS = ModuleRootSignature::analyzeModule(M);

  return false;
}

void RootSignatureAnalysisWrapper::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
}

char RootSignatureAnalysisWrapper::ID = 0;

INITIALIZE_PASS(RootSignatureAnalysisWrapper, "dx-root-signature-analysis",
                "DXIL Root Signature Analysis", true, true)
