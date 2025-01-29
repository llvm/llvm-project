//===- DXILRootSignature.cpp - DXIL Root Signature helper objects
//---------------===//
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
#include "llvm/IR/Constants.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"

using namespace llvm;
using namespace llvm::dxil;

static bool parseRootFlags(ModuleRootSignature *MRS, MDNode *RootFlagNode) {

  assert(RootFlagNode->getNumOperands() == 2 &&
         "Invalid format for RootFlag Element");
  auto *Flag = mdconst::extract<ConstantInt>(RootFlagNode->getOperand(1));
  auto Value = Flag->getZExtValue();

  // Root Element validation, as specified:
  // https://github.com/llvm/wg-hlsl/blob/main/proposals/0002-root-signature-in-clang.md#validations-during-dxil-generation
  assert((Value & ~0x80000fff) == 0 && "Invalid flag for RootFlag Element");

  MRS->Flags = Value;
  return false;
}

static bool parseRootSignatureElement(ModuleRootSignature *MRS,
                                      MDNode *Element) {
  MDString *ElementText = cast<MDString>(Element->getOperand(0));
  assert(ElementText != nullptr &&
         "First preoperty of element is not a string");

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
    llvm_unreachable("Not Implemented yet");
    break;
  }

  return true;
}

bool ModuleRootSignature::parse(int32_t Version, NamedMDNode *Root) {
  this->Version = Version;
  bool HasError = false;

  for (unsigned int Sid = 0; Sid < Root->getNumOperands(); Sid++) {
    // This should be an if, for error handling
    MDNode *Node = cast<MDNode>(Root->getOperand(Sid));

    // Not sure what use this for...
    // Metadata *Func = Node->getOperand(0).get();

    MDNode *Elements = cast<MDNode>(Node->getOperand(1).get());
    assert(Elements && "Invalid Metadata type on root signature");

    for (unsigned int Eid = 0; Eid < Elements->getNumOperands(); Eid++) {
      MDNode *Element = cast<MDNode>(Elements->getOperand(Eid));
      assert(Element && "Invalid Metadata type on root element");

      HasError = HasError || parseRootSignatureElement(this, Element);
    }
  }
  return HasError;
}

AnalysisKey RootSignatureAnalysis::Key;

ModuleRootSignature RootSignatureAnalysis::run(Module &M,
                                               ModuleAnalysisManager &AM) {
  ModuleRootSignature MRSI;

  NamedMDNode *RootSignatureNode = M.getNamedMetadata("dx.rootsignatures");
  if (RootSignatureNode) {
    MRSI.parse(1, RootSignatureNode);
  }

  return MRSI;
}

//===----------------------------------------------------------------------===//
bool RootSignatureAnalysisWrapper::runOnModule(Module &M) {
  ModuleRootSignature MRS;

  NamedMDNode *RootSignatureNode = M.getNamedMetadata("dx.rootsignatures");
  if (RootSignatureNode) {
    MRS.parse(1, RootSignatureNode);
    this->MRS = MRS;
  }

  return false;
}

void RootSignatureAnalysisWrapper::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
}

char RootSignatureAnalysisWrapper::ID = 0;

INITIALIZE_PASS(RootSignatureAnalysisWrapper, "dx-root-signature-analysis",
                "DXIL Root Signature Analysis", true, true)
