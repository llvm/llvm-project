//===- DXILRootSignature.cpp - DXIL Root Signature helper objects -------===//
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
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <optional>
#include <utility>

using namespace llvm;
using namespace llvm::dxil;

static bool reportError(LLVMContext *Ctx, Twine Message,
                        DiagnosticSeverity Severity = DS_Error) {
  Ctx->diagnose(DiagnosticInfoGeneric(Message, Severity));
  return true;
}

static bool reportValueError(LLVMContext *Ctx, Twine ParamName,
                             uint32_t Value) {
  Ctx->diagnose(DiagnosticInfoGeneric(
      "Invalid value for " + ParamName + ": " + Twine(Value), DS_Error));
  return true;
}

static std::optional<uint32_t> extractMdIntValue(MDNode *Node,
                                                 unsigned int OpId) {
  if (auto *CI =
          mdconst::dyn_extract<ConstantInt>(Node->getOperand(OpId).get()))
    return CI->getZExtValue();
  return std::nullopt;
}

static std::optional<StringRef> extractMdStringValue(MDNode *Node,
                                                     unsigned int OpId) {
  MDString *NodeText = dyn_cast<MDString>(Node->getOperand(OpId));
  if (NodeText == nullptr)
    return std::nullopt;
  return NodeText->getString();
}

static bool parseRootFlags(LLVMContext *Ctx, mcdxbc::RootSignatureDesc &RSD,
                           MDNode *RootFlagNode) {

  if (RootFlagNode->getNumOperands() != 2)
    return reportError(Ctx, "Invalid format for RootFlag Element");

  if (std::optional<uint32_t> Val = extractMdIntValue(RootFlagNode, 1))
    RSD.Flags = *Val;
  else
    return reportError(Ctx, "Invalid value for RootFlag");

  return false;
}

static bool parseRootConstants(LLVMContext *Ctx, mcdxbc::RootSignatureDesc &RSD,
                               MDNode *RootConstantNode) {

  if (RootConstantNode->getNumOperands() != 5)
    return reportError(Ctx, "Invalid format for RootConstants Element");

  dxbc::RTS0::v1::RootParameterHeader Header;
  // The parameter offset doesn't matter here - we recalculate it during
  // serialization  Header.ParameterOffset = 0;
  Header.ParameterType =
      llvm::to_underlying(dxbc::RootParameterType::Constants32Bit);

  if (std::optional<uint32_t> Val = extractMdIntValue(RootConstantNode, 1))
    Header.ShaderVisibility = *Val;
  else
    return reportError(Ctx, "Invalid value for ShaderVisibility");

  dxbc::RTS0::v1::RootConstants Constants;
  if (std::optional<uint32_t> Val = extractMdIntValue(RootConstantNode, 2))
    Constants.ShaderRegister = *Val;
  else
    return reportError(Ctx, "Invalid value for ShaderRegister");

  if (std::optional<uint32_t> Val = extractMdIntValue(RootConstantNode, 3))
    Constants.RegisterSpace = *Val;
  else
    return reportError(Ctx, "Invalid value for RegisterSpace");

  if (std::optional<uint32_t> Val = extractMdIntValue(RootConstantNode, 4))
    Constants.Num32BitValues = *Val;
  else
    return reportError(Ctx, "Invalid value for Num32BitValues");

  RSD.ParametersContainer.addParameter(Header, Constants);

  return false;
}

static bool parseRootDescriptors(LLVMContext *Ctx,
                                 mcdxbc::RootSignatureDesc &RSD,
                                 MDNode *RootDescriptorNode,
                                 RootSignatureElementKind ElementKind) {
  assert(ElementKind == RootSignatureElementKind::SRV ||
         ElementKind == RootSignatureElementKind::UAV ||
         ElementKind == RootSignatureElementKind::CBV &&
             "parseRootDescriptors should only be called with RootDescriptor "
             "element kind.");
  if (RootDescriptorNode->getNumOperands() != 5)
    return reportError(Ctx, "Invalid format for Root Descriptor Element");

  dxbc::RTS0::v1::RootParameterHeader Header;
  switch (ElementKind) {
  case RootSignatureElementKind::SRV:
    Header.ParameterType = llvm::to_underlying(dxbc::RootParameterType::SRV);
    break;
  case RootSignatureElementKind::UAV:
    Header.ParameterType = llvm::to_underlying(dxbc::RootParameterType::UAV);
    break;
  case RootSignatureElementKind::CBV:
    Header.ParameterType = llvm::to_underlying(dxbc::RootParameterType::CBV);
    break;
  default:
    llvm_unreachable("invalid Root Descriptor kind");
    break;
  }

  if (std::optional<uint32_t> Val = extractMdIntValue(RootDescriptorNode, 1))
    Header.ShaderVisibility = *Val;
  else
    return reportError(Ctx, "Invalid value for ShaderVisibility");

  dxbc::RTS0::v2::RootDescriptor Descriptor;
  if (std::optional<uint32_t> Val = extractMdIntValue(RootDescriptorNode, 2))
    Descriptor.ShaderRegister = *Val;
  else
    return reportError(Ctx, "Invalid value for ShaderRegister");

  if (std::optional<uint32_t> Val = extractMdIntValue(RootDescriptorNode, 3))
    Descriptor.RegisterSpace = *Val;
  else
    return reportError(Ctx, "Invalid value for RegisterSpace");

  if (RSD.Version == 1) {
    RSD.ParametersContainer.addParameter(Header, Descriptor);
    return false;
  }
  assert(RSD.Version > 1);

  if (std::optional<uint32_t> Val = extractMdIntValue(RootDescriptorNode, 4))
    Descriptor.Flags = *Val;
  else
    return reportError(Ctx, "Invalid value for Root Descriptor Flags");

  RSD.ParametersContainer.addParameter(Header, Descriptor);
  return false;
}

static bool parseRootSignatureElement(LLVMContext *Ctx,
                                      mcdxbc::RootSignatureDesc &RSD,
                                      MDNode *Element) {
  std::optional<StringRef> ElementText = extractMdStringValue(Element, 0);
  if (!ElementText.has_value())
    return reportError(Ctx, "Invalid format for Root Element");

  RootSignatureElementKind ElementKind =
      StringSwitch<RootSignatureElementKind>(*ElementText)
          .Case("RootFlags", RootSignatureElementKind::RootFlags)
          .Case("RootConstants", RootSignatureElementKind::RootConstants)
          .Case("RootCBV", RootSignatureElementKind::CBV)
          .Case("RootSRV", RootSignatureElementKind::SRV)
          .Case("RootUAV", RootSignatureElementKind::UAV)
          .Default(RootSignatureElementKind::Error);

  switch (ElementKind) {

  case RootSignatureElementKind::RootFlags:
    return parseRootFlags(Ctx, RSD, Element);
  case RootSignatureElementKind::RootConstants:
    return parseRootConstants(Ctx, RSD, Element);
  case RootSignatureElementKind::CBV:
  case RootSignatureElementKind::SRV:
  case RootSignatureElementKind::UAV:
    return parseRootDescriptors(Ctx, RSD, Element, ElementKind);
  case RootSignatureElementKind::Error:
    return reportError(Ctx, "Invalid Root Signature Element: " + *ElementText);
  }

  llvm_unreachable("Unhandled RootSignatureElementKind enum.");
}

static bool parse(LLVMContext *Ctx, mcdxbc::RootSignatureDesc &RSD,
                  MDNode *Node) {
  bool HasError = false;

  // Loop through the Root Elements of the root signature.
  for (const auto &Operand : Node->operands()) {
    MDNode *Element = dyn_cast<MDNode>(Operand);
    if (Element == nullptr)
      return reportError(Ctx, "Missing Root Element Metadata Node.");

    HasError = HasError || parseRootSignatureElement(Ctx, RSD, Element);
  }

  return HasError;
}

static bool verifyRootFlag(uint32_t Flags) { return (Flags & ~0xfff) == 0; }

static bool verifyVersion(uint32_t Version) {
  return (Version == 1 || Version == 2);
}

static bool verifyRegisterValue(uint32_t RegisterValue) {
  return RegisterValue != ~0U;
}

// This Range is reserverved, therefore invalid, according to the spec
// https://github.com/llvm/wg-hlsl/blob/main/proposals/0002-root-signature-in-clang.md#all-the-values-should-be-legal
static bool verifyRegisterSpace(uint32_t RegisterSpace) {
  return !(RegisterSpace >= 0xFFFFFFF0 && RegisterSpace <= 0xFFFFFFFF);
}

static bool verifyDescriptorFlag(uint32_t Flags) { return (Flags & ~0xE) == 0; }

static bool validate(LLVMContext *Ctx, const mcdxbc::RootSignatureDesc &RSD) {

  if (!verifyVersion(RSD.Version)) {
    return reportValueError(Ctx, "Version", RSD.Version);
  }

  if (!verifyRootFlag(RSD.Flags)) {
    return reportValueError(Ctx, "RootFlags", RSD.Flags);
  }

  for (const mcdxbc::RootParameterInfo &Info : RSD.ParametersContainer) {
    if (!dxbc::isValidShaderVisibility(Info.Header.ShaderVisibility))
      return reportValueError(Ctx, "ShaderVisibility",
                              Info.Header.ShaderVisibility);

    assert(dxbc::isValidParameterType(Info.Header.ParameterType) &&
           "Invalid value for ParameterType");

    switch (Info.Header.ParameterType) {

    case llvm::to_underlying(dxbc::RootParameterType::CBV):
    case llvm::to_underlying(dxbc::RootParameterType::UAV):
    case llvm::to_underlying(dxbc::RootParameterType::SRV): {
      const dxbc::RTS0::v2::RootDescriptor &Descriptor =
          RSD.ParametersContainer.getRootDescriptor(Info.Location);
      if (!verifyRegisterValue(Descriptor.ShaderRegister))
        return reportValueError(Ctx, "ShaderRegister",
                                Descriptor.ShaderRegister);

      if (!verifyRegisterSpace(Descriptor.RegisterSpace))
        return reportValueError(Ctx, "RegisterSpace", Descriptor.RegisterSpace);

      if (RSD.Version > 1) {
        if (!verifyDescriptorFlag(Descriptor.Flags))
          return reportValueError(Ctx, "DescriptorFlag", Descriptor.Flags);
      }
      break;
    }
    }
  }

  return false;
}

static SmallDenseMap<const Function *, mcdxbc::RootSignatureDesc>
analyzeModule(Module &M) {

  /** Root Signature are specified as following in the metadata:

    !dx.rootsignatures = !{!2} ; list of function/root signature pairs
    !2 = !{ ptr @main, !3 } ; function, root signature
    !3 = !{ !4, !5, !6, !7 } ; list of root signature elements

    So for each MDNode inside dx.rootsignatures NamedMDNode
    (the Root parameter of this function), the parsing process needs
    to loop through each of its operands and process the function,
    signature pair.
 */

  LLVMContext *Ctx = &M.getContext();

  SmallDenseMap<const Function *, mcdxbc::RootSignatureDesc> RSDMap;

  NamedMDNode *RootSignatureNode = M.getNamedMetadata("dx.rootsignatures");
  if (RootSignatureNode == nullptr)
    return RSDMap;

  for (const auto &RSDefNode : RootSignatureNode->operands()) {
    if (RSDefNode->getNumOperands() != 2) {
      reportError(Ctx, "Invalid format for Root Signature Definition. Pairs "
                       "of function, root signature expected.");
      continue;
    }

    // Function was pruned during compilation.
    const MDOperand &FunctionPointerMdNode = RSDefNode->getOperand(0);
    if (FunctionPointerMdNode == nullptr) {
      reportError(
          Ctx, "Function associated with Root Signature definition is null.");
      continue;
    }

    ValueAsMetadata *VAM =
        llvm::dyn_cast<ValueAsMetadata>(FunctionPointerMdNode.get());
    if (VAM == nullptr) {
      reportError(Ctx, "First element of root signature is not a Value");
      continue;
    }

    Function *F = dyn_cast<Function>(VAM->getValue());
    if (F == nullptr) {
      reportError(Ctx, "First element of root signature is not a Function");
      continue;
    }

    Metadata *RootElementListOperand = RSDefNode->getOperand(1).get();

    if (RootElementListOperand == nullptr) {
      reportError(Ctx, "Root Element mdnode is null.");
      continue;
    }

    MDNode *RootElementListNode = dyn_cast<MDNode>(RootElementListOperand);
    if (RootElementListNode == nullptr) {
      reportError(Ctx, "Root Element is not a metadata node.");
      continue;
    }

    mcdxbc::RootSignatureDesc RSD;
    // Clang emits the root signature data in dxcontainer following a specific
    // sequence. First the header, then the root parameters. So the header
    // offset will always equal to the header size.
    RSD.RootParameterOffset = sizeof(dxbc::RTS0::v1::RootSignatureHeader);

    if (parse(Ctx, RSD, RootElementListNode) || validate(Ctx, RSD)) {
      return RSDMap;
    }

    RSDMap.insert(std::make_pair(F, RSD));
  }

  return RSDMap;
}

AnalysisKey RootSignatureAnalysis::Key;

SmallDenseMap<const Function *, mcdxbc::RootSignatureDesc>
RootSignatureAnalysis::run(Module &M, ModuleAnalysisManager &AM) {
  return analyzeModule(M);
}

//===----------------------------------------------------------------------===//

PreservedAnalyses RootSignatureAnalysisPrinter::run(Module &M,
                                                    ModuleAnalysisManager &AM) {

  SmallDenseMap<const Function *, mcdxbc::RootSignatureDesc> &RSDMap =
      AM.getResult<RootSignatureAnalysis>(M);

  OS << "Root Signature Definitions"
     << "\n";
  uint8_t Space = 0;
  for (const Function &F : M) {
    auto It = RSDMap.find(&F);
    if (It == RSDMap.end())
      continue;
    const auto &RS = It->second;
    OS << "Definition for '" << F.getName() << "':\n";

    // start root signature header
    Space++;
    OS << indent(Space) << "Flags: " << format_hex(RS.Flags, 8) << "\n";
    OS << indent(Space) << "Version: " << RS.Version << "\n";
    OS << indent(Space) << "RootParametersOffset: " << RS.RootParameterOffset
       << "\n";
    OS << indent(Space) << "NumParameters: " << RS.ParametersContainer.size()
       << "\n";
    Space++;
    for (size_t I = 0; I < RS.ParametersContainer.size(); I++) {
      const auto &[Type, Loc] =
          RS.ParametersContainer.getTypeAndLocForParameter(I);
      const dxbc::RTS0::v1::RootParameterHeader Header =
          RS.ParametersContainer.getHeader(I);

      OS << indent(Space) << "- Parameter Type: " << Type << "\n";
      OS << indent(Space + 2)
         << "Shader Visibility: " << Header.ShaderVisibility << "\n";

      switch (Type) {
      case llvm::to_underlying(dxbc::RootParameterType::Constants32Bit): {
        const dxbc::RTS0::v1::RootConstants &Constants =
            RS.ParametersContainer.getConstant(Loc);
        OS << indent(Space + 2) << "Register Space: " << Constants.RegisterSpace
           << "\n";
        OS << indent(Space + 2)
           << "Shader Register: " << Constants.ShaderRegister << "\n";
        OS << indent(Space + 2)
           << "Num 32 Bit Values: " << Constants.Num32BitValues << "\n";
        break;
      }
      case llvm::to_underlying(dxbc::RootParameterType::CBV):
      case llvm::to_underlying(dxbc::RootParameterType::UAV):
      case llvm::to_underlying(dxbc::RootParameterType::SRV): {
        const dxbc::RTS0::v2::RootDescriptor &Descriptor =
            RS.ParametersContainer.getRootDescriptor(Loc);
        OS << indent(Space + 2)
           << "Register Space: " << Descriptor.RegisterSpace << "\n";
        OS << indent(Space + 2)
           << "Shader Register: " << Descriptor.ShaderRegister << "\n";
        if (RS.Version > 1)
          OS << indent(Space + 2) << "Flags: " << Descriptor.Flags << "\n";
        break;
      }
      }
      Space--;
    }
    OS << indent(Space) << "NumStaticSamplers: " << 0 << "\n";
    OS << indent(Space) << "StaticSamplersOffset: " << RS.StaticSamplersOffset
       << "\n";

    Space--;
    // end root signature header
  }
  return PreservedAnalyses::all();
}

//===----------------------------------------------------------------------===//
bool RootSignatureAnalysisWrapper::runOnModule(Module &M) {
  FuncToRsMap = analyzeModule(M);
  return false;
}

void RootSignatureAnalysisWrapper::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<DXILMetadataAnalysisWrapperPass>();
}

char RootSignatureAnalysisWrapper::ID = 0;

INITIALIZE_PASS_BEGIN(RootSignatureAnalysisWrapper,
                      "dxil-root-signature-analysis",
                      "DXIL Root Signature Analysis", true, true)
INITIALIZE_PASS_END(RootSignatureAnalysisWrapper,
                    "dxil-root-signature-analysis",
                    "DXIL Root Signature Analysis", true, true)
