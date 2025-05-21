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

  dxbc::RootParameterHeader Header;
  // The parameter offset doesn't matter here - we recalculate it during
  // serialization  Header.ParameterOffset = 0;
  Header.ParameterType =
      llvm::to_underlying(dxbc::RootParameterType::Constants32Bit);

  if (std::optional<uint32_t> Val = extractMdIntValue(RootConstantNode, 1))
    Header.ShaderVisibility = *Val;
  else
    return reportError(Ctx, "Invalid value for ShaderVisibility");

  dxbc::RootConstants Constants;
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

static bool parseRootSignatureElement(LLVMContext *Ctx,
                                      mcdxbc::RootSignatureDesc &RSD,
                                      MDNode *Element) {
  MDString *ElementText = cast<MDString>(Element->getOperand(0));
  if (ElementText == nullptr)
    return reportError(Ctx, "Invalid format for Root Element");

  RootSignatureElementKind ElementKind =
      StringSwitch<RootSignatureElementKind>(ElementText->getString())
          .Case("RootFlags", RootSignatureElementKind::RootFlags)
          .Case("RootConstants", RootSignatureElementKind::RootConstants)
          .Default(RootSignatureElementKind::Error);

  switch (ElementKind) {

  case RootSignatureElementKind::RootFlags:
    return parseRootFlags(Ctx, RSD, Element);
  case RootSignatureElementKind::RootConstants:
    return parseRootConstants(Ctx, RSD, Element);
    break;
  case RootSignatureElementKind::Error:
    return reportError(Ctx, "Invalid Root Signature Element: " +
                                ElementText->getString());
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
    RSD.RootParameterOffset = sizeof(dxbc::RootSignatureHeader);

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
      const dxbc::RootParameterHeader Header =
          RS.ParametersContainer.getHeader(I);

      OS << indent(Space) << "- Parameter Type: " << Type << "\n";
      OS << indent(Space + 2)
         << "Shader Visibility: " << Header.ShaderVisibility << "\n";

      switch (Type) {
      case llvm::to_underlying(dxbc::RootParameterType::Constants32Bit): {
        const dxbc::RootConstants &Constants =
            RS.ParametersContainer.getConstant(Loc);
        OS << indent(Space + 2) << "Register Space: " << Constants.RegisterSpace
           << "\n";
        OS << indent(Space + 2)
           << "Shader Register: " << Constants.ShaderRegister << "\n";
        OS << indent(Space + 2)
           << "Num 32 Bit Values: " << Constants.Num32BitValues << "\n";
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
