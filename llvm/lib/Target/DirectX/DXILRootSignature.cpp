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
#include "llvm/Frontend/HLSL/RootSignatureMetadata.h"
#include "llvm/Frontend/HLSL/RootSignatureValidations.h"
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

using namespace llvm;
using namespace llvm::dxil;

static std::optional<uint32_t> extractMdIntValue(MDNode *Node,
                                                 unsigned int OpId) {
  if (auto *CI =
          mdconst::dyn_extract<ConstantInt>(Node->getOperand(OpId).get()))
    return CI->getZExtValue();
  return std::nullopt;
}

static bool reportError(LLVMContext *Ctx, Twine Message,
                        DiagnosticSeverity Severity = DS_Error) {
  Ctx->diagnose(DiagnosticInfoGeneric(Message, Severity));
  return true;
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
    if (RSDefNode->getNumOperands() != 3) {
      reportError(Ctx, "Invalid Root Signature metadata - expected function, "
                       "signature, and version.");
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
    if (std::optional<uint32_t> Version = extractMdIntValue(RSDefNode, 2))
      RSD.Version = *Version;
    else {
      reportError(Ctx, "Invalid RSDefNode value, expected constant int");
      continue;
    }

    // Clang emits the root signature data in dxcontainer following a specific
    // sequence. First the header, then the root parameters. So the header
    // offset will always equal to the header size.
    RSD.RootParameterOffset = sizeof(dxbc::RTS0::v1::RootSignatureHeader);

    // static sampler offset is calculated when writting dxcontainer.
    RSD.StaticSamplersOffset = 0u;

    hlsl::rootsig::MetadataParser MDParser(RootElementListNode);

    if (MDParser.ParseRootSignature(Ctx, RSD)) {
      return RSDMap;
    }

    RSDMap.insert(std::make_pair(F, RSD));
  }

  return RSDMap;
}

AnalysisKey RootSignatureAnalysis::Key;

RootSignatureAnalysis::Result
RootSignatureAnalysis::run(Module &M, ModuleAnalysisManager &AM) {
  return RootSignatureBindingInfo(analyzeModule(M));
}

//===----------------------------------------------------------------------===//

PreservedAnalyses RootSignatureAnalysisPrinter::run(Module &M,
                                                    ModuleAnalysisManager &AM) {

  RootSignatureBindingInfo &RSDMap = AM.getResult<RootSignatureAnalysis>(M);

  OS << "Root Signature Definitions"
     << "\n";
  for (const Function &F : M) {
    auto It = RSDMap.find(&F);
    if (It == RSDMap.end())
      continue;
    const auto &RS = It->second;
    OS << "Definition for '" << F.getName() << "':\n";
    // start root signature header
    OS << "Flags: " << format_hex(RS.Flags, 8) << "\n"
       << "Version: " << RS.Version << "\n"
       << "RootParametersOffset: " << RS.RootParameterOffset << "\n"
       << "NumParameters: " << RS.ParametersContainer.size() << "\n";
    for (size_t I = 0; I < RS.ParametersContainer.size(); I++) {
      const auto &[Type, Loc] =
          RS.ParametersContainer.getTypeAndLocForParameter(I);
      const dxbc::RTS0::v1::RootParameterHeader Header =
          RS.ParametersContainer.getHeader(I);

      OS << "- Parameter Type: " << Type << "\n"
         << "  Shader Visibility: " << Header.ShaderVisibility << "\n";

      switch (Type) {
      case llvm::to_underlying(dxbc::RootParameterType::Constants32Bit): {
        const dxbc::RTS0::v1::RootConstants &Constants =
            RS.ParametersContainer.getConstant(Loc);
        OS << "  Register Space: " << Constants.RegisterSpace << "\n"
           << "  Shader Register: " << Constants.ShaderRegister << "\n"
           << "  Num 32 Bit Values: " << Constants.Num32BitValues << "\n";
        break;
      }
      case llvm::to_underlying(dxbc::RootParameterType::CBV):
      case llvm::to_underlying(dxbc::RootParameterType::UAV):
      case llvm::to_underlying(dxbc::RootParameterType::SRV): {
        const dxbc::RTS0::v2::RootDescriptor &Descriptor =
            RS.ParametersContainer.getRootDescriptor(Loc);
        OS << "  Register Space: " << Descriptor.RegisterSpace << "\n"
           << "  Shader Register: " << Descriptor.ShaderRegister << "\n";
        if (RS.Version > 1)
          OS << "  Flags: " << Descriptor.Flags << "\n";
        break;
      }
      case llvm::to_underlying(dxbc::RootParameterType::DescriptorTable): {
        const mcdxbc::DescriptorTable &Table =
            RS.ParametersContainer.getDescriptorTable(Loc);
        OS << "  NumRanges: " << Table.Ranges.size() << "\n";

        for (const dxbc::RTS0::v2::DescriptorRange Range : Table) {
          OS << "  - Range Type: " << Range.RangeType << "\n"
             << "    Register Space: " << Range.RegisterSpace << "\n"
             << "    Base Shader Register: " << Range.BaseShaderRegister << "\n"
             << "    Num Descriptors: " << Range.NumDescriptors << "\n"
             << "    Offset In Descriptors From Table Start: "
             << Range.OffsetInDescriptorsFromTableStart << "\n";
          if (RS.Version > 1)
            OS << "    Flags: " << Range.Flags << "\n";
        }
        break;
      }
      }
    }
    OS << "NumStaticSamplers: " << 0 << "\n";
    OS << "StaticSamplersOffset: " << RS.StaticSamplersOffset << "\n";
  }
  return PreservedAnalyses::all();
}

//===----------------------------------------------------------------------===//
bool RootSignatureAnalysisWrapper::runOnModule(Module &M) {
  FuncToRsMap = std::make_unique<RootSignatureBindingInfo>(
      RootSignatureBindingInfo(analyzeModule(M)));
  return false;
}

void RootSignatureAnalysisWrapper::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addPreserved<DXILMetadataAnalysisWrapperPass>();
}

char RootSignatureAnalysisWrapper::ID = 0;

INITIALIZE_PASS_BEGIN(RootSignatureAnalysisWrapper,
                      "dxil-root-signature-analysis",
                      "DXIL Root Signature Analysis", true, true)
INITIALIZE_PASS_END(RootSignatureAnalysisWrapper,
                    "dxil-root-signature-analysis",
                    "DXIL Root Signature Analysis", true, true)
