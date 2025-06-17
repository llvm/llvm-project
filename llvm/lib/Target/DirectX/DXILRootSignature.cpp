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
#include <cmath>
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

// Template function to get formatted type string based on C++ type
template <typename T> std::string getTypeFormatted() {
  if constexpr (std::is_same_v<T, MDString>) {
    return "string";
  } else if constexpr (std::is_same_v<T, MDNode *> ||
                       std::is_same_v<T, const MDNode *>) {
    return "metadata";
  } else if constexpr (std::is_same_v<T, ConstantAsMetadata *> ||
                       std::is_same_v<T, const ConstantAsMetadata *>) {
    return "constant";
  } else if constexpr (std::is_same_v<T, ConstantAsMetadata>) {
    return "constant";
  } else if constexpr (std::is_same_v<T, ConstantInt *> ||
                       std::is_same_v<T, const ConstantInt *>) {
    return "constant int";
  } else if constexpr (std::is_same_v<T, ConstantInt>) {
    return "constant int";
  }
  return "unknown";
}

// Helper function to get the actual type of a metadata operand
std::string getActualMDType(const MDNode *Node, unsigned Index) {
  if (!Node || Index >= Node->getNumOperands())
    return "null";

  Metadata *Op = Node->getOperand(Index);
  if (!Op)
    return "null";

  if (isa<MDString>(Op))
    return getTypeFormatted<MDString>();

  if (isa<ConstantAsMetadata>(Op)) {
    if (auto *CAM = dyn_cast<ConstantAsMetadata>(Op)) {
      Type *T = CAM->getValue()->getType();
      if (T->isIntegerTy())
        return (Twine("i") + Twine(T->getIntegerBitWidth())).str();
      if (T->isFloatingPointTy())
        return T->isFloatTy()    ? getTypeFormatted<float>()
               : T->isDoubleTy() ? getTypeFormatted<double>()
                                 : "fp";

      return getTypeFormatted<ConstantAsMetadata>();
    }
  }
  if (isa<MDNode>(Op))
    return getTypeFormatted<MDNode *>();

  return "unknown";
}

// Helper function to simplify error reporting for invalid metadata values
template <typename ET>
auto reportInvalidTypeError(LLVMContext *Ctx, Twine ParamName,
                            const MDNode *Node, unsigned Index) {
  std::string ExpectedType = getTypeFormatted<ET>();
  std::string ActualType = getActualMDType(Node, Index);

  return reportError(Ctx, "Root Signature Node: " + ParamName +
                              " expected metadata node of type " +
                              ExpectedType + " at index " + Twine(Index) +
                              " but got " + ActualType);
}

static std::optional<uint32_t> extractMdIntValue(MDNode *Node,
                                                 unsigned int OpId) {
  if (auto *CI =
          mdconst::dyn_extract<ConstantInt>(Node->getOperand(OpId).get()))
    return CI->getZExtValue();
  return std::nullopt;
}

static std::optional<APFloat> extractMdFloatValue(MDNode *Node,
                                                  unsigned int OpId) {
  if (auto *CI = mdconst::dyn_extract<ConstantFP>(Node->getOperand(OpId).get()))
    return CI->getValue();
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
  bool HasError = false;
  if (std::optional<uint32_t> Val = extractMdIntValue(RootFlagNode, 1))
    RSD.Flags = *Val;
  else
    HasError = reportInvalidTypeError<ConstantInt>(Ctx, "RootFlagNode",
                                                   RootFlagNode, 1) ||
               HasError;

  return HasError;
}

static bool parseRootConstants(LLVMContext *Ctx, mcdxbc::RootSignatureDesc &RSD,
                               MDNode *RootConstantNode) {

  if (RootConstantNode->getNumOperands() != 5)
    return reportError(Ctx, "Invalid format for RootConstants Element");

  bool HasError = false;
  dxbc::RTS0::v1::RootParameterHeader Header;
  // The parameter offset doesn't matter here - we recalculate it during
  // serialization  Header.ParameterOffset = 0;
  Header.ParameterType =
      llvm::to_underlying(dxbc::RootParameterType::Constants32Bit);

  if (std::optional<uint32_t> Val = extractMdIntValue(RootConstantNode, 1))
    Header.ShaderVisibility = *Val;
  else
    HasError = reportInvalidTypeError<ConstantInt>(Ctx, "RootConstantNode",
                                                   RootConstantNode, 1) ||
               HasError;

  dxbc::RTS0::v1::RootConstants Constants;
  if (std::optional<uint32_t> Val = extractMdIntValue(RootConstantNode, 2))
    Constants.ShaderRegister = *Val;
  else
    HasError = reportInvalidTypeError<ConstantInt>(Ctx, "RootConstantNode",
                                                   RootConstantNode, 2) ||
               HasError;

  if (std::optional<uint32_t> Val = extractMdIntValue(RootConstantNode, 3))
    Constants.RegisterSpace = *Val;
  else
    HasError = reportInvalidTypeError<ConstantInt>(Ctx, "RootConstantNode",
                                                   RootConstantNode, 3) ||
               HasError;

  if (std::optional<uint32_t> Val = extractMdIntValue(RootConstantNode, 4))
    Constants.Num32BitValues = *Val;
  else
    HasError = reportInvalidTypeError<ConstantInt>(Ctx, "RootConstantNode",
                                                   RootConstantNode, 4) ||
               HasError;
  if (!HasError)
    RSD.ParametersContainer.addParameter(Header, Constants);

  return HasError;
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

  bool HasError = false;
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
    HasError = reportInvalidTypeError<ConstantInt>(Ctx, "RootDescriptorNode",
                                                   RootDescriptorNode, 1) ||
               HasError;

  dxbc::RTS0::v2::RootDescriptor Descriptor;
  if (std::optional<uint32_t> Val = extractMdIntValue(RootDescriptorNode, 2))
    Descriptor.ShaderRegister = *Val;
  else
    HasError = reportInvalidTypeError<ConstantInt>(Ctx, "RootDescriptorNode",
                                                   RootDescriptorNode, 2) ||
               HasError;

  if (std::optional<uint32_t> Val = extractMdIntValue(RootDescriptorNode, 3))
    Descriptor.RegisterSpace = *Val;
  else
    HasError = reportInvalidTypeError<ConstantInt>(Ctx, "RootDescriptorNode",
                                                   RootDescriptorNode, 3) ||
               HasError;

  if (RSD.Version == 1) {
    if (!HasError)
      RSD.ParametersContainer.addParameter(Header, Descriptor);
    return HasError;
  }
  assert(RSD.Version > 1);

  if (std::optional<uint32_t> Val = extractMdIntValue(RootDescriptorNode, 4))
    Descriptor.Flags = *Val;
  else
    HasError = reportInvalidTypeError<ConstantInt>(Ctx, "RootDescriptorNode",
                                                   RootDescriptorNode, 4) ||
               HasError;
  if (!HasError)
    RSD.ParametersContainer.addParameter(Header, Descriptor);
  return HasError;
}

static bool parseDescriptorRange(LLVMContext *Ctx,
                                 mcdxbc::RootSignatureDesc &RSD,
                                 mcdxbc::DescriptorTable &Table,
                                 MDNode *RangeDescriptorNode) {

  if (RangeDescriptorNode->getNumOperands() != 6)
    return reportError(Ctx, "Invalid format for Descriptor Range");

  bool HasError = false;
  dxbc::RTS0::v2::DescriptorRange Range;

  std::optional<StringRef> ElementText =
      extractMdStringValue(RangeDescriptorNode, 0);

  if (!ElementText.has_value())
    HasError = reportInvalidTypeError<MDString>(Ctx, "RangeDescriptorNode",
                                                RangeDescriptorNode, 0) ||
               HasError;

  Range.RangeType =
      StringSwitch<uint32_t>(*ElementText)
          .Case("CBV", llvm::to_underlying(dxbc::DescriptorRangeType::CBV))
          .Case("SRV", llvm::to_underlying(dxbc::DescriptorRangeType::SRV))
          .Case("UAV", llvm::to_underlying(dxbc::DescriptorRangeType::UAV))
          .Case("Sampler",
                llvm::to_underlying(dxbc::DescriptorRangeType::Sampler))
          .Default(-1u);

  if (Range.RangeType == -1u)
    HasError =
        reportError(Ctx, "Invalid Descriptor Range type: " + *ElementText) ||
        HasError;

  if (std::optional<uint32_t> Val = extractMdIntValue(RangeDescriptorNode, 1))
    Range.NumDescriptors = *Val;
  else
    HasError = reportInvalidTypeError<ConstantInt>(Ctx, "RangeDescriptorNode",
                                                   RangeDescriptorNode, 1) ||
               HasError;

  if (std::optional<uint32_t> Val = extractMdIntValue(RangeDescriptorNode, 2))
    Range.BaseShaderRegister = *Val;
  else
    HasError = reportInvalidTypeError<ConstantInt>(Ctx, "RangeDescriptorNode",
                                                   RangeDescriptorNode, 2) ||
               HasError;

  if (std::optional<uint32_t> Val = extractMdIntValue(RangeDescriptorNode, 3))
    Range.RegisterSpace = *Val;
  else
    HasError = reportInvalidTypeError<ConstantInt>(Ctx, "RangeDescriptorNode",
                                                   RangeDescriptorNode, 3) ||
               HasError;

  if (std::optional<uint32_t> Val = extractMdIntValue(RangeDescriptorNode, 4))
    Range.OffsetInDescriptorsFromTableStart = *Val;
  else
    HasError = reportInvalidTypeError<ConstantInt>(Ctx, "RangeDescriptorNode",
                                                   RangeDescriptorNode, 4) ||
               HasError;

  if (std::optional<uint32_t> Val = extractMdIntValue(RangeDescriptorNode, 5))
    Range.Flags = *Val;
  else
    HasError = reportInvalidTypeError<ConstantInt>(Ctx, "RangeDescriptorNode",
                                                   RangeDescriptorNode, 5) ||
               HasError;
  if (!HasError)
    Table.Ranges.push_back(Range);
  return HasError;
}

static bool parseDescriptorTable(LLVMContext *Ctx,
                                 mcdxbc::RootSignatureDesc &RSD,
                                 MDNode *DescriptorTableNode) {
  const unsigned int NumOperands = DescriptorTableNode->getNumOperands();
  if (NumOperands < 2)
    return reportError(Ctx, "Invalid format for Descriptor Table");
  bool HasError = false;
  dxbc::RTS0::v1::RootParameterHeader Header;
  if (std::optional<uint32_t> Val = extractMdIntValue(DescriptorTableNode, 1))
    Header.ShaderVisibility = *Val;
  else
    HasError = reportInvalidTypeError<ConstantInt>(Ctx, "DescriptorTableNode",
                                                   DescriptorTableNode, 1) ||
               HasError;

  mcdxbc::DescriptorTable Table;
  Header.ParameterType =
      llvm::to_underlying(dxbc::RootParameterType::DescriptorTable);

  for (unsigned int I = 2; I < NumOperands; I++) {
    MDNode *Element = dyn_cast<MDNode>(DescriptorTableNode->getOperand(I));
    if (Element == nullptr)
      HasError = reportInvalidTypeError<MDNode>(Ctx, "DescriptorTableNode",
                                                DescriptorTableNode, I) ||
                 HasError;

    if (parseDescriptorRange(Ctx, RSD, Table, Element))
      HasError = true || HasError;
  }
  if (!HasError)
    RSD.ParametersContainer.addParameter(Header, Table);
  return HasError;
}

static bool parseStaticSampler(LLVMContext *Ctx, mcdxbc::RootSignatureDesc &RSD,
                               MDNode *StaticSamplerNode) {
  if (StaticSamplerNode->getNumOperands() != 14)
    return reportError(Ctx, "Invalid format for Static Sampler");

  bool HasError = false;
  dxbc::RTS0::v1::StaticSampler Sampler;
  if (std::optional<uint32_t> Val = extractMdIntValue(StaticSamplerNode, 1))
    Sampler.Filter = *Val;
  else
    HasError = reportInvalidTypeError<ConstantInt>(Ctx, "StaticSamplerNode",
                                                   StaticSamplerNode, 1) ||
               HasError;

  if (std::optional<uint32_t> Val = extractMdIntValue(StaticSamplerNode, 2))
    Sampler.AddressU = *Val;
  else
    HasError = reportInvalidTypeError<ConstantInt>(Ctx, "StaticSamplerNode",
                                                   StaticSamplerNode, 2) ||
               HasError;

  if (std::optional<uint32_t> Val = extractMdIntValue(StaticSamplerNode, 3))
    Sampler.AddressV = *Val;
  else
    HasError = reportInvalidTypeError<ConstantInt>(Ctx, "StaticSamplerNode",
                                                   StaticSamplerNode, 3) ||
               HasError;

  if (std::optional<uint32_t> Val = extractMdIntValue(StaticSamplerNode, 4))
    Sampler.AddressW = *Val;
  else
    HasError = reportInvalidTypeError<ConstantInt>(Ctx, "StaticSamplerNode",
                                                   StaticSamplerNode, 4) ||
               HasError;

  if (std::optional<APFloat> Val = extractMdFloatValue(StaticSamplerNode, 5))
    Sampler.MipLODBias = Val->convertToFloat();
  else
    HasError = reportInvalidTypeError<ConstantInt>(Ctx, "StaticSamplerNode",
                                                   StaticSamplerNode, 5) ||
               HasError;

  if (std::optional<uint32_t> Val = extractMdIntValue(StaticSamplerNode, 6))
    Sampler.MaxAnisotropy = *Val;
  else
    HasError = reportInvalidTypeError<ConstantInt>(Ctx, "StaticSamplerNode",
                                                   StaticSamplerNode, 6) ||
               HasError;

  if (std::optional<uint32_t> Val = extractMdIntValue(StaticSamplerNode, 7))
    Sampler.ComparisonFunc = *Val;
  else
    HasError = reportInvalidTypeError<ConstantInt>(Ctx, "StaticSamplerNode",
                                                   StaticSamplerNode, 7) ||
               HasError;

  if (std::optional<uint32_t> Val = extractMdIntValue(StaticSamplerNode, 8))
    Sampler.BorderColor = *Val;
  else
    HasError = reportInvalidTypeError<ConstantInt>(Ctx, "StaticSamplerNode",
                                                   StaticSamplerNode, 8) ||
               HasError;

  if (std::optional<APFloat> Val = extractMdFloatValue(StaticSamplerNode, 9))
    Sampler.MinLOD = Val->convertToFloat();
  else
    HasError = reportInvalidTypeError<ConstantInt>(Ctx, "StaticSamplerNode",
                                                   StaticSamplerNode, 9) ||
               HasError;

  if (std::optional<APFloat> Val = extractMdFloatValue(StaticSamplerNode, 10))
    Sampler.MaxLOD = Val->convertToFloat();
  else
    HasError = reportInvalidTypeError<ConstantInt>(Ctx, "StaticSamplerNode",
                                                   StaticSamplerNode, 10) ||
               HasError;

  if (std::optional<uint32_t> Val = extractMdIntValue(StaticSamplerNode, 11))
    Sampler.ShaderRegister = *Val;
  else
    HasError = reportInvalidTypeError<ConstantInt>(Ctx, "StaticSamplerNode",
                                                   StaticSamplerNode, 11) ||
               HasError;

  if (std::optional<uint32_t> Val = extractMdIntValue(StaticSamplerNode, 12))
    Sampler.RegisterSpace = *Val;
  else
    HasError = reportInvalidTypeError<ConstantInt>(Ctx, "StaticSamplerNode",
                                                   StaticSamplerNode, 12) ||
               HasError;

  if (std::optional<uint32_t> Val = extractMdIntValue(StaticSamplerNode, 13))
    Sampler.ShaderVisibility = *Val;
  else
    HasError = reportInvalidTypeError<ConstantInt>(Ctx, "StaticSamplerNode",
                                                   StaticSamplerNode, 13) ||
               HasError;
  if (!HasError)
    RSD.StaticSamplers.push_back(Sampler);
  return HasError;
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
          .Case("DescriptorTable", RootSignatureElementKind::DescriptorTable)
          .Case("StaticSampler", RootSignatureElementKind::StaticSamplers)
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
  case RootSignatureElementKind::DescriptorTable:
    return parseDescriptorTable(Ctx, RSD, Element);
  case RootSignatureElementKind::StaticSamplers:
    return parseStaticSampler(Ctx, RSD, Element);
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

    HasError = parseRootSignatureElement(Ctx, RSD, Element) || HasError;
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

static bool verifyRangeType(uint32_t Type) {
  switch (Type) {
  case llvm::to_underlying(dxbc::DescriptorRangeType::CBV):
  case llvm::to_underlying(dxbc::DescriptorRangeType::SRV):
  case llvm::to_underlying(dxbc::DescriptorRangeType::UAV):
  case llvm::to_underlying(dxbc::DescriptorRangeType::Sampler):
    return true;
  };

  return false;
}

template <typename... FlagTypes>
static bool isFlagSet(uint32_t Flags, FlagTypes... FlagsToCheck) {
  return ((Flags & llvm::to_underlying(FlagsToCheck)) | ...) == Flags;
}

static bool verifyDescriptorRangeFlag(uint32_t Version, uint32_t Type,
                                      uint32_t FlagsVal) {
  using FlagT = dxbc::DescriptorRangeFlag;
  FlagT Flags = FlagT(FlagsVal);

  const bool IsSampler =
      (Type == llvm::to_underlying(dxbc::DescriptorRangeType::Sampler));

  if (Version == 1) {
    if (IsSampler)
      return Flags == FlagT::NONE;
    return Flags == FlagT::DESCRIPTORS_VOLATILE;
  }

  // The data-specific flags are mutually exclusive.
  FlagT DataFlags = FlagT::DATA_VOLATILE | FlagT::DATA_STATIC |
                    FlagT::DATA_STATIC_WHILE_SET_AT_EXECUTE;

  if (popcount(llvm::to_underlying(Flags & DataFlags)) > 1)
    return false;

  // For volatile descriptors, DATA_STATIC is never valid.
  if ((Flags & FlagT::DESCRIPTORS_VOLATILE) == FlagT::DESCRIPTORS_VOLATILE) {
    FlagT Mask = FlagT::DESCRIPTORS_VOLATILE;
    if (!IsSampler) {
      Mask |= FlagT::DATA_VOLATILE;
      Mask |= FlagT::DATA_STATIC_WHILE_SET_AT_EXECUTE;
    }
    return (Flags & ~Mask) == FlagT::NONE;
  }

  // For "STATIC_KEEPING_BUFFER_BOUNDS_CHECKS" descriptors,
  // the other data-specific flags may all be set.
  if ((Flags & FlagT::DESCRIPTORS_STATIC_KEEPING_BUFFER_BOUNDS_CHECKS) ==
      FlagT::DESCRIPTORS_STATIC_KEEPING_BUFFER_BOUNDS_CHECKS) {
    FlagT Mask = FlagT::DESCRIPTORS_STATIC_KEEPING_BUFFER_BOUNDS_CHECKS;
    if (!IsSampler) {
      Mask |= FlagT::DATA_VOLATILE;
      Mask |= FlagT::DATA_STATIC;
      Mask |= FlagT::DATA_STATIC_WHILE_SET_AT_EXECUTE;
    }
    return (Flags & ~Mask) == FlagT::NONE;
  }

  // When no descriptor flag is set, any data flag is allowed.
  return (Flags & ~DataFlags) == FlagT::NONE;
}

static bool verifySamplerFilter(uint32_t Filter) {
  switch (Filter) {
  case llvm::to_underlying(dxbc::StaticSamplerFilter::MIN_MAG_MIP_POINT):
  case llvm::to_underlying(dxbc::StaticSamplerFilter::MIN_MAG_POINT_MIP_LINEAR):
  case llvm::to_underlying(
      dxbc::StaticSamplerFilter::MIN_POINT_MAG_LINEAR_MIP_POINT):
  case llvm::to_underlying(dxbc::StaticSamplerFilter::MIN_POINT_MAG_MIP_LINEAR):
  case llvm::to_underlying(dxbc::StaticSamplerFilter::MIN_LINEAR_MAG_MIP_POINT):
  case llvm::to_underlying(
      dxbc::StaticSamplerFilter::MIN_LINEAR_MAG_POINT_MIP_LINEAR):
  case llvm::to_underlying(dxbc::StaticSamplerFilter::MIN_MAG_LINEAR_MIP_POINT):
  case llvm::to_underlying(dxbc::StaticSamplerFilter::MIN_MAG_MIP_LINEAR):
  case llvm::to_underlying(dxbc::StaticSamplerFilter::ANISOTROPIC):
  case llvm::to_underlying(
      dxbc::StaticSamplerFilter::COMPARISON_MIN_MAG_MIP_POINT):
  case llvm::to_underlying(
      dxbc::StaticSamplerFilter::COMPARISON_MIN_MAG_POINT_MIP_LINEAR):
  case llvm::to_underlying(
      dxbc::StaticSamplerFilter::COMPARISON_MIN_POINT_MAG_LINEAR_MIP_POINT):
  case llvm::to_underlying(
      dxbc::StaticSamplerFilter::COMPARISON_MIN_POINT_MAG_MIP_LINEAR):
  case llvm::to_underlying(
      dxbc::StaticSamplerFilter::COMPARISON_MIN_LINEAR_MAG_MIP_POINT):
  case llvm::to_underlying(
      dxbc::StaticSamplerFilter::COMPARISON_MIN_LINEAR_MAG_POINT_MIP_LINEAR):
  case llvm::to_underlying(
      dxbc::StaticSamplerFilter::COMPARISON_MIN_MAG_LINEAR_MIP_POINT):
  case llvm::to_underlying(
      dxbc::StaticSamplerFilter::COMPARISON_MIN_MAG_MIP_LINEAR):
  case llvm::to_underlying(dxbc::StaticSamplerFilter::COMPARISON_ANISOTROPIC):
  case llvm::to_underlying(
      dxbc::StaticSamplerFilter::MINIMUM_MIN_MAG_MIP_POINT):
  case llvm::to_underlying(
      dxbc::StaticSamplerFilter::MINIMUM_MIN_MAG_POINT_MIP_LINEAR):
  case llvm::to_underlying(
      dxbc::StaticSamplerFilter::MINIMUM_MIN_POINT_MAG_LINEAR_MIP_POINT):
  case llvm::to_underlying(
      dxbc::StaticSamplerFilter::MINIMUM_MIN_POINT_MAG_MIP_LINEAR):
  case llvm::to_underlying(
      dxbc::StaticSamplerFilter::MINIMUM_MIN_LINEAR_MAG_MIP_POINT):
  case llvm::to_underlying(
      dxbc::StaticSamplerFilter::MINIMUM_MIN_LINEAR_MAG_POINT_MIP_LINEAR):
  case llvm::to_underlying(
      dxbc::StaticSamplerFilter::MINIMUM_MIN_MAG_LINEAR_MIP_POINT):
  case llvm::to_underlying(
      dxbc::StaticSamplerFilter::MINIMUM_MIN_MAG_MIP_LINEAR):
  case llvm::to_underlying(dxbc::StaticSamplerFilter::MINIMUM_ANISOTROPIC):
  case llvm::to_underlying(
      dxbc::StaticSamplerFilter::MAXIMUM_MIN_MAG_MIP_POINT):
  case llvm::to_underlying(
      dxbc::StaticSamplerFilter::MAXIMUM_MIN_MAG_POINT_MIP_LINEAR):
  case llvm::to_underlying(
      dxbc::StaticSamplerFilter::MAXIMUM_MIN_POINT_MAG_LINEAR_MIP_POINT):
  case llvm::to_underlying(
      dxbc::StaticSamplerFilter::MAXIMUM_MIN_POINT_MAG_MIP_LINEAR):
  case llvm::to_underlying(
      dxbc::StaticSamplerFilter::MAXIMUM_MIN_LINEAR_MAG_MIP_POINT):
  case llvm::to_underlying(
      dxbc::StaticSamplerFilter::MAXIMUM_MIN_LINEAR_MAG_POINT_MIP_LINEAR):
  case llvm::to_underlying(
      dxbc::StaticSamplerFilter::MAXIMUM_MIN_MAG_LINEAR_MIP_POINT):
  case llvm::to_underlying(
      dxbc::StaticSamplerFilter::MAXIMUM_MIN_MAG_MIP_LINEAR):
  case llvm::to_underlying(dxbc::StaticSamplerFilter::MAXIMUM_ANISOTROPIC):
    return true;
  }
  return false;
}

// Values allowed here:
// https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_texture_address_mode#syntax
static bool verifyAddress(uint32_t Address) {
  switch (Address) {
  case llvm::to_underlying(dxbc::TextureAddressMode::Border):
  case llvm::to_underlying(dxbc::TextureAddressMode::Clamp):
  case llvm::to_underlying(dxbc::TextureAddressMode::Mirror):
  case llvm::to_underlying(dxbc::TextureAddressMode::MirrorOnce):
  case llvm::to_underlying(dxbc::TextureAddressMode::Wrap):
    return true;
  }

  return false;
}

static bool verifyMipLODBias(float MipLODBias) {
  return MipLODBias >= -16.f && MipLODBias <= 16.f;
}

static bool verifyMaxAnisotropy(uint32_t MaxAnisotropy) {
  return MaxAnisotropy <= 16u;
}

static bool verifyComparisonFunc(uint32_t ComparisonFunc) {
  switch (ComparisonFunc) {
  case llvm::to_underlying(dxbc::SamplersComparisonFunction::Never):
  case llvm::to_underlying(dxbc::SamplersComparisonFunction::Less):
  case llvm::to_underlying(dxbc::SamplersComparisonFunction::Equal):
  case llvm::to_underlying(dxbc::SamplersComparisonFunction::LessEqual):
  case llvm::to_underlying(dxbc::SamplersComparisonFunction::Greater):
  case llvm::to_underlying(dxbc::SamplersComparisonFunction::NotEqual):
  case llvm::to_underlying(dxbc::SamplersComparisonFunction::GreaterEqual):
  case llvm::to_underlying(dxbc::SamplersComparisonFunction::Always):
    return true;
  }
  return false;
}

static bool verifyBorderColor(uint32_t BorderColor) {
  switch (BorderColor) {
  case llvm::to_underlying(dxbc::SamplersBorderColor::TransparentBlack):
  case llvm::to_underlying(dxbc::SamplersBorderColor::OpaqueBlack):
  case llvm::to_underlying(dxbc::SamplersBorderColor::OpaqueWhite):
  case llvm::to_underlying(dxbc::SamplersBorderColor::OpaqueBlackUint):
  case llvm::to_underlying(dxbc::SamplersBorderColor::OpaqueWhiteUint):
    return true;
  }
  return false;
}

static bool verifyLOD(float LOD) { return !std::isnan(LOD); }

static bool validate(LLVMContext *Ctx, const mcdxbc::RootSignatureDesc &RSD) {
  bool HasError = false;
  if (!verifyVersion(RSD.Version)) {
    HasError = reportValueError(Ctx, "Version", RSD.Version) || HasError;
  }

  if (!verifyRootFlag(RSD.Flags)) {
    HasError = reportValueError(Ctx, "RootFlags", RSD.Flags) || HasError;
  }

  for (const mcdxbc::RootParameterInfo &Info : RSD.ParametersContainer) {
    if (!dxbc::isValidShaderVisibility(Info.Header.ShaderVisibility))
      HasError = reportValueError(Ctx, "ShaderVisibility",
                                  Info.Header.ShaderVisibility) ||
                 HasError;

    assert(dxbc::isValidParameterType(Info.Header.ParameterType) &&
           "Invalid value for ParameterType");

    switch (Info.Header.ParameterType) {

    case llvm::to_underlying(dxbc::RootParameterType::CBV):
    case llvm::to_underlying(dxbc::RootParameterType::UAV):
    case llvm::to_underlying(dxbc::RootParameterType::SRV): {
      const dxbc::RTS0::v2::RootDescriptor &Descriptor =
          RSD.ParametersContainer.getRootDescriptor(Info.Location);
      if (!verifyRegisterValue(Descriptor.ShaderRegister))
        HasError = reportValueError(Ctx, "ShaderRegister",
                                    Descriptor.ShaderRegister) ||
                   HasError;

      if (!verifyRegisterSpace(Descriptor.RegisterSpace))
        HasError =
            reportValueError(Ctx, "RegisterSpace", Descriptor.RegisterSpace) ||
            HasError;

      if (RSD.Version > 1) {
        if (!verifyDescriptorFlag(Descriptor.Flags))
          HasError =
              reportValueError(Ctx, "DescriptorFlag", Descriptor.Flags) ||
              HasError;
      }
      break;
    }
    case llvm::to_underlying(dxbc::RootParameterType::DescriptorTable): {
      const mcdxbc::DescriptorTable &Table =
          RSD.ParametersContainer.getDescriptorTable(Info.Location);
      for (const dxbc::RTS0::v2::DescriptorRange &Range : Table) {
        if (!verifyRangeType(Range.RangeType))
          HasError =
              reportValueError(Ctx, "RangeType", Range.RangeType) || HasError;

        if (!verifyRegisterSpace(Range.RegisterSpace))
          HasError =
              reportValueError(Ctx, "RegisterSpace", Range.RegisterSpace) ||
              HasError;

        if (!verifyDescriptorRangeFlag(RSD.Version, Range.RangeType,
                                       Range.Flags))
          HasError =
              reportValueError(Ctx, "DescriptorFlag", Range.Flags) || HasError;
      }
      break;
    }
    }
  }

  for (const dxbc::RTS0::v1::StaticSampler &Sampler : RSD.StaticSamplers) {
    if (!verifySamplerFilter(Sampler.Filter))
      HasError = reportValueError(Ctx, "Filter", Sampler.Filter) || HasError;

    if (!verifyAddress(Sampler.AddressU))
      HasError =
          reportValueError(Ctx, "AddressU", Sampler.AddressU) || HasError;

    if (!verifyAddress(Sampler.AddressV))
      HasError =
          reportValueError(Ctx, "AddressV", Sampler.AddressV) || HasError;

    if (!verifyAddress(Sampler.AddressW))
      HasError =
          reportValueError(Ctx, "AddressW", Sampler.AddressW) || HasError;

    if (!verifyMipLODBias(Sampler.MipLODBias))
      HasError =
          reportValueError(Ctx, "MipLODBias", Sampler.MipLODBias) || HasError;

    if (!verifyMaxAnisotropy(Sampler.MaxAnisotropy))
      HasError =
          reportValueError(Ctx, "MaxAnisotropy", Sampler.MaxAnisotropy) ||
          HasError;

    if (!verifyComparisonFunc(Sampler.ComparisonFunc))
      HasError =
          reportValueError(Ctx, "ComparisonFunc", Sampler.ComparisonFunc) ||
          HasError;

    if (!verifyBorderColor(Sampler.BorderColor))
      HasError =
          reportValueError(Ctx, "BorderColor", Sampler.BorderColor) || HasError;

    if (!verifyLOD(Sampler.MinLOD))
      HasError = reportValueError(Ctx, "MinLOD", Sampler.MinLOD) || HasError;

    if (!verifyLOD(Sampler.MaxLOD))
      HasError = reportValueError(Ctx, "MaxLOD", Sampler.MaxLOD) || HasError;

    if (!verifyRegisterValue(Sampler.ShaderRegister))
      HasError =
          reportValueError(Ctx, "ShaderRegister", Sampler.ShaderRegister) ||
          HasError;

    if (!verifyRegisterSpace(Sampler.RegisterSpace))
      HasError =
          reportValueError(Ctx, "RegisterSpace", Sampler.RegisterSpace) ||
          HasError;

    if (!dxbc::isValidShaderVisibility(Sampler.ShaderVisibility))
      HasError =
          reportValueError(Ctx, "ShaderVisibility", Sampler.ShaderVisibility) ||
          HasError;
  }

  return HasError;
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

    // static sampler offset is calculated when writting dxcontainer.
    RSD.StaticSamplersOffset = 0u;

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
  for (const Function &F : M) {
    auto It = RSDMap.find(&F);
    if (It == RSDMap.end())
      continue;
    const auto &RS = It->second;
    OS << "Definition for '" << F.getName() << "':\n";

    // start root signature header
    OS << "Flags: " << format_hex(RS.Flags, 8) << "\n";
    OS << "Version: " << RS.Version << "\n";
    OS << "RootParametersOffset: " << RS.RootParameterOffset << "\n";
    OS << "NumParameters: " << RS.ParametersContainer.size() << "\n";
    for (size_t I = 0; I < RS.ParametersContainer.size(); I++) {
      const auto &[Type, Loc] =
          RS.ParametersContainer.getTypeAndLocForParameter(I);
      const dxbc::RTS0::v1::RootParameterHeader Header =
          RS.ParametersContainer.getHeader(I);

      OS << "- Parameter Type: " << Type << "\n";
      OS << indent(2) << "Shader Visibility: " << Header.ShaderVisibility
         << "\n";

      switch (Type) {
      case llvm::to_underlying(dxbc::RootParameterType::Constants32Bit): {
        const dxbc::RTS0::v1::RootConstants &Constants =
            RS.ParametersContainer.getConstant(Loc);
        OS << indent(2) << "Register Space: " << Constants.RegisterSpace
           << "\n";
        OS << indent(2) << "Shader Register: " << Constants.ShaderRegister
           << "\n";
        OS << indent(2) << "Num 32 Bit Values: " << Constants.Num32BitValues
           << "\n";
        break;
      }
      case llvm::to_underlying(dxbc::RootParameterType::CBV):
      case llvm::to_underlying(dxbc::RootParameterType::UAV):
      case llvm::to_underlying(dxbc::RootParameterType::SRV): {
        const dxbc::RTS0::v2::RootDescriptor &Descriptor =
            RS.ParametersContainer.getRootDescriptor(Loc);
        OS << indent(2) << "Register Space: " << Descriptor.RegisterSpace
           << "\n";
        OS << indent(2) << "Shader Register: " << Descriptor.ShaderRegister
           << "\n";
        if (RS.Version > 1)
          OS << indent(2) << "Flags: " << Descriptor.Flags << "\n";
        break;
      }
      case llvm::to_underlying(dxbc::RootParameterType::DescriptorTable): {
        const mcdxbc::DescriptorTable &Table =
            RS.ParametersContainer.getDescriptorTable(Loc);
        OS << indent(2) << "NumRanges: " << Table.Ranges.size() << "\n";

        for (const dxbc::RTS0::v2::DescriptorRange Range : Table) {
          OS << indent(2) << "- Range Type: " << Range.RangeType << "\n";
          OS << indent(4) << "Register Space: " << Range.RegisterSpace << "\n";
          OS << indent(4)
             << "Base Shader Register: " << Range.BaseShaderRegister << "\n";
          OS << indent(4) << "Num Descriptors: " << Range.NumDescriptors
             << "\n";
          OS << indent(4) << "Offset In Descriptors From Table Start: "
             << Range.OffsetInDescriptorsFromTableStart << "\n";
          if (RS.Version > 1)
            OS << indent(4) << "Flags: " << Range.Flags << "\n";
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
