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

static std::optional<uint32_t> extractMdIntValue(MDNode *Node,
                                                 unsigned int OpId) {
  if (auto *CI =
          mdconst::dyn_extract<ConstantInt>(Node->getOperand(OpId).get()))
    return CI->getZExtValue();
  return std::nullopt;
}

static std::optional<float> extractMdFloatValue(MDNode *Node,
                                                unsigned int OpId) {
  if (auto *CI = mdconst::dyn_extract<ConstantFP>(Node->getOperand(OpId).get()))
    return CI->getValueAPF().convertToFloat();
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

static bool parseDescriptorRange(LLVMContext *Ctx,
                                 mcdxbc::DescriptorTable &Table,
                                 MDNode *RangeDescriptorNode) {

  if (RangeDescriptorNode->getNumOperands() != 6)
    return reportError(Ctx, "Invalid format for Descriptor Range");

  dxbc::RTS0::v2::DescriptorRange Range;

  std::optional<StringRef> ElementText =
      extractMdStringValue(RangeDescriptorNode, 0);

  if (!ElementText.has_value())
    return reportError(Ctx, "Descriptor Range, first element is not a string.");

  Range.RangeType =
      StringSwitch<uint32_t>(*ElementText)
          .Case("CBV", llvm::to_underlying(dxbc::DescriptorRangeType::CBV))
          .Case("SRV", llvm::to_underlying(dxbc::DescriptorRangeType::SRV))
          .Case("UAV", llvm::to_underlying(dxbc::DescriptorRangeType::UAV))
          .Case("Sampler",
                llvm::to_underlying(dxbc::DescriptorRangeType::Sampler))
          .Default(~0U);

  if (Range.RangeType == ~0U)
    return reportError(Ctx, "Invalid Descriptor Range type: " + *ElementText);

  if (std::optional<uint32_t> Val = extractMdIntValue(RangeDescriptorNode, 1))
    Range.NumDescriptors = *Val;
  else
    return reportError(Ctx, "Invalid value for Number of Descriptor in Range");

  if (std::optional<uint32_t> Val = extractMdIntValue(RangeDescriptorNode, 2))
    Range.BaseShaderRegister = *Val;
  else
    return reportError(Ctx, "Invalid value for BaseShaderRegister");

  if (std::optional<uint32_t> Val = extractMdIntValue(RangeDescriptorNode, 3))
    Range.RegisterSpace = *Val;
  else
    return reportError(Ctx, "Invalid value for RegisterSpace");

  if (std::optional<uint32_t> Val = extractMdIntValue(RangeDescriptorNode, 4))
    Range.OffsetInDescriptorsFromTableStart = *Val;
  else
    return reportError(Ctx,
                       "Invalid value for OffsetInDescriptorsFromTableStart");

  if (std::optional<uint32_t> Val = extractMdIntValue(RangeDescriptorNode, 5))
    Range.Flags = *Val;
  else
    return reportError(Ctx, "Invalid value for Descriptor Range Flags");

  Table.Ranges.push_back(Range);
  return false;
}

static bool parseDescriptorTable(LLVMContext *Ctx,
                                 mcdxbc::RootSignatureDesc &RSD,
                                 MDNode *DescriptorTableNode) {
  const unsigned int NumOperands = DescriptorTableNode->getNumOperands();
  if (NumOperands < 2)
    return reportError(Ctx, "Invalid format for Descriptor Table");

  dxbc::RTS0::v1::RootParameterHeader Header;
  if (std::optional<uint32_t> Val = extractMdIntValue(DescriptorTableNode, 1))
    Header.ShaderVisibility = *Val;
  else
    return reportError(Ctx, "Invalid value for ShaderVisibility");

  mcdxbc::DescriptorTable Table;
  Header.ParameterType =
      llvm::to_underlying(dxbc::RootParameterType::DescriptorTable);

  for (unsigned int I = 2; I < NumOperands; I++) {
    MDNode *Element = dyn_cast<MDNode>(DescriptorTableNode->getOperand(I));
    if (Element == nullptr)
      return reportError(Ctx, "Missing Root Element Metadata Node.");

    if (parseDescriptorRange(Ctx, Table, Element))
      return true;
  }

  RSD.ParametersContainer.addParameter(Header, Table);
  return false;
}

static bool parseStaticSampler(LLVMContext *Ctx, mcdxbc::RootSignatureDesc &RSD,
                               MDNode *StaticSamplerNode) {
  if (StaticSamplerNode->getNumOperands() != 14)
    return reportError(Ctx, "Invalid format for Static Sampler");

  dxbc::RTS0::v1::StaticSampler Sampler;
  if (std::optional<uint32_t> Val = extractMdIntValue(StaticSamplerNode, 1))
    Sampler.Filter = *Val;
  else
    return reportError(Ctx, "Invalid value for Filter");

  if (std::optional<uint32_t> Val = extractMdIntValue(StaticSamplerNode, 2))
    Sampler.AddressU = *Val;
  else
    return reportError(Ctx, "Invalid value for AddressU");

  if (std::optional<uint32_t> Val = extractMdIntValue(StaticSamplerNode, 3))
    Sampler.AddressV = *Val;
  else
    return reportError(Ctx, "Invalid value for AddressV");

  if (std::optional<uint32_t> Val = extractMdIntValue(StaticSamplerNode, 4))
    Sampler.AddressW = *Val;
  else
    return reportError(Ctx, "Invalid value for AddressW");

  if (std::optional<float> Val = extractMdFloatValue(StaticSamplerNode, 5))
    Sampler.MipLODBias = *Val;
  else
    return reportError(Ctx, "Invalid value for MipLODBias");

  if (std::optional<uint32_t> Val = extractMdIntValue(StaticSamplerNode, 6))
    Sampler.MaxAnisotropy = *Val;
  else
    return reportError(Ctx, "Invalid value for MaxAnisotropy");

  if (std::optional<uint32_t> Val = extractMdIntValue(StaticSamplerNode, 7))
    Sampler.ComparisonFunc = *Val;
  else
    return reportError(Ctx, "Invalid value for ComparisonFunc ");

  if (std::optional<uint32_t> Val = extractMdIntValue(StaticSamplerNode, 8))
    Sampler.BorderColor = *Val;
  else
    return reportError(Ctx, "Invalid value for ComparisonFunc ");

  if (std::optional<float> Val = extractMdFloatValue(StaticSamplerNode, 9))
    Sampler.MinLOD = *Val;
  else
    return reportError(Ctx, "Invalid value for MinLOD");

  if (std::optional<float> Val = extractMdFloatValue(StaticSamplerNode, 10))
    Sampler.MaxLOD = *Val;
  else
    return reportError(Ctx, "Invalid value for MaxLOD");

  if (std::optional<uint32_t> Val = extractMdIntValue(StaticSamplerNode, 11))
    Sampler.ShaderRegister = *Val;
  else
    return reportError(Ctx, "Invalid value for ShaderRegister");

  if (std::optional<uint32_t> Val = extractMdIntValue(StaticSamplerNode, 12))
    Sampler.RegisterSpace = *Val;
  else
    return reportError(Ctx, "Invalid value for RegisterSpace");

  if (std::optional<uint32_t> Val = extractMdIntValue(StaticSamplerNode, 13))
    Sampler.ShaderVisibility = *Val;
  else
    return reportError(Ctx, "Invalid value for ShaderVisibility");

  RSD.StaticSamplers.push_back(Sampler);
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

static bool verifyDescriptorRangeFlag(uint32_t Version, uint32_t Type,
                                      uint32_t FlagsVal) {
  using FlagT = dxbc::DescriptorRangeFlags;
  FlagT Flags = FlagT(FlagsVal);

  const bool IsSampler =
      (Type == llvm::to_underlying(dxbc::DescriptorRangeType::Sampler));

  if (Version == 1) {
    // Since the metadata is unversioned, we expect to explicitly see the values
    // that map to the version 1 behaviour here.
    if (IsSampler)
      return Flags == FlagT::DescriptorsVolatile;
    return Flags == (FlagT::DataVolatile | FlagT::DescriptorsVolatile);
  }

  // The data-specific flags are mutually exclusive.
  FlagT DataFlags = FlagT::DataVolatile | FlagT::DataStatic |
                    FlagT::DataStaticWhileSetAtExecute;

  if (popcount(llvm::to_underlying(Flags & DataFlags)) > 1)
    return false;

  // The descriptor-specific flags are mutually exclusive.
  FlagT DescriptorFlags = FlagT::DescriptorsStaticKeepingBufferBoundsChecks |
                          FlagT::DescriptorsVolatile;
  if (popcount(llvm::to_underlying(Flags & DescriptorFlags)) > 1)
    return false;

  // For volatile descriptors, DATA_STATIC is never valid.
  if ((Flags & FlagT::DescriptorsVolatile) == FlagT::DescriptorsVolatile) {
    FlagT Mask = FlagT::DescriptorsVolatile;
    if (!IsSampler) {
      Mask |= FlagT::DataVolatile;
      Mask |= FlagT::DataStaticWhileSetAtExecute;
    }
    return (Flags & ~Mask) == FlagT::None;
  }

  // For "STATIC_KEEPING_BUFFER_BOUNDS_CHECKS" descriptors,
  // the other data-specific flags may all be set.
  if ((Flags & FlagT::DescriptorsStaticKeepingBufferBoundsChecks) ==
      FlagT::DescriptorsStaticKeepingBufferBoundsChecks) {
    FlagT Mask = FlagT::DescriptorsStaticKeepingBufferBoundsChecks;
    if (!IsSampler) {
      Mask |= FlagT::DataVolatile;
      Mask |= FlagT::DataStatic;
      Mask |= FlagT::DataStaticWhileSetAtExecute;
    }
    return (Flags & ~Mask) == FlagT::None;
  }

  // When no descriptor flag is set, any data flag is allowed.
  FlagT Mask = FlagT::None;
  if (!IsSampler) {
    Mask |= FlagT::DataVolatile;
    Mask |= FlagT::DataStaticWhileSetAtExecute;
    Mask |= FlagT::DataStatic;
  }
  return (Flags & ~Mask) == FlagT::None;
}

static bool verifySamplerFilter(uint32_t Value) {
  switch (Value) {
#define FILTER(Num, Val) case llvm::to_underlying(dxbc::SamplerFilter::Val):
#include "llvm/BinaryFormat/DXContainerConstants.def"
    return true;
  }
  return false;
}

// Values allowed here:
// https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_texture_address_mode#syntax
static bool verifyAddress(uint32_t Address) {
  switch (Address) {
#define TEXTURE_ADDRESS_MODE(Num, Val)                                         \
  case llvm::to_underlying(dxbc::TextureAddressMode::Val):
#include "llvm/BinaryFormat/DXContainerConstants.def"
    return true;
  }
  return false;
}

static bool verifyMipLODBias(float MipLODBias) {
  return MipLODBias >= -16.f && MipLODBias <= 15.99f;
}

static bool verifyMaxAnisotropy(uint32_t MaxAnisotropy) {
  return MaxAnisotropy <= 16u;
}

static bool verifyComparisonFunc(uint32_t ComparisonFunc) {
  switch (ComparisonFunc) {
#define COMPARISON_FUNC(Num, Val)                                              \
  case llvm::to_underlying(dxbc::ComparisonFunc::Val):
#include "llvm/BinaryFormat/DXContainerConstants.def"
    return true;
  }
  return false;
}

static bool verifyBorderColor(uint32_t BorderColor) {
  switch (BorderColor) {
#define STATIC_BORDER_COLOR(Num, Val)                                          \
  case llvm::to_underlying(dxbc::StaticBorderColor::Val):
#include "llvm/BinaryFormat/DXContainerConstants.def"
    return true;
  }
  return false;
}

static bool verifyLOD(float LOD) { return !std::isnan(LOD); }

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
          return reportValueError(Ctx, "DescriptorRangeFlag", Descriptor.Flags);
      }
      break;
    }
    case llvm::to_underlying(dxbc::RootParameterType::DescriptorTable): {
      const mcdxbc::DescriptorTable &Table =
          RSD.ParametersContainer.getDescriptorTable(Info.Location);
      for (const dxbc::RTS0::v2::DescriptorRange &Range : Table) {
        if (!verifyRangeType(Range.RangeType))
          return reportValueError(Ctx, "RangeType", Range.RangeType);

        if (!verifyRegisterSpace(Range.RegisterSpace))
          return reportValueError(Ctx, "RegisterSpace", Range.RegisterSpace);

        if (!verifyDescriptorRangeFlag(RSD.Version, Range.RangeType,
                                       Range.Flags))
          return reportValueError(Ctx, "DescriptorFlag", Range.Flags);
      }
      break;
    }
    }
  }

  for (const dxbc::RTS0::v1::StaticSampler &Sampler : RSD.StaticSamplers) {
    if (!verifySamplerFilter(Sampler.Filter))
      return reportValueError(Ctx, "Filter", Sampler.Filter);

    if (!verifyAddress(Sampler.AddressU))
      return reportValueError(Ctx, "AddressU", Sampler.AddressU);

    if (!verifyAddress(Sampler.AddressV))
      return reportValueError(Ctx, "AddressV", Sampler.AddressV);

    if (!verifyAddress(Sampler.AddressW))
      return reportValueError(Ctx, "AddressW", Sampler.AddressW);

    if (!verifyMipLODBias(Sampler.MipLODBias))
      return reportValueError(Ctx, "MipLODBias", Sampler.MipLODBias);

    if (!verifyMaxAnisotropy(Sampler.MaxAnisotropy))
      return reportValueError(Ctx, "MaxAnisotropy", Sampler.MaxAnisotropy);

    if (!verifyComparisonFunc(Sampler.ComparisonFunc))
      return reportValueError(Ctx, "ComparisonFunc", Sampler.ComparisonFunc);

    if (!verifyBorderColor(Sampler.BorderColor))
      return reportValueError(Ctx, "BorderColor", Sampler.BorderColor);

    if (!verifyLOD(Sampler.MinLOD))
      return reportValueError(Ctx, "MinLOD", Sampler.MinLOD);

    if (!verifyLOD(Sampler.MaxLOD))
      return reportValueError(Ctx, "MaxLOD", Sampler.MaxLOD);

    if (!verifyRegisterValue(Sampler.ShaderRegister))
      return reportValueError(Ctx, "ShaderRegister", Sampler.ShaderRegister);

    if (!verifyRegisterSpace(Sampler.RegisterSpace))
      return reportValueError(Ctx, "RegisterSpace", Sampler.RegisterSpace);

    if (!dxbc::isValidShaderVisibility(Sampler.ShaderVisibility))
      return reportValueError(Ctx, "ShaderVisibility",
                              Sampler.ShaderVisibility);
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
