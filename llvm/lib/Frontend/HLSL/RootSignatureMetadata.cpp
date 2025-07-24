//===- RootSignatureMetadata.h - HLSL Root Signature helpers --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file implements a library for working with HLSL Root Signatures
/// and their metadata representation.
///
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/HLSL/RootSignatureMetadata.h"
#include "llvm/Frontend/HLSL/RootSignatureValidations.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/ScopedPrinter.h"

namespace llvm {
namespace hlsl {
namespace rootsig {

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

static const EnumEntry<dxil::ResourceClass> ResourceClassNames[] = {
    {"CBV", dxil::ResourceClass::CBuffer},
    {"SRV", dxil::ResourceClass::SRV},
    {"UAV", dxil::ResourceClass::UAV},
    {"Sampler", dxil::ResourceClass::Sampler},
};

static std::optional<StringRef> getResourceName(dxil::ResourceClass Class) {
  for (const auto &ClassEnum : ResourceClassNames)
    if (ClassEnum.Value == Class)
      return ClassEnum.Name;
  return std::nullopt;
}

namespace {

// We use the OverloadVisit with std::visit to ensure the compiler catches if a
// new RootElement variant type is added but it's metadata generation isn't
// handled.
template <class... Ts> struct OverloadedVisit : Ts... {
  using Ts::operator()...;
};
template <class... Ts> OverloadedVisit(Ts...) -> OverloadedVisit<Ts...>;

} // namespace

MDNode *MetadataBuilder::BuildRootSignature() {
  const auto Visitor = OverloadedVisit{
      [this](const dxbc::RootFlags &Flags) -> MDNode * {
        return BuildRootFlags(Flags);
      },
      [this](const RootConstants &Constants) -> MDNode * {
        return BuildRootConstants(Constants);
      },
      [this](const RootDescriptor &Descriptor) -> MDNode * {
        return BuildRootDescriptor(Descriptor);
      },
      [this](const DescriptorTableClause &Clause) -> MDNode * {
        return BuildDescriptorTableClause(Clause);
      },
      [this](const DescriptorTable &Table) -> MDNode * {
        return BuildDescriptorTable(Table);
      },
      [this](const StaticSampler &Sampler) -> MDNode * {
        return BuildStaticSampler(Sampler);
      },
  };

  for (const RootElement &Element : Elements) {
    MDNode *ElementMD = std::visit(Visitor, Element);
    assert(ElementMD != nullptr &&
           "Root Element must be initialized and validated");
    GeneratedMetadata.push_back(ElementMD);
  }

  return MDNode::get(Ctx, GeneratedMetadata);
}

MDNode *MetadataBuilder::BuildRootFlags(const dxbc::RootFlags &Flags) {
  IRBuilder<> Builder(Ctx);
  Metadata *Operands[] = {
      MDString::get(Ctx, "RootFlags"),
      ConstantAsMetadata::get(Builder.getInt32(llvm::to_underlying(Flags))),
  };
  return MDNode::get(Ctx, Operands);
}

MDNode *MetadataBuilder::BuildRootConstants(const RootConstants &Constants) {
  IRBuilder<> Builder(Ctx);
  Metadata *Operands[] = {
      MDString::get(Ctx, "RootConstants"),
      ConstantAsMetadata::get(
          Builder.getInt32(llvm::to_underlying(Constants.Visibility))),
      ConstantAsMetadata::get(Builder.getInt32(Constants.Reg.Number)),
      ConstantAsMetadata::get(Builder.getInt32(Constants.Space)),
      ConstantAsMetadata::get(Builder.getInt32(Constants.Num32BitConstants)),
  };
  return MDNode::get(Ctx, Operands);
}

MDNode *MetadataBuilder::BuildRootDescriptor(const RootDescriptor &Descriptor) {
  IRBuilder<> Builder(Ctx);
  std::optional<StringRef> ResName = getResourceName(
      dxil::ResourceClass(llvm::to_underlying(Descriptor.Type)));
  assert(ResName && "Provided an invalid Resource Class");
  llvm::SmallString<7> Name({"Root", *ResName});
  Metadata *Operands[] = {
      MDString::get(Ctx, Name),
      ConstantAsMetadata::get(
          Builder.getInt32(llvm::to_underlying(Descriptor.Visibility))),
      ConstantAsMetadata::get(Builder.getInt32(Descriptor.Reg.Number)),
      ConstantAsMetadata::get(Builder.getInt32(Descriptor.Space)),
      ConstantAsMetadata::get(
          Builder.getInt32(llvm::to_underlying(Descriptor.Flags))),
  };
  return MDNode::get(Ctx, Operands);
}

MDNode *MetadataBuilder::BuildDescriptorTable(const DescriptorTable &Table) {
  IRBuilder<> Builder(Ctx);
  SmallVector<Metadata *> TableOperands;
  // Set the mandatory arguments
  TableOperands.push_back(MDString::get(Ctx, "DescriptorTable"));
  TableOperands.push_back(ConstantAsMetadata::get(
      Builder.getInt32(llvm::to_underlying(Table.Visibility))));

  // Remaining operands are references to the table's clauses. The in-memory
  // representation of the Root Elements created from parsing will ensure that
  // the previous N elements are the clauses for this table.
  assert(Table.NumClauses <= GeneratedMetadata.size() &&
         "Table expected all owned clauses to be generated already");
  // So, add a refence to each clause to our operands
  TableOperands.append(GeneratedMetadata.end() - Table.NumClauses,
                       GeneratedMetadata.end());
  // Then, remove those clauses from the general list of Root Elements
  GeneratedMetadata.pop_back_n(Table.NumClauses);

  return MDNode::get(Ctx, TableOperands);
}

MDNode *MetadataBuilder::BuildDescriptorTableClause(
    const DescriptorTableClause &Clause) {
  IRBuilder<> Builder(Ctx);
  std::optional<StringRef> ResName =
      getResourceName(dxil::ResourceClass(llvm::to_underlying(Clause.Type)));
  assert(ResName && "Provided an invalid Resource Class");
  Metadata *Operands[] = {
      MDString::get(Ctx, *ResName),
      ConstantAsMetadata::get(Builder.getInt32(Clause.NumDescriptors)),
      ConstantAsMetadata::get(Builder.getInt32(Clause.Reg.Number)),
      ConstantAsMetadata::get(Builder.getInt32(Clause.Space)),
      ConstantAsMetadata::get(Builder.getInt32(Clause.Offset)),
      ConstantAsMetadata::get(
          Builder.getInt32(llvm::to_underlying(Clause.Flags))),
  };
  return MDNode::get(Ctx, Operands);
}

MDNode *MetadataBuilder::BuildStaticSampler(const StaticSampler &Sampler) {
  IRBuilder<> Builder(Ctx);
  Metadata *Operands[] = {
      MDString::get(Ctx, "StaticSampler"),
      ConstantAsMetadata::get(
          Builder.getInt32(llvm::to_underlying(Sampler.Filter))),
      ConstantAsMetadata::get(
          Builder.getInt32(llvm::to_underlying(Sampler.AddressU))),
      ConstantAsMetadata::get(
          Builder.getInt32(llvm::to_underlying(Sampler.AddressV))),
      ConstantAsMetadata::get(
          Builder.getInt32(llvm::to_underlying(Sampler.AddressW))),
      ConstantAsMetadata::get(llvm::ConstantFP::get(llvm::Type::getFloatTy(Ctx),
                                                    Sampler.MipLODBias)),
      ConstantAsMetadata::get(Builder.getInt32(Sampler.MaxAnisotropy)),
      ConstantAsMetadata::get(
          Builder.getInt32(llvm::to_underlying(Sampler.CompFunc))),
      ConstantAsMetadata::get(
          Builder.getInt32(llvm::to_underlying(Sampler.BorderColor))),
      ConstantAsMetadata::get(
          llvm::ConstantFP::get(llvm::Type::getFloatTy(Ctx), Sampler.MinLOD)),
      ConstantAsMetadata::get(
          llvm::ConstantFP::get(llvm::Type::getFloatTy(Ctx), Sampler.MaxLOD)),
      ConstantAsMetadata::get(Builder.getInt32(Sampler.Reg.Number)),
      ConstantAsMetadata::get(Builder.getInt32(Sampler.Space)),
      ConstantAsMetadata::get(
          Builder.getInt32(llvm::to_underlying(Sampler.Visibility))),
  };
  return MDNode::get(Ctx, Operands);
}

bool MetadataParser::parseRootFlags(LLVMContext *Ctx,
                                    mcdxbc::RootSignatureDesc &RSD,
                                    MDNode *RootFlagNode) {

  if (RootFlagNode->getNumOperands() != 2)
    return reportError(Ctx, "Invalid format for RootFlag Element");

  if (std::optional<uint32_t> Val = extractMdIntValue(RootFlagNode, 1))
    RSD.Flags = *Val;
  else
    return reportError(Ctx, "Invalid value for RootFlag");

  return false;
}

bool MetadataParser::parseRootConstants(LLVMContext *Ctx,
                                        mcdxbc::RootSignatureDesc &RSD,
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

bool MetadataParser::parseRootDescriptors(
    LLVMContext *Ctx, mcdxbc::RootSignatureDesc &RSD,
    MDNode *RootDescriptorNode, RootSignatureElementKind ElementKind) {
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

bool MetadataParser::parseDescriptorRange(LLVMContext *Ctx,
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

bool MetadataParser::parseDescriptorTable(LLVMContext *Ctx,
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

bool MetadataParser::parseStaticSampler(LLVMContext *Ctx,
                                        mcdxbc::RootSignatureDesc &RSD,
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

bool MetadataParser::parseRootSignatureElement(LLVMContext *Ctx,
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

bool MetadataParser::validateRootSignature(
    LLVMContext *Ctx, const llvm::mcdxbc::RootSignatureDesc &RSD) {
  if (!llvm::hlsl::rootsig::verifyVersion(RSD.Version)) {
    return reportValueError(Ctx, "Version", RSD.Version);
  }

  if (!llvm::hlsl::rootsig::verifyRootFlag(RSD.Flags)) {
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
      if (!llvm::hlsl::rootsig::verifyRegisterValue(Descriptor.ShaderRegister))
        return reportValueError(Ctx, "ShaderRegister",
                                Descriptor.ShaderRegister);

      if (!llvm::hlsl::rootsig::verifyRegisterSpace(Descriptor.RegisterSpace))
        return reportValueError(Ctx, "RegisterSpace", Descriptor.RegisterSpace);

      if (RSD.Version > 1) {
        if (!llvm::hlsl::rootsig::verifyRootDescriptorFlag(RSD.Version,
                                                           Descriptor.Flags))
          return reportValueError(Ctx, "RootDescriptorFlag", Descriptor.Flags);
      }
      break;
    }
    case llvm::to_underlying(dxbc::RootParameterType::DescriptorTable): {
      const mcdxbc::DescriptorTable &Table =
          RSD.ParametersContainer.getDescriptorTable(Info.Location);
      for (const dxbc::RTS0::v2::DescriptorRange &Range : Table) {
        if (!llvm::hlsl::rootsig::verifyRangeType(Range.RangeType))
          return reportValueError(Ctx, "RangeType", Range.RangeType);

        if (!llvm::hlsl::rootsig::verifyRegisterSpace(Range.RegisterSpace))
          return reportValueError(Ctx, "RegisterSpace", Range.RegisterSpace);

        if (!llvm::hlsl::rootsig::verifyNumDescriptors(Range.NumDescriptors))
          return reportValueError(Ctx, "NumDescriptors", Range.NumDescriptors);

        if (!llvm::hlsl::rootsig::verifyDescriptorRangeFlag(
                RSD.Version, Range.RangeType, Range.Flags))
          return reportValueError(Ctx, "DescriptorFlag", Range.Flags);
      }
      break;
    }
    }
  }

  for (const dxbc::RTS0::v1::StaticSampler &Sampler : RSD.StaticSamplers) {
    if (!llvm::hlsl::rootsig::verifySamplerFilter(Sampler.Filter))
      return reportValueError(Ctx, "Filter", Sampler.Filter);

    if (!llvm::hlsl::rootsig::verifyAddress(Sampler.AddressU))
      return reportValueError(Ctx, "AddressU", Sampler.AddressU);

    if (!llvm::hlsl::rootsig::verifyAddress(Sampler.AddressV))
      return reportValueError(Ctx, "AddressV", Sampler.AddressV);

    if (!llvm::hlsl::rootsig::verifyAddress(Sampler.AddressW))
      return reportValueError(Ctx, "AddressW", Sampler.AddressW);

    if (!llvm::hlsl::rootsig::verifyMipLODBias(Sampler.MipLODBias))
      return reportValueError(Ctx, "MipLODBias", Sampler.MipLODBias);

    if (!llvm::hlsl::rootsig::verifyMaxAnisotropy(Sampler.MaxAnisotropy))
      return reportValueError(Ctx, "MaxAnisotropy", Sampler.MaxAnisotropy);

    if (!llvm::hlsl::rootsig::verifyComparisonFunc(Sampler.ComparisonFunc))
      return reportValueError(Ctx, "ComparisonFunc", Sampler.ComparisonFunc);

    if (!llvm::hlsl::rootsig::verifyBorderColor(Sampler.BorderColor))
      return reportValueError(Ctx, "BorderColor", Sampler.BorderColor);

    if (!llvm::hlsl::rootsig::verifyLOD(Sampler.MinLOD))
      return reportValueError(Ctx, "MinLOD", Sampler.MinLOD);

    if (!llvm::hlsl::rootsig::verifyLOD(Sampler.MaxLOD))
      return reportValueError(Ctx, "MaxLOD", Sampler.MaxLOD);

    if (!llvm::hlsl::rootsig::verifyRegisterValue(Sampler.ShaderRegister))
      return reportValueError(Ctx, "ShaderRegister", Sampler.ShaderRegister);

    if (!llvm::hlsl::rootsig::verifyRegisterSpace(Sampler.RegisterSpace))
      return reportValueError(Ctx, "RegisterSpace", Sampler.RegisterSpace);

    if (!dxbc::isValidShaderVisibility(Sampler.ShaderVisibility))
      return reportValueError(Ctx, "ShaderVisibility",
                              Sampler.ShaderVisibility);
  }

  return false;
}

bool MetadataParser::ParseRootSignature(LLVMContext *Ctx,
                                        mcdxbc::RootSignatureDesc &RSD) {
  bool HasError = false;

  // Loop through the Root Elements of the root signature.
  for (const auto &Operand : Root->operands()) {
    MDNode *Element = dyn_cast<MDNode>(Operand);
    if (Element == nullptr)
      return reportError(Ctx, "Missing Root Element Metadata Node.");

    HasError = HasError || parseRootSignatureElement(Ctx, RSD, Element) ||
               validateRootSignature(Ctx, RSD);
  }

  return HasError;
}
} // namespace rootsig
} // namespace hlsl
} // namespace llvm
