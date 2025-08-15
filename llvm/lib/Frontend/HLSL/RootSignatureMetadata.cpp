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
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/ScopedPrinter.h"

using namespace llvm;

namespace llvm {
namespace hlsl {
namespace rootsig {

char GenericRSMetadataError::ID;
char InvalidRSMetadataFormat::ID;
char InvalidRSMetadataValue::ID;
template <typename T> char RootSignatureValidationError<T>::ID;

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
      ConstantAsMetadata::get(Builder.getInt32(to_underlying(Flags))),
  };
  return MDNode::get(Ctx, Operands);
}

MDNode *MetadataBuilder::BuildRootConstants(const RootConstants &Constants) {
  IRBuilder<> Builder(Ctx);
  Metadata *Operands[] = {
      MDString::get(Ctx, "RootConstants"),
      ConstantAsMetadata::get(
          Builder.getInt32(to_underlying(Constants.Visibility))),
      ConstantAsMetadata::get(Builder.getInt32(Constants.Reg.Number)),
      ConstantAsMetadata::get(Builder.getInt32(Constants.Space)),
      ConstantAsMetadata::get(Builder.getInt32(Constants.Num32BitConstants)),
  };
  return MDNode::get(Ctx, Operands);
}

MDNode *MetadataBuilder::BuildRootDescriptor(const RootDescriptor &Descriptor) {
  IRBuilder<> Builder(Ctx);
  StringRef ResName =
      enumToStringRef(dxil::ResourceClass(to_underlying(Descriptor.Type)),
                      dxil::getResourceClasses());
  assert(!ResName.empty() && "Provided an invalid Resource Class");
  SmallString<7> Name({"Root", ResName});
  Metadata *Operands[] = {
      MDString::get(Ctx, Name),
      ConstantAsMetadata::get(
          Builder.getInt32(to_underlying(Descriptor.Visibility))),
      ConstantAsMetadata::get(Builder.getInt32(Descriptor.Reg.Number)),
      ConstantAsMetadata::get(Builder.getInt32(Descriptor.Space)),
      ConstantAsMetadata::get(
          Builder.getInt32(to_underlying(Descriptor.Flags))),
  };
  return MDNode::get(Ctx, Operands);
}

MDNode *MetadataBuilder::BuildDescriptorTable(const DescriptorTable &Table) {
  IRBuilder<> Builder(Ctx);
  SmallVector<Metadata *> TableOperands;
  // Set the mandatory arguments
  TableOperands.push_back(MDString::get(Ctx, "DescriptorTable"));
  TableOperands.push_back(ConstantAsMetadata::get(
      Builder.getInt32(to_underlying(Table.Visibility))));

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
  StringRef ResName =
      enumToStringRef(dxil::ResourceClass(to_underlying(Clause.Type)),
                      dxil::getResourceClasses());
  assert(!ResName.empty() && "Provided an invalid Resource Class");
  Metadata *Operands[] = {
      MDString::get(Ctx, ResName),
      ConstantAsMetadata::get(Builder.getInt32(Clause.NumDescriptors)),
      ConstantAsMetadata::get(Builder.getInt32(Clause.Reg.Number)),
      ConstantAsMetadata::get(Builder.getInt32(Clause.Space)),
      ConstantAsMetadata::get(Builder.getInt32(Clause.Offset)),
      ConstantAsMetadata::get(Builder.getInt32(to_underlying(Clause.Flags))),
  };
  return MDNode::get(Ctx, Operands);
}

MDNode *MetadataBuilder::BuildStaticSampler(const StaticSampler &Sampler) {
  IRBuilder<> Builder(Ctx);
  Metadata *Operands[] = {
      MDString::get(Ctx, "StaticSampler"),
      ConstantAsMetadata::get(Builder.getInt32(to_underlying(Sampler.Filter))),
      ConstantAsMetadata::get(
          Builder.getInt32(to_underlying(Sampler.AddressU))),
      ConstantAsMetadata::get(
          Builder.getInt32(to_underlying(Sampler.AddressV))),
      ConstantAsMetadata::get(
          Builder.getInt32(to_underlying(Sampler.AddressW))),
      ConstantAsMetadata::get(
          ConstantFP::get(Type::getFloatTy(Ctx), Sampler.MipLODBias)),
      ConstantAsMetadata::get(Builder.getInt32(Sampler.MaxAnisotropy)),
      ConstantAsMetadata::get(
          Builder.getInt32(to_underlying(Sampler.CompFunc))),
      ConstantAsMetadata::get(
          Builder.getInt32(to_underlying(Sampler.BorderColor))),
      ConstantAsMetadata::get(
          ConstantFP::get(Type::getFloatTy(Ctx), Sampler.MinLOD)),
      ConstantAsMetadata::get(
          ConstantFP::get(Type::getFloatTy(Ctx), Sampler.MaxLOD)),
      ConstantAsMetadata::get(Builder.getInt32(Sampler.Reg.Number)),
      ConstantAsMetadata::get(Builder.getInt32(Sampler.Space)),
      ConstantAsMetadata::get(
          Builder.getInt32(to_underlying(Sampler.Visibility))),
  };
  return MDNode::get(Ctx, Operands);
}

Error MetadataParser::parseRootFlags(mcdxbc::RootSignatureDesc &RSD,
                                     MDNode *RootFlagNode) {
  if (RootFlagNode->getNumOperands() != 2)
    return make_error<InvalidRSMetadataFormat>("RootFlag Element");

  if (std::optional<uint32_t> Val = extractMdIntValue(RootFlagNode, 1))
    RSD.Flags = *Val;
  else
    return make_error<InvalidRSMetadataValue>("RootFlag");

  return Error::success();
}

Error MetadataParser::parseRootConstants(mcdxbc::RootSignatureDesc &RSD,
                                         MDNode *RootConstantNode) {
  if (RootConstantNode->getNumOperands() != 5)
    return make_error<InvalidRSMetadataFormat>("RootConstants Element");

  dxbc::RTS0::v1::RootParameterHeader Header;
  // The parameter offset doesn't matter here - we recalculate it during
  // serialization  Header.ParameterOffset = 0;
  Header.ParameterType = to_underlying(dxbc::RootParameterType::Constants32Bit);

  if (std::optional<uint32_t> Val = extractMdIntValue(RootConstantNode, 1))
    Header.ShaderVisibility = *Val;
  else
    return make_error<InvalidRSMetadataValue>("ShaderVisibility");

  dxbc::RTS0::v1::RootConstants Constants;
  if (std::optional<uint32_t> Val = extractMdIntValue(RootConstantNode, 2))
    Constants.ShaderRegister = *Val;
  else
    return make_error<InvalidRSMetadataValue>("ShaderRegister");

  if (std::optional<uint32_t> Val = extractMdIntValue(RootConstantNode, 3))
    Constants.RegisterSpace = *Val;
  else
    return make_error<InvalidRSMetadataValue>("RegisterSpace");

  if (std::optional<uint32_t> Val = extractMdIntValue(RootConstantNode, 4))
    Constants.Num32BitValues = *Val;
  else
    return make_error<InvalidRSMetadataValue>("Num32BitValues");

  RSD.ParametersContainer.addParameter(Header, Constants);

  return Error::success();
}

Error MetadataParser::parseRootDescriptors(
    mcdxbc::RootSignatureDesc &RSD, MDNode *RootDescriptorNode,
    RootSignatureElementKind ElementKind) {
  assert((ElementKind == RootSignatureElementKind::SRV ||
          ElementKind == RootSignatureElementKind::UAV ||
          ElementKind == RootSignatureElementKind::CBV) &&
         "parseRootDescriptors should only be called with RootDescriptor "
         "element kind.");
  if (RootDescriptorNode->getNumOperands() != 5)
    return make_error<InvalidRSMetadataFormat>("Root Descriptor Element");

  dxbc::RTS0::v1::RootParameterHeader Header;
  switch (ElementKind) {
  case RootSignatureElementKind::SRV:
    Header.ParameterType = to_underlying(dxbc::RootParameterType::SRV);
    break;
  case RootSignatureElementKind::UAV:
    Header.ParameterType = to_underlying(dxbc::RootParameterType::UAV);
    break;
  case RootSignatureElementKind::CBV:
    Header.ParameterType = to_underlying(dxbc::RootParameterType::CBV);
    break;
  default:
    llvm_unreachable("invalid Root Descriptor kind");
    break;
  }

  if (std::optional<uint32_t> Val = extractMdIntValue(RootDescriptorNode, 1))
    Header.ShaderVisibility = *Val;
  else
    return make_error<InvalidRSMetadataValue>("ShaderVisibility");

  dxbc::RTS0::v2::RootDescriptor Descriptor;
  if (std::optional<uint32_t> Val = extractMdIntValue(RootDescriptorNode, 2))
    Descriptor.ShaderRegister = *Val;
  else
    return make_error<InvalidRSMetadataValue>("ShaderRegister");

  if (std::optional<uint32_t> Val = extractMdIntValue(RootDescriptorNode, 3))
    Descriptor.RegisterSpace = *Val;
  else
    return make_error<InvalidRSMetadataValue>("RegisterSpace");

  if (RSD.Version == 1) {
    RSD.ParametersContainer.addParameter(Header, Descriptor);
    return Error::success();
  }
  assert(RSD.Version > 1);

  if (std::optional<uint32_t> Val = extractMdIntValue(RootDescriptorNode, 4))
    Descriptor.Flags = *Val;
  else
    return make_error<InvalidRSMetadataValue>("Root Descriptor Flags");

  RSD.ParametersContainer.addParameter(Header, Descriptor);
  return Error::success();
}

Error MetadataParser::parseDescriptorRange(mcdxbc::DescriptorTable &Table,
                                           MDNode *RangeDescriptorNode) {
  if (RangeDescriptorNode->getNumOperands() != 6)
    return make_error<InvalidRSMetadataFormat>("Descriptor Range");

  dxbc::RTS0::v2::DescriptorRange Range;

  std::optional<StringRef> ElementText =
      extractMdStringValue(RangeDescriptorNode, 0);

  if (!ElementText.has_value())
    return make_error<InvalidRSMetadataFormat>("Descriptor Range");

  Range.RangeType =
      StringSwitch<uint32_t>(*ElementText)
          .Case("CBV", to_underlying(dxbc::DescriptorRangeType::CBV))
          .Case("SRV", to_underlying(dxbc::DescriptorRangeType::SRV))
          .Case("UAV", to_underlying(dxbc::DescriptorRangeType::UAV))
          .Case("Sampler", to_underlying(dxbc::DescriptorRangeType::Sampler))
          .Default(~0U);

  if (Range.RangeType == ~0U)
    return make_error<GenericRSMetadataError>("Invalid Descriptor Range type.",
                                              RangeDescriptorNode);

  if (std::optional<uint32_t> Val = extractMdIntValue(RangeDescriptorNode, 1))
    Range.NumDescriptors = *Val;
  else
    return make_error<GenericRSMetadataError>("Number of Descriptor in Range",
                                              RangeDescriptorNode);

  if (std::optional<uint32_t> Val = extractMdIntValue(RangeDescriptorNode, 2))
    Range.BaseShaderRegister = *Val;
  else
    return make_error<InvalidRSMetadataValue>("BaseShaderRegister");

  if (std::optional<uint32_t> Val = extractMdIntValue(RangeDescriptorNode, 3))
    Range.RegisterSpace = *Val;
  else
    return make_error<InvalidRSMetadataValue>("RegisterSpace");

  if (std::optional<uint32_t> Val = extractMdIntValue(RangeDescriptorNode, 4))
    Range.OffsetInDescriptorsFromTableStart = *Val;
  else
    return make_error<InvalidRSMetadataValue>(
        "OffsetInDescriptorsFromTableStart");

  if (std::optional<uint32_t> Val = extractMdIntValue(RangeDescriptorNode, 5))
    Range.Flags = *Val;
  else
    return make_error<InvalidRSMetadataValue>("Descriptor Range Flags");

  Table.Ranges.push_back(Range);
  return Error::success();
}

Error MetadataParser::parseDescriptorTable(mcdxbc::RootSignatureDesc &RSD,
                                           MDNode *DescriptorTableNode) {
  const unsigned int NumOperands = DescriptorTableNode->getNumOperands();
  if (NumOperands < 2)
    return make_error<InvalidRSMetadataFormat>("Descriptor Table");

  dxbc::RTS0::v1::RootParameterHeader Header;
  if (std::optional<uint32_t> Val = extractMdIntValue(DescriptorTableNode, 1))
    Header.ShaderVisibility = *Val;
  else
    return make_error<InvalidRSMetadataValue>("ShaderVisibility");

  mcdxbc::DescriptorTable Table;
  Header.ParameterType =
      to_underlying(dxbc::RootParameterType::DescriptorTable);

  for (unsigned int I = 2; I < NumOperands; I++) {
    MDNode *Element = dyn_cast<MDNode>(DescriptorTableNode->getOperand(I));
    if (Element == nullptr)
      return make_error<GenericRSMetadataError>(
          "Missing Root Element Metadata Node.", DescriptorTableNode);

    if (auto Err = parseDescriptorRange(Table, Element))
      return Err;
  }

  RSD.ParametersContainer.addParameter(Header, Table);
  return Error::success();
}

Error MetadataParser::parseStaticSampler(mcdxbc::RootSignatureDesc &RSD,
                                         MDNode *StaticSamplerNode) {
  if (StaticSamplerNode->getNumOperands() != 14)
    return make_error<InvalidRSMetadataFormat>("Static Sampler");

  dxbc::RTS0::v1::StaticSampler Sampler;
  if (std::optional<uint32_t> Val = extractMdIntValue(StaticSamplerNode, 1))
    Sampler.Filter = *Val;
  else
    return make_error<InvalidRSMetadataValue>("Filter");

  if (std::optional<uint32_t> Val = extractMdIntValue(StaticSamplerNode, 2))
    Sampler.AddressU = *Val;
  else
    return make_error<InvalidRSMetadataValue>("AddressU");

  if (std::optional<uint32_t> Val = extractMdIntValue(StaticSamplerNode, 3))
    Sampler.AddressV = *Val;
  else
    return make_error<InvalidRSMetadataValue>("AddressV");

  if (std::optional<uint32_t> Val = extractMdIntValue(StaticSamplerNode, 4))
    Sampler.AddressW = *Val;
  else
    return make_error<InvalidRSMetadataValue>("AddressW");

  if (std::optional<float> Val = extractMdFloatValue(StaticSamplerNode, 5))
    Sampler.MipLODBias = *Val;
  else
    return make_error<InvalidRSMetadataValue>("MipLODBias");

  if (std::optional<uint32_t> Val = extractMdIntValue(StaticSamplerNode, 6))
    Sampler.MaxAnisotropy = *Val;
  else
    return make_error<InvalidRSMetadataValue>("MaxAnisotropy");

  if (std::optional<uint32_t> Val = extractMdIntValue(StaticSamplerNode, 7))
    Sampler.ComparisonFunc = *Val;
  else
    return make_error<InvalidRSMetadataValue>("ComparisonFunc");

  if (std::optional<uint32_t> Val = extractMdIntValue(StaticSamplerNode, 8))
    Sampler.BorderColor = *Val;
  else
    return make_error<InvalidRSMetadataValue>("ComparisonFunc");

  if (std::optional<float> Val = extractMdFloatValue(StaticSamplerNode, 9))
    Sampler.MinLOD = *Val;
  else
    return make_error<InvalidRSMetadataValue>("MinLOD");

  if (std::optional<float> Val = extractMdFloatValue(StaticSamplerNode, 10))
    Sampler.MaxLOD = *Val;
  else
    return make_error<InvalidRSMetadataValue>("MaxLOD");

  if (std::optional<uint32_t> Val = extractMdIntValue(StaticSamplerNode, 11))
    Sampler.ShaderRegister = *Val;
  else
    return make_error<InvalidRSMetadataValue>("ShaderRegister");

  if (std::optional<uint32_t> Val = extractMdIntValue(StaticSamplerNode, 12))
    Sampler.RegisterSpace = *Val;
  else
    return make_error<InvalidRSMetadataValue>("RegisterSpace");

  if (std::optional<uint32_t> Val = extractMdIntValue(StaticSamplerNode, 13))
    Sampler.ShaderVisibility = *Val;
  else
    return make_error<InvalidRSMetadataValue>("ShaderVisibility");

  RSD.StaticSamplers.push_back(Sampler);
  return Error::success();
}

Error MetadataParser::parseRootSignatureElement(mcdxbc::RootSignatureDesc &RSD,
                                                MDNode *Element) {
  std::optional<StringRef> ElementText = extractMdStringValue(Element, 0);
  if (!ElementText.has_value())
    return make_error<InvalidRSMetadataFormat>("Root Element");

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
    return parseRootFlags(RSD, Element);
  case RootSignatureElementKind::RootConstants:
    return parseRootConstants(RSD, Element);
  case RootSignatureElementKind::CBV:
  case RootSignatureElementKind::SRV:
  case RootSignatureElementKind::UAV:
    return parseRootDescriptors(RSD, Element, ElementKind);
  case RootSignatureElementKind::DescriptorTable:
    return parseDescriptorTable(RSD, Element);
  case RootSignatureElementKind::StaticSamplers:
    return parseStaticSampler(RSD, Element);
  case RootSignatureElementKind::Error:
    return make_error<GenericRSMetadataError>("Invalid Root Signature Element",
                                              Element);
  }

  llvm_unreachable("Unhandled RootSignatureElementKind enum.");
}

Error MetadataParser::validateRootSignature(
    const mcdxbc::RootSignatureDesc &RSD) {
  Error DeferredErrs = Error::success();
  if (!hlsl::rootsig::verifyVersion(RSD.Version)) {
    DeferredErrs =
        joinErrors(std::move(DeferredErrs),
                   make_error<RootSignatureValidationError<uint32_t>>(
                       "Version", RSD.Version));
  }

  if (!hlsl::rootsig::verifyRootFlag(RSD.Flags)) {
    DeferredErrs =
        joinErrors(std::move(DeferredErrs),
                   make_error<RootSignatureValidationError<uint32_t>>(
                       "RootFlags", RSD.Flags));
  }

  for (const mcdxbc::RootParameterInfo &Info : RSD.ParametersContainer) {
    if (!dxbc::isValidShaderVisibility(Info.Header.ShaderVisibility))
      DeferredErrs =
          joinErrors(std::move(DeferredErrs),
                     make_error<RootSignatureValidationError<uint32_t>>(
                         "ShaderVisibility", Info.Header.ShaderVisibility));

    assert(dxbc::isValidParameterType(Info.Header.ParameterType) &&
           "Invalid value for ParameterType");

    switch (Info.Header.ParameterType) {

    case to_underlying(dxbc::RootParameterType::CBV):
    case to_underlying(dxbc::RootParameterType::UAV):
    case to_underlying(dxbc::RootParameterType::SRV): {
      const dxbc::RTS0::v2::RootDescriptor &Descriptor =
          RSD.ParametersContainer.getRootDescriptor(Info.Location);
      if (!hlsl::rootsig::verifyRegisterValue(Descriptor.ShaderRegister))
        DeferredErrs =
            joinErrors(std::move(DeferredErrs),
                       make_error<RootSignatureValidationError<uint32_t>>(
                           "ShaderRegister", Descriptor.ShaderRegister));

      if (!hlsl::rootsig::verifyRegisterSpace(Descriptor.RegisterSpace))
        DeferredErrs =
            joinErrors(std::move(DeferredErrs),
                       make_error<RootSignatureValidationError<uint32_t>>(
                           "RegisterSpace", Descriptor.RegisterSpace));

      if (RSD.Version > 1) {
        if (!hlsl::rootsig::verifyRootDescriptorFlag(RSD.Version,
                                                     Descriptor.Flags))
          DeferredErrs =
              joinErrors(std::move(DeferredErrs),
                         make_error<RootSignatureValidationError<uint32_t>>(
                             "RootDescriptorFlag", Descriptor.Flags));
      }
      break;
    }
    case to_underlying(dxbc::RootParameterType::DescriptorTable): {
      const mcdxbc::DescriptorTable &Table =
          RSD.ParametersContainer.getDescriptorTable(Info.Location);
      for (const dxbc::RTS0::v2::DescriptorRange &Range : Table) {
        if (!hlsl::rootsig::verifyRangeType(Range.RangeType))
          DeferredErrs =
              joinErrors(std::move(DeferredErrs),
                         make_error<RootSignatureValidationError<uint32_t>>(
                             "RangeType", Range.RangeType));

        if (!hlsl::rootsig::verifyRegisterSpace(Range.RegisterSpace))
          DeferredErrs =
              joinErrors(std::move(DeferredErrs),
                         make_error<RootSignatureValidationError<uint32_t>>(
                             "RegisterSpace", Range.RegisterSpace));

        if (!hlsl::rootsig::verifyNumDescriptors(Range.NumDescriptors))
          DeferredErrs =
              joinErrors(std::move(DeferredErrs),
                         make_error<RootSignatureValidationError<uint32_t>>(
                             "NumDescriptors", Range.NumDescriptors));

        if (!hlsl::rootsig::verifyDescriptorRangeFlag(
                RSD.Version, Range.RangeType, Range.Flags))
          DeferredErrs =
              joinErrors(std::move(DeferredErrs),
                         make_error<RootSignatureValidationError<uint32_t>>(
                             "DescriptorFlag", Range.Flags));
      }
      break;
    }
    }
  }

  for (const dxbc::RTS0::v1::StaticSampler &Sampler : RSD.StaticSamplers) {
    if (!hlsl::rootsig::verifySamplerFilter(Sampler.Filter))
      DeferredErrs =
          joinErrors(std::move(DeferredErrs),
                     make_error<RootSignatureValidationError<uint32_t>>(
                         "Filter", Sampler.Filter));

    if (!hlsl::rootsig::verifyAddress(Sampler.AddressU))
      DeferredErrs =
          joinErrors(std::move(DeferredErrs),
                     make_error<RootSignatureValidationError<uint32_t>>(
                         "AddressU", Sampler.AddressU));

    if (!hlsl::rootsig::verifyAddress(Sampler.AddressV))
      DeferredErrs =
          joinErrors(std::move(DeferredErrs),
                     make_error<RootSignatureValidationError<uint32_t>>(
                         "AddressV", Sampler.AddressV));

    if (!hlsl::rootsig::verifyAddress(Sampler.AddressW))
      DeferredErrs =
          joinErrors(std::move(DeferredErrs),
                     make_error<RootSignatureValidationError<uint32_t>>(
                         "AddressW", Sampler.AddressW));

    if (!hlsl::rootsig::verifyMipLODBias(Sampler.MipLODBias))
      DeferredErrs = joinErrors(std::move(DeferredErrs),
                                make_error<RootSignatureValidationError<float>>(
                                    "MipLODBias", Sampler.MipLODBias));

    if (!hlsl::rootsig::verifyMaxAnisotropy(Sampler.MaxAnisotropy))
      DeferredErrs =
          joinErrors(std::move(DeferredErrs),
                     make_error<RootSignatureValidationError<uint32_t>>(
                         "MaxAnisotropy", Sampler.MaxAnisotropy));

    if (!hlsl::rootsig::verifyComparisonFunc(Sampler.ComparisonFunc))
      DeferredErrs =
          joinErrors(std::move(DeferredErrs),
                     make_error<RootSignatureValidationError<uint32_t>>(
                         "ComparisonFunc", Sampler.ComparisonFunc));

    if (!hlsl::rootsig::verifyBorderColor(Sampler.BorderColor))
      DeferredErrs =
          joinErrors(std::move(DeferredErrs),
                     make_error<RootSignatureValidationError<uint32_t>>(
                         "BorderColor", Sampler.BorderColor));

    if (!hlsl::rootsig::verifyLOD(Sampler.MinLOD))
      DeferredErrs = joinErrors(std::move(DeferredErrs),
                                make_error<RootSignatureValidationError<float>>(
                                    "MinLOD", Sampler.MinLOD));

    if (!hlsl::rootsig::verifyLOD(Sampler.MaxLOD))
      DeferredErrs = joinErrors(std::move(DeferredErrs),
                                make_error<RootSignatureValidationError<float>>(
                                    "MaxLOD", Sampler.MaxLOD));

    if (!hlsl::rootsig::verifyRegisterValue(Sampler.ShaderRegister))
      DeferredErrs =
          joinErrors(std::move(DeferredErrs),
                     make_error<RootSignatureValidationError<uint32_t>>(
                         "ShaderRegister", Sampler.ShaderRegister));

    if (!hlsl::rootsig::verifyRegisterSpace(Sampler.RegisterSpace))
      DeferredErrs =
          joinErrors(std::move(DeferredErrs),
                     make_error<RootSignatureValidationError<uint32_t>>(
                         "RegisterSpace", Sampler.RegisterSpace));

    if (!dxbc::isValidShaderVisibility(Sampler.ShaderVisibility))
      DeferredErrs =
          joinErrors(std::move(DeferredErrs),
                     make_error<RootSignatureValidationError<uint32_t>>(
                         "ShaderVisibility", Sampler.ShaderVisibility));
  }

  return DeferredErrs;
}

Expected<mcdxbc::RootSignatureDesc>
MetadataParser::ParseRootSignature(uint32_t Version) {
  Error DeferredErrs = Error::success();
  mcdxbc::RootSignatureDesc RSD;
  RSD.Version = Version;
  for (const auto &Operand : Root->operands()) {
    MDNode *Element = dyn_cast<MDNode>(Operand);
    if (Element == nullptr)
      return joinErrors(std::move(DeferredErrs),
                        make_error<GenericRSMetadataError>(
                            "Missing Root Element Metadata Node.", nullptr));

    if (auto Err = parseRootSignatureElement(RSD, Element))
      DeferredErrs = joinErrors(std::move(DeferredErrs), std::move(Err));
  }

  if (auto Err = validateRootSignature(RSD))
    DeferredErrs = joinErrors(std::move(DeferredErrs), std::move(Err));

  if (DeferredErrs)
    return std::move(DeferredErrs);

  return std::move(RSD);
}
} // namespace rootsig
} // namespace hlsl
} // namespace llvm
