
//===-- llvm/BinaryFormat/DXContainer.cpp - DXContainer Utils ----*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains utility functions for working with DXContainers.
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ScopedPrinter.h"

using namespace llvm;
using namespace llvm::dxbc;

#define ROOT_PARAMETER(Val, Enum)                                              \
  case Val:                                                                    \
    return true;
bool llvm::dxbc::isValidParameterType(uint32_t V) {
  switch (V) {
#include "llvm/BinaryFormat/DXContainerConstants.def"
  }
  return false;
}

bool llvm::dxbc::isValidRangeType(uint32_t V) {
  return V <= llvm::to_underlying(dxil::ResourceClass::LastEntry);
}

#define SHADER_VISIBILITY(Val, Enum)                                           \
  case Val:                                                                    \
    return true;
bool llvm::dxbc::isValidShaderVisibility(uint32_t V) {
  switch (V) {
#include "llvm/BinaryFormat/DXContainerConstants.def"
  }
  return false;
}

#define FILTER(Val, Enum)                                                      \
  case Val:                                                                    \
    return true;
bool llvm::dxbc::isValidSamplerFilter(uint32_t V) {
  switch (V) {
#include "llvm/BinaryFormat/DXContainerConstants.def"
  }
  return false;
}

#define TEXTURE_ADDRESS_MODE(Val, Enum)                                        \
  case Val:                                                                    \
    return true;
bool llvm::dxbc::isValidAddress(uint32_t V) {
  switch (V) {
#include "llvm/BinaryFormat/DXContainerConstants.def"
  }
  return false;
}

#define COMPARISON_FUNC(Val, Enum)                                             \
  case Val:                                                                    \
    return true;
bool llvm::dxbc::isValidComparisonFunc(uint32_t V) {
  switch (V) {
#include "llvm/BinaryFormat/DXContainerConstants.def"
  }
  return false;
}

#define STATIC_BORDER_COLOR(Val, Enum)                                         \
  case Val:                                                                    \
    return true;
bool llvm::dxbc::isValidBorderColor(uint32_t V) {
  switch (V) {
#include "llvm/BinaryFormat/DXContainerConstants.def"
  }
  return false;
}

template <typename FlagT>
static bool isValidFlags(std::underlying_type_t<FlagT> V) {
  decltype(V) LargestValue =
      llvm::to_underlying(FlagT::LLVM_BITMASK_LARGEST_ENUMERATOR);
  return V < NextPowerOf2(LargestValue);
}

bool llvm::dxbc::isValidRootDesciptorFlags(uint32_t V) {
  return isValidFlags<dxbc::RootDescriptorFlags>(V);
}

bool llvm::dxbc::isValidDescriptorRangeFlags(uint32_t V) {
  return isValidFlags<dxbc::DescriptorRangeFlags>(V);
}

bool llvm::dxbc::isValidStaticSamplerFlags(uint32_t V) {
  return isValidFlags<dxbc::StaticSamplerFlags>(V);
}

bool llvm::dxbc::isValidCompilerVersionFlags(uint32_t V) {
  return isValidFlags<dxbc::CompilerVersionFlags>(V);
}

template <typename EnumT>
static bool isValidEnumValue(std::underlying_type_t<EnumT> V) {
  decltype(V) LargestValue =
      llvm::to_underlying(EnumT::LLVM_BITMASK_LARGEST_ENUMERATOR);
  return V <= LargestValue;
}

bool llvm::dxbc::SourceInfo::Contents::isValidCompressionType(uint16_t V) {
  return isValidEnumValue<CompressionType>(V);
}

bool SourceInfo::isValidSectionType(uint16_t V) {
  return isValidEnumValue<SourceInfo::SectionType>(V);
}

dxbc::PartType dxbc::parsePartType(StringRef S) {
#define CONTAINER_PART(PartName) .Case(#PartName, PartType::PartName)
  return StringSwitch<dxbc::PartType>(S)
#include "llvm/BinaryFormat/DXContainerConstants.def"
      .Default(dxbc::PartType::Unknown);
}

bool dxbc::isDebugProgramPart(PartType PT) { return PT == PartType::ILDB; }

const char *dxbc::getProgramPartName(bool IsDebug) {
  return IsDebug ? "ILDB" : "DXIL";
}

bool dxbc::isProgramPart(StringRef PartName) {
  return PartName == "DXIL" || PartName == "ILDB";
}

bool ShaderHash::isPopulated() {
  static uint8_t Zeros[16] = {0};
  return Flags > 0 || 0 != memcmp(&Digest, &Zeros, 16);
}

EnumStrings<SigMinPrecision> dxbc::getSigMinPrecisions() {
  constexpr EnumStringDef<SigMinPrecision> SigMinPrecisionNameDefs[] = {
#define COMPONENT_PRECISION(Val, Enum) {{#Enum}, SigMinPrecision::Enum},
#include "llvm/BinaryFormat/DXContainerConstants.def"
  };
  static constexpr auto SigMinPrecisionNames =
      BUILD_ENUM_STRINGS(SigMinPrecisionNameDefs);
  return EnumStrings(SigMinPrecisionNames);
}

EnumStrings<D3DSystemValue> dxbc::getD3DSystemValues() {
  constexpr EnumStringDef<D3DSystemValue> D3DSystemValueNameDefs[] = {
#define D3D_SYSTEM_VALUE(Val, Enum) {{#Enum}, D3DSystemValue::Enum},
#include "llvm/BinaryFormat/DXContainerConstants.def"
  };
  static constexpr auto D3DSystemValueNames =
      BUILD_ENUM_STRINGS(D3DSystemValueNameDefs);
  return EnumStrings(D3DSystemValueNames);
}

EnumStrings<SigComponentType> dxbc::getSigComponentTypes() {
  constexpr EnumStringDef<SigComponentType> SigComponentTypeDefs[] = {
#define COMPONENT_TYPE(Val, Enum) {{#Enum}, SigComponentType::Enum},
#include "llvm/BinaryFormat/DXContainerConstants.def"
  };
  static constexpr auto SigComponentTypes =
      BUILD_ENUM_STRINGS(SigComponentTypeDefs);
  return EnumStrings(SigComponentTypes);
}

EnumStrings<RootFlags> dxbc::getRootFlags() {
  constexpr EnumStringDef<RootFlags> RootFlagNameDefs[] = {
#define ROOT_SIGNATURE_FLAG(Val, Enum) {{#Enum}, RootFlags::Enum},
#include "llvm/BinaryFormat/DXContainerConstants.def"
  };
  static constexpr auto RootFlagNames = BUILD_ENUM_STRINGS(RootFlagNameDefs);
  return EnumStrings(RootFlagNames);
}

EnumStrings<RootDescriptorFlags> dxbc::getRootDescriptorFlags() {
  constexpr EnumStringDef<RootDescriptorFlags> RootDescriptorFlagNameDefs[] = {
#define ROOT_DESCRIPTOR_FLAG(Val, Enum, Flag)                                  \
  {{#Enum}, RootDescriptorFlags::Enum},
#include "llvm/BinaryFormat/DXContainerConstants.def"
  };
  static constexpr auto RootDescriptorFlagNames =
      BUILD_ENUM_STRINGS(RootDescriptorFlagNameDefs);
  return EnumStrings(RootDescriptorFlagNames);
}

EnumStrings<DescriptorRangeFlags> dxbc::getDescriptorRangeFlags() {
  constexpr EnumStringDef<DescriptorRangeFlags> DescriptorRangeFlagNameDefs[] =
      {
#define DESCRIPTOR_RANGE_FLAG(Val, Enum, Flag)                                 \
  {{#Enum}, DescriptorRangeFlags::Enum},
#include "llvm/BinaryFormat/DXContainerConstants.def"
      };
  static constexpr auto DescriptorRangeFlagNames =
      BUILD_ENUM_STRINGS(DescriptorRangeFlagNameDefs);
  return EnumStrings(DescriptorRangeFlagNames);
}

EnumStrings<StaticSamplerFlags> dxbc::getStaticSamplerFlags() {
  constexpr EnumStringDef<StaticSamplerFlags> StaticSamplerFlagNameDefs[] = {
#define STATIC_SAMPLER_FLAG(Val, Enum, Flag)                                   \
  {{#Enum}, StaticSamplerFlags::Enum},
#include "llvm/BinaryFormat/DXContainerConstants.def"
  };
  static constexpr auto StaticSamplerFlagNames =
      BUILD_ENUM_STRINGS(StaticSamplerFlagNameDefs);
  return EnumStrings(StaticSamplerFlagNames);
}

EnumStrings<ShaderVisibility> dxbc::getShaderVisibility() {
  constexpr EnumStringDef<ShaderVisibility> ShaderVisibilityValueDefs[] = {
#define SHADER_VISIBILITY(Val, Enum) {{#Enum}, ShaderVisibility::Enum},
#include "llvm/BinaryFormat/DXContainerConstants.def"
  };
  static constexpr auto ShaderVisibilityValues =
      BUILD_ENUM_STRINGS(ShaderVisibilityValueDefs);
  return EnumStrings(ShaderVisibilityValues);
}

EnumStrings<SamplerFilter> dxbc::getSamplerFilters() {
  constexpr EnumStringDef<SamplerFilter> SamplerFilterNameDefs[] = {
#define FILTER(Val, Enum) {{#Enum}, SamplerFilter::Enum},
#include "llvm/BinaryFormat/DXContainerConstants.def"
  };
  static constexpr auto SamplerFilterNames =
      BUILD_ENUM_STRINGS(SamplerFilterNameDefs);
  return EnumStrings(SamplerFilterNames);
}

EnumStrings<TextureAddressMode> dxbc::getTextureAddressModes() {
  constexpr EnumStringDef<TextureAddressMode> TextureAddressModeNameDefs[] = {
#define TEXTURE_ADDRESS_MODE(Val, Enum) {{#Enum}, TextureAddressMode::Enum},
#include "llvm/BinaryFormat/DXContainerConstants.def"
  };
  static constexpr auto TextureAddressModeNames =
      BUILD_ENUM_STRINGS(TextureAddressModeNameDefs);
  return EnumStrings(TextureAddressModeNames);
}

EnumStrings<ComparisonFunc> dxbc::getComparisonFuncs() {
  constexpr EnumStringDef<ComparisonFunc> ComparisonFuncNameDefs[] = {
#define COMPARISON_FUNC(Val, Enum) {{#Enum}, ComparisonFunc::Enum},
#include "llvm/BinaryFormat/DXContainerConstants.def"
  };
  static constexpr auto ComparisonFuncNames =
      BUILD_ENUM_STRINGS(ComparisonFuncNameDefs);
  return EnumStrings(ComparisonFuncNames);
}

EnumStrings<StaticBorderColor> dxbc::getStaticBorderColors() {
  constexpr EnumStringDef<StaticBorderColor> StaticBorderColorValueDefs[] = {
#define STATIC_BORDER_COLOR(Val, Enum) {{#Enum}, StaticBorderColor::Enum},
#include "llvm/BinaryFormat/DXContainerConstants.def"
  };
  static constexpr auto StaticBorderColorValues =
      BUILD_ENUM_STRINGS(StaticBorderColorValueDefs);
  return EnumStrings(StaticBorderColorValues);
}

EnumStrings<RootParameterType> dxbc::getRootParameterTypes() {
  constexpr EnumStringDef<RootParameterType> RootParameterTypeDefs[] = {
#define ROOT_PARAMETER(Val, Enum) {{#Enum}, RootParameterType::Enum},
#include "llvm/BinaryFormat/DXContainerConstants.def"
  };
  static constexpr auto RootParameterTypes =
      BUILD_ENUM_STRINGS(RootParameterTypeDefs);
  return EnumStrings(RootParameterTypes);
}

EnumStrings<PSV::SemanticKind> PSV::getSemanticKinds() {
  constexpr EnumStringDef<PSV::SemanticKind> SemanticKindNameDefs[] = {
#define SEMANTIC_KIND(Val, Enum) {{#Enum}, PSV::SemanticKind::Enum},
#include "llvm/BinaryFormat/DXContainerConstants.def"
  };
  static constexpr auto SemanticKindNames =
      BUILD_ENUM_STRINGS(SemanticKindNameDefs);
  return EnumStrings(SemanticKindNames);
}

EnumStrings<PSV::ComponentType> PSV::getComponentTypes() {
  constexpr EnumStringDef<PSV::ComponentType> ComponentTypeNameDefs[] = {
#define COMPONENT_TYPE(Val, Enum) {{#Enum}, PSV::ComponentType::Enum},
#include "llvm/BinaryFormat/DXContainerConstants.def"
  };
  static constexpr auto ComponentTypeNames =
      BUILD_ENUM_STRINGS(ComponentTypeNameDefs);
  return EnumStrings(ComponentTypeNames);
}

EnumStrings<PSV::InterpolationMode> PSV::getInterpolationModes() {
  constexpr EnumStringDef<PSV::InterpolationMode> InterpolationModeNameDefs[] =
      {
#define INTERPOLATION_MODE(Val, Enum) {{#Enum}, PSV::InterpolationMode::Enum},
#include "llvm/BinaryFormat/DXContainerConstants.def"
      };
  static constexpr auto InterpolationModeNames =
      BUILD_ENUM_STRINGS(InterpolationModeNameDefs);
  return EnumStrings(InterpolationModeNames);
}

EnumStrings<PSV::ResourceType> PSV::getResourceTypes() {
  constexpr EnumStringDef<PSV::ResourceType> ResourceTypeNameDefs[] = {
#define RESOURCE_TYPE(Val, Enum) {{#Enum}, PSV::ResourceType::Enum},
#include "llvm/BinaryFormat/DXContainerConstants.def"
  };
  static constexpr auto ResourceTypeNames =
      BUILD_ENUM_STRINGS(ResourceTypeNameDefs);
  return EnumStrings(ResourceTypeNames);
}

EnumStrings<PSV::ResourceKind> PSV::getResourceKinds() {
  constexpr EnumStringDef<PSV::ResourceKind> ResourceKindNameDefs[] = {
#define RESOURCE_KIND(Val, Enum) {{#Enum}, PSV::ResourceKind::Enum},
#include "llvm/BinaryFormat/DXContainerConstants.def"
  };
  static constexpr auto ResourceKindNames =
      BUILD_ENUM_STRINGS(ResourceKindNameDefs);
  return EnumStrings(ResourceKindNames);
}

EnumStrings<SourceInfo::SectionType> SourceInfo::getSectionTypes() {
  constexpr EnumStringDef<SectionType> SectionNameDefs[] = {
#define SOURCE_INFO_TYPE(Num, Val) {{#Val}, SourceInfo::SectionType::Val},
#include "llvm/BinaryFormat/DXContainerConstants.def"
  };
  static constexpr auto SectionNames = BUILD_ENUM_STRINGS(SectionNameDefs);
  return EnumStrings(SectionNames);
}

StringRef SourceInfo::getSectionName(SourceInfo::SectionType Type) {
  auto V = to_underlying(Type);
  if (!isValidSectionType(V))
    return StringRef();
  return getSectionTypes()[V].name();
}

EnumStrings<SourceInfo::Contents::CompressionType>
SourceInfo::Contents::getCompressionTypes() {
  constexpr EnumStringDef<CompressionType> CompressionTypeDefs[] = {
#define COMPRESSION_TYPE(Num, Val)                                             \
  {{#Val}, SourceInfo::Contents::CompressionType::Val},
#include "llvm/BinaryFormat/DXContainerConstants.def"
  };
  static constexpr auto CompressionTypes =
      BUILD_ENUM_STRINGS(CompressionTypeDefs);
  return EnumStrings(CompressionTypes);
}
