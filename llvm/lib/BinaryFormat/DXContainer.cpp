
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

#define COMPONENT_PRECISION(Val, Enum) {#Enum, SigMinPrecision::Enum},

static const EnumEntry<SigMinPrecision> SigMinPrecisionNames[] = {
#include "llvm/BinaryFormat/DXContainerConstants.def"
};

ArrayRef<EnumEntry<SigMinPrecision>> dxbc::getSigMinPrecisions() {
  return ArrayRef(SigMinPrecisionNames);
}

#define D3D_SYSTEM_VALUE(Val, Enum) {#Enum, D3DSystemValue::Enum},

static const EnumEntry<D3DSystemValue> D3DSystemValueNames[] = {
#include "llvm/BinaryFormat/DXContainerConstants.def"
};

ArrayRef<EnumEntry<D3DSystemValue>> dxbc::getD3DSystemValues() {
  return ArrayRef(D3DSystemValueNames);
}

#define COMPONENT_TYPE(Val, Enum) {#Enum, SigComponentType::Enum},

static const EnumEntry<SigComponentType> SigComponentTypes[] = {
#include "llvm/BinaryFormat/DXContainerConstants.def"
};

ArrayRef<EnumEntry<SigComponentType>> dxbc::getSigComponentTypes() {
  return ArrayRef(SigComponentTypes);
}

static const EnumEntry<RootFlags> RootFlagNames[] = {
#define ROOT_SIGNATURE_FLAG(Val, Enum) {#Enum, RootFlags::Enum},
#include "llvm/BinaryFormat/DXContainerConstants.def"
};

ArrayRef<EnumEntry<RootFlags>> dxbc::getRootFlags() {
  return ArrayRef(RootFlagNames);
}

static const EnumEntry<RootDescriptorFlags> RootDescriptorFlagNames[] = {
#define ROOT_DESCRIPTOR_FLAG(Val, Enum, Flag)                                  \
  {#Enum, RootDescriptorFlags::Enum},
#include "llvm/BinaryFormat/DXContainerConstants.def"
};

ArrayRef<EnumEntry<RootDescriptorFlags>> dxbc::getRootDescriptorFlags() {
  return ArrayRef(RootDescriptorFlagNames);
}

static const EnumEntry<DescriptorRangeFlags> DescriptorRangeFlagNames[] = {
#define DESCRIPTOR_RANGE_FLAG(Val, Enum, Flag)                                 \
  {#Enum, DescriptorRangeFlags::Enum},
#include "llvm/BinaryFormat/DXContainerConstants.def"
};

ArrayRef<EnumEntry<DescriptorRangeFlags>> dxbc::getDescriptorRangeFlags() {
  return ArrayRef(DescriptorRangeFlagNames);
}

static const EnumEntry<StaticSamplerFlags> StaticSamplerFlagNames[] = {
#define STATIC_SAMPLER_FLAG(Val, Enum, Flag) {#Enum, StaticSamplerFlags::Enum},
#include "llvm/BinaryFormat/DXContainerConstants.def"
};

ArrayRef<EnumEntry<StaticSamplerFlags>> dxbc::getStaticSamplerFlags() {
  return ArrayRef(StaticSamplerFlagNames);
}

#define SHADER_VISIBILITY(Val, Enum) {#Enum, ShaderVisibility::Enum},

static const EnumEntry<ShaderVisibility> ShaderVisibilityValues[] = {
#include "llvm/BinaryFormat/DXContainerConstants.def"
};

ArrayRef<EnumEntry<ShaderVisibility>> dxbc::getShaderVisibility() {
  return ArrayRef(ShaderVisibilityValues);
}

#define FILTER(Val, Enum) {#Enum, SamplerFilter::Enum},

static const EnumEntry<SamplerFilter> SamplerFilterNames[] = {
#include "llvm/BinaryFormat/DXContainerConstants.def"
};

ArrayRef<EnumEntry<SamplerFilter>> dxbc::getSamplerFilters() {
  return ArrayRef(SamplerFilterNames);
}

#define TEXTURE_ADDRESS_MODE(Val, Enum) {#Enum, TextureAddressMode::Enum},

static const EnumEntry<TextureAddressMode> TextureAddressModeNames[] = {
#include "llvm/BinaryFormat/DXContainerConstants.def"
};

ArrayRef<EnumEntry<TextureAddressMode>> dxbc::getTextureAddressModes() {
  return ArrayRef(TextureAddressModeNames);
}

#define COMPARISON_FUNC(Val, Enum) {#Enum, ComparisonFunc::Enum},

static const EnumEntry<ComparisonFunc> ComparisonFuncNames[] = {
#include "llvm/BinaryFormat/DXContainerConstants.def"
};

ArrayRef<EnumEntry<ComparisonFunc>> dxbc::getComparisonFuncs() {
  return ArrayRef(ComparisonFuncNames);
}

#define STATIC_BORDER_COLOR(Val, Enum) {#Enum, StaticBorderColor::Enum},

static const EnumEntry<StaticBorderColor> StaticBorderColorValues[] = {
#include "llvm/BinaryFormat/DXContainerConstants.def"
};

ArrayRef<EnumEntry<StaticBorderColor>> dxbc::getStaticBorderColors() {
  return ArrayRef(StaticBorderColorValues);
}

#define ROOT_PARAMETER(Val, Enum) {#Enum, RootParameterType::Enum},

static const EnumEntry<RootParameterType> RootParameterTypes[] = {
#include "llvm/BinaryFormat/DXContainerConstants.def"
};

ArrayRef<EnumEntry<RootParameterType>> dxbc::getRootParameterTypes() {
  return ArrayRef(RootParameterTypes);
}

#define SEMANTIC_KIND(Val, Enum) {#Enum, PSV::SemanticKind::Enum},

static const EnumEntry<PSV::SemanticKind> SemanticKindNames[] = {
#include "llvm/BinaryFormat/DXContainerConstants.def"
};

ArrayRef<EnumEntry<PSV::SemanticKind>> PSV::getSemanticKinds() {
  return ArrayRef(SemanticKindNames);
}

#define COMPONENT_TYPE(Val, Enum) {#Enum, PSV::ComponentType::Enum},

static const EnumEntry<PSV::ComponentType> ComponentTypeNames[] = {
#include "llvm/BinaryFormat/DXContainerConstants.def"
};

ArrayRef<EnumEntry<PSV::ComponentType>> PSV::getComponentTypes() {
  return ArrayRef(ComponentTypeNames);
}

#define INTERPOLATION_MODE(Val, Enum) {#Enum, PSV::InterpolationMode::Enum},

static const EnumEntry<PSV::InterpolationMode> InterpolationModeNames[] = {
#include "llvm/BinaryFormat/DXContainerConstants.def"
};

ArrayRef<EnumEntry<PSV::InterpolationMode>> PSV::getInterpolationModes() {
  return ArrayRef(InterpolationModeNames);
}

#define RESOURCE_TYPE(Val, Enum) {#Enum, PSV::ResourceType::Enum},

static const EnumEntry<PSV::ResourceType> ResourceTypeNames[] = {
#include "llvm/BinaryFormat/DXContainerConstants.def"
};

ArrayRef<EnumEntry<PSV::ResourceType>> PSV::getResourceTypes() {
  return ArrayRef(ResourceTypeNames);
}

#define RESOURCE_KIND(Val, Enum) {#Enum, PSV::ResourceKind::Enum},

static const EnumEntry<PSV::ResourceKind> ResourceKindNames[] = {
#include "llvm/BinaryFormat/DXContainerConstants.def"
};

ArrayRef<EnumEntry<PSV::ResourceKind>> PSV::getResourceKinds() {
  return ArrayRef(ResourceKindNames);
}

static const EnumEntry<SourceInfo::SectionType> SectionNames[] = {
#define SOURCE_INFO_TYPE(Num, Val) {#Val, SourceInfo::SectionType::Val},
#include "llvm/BinaryFormat/DXContainerConstants.def"
};

ArrayRef<EnumEntry<SourceInfo::SectionType>> SourceInfo::getSectionTypes() {
  return ArrayRef(SectionNames);
}

StringRef SourceInfo::getSectionName(SourceInfo::SectionType Type) {
  auto V = to_underlying(Type);
  if (!isValidSectionType(V))
    return StringRef();
  return getSectionTypes()[V].getName();
}

static const EnumEntry<SourceInfo::Contents::CompressionType>
    CompressionTypes[] = {
#define COMPRESSION_TYPE(Num, Val)                                             \
  {#Val, SourceInfo::Contents::CompressionType::Val},
#include "llvm/BinaryFormat/DXContainerConstants.def"
};

ArrayRef<EnumEntry<SourceInfo::Contents::CompressionType>>
SourceInfo::Contents::getCompressionTypes() {
  return ArrayRef(CompressionTypes);
}
