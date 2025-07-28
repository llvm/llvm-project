//===- DXContainerYAML.h - DXContainer YAMLIO implementation ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares classes for handling the YAML representation
/// of DXContainer.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECTYAML_DXCONTAINERYAML_H
#define LLVM_OBJECTYAML_DXCONTAINERYAML_H

#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/Object/DXContainer.h"
#include "llvm/ObjectYAML/YAML.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/YAMLTraits.h"
#include <array>
#include <optional>
#include <string>
#include <vector>

namespace llvm {
namespace DXContainerYAML {

struct VersionTuple {
  uint16_t Major;
  uint16_t Minor;
};

// The optional header fields are required in the binary and will be populated
// when reading from binary, but can be omitted in the YAML text because the
// emitter can calculate them.
struct FileHeader {
  std::vector<llvm::yaml::Hex8> Hash;
  VersionTuple Version;
  std::optional<uint32_t> FileSize;
  uint32_t PartCount;
  std::optional<std::vector<uint32_t>> PartOffsets;
};

struct DXILProgram {
  uint8_t MajorVersion;
  uint8_t MinorVersion;
  uint16_t ShaderKind;
  std::optional<uint32_t> Size;
  uint16_t DXILMajorVersion;
  uint16_t DXILMinorVersion;
  std::optional<uint32_t> DXILOffset;
  std::optional<uint32_t> DXILSize;
  std::optional<std::vector<llvm::yaml::Hex8>> DXIL;
};

#define SHADER_FEATURE_FLAG(Num, DxilModuleNum, Val, Str) bool Val = false;
struct ShaderFeatureFlags {
  ShaderFeatureFlags() = default;
  LLVM_ABI ShaderFeatureFlags(uint64_t FlagData);
  LLVM_ABI uint64_t getEncodedFlags();
#include "llvm/BinaryFormat/DXContainerConstants.def"
};

struct ShaderHash {
  ShaderHash() = default;
  LLVM_ABI ShaderHash(const dxbc::ShaderHash &Data);

  bool IncludesSource;
  std::vector<llvm::yaml::Hex8> Digest;
};

struct RootConstantsYaml {
  uint32_t ShaderRegister;
  uint32_t RegisterSpace;
  uint32_t Num32BitValues;
};

struct RootDescriptorYaml {
  RootDescriptorYaml() = default;

  uint32_t ShaderRegister;
  uint32_t RegisterSpace;

  LLVM_ABI uint32_t getEncodedFlags() const;

#define ROOT_DESCRIPTOR_FLAG(Num, Enum, Flag) bool Enum = false;
#include "llvm/BinaryFormat/DXContainerConstants.def"
};

struct DescriptorRangeYaml {
  uint32_t RangeType;
  uint32_t NumDescriptors;
  uint32_t BaseShaderRegister;
  uint32_t RegisterSpace;
  uint32_t OffsetInDescriptorsFromTableStart;

  LLVM_ABI uint32_t getEncodedFlags() const;

#define DESCRIPTOR_RANGE_FLAG(Num, Enum, Flag) bool Enum = false;
#include "llvm/BinaryFormat/DXContainerConstants.def"
};

struct DescriptorTableYaml {
  uint32_t NumRanges;
  uint32_t RangesOffset;
  SmallVector<DescriptorRangeYaml> Ranges;
};

struct RootParameterHeaderYaml {
  uint32_t Type;
  uint32_t Visibility;
  uint32_t Offset;

  RootParameterHeaderYaml(){};
  RootParameterHeaderYaml(uint32_t T) : Type(T) {}
};

struct RootParameterLocationYaml {
  RootParameterHeaderYaml Header;
  std::optional<size_t> IndexInSignature;

  RootParameterLocationYaml(){};
  explicit RootParameterLocationYaml(RootParameterHeaderYaml Header)
      : Header(Header) {}
};

struct RootParameterYamlDesc {
  SmallVector<RootParameterLocationYaml> Locations;

  SmallVector<RootConstantsYaml> Constants;
  SmallVector<RootDescriptorYaml> Descriptors;
  SmallVector<DescriptorTableYaml> Tables;

  template <typename T>
  T &getOrInsertImpl(RootParameterLocationYaml &ParamDesc,
                     SmallVectorImpl<T> &Container) {
    if (!ParamDesc.IndexInSignature) {
      ParamDesc.IndexInSignature = Container.size();
      Container.emplace_back();
    }
    return Container[*ParamDesc.IndexInSignature];
  }

  RootConstantsYaml &
  getOrInsertConstants(RootParameterLocationYaml &ParamDesc) {
    return getOrInsertImpl(ParamDesc, Constants);
  }

  RootDescriptorYaml &
  getOrInsertDescriptor(RootParameterLocationYaml &ParamDesc) {
    return getOrInsertImpl(ParamDesc, Descriptors);
  }

  DescriptorTableYaml &getOrInsertTable(RootParameterLocationYaml &ParamDesc) {
    return getOrInsertImpl(ParamDesc, Tables);
  }

  void insertLocation(RootParameterLocationYaml &Location) {
    Locations.push_back(Location);
  }
};

struct StaticSamplerYamlDesc {
  uint32_t Filter = llvm::to_underlying(dxbc::SamplerFilter::Anisotropic);
  uint32_t AddressU = llvm::to_underlying(dxbc::TextureAddressMode::Wrap);
  uint32_t AddressV = llvm::to_underlying(dxbc::TextureAddressMode::Wrap);
  uint32_t AddressW = llvm::to_underlying(dxbc::TextureAddressMode::Wrap);
  float MipLODBias = 0.f;
  uint32_t MaxAnisotropy = 16u;
  uint32_t ComparisonFunc =
      llvm::to_underlying(dxbc::ComparisonFunc::LessEqual);
  uint32_t BorderColor =
      llvm::to_underlying(dxbc::StaticBorderColor::OpaqueWhite);
  float MinLOD = 0.f;
  float MaxLOD = std::numeric_limits<float>::max();
  uint32_t ShaderRegister;
  uint32_t RegisterSpace;
  uint32_t ShaderVisibility;
};

struct RootSignatureYamlDesc {
  RootSignatureYamlDesc() = default;

  uint32_t Version;
  uint32_t NumRootParameters;
  uint32_t RootParametersOffset;
  uint32_t NumStaticSamplers;
  uint32_t StaticSamplersOffset;

  RootParameterYamlDesc Parameters;
  SmallVector<StaticSamplerYamlDesc> StaticSamplers;

  LLVM_ABI uint32_t getEncodedFlags();

  iterator_range<StaticSamplerYamlDesc *> samplers() {
    return make_range(StaticSamplers.begin(), StaticSamplers.end());
  }

  LLVM_ABI static llvm::Expected<DXContainerYAML::RootSignatureYamlDesc>
  create(const object::DirectX::RootSignature &Data);

#define ROOT_SIGNATURE_FLAG(Num, Val) bool Val = false;
#include "llvm/BinaryFormat/DXContainerConstants.def"
};

using ResourceFlags = dxbc::PSV::ResourceFlags;
using ResourceBindInfo = dxbc::PSV::v2::ResourceBindInfo;

struct SignatureElement {
  SignatureElement() = default;

  SignatureElement(dxbc::PSV::v0::SignatureElement El, StringRef StringTable,
                   ArrayRef<uint32_t> IdxTable)
      : Name(StringTable.substr(El.NameOffset,
                                StringTable.find('\0', El.NameOffset) -
                                    El.NameOffset)),
        Indices(IdxTable.slice(El.IndicesOffset, El.Rows)),
        StartRow(El.StartRow), Cols(El.Cols), StartCol(El.StartCol),
        Allocated(El.Allocated != 0), Kind(El.Kind), Type(El.Type),
        Mode(El.Mode), DynamicMask(El.DynamicMask), Stream(El.Stream) {}
  StringRef Name;
  SmallVector<uint32_t> Indices;

  uint8_t StartRow;
  uint8_t Cols;
  uint8_t StartCol;
  bool Allocated;
  dxbc::PSV::SemanticKind Kind;

  dxbc::PSV::ComponentType Type;
  dxbc::PSV::InterpolationMode Mode;
  llvm::yaml::Hex8 DynamicMask;
  uint8_t Stream;
};

struct PSVInfo {
  // The version field isn't actually encoded in the file, but it is inferred by
  // the size of data regions. We include it in the yaml because it simplifies
  // the format.
  uint32_t Version;

  dxbc::PSV::v3::RuntimeInfo Info;
  uint32_t ResourceStride;
  SmallVector<ResourceBindInfo> Resources;
  SmallVector<SignatureElement> SigInputElements;
  SmallVector<SignatureElement> SigOutputElements;
  SmallVector<SignatureElement> SigPatchOrPrimElements;

  using MaskVector = SmallVector<llvm::yaml::Hex32>;
  std::array<MaskVector, 4> OutputVectorMasks;
  MaskVector PatchOrPrimMasks;
  std::array<MaskVector, 4> InputOutputMap;
  MaskVector InputPatchMap;
  MaskVector PatchOutputMap;

  StringRef EntryName;

  LLVM_ABI void mapInfoForVersion(yaml::IO &IO);

  LLVM_ABI PSVInfo();
  LLVM_ABI PSVInfo(const dxbc::PSV::v0::RuntimeInfo *P, uint16_t Stage);
  LLVM_ABI PSVInfo(const dxbc::PSV::v1::RuntimeInfo *P);
  LLVM_ABI PSVInfo(const dxbc::PSV::v2::RuntimeInfo *P);
  LLVM_ABI PSVInfo(const dxbc::PSV::v3::RuntimeInfo *P, StringRef StringTable);
};

struct SignatureParameter {
  uint32_t Stream;
  std::string Name;
  uint32_t Index;
  dxbc::D3DSystemValue SystemValue;
  dxbc::SigComponentType CompType;
  uint32_t Register;
  uint8_t Mask;
  uint8_t ExclusiveMask;
  dxbc::SigMinPrecision MinPrecision;
};

struct Signature {
  llvm::SmallVector<SignatureParameter> Parameters;
};

struct Part {
  Part() = default;
  Part(std::string N, uint32_t S) : Name(N), Size(S) {}
  std::string Name;
  uint32_t Size;
  std::optional<DXILProgram> Program;
  std::optional<ShaderFeatureFlags> Flags;
  std::optional<ShaderHash> Hash;
  std::optional<PSVInfo> Info;
  std::optional<DXContainerYAML::Signature> Signature;
  std::optional<DXContainerYAML::RootSignatureYamlDesc> RootSignature;
};

struct Object {
  FileHeader Header;
  std::vector<Part> Parts;
};

} // namespace DXContainerYAML
} // namespace llvm

LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::DXContainerYAML::Part)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::DXContainerYAML::ResourceBindInfo)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::DXContainerYAML::SignatureElement)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::DXContainerYAML::PSVInfo::MaskVector)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::DXContainerYAML::SignatureParameter)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::DXContainerYAML::RootParameterLocationYaml)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::DXContainerYAML::DescriptorRangeYaml)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::DXContainerYAML::StaticSamplerYamlDesc)
LLVM_YAML_DECLARE_ENUM_TRAITS(llvm::dxbc::PSV::SemanticKind)
LLVM_YAML_DECLARE_ENUM_TRAITS(llvm::dxbc::PSV::ComponentType)
LLVM_YAML_DECLARE_ENUM_TRAITS(llvm::dxbc::PSV::InterpolationMode)
LLVM_YAML_DECLARE_ENUM_TRAITS(llvm::dxbc::PSV::ResourceType)
LLVM_YAML_DECLARE_ENUM_TRAITS(llvm::dxbc::PSV::ResourceKind)
LLVM_YAML_DECLARE_ENUM_TRAITS(llvm::dxbc::D3DSystemValue)
LLVM_YAML_DECLARE_ENUM_TRAITS(llvm::dxbc::SigComponentType)
LLVM_YAML_DECLARE_ENUM_TRAITS(llvm::dxbc::SigMinPrecision)

namespace llvm {

class raw_ostream;

namespace yaml {

template <> struct MappingTraits<DXContainerYAML::VersionTuple> {
  LLVM_ABI static void mapping(IO &IO, DXContainerYAML::VersionTuple &Version);
};

template <> struct MappingTraits<DXContainerYAML::FileHeader> {
  LLVM_ABI static void mapping(IO &IO, DXContainerYAML::FileHeader &Header);
};

template <> struct MappingTraits<DXContainerYAML::DXILProgram> {
  LLVM_ABI static void mapping(IO &IO, DXContainerYAML::DXILProgram &Program);
};

template <> struct MappingTraits<DXContainerYAML::ShaderFeatureFlags> {
  LLVM_ABI static void mapping(IO &IO,
                               DXContainerYAML::ShaderFeatureFlags &Flags);
};

template <> struct MappingTraits<DXContainerYAML::ShaderHash> {
  LLVM_ABI static void mapping(IO &IO, DXContainerYAML::ShaderHash &Hash);
};

template <> struct MappingTraits<DXContainerYAML::PSVInfo> {
  LLVM_ABI static void mapping(IO &IO, DXContainerYAML::PSVInfo &PSV);
};

template <> struct MappingTraits<DXContainerYAML::Part> {
  LLVM_ABI static void mapping(IO &IO, DXContainerYAML::Part &Version);
};

template <> struct MappingTraits<DXContainerYAML::Object> {
  LLVM_ABI static void mapping(IO &IO, DXContainerYAML::Object &Obj);
};

template <> struct MappingTraits<DXContainerYAML::ResourceFlags> {
  LLVM_ABI static void mapping(IO &IO, DXContainerYAML::ResourceFlags &Flags);
};

template <> struct MappingTraits<DXContainerYAML::ResourceBindInfo> {
  LLVM_ABI static void mapping(IO &IO, DXContainerYAML::ResourceBindInfo &Res);
};

template <> struct MappingTraits<DXContainerYAML::SignatureElement> {
  LLVM_ABI static void mapping(IO &IO,
                               llvm::DXContainerYAML::SignatureElement &El);
};

template <> struct MappingTraits<DXContainerYAML::SignatureParameter> {
  LLVM_ABI static void mapping(IO &IO,
                               llvm::DXContainerYAML::SignatureParameter &El);
};

template <> struct MappingTraits<DXContainerYAML::Signature> {
  LLVM_ABI static void mapping(IO &IO, llvm::DXContainerYAML::Signature &El);
};

template <> struct MappingTraits<DXContainerYAML::RootSignatureYamlDesc> {
  LLVM_ABI static void
  mapping(IO &IO, DXContainerYAML::RootSignatureYamlDesc &RootSignature);
};

template <>
struct MappingContextTraits<DXContainerYAML::RootParameterLocationYaml,
                            DXContainerYAML::RootSignatureYamlDesc> {
  LLVM_ABI static void
  mapping(IO &IO, llvm::DXContainerYAML::RootParameterLocationYaml &L,
          DXContainerYAML::RootSignatureYamlDesc &S);
};

template <> struct MappingTraits<llvm::DXContainerYAML::RootConstantsYaml> {
  LLVM_ABI static void mapping(IO &IO,
                               llvm::DXContainerYAML::RootConstantsYaml &C);
};

template <> struct MappingTraits<llvm::DXContainerYAML::RootDescriptorYaml> {
  LLVM_ABI static void mapping(IO &IO,
                               llvm::DXContainerYAML::RootDescriptorYaml &D);
};

template <> struct MappingTraits<llvm::DXContainerYAML::DescriptorTableYaml> {
  LLVM_ABI static void mapping(IO &IO,
                               llvm::DXContainerYAML::DescriptorTableYaml &D);
};

template <> struct MappingTraits<llvm::DXContainerYAML::DescriptorRangeYaml> {
  LLVM_ABI static void mapping(IO &IO,
                               llvm::DXContainerYAML::DescriptorRangeYaml &D);
};

template <> struct MappingTraits<llvm::DXContainerYAML::StaticSamplerYamlDesc> {
  LLVM_ABI static void mapping(IO &IO,
                               llvm::DXContainerYAML::StaticSamplerYamlDesc &S);
};

} // namespace yaml

} // namespace llvm

#endif // LLVM_OBJECTYAML_DXCONTAINERYAML_H
