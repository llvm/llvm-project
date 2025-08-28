//===- DXContainerYAML.cpp - DXContainer YAMLIO implementation ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines classes for handling the YAML representation of
// DXContainerYAML.
//
//===----------------------------------------------------------------------===//

#include "llvm/ObjectYAML/DXContainerYAML.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ScopedPrinter.h"
#include <cstdint>
#include <system_error>

namespace llvm {

// This assert is duplicated here to leave a breadcrumb of the places that need
// to be updated if flags grow past 64-bits.
static_assert((uint64_t)dxbc::FeatureFlags::NextUnusedBit <= 1ull << 63,
              "Shader flag bits exceed enum size.");

DXContainerYAML::ShaderFeatureFlags::ShaderFeatureFlags(uint64_t FlagData) {
#define SHADER_FEATURE_FLAG(Num, DxilModuleNum, Val, Str)                      \
  Val = (FlagData & (uint64_t)dxbc::FeatureFlags::Val) > 0;
#include "llvm/BinaryFormat/DXContainerConstants.def"
}

template <typename T>
static llvm::Error
readDescriptorRanges(DXContainerYAML::RootParameterHeaderYaml &Header,
                     DXContainerYAML::RootSignatureYamlDesc &RootSigDesc,
                     object::DirectX::DescriptorTableView *DTV) {

  llvm::Expected<object::DirectX::DescriptorTable<T>> TableOrErr =
      DTV->read<T>();
  if (Error E = TableOrErr.takeError())
    return E;
  auto Table = *TableOrErr;

  DXContainerYAML::RootParameterLocationYaml Location(Header);
  DXContainerYAML::DescriptorTableYaml &TableYaml =
      RootSigDesc.Parameters.getOrInsertTable(Location);
  RootSigDesc.Parameters.insertLocation(Location);

  TableYaml.NumRanges = Table.NumRanges;
  TableYaml.RangesOffset = Table.RangesOffset;

  for (const auto &R : Table.Ranges) {
    DXContainerYAML::DescriptorRangeYaml NewR;
    NewR.OffsetInDescriptorsFromTableStart =
        R.OffsetInDescriptorsFromTableStart;
    NewR.NumDescriptors = R.NumDescriptors;
    NewR.BaseShaderRegister = R.BaseShaderRegister;
    NewR.RegisterSpace = R.RegisterSpace;
    NewR.RangeType = R.RangeType;
    if constexpr (std::is_same_v<T, dxbc::RTS0::v2::DescriptorRange>) {
      // Set all flag fields for v2
#define DESCRIPTOR_RANGE_FLAG(Num, Enum, Flag)                                 \
  NewR.Enum =                                                                  \
      (R.Flags & llvm::to_underlying(dxbc::DescriptorRangeFlags::Enum)) != 0;
#include "llvm/BinaryFormat/DXContainerConstants.def"
    }
    TableYaml.Ranges.push_back(NewR);
  }

  return Error::success();
}

llvm::Expected<DXContainerYAML::RootSignatureYamlDesc>
DXContainerYAML::RootSignatureYamlDesc::create(
    const object::DirectX::RootSignature &Data) {

  RootSignatureYamlDesc RootSigDesc;
  uint32_t Version = Data.getVersion();

  RootSigDesc.Version = Version;
  RootSigDesc.NumStaticSamplers = Data.getNumStaticSamplers();
  RootSigDesc.StaticSamplersOffset = Data.getStaticSamplersOffset();
  RootSigDesc.NumRootParameters = Data.getNumRootParameters();
  RootSigDesc.RootParametersOffset = Data.getRootParametersOffset();

  uint32_t Flags = Data.getFlags();
  for (const dxbc::RTS0::v1::RootParameterHeader &PH : Data.param_headers()) {

    if (!dxbc::isValidParameterType(PH.ParameterType))
      return createStringError(std::errc::invalid_argument,
                               "Invalid value for parameter type");

    RootParameterHeaderYaml Header(PH.ParameterType);
    Header.Offset = PH.ParameterOffset;
    Header.Type = PH.ParameterType;

    if (!dxbc::isValidShaderVisibility(PH.ShaderVisibility))
      return createStringError(std::errc::invalid_argument,
                               "Invalid value for shader visibility");

    Header.Visibility = PH.ShaderVisibility;

    llvm::Expected<object::DirectX::RootParameterView> ParamViewOrErr =
        Data.getParameter(PH);
    if (Error E = ParamViewOrErr.takeError())
      return std::move(E);
    object::DirectX::RootParameterView ParamView = ParamViewOrErr.get();

    if (auto *RCV = dyn_cast<object::DirectX::RootConstantView>(&ParamView)) {
      llvm::Expected<dxbc::RTS0::v1::RootConstants> ConstantsOrErr =
          RCV->read();
      if (Error E = ConstantsOrErr.takeError())
        return std::move(E);

      auto Constants = *ConstantsOrErr;
      RootParameterLocationYaml Location(Header);
      RootConstantsYaml &ConstantYaml =
          RootSigDesc.Parameters.getOrInsertConstants(Location);
      RootSigDesc.Parameters.insertLocation(Location);
      ConstantYaml.Num32BitValues = Constants.Num32BitValues;
      ConstantYaml.ShaderRegister = Constants.ShaderRegister;
      ConstantYaml.RegisterSpace = Constants.RegisterSpace;

    } else if (auto *RDV =
                   dyn_cast<object::DirectX::RootDescriptorView>(&ParamView)) {
      llvm::Expected<dxbc::RTS0::v2::RootDescriptor> DescriptorOrErr =
          RDV->read(Version);
      if (Error E = DescriptorOrErr.takeError())
        return std::move(E);
      auto Descriptor = *DescriptorOrErr;
      RootParameterLocationYaml Location(Header);
      RootDescriptorYaml &YamlDescriptor =
          RootSigDesc.Parameters.getOrInsertDescriptor(Location);
      RootSigDesc.Parameters.insertLocation(Location);

      YamlDescriptor.ShaderRegister = Descriptor.ShaderRegister;
      YamlDescriptor.RegisterSpace = Descriptor.RegisterSpace;
      if (Version > 1) {
#define ROOT_DESCRIPTOR_FLAG(Num, Enum, Flag)                                  \
  YamlDescriptor.Enum =                                                        \
      (Descriptor.Flags &                                                      \
       llvm::to_underlying(dxbc::RootDescriptorFlags::Enum)) > 0;
#include "llvm/BinaryFormat/DXContainerConstants.def"
      }
    } else if (auto *DTV =
                   dyn_cast<object::DirectX::DescriptorTableView>(&ParamView)) {
      if (Version == 1) {
        if (Error E = readDescriptorRanges<dxbc::RTS0::v1::DescriptorRange>(
                Header, RootSigDesc, DTV))
          return std::move(E);
      } else if (Version == 2) {
        if (Error E = readDescriptorRanges<dxbc::RTS0::v2::DescriptorRange>(
                Header, RootSigDesc, DTV))
          return std::move(E);
      } else
        llvm_unreachable("Unknown version for DescriptorRanges");
    }
  }

  for (const auto &S : Data.samplers()) {
    StaticSamplerYamlDesc NewS;
    NewS.Filter = S.Filter;
    NewS.AddressU = S.AddressU;
    NewS.AddressV = S.AddressV;
    NewS.AddressW = S.AddressW;
    NewS.MipLODBias = S.MipLODBias;
    NewS.MaxAnisotropy = S.MaxAnisotropy;
    NewS.ComparisonFunc = S.ComparisonFunc;
    NewS.BorderColor = S.BorderColor;
    NewS.MinLOD = S.MinLOD;
    NewS.MaxLOD = S.MaxLOD;
    NewS.ShaderRegister = S.ShaderRegister;
    NewS.RegisterSpace = S.RegisterSpace;
    NewS.ShaderVisibility = S.ShaderVisibility;

    RootSigDesc.StaticSamplers.push_back(NewS);
  }

#define ROOT_SIGNATURE_FLAG(Num, Val)                                          \
  RootSigDesc.Val = (Flags & llvm::to_underlying(dxbc::RootFlags::Val)) > 0;
#include "llvm/BinaryFormat/DXContainerConstants.def"
  return RootSigDesc;
}

uint32_t DXContainerYAML::RootDescriptorYaml::getEncodedFlags() const {
  uint64_t Flags = 0;
#define ROOT_DESCRIPTOR_FLAG(Num, Enum, Flag)                                  \
  if (Enum)                                                                    \
    Flags |= (uint32_t)dxbc::RootDescriptorFlags::Enum;
#include "llvm/BinaryFormat/DXContainerConstants.def"
  return Flags;
}

uint32_t DXContainerYAML::RootSignatureYamlDesc::getEncodedFlags() {
  uint64_t Flag = 0;
#define ROOT_SIGNATURE_FLAG(Num, Val)                                          \
  if (Val)                                                                     \
    Flag |= (uint32_t)dxbc::RootFlags::Val;
#include "llvm/BinaryFormat/DXContainerConstants.def"
  return Flag;
}

uint32_t DXContainerYAML::DescriptorRangeYaml::getEncodedFlags() const {
  uint64_t Flags = 0;
#define DESCRIPTOR_RANGE_FLAG(Num, Enum, Flag)                                 \
  if (Enum)                                                                    \
    Flags |= (uint32_t)dxbc::DescriptorRangeFlags::Enum;
#include "llvm/BinaryFormat/DXContainerConstants.def"
  return Flags;
}

uint64_t DXContainerYAML::ShaderFeatureFlags::getEncodedFlags() {
  uint64_t Flag = 0;
#define SHADER_FEATURE_FLAG(Num, DxilModuleNum, Val, Str)                      \
  if (Val)                                                                     \
    Flag |= (uint64_t)dxbc::FeatureFlags::Val;
#include "llvm/BinaryFormat/DXContainerConstants.def"
  return Flag;
}

DXContainerYAML::ShaderHash::ShaderHash(const dxbc::ShaderHash &Data)
    : IncludesSource((Data.Flags & static_cast<uint32_t>(
                                       dxbc::HashFlags::IncludesSource)) != 0),
      Digest(16, 0) {
  memcpy(Digest.data(), &Data.Digest[0], 16);
}

DXContainerYAML::PSVInfo::PSVInfo() : Version(0) {
  memset(&Info, 0, sizeof(Info));
}

DXContainerYAML::PSVInfo::PSVInfo(const dxbc::PSV::v0::RuntimeInfo *P,
                                  uint16_t Stage)
    : Version(0) {
  memset(&Info, 0, sizeof(Info));
  memcpy(&Info, P, sizeof(dxbc::PSV::v0::RuntimeInfo));

  assert(Stage < std::numeric_limits<uint8_t>::max() &&
         "Stage should be a very small number");
  // We need to bring the stage in separately since it isn't part of the v1 data
  // structure.
  Info.ShaderStage = static_cast<uint8_t>(Stage);
}

DXContainerYAML::PSVInfo::PSVInfo(const dxbc::PSV::v1::RuntimeInfo *P)
    : Version(1) {
  memset(&Info, 0, sizeof(Info));
  memcpy(&Info, P, sizeof(dxbc::PSV::v1::RuntimeInfo));
}

DXContainerYAML::PSVInfo::PSVInfo(const dxbc::PSV::v2::RuntimeInfo *P)
    : Version(2) {
  memset(&Info, 0, sizeof(Info));
  memcpy(&Info, P, sizeof(dxbc::PSV::v2::RuntimeInfo));
}

DXContainerYAML::PSVInfo::PSVInfo(const dxbc::PSV::v3::RuntimeInfo *P,
                                  StringRef StringTable)
    : Version(3),
      EntryName(StringTable.substr(P->EntryNameOffset,
                                   StringTable.find('\0', P->EntryNameOffset) -
                                       P->EntryNameOffset)) {
  memset(&Info, 0, sizeof(Info));
  memcpy(&Info, P, sizeof(dxbc::PSV::v3::RuntimeInfo));
}

namespace yaml {

void MappingTraits<DXContainerYAML::VersionTuple>::mapping(
    IO &IO, DXContainerYAML::VersionTuple &Version) {
  IO.mapRequired("Major", Version.Major);
  IO.mapRequired("Minor", Version.Minor);
}

void MappingTraits<DXContainerYAML::FileHeader>::mapping(
    IO &IO, DXContainerYAML::FileHeader &Header) {
  IO.mapRequired("Hash", Header.Hash);
  IO.mapRequired("Version", Header.Version);
  IO.mapOptional("FileSize", Header.FileSize);
  IO.mapRequired("PartCount", Header.PartCount);
  IO.mapOptional("PartOffsets", Header.PartOffsets);
}

void MappingTraits<DXContainerYAML::DXILProgram>::mapping(
    IO &IO, DXContainerYAML::DXILProgram &Program) {
  IO.mapRequired("MajorVersion", Program.MajorVersion);
  IO.mapRequired("MinorVersion", Program.MinorVersion);
  IO.mapRequired("ShaderKind", Program.ShaderKind);
  IO.mapOptional("Size", Program.Size);
  IO.mapRequired("DXILMajorVersion", Program.DXILMajorVersion);
  IO.mapRequired("DXILMinorVersion", Program.DXILMinorVersion);
  IO.mapOptional("DXILSize", Program.DXILSize);
  IO.mapOptional("DXIL", Program.DXIL);
}

void MappingTraits<DXContainerYAML::ShaderFeatureFlags>::mapping(
    IO &IO, DXContainerYAML::ShaderFeatureFlags &Flags) {
#define SHADER_FEATURE_FLAG(Num, DxilModuleNum, Val, Str)                      \
  IO.mapRequired(#Val, Flags.Val);
#include "llvm/BinaryFormat/DXContainerConstants.def"
}

void MappingTraits<DXContainerYAML::ShaderHash>::mapping(
    IO &IO, DXContainerYAML::ShaderHash &Hash) {
  IO.mapRequired("IncludesSource", Hash.IncludesSource);
  IO.mapRequired("Digest", Hash.Digest);
}

void MappingTraits<DXContainerYAML::PSVInfo>::mapping(
    IO &IO, DXContainerYAML::PSVInfo &PSV) {
  IO.mapRequired("Version", PSV.Version);

  // Store the PSV version in the YAML context.
  void *OldContext = IO.getContext();
  uint32_t Version = PSV.Version;
  IO.setContext(&Version);

  // Restore the YAML context on function exit.
  auto RestoreContext = make_scope_exit([&]() { IO.setContext(OldContext); });

  // Shader stage is only included in binaries for v1 and later, but we always
  // include it since it simplifies parsing and file construction.
  IO.mapRequired("ShaderStage", PSV.Info.ShaderStage);
  PSV.mapInfoForVersion(IO);

  IO.mapRequired("ResourceStride", PSV.ResourceStride);
  IO.mapRequired("Resources", PSV.Resources);
  if (PSV.Version == 0)
    return;
  IO.mapRequired("SigInputElements", PSV.SigInputElements);
  IO.mapRequired("SigOutputElements", PSV.SigOutputElements);
  IO.mapRequired("SigPatchOrPrimElements", PSV.SigPatchOrPrimElements);

  Triple::EnvironmentType Stage = dxbc::getShaderStage(PSV.Info.ShaderStage);
  if (PSV.Info.UsesViewID) {
    MutableArrayRef<SmallVector<llvm::yaml::Hex32>> MutableOutMasks(
        PSV.OutputVectorMasks);
    IO.mapRequired("OutputVectorMasks", MutableOutMasks);
    if (Stage == Triple::EnvironmentType::Hull)
      IO.mapRequired("PatchOrPrimMasks", PSV.PatchOrPrimMasks);
  }
  MutableArrayRef<SmallVector<llvm::yaml::Hex32>> MutableIOMap(
      PSV.InputOutputMap);
  IO.mapRequired("InputOutputMap", MutableIOMap);

  if (Stage == Triple::EnvironmentType::Hull)
    IO.mapRequired("InputPatchMap", PSV.InputPatchMap);

  if (Stage == Triple::EnvironmentType::Domain)
    IO.mapRequired("PatchOutputMap", PSV.PatchOutputMap);
}

void MappingTraits<DXContainerYAML::SignatureParameter>::mapping(
    IO &IO, DXContainerYAML::SignatureParameter &S) {
  IO.mapRequired("Stream", S.Stream);
  IO.mapRequired("Name", S.Name);
  IO.mapRequired("Index", S.Index);
  IO.mapRequired("SystemValue", S.SystemValue);
  IO.mapRequired("CompType", S.CompType);
  IO.mapRequired("Register", S.Register);
  IO.mapRequired("Mask", S.Mask);
  IO.mapRequired("ExclusiveMask", S.ExclusiveMask);
  IO.mapRequired("MinPrecision", S.MinPrecision);
}

void MappingTraits<DXContainerYAML::Signature>::mapping(
    IO &IO, DXContainerYAML::Signature &S) {
  IO.mapRequired("Parameters", S.Parameters);
}

void MappingTraits<DXContainerYAML::RootSignatureYamlDesc>::mapping(
    IO &IO, DXContainerYAML::RootSignatureYamlDesc &S) {
  IO.mapRequired("Version", S.Version);
  IO.mapRequired("NumRootParameters", S.NumRootParameters);
  IO.mapOptional("RootParametersOffset", S.RootParametersOffset, std::nullopt);
  IO.mapRequired("NumStaticSamplers", S.NumStaticSamplers);
  IO.mapOptional("StaticSamplersOffset", S.StaticSamplersOffset, std::nullopt);
  IO.mapRequired("Parameters", S.Parameters.Locations, S);
  IO.mapOptional("Samplers", S.StaticSamplers);
#define ROOT_SIGNATURE_FLAG(Num, Val) IO.mapOptional(#Val, S.Val, false);
#include "llvm/BinaryFormat/DXContainerConstants.def"
}

void MappingTraits<llvm::DXContainerYAML::DescriptorRangeYaml>::mapping(
    IO &IO, llvm::DXContainerYAML::DescriptorRangeYaml &R) {
  IO.mapRequired("RangeType", R.RangeType);
  // handling the edge case where NumDescriptors might be -1
  if (IO.outputting()) {
    if (R.NumDescriptors == UINT_MAX) {
      int32_t NegOne = -1;
      IO.mapRequired("NumDescriptors", NegOne);
    } else
      IO.mapRequired("NumDescriptors", R.NumDescriptors);
  } else {
    int32_t TmpNumDesc = 0;
    IO.mapRequired("NumDescriptors", TmpNumDesc);
    R.NumDescriptors = static_cast<uint32_t>(TmpNumDesc);
  }

  IO.mapRequired("BaseShaderRegister", R.BaseShaderRegister);
  IO.mapRequired("RegisterSpace", R.RegisterSpace);
  IO.mapRequired("OffsetInDescriptorsFromTableStart",
                 R.OffsetInDescriptorsFromTableStart);
#define DESCRIPTOR_RANGE_FLAG(Num, Enum, Flag)                                 \
  IO.mapOptional(#Flag, R.Enum, false);
#include "llvm/BinaryFormat/DXContainerConstants.def"
}

void MappingTraits<llvm::DXContainerYAML::DescriptorTableYaml>::mapping(
    IO &IO, llvm::DXContainerYAML::DescriptorTableYaml &T) {
  IO.mapRequired("NumRanges", T.NumRanges);
  IO.mapOptional("RangesOffset", T.RangesOffset);
  IO.mapRequired("Ranges", T.Ranges);
}

void MappingContextTraits<DXContainerYAML::RootParameterLocationYaml,
                          DXContainerYAML::RootSignatureYamlDesc>::
    mapping(IO &IO, DXContainerYAML::RootParameterLocationYaml &L,
            DXContainerYAML::RootSignatureYamlDesc &S) {
  IO.mapRequired("ParameterType", L.Header.Type);
  IO.mapRequired("ShaderVisibility", L.Header.Visibility);

  switch (L.Header.Type) {
  case llvm::to_underlying(dxbc::RootParameterType::Constants32Bit): {
    DXContainerYAML::RootConstantsYaml &Constants =
        S.Parameters.getOrInsertConstants(L);
    IO.mapRequired("Constants", Constants);
    break;
  }
  case llvm::to_underlying(dxbc::RootParameterType::CBV):
  case llvm::to_underlying(dxbc::RootParameterType::SRV):
  case llvm::to_underlying(dxbc::RootParameterType::UAV): {
    DXContainerYAML::RootDescriptorYaml &Descriptor =
        S.Parameters.getOrInsertDescriptor(L);
    IO.mapRequired("Descriptor", Descriptor);
    break;
  }
  case llvm::to_underlying(dxbc::RootParameterType::DescriptorTable): {
    DXContainerYAML::DescriptorTableYaml &Table =
        S.Parameters.getOrInsertTable(L);
    IO.mapRequired("Table", Table);
    break;
  }
  }
}

void MappingTraits<llvm::DXContainerYAML::RootConstantsYaml>::mapping(
    IO &IO, llvm::DXContainerYAML::RootConstantsYaml &C) {
  IO.mapRequired("Num32BitValues", C.Num32BitValues);
  IO.mapRequired("RegisterSpace", C.RegisterSpace);
  IO.mapRequired("ShaderRegister", C.ShaderRegister);
}

void MappingTraits<llvm::DXContainerYAML::RootDescriptorYaml>::mapping(
    IO &IO, llvm::DXContainerYAML::RootDescriptorYaml &D) {
  IO.mapRequired("RegisterSpace", D.RegisterSpace);
  IO.mapRequired("ShaderRegister", D.ShaderRegister);
#define ROOT_DESCRIPTOR_FLAG(Num, Enum, Flag)                                  \
  IO.mapOptional(#Flag, D.Enum, false);
#include "llvm/BinaryFormat/DXContainerConstants.def"
}

void MappingTraits<llvm::DXContainerYAML::StaticSamplerYamlDesc>::mapping(
    IO &IO, llvm::DXContainerYAML::StaticSamplerYamlDesc &S) {

  IO.mapOptional("Filter", S.Filter);
  IO.mapOptional("AddressU", S.AddressU);
  IO.mapOptional("AddressV", S.AddressV);
  IO.mapOptional("AddressW", S.AddressW);
  IO.mapOptional("MipLODBias", S.MipLODBias);
  IO.mapOptional("MaxAnisotropy", S.MaxAnisotropy);
  IO.mapOptional("ComparisonFunc", S.ComparisonFunc);
  IO.mapOptional("BorderColor", S.BorderColor);
  IO.mapOptional("MinLOD", S.MinLOD);
  IO.mapOptional("MaxLOD", S.MaxLOD);
  IO.mapRequired("ShaderRegister", S.ShaderRegister);
  IO.mapRequired("RegisterSpace", S.RegisterSpace);
  IO.mapRequired("ShaderVisibility", S.ShaderVisibility);
}

void MappingTraits<DXContainerYAML::Part>::mapping(IO &IO,
                                                   DXContainerYAML::Part &P) {
  IO.mapRequired("Name", P.Name);
  IO.mapRequired("Size", P.Size);
  IO.mapOptional("Program", P.Program);
  IO.mapOptional("Flags", P.Flags);
  IO.mapOptional("Hash", P.Hash);
  IO.mapOptional("PSVInfo", P.Info);
  IO.mapOptional("Signature", P.Signature);
  IO.mapOptional("RootSignature", P.RootSignature);
}

void MappingTraits<DXContainerYAML::Object>::mapping(
    IO &IO, DXContainerYAML::Object &Obj) {
  IO.mapTag("!dxcontainer", true);
  IO.mapRequired("Header", Obj.Header);
  IO.mapRequired("Parts", Obj.Parts);
}

void MappingTraits<DXContainerYAML::ResourceFlags>::mapping(
    IO &IO, DXContainerYAML::ResourceFlags &Flags) {
#define RESOURCE_FLAG(FlagIndex, Enum) IO.mapRequired(#Enum, Flags.Bits.Enum);
#include "llvm/BinaryFormat/DXContainerConstants.def"
}

void MappingTraits<DXContainerYAML::ResourceBindInfo>::mapping(
    IO &IO, DXContainerYAML::ResourceBindInfo &Res) {
  IO.mapRequired("Type", Res.Type);
  IO.mapRequired("Space", Res.Space);
  IO.mapRequired("LowerBound", Res.LowerBound);
  IO.mapRequired("UpperBound", Res.UpperBound);

  const uint32_t *PSVVersion = static_cast<uint32_t *>(IO.getContext());
  if (*PSVVersion < 2)
    return;

  IO.mapRequired("Kind", Res.Kind);
  IO.mapRequired("Flags", Res.Flags);
}

void MappingTraits<DXContainerYAML::SignatureElement>::mapping(
    IO &IO, DXContainerYAML::SignatureElement &El) {
  IO.mapRequired("Name", El.Name);
  IO.mapRequired("Indices", El.Indices);
  IO.mapRequired("StartRow", El.StartRow);
  IO.mapRequired("Cols", El.Cols);
  IO.mapRequired("StartCol", El.StartCol);
  IO.mapRequired("Allocated", El.Allocated);
  IO.mapRequired("Kind", El.Kind);
  IO.mapRequired("ComponentType", El.Type);
  IO.mapRequired("Interpolation", El.Mode);
  IO.mapRequired("DynamicMask", El.DynamicMask);
  IO.mapRequired("Stream", El.Stream);
}

void ScalarEnumerationTraits<dxbc::PSV::SemanticKind>::enumeration(
    IO &IO, dxbc::PSV::SemanticKind &Value) {
  for (const auto &E : dxbc::PSV::getSemanticKinds())
    IO.enumCase(Value, E.Name.str().c_str(), E.Value);
}

void ScalarEnumerationTraits<dxbc::PSV::ComponentType>::enumeration(
    IO &IO, dxbc::PSV::ComponentType &Value) {
  for (const auto &E : dxbc::PSV::getComponentTypes())
    IO.enumCase(Value, E.Name.str().c_str(), E.Value);
}

void ScalarEnumerationTraits<dxbc::PSV::InterpolationMode>::enumeration(
    IO &IO, dxbc::PSV::InterpolationMode &Value) {
  for (const auto &E : dxbc::PSV::getInterpolationModes())
    IO.enumCase(Value, E.Name.str().c_str(), E.Value);
}

void ScalarEnumerationTraits<dxbc::PSV::ResourceType>::enumeration(
    IO &IO, dxbc::PSV::ResourceType &Value) {
  for (const auto &E : dxbc::PSV::getResourceTypes())
    IO.enumCase(Value, E.Name.str().c_str(), E.Value);
}

void ScalarEnumerationTraits<dxbc::PSV::ResourceKind>::enumeration(
    IO &IO, dxbc::PSV::ResourceKind &Value) {
  for (const auto &E : dxbc::PSV::getResourceKinds())
    IO.enumCase(Value, E.Name.str().c_str(), E.Value);
}

void ScalarEnumerationTraits<dxbc::D3DSystemValue>::enumeration(
    IO &IO, dxbc::D3DSystemValue &Value) {
  for (const auto &E : dxbc::getD3DSystemValues())
    IO.enumCase(Value, E.Name.str().c_str(), E.Value);
}

void ScalarEnumerationTraits<dxbc::SigMinPrecision>::enumeration(
    IO &IO, dxbc::SigMinPrecision &Value) {
  for (const auto &E : dxbc::getSigMinPrecisions())
    IO.enumCase(Value, E.Name.str().c_str(), E.Value);
}

void ScalarEnumerationTraits<dxbc::SigComponentType>::enumeration(
    IO &IO, dxbc::SigComponentType &Value) {
  for (const auto &E : dxbc::getSigComponentTypes())
    IO.enumCase(Value, E.Name.str().c_str(), E.Value);
}

} // namespace yaml

void DXContainerYAML::PSVInfo::mapInfoForVersion(yaml::IO &IO) {
  dxbc::PipelinePSVInfo &StageInfo = Info.StageInfo;
  Triple::EnvironmentType Stage = dxbc::getShaderStage(Info.ShaderStage);

  switch (Stage) {
  case Triple::EnvironmentType::Pixel:
    IO.mapRequired("DepthOutput", StageInfo.PS.DepthOutput);
    IO.mapRequired("SampleFrequency", StageInfo.PS.SampleFrequency);
    break;
  case Triple::EnvironmentType::Vertex:
    IO.mapRequired("OutputPositionPresent", StageInfo.VS.OutputPositionPresent);
    break;
  case Triple::EnvironmentType::Geometry:
    IO.mapRequired("InputPrimitive", StageInfo.GS.InputPrimitive);
    IO.mapRequired("OutputTopology", StageInfo.GS.OutputTopology);
    IO.mapRequired("OutputStreamMask", StageInfo.GS.OutputStreamMask);
    IO.mapRequired("OutputPositionPresent", StageInfo.GS.OutputPositionPresent);
    break;
  case Triple::EnvironmentType::Hull:
    IO.mapRequired("InputControlPointCount",
                   StageInfo.HS.InputControlPointCount);
    IO.mapRequired("OutputControlPointCount",
                   StageInfo.HS.OutputControlPointCount);
    IO.mapRequired("TessellatorDomain", StageInfo.HS.TessellatorDomain);
    IO.mapRequired("TessellatorOutputPrimitive",
                   StageInfo.HS.TessellatorOutputPrimitive);
    break;
  case Triple::EnvironmentType::Domain:
    IO.mapRequired("InputControlPointCount",
                   StageInfo.DS.InputControlPointCount);
    IO.mapRequired("OutputPositionPresent", StageInfo.DS.OutputPositionPresent);
    IO.mapRequired("TessellatorDomain", StageInfo.DS.TessellatorDomain);
    break;
  case Triple::EnvironmentType::Mesh:
    IO.mapRequired("GroupSharedBytesUsed", StageInfo.MS.GroupSharedBytesUsed);
    IO.mapRequired("GroupSharedBytesDependentOnViewID",
                   StageInfo.MS.GroupSharedBytesDependentOnViewID);
    IO.mapRequired("PayloadSizeInBytes", StageInfo.MS.PayloadSizeInBytes);
    IO.mapRequired("MaxOutputVertices", StageInfo.MS.MaxOutputVertices);
    IO.mapRequired("MaxOutputPrimitives", StageInfo.MS.MaxOutputPrimitives);
    break;
  case Triple::EnvironmentType::Amplification:
    IO.mapRequired("PayloadSizeInBytes", StageInfo.AS.PayloadSizeInBytes);
    break;
  default:
    break;
  }

  IO.mapRequired("MinimumWaveLaneCount", Info.MinimumWaveLaneCount);
  IO.mapRequired("MaximumWaveLaneCount", Info.MaximumWaveLaneCount);

  if (Version == 0)
    return;

  IO.mapRequired("UsesViewID", Info.UsesViewID);

  switch (Stage) {
  case Triple::EnvironmentType::Geometry:
    IO.mapRequired("MaxVertexCount", Info.GeomData.MaxVertexCount);
    break;
  case Triple::EnvironmentType::Hull:
  case Triple::EnvironmentType::Domain:
    IO.mapRequired("SigPatchConstOrPrimVectors",
                   Info.GeomData.SigPatchConstOrPrimVectors);
    break;
  case Triple::EnvironmentType::Mesh:
    IO.mapRequired("SigPrimVectors", Info.GeomData.MeshInfo.SigPrimVectors);
    IO.mapRequired("MeshOutputTopology",
                   Info.GeomData.MeshInfo.MeshOutputTopology);
    break;
  default:
    break;
  }

  IO.mapRequired("SigInputVectors", Info.SigInputVectors);
  MutableArrayRef<uint8_t> Vec(Info.SigOutputVectors);
  IO.mapRequired("SigOutputVectors", Vec);

  if (Version == 1)
    return;

  IO.mapRequired("NumThreadsX", Info.NumThreadsX);
  IO.mapRequired("NumThreadsY", Info.NumThreadsY);
  IO.mapRequired("NumThreadsZ", Info.NumThreadsZ);

  if (Version == 2)
    return;

  IO.mapRequired("EntryName", EntryName);
}

} // namespace llvm
