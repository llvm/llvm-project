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
#include "llvm/Object/DXContainer.h"
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
  for (const dxbc::RTS0::v0::RootParameterHeader &PH : Data.param_headers()) {

    if (!dxbc::RTS0::isValidParameterType(PH.ParameterType))
      return createStringError(std::errc::invalid_argument,
                               "Invalid value for parameter type");

    RootParameterYamlDesc NewP(PH.ParameterType);
    NewP.Offset = PH.ParameterOffset;
    NewP.Type = PH.ParameterType;

    if (!dxbc::RTS0::isValidShaderVisibility(PH.ShaderVisibility))
      return createStringError(std::errc::invalid_argument,
                               "Invalid value for shader visibility");

    NewP.Visibility = PH.ShaderVisibility;

    llvm::Expected<object::DirectX::RootParameterView> ParamViewOrErr =
        Data.getParameter(PH);
    if (Error E = ParamViewOrErr.takeError())
      return std::move(E);
    object::DirectX::RootParameterView ParamView = ParamViewOrErr.get();

    if (auto *RCV = dyn_cast<object::DirectX::RootConstantView>(&ParamView)) {
      llvm::Expected<dxbc::RTS0::v0::RootConstants> ConstantsOrErr =
          RCV->read();
      if (Error E = ConstantsOrErr.takeError())
        return std::move(E);

      auto Constants = *ConstantsOrErr;
      RootConstantsYaml ConstantYaml;
      ConstantYaml.Num32BitValues = Constants.Num32BitValues;
      ConstantYaml.ShaderRegister = Constants.ShaderRegister;
      ConstantYaml.RegisterSpace = Constants.RegisterSpace;
      NewP.Data = ConstantYaml;
    } else if (auto *RDV =
                   dyn_cast<object::DirectX::RootDescriptorView>(&ParamView)) {
      llvm::Expected<dxbc::RTS0::v1::RootDescriptor> DescriptorOrErr =
          RDV->read(Version);
      if (Error E = DescriptorOrErr.takeError())
        return std::move(E);
      auto Descriptor = *DescriptorOrErr;
      RootDescriptorYaml YamlDescriptor;
      YamlDescriptor.ShaderRegister = Descriptor.ShaderRegister;
      YamlDescriptor.RegisterSpace = Descriptor.RegisterSpace;
      if (Version > 1) {
#define ROOT_DESCRIPTOR_FLAG(Num, Val)                                         \
  YamlDescriptor.Val =                                                         \
      (Descriptor.Flags &                                                      \
       llvm::to_underlying(dxbc::RTS0::RootDescriptorFlag::Val)) > 0;
#include "llvm/BinaryFormat/DXContainerConstants.def"
      }
      NewP.Data = YamlDescriptor;
    } else if (auto *TDV = dyn_cast<object::DirectX::DescriptorTableView<
                   dxbc::RTS0::v0::DescriptorRange>>(&ParamView)) {
      llvm::Expected<
          object::DirectX::DescriptorTable<dxbc::RTS0::v0::DescriptorRange>>
          TableOrErr = TDV->read();
      if (Error E = TableOrErr.takeError())
        return std::move(E);
      auto Table = *TableOrErr;
      DescriptorTableYaml YamlTable;
      YamlTable.NumRanges = Table.NumRanges;
      YamlTable.RangesOffset = Table.RangesOffset;

      for (const auto &R : Table) {
        DescriptorRangeYaml NewR;

        NewR.OffsetInDescriptorsFromTableStart =
            R.OffsetInDescriptorsFromTableStart;
        NewR.NumDescriptors = R.NumDescriptors;
        NewR.BaseShaderRegister = R.BaseShaderRegister;
        NewR.RegisterSpace = R.RegisterSpace;
        NewR.RangeType = R.RangeType;

        YamlTable.Ranges.push_back(NewR);
      }
      NewP.Data = YamlTable;
    } else if (auto *TDV = dyn_cast<object::DirectX::DescriptorTableView<
                   dxbc::RTS0::v1::DescriptorRange>>(&ParamView)) {
      llvm::Expected<
          object::DirectX::DescriptorTable<dxbc::RTS0::v1::DescriptorRange>>
          TableOrErr = TDV->read();
      if (Error E = TableOrErr.takeError())
        return std::move(E);
      auto Table = *TableOrErr;
      DescriptorTableYaml YamlTable;
      YamlTable.NumRanges = Table.NumRanges;
      YamlTable.RangesOffset = Table.RangesOffset;

      for (const auto &R : Table) {
        DescriptorRangeYaml NewR;

        NewR.OffsetInDescriptorsFromTableStart =
            R.OffsetInDescriptorsFromTableStart;
        NewR.NumDescriptors = R.NumDescriptors;
        NewR.BaseShaderRegister = R.BaseShaderRegister;
        NewR.RegisterSpace = R.RegisterSpace;
        NewR.RangeType = R.RangeType;
#define DESCRIPTOR_RANGE_FLAG(Num, Val)                                        \
  NewR.Val = (R.Flags &                                                        \
              llvm::to_underlying(dxbc::RTS0::DescriptorRangeFlag::Val)) > 0;
#include "llvm/BinaryFormat/DXContainerConstants.def"
        YamlTable.Ranges.push_back(NewR);
      }
      NewP.Data = YamlTable;
    }

    RootSigDesc.Parameters.push_back(NewP);
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
#define ROOT_ELEMENT_FLAG(Num, Val)                                            \
  RootSigDesc.Val =                                                            \
      (Flags & llvm::to_underlying(dxbc::RTS0::RootElementFlag::Val)) > 0;
#include "llvm/BinaryFormat/DXContainerConstants.def"
  return RootSigDesc;
}

uint32_t DXContainerYAML::RootDescriptorYaml::getEncodedFlags() const {
  uint64_t Flag = 0;
#define ROOT_DESCRIPTOR_FLAG(Num, Val)                                         \
  if (Val)                                                                     \
    Flag |= (uint32_t)dxbc::RTS0::RootDescriptorFlag::Val;
#include "llvm/BinaryFormat/DXContainerConstants.def"
  return Flag;
}

uint32_t DXContainerYAML::RootSignatureYamlDesc::getEncodedFlags() {
  uint64_t Flag = 0;
#define ROOT_ELEMENT_FLAG(Num, Val)                                            \
  if (Val)                                                                     \
    Flag |= (uint32_t)dxbc::RTS0::RootElementFlag::Val;
#include "llvm/BinaryFormat/DXContainerConstants.def"
  return Flag;
}

uint32_t DXContainerYAML::DescriptorRangeYaml::getEncodedFlags() const {
  uint64_t Flag = 0;
#define DESCRIPTOR_RANGE_FLAG(Num, Val)                                        \
  if (Val)                                                                     \
    Flag |= (uint32_t)dxbc::RTS0::DescriptorRangeFlag::Val;
#include "llvm/BinaryFormat/DXContainerConstants.def"
  return Flag;
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
  IO.mapRequired("RootParametersOffset", S.RootParametersOffset);
  IO.mapRequired("NumStaticSamplers", S.NumStaticSamplers);
  IO.mapRequired("StaticSamplersOffset", S.StaticSamplersOffset);
  IO.mapRequired("Parameters", S.Parameters);
  IO.mapRequired("Samplers", S.StaticSamplers);
#define ROOT_ELEMENT_FLAG(Num, Val) IO.mapOptional(#Val, S.Val, false);
#include "llvm/BinaryFormat/DXContainerConstants.def"
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
#define ROOT_DESCRIPTOR_FLAG(Num, Val) IO.mapOptional(#Val, D.Val, false);
#include "llvm/BinaryFormat/DXContainerConstants.def"
}

void MappingTraits<llvm::DXContainerYAML::DescriptorRangeYaml>::mapping(
    IO &IO, llvm::DXContainerYAML::DescriptorRangeYaml &R) {
  IO.mapRequired("RangeType", R.RangeType);
  IO.mapRequired("NumDescriptors", R.NumDescriptors);
  IO.mapRequired("BaseShaderRegister", R.BaseShaderRegister);
  IO.mapRequired("RegisterSpace", R.RegisterSpace);
  IO.mapRequired("OffsetInDescriptorsFromTableStart",
                 R.OffsetInDescriptorsFromTableStart);
#define DESCRIPTOR_RANGE_FLAG(Num, Val) IO.mapOptional(#Val, R.Val, false);
#include "llvm/BinaryFormat/DXContainerConstants.def"
}

void MappingTraits<llvm::DXContainerYAML::DescriptorTableYaml>::mapping(
    IO &IO, llvm::DXContainerYAML::DescriptorTableYaml &T) {
  IO.mapRequired("NumRanges", T.NumRanges);
  IO.mapOptional("RangesOffset", T.RangesOffset);
  IO.mapRequired("Ranges", T.Ranges);
}

void MappingTraits<llvm::DXContainerYAML::RootParameterYamlDesc>::mapping(
    IO &IO, llvm::DXContainerYAML::RootParameterYamlDesc &P) {
  IO.mapRequired("ParameterType", P.Type);
  IO.mapRequired("ShaderVisibility", P.Visibility);

  switch (P.Type) {
  case llvm::to_underlying(dxbc::RTS0::RootParameterType::Constants32Bit): {
    DXContainerYAML::RootConstantsYaml Constants;
    if (IO.outputting())
      Constants = std::get<DXContainerYAML::RootConstantsYaml>(P.Data);
    IO.mapRequired("Constants", Constants);
    P.Data = Constants;
  } break;
  case llvm::to_underlying(dxbc::RTS0::RootParameterType::CBV):
  case llvm::to_underlying(dxbc::RTS0::RootParameterType::SRV):
  case llvm::to_underlying(dxbc::RTS0::RootParameterType::UAV): {
    DXContainerYAML::RootDescriptorYaml Descriptor;
    if (IO.outputting())
      Descriptor = std::get<DXContainerYAML::RootDescriptorYaml>(P.Data);
    IO.mapRequired("Descriptor", Descriptor);
    P.Data = Descriptor;
  } break;
  case llvm::to_underlying(dxbc::RTS0::RootParameterType::DescriptorTable): {
    DXContainerYAML::DescriptorTableYaml Table;
    if (IO.outputting())
      Table = std::get<DXContainerYAML::DescriptorTableYaml>(P.Data);
    IO.mapRequired("Table", Table);
    P.Data = Table;
  } break;
  }
}

void MappingTraits<llvm::DXContainerYAML::StaticSamplerYamlDesc>::mapping(
    IO &IO, llvm::DXContainerYAML::StaticSamplerYamlDesc &S) {

  IO.mapRequired("Filter", S.Filter);
  IO.mapRequired("AddressU", S.AddressU);
  IO.mapRequired("AddressV", S.AddressV);
  IO.mapRequired("AddressW", S.AddressW);
  IO.mapRequired("MipLODBias", S.MipLODBias);
  IO.mapRequired("MaxAnisotropy", S.MaxAnisotropy);
  IO.mapRequired("ComparisonFunc", S.ComparisonFunc);
  IO.mapRequired("BorderColor", S.BorderColor);
  IO.mapRequired("MinLOD", S.MinLOD);
  IO.mapRequired("MaxLOD", S.MaxLOD);
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
