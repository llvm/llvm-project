//===-------- HLSLRootSignatureDumpTest.cpp - RootSignature dump tests ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/HLSL/HLSLRootSignatureUtils.h"
#include "gtest/gtest.h"

using namespace llvm::hlsl::rootsig;

namespace {

TEST(HLSLRootSignatureTest, DescriptorCBVClauseDump) {
  DescriptorTableClause Clause;
  Clause.Type = ClauseType::CBuffer;
  Clause.Reg = {RegisterType::BReg, 0};
  Clause.setDefaultFlags();

  std::string Out;
  llvm::raw_string_ostream OS(Out);
  OS << Clause;
  OS.flush();

  std::string Expected = "CBV(b0, numDescriptors = 1, space = 0, "
                         "offset = DescriptorTableOffsetAppend, "
                         "flags = DataStaticWhileSetAtExecute)";
  EXPECT_EQ(Out, Expected);
}

TEST(HLSLRootSignatureTest, DescriptorSRVClauseDump) {
  DescriptorTableClause Clause;
  Clause.Type = ClauseType::SRV;
  Clause.Reg = {RegisterType::TReg, 0};
  Clause.NumDescriptors = NumDescriptorsUnbounded;
  Clause.Space = 42;
  Clause.Offset = 3;
  Clause.Flags = llvm::dxbc::DescriptorRangeFlags::None;

  std::string Out;
  llvm::raw_string_ostream OS(Out);
  OS << Clause;
  OS.flush();

  std::string Expected = "SRV(t0, numDescriptors = unbounded, space = 42, "
                         "offset = 3, flags = None)";
  EXPECT_EQ(Out, Expected);
}

TEST(HLSLRootSignatureTest, DescriptorUAVClauseDump) {
  using llvm::dxbc::DescriptorRangeFlags;
  DescriptorTableClause Clause;
  Clause.Type = ClauseType::UAV;
  Clause.Reg = {RegisterType::UReg, 92374};
  Clause.NumDescriptors = 3298;
  Clause.Space = 932847;
  Clause.Offset = 1;
  auto ValidDescriptorRangeFlags =
      DescriptorRangeFlags::DescriptorsVolatile |
      DescriptorRangeFlags::DataVolatile |
      DescriptorRangeFlags::DataStaticWhileSetAtExecute |
      DescriptorRangeFlags::DataStatic |
      DescriptorRangeFlags::DescriptorsStaticKeepingBufferBoundsChecks;
  Clause.Flags = ValidDescriptorRangeFlags;

  std::string Out;
  llvm::raw_string_ostream OS(Out);
  OS << Clause;
  OS.flush();

  std::string Expected =
      "UAV(u92374, numDescriptors = 3298, space = 932847, offset = 1, flags = "
      "DescriptorsVolatile | "
      "DataVolatile | "
      "DataStaticWhileSetAtExecute | "
      "DataStatic | "
      "DescriptorsStaticKeepingBufferBoundsChecks)";
  EXPECT_EQ(Out, Expected);
}

TEST(HLSLRootSignatureTest, DescriptorSamplerClauseDump) {
  DescriptorTableClause Clause;
  Clause.Type = ClauseType::Sampler;
  Clause.Reg = {RegisterType::SReg, 0};
  Clause.NumDescriptors = 2;
  Clause.Space = 42;
  Clause.Offset = DescriptorTableOffsetAppend;
  Clause.Flags = llvm::dxbc::DescriptorRangeFlags::DescriptorsVolatile;

  std::string Out;
  llvm::raw_string_ostream OS(Out);
  OS << Clause;
  OS.flush();

  std::string Expected = "Sampler(s0, numDescriptors = 2, space = 42, offset = "
                         "DescriptorTableOffsetAppend, "
                         "flags = DescriptorsVolatile)";
  EXPECT_EQ(Out, Expected);
}

TEST(HLSLRootSignatureTest, DescriptorTableDump) {
  DescriptorTable Table;
  Table.NumClauses = 4;
  Table.Visibility = llvm::dxbc::ShaderVisibility::Geometry;

  std::string Out;
  llvm::raw_string_ostream OS(Out);
  OS << Table;
  OS.flush();

  std::string Expected =
      "DescriptorTable(numClauses = 4, visibility = Geometry)";
  EXPECT_EQ(Out, Expected);
}

TEST(HLSLRootSignatureTest, RootCBVDump) {
  RootDescriptor Descriptor;
  Descriptor.Type = DescriptorType::CBuffer;
  Descriptor.Reg = {RegisterType::BReg, 0};
  Descriptor.setDefaultFlags();

  std::string Out;
  llvm::raw_string_ostream OS(Out);
  OS << Descriptor;
  OS.flush();

  std::string Expected = "RootCBV(b0, space = 0, "
                         "visibility = All, "
                         "flags = DataStaticWhileSetAtExecute)";
  EXPECT_EQ(Out, Expected);
}

TEST(HLSLRootSignatureTest, RootSRVDump) {
  RootDescriptor Descriptor;
  Descriptor.Type = DescriptorType::SRV;
  Descriptor.Reg = {RegisterType::TReg, 0};
  Descriptor.Space = 42;
  Descriptor.Visibility = llvm::dxbc::ShaderVisibility::Geometry;
  Descriptor.Flags = llvm::dxbc::RootDescriptorFlags::None;

  std::string Out;
  llvm::raw_string_ostream OS(Out);
  OS << Descriptor;
  OS.flush();

  std::string Expected =
      "RootSRV(t0, space = 42, visibility = Geometry, flags = None)";
  EXPECT_EQ(Out, Expected);
}

TEST(HLSLRootSignatureTest, RootUAVDump) {
  using llvm::dxbc::RootDescriptorFlags;
  RootDescriptor Descriptor;
  Descriptor.Type = DescriptorType::UAV;
  Descriptor.Reg = {RegisterType::UReg, 92374};
  Descriptor.Space = 932847;
  Descriptor.Visibility = llvm::dxbc::ShaderVisibility::Hull;
  auto ValidRootDescriptorFlags =
      RootDescriptorFlags::DataVolatile |
      RootDescriptorFlags::DataStaticWhileSetAtExecute |
      RootDescriptorFlags::DataStatic;
  Descriptor.Flags = ValidRootDescriptorFlags;

  std::string Out;
  llvm::raw_string_ostream OS(Out);
  OS << Descriptor;
  OS.flush();

  std::string Expected =
      "RootUAV(u92374, space = 932847, visibility = Hull, flags = "
      "DataVolatile | "
      "DataStaticWhileSetAtExecute | "
      "DataStatic)";
  EXPECT_EQ(Out, Expected);
}

TEST(HLSLRootSignatureTest, DefaultStaticSamplerDump) {
  StaticSampler Sampler;
  Sampler.Reg = {RegisterType::SReg, 0};

  std::string Out;
  llvm::raw_string_ostream OS(Out);
  OS << Sampler;
  OS.flush();

  std::string Expected = "StaticSampler(s0, "
                         "filter = Anisotropic, "
                         "addressU = Wrap, "
                         "addressV = Wrap, "
                         "addressW = Wrap, "
                         "mipLODBias = 0.000000e+00, "
                         "maxAnisotropy = 16, "
                         "comparisonFunc = LessEqual, "
                         "borderColor = OpaqueWhite, "
                         "minLOD = 0.000000e+00, "
                         "maxLOD = 3.402823e+38, "
                         "space = 0, "
                         "visibility = All"
                         ")";
  EXPECT_EQ(Out, Expected);
}

TEST(HLSLRootSignatureTest, DefinedStaticSamplerDump) {
  StaticSampler Sampler;
  Sampler.Reg = {RegisterType::SReg, 0};

  Sampler.Filter = llvm::dxbc::SamplerFilter::ComparisonMinMagLinearMipPoint;
  Sampler.AddressU = llvm::dxbc::TextureAddressMode::Mirror;
  Sampler.AddressV = llvm::dxbc::TextureAddressMode::Border;
  Sampler.AddressW = llvm::dxbc::TextureAddressMode::Clamp;
  Sampler.MipLODBias = 4.8f;
  Sampler.MaxAnisotropy = 32;
  Sampler.CompFunc = llvm::dxbc::ComparisonFunc::NotEqual;
  Sampler.BorderColor = llvm::dxbc::StaticBorderColor::OpaqueBlack;
  Sampler.MinLOD = 1.0f;
  Sampler.MaxLOD = 32.0f;
  Sampler.Space = 7;
  Sampler.Visibility = llvm::dxbc::ShaderVisibility::Domain;

  std::string Out;
  llvm::raw_string_ostream OS(Out);
  OS << Sampler;
  OS.flush();

  std::string Expected = "StaticSampler(s0, "
                         "filter = ComparisonMinMagLinearMipPoint, "
                         "addressU = Mirror, "
                         "addressV = Border, "
                         "addressW = Clamp, "
                         "mipLODBias = 4.800000e+00, "
                         "maxAnisotropy = 32, "
                         "comparisonFunc = NotEqual, "
                         "borderColor = OpaqueBlack, "
                         "minLOD = 1.000000e+00, "
                         "maxLOD = 3.200000e+01, "
                         "space = 7, "
                         "visibility = Domain"
                         ")";
  EXPECT_EQ(Out, Expected);
}

TEST(HLSLRootSignatureTest, DefaultRootConstantsDump) {
  RootConstants Constants;
  Constants.Num32BitConstants = 1;
  Constants.Reg = {RegisterType::BReg, 3};

  std::string Out;
  llvm::raw_string_ostream OS(Out);
  OS << Constants;
  OS.flush();

  std::string Expected = "RootConstants(num32BitConstants = 1, b3, space = 0, "
                         "visibility = All)";
  EXPECT_EQ(Out, Expected);
}

TEST(HLSLRootSignatureTest, SetRootConstantsDump) {
  RootConstants Constants;
  Constants.Num32BitConstants = 983;
  Constants.Reg = {RegisterType::BReg, 34593};
  Constants.Space = 7;
  Constants.Visibility = llvm::dxbc::ShaderVisibility::Pixel;

  std::string Out;
  llvm::raw_string_ostream OS(Out);
  OS << Constants;
  OS.flush();

  std::string Expected = "RootConstants(num32BitConstants = 983, b34593, "
                         "space = 7, visibility = Pixel)";
  EXPECT_EQ(Out, Expected);
}

TEST(HLSLRootSignatureTest, NoneRootFlagsDump) {
  llvm::dxbc::RootFlags Flags = llvm::dxbc::RootFlags::None;

  std::string Out;
  llvm::raw_string_ostream OS(Out);
  OS << Flags;
  OS.flush();

  std::string Expected = "RootFlags(None)";
  EXPECT_EQ(Out, Expected);
}

TEST(HLSLRootSignatureTest, AllRootFlagsDump) {
  using llvm::dxbc::RootFlags;
  auto ValidRootFlags = RootFlags::AllowInputAssemblerInputLayout |
                        RootFlags::DenyVertexShaderRootAccess |
                        RootFlags::DenyHullShaderRootAccess |
                        RootFlags::DenyDomainShaderRootAccess |
                        RootFlags::DenyGeometryShaderRootAccess |
                        RootFlags::DenyPixelShaderRootAccess |
                        RootFlags::AllowStreamOutput |
                        RootFlags::LocalRootSignature |
                        RootFlags::DenyAmplificationShaderRootAccess |
                        RootFlags::DenyMeshShaderRootAccess |
                        RootFlags::CBVSRVUAVHeapDirectlyIndexed |
                        RootFlags::SamplerHeapDirectlyIndexed;

  std::string Out;
  llvm::raw_string_ostream OS(Out);
  OS << ValidRootFlags;
  OS.flush();

  std::string Expected = "RootFlags("
                         "AllowInputAssemblerInputLayout | "
                         "DenyVertexShaderRootAccess | "
                         "DenyHullShaderRootAccess | "
                         "DenyDomainShaderRootAccess | "
                         "DenyGeometryShaderRootAccess | "
                         "DenyPixelShaderRootAccess | "
                         "AllowStreamOutput | "
                         "LocalRootSignature | "
                         "DenyAmplificationShaderRootAccess | "
                         "DenyMeshShaderRootAccess | "
                         "CBVSRVUAVHeapDirectlyIndexed | "
                         "SamplerHeapDirectlyIndexed)";
  EXPECT_EQ(Out, Expected);
}

} // namespace
