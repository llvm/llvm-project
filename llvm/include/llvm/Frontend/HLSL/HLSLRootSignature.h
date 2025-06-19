//===- HLSLRootSignature.h - HLSL Root Signature helper objects -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains structure definitions of HLSL Root Signature
/// objects.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_HLSL_HLSLROOTSIGNATURE_H
#define LLVM_FRONTEND_HLSL_HLSLROOTSIGNATURE_H

#include "llvm/Support/Compiler.h"
#include "llvm/Support/DXILABI.h"
#include <limits>
#include <variant>

namespace llvm {
namespace hlsl {
namespace rootsig {

// Definition of the various enumerations and flags. The definitions of all
// values here correspond to their description in the d3d12.h header and are
// carried over from their values in DXC. For reference:
// https://learn.microsoft.com/en-us/windows/win32/api/d3d12/

enum class RootFlags : uint32_t {
  None = 0,
  AllowInputAssemblerInputLayout = 0x1,
  DenyVertexShaderRootAccess = 0x2,
  DenyHullShaderRootAccess = 0x4,
  DenyDomainShaderRootAccess = 0x8,
  DenyGeometryShaderRootAccess = 0x10,
  DenyPixelShaderRootAccess = 0x20,
  AllowStreamOutput = 0x40,
  LocalRootSignature = 0x80,
  DenyAmplificationShaderRootAccess = 0x100,
  DenyMeshShaderRootAccess = 0x200,
  CBVSRVUAVHeapDirectlyIndexed = 0x400,
  SamplerHeapDirectlyIndexed = 0x800,
  ValidFlags = 0x00000fff
};

enum class RootDescriptorFlags : unsigned {
  None = 0,
  DataVolatile = 0x2,
  DataStaticWhileSetAtExecute = 0x4,
  DataStatic = 0x8,
  ValidFlags = 0xe,
};

enum class DescriptorRangeFlags : unsigned {
  None = 0,
  DescriptorsVolatile = 0x1,
  DataVolatile = 0x2,
  DataStaticWhileSetAtExecute = 0x4,
  DataStatic = 0x8,
  DescriptorsStaticKeepingBufferBoundsChecks = 0x10000,
  ValidFlags = 0x1000f,
  ValidSamplerFlags = DescriptorsVolatile,
};

enum class ShaderVisibility {
  All = 0,
  Vertex = 1,
  Hull = 2,
  Domain = 3,
  Geometry = 4,
  Pixel = 5,
  Amplification = 6,
  Mesh = 7,
};

// D3D12_FILTER enumeration:
// https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_filter
enum class SamplerFilter {
  MinMagMipPoint = 0,
  MinMagPointMipLinear = 0x1,
  MinPointMagLinearMipPoint = 0x4,
  MinPointMagMipLinear = 0x5,
  MinLinearMagMipPoint = 0x10,
  MinLinearMagPointMipLinear = 0x11,
  MinMagLinearMipPoint = 0x14,
  MinMagMipLinear = 0x15,
  Anisotropic = 0x55,
  ComparisonMinMagMipPoint = 0x80,
  ComparisonMinMagPointMipLinear = 0x81,
  ComparisonMinPointMagLinearMipPoint = 0x84,
  ComparisonMinPointMagMipLinear = 0x85,
  ComparisonMinLinearMagMipPoint = 0x90,
  ComparisonMinLinearMagPointMipLinear = 0x91,
  ComparisonMinMagLinearMipPoint = 0x94,
  ComparisonMinMagMipLinear = 0x95,
  ComparisonAnisotropic = 0xd5,
  MinimumMinMagMipPoint = 0x100,
  MinimumMinMagPointMipLinear = 0x101,
  MinimumMinPointMagLinearMipPoint = 0x104,
  MinimumMinPointMagMipLinear = 0x105,
  MinimumMinLinearMagMipPoint = 0x110,
  MinimumMinLinearMagPointMipLinear = 0x111,
  MinimumMinMagLinearMipPoint = 0x114,
  MinimumMinMagMipLinear = 0x115,
  MinimumAnisotropic = 0x155,
  MaximumMinMagMipPoint = 0x180,
  MaximumMinMagPointMipLinear = 0x181,
  MaximumMinPointMagLinearMipPoint = 0x184,
  MaximumMinPointMagMipLinear = 0x185,
  MaximumMinLinearMagMipPoint = 0x190,
  MaximumMinLinearMagPointMipLinear = 0x191,
  MaximumMinMagLinearMipPoint = 0x194,
  MaximumMinMagMipLinear = 0x195,
  MaximumAnisotropic = 0x1d5
};

enum class TextureAddressMode {
  Wrap = 1,
  Mirror = 2,
  Clamp = 3,
  Border = 4,
  MirrorOnce = 5
};

enum class ComparisonFunc : unsigned {
  Never = 1,
  Less = 2,
  Equal = 3,
  LessEqual = 4,
  Greater = 5,
  NotEqual = 6,
  GreaterEqual = 7,
  Always = 8
};

enum class StaticBorderColor {
  TransparentBlack = 0,
  OpaqueBlack = 1,
  OpaqueWhite = 2,
  OpaqueBlackUint = 3,
  OpaqueWhiteUint = 4
};

// Definitions of the in-memory data layout structures

// Models the different registers: bReg | tReg | uReg | sReg
enum class RegisterType { BReg, TReg, UReg, SReg };
struct Register {
  RegisterType ViewType;
  uint32_t Number;
};

// Models the parameter values of root constants
struct RootConstants {
  uint32_t Num32BitConstants;
  Register Reg;
  uint32_t Space = 0;
  ShaderVisibility Visibility = ShaderVisibility::All;
};

enum class DescriptorType : uint8_t { SRV = 0, UAV, CBuffer };
// Models RootDescriptor : CBV | SRV | UAV, by collecting like parameters
struct RootDescriptor {
  DescriptorType Type;
  Register Reg;
  uint32_t Space = 0;
  ShaderVisibility Visibility = ShaderVisibility::All;
  RootDescriptorFlags Flags;

  void setDefaultFlags() {
    switch (Type) {
    case DescriptorType::CBuffer:
    case DescriptorType::SRV:
      Flags = RootDescriptorFlags::DataStaticWhileSetAtExecute;
      break;
    case DescriptorType::UAV:
      Flags = RootDescriptorFlags::DataVolatile;
      break;
    }
  }
};

// Models the end of a descriptor table and stores its visibility
struct DescriptorTable {
  ShaderVisibility Visibility = ShaderVisibility::All;
  // Denotes that the previous NumClauses in the RootElement array
  // are the clauses in the table.
  uint32_t NumClauses = 0;
};

static const uint32_t NumDescriptorsUnbounded = 0xffffffff;
static const uint32_t DescriptorTableOffsetAppend = 0xffffffff;
// Models DTClause : CBV | SRV | UAV | Sampler, by collecting like parameters
using ClauseType = llvm::dxil::ResourceClass;
struct DescriptorTableClause {
  ClauseType Type;
  Register Reg;
  uint32_t NumDescriptors = 1;
  uint32_t Space = 0;
  uint32_t Offset = DescriptorTableOffsetAppend;
  DescriptorRangeFlags Flags;

  void setDefaultFlags() {
    switch (Type) {
    case ClauseType::CBuffer:
    case ClauseType::SRV:
      Flags = DescriptorRangeFlags::DataStaticWhileSetAtExecute;
      break;
    case ClauseType::UAV:
      Flags = DescriptorRangeFlags::DataVolatile;
      break;
    case ClauseType::Sampler:
      Flags = DescriptorRangeFlags::None;
      break;
    }
  }
};

struct StaticSampler {
  Register Reg;
  SamplerFilter Filter = SamplerFilter::Anisotropic;
  TextureAddressMode AddressU = TextureAddressMode::Wrap;
  TextureAddressMode AddressV = TextureAddressMode::Wrap;
  TextureAddressMode AddressW = TextureAddressMode::Wrap;
  float MipLODBias = 0.f;
  uint32_t MaxAnisotropy = 16;
  ComparisonFunc CompFunc = ComparisonFunc::LessEqual;
  StaticBorderColor BorderColor = StaticBorderColor::OpaqueWhite;
  float MinLOD = 0.f;
  float MaxLOD = std::numeric_limits<float>::max();
  uint32_t Space = 0;
  ShaderVisibility Visibility = ShaderVisibility::All;
};

/// Models RootElement : RootFlags | RootConstants | RootParam
///  | DescriptorTable | DescriptorTableClause | StaticSampler
///
/// A Root Signature is modeled in-memory by an array of RootElements. These
/// aim to map closely to their DSL grammar reprsentation defined in the spec.
///
/// Each optional parameter has its default value defined in the struct, and,
/// each mandatory parameter does not have a default initialization.
///
/// For the variants RootFlags, RootConstants, RootParam, StaticSampler and
/// DescriptorTableClause: each data member maps directly to a parameter in the
/// grammar.
///
/// The DescriptorTable is modelled by having its Clauses as the previous
/// RootElements in the array, and it holds a data member for the Visibility
/// parameter.
using RootElement =
    std::variant<RootFlags, RootConstants, RootDescriptor, DescriptorTable,
                 DescriptorTableClause, StaticSampler>;

} // namespace rootsig
} // namespace hlsl
} // namespace llvm

#endif // LLVM_FRONTEND_HLSL_HLSLROOTSIGNATURE_H
