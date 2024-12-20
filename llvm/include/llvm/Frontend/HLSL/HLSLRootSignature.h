//===- HLSLRootSignature.h - HLSL Root Signature helper objects -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains helper objects for working with HLSL Root
/// Signatures.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_HLSL_HLSLROOTSIGNATURE_H
#define LLVM_FRONTEND_HLSL_HLSLROOTSIGNATURE_H

#include <stdint.h>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Endian.h"

namespace llvm {
namespace hlsl {
namespace root_signature {

// This is a copy from DebugInfo/CodeView/CodeView.h
#define RS_DEFINE_ENUM_CLASS_FLAGS_OPERATORS(Class)                            \
  inline Class operator|(Class a, Class b) {                                   \
    return static_cast<Class>(llvm::to_underlying(a) |                         \
                              llvm::to_underlying(b));                         \
  }                                                                            \
  inline Class operator&(Class a, Class b) {                                   \
    return static_cast<Class>(llvm::to_underlying(a) &                         \
                              llvm::to_underlying(b));                         \
  }                                                                            \
  inline Class operator~(Class a) {                                            \
    return static_cast<Class>(~llvm::to_underlying(a));                        \
  }                                                                            \
  inline Class &operator|=(Class &a, Class b) {                                \
    a = a | b;                                                                 \
    return a;                                                                  \
  }                                                                            \
  inline Class &operator&=(Class &a, Class b) {                                \
    a = a & b;                                                                 \
    return a;                                                                  \
  }

// Various enumerations and flags

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
  AllowLowTierReservedHwCbLimit = 0x80000000,
  ValidFlags = 0x80000fff
};
RS_DEFINE_ENUM_CLASS_FLAGS_OPERATORS(RootFlags)

enum class RootDescriptorFlags : unsigned {
  None = 0,
  DataVolatile = 0x2,
  DataStaticWhileSetAtExecute = 0x4,
  DataStatic = 0x8,
  ValidFlags = 0xe
};
RS_DEFINE_ENUM_CLASS_FLAGS_OPERATORS(RootDescriptorFlags)

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
RS_DEFINE_ENUM_CLASS_FLAGS_OPERATORS(DescriptorRangeFlags)

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

enum class Filter {
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

// Define the in-memory layout structures

// Models the different registers: bReg | tReg | uReg | sReg
enum class RegisterType { BReg, TReg, UReg, SReg };
struct Register {
  RegisterType ViewType;
  uint32_t Number;
};

// Models RootConstants | RootCBV | RootSRV | RootUAV collecting like
// parameters
enum class RootType { CBV, SRV, UAV, Constants };
struct RootParameter {
  RootType Type;
  Register Register;
  union {
    uint32_t Num32BitConstants;
    RootDescriptorFlags Flags = RootDescriptorFlags::None;
  };
  uint32_t Space = 0;
  ShaderVisibility Visibility = ShaderVisibility::All;
};

static const uint32_t DescriptorTableOffsetAppend = 0xffffffff;
// Models DTClause : CBV | SRV | UAV | Sampler collecting like parameters
enum class ClauseType { CBV, SRV, UAV, Sampler };
struct DescriptorTableClause {
  ClauseType Type;
  Register Register;
  uint32_t NumDescriptors = 1;
  uint32_t Space = 0;
  uint32_t Offset = DescriptorTableOffsetAppend;
  DescriptorRangeFlags Flags = DescriptorRangeFlags::None;
};

// Models the start of a descriptor table
struct DescriptorTable {
  ShaderVisibility Visibility = ShaderVisibility::All;
  uint32_t NumClauses = 0;
};

struct StaticSampler {
  Register Register;
  Filter Filter = Filter::Anisotropic;
  TextureAddressMode AddressU = TextureAddressMode::Wrap;
  TextureAddressMode AddressV = TextureAddressMode::Wrap;
  TextureAddressMode AddressW = TextureAddressMode::Wrap;
  float MipLODBias = 0.f;
  uint32_t MaxAnisotropy = 16;
  ComparisonFunc ComparisonFunc = ComparisonFunc::LessEqual;
  StaticBorderColor BorderColor = StaticBorderColor::OpaqueWhite;
  float MinLOD = 0.f;
  float MaxLODBias = 3.402823466e+38f;
  uint32_t Space = 0;
  ShaderVisibility Visibility = ShaderVisibility::All;
};

struct RootElement {
  enum class ElementType {
    RootFlags,
    RootParameter,
    DescriptorTable,
    DescriptorTableClause,
    StaticSampler
  };

  ElementType Tag;
  union {
    RootFlags Flags;
    RootParameter Parameter;
    DescriptorTable Table;
    DescriptorTableClause Clause;
    StaticSampler Sampler;
  };

  // Constructors
  RootElement(RootFlags Flags) : Tag(ElementType::RootFlags), Flags(Flags) {}
  RootElement(RootParameter Parameter)
      : Tag(ElementType::RootParameter), Parameter(Parameter) {}
  RootElement(DescriptorTable Table)
      : Tag(ElementType::DescriptorTable), Table(Table) {}
  RootElement(DescriptorTableClause Clause)
      : Tag(ElementType::DescriptorTableClause), Clause(Clause) {}
  RootElement(StaticSampler Sampler)
      : Tag(ElementType::StaticSampler), Sampler(Sampler) {}
};

} // namespace root_signature
} // namespace hlsl
} // namespace llvm

#endif // LLVM_FRONTEND_HLSL_HLSLROOTSIGNATURE_H
