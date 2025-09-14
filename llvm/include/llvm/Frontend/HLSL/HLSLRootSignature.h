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

#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DXILABI.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>
#include <variant>

namespace llvm {
namespace hlsl {
namespace rootsig {

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
  dxbc::ShaderVisibility Visibility = dxbc::ShaderVisibility::All;
};

// Models RootDescriptor : CBV | SRV | UAV, by collecting like parameters
struct RootDescriptor {
  dxil::ResourceClass Type;
  Register Reg;
  uint32_t Space = 0;
  dxbc::ShaderVisibility Visibility = dxbc::ShaderVisibility::All;
  dxbc::RootDescriptorFlags Flags;

  void setDefaultFlags(dxbc::RootSignatureVersion Version) {
    if (Version == dxbc::RootSignatureVersion::V1_0) {
      Flags = dxbc::RootDescriptorFlags::DataVolatile;
      return;
    }

    assert(Version == llvm::dxbc::RootSignatureVersion::V1_1 &&
           "Specified an invalid root signature version");
    switch (Type) {
    case dxil::ResourceClass::CBuffer:
    case dxil::ResourceClass::SRV:
      Flags = dxbc::RootDescriptorFlags::DataStaticWhileSetAtExecute;
      break;
    case dxil::ResourceClass::UAV:
      Flags = dxbc::RootDescriptorFlags::DataVolatile;
      break;
    case dxil::ResourceClass::Sampler:
      llvm_unreachable(
          "ResourceClass::Sampler is not valid for RootDescriptors");
    }
  }
};

// Models the end of a descriptor table and stores its visibility
struct DescriptorTable {
  dxbc::ShaderVisibility Visibility = dxbc::ShaderVisibility::All;
  // Denotes that the previous NumClauses in the RootElement array
  // are the clauses in the table.
  uint32_t NumClauses = 0;
};

static const uint32_t NumDescriptorsUnbounded = 0xffffffff;
static const uint32_t DescriptorTableOffsetAppend = 0xffffffff;
// Models DTClause : CBV | SRV | UAV | Sampler, by collecting like parameters
struct DescriptorTableClause {
  dxil::ResourceClass Type;
  Register Reg;
  uint32_t NumDescriptors = 1;
  uint32_t Space = 0;
  uint32_t Offset = DescriptorTableOffsetAppend;
  dxbc::DescriptorRangeFlags Flags;

  void setDefaultFlags(dxbc::RootSignatureVersion Version) {
    if (Version == dxbc::RootSignatureVersion::V1_0) {
      Flags = dxbc::DescriptorRangeFlags::DescriptorsVolatile;
      if (Type != dxil::ResourceClass::Sampler)
        Flags |= dxbc::DescriptorRangeFlags::DataVolatile;
      return;
    }

    assert(Version == dxbc::RootSignatureVersion::V1_1 &&
           "Specified an invalid root signature version");
    switch (Type) {
    case dxil::ResourceClass::CBuffer:
    case dxil::ResourceClass::SRV:
      Flags = dxbc::DescriptorRangeFlags::DataStaticWhileSetAtExecute;
      break;
    case dxil::ResourceClass::UAV:
      Flags = dxbc::DescriptorRangeFlags::DataVolatile;
      break;
    case dxil::ResourceClass::Sampler:
      Flags = dxbc::DescriptorRangeFlags::None;
      break;
    }
  }
};

struct StaticSampler {
  Register Reg;
  dxbc::SamplerFilter Filter = dxbc::SamplerFilter::Anisotropic;
  dxbc::TextureAddressMode AddressU = dxbc::TextureAddressMode::Wrap;
  dxbc::TextureAddressMode AddressV = dxbc::TextureAddressMode::Wrap;
  dxbc::TextureAddressMode AddressW = dxbc::TextureAddressMode::Wrap;
  float MipLODBias = 0.f;
  uint32_t MaxAnisotropy = 16;
  dxbc::ComparisonFunc CompFunc = dxbc::ComparisonFunc::LessEqual;
  dxbc::StaticBorderColor BorderColor = dxbc::StaticBorderColor::OpaqueWhite;
  float MinLOD = 0.f;
  float MaxLOD = std::numeric_limits<float>::max();
  uint32_t Space = 0;
  dxbc::ShaderVisibility Visibility = dxbc::ShaderVisibility::All;
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
    std::variant<dxbc::RootFlags, RootConstants, RootDescriptor,
                 DescriptorTable, DescriptorTableClause, StaticSampler>;

/// The following contains the serialization interface for root elements
LLVM_ABI raw_ostream &operator<<(raw_ostream &OS, const dxbc::RootFlags &Flags);
LLVM_ABI raw_ostream &operator<<(raw_ostream &OS,
                                 const RootConstants &Constants);
LLVM_ABI raw_ostream &operator<<(raw_ostream &OS,
                                 const DescriptorTableClause &Clause);
LLVM_ABI raw_ostream &operator<<(raw_ostream &OS, const DescriptorTable &Table);
LLVM_ABI raw_ostream &operator<<(raw_ostream &OS,
                                 const RootDescriptor &Descriptor);
LLVM_ABI raw_ostream &operator<<(raw_ostream &OS,
                                 const StaticSampler &StaticSampler);
LLVM_ABI raw_ostream &operator<<(raw_ostream &OS, const RootElement &Element);

LLVM_ABI void dumpRootElements(raw_ostream &OS, ArrayRef<RootElement> Elements);

} // namespace rootsig
} // namespace hlsl
} // namespace llvm

#endif // LLVM_FRONTEND_HLSL_HLSLROOTSIGNATURE_H
