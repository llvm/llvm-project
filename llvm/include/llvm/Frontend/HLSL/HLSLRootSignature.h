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

#include "llvm/ADT/IntervalMap.h"
#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DXILABI.h"
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

enum class DescriptorType : uint8_t { SRV = 0, UAV, CBuffer };
// Models RootDescriptor : CBV | SRV | UAV, by collecting like parameters
struct RootDescriptor {
  DescriptorType Type;
  Register Reg;
  uint32_t Space = 0;
  dxbc::ShaderVisibility Visibility = dxbc::ShaderVisibility::All;
  dxbc::RootDescriptorFlags Flags;

  void setDefaultFlags() {
    switch (Type) {
    case DescriptorType::CBuffer:
    case DescriptorType::SRV:
      Flags = dxbc::RootDescriptorFlags::DataStaticWhileSetAtExecute;
      break;
    case DescriptorType::UAV:
      Flags = dxbc::RootDescriptorFlags::DataVolatile;
      break;
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
using ClauseType = llvm::dxil::ResourceClass;
struct DescriptorTableClause {
  ClauseType Type;
  Register Reg;
  uint32_t NumDescriptors = 1;
  uint32_t Space = 0;
  uint32_t Offset = DescriptorTableOffsetAppend;
  dxbc::DescriptorRangeFlags Flags;

  void setDefaultFlags() {
    switch (Type) {
    case ClauseType::CBuffer:
    case ClauseType::SRV:
      Flags = dxbc::DescriptorRangeFlags::DataStaticWhileSetAtExecute;
      break;
    case ClauseType::UAV:
      Flags = dxbc::DescriptorRangeFlags::DataVolatile;
      break;
    case ClauseType::Sampler:
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

struct RangeInfo {
  const static uint32_t Unbounded = ~0u;

  // Interval information
  uint32_t LowerBound;
  uint32_t UpperBound;

  // Information retained for diagnostics
  llvm::dxil::ResourceClass Class;
  uint32_t Space;
  llvm::dxbc::ShaderVisibility Visibility;
};

class ResourceRange {
public:
  using MapT = llvm::IntervalMap<uint32_t, const RangeInfo *, 16,
                                 llvm::IntervalMapInfo<uint32_t>>;

private:
  MapT Intervals;

public:
  ResourceRange(MapT::Allocator &Allocator) : Intervals(MapT(Allocator)) {}

  // Returns a reference to the first RangeInfo that overlaps with
  // [Info.LowerBound;Info.UpperBound], or, std::nullopt if there is no overlap
  LLVM_ABI std::optional<const RangeInfo *>
  getOverlapping(const RangeInfo &Info) const;

  // Return the mapped RangeInfo at X or nullptr if no mapping exists
  const RangeInfo *lookup(uint32_t X) const;

  // Removes all entries of the ResourceRange
  LLVM_ABI void clear();

  // Insert the required (sub-)intervals such that the interval of [a;b] =
  // [Info.LowerBound, Info.UpperBound] is covered and points to a valid
  // RangeInfo &.
  //
  // For instance consider the following chain of inserting RangeInfos with the
  // intervals denoting the Lower/Upper-bounds:
  //
  // A = [0;2]
  //   insert(A) -> false
  //   intervals: [0;2] -> &A
  // B = [5;7]
  //   insert(B) -> false
  //   intervals: [0;2] -> &A, [5;7] -> &B
  // C = [4;7]
  //   insert(C) -> true
  //   intervals: [0;2] -> &A, [4;7] -> &C
  // D = [1;5]
  //   insert(D) -> true
  //   intervals: [0;2] -> &A, [3;3] -> &D, [4;7] -> &C
  // E = [0;unbounded]
  //   insert(E) -> true
  //   intervals: [0;unbounded] -> E
  //
  // Returns a reference to the first RangeInfo that overlaps with
  // [Info.LowerBound;Info.UpperBound], or, std::nullopt if there is no overlap
  // (equivalent to getOverlapping)
  LLVM_ABI std::optional<const RangeInfo *> insert(const RangeInfo &Info);
};

} // namespace rootsig
} // namespace hlsl
} // namespace llvm

#endif // LLVM_FRONTEND_HLSL_HLSLROOTSIGNATURE_H
