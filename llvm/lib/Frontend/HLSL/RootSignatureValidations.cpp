//===- HLSLRootSignatureValidations.cpp - HLSL Root Signature helpers -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains helpers for working with HLSL Root Signatures.
///
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/HLSL/RootSignatureValidations.h"

#include <cmath>

namespace llvm {
namespace hlsl {
namespace rootsig {

bool verifyRootFlag(uint32_t Flags) { return (Flags & ~0xfff) == 0; }

bool verifyVersion(uint32_t Version) { return (Version == 1 || Version == 2); }

bool verifyRegisterValue(uint32_t RegisterValue) {
  return RegisterValue != ~0U;
}

// This Range is reserverved, therefore invalid, according to the spec
// https://github.com/llvm/wg-hlsl/blob/main/proposals/0002-root-signature-in-clang.md#all-the-values-should-be-legal
bool verifyRegisterSpace(uint32_t RegisterSpace) {
  return !(RegisterSpace >= 0xFFFFFFF0);
}

bool verifyRootDescriptorFlag(uint32_t Version, uint32_t FlagsVal) {
  using FlagT = dxbc::RootDescriptorFlags;
  FlagT Flags = FlagT(FlagsVal);
  if (Version == 1)
    return Flags == FlagT::DataVolatile;

  assert(Version == 2 && "Provided invalid root signature version");

  // The data-specific flags are mutually exclusive.
  FlagT DataFlags = FlagT::DataVolatile | FlagT::DataStatic |
                    FlagT::DataStaticWhileSetAtExecute;

  if (popcount(llvm::to_underlying(Flags & DataFlags)) > 1)
    return false;

  // Only a data flag or no flags is valid
  return (Flags | DataFlags) == DataFlags;
}

bool verifyDescriptorRangeFlag(uint32_t Version, dxil::ResourceClass Type,
                               dxbc::DescriptorRangeFlags Flags) {
  using FlagT = dxbc::DescriptorRangeFlags;

  const bool IsSampler = (Type == dxil::ResourceClass::Sampler);

  if (Version == 1) {
    // Since the metadata is unversioned, we expect to explicitly see the values
    // that map to the version 1 behaviour here.
    if (IsSampler)
      return Flags == FlagT::DescriptorsVolatile;
    return Flags == (FlagT::DataVolatile | FlagT::DescriptorsVolatile);
  }

  // The data-specific flags are mutually exclusive.
  FlagT DataFlags = FlagT::DataVolatile | FlagT::DataStatic |
                    FlagT::DataStaticWhileSetAtExecute;

  if (popcount(llvm::to_underlying(Flags & DataFlags)) > 1)
    return false;

  // The descriptor-specific flags are mutually exclusive.
  FlagT DescriptorFlags = FlagT::DescriptorsStaticKeepingBufferBoundsChecks |
                          FlagT::DescriptorsVolatile;
  if (popcount(llvm::to_underlying(Flags & DescriptorFlags)) > 1)
    return false;

  // For volatile descriptors, DATA_is never valid.
  if ((Flags & FlagT::DescriptorsVolatile) == FlagT::DescriptorsVolatile) {
    FlagT Mask = FlagT::DescriptorsVolatile;
    if (!IsSampler) {
      Mask |= FlagT::DataVolatile;
      Mask |= FlagT::DataStaticWhileSetAtExecute;
    }
    return (Flags & ~Mask) == FlagT::None;
  }

  // For "KEEPING_BUFFER_BOUNDS_CHECKS" descriptors,
  // the other data-specific flags may all be set.
  if ((Flags & FlagT::DescriptorsStaticKeepingBufferBoundsChecks) ==
      FlagT::DescriptorsStaticKeepingBufferBoundsChecks) {
    FlagT Mask = FlagT::DescriptorsStaticKeepingBufferBoundsChecks;
    if (!IsSampler) {
      Mask |= FlagT::DataVolatile;
      Mask |= FlagT::DataStatic;
      Mask |= FlagT::DataStaticWhileSetAtExecute;
    }
    return (Flags & ~Mask) == FlagT::None;
  }

  // When no descriptor flag is set, any data flag is allowed.
  FlagT Mask = FlagT::None;
  if (!IsSampler) {
    Mask |= FlagT::DataVolatile;
    Mask |= FlagT::DataStaticWhileSetAtExecute;
    Mask |= FlagT::DataStatic;
  }
  return (Flags & ~Mask) == FlagT::None;
}

bool verifyNumDescriptors(uint32_t NumDescriptors) {
  return NumDescriptors > 0;
}

bool verifySamplerFilter(uint32_t Value) {
  switch (Value) {
#define FILTER(Num, Val) case llvm::to_underlying(dxbc::SamplerFilter::Val):
#include "llvm/BinaryFormat/DXContainerConstants.def"
    return true;
  }
  return false;
}

// Values allowed here:
// https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_texture_address_mode#syntax
bool verifyAddress(uint32_t Address) {
  switch (Address) {
#define TEXTURE_ADDRESS_MODE(Num, Val)                                         \
  case llvm::to_underlying(dxbc::TextureAddressMode::Val):
#include "llvm/BinaryFormat/DXContainerConstants.def"
    return true;
  }
  return false;
}

bool verifyMipLODBias(float MipLODBias) {
  return MipLODBias >= -16.f && MipLODBias <= 15.99f;
}

bool verifyMaxAnisotropy(uint32_t MaxAnisotropy) {
  return MaxAnisotropy <= 16u;
}

bool verifyComparisonFunc(uint32_t ComparisonFunc) {
  switch (ComparisonFunc) {
#define COMPARISON_FUNC(Num, Val)                                              \
  case llvm::to_underlying(dxbc::ComparisonFunc::Val):
#include "llvm/BinaryFormat/DXContainerConstants.def"
    return true;
  }
  return false;
}

bool verifyBorderColor(uint32_t BorderColor) {
  switch (BorderColor) {
#define STATIC_BORDER_COLOR(Num, Val)                                          \
  case llvm::to_underlying(dxbc::StaticBorderColor::Val):
#include "llvm/BinaryFormat/DXContainerConstants.def"
    return true;
  }
  return false;
}

bool verifyLOD(float LOD) { return !std::isnan(LOD); }

bool verifyBoundOffset(uint32_t Offset) {
  return Offset != NumDescriptorsUnbounded;
}

bool verifyNoOverflowedOffset(uint64_t Offset) {
  return Offset <= std::numeric_limits<uint32_t>::max();
}

uint64_t computeRangeBound(uint32_t Offset, uint32_t Size) {
  assert(0 < Size && "Must be a non-empty range");
  if (Size == NumDescriptorsUnbounded)
    return NumDescriptorsUnbounded;

  return uint64_t(Offset) + uint64_t(Size) - 1;
}

} // namespace rootsig
} // namespace hlsl
} // namespace llvm
