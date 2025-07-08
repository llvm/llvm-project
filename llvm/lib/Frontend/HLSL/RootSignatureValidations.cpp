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
  return !(RegisterSpace >= 0xFFFFFFF0 && RegisterSpace <= 0xFFFFFFFF);
}

bool verifyDescriptorFlag(uint32_t Flags) { return (Flags & ~0xE) == 0; }

bool verifyRangeType(uint32_t Type) {
  switch (Type) {
  case llvm::to_underlying(dxbc::DescriptorRangeType::CBV):
  case llvm::to_underlying(dxbc::DescriptorRangeType::SRV):
  case llvm::to_underlying(dxbc::DescriptorRangeType::UAV):
  case llvm::to_underlying(dxbc::DescriptorRangeType::Sampler):
    return true;
  };

  return false;
}

bool verifyDescriptorRangeFlag(uint32_t Version, uint32_t Type,
                               uint32_t FlagsVal) {
  using FlagT = dxbc::DescriptorRangeFlags;
  FlagT Flags = FlagT(FlagsVal);

  const bool IsSampler =
      (Type == llvm::to_underlying(dxbc::DescriptorRangeType::Sampler));

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

std::optional<const RangeInfo *>
ResourceRange::getOverlapping(const RangeInfo &Info) const {
  MapT::const_iterator Interval = Intervals.find(Info.LowerBound);
  if (!Interval.valid() || Info.UpperBound < Interval.start())
    return std::nullopt;
  return Interval.value();
}

const RangeInfo *ResourceRange::lookup(uint32_t X) const {
  return Intervals.lookup(X, nullptr);
}

void ResourceRange::clear() { return Intervals.clear(); }

std::optional<const RangeInfo *> ResourceRange::insert(const RangeInfo &Info) {
  uint32_t LowerBound = Info.LowerBound;
  uint32_t UpperBound = Info.UpperBound;

  std::optional<const RangeInfo *> Res = std::nullopt;
  MapT::iterator Interval = Intervals.begin();

  while (true) {
    if (UpperBound < LowerBound)
      break;

    Interval.advanceTo(LowerBound);
    if (!Interval.valid()) // No interval found
      break;

    // Let Interval = [x;y] and [LowerBound;UpperBound] = [a;b] and note that
    // a <= y implicitly from Intervals.find(LowerBound)
    if (UpperBound < Interval.start())
      break; // found interval does not overlap with inserted one

    if (!Res.has_value()) // Update to be the first found intersection
      Res = Interval.value();

    if (Interval.start() <= LowerBound && UpperBound <= Interval.stop()) {
      // x <= a <= b <= y implies that [a;b] is covered by [x;y]
      //  -> so we don't need to insert this, report an overlap
      return Res;
    } else if (LowerBound <= Interval.start() &&
               Interval.stop() <= UpperBound) {
      // a <= x <= y <= b implies that [x;y] is covered by [a;b]
      //  -> so remove the existing interval that we will cover with the
      //  overwrite
      Interval.erase();
    } else if (LowerBound < Interval.start() && UpperBound <= Interval.stop()) {
      // a < x <= b <= y implies that [a; x] is not covered but [x;b] is
      //  -> so set b = x - 1 such that [a;x-1] is now the interval to insert
      UpperBound = Interval.start() - 1;
    } else if (Interval.start() <= LowerBound && Interval.stop() < UpperBound) {
      // a < x <= b <= y implies that [y; b] is not covered but [a;y] is
      //  -> so set a = y + 1 such that [y+1;b] is now the interval to insert
      LowerBound = Interval.stop() + 1;
    }
  }

  assert(LowerBound <= UpperBound && "Attempting to insert an empty interval");
  Intervals.insert(LowerBound, UpperBound, &Info);
  return Res;
}

} // namespace rootsig
} // namespace hlsl
} // namespace llvm
