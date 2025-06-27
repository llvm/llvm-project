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

namespace llvm {
namespace hlsl {
namespace rootsig {

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

llvm::SmallVector<OverlappingRanges>
findOverlappingRanges(llvm::SmallVector<RangeInfo> &Infos) {
  // 1. The user has provided the corresponding range information
  llvm::SmallVector<OverlappingRanges> Overlaps;
  using GroupT = std::pair<dxil::ResourceClass, /*Space*/ uint32_t>;

  // 2. Sort the RangeInfo's by their GroupT to form groupings
  std::sort(Infos.begin(), Infos.end(), [](RangeInfo A, RangeInfo B) {
    return std::tie(A.Class, A.Space) < std::tie(B.Class, B.Space);
  });

  // 3. First we will init our state to track:
  if (Infos.size() == 0)
    return Overlaps; // No ranges to overlap
  GroupT CurGroup = {Infos[0].Class, Infos[0].Space};

  // Create a ResourceRange for each Visibility
  ResourceRange::MapT::Allocator Allocator;
  std::array<ResourceRange, 8> Ranges = {
      ResourceRange(Allocator), // All
      ResourceRange(Allocator), // Vertex
      ResourceRange(Allocator), // Hull
      ResourceRange(Allocator), // Domain
      ResourceRange(Allocator), // Geometry
      ResourceRange(Allocator), // Pixel
      ResourceRange(Allocator), // Amplification
      ResourceRange(Allocator), // Mesh
  };

  // Reset the ResourceRanges for when we iterate through a new group
  auto ClearRanges = [&Ranges]() {
    for (ResourceRange &Range : Ranges)
      Range.clear();
  };

  // 3: Iterate through collected RangeInfos
  for (const RangeInfo &Info : Infos) {
    GroupT InfoGroup = {Info.Class, Info.Space};
    // Reset our ResourceRanges when we enter a new group
    if (CurGroup != InfoGroup) {
      ClearRanges();
      CurGroup = InfoGroup;
    }

    // 3A: Insert range info into corresponding Visibility ResourceRange
    ResourceRange &VisRange = Ranges[llvm::to_underlying(Info.Visibility)];
    if (std::optional<const RangeInfo *> Overlapping = VisRange.insert(Info))
      Overlaps.push_back(OverlappingRanges(&Info, Overlapping.value()));

    // 3B: Check for overlap in all overlapping Visibility ResourceRanges
    //
    // If the range that we are inserting has ShaderVisiblity::All it needs to
    // check for an overlap in all other visibility types as well.
    // Otherwise, the range that is inserted needs to check that it does not
    // overlap with ShaderVisibility::All.
    //
    // OverlapRanges will be an ArrayRef to all non-all visibility
    // ResourceRanges in the former case and it will be an ArrayRef to just the
    // all visiblity ResourceRange in the latter case.
    ArrayRef<ResourceRange> OverlapRanges =
        Info.Visibility == llvm::dxbc::ShaderVisibility::All
            ? ArrayRef<ResourceRange>{Ranges}.drop_front()
            : ArrayRef<ResourceRange>{Ranges}.take_front();

    for (const ResourceRange &Range : OverlapRanges)
      if (std::optional<const RangeInfo *> Overlapping =
              Range.getOverlapping(Info))
        Overlaps.push_back(OverlappingRanges(&Info, Overlapping.value()));
  }

  return Overlaps;
}

} // namespace rootsig
} // namespace hlsl
} // namespace llvm
