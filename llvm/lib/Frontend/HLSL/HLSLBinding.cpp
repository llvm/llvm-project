//===- HLSLBinding.cpp - Representation for resource bindings in HLSL -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/HLSL/HLSLBinding.h"
#include "llvm/ADT/STLExtras.h"

using namespace llvm;
using namespace hlsl;

std::optional<uint32_t>
BindingInfo::findAvailableBinding(dxil::ResourceClass RC, uint32_t Space,
                                  int32_t Size) {
  BindingSpaces &BS = getBindingSpaces(RC);
  RegisterSpace &RS = BS.getOrInsertSpace(Space);
  return RS.findAvailableBinding(Size);
}

BindingInfo::RegisterSpace &
BindingInfo::BindingSpaces::getOrInsertSpace(uint32_t Space) {
  for (auto It = Spaces.begin(), End = Spaces.end(); It != End; ++It) {
    if (It->Space == Space)
      return *It;
    if (It->Space < Space)
      continue;
    return *Spaces.insert(It, Space);
  }
  return Spaces.emplace_back(Space);
}

std::optional<uint32_t>
BindingInfo::RegisterSpace::findAvailableBinding(int32_t Size) {
  assert((Size == -1 || Size > 0) && "invalid size");

  if (FreeRanges.empty())
    return std::nullopt;

  // unbounded array
  if (Size == -1) {
    BindingRange &Last = FreeRanges.back();
    if (Last.UpperBound != ~0u)
      // this space is already occupied by an unbounded array
      return std::nullopt;
    uint32_t RegSlot = Last.LowerBound;
    FreeRanges.pop_back();
    return RegSlot;
  }

  // single resource or fixed-size array
  for (BindingRange &R : FreeRanges) {
    // compare the size as uint64_t to prevent overflow for range (0, ~0u)
    if ((uint64_t)R.UpperBound - R.LowerBound + 1 < (uint64_t)Size)
      continue;
    uint32_t RegSlot = R.LowerBound;
    // This might create a range where (LowerBound == UpperBound + 1). When
    // that happens, the next time this function is called the range will
    // skipped over by the check above (at this point Size is always > 0).
    R.LowerBound += Size;
    return RegSlot;
  }

  return std::nullopt;
}

BindingInfo BindingInfoBuilder::calculateBindingInfo(
    llvm::function_ref<void(const BindingInfoBuilder &Builder,
                            const Binding &Overlapping)>
        ReportOverlap) {
  // sort all the collected bindings
  llvm::stable_sort(Bindings);

  // remove duplicates
  Binding *NewEnd = llvm::unique(Bindings);
  if (NewEnd != Bindings.end())
    Bindings.erase(NewEnd, Bindings.end());

  BindingInfo Info;

  // Go over the sorted bindings and build up lists of free register ranges
  // for each binding type and used spaces. Bindings are sorted by resource
  // class, space, and lower bound register slot.
  BindingInfo::BindingSpaces *BS =
      &Info.getBindingSpaces(dxil::ResourceClass::SRV);
  for (const Binding &B : Bindings) {
    if (BS->RC != B.RC)
      // move to the next resource class spaces
      BS = &Info.getBindingSpaces(B.RC);

    BindingInfo::RegisterSpace *S = BS->Spaces.empty()
                                        ? &BS->Spaces.emplace_back(B.Space)
                                        : &BS->Spaces.back();
    assert(S->Space <= B.Space && "bindings not sorted correctly?");
    if (B.Space != S->Space)
      // add new space
      S = &BS->Spaces.emplace_back(B.Space);

    // The space is full - there are no free slots left, or the rest of the
    // slots are taken by an unbounded array. Report the overlapping to the
    // caller.
    if (S->FreeRanges.empty() || S->FreeRanges.back().UpperBound < ~0u) {
      ReportOverlap(*this, B);
      continue;
    }
    // adjust the last free range lower bound, split it in two, or remove it
    BindingInfo::BindingRange &LastFreeRange = S->FreeRanges.back();
    if (LastFreeRange.LowerBound == B.LowerBound) {
      if (B.UpperBound < ~0u)
        LastFreeRange.LowerBound = B.UpperBound + 1;
      else
        S->FreeRanges.pop_back();
    } else if (LastFreeRange.LowerBound < B.LowerBound) {
      LastFreeRange.UpperBound = B.LowerBound - 1;
      if (B.UpperBound < ~0u)
        S->FreeRanges.emplace_back(B.UpperBound + 1, ~0u);
    } else {
      // We don't have room here. Report the overlapping binding to the caller
      // and mark any extra space this binding would use as unavailable.
      ReportOverlap(*this, B);
      if (B.UpperBound < ~0u)
        LastFreeRange.LowerBound =
            std::max(LastFreeRange.LowerBound, B.UpperBound + 1);
      else
        S->FreeRanges.pop_back();
    }
  }

  return Info;
}

const BindingInfoBuilder::Binding &BindingInfoBuilder::findOverlapping(
    const BindingInfoBuilder::Binding &ReportedBinding) const {
  for (const BindingInfoBuilder::Binding &Other : Bindings)
    if (ReportedBinding.LowerBound <= Other.UpperBound &&
        Other.LowerBound <= ReportedBinding.UpperBound)
      return Other;

  llvm_unreachable("Searching for overlap for binding that does not overlap");
}
