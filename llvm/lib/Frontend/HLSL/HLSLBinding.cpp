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
  BindingSpaces<FreeRegisterSpace> &BS = getBindingSpaces(RC);
  FreeRegisterSpace &RS = BS.getOrInsertSpace(Space);
  return RS.findAvailableBinding(Size);
}

template <typename T> T &BindingSpaces<T>::getOrInsertSpace(uint32_t Space) {
  for (auto It = Spaces.begin(), End = Spaces.end(); It != End; ++It) {
    if (It->Space == Space)
      return *It;
    if (It->Space < Space)
      continue;
    return *Spaces.insert(It, Space);
  }
  return Spaces.emplace_back(Space);
}

template <typename T>
std::optional<const T *> BindingSpaces<T>::contains(uint32_t Space) const {
  const T *It = llvm::find(Spaces, Space);
  if (It == Spaces.end())
    return std::nullopt;
  return It;
}

std::optional<uint32_t> FreeRegisterSpace::findAvailableBinding(int32_t Size) {
  assert((Size == -1 || Size > 0) && "invalid size");

  if (Ranges.empty())
    return std::nullopt;

  // unbounded array
  if (Size == -1) {
    BindingRange &Last = Ranges.back();
    if (Last.UpperBound != ~0u)
      // this space is already occupied by an unbounded array
      return std::nullopt;
    uint32_t RegSlot = Last.LowerBound;
    Ranges.pop_back();
    return RegSlot;
  }

  // single resource or fixed-size array
  for (BindingRange &R : Ranges) {
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
  BindingSpaces<FreeRegisterSpace> *BS =
      &Info.getBindingSpaces(dxil::ResourceClass::SRV);
  for (const Binding &B : Bindings) {
    if (BS->RC != B.RC)
      // move to the next resource class spaces
      BS = &Info.getBindingSpaces(B.RC);

    FreeRegisterSpace *S = BS->Spaces.empty()
                               ? &BS->Spaces.emplace_back(B.Space)
                               : &BS->Spaces.back();
    assert(S->Space <= B.Space && "bindings not sorted correctly?");
    if (B.Space != S->Space)
      // add new space
      S = &BS->Spaces.emplace_back(B.Space);

    // The space is full - there are no free slots left, or the rest of the
    // slots are taken by an unbounded array. Report the overlapping to the
    // caller.
    if (S->Ranges.empty() || S->Ranges.back().UpperBound < ~0u) {
      ReportOverlap(*this, B);
      continue;
    }
    // adjust the last free range lower bound, split it in two, or remove it
    BindingRange &LastFreeRange = S->Ranges.back();
    if (LastFreeRange.LowerBound == B.LowerBound) {
      if (B.UpperBound < ~0u)
        LastFreeRange.LowerBound = B.UpperBound + 1;
      else
        S->Ranges.pop_back();
    } else if (LastFreeRange.LowerBound < B.LowerBound) {
      LastFreeRange.UpperBound = B.LowerBound - 1;
      if (B.UpperBound < ~0u)
        S->Ranges.emplace_back(B.UpperBound + 1, ~0u);
    } else {
      // We don't have room here. Report the overlapping binding to the caller
      // and mark any extra space this binding would use as unavailable.
      ReportOverlap(*this, B);
      if (B.UpperBound < ~0u)
        LastFreeRange.LowerBound =
            std::max(LastFreeRange.LowerBound, B.UpperBound + 1);
      else
        S->Ranges.pop_back();
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

bool BusyRegisterSpace::isBound(const BindingRange &Range) const {
  const BindingRange *It = llvm::lower_bound(
      Ranges, Range.LowerBound,
      [](const BindingRange &R, uint32_t Val) { return R.UpperBound < Val; });

  if (It == Ranges.end())
    return false;
  return ((Range.LowerBound >= It->LowerBound) &&
          (Range.UpperBound <= It->UpperBound));
}

bool BusyBindingInfo::isBound(dxil::ResourceClass RC, uint32_t Space,
                              const BindingRange &Range) const {
  const BindingSpaces<BusyRegisterSpace> &BS = getBindingSpaces(RC);
  std::optional<const BusyRegisterSpace *> RS = BS.contains(Space);
  if (!RS)
    return false;
  return RS.value()->isBound(Range);
}

BusyBindingInfo BindingInfoBuilder::calculateBusyBindingInfo(
    llvm::function_ref<void(const BindingInfoBuilder &Builder,
                            const Binding &Overlapping)>
        ReportOverlap) {
  // sort all the collected bindings
  llvm::stable_sort(Bindings);

  // remove duplicates
  Binding *NewEnd = llvm::unique(Bindings);
  if (NewEnd != Bindings.end())
    Bindings.erase(NewEnd, Bindings.end());

  BusyBindingInfo Info;

  BindingSpaces<BusyRegisterSpace> *BS =
      &Info.getBindingSpaces(dxil::ResourceClass::SRV);
  for (const Binding &B : Bindings) {
    if (BS->RC != B.RC)
      // move to the next resource class spaces
      BS = &Info.getBindingSpaces(B.RC);

    BusyRegisterSpace *S = BS->Spaces.empty()
                               ? &BS->Spaces.emplace_back(B.Space)
                               : &BS->Spaces.back();
    assert(S->Space <= B.Space && "bindings not sorted correctly?");

    if (B.Space != S->Space)
      S = &BS->Spaces.emplace_back(B.Space);

    if (!S->Ranges.empty()) {
      // check for overlap with the last range only, since the bindings are
      // sorted and there cannot be any overlap with earlier ranges.
      const BindingRange Back = S->Ranges.back();
      if (Back.overlapsWith({B.LowerBound, B.UpperBound})) {
        ReportOverlap(*this, B);
        continue;
      }
    }
    S->Ranges.emplace_back(B.LowerBound, B.UpperBound);
  }
  return Info;
}
