//===- AMDGPUHWEvents.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPUHWEVENTS_H
#define LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPUHWEVENTS_H

#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class raw_ostream;

namespace AMDGPU {

/// TODO: This should be a bitmask from the start instead of having this enum
///       + \ref HWEventSet below.
enum class HWEvent : unsigned char {
#define AMDGPU_HW_EVENT(X) X,
#define AMDGPU_FIRST_HW_EVENT(X) FIRST_WAIT_EVENT = X,
#define AMDGPU_LAST_HW_EVENT(X) NUM_WAIT_EVENTS = X,
#include "AMDGPUHWEvents.def"
};

} // namespace AMDGPU

template <> struct enum_iteration_traits<AMDGPU::HWEvent> {
  static constexpr bool is_iterable = true; // NOLINT
};

namespace AMDGPU {

static constexpr StringLiteral toString(HWEvent Event) {
  switch (Event) {
#define AMDGPU_HW_EVENT(EVENT)                                                 \
  case HWEvent::EVENT:                                                         \
    return #EVENT;
#include "AMDGPUHWEvents.def"
  }

  return "";
}

/// Return an iterator over all events between FIRST_WAIT_EVENT
/// and \c MaxEvent (exclusive, default value yields an enumeration over
/// all counters).
// NOLINTNEXTLINE
inline iota_range<HWEvent>
hw_events(HWEvent MaxEvent = HWEvent::NUM_WAIT_EVENTS) {
  return enum_seq(HWEvent::FIRST_WAIT_EVENT, MaxEvent);
}

class HWEventSet {
  unsigned Mask = 0;

public:
  HWEventSet() = default;
  explicit constexpr HWEventSet(HWEvent Event) {
    static_assert(static_cast<unsigned>(HWEvent::NUM_WAIT_EVENTS) <=
                      sizeof(Mask) * 8,
                  "Not enough bits in Mask for all the events");
    Mask |= 1 << static_cast<unsigned>(Event);
  }
  constexpr HWEventSet(std::initializer_list<HWEvent> Events) {
    for (auto &E : Events) {
      Mask |= 1 << static_cast<unsigned>(E);
    }
  }
  void insert(const HWEvent &Event) {
    Mask |= 1 << static_cast<unsigned>(Event);
  }
  void remove(const HWEvent &Event) {
    Mask &= ~(1 << static_cast<unsigned>(Event));
  }
  void remove(const HWEventSet &Other) { Mask &= ~Other.Mask; }
  bool contains(const HWEvent &Event) const {
    return Mask & (1 << static_cast<unsigned>(Event));
  }
  /// \returns true if this set contains all elements of \p Other.
  bool contains(const HWEventSet &Other) const {
    return (~Mask & Other.Mask) == 0;
  }
  /// \returns the intersection of this and \p Other.
  HWEventSet operator&(const HWEventSet &Other) const {
    auto Copy = *this;
    Copy.Mask &= Other.Mask;
    return Copy;
  }
  /// \returns the union of this and \p Other.
  HWEventSet operator|(const HWEventSet &Other) const {
    auto Copy = *this;
    Copy.Mask |= Other.Mask;
    return Copy;
  }
  /// This set becomes the union of this and \p Other.
  HWEventSet &operator|=(const HWEventSet &Other) {
    Mask |= Other.Mask;
    return *this;
  }
  /// This set becomes the intersection of this and \p Other.
  HWEventSet &operator&=(const HWEventSet &Other) {
    Mask &= Other.Mask;
    return *this;
  }
  bool operator==(const HWEventSet &Other) const { return Mask == Other.Mask; }
  bool operator!=(const HWEventSet &Other) const { return !(*this == Other); }
  bool empty() const { return Mask == 0; }
  /// \returns true if the set contains more than one element.
  bool twoOrMore() const { return Mask & (Mask - 1); }
  operator bool() const { return !empty(); }
  void print(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
};

} // namespace AMDGPU
} // namespace llvm

#endif
