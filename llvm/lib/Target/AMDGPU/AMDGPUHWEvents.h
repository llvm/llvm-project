//===- AMDGPUHWEvents.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPUHWEVENTS_H
#define LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPUHWEVENTS_H

#include "llvm/ADT/bit.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>
#include <cstdint>
#include <iterator>

namespace llvm {
class GCNSubtarget;
class MachineInstr;
class raw_ostream;
class SIInstrInfo;

namespace AMDGPU {

/// Bit mask of hardware events.
///
/// This is useful to manipulate events as hardware events rarely come alone.
/// This class implements all the usual operators one would need to manipulate a
/// bit mask, and also supports printing to a \ref raw_ostream and iterating
/// over all the set bits of the event mask.
///
/// This class behaves like a constexpr set of flags. None of the methods should
/// be able to mutate the data unless they are assignment operators. Examples:
/// \verbatim
///   A |= B;   // Add flags (union).
///   A -= B;   // Remove flags (substraction).
///   A &= B;   // Intersection.
///   A ^= B;   // Bitwise XOR.
///   (bool)A;  // Check whether any bits are set; A.any() also works.
///   !A;       // Check if no bits are set; A.none() also works.
///   A.size(); // Check how many bits are set.
/// \endverbatim
///
/// This type also provides certain stronger guarantees than a simple integer:
///   - Default constructor initializes the mask to zero.
///   - Constructor ensures undefined bits cannot be set.
class HWEvents {
public:
  using value_type = uint32_t;

  enum : value_type {
    NONE = 0,
#define AMDGPU_HW_EVENT(X, V) X = (1 << V),
#define AMDGPU_LAST_HW_EVENT(X) HWEVENT_LAST_EVENT = X,
#include "AMDGPUHWEvents.def"

    ALL = ((HWEVENT_LAST_EVENT << 1) - 1)
  };

  /// Iterates over the set bits of an HWEvent.
  /// NOLINTNEXTLINE
  class const_iterator
      : public iterator_facade_base<const_iterator, std::forward_iterator_tag,
                                    HWEvents> {
    // The "end" iterator is also the default-constructed iterator.
    // We naturally move towards the "end" by clearing the set bits from least
    // to most significant.
    HWEvents::value_type Cur = 0;

  public:
    const_iterator() = default;
    const_iterator(HWEvents H) : Cur(H.value()) {}

    bool operator==(const const_iterator &Other) const {
      return Cur == Other.Cur;
    }

    HWEvents operator*() const {
      // Return only rightmost (least significant) bit set.
      return Cur ? (Cur & (1 << countr_zero(Cur))) : 0;
    }

    const_iterator &operator++() {
      // Keep all bits except the least significant bit set.
      Cur &= maskTrailingZeros<HWEvents::value_type>(countr_zero(Cur) + 1);
      return *this;
    }
  };

  constexpr HWEvents() = default;
  constexpr HWEvents(value_type V) : Data(V) {
    assert((V & ALL) == V && "Bits set out of bounds!");
  }

  constexpr unsigned size() const { return popcount(Data); }
  constexpr bool any() const { return Data != 0; }
  constexpr bool none() const { return Data == 0; }
  constexpr value_type value() const { return Data; }

  explicit constexpr operator bool() const { return any(); }

  constexpr bool contains(HWEvents Other) const {
    return (~Data & Other.Data) == 0;
  }

  const_iterator begin() const { return *this; }
  const_iterator end() const { return {}; }

  constexpr HWEvents operator|(HWEvents Other) const {
    return Data | Other.Data;
  }
  constexpr HWEvents operator&(HWEvents Other) const {
    return Data & Other.Data;
  }
  constexpr HWEvents operator^(HWEvents Other) const {
    return Data ^ Other.Data;
  }

  constexpr HWEvents operator-(HWEvents Other) const {
    return Data & ~Other.Data;
  }

  constexpr HWEvents operator~() const { return Data ^ ALL; }

  constexpr bool operator==(HWEvents Other) const { return Data == Other.Data; }
  constexpr bool operator!=(HWEvents Other) const { return Data != Other.Data; }

  constexpr HWEvents &operator|=(HWEvents Other) {
    Data |= Other.Data;
    return *this;
  }
  constexpr HWEvents &operator&=(HWEvents Other) {
    Data &= Other.Data;
    return *this;
  }
  constexpr HWEvents &operator^=(HWEvents Other) {
    Data ^= Other.Data;
    return *this;
  }

  constexpr HWEvents operator-=(HWEvents Other) {
    Data &= ~Other.Data;
    return *this;
  }

  /// Overload both bitwise AND operators w/ the value_type to avoid an implicit
  /// conversion to HWEvent in this common pattern used to clear an event bit:
  /// `Events & ~HWEvent::EVENT_TO_CLEAR`.
  /// If we had the implicit conversion to HWEvent, we'd assert because
  /// `~HWEvent::EVENT_TO_CLEAR` has bits set outside of `HWEvent::ALL`.
  constexpr HWEvents operator&(value_type Other) const { return Data & Other; }
  constexpr HWEvents &operator&=(value_type Other) {
    Data &= Other;
    return *this;
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const;
#endif

private:
  value_type Data = NONE;
};

/// \param Inst A VMEM instruction (as per `SIInstrInfo::isVMEM`).
/// \returns the simplified set of events triggered by the VMEM instruction \p
/// Inst. The returned mask is not exhaustive, but is guaranteed to be a subset
/// of the mask that'd be returned by \ref getEventsFor.
///
/// Useful to quickly categorize VMEM instructions without having to fetch all
/// events.
HWEvents getSimplifiedVMEMEventsFor(const MachineInstr &Inst,
                                    const SIInstrInfo &TII);

/// \returns A bitmask of HWEvent triggered by \p Inst
HWEvents getEventsFor(const MachineInstr &Inst, const GCNSubtarget &ST,
                      bool IsExpertMode, bool TgSplit);

} // namespace AMDGPU

raw_ostream &operator<<(raw_ostream &OS, AMDGPU::HWEvents E);
} // namespace llvm

#endif
