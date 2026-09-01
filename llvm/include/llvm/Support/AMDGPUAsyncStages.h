//===- AMDGPUAsyncStages.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Shared AMDGPU asyncmark stage definitions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_AMDGPUASYNCSTAGES_H
#define LLVM_SUPPORT_AMDGPUASYNCSTAGES_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/bit.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <iterator>

namespace llvm {
namespace AMDGPU {

/// Bit mask of the async stages tracked by the asyncmark / wait_asyncmark
/// intrinsics. Each stage has its own independent sequence of marks.
///
/// A single stage is represented as a mask with exactly one bit set. Iterating
/// over a mask visits each stage it names, from the least significant bit up.
///
/// This type also provides certain stronger guarantees than a simple integer:
///   - Default constructor initializes the mask to zero.
///   - Constructor ensures undefined bits cannot be set.
class AsyncStages {
public:
  using value_type = uint32_t;

  static constexpr unsigned NUM_STAGES = 11;

  // The bit of each stage is part of the IR, so do not renumber them. Some
  // values are RESERVED for later use.
  enum : value_type {
    NONE = 0,
    // Tensor loads to LDS and tensor stores from LDS.
    TENSOR = 1u << 0,
    // Asynchronous global loads to LDS.
    GLOBAL_LOAD_ASYNC_TO_LDS = 1u << 1,
    // Asynchronous multicast (cluster) global loads to LDS.
    GLOBAL_LOAD_ASYNC_TO_LDS_MCAST = 1u << 2,
    // Asynchronous global stores from LDS.
    GLOBAL_STORE_ASYNC_FROM_LDS = 1u << 3,
    // Held for downstream use.
    RESERVED_4 = 1u << 4,
    // Buffer loads to LDS and pre-gfx1250 global loads to LDS.
    BUFFER_GLOBAL_LOAD = 1u << 5,
    RESERVED_6 = 1u << 6,
    RESERVED_7 = 1u << 7,
    RESERVED_8 = 1u << 8,
    RESERVED_9 = 1u << 9,
    RESERVED_10 = 1u << 10,

    RESERVED = RESERVED_4 | RESERVED_6 | RESERVED_7 | RESERVED_8 | RESERVED_9 |
               RESERVED_10,

    // Bits that a mask may legally set. Reserved stages are included: naming a
    // stage whose operations do not exist yet is harmless, and accepting the
    // bit keeps masks portable as stages are filled in.
    ALL = (1u << NUM_STAGES) - 1
  };

  /// Iterates over the stages named by a mask.
  /// NOLINTNEXTLINE
  class const_iterator
      : public iterator_facade_base<const_iterator, std::forward_iterator_tag,
                                    AsyncStages> {
    // The "end" iterator is also the default-constructed iterator.
    // We naturally move towards the "end" by clearing the set bits from least
    // to most significant.
    AsyncStages::value_type Cur = 0;

  public:
    const_iterator() = default;
    const_iterator(AsyncStages S) : Cur(S.value()) {}

    bool operator==(const const_iterator &Other) const {
      return Cur == Other.Cur;
    }

    AsyncStages operator*() const {
      // Return only rightmost (least significant) bit set.
      return Cur ? (Cur & (1u << countr_zero(Cur))) : 0;
    }

    const_iterator &operator++() {
      // Keep all bits except the least significant bit set.
      Cur &= maskTrailingZeros<AsyncStages::value_type>(countr_zero(Cur) + 1);
      return *this;
    }
  };

  constexpr AsyncStages() = default;
  constexpr AsyncStages(value_type V) : Data(V) {
    assert((V & ALL) == V && "Bits set out of bounds!");
  }

  /// \returns true if \p Imm is a legal stage mask operand of the asyncmark /
  /// wait_asyncmark intrinsics and pseudos.
  static constexpr bool isValidMask(int64_t Imm) {
    return Imm >= 0 && (Imm & ~int64_t(ALL)) == 0;
  }

  /// \returns the stages named by the stage mask operand \p Imm of the
  /// asyncmark / wait_asyncmark intrinsics and pseudos. The zero mask is the
  /// one exception to a set bit naming a stage: it names every stage rather
  /// than none.
  static constexpr AsyncStages fromMaskOperand(value_type Imm) {
    return Imm ? AsyncStages(Imm) : AsyncStages(ALL);
  }

  constexpr unsigned size() const { return popcount(Data); }
  constexpr bool any() const { return Data != 0; }
  constexpr bool none() const { return Data == 0; }
  constexpr value_type value() const { return Data; }

  explicit constexpr operator bool() const { return any(); }

  const_iterator begin() const { return *this; }
  const_iterator end() const { return {}; }

  /// \returns the position of this single stage in a mask, for indexing
  /// per-stage data.
  unsigned index() const {
    assert(size() == 1 && "Expected a single stage");
    return countr_zero(Data);
  }

  /// \returns the name of this single stage.
  const char *getName() const {
    switch (Data) {
    case TENSOR:
      return "TENSOR";
    case GLOBAL_LOAD_ASYNC_TO_LDS:
      return "GLOBAL_LOAD_ASYNC_TO_LDS";
    case GLOBAL_LOAD_ASYNC_TO_LDS_MCAST:
      return "GLOBAL_LOAD_ASYNC_TO_LDS_MCAST";
    case GLOBAL_STORE_ASYNC_FROM_LDS:
      return "GLOBAL_STORE_ASYNC_FROM_LDS";
    case RESERVED_4:
      return "RESERVED_4";
    case BUFFER_GLOBAL_LOAD:
      return "BUFFER_GLOBAL_LOAD";
    case RESERVED_6:
      return "RESERVED_6";
    case RESERVED_7:
      return "RESERVED_7";
    case RESERVED_8:
      return "RESERVED_8";
    case RESERVED_9:
      return "RESERVED_9";
    case RESERVED_10:
      return "RESERVED_10";
    }
    llvm_unreachable("Expected a single stage");
  }

  constexpr AsyncStages operator&(AsyncStages Other) const {
    return Data & Other.Data;
  }
  constexpr AsyncStages operator-(AsyncStages Other) const {
    return Data & ~Other.Data;
  }

  constexpr bool operator==(AsyncStages Other) const {
    return Data == Other.Data;
  }

  constexpr AsyncStages &operator|=(AsyncStages Other) {
    Data |= Other.Data;
    return *this;
  }

private:
  value_type Data = NONE;
};

/// Print the stages named by \p S as a '|'-separated list, or "all" if it
/// names every stage.
inline raw_ostream &operator<<(raw_ostream &OS, AsyncStages S) {
  if (S == AsyncStages::ALL)
    return OS << "all";
  if (S.none())
    return OS << "none";
  interleave(S, OS, [&OS](AsyncStages Stage) { OS << Stage.getName(); }, "|");
  return OS;
}

} // namespace AMDGPU
} // namespace llvm

#endif // LLVM_SUPPORT_AMDGPUASYNCSTAGES_H
