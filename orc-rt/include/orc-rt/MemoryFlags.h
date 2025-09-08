//===--------- MemoryFlags.h -- Memory allocation flags ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Memory allocation flags.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_MEMORYFLAGS_H
#define ORC_RT_MEMORYFLAGS_H

#include "orc-rt/BitmaskEnum.h"
#include "orc-rt/bit.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace orc_rt {

/// Describes Read/Write/Exec permissions for memory.
enum class MemProt : unsigned {
  None = 0,
  Read = 1U << 0,
  Write = 1U << 1,
  Exec = 1U << 2,
  ORC_RT_MARK_AS_BITMASK_ENUM(/* LargestValue = */ Exec)
};

/// Describes a memory lifetime policy.
enum class MemLifetime : unsigned {
  /// Standard memory should be deallocated by the corresponding call to
  /// deallocate.
  Standard,

  /// Finalize memory should be deallocated at the end of the finalization
  /// process.
  Finalize
};

/// A pair of memory protections and lifetime policy.
class AllocGroup {
private:
  static constexpr int NumProtBits = bitmask_enum_num_bits_v<MemProt>;
  static constexpr int NumLifetimeBits = 1;
  static constexpr int NumBits = NumProtBits + NumLifetimeBits;

  typedef uint8_t underlying_type;

  static_assert(NumBits <= std::numeric_limits<underlying_type>::digits,
                "Not enough bits to hold (prot, lifetime) pair");

  constexpr static underlying_type ProtMask = (1U << NumProtBits) - 1;
  constexpr static underlying_type LifetimeMask = (1U << NumLifetimeBits) - 1;

public:
  static constexpr size_t MaxValues = 1U << NumBits;

  AllocGroup() = default;
  AllocGroup(MemProt MP, MemLifetime ML = MemLifetime::Standard)
      : Id((static_cast<underlying_type>(ML) << NumProtBits) |
           static_cast<underlying_type>(MP)) {}

  MemProt getMemProt() const { return static_cast<MemProt>(Id & ProtMask); }

  MemLifetime getMemLifetime() const {
    return static_cast<MemLifetime>((Id >> NumProtBits) & LifetimeMask);
  }

  friend bool operator==(const AllocGroup &LHS, const AllocGroup &RHS) {
    return LHS.Id == RHS.Id;
  }

  friend bool operator!=(const AllocGroup &LHS, const AllocGroup &RHS) {
    return !(LHS == RHS);
  }

  friend bool operator<(const AllocGroup &LHS, const AllocGroup &RHS) {
    return LHS.Id < RHS.Id;
  }

private:
  underlying_type Id = 0;
};

/// A specialized small-map for AllocGroups.
///
/// Iteration order is guaranteed to match key ordering.
template <typename T> class AllocGroupSmallMap {
private:
  using ElemT = std::pair<AllocGroup, T>;
  using VectorTy = std::vector<ElemT>;

  static bool compareKey(const ElemT &E, const AllocGroup &G) {
    return E.first < G;
  }

public:
  using iterator = typename VectorTy::iterator;

  AllocGroupSmallMap() = default;
  AllocGroupSmallMap(std::initializer_list<std::pair<AllocGroup, T>> Inits)
      : Elems(Inits) {
    std::sort(Elems, [](const ElemT &LHS, const ElemT &RHS) {
      return LHS.first < RHS.first;
    });
  }

  iterator begin() { return Elems.begin(); }
  iterator end() { return Elems.end(); }
  iterator find(AllocGroup G) {
    auto I = std::lower_bound(Elems.begin(), Elems.end(), G, compareKey);
    return (I == end() || I->first == G) ? I : end();
  }

  bool empty() const { return Elems.empty(); }
  size_t size() const { return Elems.size(); }

  T &operator[](AllocGroup G) {
    auto I = std::lower_bound(Elems.begin(), Elems.end(), G, compareKey);
    if (I == Elems.end() || I->first != G)
      I = Elems.insert(I, std::make_pair(G, T()));
    return I->second;
  }

private:
  VectorTy Elems;
};

} // namespace orc_rt

#endif // ORC_RT_MEMORYFLAGS_H
