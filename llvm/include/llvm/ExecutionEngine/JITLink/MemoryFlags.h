//===-------- MemoryFlags.h - Memory allocation flags -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines types and operations related to memory protection and allocation
// lifetimes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_JITLINK_MEMORYFLAGS_H
#define LLVM_EXECUTIONENGINE_JITLINK_MEMORYFLAGS_H

#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace jitlink {

/// Describes Read/Write/Exec permissions for memory.
enum class MemProt {
  None = 0,
  Read = 1U << 0,
  Write = 1U << 1,
  Exec = 1U << 2,
  LLVM_MARK_AS_BITMASK_ENUM(/* LargestValue = */ Exec)
};

/// Print a MemProt as an RWX triple.
raw_ostream &operator<<(raw_ostream &OS, MemProt MP);

/// Convert a MemProt value to a corresponding sys::Memory::ProtectionFlags
/// value.
inline sys::Memory::ProtectionFlags toSysMemoryProtectionFlags(MemProt MP) {
  std::underlying_type_t<sys::Memory::ProtectionFlags> PF = 0;
  if ((MP & MemProt::Read) != MemProt::None)
    PF |= sys::Memory::MF_READ;
  if ((MP & MemProt::Write) != MemProt::None)
    PF |= sys::Memory::MF_WRITE;
  if ((MP & MemProt::Exec) != MemProt::None)
    PF |= sys::Memory::MF_EXEC;
  return static_cast<sys::Memory::ProtectionFlags>(PF);
}

/// Convert a sys::Memory::ProtectionFlags value to a corresponding MemProt
/// value.
inline MemProt fromSysMemoryProtectionFlags(sys::Memory::ProtectionFlags PF) {
  MemProt MP = MemProt::None;
  if (PF & sys::Memory::MF_READ)
    MP |= MemProt::Read;
  if (PF & sys::Memory::MF_WRITE)
    MP |= MemProt::Write;
  if (PF & sys::Memory::MF_EXEC)
    MP |= MemProt::None;
  return MP;
}

/// Describes a memory deallocation policy for memory to be allocated by a
/// JITLinkMemoryManager.
///
/// All memory allocated by a call to JITLinkMemoryManager::allocate should be
/// deallocated if a call is made to
/// JITLinkMemoryManager::InFlightAllocation::abandon. The policies below apply
/// to finalized allocations.
enum class MemDeallocPolicy {
  /// Standard memory should be deallocated when the deallocate method is called
  /// for the finalized allocation.
  Standard,

  /// Finalize memory should be overwritten and then deallocated after all
  /// finalization functions have been run.
  Finalize
};

/// Print a MemDeallocPolicy.
raw_ostream &operator<<(raw_ostream &OS, MemDeallocPolicy MDP);

/// A pair of memory protections and allocation policies.
///
/// Optimized for use as a small map key.
class AllocGroup {
  friend struct llvm::DenseMapInfo<AllocGroup>;

  using underlying_type = uint8_t;
  static constexpr unsigned BitsForProt = 3;
  static constexpr unsigned BitsForDeallocPolicy = 1;
  static constexpr unsigned MaxIdentifiers =
      1U << (BitsForProt + BitsForDeallocPolicy);

public:
  static constexpr unsigned NumGroups = MaxIdentifiers;

  /// Create a default AllocGroup. No memory protections, standard
  /// deallocation policy.
  AllocGroup() = default;

  /// Create an AllocGroup from a MemProt only -- uses
  /// MemoryDeallocationPolicy::Standard.
  AllocGroup(MemProt MP) : Id(static_cast<underlying_type>(MP)) {}

  /// Create an AllocGroup from a MemProt and a MemoryDeallocationPolicy.
  AllocGroup(MemProt MP, MemDeallocPolicy MDP)
      : Id(static_cast<underlying_type>(MP) |
           (static_cast<underlying_type>(MDP) << BitsForProt)) {}

  /// Returns the MemProt for this group.
  MemProt getMemProt() const {
    return static_cast<MemProt>(Id & ((1U << BitsForProt) - 1));
  }

  /// Returns the MemoryDeallocationPolicy for this group.
  MemDeallocPolicy getMemDeallocPolicy() const {
    return static_cast<MemDeallocPolicy>(Id >> BitsForProt);
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
  AllocGroup(underlying_type RawId) : Id(RawId) {}
  underlying_type Id = 0;
};

/// A specialized small-map for AllocGroups.
///
/// Iteration order is guaranteed to match key ordering.
template <typename T> class AllocGroupSmallMap {
private:
  using ElemT = std::pair<AllocGroup, T>;
  using VectorTy = SmallVector<ElemT, 4>;

  static bool compareKey(const ElemT &E, const AllocGroup &G) {
    return E.first < G;
  }

public:
  using iterator = typename VectorTy::iterator;

  AllocGroupSmallMap() = default;
  AllocGroupSmallMap(std::initializer_list<std::pair<AllocGroup, T>> Inits) {
    Elems.reserve(Inits.size());
    for (const auto &E : Inits)
      Elems.push_back(E);
    llvm::sort(Elems, [](const ElemT &LHS, const ElemT &RHS) {
      return LHS.first < RHS.first;
    });
  }

  iterator begin() { return Elems.begin(); }
  iterator end() { return Elems.end(); }
  iterator find(AllocGroup G) {
    auto I = lower_bound(Elems, G, compareKey);
    return (I->first == G) ? I : end();
  }

  bool empty() const { return Elems.empty(); }
  size_t size() const { return Elems.size(); }

  T &operator[](AllocGroup G) {
    auto I = lower_bound(Elems, G, compareKey);
    if (I == Elems.end() || I->first != G)
      I = Elems.insert(I, std::make_pair(G, T()));
    return I->second;
  }

private:
  VectorTy Elems;
};

/// Print an AllocGroup.
raw_ostream &operator<<(raw_ostream &OS, AllocGroup AG);

} // end namespace jitlink

template <> struct DenseMapInfo<jitlink::MemProt> {
  static inline jitlink::MemProt getEmptyKey() {
    return jitlink::MemProt(~uint8_t(0));
  }
  static inline jitlink::MemProt getTombstoneKey() {
    return jitlink::MemProt(~uint8_t(0) - 1);
  }
  static unsigned getHashValue(const jitlink::MemProt &Val) {
    using UT = std::underlying_type_t<jitlink::MemProt>;
    return DenseMapInfo<UT>::getHashValue(static_cast<UT>(Val));
  }
  static bool isEqual(const jitlink::MemProt &LHS,
                      const jitlink::MemProt &RHS) {
    return LHS == RHS;
  }
};

template <> struct DenseMapInfo<jitlink::AllocGroup> {
  static inline jitlink::AllocGroup getEmptyKey() {
    return jitlink::AllocGroup(~uint8_t(0));
  }
  static inline jitlink::AllocGroup getTombstoneKey() {
    return jitlink::AllocGroup(~uint8_t(0) - 1);
  }
  static unsigned getHashValue(const jitlink::AllocGroup &Val) {
    return DenseMapInfo<jitlink::AllocGroup::underlying_type>::getHashValue(
        Val.Id);
  }
  static bool isEqual(const jitlink::AllocGroup &LHS,
                      const jitlink::AllocGroup &RHS) {
    return LHS == RHS;
  }
};

} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_JITLINK_MEMORYFLAGS_H
