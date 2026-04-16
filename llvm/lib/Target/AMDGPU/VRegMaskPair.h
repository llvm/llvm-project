//===-- VRegMaskPair.h - Virtual Register and Lane Mask Pair ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Defines VRegMaskPair and VRegMaskPairSet for managing sets of
/// virtual registers and their lane masks.
///
/// Set operations (union, intersection, subtraction) are implemented based on
/// *subregister coverage logic* rather than exact equality. This means:
/// - Two VRegMaskPairs are considered overlapping if their LaneMasks overlap.
/// - Intersection and subtraction operate on *overlapping masks*, not exact
/// matches.
///
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIB_TARGET_VREGMASKPAIR_H
#define LLVM_LIB_TARGET_VREGMASKPAIR_H

#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/MC/LaneBitmask.h"
#include "llvm/Support/Compiler.h"
#include <cassert>

using namespace llvm;

class VRegMaskPairSet;

class VRegMaskPair {
  friend class VRegMaskPairSet;

  Register VReg;
  LaneBitmask LaneMask;

public:
  VRegMaskPair(Register VReg, LaneBitmask LaneMask)
      : VReg(VReg), LaneMask(LaneMask) {}

  VRegMaskPair() : VReg(AMDGPU::NoRegister), LaneMask(LaneBitmask::getNone()) {}
  VRegMaskPair(const VRegMaskPair &Other) = default;
  VRegMaskPair(VRegMaskPair &&Other) = default;
  VRegMaskPair &operator=(const VRegMaskPair &Other) = default;
  VRegMaskPair &operator=(VRegMaskPair &&Other) = default;

  VRegMaskPair(const MachineOperand MO, const SIRegisterInfo *TRI,
               const MachineRegisterInfo *MRI) {
    assert(MO.isReg() && "Not a register operand!");
    assert(MO.getReg().isVirtual() && "Not a virtual register!");
    VReg = MO.getReg();
    LaneMask = MO.getSubReg() ? TRI->getSubRegIndexLaneMask(MO.getSubReg())
                              : MRI->getMaxLaneMaskForVReg(VReg);
  }

  const Register getVReg() const { return VReg; }
  const LaneBitmask getLaneMask() const { return LaneMask; }

  unsigned getSubReg(const MachineRegisterInfo *MRI,
                     const SIRegisterInfo *TRI) const {
    LaneBitmask Mask = MRI->getMaxLaneMaskForVReg(VReg);
    if (LaneMask == Mask)
      return AMDGPU::NoRegister;
    return TRI->getSubRegIndexForLaneMask(LaneMask);
  }

  const TargetRegisterClass *getRegClass(const MachineRegisterInfo *MRI,
                                         const SIRegisterInfo *TRI) const {
    const TargetRegisterClass *RC = TRI->getRegClassForReg(*MRI, VReg);
    LaneBitmask Mask = MRI->getMaxLaneMaskForVReg(VReg);
    if (LaneMask != Mask) {
      unsigned SubRegIdx = TRI->getSubRegIndexForLaneMask(LaneMask);
      return TRI->getSubRegisterClass(RC, SubRegIdx);
    }
    return RC;
  }

  unsigned getSizeInRegs(const SIRegisterInfo *TRI) const {
    return TRI->getNumCoveredRegs(LaneMask);
  }

  bool operator==(const VRegMaskPair &other) const {
    return VReg == other.VReg && LaneMask == other.LaneMask;
  }
};

class LaneCoverageResult {
  friend class VRegMaskPairSet;
  LaneBitmask Data;
  LaneBitmask Covered;
  LaneBitmask NotCovered;

public:
  LaneCoverageResult() = default;
  LaneCoverageResult(const LaneBitmask Mask) : Data(Mask), NotCovered(Mask) {};
  bool isFullyCovered() { return Data == Covered; }
  bool isFullyUncovered() { return Data == NotCovered; }
  LaneBitmask getCovered() { return Covered; }
  LaneBitmask getNotCovered() { return NotCovered; }
};

class VRegMaskPairSet {
  // std::set provides both sorted iteration (required for getCoverage and
  // iteration semantics) AND efficient membership testing (required by
  // insert/contains/count). A sorted vector would provide ordering but
  // only O(log n) lookup via binary search, while std::set provides O(log n)
  // for both operations with better performance characteristics for our
  // typical workload of frequent inserts and lookups.
  using MaskSet = std::set<LaneBitmask>;
  using SetStorageT = DenseMap<Register, MaskSet>;
  using LinearStorageT = std::vector<VRegMaskPair>;

  SetStorageT SetStorage;
  LinearStorageT LinearStorage;

public:
  VRegMaskPairSet() = default;

  template <typename ContainerT,
            typename = std::enable_if_t<std::is_same<
                typename ContainerT::value_type, VRegMaskPair>::value>>
  VRegMaskPairSet(const ContainerT &Vec) {
    for (const auto &VMP : Vec)
      insert(VMP);
  }

  template <typename ContainerT,
            typename = std::enable_if_t<std::is_same<
                typename ContainerT::value_type, VRegMaskPair>::value>>
  VRegMaskPairSet(ContainerT &&Vec) {
    for (auto &&VMP : Vec)
      insert(std::move(VMP));
  }

  bool insert(const VRegMaskPair &VMP) {
    auto &MaskSet = SetStorage[VMP.VReg];
    auto Inserted = MaskSet.insert(VMP.LaneMask);
    if (!Inserted.second)
      return false;
    LinearStorage.push_back(VMP);
    return true;
  }

  template <typename InputIt> void insert(InputIt First, InputIt Last) {
    for (auto It = First; It != Last; ++It)
      insert(*It);
  }

  void remove(const VRegMaskPair &VMP) {
    auto MapIt = SetStorage.find(VMP.VReg);
    if (MapIt == SetStorage.end())
      return;

    size_t Erased = MapIt->second.erase(VMP.LaneMask);
    if (!Erased)
      return;

    if (MapIt->second.empty())
      SetStorage.erase(MapIt);

    auto VecIt = std::find(LinearStorage.begin(), LinearStorage.end(), VMP);
    if (VecIt != LinearStorage.end()) {
      LinearStorage.erase(VecIt);
    } else {
      llvm_unreachable("Inconsistent LinearStorage: VMP missing on remove");
    }
  }

  template <typename Predicate> void remove_if(Predicate Pred) {
    for (auto It = LinearStorage.begin(); It != LinearStorage.end();) {
      const VRegMaskPair VMP = *It;
      if (Pred(VMP)) {
        It = LinearStorage.erase(It);
        SetStorage[VMP.VReg].erase(VMP.LaneMask);
        if (SetStorage[VMP.VReg].empty())
          SetStorage.erase(VMP.VReg);
      } else {
        ++It;
      }
    }
  }

  bool count(const VRegMaskPair &VMP) const {
    auto It = SetStorage.find(VMP.VReg);
    if (It == SetStorage.end())
      return false;

    return It->second.count(VMP.LaneMask) > 0;
  }

  bool contains(const VRegMaskPair &VMP) const {
    auto It = SetStorage.find(VMP.VReg);
    return It != SetStorage.end() &&
           It->second.find(VMP.LaneMask) != It->second.end();
  }

  void clear() {
    SetStorage.clear();
    LinearStorage.clear();
  }

  size_t size() const { return LinearStorage.size(); }
  bool empty() const { return LinearStorage.empty(); }

  void sort(llvm::function_ref<bool(const VRegMaskPair &, const VRegMaskPair &)>
                Cmp) {
    std::sort(LinearStorage.begin(), LinearStorage.end(), Cmp);
  }

  VRegMaskPair pop_back_val() {
    assert(!LinearStorage.empty() && "Pop from empty set");
    VRegMaskPair VMP = LinearStorage.back();
    LinearStorage.pop_back();

    auto It = SetStorage.find(VMP.VReg);
    assert(It != SetStorage.end() && "Inconsistent SetStorage");
    It->second.erase(VMP.LaneMask);
    if (It->second.empty())
      SetStorage.erase(It);

    return VMP;
  }

  LaneCoverageResult getCoverage(const VRegMaskPair &VMP) const {
    LaneCoverageResult Result(VMP.LaneMask);
    auto It = SetStorage.find(VMP.VReg);
    if (It != SetStorage.end()) {
      MaskSet Masks = It->second;
      for (auto Mask : Masks) {
        Result.Covered |= (Mask & VMP.LaneMask);
      }
      Result.NotCovered = (VMP.LaneMask & ~Result.Covered);
    }
    return Result;
  }

  bool operator==(const VRegMaskPairSet &Other) const {
    if (SetStorage.size() != Other.SetStorage.size())
      return false;

    for (const auto &Entry : SetStorage) {
      auto It = Other.SetStorage.find(Entry.first);
      if (It == Other.SetStorage.end())
        return false;

      if (Entry.second != It->second)
        return false;
    }

    return true;
  }

  template <typename ContainerT>
  VRegMaskPairSet &operator=(const ContainerT &Vec) {
    static_assert(
        std::is_same<typename ContainerT::value_type, VRegMaskPair>::value,
        "Container must hold VRegMaskPair elements");

    clear();
    for (const auto &VMP : Vec)
      insert(VMP);
    return *this;
  }

  // Set operations based on subregister coverage logic

  /// Adds all elements from Other whose (VReg, LaneMask) overlap with none
  /// in *this.
  void set_union(const VRegMaskPairSet &Other) {
    for (const auto &VMP : Other)
      insert(VMP);
  }

  /// Keeps only those elements in *this that are at least partially covered
  /// by Other.
  void set_intersect(const VRegMaskPairSet &Other) {
    std::vector<VRegMaskPair> ToInsert;
    remove_if([&](const VRegMaskPair &VMP) {
      LaneCoverageResult Cov = Other.getCoverage(VMP);
      if (Cov.isFullyUncovered())
        return true;

      if (!Cov.isFullyCovered()) {
        ToInsert.push_back({VMP.VReg, Cov.getCovered()});
        return true; // remove current, will reinsert trimmed version
      }

      return false; // keep as-is
    });

    insert(ToInsert.begin(), ToInsert.end());
  }

  /// Removes elements from *this that are at least partially covered by
  /// Other.
  void set_subtract(const VRegMaskPairSet &Other) {
    std::vector<VRegMaskPair> ToInsert;
    remove_if([&](const VRegMaskPair &VMP) {
      LaneCoverageResult Cov = Other.getCoverage(VMP);
      if (Cov.isFullyCovered())
        return true;

      if (!Cov.isFullyUncovered()) {
        ToInsert.push_back({VMP.VReg, Cov.getNotCovered()});
        return true; // remove and reinsert uncovered part
      }

      return false;
    });

    insert(ToInsert.begin(), ToInsert.end());
  }

  /// Returns the union (join) of this set and Other under coverage logic.
  VRegMaskPairSet set_join(const VRegMaskPairSet &Other) const {
    VRegMaskPairSet Result = *this;
    Result.set_union(Other);
    return Result;
  }

  /// Returns the intersection of this set and Other based on partial
  /// overlap.
  VRegMaskPairSet set_intersection(const VRegMaskPairSet &Other) const {
    VRegMaskPairSet Result;
    for (const auto &VMP : *this) {
      LaneCoverageResult Cov = Other.getCoverage(VMP);
      if (!Cov.isFullyUncovered()) {
        Result.insert({VMP.VReg, Cov.getCovered()});
      }
    }
    return Result;
  }

  /// Returns all elements of *this that do not overlap with anything in
  /// Other.
  VRegMaskPairSet set_difference(const VRegMaskPairSet &Other) const {
    VRegMaskPairSet Result;
    for (const auto &VMP : *this) {
      LaneCoverageResult Cov = Other.getCoverage(VMP);
      if (!Cov.isFullyCovered()) {
        Result.insert({VMP.VReg, Cov.getNotCovered()});
      }
    }
    return Result;
  }

  // Debug
  void dump() const {
    dbgs() << "=== VRegMaskPairSet Dump ===\n";

    dbgs() << "SetStorage:\n";
    for (const auto &Entry : SetStorage) {
      dbgs() << "  VReg: " << printReg(Entry.first) << " => { ";
      for (const auto &Mask : Entry.second) {
        dbgs() << PrintLaneMask(Mask) << " ";
      }
      dbgs() << "}\n";
    }

    dbgs() << "LinearStorage (insertion order):\n";
    for (const auto &VMP : LinearStorage) {
      dbgs() << "  (" << printReg(VMP.getVReg()) << ", "
             << PrintLaneMask(VMP.getLaneMask()) << ")\n";
    }

    dbgs() << "=============================\n";
  }

  // Iterators
  using iterator = LinearStorageT::const_iterator;
  iterator begin() const { return LinearStorage.begin(); }
  iterator end() const { return LinearStorage.end(); }
};

namespace llvm {
template <> struct DenseMapInfo<VRegMaskPair> {
  static inline VRegMaskPair getEmptyKey() {
    return {Register(DenseMapInfo<unsigned>::getEmptyKey()),
            LaneBitmask(0xFFFFFFFFFFFFFFFFULL)};
  }

  static inline VRegMaskPair getTombstoneKey() {
    return {Register(DenseMapInfo<unsigned>::getTombstoneKey()),
            LaneBitmask(0xFFFFFFFFFFFFFFFEULL)};
  }

  static unsigned getHashValue(const VRegMaskPair &P) {
    return DenseMapInfo<unsigned>::getHashValue(P.getVReg().id()) ^
           DenseMapInfo<uint64_t>::getHashValue(P.getLaneMask().getAsInteger());
  }

  static bool isEqual(const VRegMaskPair &LHS, const VRegMaskPair &RHS) {
    return DenseMapInfo<unsigned>::isEqual(LHS.getVReg().id(),
                                           RHS.getVReg().id()) &&
           DenseMapInfo<uint64_t>::isEqual(LHS.getLaneMask().getAsInteger(),
                                           RHS.getLaneMask().getAsInteger());
  }
};

} // namespace llvm
#endif // LLVM_LIB_TARGET_VREGMASKPAIR_H
