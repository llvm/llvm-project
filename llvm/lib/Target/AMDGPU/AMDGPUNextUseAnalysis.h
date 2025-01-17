//===- AMDGPUNextUseAnalysis.h ----------------------------------------*- C++-
//*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_NEXT_USE_ANALYSIS_H
#define LLVM_LIB_TARGET_AMDGPU_NEXT_USE_ANALYSIS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"

#include "SIRegisterInfo.h"
#include "GCNSubtarget.h"

#include <algorithm>
#include <limits>
#include <set>

using namespace llvm;

// namespace {

struct VRegMaskPair {
public:
  Register VReg;
  LaneBitmask LaneMask;

  VRegMaskPair(Register VReg, LaneBitmask LaneMask)
      : VReg(VReg), LaneMask(LaneMask) {}

  VRegMaskPair(const MachineOperand MO, const TargetRegisterInfo &TRI) {
    assert(MO.isReg() && "Not a register operand!");
    Register R = MO.getReg();
    assert(R.isVirtual() && "Not a virtual register!");
    VReg = R;
    LaneMask = LaneBitmask::getAll();
    unsigned subRegIndex = MO.getSubReg();
    if (subRegIndex) {
      LaneMask = TRI.getSubRegIndexLaneMask(subRegIndex);
    }
  }

  bool operator==(const VRegMaskPair &other) const {
    return VReg == other.VReg && LaneMask == other.LaneMask;
  }
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
    return DenseMapInfo<unsigned>::getHashValue(P.VReg.id()) ^
           DenseMapInfo<uint64_t>::getHashValue(P.LaneMask.getAsInteger());
  }

  static bool isEqual(const VRegMaskPair &LHS, const VRegMaskPair &RHS) {
    return DenseMapInfo<unsigned>::isEqual(LHS.VReg.id(), RHS.VReg.id()) &&
           DenseMapInfo<uint64_t>::isEqual(LHS.LaneMask.getAsInteger(),
                                           RHS.LaneMask.getAsInteger());
  }
};
} // namespace llvm
class NextUseResult {
  friend class AMDGPUNextUseAnalysisWrapper;
  SlotIndexes *Indexes;
  const MachineRegisterInfo *MRI;
  const SIRegisterInfo *TRI;
  MachineLoopInfo *LI;

  TimerGroup *TG;
  Timer *T1;
  Timer *T2;

  class VRegDistances {

    using Record = std::pair<LaneBitmask, unsigned>;
    struct CompareByDist {
      bool operator()(const Record &LHS, const Record &RHS) const {
        return LHS.second < RHS.second;
      };
    };

public:
    using SortedRecords = std::set<Record, CompareByDist>;
  private:
    DenseMap<unsigned, SortedRecords> NextUseMap;

  public:
    auto begin() { return NextUseMap.begin(); }
    auto end() { return NextUseMap.end(); }

    auto begin() const { return NextUseMap.begin(); }
    auto end() const { return NextUseMap.end(); }

    size_t size() { return NextUseMap.size(); }
    std::pair<bool, SortedRecords> get(unsigned Key) {
      if (NextUseMap.contains(Key))
        return {true, NextUseMap.find(Key)->second};
      return {false, SortedRecords()};
    }

    SortedRecords operator[] (unsigned Key) {
      return NextUseMap[Key];
    }

    SmallVector<unsigned> keys() {
      SmallVector<unsigned> Keys;
      for (auto P : NextUseMap)
        Keys.push_back(P.first);
      return std::move(Keys);
    }

    bool contains(unsigned Key) {
      return NextUseMap.contains(Key);
    }

    bool insert(VRegMaskPair VMP, unsigned Dist) {
      SortedRecords &Dists = NextUseMap[VMP.VReg];
      return Dists.insert({VMP.LaneMask, Dist}).second;
    }

    void clear(VRegMaskPair VMP) {
      if (NextUseMap.contains(VMP.VReg)) {
        auto &Dists = NextUseMap[VMP.VReg];
        std::erase_if(Dists,
                  [&](Record R) { return (R.first &= ~VMP.LaneMask).none(); });
      }
    }

    bool operator == (VRegDistances Other) {
      
      if (Other.size() != size())
        return false;

      for (auto P : NextUseMap) {

        std::pair<bool, SortedRecords> OtherDists = Other.get(P.getFirst());
        if (!OtherDists.first)
          return false;
        SortedRecords &Dists = P.getSecond();

        if (Dists.size() != OtherDists.second.size())
          return false;

        for (auto R : OtherDists.second) {
          SortedRecords::iterator I = Dists.find(R);
          if (I == Dists.end())
            return false;
          if (R.second != I->second)
            return false;
        }
      }

      return true;
    }

    bool operator != (VRegDistances Other) {
      return !operator == (Other);
    }

    void merge(VRegDistances Other, unsigned Weight = 0) {
      for (auto P : Other) {
        unsigned Key = P.getFirst();
        auto Dists = P.getSecond();
        if (NextUseMap.contains(Key)) {
          auto &MineDists = NextUseMap[Key];
          // Merge it!
          for (auto D : Dists) {
            auto It = MineDists.find(D);
            if (It == MineDists.end()) {
              // Not found! We have a subreg use to merge in.
              for (auto D1 : MineDists) {
                if (D1.first == D.first && D1.second > D.second + Weight) {
                  // We have a closer use of the same reg and mask.
                  // Erase and insert new to keep it properly sorted.
                  MineDists.erase(D1);
                  MineDists.insert({D.first, D.second + Weight});
                  break;
                }
              }
              // Just add a new one.
              MineDists.insert(*It);
            }
          }
        } else {
          // Just add it!
          if (Weight)
            for (auto D : Dists)
              D.second += Weight;
          NextUseMap[Key] = Dists;
        }
      }
    }
  };
  class NextUseInfo {
    // FIXME: need to elaborate proper class interface!
    public:
    VRegDistances Bottom;
    DenseMap<const MachineInstr *, VRegDistances> InstrDist;
  };

  // VRegMaskPair getFromOperand(const MachineOperand &MO) {
  //   assert(MO.isReg() && "Not a register operand!");
  //   Register R = MO.getReg();
  //   assert(R.isVirtual() && "Not a virtual register!");
  //   LaneBitmask Mask = LaneBitmask::getAll();
  //   unsigned subRegIndex = MO.getSubReg();
  //   if (subRegIndex) {
  //     Mask = TRI->getSubRegIndexLaneMask(subRegIndex);
  //   }
  //   return {R, Mask};
  // }

  DenseMap<unsigned, NextUseInfo> NextUseMap;

public:
  

private:
  DenseMap<unsigned, SetVector<VRegMaskPair>> UsedInBlock;
  DenseMap<int, int> EdgeWeigths;
  const uint16_t Infinity = std::numeric_limits<unsigned short>::max();
  void init(const MachineFunction &MF);
  void analyze(const MachineFunction &MF);

  void printVregDistances(const VRegDistances &D,
                          raw_ostream &O = dbgs()) const {
    O << "\n";
    for (auto P : D) {
      SmallVector<unsigned> Idxs;
      const TargetRegisterClass *RC =
          TRI->getRegClassForReg(*MRI, P.first);
      for (auto X : P.second) {
        bool HasSubReg =
            TRI->getCoveringSubRegIndexes(*MRI, RC, X.first, Idxs);
        O << "Vreg: ";
        if (HasSubReg)
          for (auto i : Idxs)
            O << printReg(P.first, TRI, i, MRI) << "[ " << X.second
              << "]\n";
        else
          O << printReg(P.first) << "[ " << X.second << "]\n";
      }
    }
  }

  void clear() {
    NextUseMap.clear();
    EdgeWeigths.clear();
  }

public:
  NextUseResult() = default;
  NextUseResult(const MachineFunction &MF, SlotIndexes &SI, MachineLoopInfo &LI)
      : Indexes(&SI), MRI(&MF.getRegInfo()), LI(&LI) {
    init(MF);
    analyze(MF);
  }
  ~NextUseResult() { clear(); }

  // void print(raw_ostream &O) const { dump(O); }

  unsigned getNextUseDistance(const MachineBasicBlock &MBB,
                              const VRegMaskPair VMP);
  unsigned getNextUseDistance(const MachineBasicBlock::iterator I,
                              const VRegMaskPair VMP);
  void getFromSortedRecords(const VRegDistances::SortedRecords Dists,
                            LaneBitmask Mask, unsigned &D);

  bool isDead(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
              const VRegMaskPair VMP) {
    if (!VMP.VReg.isVirtual())
      report_fatal_error("Only virtual registers allowed!\n", true);
    return I == MBB.end() ? getNextUseDistance(MBB, VMP) == Infinity
                          : getNextUseDistance(I, VMP) == Infinity;
  }

  SetVector<VRegMaskPair> usedInBlock(MachineBasicBlock &MBB) {
    return std::move(UsedInBlock[MBB.getNumber()]);
  }
};

class AMDGPUNextUseAnalysis : public AnalysisInfoMixin<AMDGPUNextUseAnalysis> {
  friend AnalysisInfoMixin<AMDGPUNextUseAnalysis>;
  static AnalysisKey Key;

public:
  using Result = NextUseResult;
  Result run(MachineFunction &MF, MachineFunctionAnalysisManager &MFAM);
};

class AMDGPUNextUseAnalysisWrapper : public MachineFunctionPass {
  NextUseResult NU;

public:
  static char ID;

  AMDGPUNextUseAnalysisWrapper();

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  /// Pass entry point; Calculates LiveIntervals.
  bool runOnMachineFunction(MachineFunction &) override;
  void releaseMemory() override { NU.clear(); }

  // /// Implement the dump method.
  // void print(raw_ostream &O, const Module * = nullptr) const override {
  //   NU.print(O);
  // }

  NextUseResult &getNU() { return NU; }
};

//}

#endif // LLVM_LIB_TARGET_AMDGPU_NEXT_USE_ANALYSIS_H
