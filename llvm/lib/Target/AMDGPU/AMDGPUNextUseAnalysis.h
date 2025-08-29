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
#include "AMDGPUSSARAUtils.h"
#include "VRegMaskPair.h"

#include <algorithm>
#include <limits>
#include <set>

using namespace llvm;

// namespace {


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
        if (LHS.first ==
            RHS.first) // Same LaneBitmask → prefer furthest distance
          return LHS.second > RHS.second;
        return LHS.first.getAsInteger() <
               RHS.first.getAsInteger(); // Otherwise sort by LaneBitmask so
                                         // that smaller Mask first
      }
    };

    using SortedRecords = std::set<Record, CompareByDist>;

    DenseMap<unsigned, SortedRecords> NextUseMap;

  public:
    auto begin() { return NextUseMap.begin(); }
    auto end() { return NextUseMap.end(); }

    auto begin() const { return NextUseMap.begin(); }
    auto end() const { return NextUseMap.end(); }

    size_t size() const { return NextUseMap.size(); }
    std::pair<bool, SortedRecords> get(unsigned Key) const {
      if (NextUseMap.contains(Key))
        return {true, NextUseMap.find(Key)->second};
      return {false, SortedRecords()};
    }

    SortedRecords &operator[](unsigned Key) { return NextUseMap[Key]; }

    SmallVector<unsigned> keys() {
      SmallVector<unsigned> Keys;
      for (auto P : NextUseMap)
        Keys.push_back(P.first);
      return Keys;
    }

    bool contains(unsigned Key) {
      return NextUseMap.contains(Key);
    }

    bool insert(VRegMaskPair VMP, unsigned Dist) {
      Record R(VMP.getLaneMask(), Dist);
      if (NextUseMap.contains(VMP.getVReg())) {
        SortedRecords &Dists = NextUseMap[VMP.getVReg()];

        if (!Dists.contains(R)) {
          for (auto D : Dists) {
            if (D.first == R.first) {
              if (D.second > R.second) {
                // Change to record with less distance
                Dists.erase(D);
                return Dists.insert(R).second;
              } else {
                return false;
              }
            }
          }
          // add new record
          return Dists.insert(R).second;
        } else {
          // record already exists!
          return false;
        }
      } else
        return NextUseMap[VMP.getVReg()].insert(R).second;
    }

    void clear(VRegMaskPair VMP) {
      if (NextUseMap.contains(VMP.getVReg())) {
        auto &Dists = NextUseMap[VMP.getVReg()];
        std::erase_if(Dists,
                  [&](Record R) { return (R.first &= ~VMP.getLaneMask()).none(); });
        if (Dists.empty())
          NextUseMap.erase(VMP.getVReg());
      }
    }

    bool operator == (const VRegDistances Other) const {
      
      if (Other.size() != size())
        return false;

      for (auto P : NextUseMap) {
        unsigned Key = P.getFirst();
        
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

    bool operator!=(const VRegDistances &Other) const {
      return !operator==(Other);
    }

    void merge(const VRegDistances &Other, unsigned Weight = 0) {
      for (const auto &P : Other) {
        unsigned Key = P.getFirst();
        const auto &OtherDists = P.getSecond();
        auto &MineDists = NextUseMap[Key]; // creates empty if not present

        for (const auto &D : OtherDists) {
          Record Adjusted = {D.first, D.second + Weight};

          // Try to find existing record with the same LaneBitmask
          auto It =
              std::find_if(MineDists.begin(), MineDists.end(),
                           [&](const Record &R) { return R.first == D.first; });

          if (It == MineDists.end()) {
            // No record → insert
            MineDists.insert(Adjusted);
          } else if (It->second > Adjusted.second) {
            // Furthest wins (adjusted is more distant) → replace
            MineDists.erase(It);
            MineDists.insert(Adjusted);
          }
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

  DenseMap<unsigned, NextUseInfo> NextUseMap;

public:
  

private:
  DenseMap<unsigned, SetVector<VRegMaskPair>> UsedInBlock;
  DenseMap<int, int> LoopExits;
  const uint16_t Infinity = std::numeric_limits<unsigned short>::max();
  void init(const MachineFunction &MF);
  void analyze(const MachineFunction &MF);
  LLVM_ATTRIBUTE_NOINLINE void
  printSortedRecords(VRegDistances::SortedRecords Records, unsigned VReg,
                     raw_ostream &O = dbgs()) const {
    for (auto X : Records) {
      O << "Vreg: ";
      LaneBitmask FullMask = MRI->getMaxLaneMaskForVReg(VReg);
      if (X.first != FullMask) {
        unsigned SubRegIdx = getSubRegIndexForLaneMask(X.first, TRI);
        O << printReg(VReg, TRI, SubRegIdx, MRI) << "[ " << X.second << "]\n";
      } else
        O << printReg(VReg) << "[ " << X.second << "]\n";
    }
  }

  LLVM_ATTRIBUTE_NOINLINE
  void printVregDistances(const VRegDistances &D,
                          raw_ostream &O = dbgs()) const {
    O << "\n";
    for (auto P : D) {
      printSortedRecords(P.second, P.first);
    }
  }

  void clear() {
    NextUseMap.clear();
    LoopExits.clear();
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

  SmallVector<VRegMaskPair>
  getSortedSubregUses(const MachineBasicBlock::iterator I,
                      const VRegMaskPair VMP);

  SmallVector<VRegMaskPair>
  getSortedSubregUses(const MachineBasicBlock &MBB,
                      const VRegMaskPair VMP);

  bool isDead(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
              const VRegMaskPair VMP) {
    if (!VMP.getVReg().isVirtual())
      report_fatal_error("Only virtual registers allowed!\n", true);
    // FIXME: We use the same Infinity value to indicate both invalid distance
    // and too long for out of block values. It is okay if the use out of block
    // is at least one instruction further then the end of loop exit. In this
    // case we have a distance Infinity + 1 and hence register is not considered
    // dead. What if the register is defined by the last instruction in the loop
    // exit block and out of loop use is in PHI? By design the dist of all PHIs
    // from the beginning of block are ZERO and hence the distance of
    // out-of-the-loop use will be exactly Infinity So, the register will be
    // mistakenly considered DEAD! On another hand, any predecessor of the block
    // containing PHI must have a branch as the last instruction. In this case
    // the current design works.
    return I == MBB.end() ? getNextUseDistance(MBB, VMP) == Infinity
                          : getNextUseDistance(I, VMP) == Infinity;
  }

  SetVector<VRegMaskPair>& usedInBlock(MachineBasicBlock &MBB) {
    return UsedInBlock[MBB.getNumber()];
  }

  void dumpUsedInBlock();

  /// Dump complete next-use analysis results for testing
  void dumpAllNextUseDistances(const MachineFunction &MF);
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
