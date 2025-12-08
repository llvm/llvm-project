//===-- AMDGPUNextUseAnalysis.h - Next Use Analysis ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file defines the Next Use Analysis for AMDGPU targets, which computes
/// distances to next uses of virtual registers to guide register allocation
/// and spilling decisions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_NEXT_USE_ANALYSIS_H
#define LLVM_LIB_TARGET_AMDGPU_NEXT_USE_ANALYSIS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"

#include "AMDGPUSSARAUtils.h"
#include "GCNSubtarget.h"
#include "SIRegisterInfo.h"
#include "VRegMaskPair.h"

#include <algorithm>
#include <limits>
#include <set>

using namespace llvm;

// namespace {

// Helper function for rebasing successor distances into current block frame
static inline int64_t rebaseFromSucc(int64_t SuccStored, unsigned SuccEntryOff,
                                     int64_t EdgeWeight /*0 or LoopTag*/) {
  // Move succ-relative value into "current block end" frame.
  return (int64_t)SuccStored + (int64_t)SuccEntryOff + (int64_t)EdgeWeight;
}

class NextUseResult {
  friend class AMDGPUNextUseAnalysisWrapper;
  SlotIndexes *Indexes;
  const MachineRegisterInfo *MRI;
  const SIRegisterInfo *TRI;
  MachineLoopInfo *LI;

  class VRegDistances {

    using Record = std::pair<LaneBitmask, int64_t>;
    struct CompareByDist {
      bool operator()(const Record &LHS, const Record &RHS) const {
        if (LHS.second != RHS.second)     // Different distances
          return LHS.second < RHS.second; // Closest first
        return LHS.first.getAsInteger() <
               RHS.first.getAsInteger(); // Tiebreaker
      }
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

    bool contains(unsigned Key) { return NextUseMap.contains(Key); }

    // Compare two stored distances: returns true if A is closer or equal to B.
    // Handles mixed-sign values correctly:
    // - Negative stored values (finite distances): larger (less negative) =
    // closer
    // - Non-negative stored values (LoopTag distances): smaller = closer
    // - Mixed: negative (finite) is always closer than non-negative
    // (loop-tagged)
    // TODO: Investigate making LoopTag/DeadTag negative for consistent sign
    // convention
    static bool isCloserOrEqual(int64_t A, int64_t B) {
      // Both negative (finite): larger = closer
      if (A < 0 && B < 0)
        return A >= B;
      // Both non-negative (loop-tagged): smaller = closer
      if (A >= 0 && B >= 0)
        return A <= B;
      // Mixed: negative (finite) is always closer than non-negative
      // (loop-tagged)
      return A < 0;
    }

    bool insert(VRegMaskPair VMP, int64_t Dist) {
      Record R(VMP.getLaneMask(), Dist);
      if (NextUseMap.contains(VMP.getVReg())) {
        SortedRecords &Dists = NextUseMap[VMP.getVReg()];

        if (Dists.find(R) == Dists.end()) {
          SmallVector<SortedRecords::iterator, 4> ToErase;

          for (auto It = Dists.begin(); It != Dists.end(); ++It) {
            const Record &D = *It;

            // Check if existing use covers the new use
            if ((R.first & D.first) == R.first) {
              // Existing use covers new use - keep if existing is closer
              if (isCloserOrEqual(D.second, R.second)) {
                // Existing use is closer or equal → reject new use
                return false;
              }
              // Existing use is further → continue (might replace it)
            }

            // Check if new use covers existing use
            if ((D.first & R.first) == D.first) {
              // New use covers existing use - evict if new is closer
              if (isCloserOrEqual(R.second, D.second)) {
                // New use is closer → mark existing for removal
                ToErase.push_back(It);
              } else {
                // New use is further → reject it
                return false;
              }
            }
          }

          // Remove all records that the new use supersedes
          for (auto It : ToErase) {
            Dists.erase(It);
          }

          // Add new record
          return Dists.insert(R).second;
        }
        // Record already exists!
        return false;
      }
      return NextUseMap[VMP.getVReg()].insert(R).second;
    }

    void clear(VRegMaskPair VMP) {
      if (NextUseMap.contains(VMP.getVReg())) {
        auto &Dists = NextUseMap[VMP.getVReg()];
        for (auto It = Dists.begin(); It != Dists.end();) {
          LaneBitmask Masked = It->first & ~VMP.getLaneMask();
          if (Masked.none()) {
            It = Dists.erase(It);
          } else {
            ++It;
          }
        }
        if (Dists.empty())
          NextUseMap.erase(VMP.getVReg());
      }
    }

    bool operator==(const VRegDistances Other) const {

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

    bool operator!=(const VRegDistances &Other) const {
      return !operator==(Other);
    }

    // Adjust 'Other' (which is in successor's frame) into *this* frame,
    // then merge using insert's coverage logic.
    void merge(const VRegDistances &Other, unsigned SuccEntryOff,
               int64_t EdgeWeight = 0) {
      for (const auto &P : Other) {
        unsigned Key = P.getFirst();
        const auto &OtherDists = P.getSecond();

        for (const auto &D : OtherDists) {
          int64_t Rebased = rebaseFromSucc(D.second, SuccEntryOff, EdgeWeight);
          // Use insert's coverage logic for consistent handling
          insert(VRegMaskPair(Register(Key), D.first), Rebased);
        }
      }
    }
  };
  class NextUseInfo {
    // FIXME: need to elaborate proper class interface!
  public:
    VRegDistances Bottom;
    DenseMap<const MachineInstr *, VRegDistances> InstrDist;
    DenseMap<const MachineInstr *, unsigned>
        InstrOffset; // offset at that MI snapshot
  };

  DenseMap<unsigned, NextUseInfo> NextUseMap;
  // Map MBB number to the maximal offset in given by the bottm-up walk
  DenseMap<unsigned, unsigned> EntryOff;

public:
private:
  DenseMap<unsigned, VRegMaskPairSet> UsedInBlock;
  DenseMap<unsigned, unsigned> LoopExits;
  // Signed tag used to mark "outside current loop" in stored values.
  // Must be >> any finite distance you can accumulate in one function.
  static constexpr int64_t LoopTag = (int64_t)1 << 40; // ~1e12 headroom
  static constexpr int64_t DeadTag = (int64_t)1 << 60; // ~1e18, >> LoopTag

  // Sentinel returned by getNextUseDistance() for dead/unused registers.
  static constexpr unsigned DeadDistance = std::numeric_limits<uint16_t>::max();

  void init(const MachineFunction &MF);
  void analyze(const MachineFunction &MF);

  // Core materialization: convert stored relative value + snapshot offset
  // to full materialized distance with bounds checking.
  int64_t materialize(int64_t Stored, unsigned SnapshotOffset) const {
    int64_t Mat64 = Stored + static_cast<int64_t>(SnapshotOffset);
    return (Mat64 <= 0) ? 0 : Mat64;
  }

  // Structure for enhanced distance ranking and printing
  struct PrintDist {
    bool IsInfinity;
    bool IsDead;
    int64_t LoopMultiplier; // How many LoopTags are in the distance
    int64_t Rema;           // remainder after extracting LoopTags

    PrintDist(int64_t Mat64) {
      if (Mat64 >= DeadTag) {
        IsInfinity = false;
        IsDead = true;
        LoopMultiplier = 0;
        Rema = Mat64 - DeadTag;
      } else if (Mat64 >= LoopTag) {
        IsInfinity = true;
        IsDead = false;
        // Extract LoopTag multiples and remainder
        LoopMultiplier = Mat64 / LoopTag;
        Rema = Mat64 % LoopTag;
      } else {
        IsInfinity = false;
        IsDead = false;
        LoopMultiplier = 0;
        Rema = Mat64;
      }
    }
  };

  // Materializer for spiller use: applies three-tier ranking system
  unsigned materializeForRank(int64_t Stored, unsigned SnapshotOffset) const;

  // Materializer for printing: returns PrintDist structure
  PrintDist materializeForPrint(int64_t Stored, unsigned SnapshotOffset) const {
    return PrintDist(materialize(Stored, SnapshotOffset));
  }

  // Print each (VReg, LaneMask) entry with materialized distances.
  // Returns true if at least one record was printed.
  LLVM_DUMP_METHOD
  bool printSortedRecords(const VRegDistances::SortedRecords &Records,
                          unsigned VReg, unsigned SnapshotOffset,
                          int64_t EdgeWeigth = 0, raw_ostream &O = dbgs(),
                          StringRef Indent = "      ") const {
    bool Any = false;
    O << "\n";
    for (const auto &X : Records) {
      const LaneBitmask UseMask = X.first;
      const int64_t Stored = X.second; // stored relative (may be negative)

      // Use enhanced materialization for display that shows three-tier
      // structure
      PrintDist PDist =
          materializeForPrint(Stored + EdgeWeigth, SnapshotOffset);

      O << Indent << "Vreg: ";
      const LaneBitmask FullMask = MRI->getMaxLaneMaskForVReg(VReg);
      if (UseMask != FullMask) {
        const unsigned SubRegIdx = getSubRegIndexForLaneMask(UseMask, TRI);
        O << printReg(VReg, TRI, SubRegIdx, MRI);
      } else {
        O << printReg(VReg, TRI);
      }

      if (PDist.IsDead)
        O << "[ DEAD ]\n";
      else if (PDist.IsInfinity)
        if (PDist.LoopMultiplier == 1)
          O << "[ LoopTag+" << PDist.Rema << " ]\n";
        else if (PDist.LoopMultiplier > 1)
          O << "[ LoopTag*" << PDist.LoopMultiplier << "+" << PDist.Rema
            << " ]\n";
        else
          O << "[ INF+" << PDist.Rema << " ]\n";
      else
        O << "[ " << PDist.Rema << " ]\n";

      Any = true;
    }
    return Any;
  }

  // Iterate VRegs and delegate to printSortedRecords.
  // Returns true if anything was printed.
  LLVM_DUMP_METHOD
  bool printVregDistances(const VRegDistances &D, unsigned SnapshotOffset,
                          int64_t EdgeWeight = 0, raw_ostream &O = dbgs(),
                          StringRef Indent = "      ") const {
    bool Any = false;
    for (const auto &P : D) {
      Any |= printSortedRecords(P.second, P.first, SnapshotOffset, EdgeWeight,
                                O, Indent);
    }
    return Any;
  }

  // Backward-compat shim for block-end printing (offset = 0, default indent).
  LLVM_DUMP_METHOD
  bool printVregDistances(const VRegDistances &D,
                          raw_ostream &O = dbgs()) const {
    return printVregDistances(D, /*SnapshotOffset=*/0, /*EdgeWeight*/ 0, O);
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
  void getFromSortedRecords(const VRegDistances::SortedRecords &Dists,
                            LaneBitmask Mask, unsigned SnapshotOffset,
                            unsigned &D);

  SmallVector<VRegMaskPair>
  getSortedSubregUses(const MachineBasicBlock::iterator I,
                      const VRegMaskPair VMP);

  SmallVector<VRegMaskPair> getSortedSubregUses(const MachineBasicBlock &MBB,
                                                const VRegMaskPair VMP);

  bool isDead(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
              const VRegMaskPair VMP) {
    if (!VMP.getVReg().isVirtual())
      report_fatal_error("Only virtual registers allowed!\n", true);
    return I == MBB.end() ? getNextUseDistance(MBB, VMP) == DeadDistance
                          : getNextUseDistance(I, VMP) == DeadDistance;
  }

  VRegMaskPairSet &usedInBlock(MachineBasicBlock &MBB) {
    return UsedInBlock[MBB.getNumber()];
  }

  LLVM_DUMP_METHOD void dumpUsedInBlock();

  /// Dump complete next-use analysis results for testing
  LLVM_DUMP_METHOD void dumpAllNextUseDistances(const MachineFunction &MF);
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
