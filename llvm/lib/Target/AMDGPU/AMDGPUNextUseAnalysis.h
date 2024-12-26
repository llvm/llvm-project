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

#include <limits>

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

template<>
struct DenseMapInfo<VRegMaskPair> {
  static inline VRegMaskPair getEmptyKey() {
    return {Register(DenseMapInfo<unsigned>::getEmptyKey()),
            LaneBitmask(0xFFFFFFFFFFFFFFFFULL)};
  }

  static inline VRegMaskPair getTombstoneKey() {
    return { Register(DenseMapInfo<unsigned>::getTombstoneKey()),
                    LaneBitmask(0xFFFFFFFFFFFFFFFEULL) };
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

class NextUseResult {
  friend class AMDGPUNextUseAnalysisWrapper;
  SlotIndexes *Indexes;
  const MachineRegisterInfo *MRI;
  const SIRegisterInfo *TRI;
  MachineLoopInfo *LI;

  TimerGroup *TG;
  Timer *T1;
  Timer *T2;

  using VRegDistances = DenseMap<VRegMaskPair, unsigned>;
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
  DenseMap<unsigned, SetVector<Register>> UsedInBlock;
  DenseMap<int, int> EdgeWeigths;
  const uint16_t Infinity = std::numeric_limits<unsigned short>::max();
  void init(const MachineFunction &MF);
  void analyze(const MachineFunction &MF);
  bool diff(const VRegDistances &LHS, const VRegDistances &RHS) {
    for (auto P : LHS) {
      if (!RHS.contains(P.getFirst()) ||
          RHS.lookup(P.getFirst()) != P.getSecond())
        return true;
    }
    for (auto P : RHS) {
      if (!LHS.contains(P.getFirst()))
        return true;
    }
    return false;
  }

  void printVregDistances(const VRegDistances &D,
                          raw_ostream &O = dbgs()) const {
    O << "\n";
    for (auto P : D) {
      SmallVector<unsigned> Idxs;
      const TargetRegisterClass *RC =
          TRI->getRegClassForReg(*MRI, P.first.VReg);
      bool HasSubReg =
          TRI->getCoveringSubRegIndexes(*MRI, RC, P.first.LaneMask, Idxs);
      O << "Vreg: ";
      if (HasSubReg)
        for (auto i : Idxs)
          O << printReg(P.first.VReg, TRI, i, MRI) << "[ " << P.second << "]\n";
      else
        O << printReg(P.first.VReg) << "[ " << P.second << "]\n";
    }
  }

  void printVregDistancesD(const VRegDistances &D) const {
    dbgs() << "\n";
    for (auto P : D) {
      SmallVector<unsigned> Idxs;
      const TargetRegisterClass *RC =
          TRI->getRegClassForReg(*MRI, P.first.VReg);
      bool HasSubReg =
          TRI->getCoveringSubRegIndexes(*MRI, RC, P.first.LaneMask, Idxs);
      dbgs() << "Vreg: ";
      if (HasSubReg)
        for (auto i : Idxs)
          dbgs() << printReg(P.first.VReg, TRI, i, MRI) << "[ " << P.second
                 << "]\n";
      else
        dbgs() << printReg(P.first.VReg) << "[ " << P.second << "]\n";
    }
  }

  // void dump(raw_ostream &O = dbgs()) const {
  //   for (auto P : NextUseMap) {
  //     O << "\nMBB_" << P.first << "\n";
  //     printVregDistances(P.second, O);
  //   }
  // }

  VRegDistances &mergeDistances(VRegDistances &LHS, const VRegDistances &RHS,
                                unsigned Weight = 0) {
    for (auto Pair : LHS) {
      VRegMaskPair VRegMP = Pair.getFirst();
      if (RHS.contains(VRegMP)) {
        LHS[VRegMP] = std::min(Pair.getSecond(), RHS.lookup(VRegMP) + Weight);
      }
    }
    for (auto Pair : RHS) {
      if (LHS.contains(Pair.getFirst()))
        continue;
      LHS[Pair.getFirst()] = Pair.getSecond() + Weight;
    }
    return LHS;
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

  bool isDead(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
              const VRegMaskPair VMP) {
    if (!VMP.VReg.isVirtual())
      report_fatal_error("Only virtual registers allowed!\n", true);
    return I == MBB.end() ? getNextUseDistance(MBB, VMP) == Infinity
                          : getNextUseDistance(I, VMP) == Infinity;
  }

  SetVector<Register> usedInBlock(MachineBasicBlock &MBB) {
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
