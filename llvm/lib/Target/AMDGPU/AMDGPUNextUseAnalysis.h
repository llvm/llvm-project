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

class NextUseResult {
  friend class AMDGPUNextUseAnalysisWrapper;
  SlotIndexes *Indexes;
  const MachineRegisterInfo *MRI;
  const SIRegisterInfo *TRI;
  MachineLoopInfo *LI;

  TimerGroup *TG;
  Timer *T1;
  Timer *T2;

  using VRegDistances = DenseMap<Register, unsigned>;
  class NextUseInfo {
    // FIXME: need to elaborate proper class interface!
    public:
    VRegDistances Bottom;
    DenseMap<const MachineInstr *, VRegDistances> InstrDist;
  };
  
  
  DenseMap<unsigned, NextUseInfo> NextUseMap;

public:
  

private:
  //DenseMap<unsigned, VRegDistances> NextUseMap;
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
      O << "Vreg: " << printReg(P.first) << "[ " << P.second << "]\n";
    }
  }

  void printVregDistancesD(const VRegDistances &D) const {
    dbgs() << "\n";
    for (auto P : D) {
      dbgs() << "Vreg: " << printReg(P.first) << "[ " << P.second << "]\n";
    }
  }

  // void dump(raw_ostream &O = dbgs()) const {
  //   for (auto P : NextUseMap) {
  //     O << "\nMBB_" << P.first << "\n";
  //     printVregDistances(P.second, O);
  //   }
  // }

  // std::optional<std::reference_wrapper<VRegDistances>>
  // getVRegMap(const MachineBasicBlock *MBB) {
  //   if (NextUseMap.contains(MBB->getNumber())) {
  //     return NextUseMap[MBB->getNumber()];
  //   }
  //   return std::nullopt;
  // }

  VRegDistances &mergeDistances(VRegDistances &LHS, const VRegDistances &RHS,
                                unsigned Weight = 0) {
    for (auto Pair : LHS) {
      Register VReg = Pair.getFirst();
      if (RHS.contains(VReg)) {
        LHS[VReg] = std::min(Pair.getSecond(), RHS.lookup(VReg) + Weight);
      }
    }
    for (auto Pair : RHS) {
      if (LHS.contains(Pair.getFirst()))
        continue;
      LHS[Pair.getFirst()] = Pair.getSecond() + Weight;
    }
    return LHS;
  }

  // void setNextUseDistance(const MachineBasicBlock *MBB, Register VReg,
  //                         int Distance) {
  //   auto VMapRef = getVRegMap(MBB);
  //   if (!VMapRef)
  //     VMapRef = NextUseMap[MBB->getNumber()];
  //   VRegDistances &VRegs = VMapRef.value();
  //   VRegs[VReg] = Distance;
  // }

  // unsigned computeNextUseDistance(const MachineBasicBlock &MBB,
  //                                 const SlotIndex I, Register Vreg);

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

  // unsigned getNextUseDistance(const MachineInstr &MI, Register VReg);
  unsigned getNextUseDistance(const MachineBasicBlock &MBB,
                              const Register VReg);
  unsigned getNextUseDistance(const MachineBasicBlock::iterator I,
                              const Register VReg);

  bool isDead(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
              Register R) {
    if (!R.isVirtual())
      report_fatal_error("Only virtual registers allowed!\n", true);
    return I == MBB.end() ? getNextUseDistance(MBB, R) == Infinity
                          : getNextUseDistance(I, R) == Infinity;
  }

  // bool isDead(MachineBasicBlock &MBB, Register R) {
  //   if (!R.isVirtual())
  //     report_fatal_error("Only virtual registers allowed!\n", true);
  //   return getNextUseDistance(MBB, R) == Infinity;
  // }

  // bool isDead(MachineInstr &MI, Register R) {
  //   if (!R.isVirtual())
  //     report_fatal_error("Only virtual registers allowed!\n", true);
  //   return getNextUseDistance(MI, R) == Infinity;
  // }

//   void getSortedForBlockEnd(MachineBasicBlock &MBB,
//                                SetVector<Register> &Regs) {
//     auto SortByDist = [&](const Register LHS, const Register RHS) {
//       return getNextUseDistance(MBB, LHS) < getNextUseDistance(MBB, RHS);
//     };
//     SmallVector<Register> Tmp(Regs.takeVector());
//     sort(Tmp, SortByDist);
//     Regs.insert(Tmp.begin(), Tmp.end());
//   }

//   void getSortedForInstruction(const MachineInstr &MI,
//                                    SetVector<Register> &Regs) {
//     // auto SortByDist = [&](const Register LHS, const Register RHS) {
//     //   unsigned LDist = getNextUseDistance(MI, LHS);
//     //   unsigned RDist = getNextUseDistance(MI, RHS);
//     //   if (LDist == RDist) {
//     //     const TargetRegisterClass *LRC = TRI->getRegClassForReg(*MRI, LHS);
//     //     unsigned LSize = TRI->getRegClassWeight(LRC).RegWeight;
//     //     const TargetRegisterClass *RRC = TRI->getRegClassForReg(*MRI, RHS);
//     //     unsigned RSize = TRI->getRegClassWeight(RRC).RegWeight;
//     //     return LSize < RSize;
//     //   }
//     //   return LDist < RDist;
//     // };
//     auto SortByDist = [&](const Register LHS, const Register RHS) {
//       return getNextUseDistance(MI, LHS) < getNextUseDistance(MI, RHS);
//     };
//     SmallVector<Register> Tmp(Regs.takeVector());
//     sort(Tmp, SortByDist);
//     Regs.insert(Tmp.begin(), Tmp.end());
//   }

//   std::vector<std::pair<Register, unsigned>>
//   getSortedByDistance(const MachineInstr &MI, std::vector<Register> &W) {
//     std::vector<std::pair<Register, unsigned>> Result;
//     auto compareByVal = [](std::pair<Register, unsigned> &LHS,
//                            std::pair<Register, unsigned> &RHS) -> bool {
//       return LHS.second < RHS.second;
//     };

//     for (auto R : W) {
//       dbgs() << printReg(R);
//       Result.push_back(std::make_pair(R, getNextUseDistance(MI, R)));
//     }

//     std::sort(Result.begin(), Result.end(), compareByVal);

//     return std::move(Result);
//   }

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
