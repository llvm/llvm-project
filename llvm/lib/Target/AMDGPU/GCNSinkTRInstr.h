//===-- GCNSinkTRIntstr.h - GCN Scheduler Strategy -*- C++ -*--------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_GCNSINKTRINSTR_H
#define LLVM_LIB_TARGET_AMDGPU_GCNSINKTRINSTR_H

#include "GCNRegPressure.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include <functional>
#include <map>

namespace llvm {

struct GCNRegPressure;
class GCNSubtarget;
class LiveIntervals;
class MachineBasicBlock;
class MachineFunction;
class MachineRegisterInfo;
class TargetInstrInfo;
class TargetRegisterInfo;

class SinkTrivallyRematInstr {
public:
  SinkTrivallyRematInstr(
      MachineFunction &MF, LiveIntervals *LIS,
      const DenseMap<MachineBasicBlock *, GCNRegPressure> &MBBPressure);

  unsigned collectSinkableRegs(DenseSet<Register> &SelectedRegs,
                               unsigned MinOccupancy,
                               unsigned MaxOccupancy) const;

  void sinkTriviallyRematInstrs(const DenseSet<Register> &Regs) const;

  void forEveryMBBRegIsLiveIn(
      Register Reg, std::function<void(MachineBasicBlock *MBB)> Callback) const;

  MachineInstr *getDefInstr(Register Reg) const;
  MachineInstr *getUserInstr(Register Reg) const;

private:
  MachineFunction &MF;
  const GCNSubtarget &ST;
  const MachineRegisterInfo &MRI;
  const TargetInstrInfo &TII;
  const TargetRegisterInfo &TRI;
  LiveIntervals &LIS;

  mutable DenseMap<MachineBasicBlock *, GCNRegPressure> MBBPressure;
  mutable SmallVector<SlotIndex, 16> MBBFirstInstSlot;
  mutable DenseMap<MachineBasicBlock *, GCNRPTracker::LiveRegSet> LiveOuts;

  void init() const;

  MachineBasicBlock *getSinkSourceMBB(Register Reg) const {
    return getDefInstr(Reg)->getParent();
  }

  MachineBasicBlock *getSinkTargetMBB(Register Reg) const {
    return getUserInstr(Reg)->getParent();
  }

  const GCNRegPressure &getPressure(MachineBasicBlock *MBB) const;

  void cacheLiveOuts(
      const DenseMap<MachineBasicBlock *, SmallVector<Register>> &MBBs) const;

  bool
  fromHighToLowRP(DenseMap<MachineBasicBlock *, SmallVector<Register>> &MBBs,
                  unsigned Occupancy,
                  std::function<bool(MachineBasicBlock *MBB)> Callback) const;

  bool isSinkableReg(Register Reg) const;

  bool selectRegsFromSinkSourceMBB(MachineBasicBlock *MBB,
                                   DenseSet<Register> &SelectedRegs,
                                   const SmallVectorImpl<Register> &Regs,
                                   const GCNRegPressure &TargetRP) const;

  bool selectRegsFromSinkTargetMBB(MachineBasicBlock *MBB,
                                   DenseSet<Register> &SelectedRegs,
                                   const SmallVectorImpl<Register> &Regs,
                                   const GCNRegPressure &TargetRP) const;

  void findExcessiveSinkSourceMBBs(
      const DenseMap<Register, unsigned> &Regs, const GCNRegPressure &TargetRP,
      DenseMap<MachineBasicBlock *, SmallVector<Register>> &SinkSrc) const;

  void findExcessiveSinkTargetMBBs(
      const DenseMap<Register, unsigned> &Regs, const GCNRegPressure &TargetRP,
      DenseMap<MachineBasicBlock *, SmallVector<Register>> &SinkTgt) const;

  void selectLiveThroughRegs(
      DenseSet<Register> &SelectedRegs, const SmallVectorImpl<Register> &Regs,
      const GCNRegPressure &TargetRP,
      std::function<bool(MachineBasicBlock *MBB)> IsMBBProcessed) const;

  Printable printReg(Register Reg) const;
  Printable printMBBHeader(MachineBasicBlock *MBB,
                           const SmallVectorImpl<Register> &Regs,
                           const DenseSet<Register> &SelectedRegs,
                           const GCNRegPressure &TargetRP) const;
};

} // End namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_GCNSINKTRINSTR_H