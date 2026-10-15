//===- AMDGPUMemoryClauseUtils.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUMEMORYCLAUSESIMPL_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUMEMORYCLAUSESIMPL_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/MC/LaneBitmask.h"

namespace llvm {

class GCNDownwardRPTracker;
class GCNSubtarget;
class SIRegisterInfo;
class MachineRegisterInfo;
class SIMachineFunctionInfo;
class LiveIntervals;

namespace AMDGPU {

class AMDGPUFormMemoryClausesImpl {
  using RegUse = DenseMap<unsigned, std::pair<RegState, LaneBitmask>>;

  bool canBundle(const MachineInstr &MI, const RegUse &Defs,
                 const RegUse &Uses) const;
  bool checkPressure(const MachineInstr &MI, GCNDownwardRPTracker &RPT);
  void collectRegUses(const MachineInstr &MI, RegUse &Defs, RegUse &Uses) const;
  bool processRegUses(const MachineInstr &MI, RegUse &Defs, RegUse &Uses,
                      GCNDownwardRPTracker &RPT);

  const GCNSubtarget *ST;
  const SIRegisterInfo *TRI;
  const MachineRegisterInfo *MRI;
  SIMachineFunctionInfo *MFI;
  LiveIntervals *LIS;

  unsigned LastRecordedOccupancy;
  unsigned MaxVGPRs;
  unsigned MaxSGPRs;

public:
  AMDGPUFormMemoryClausesImpl(LiveIntervals *LS) : LIS(LS) {}
  bool run(MachineFunction &MF);
};

} // namespace AMDGPU

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUMEMORYCLAUSESIMPL_H
