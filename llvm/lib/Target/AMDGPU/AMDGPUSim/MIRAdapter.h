//===- AMDGPUSim/MIRAdapter.h - MachineInstr Adapter -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Declares the SimInstInfo adapter that exposes MachineInstr properties
/// without coupling the simulator core to LLVM code generation types.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_MIRADAPTER_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_MIRADAPTER_H

#include "SimInstInfo.h"

namespace llvm {

class MachineInstr;
class SIInstrInfo;
class SIRegisterInfo;

namespace AMDGPUSim {

/// Provides MachineInstr property queries for the representation independent
/// simulator.
class MachineInstrInfo final : public SimInstInfo {
  const SIInstrInfo &TII;
  const SIRegisterInfo &TRI;

public:
  /// Construct an adapter that retains references to \p TII and \p TRI.
  MachineInstrInfo(const SIInstrInfo &TII, const SIRegisterInfo &TRI);

  /// Create a SimInst that references \p MI.
  SimInst createSimInst(const MachineInstr &MI) const;

  /// Return the MachineInstr repeat rate, treating VOPD as one paired issue.
  unsigned getRepeatRate(const SimInst &SI) const override;

  /// Return whether \p SI is a VALU with a repeat rate greater than one.
  bool isLOLVALU(const SimInst &SI) const override;

  /// Derive resource occupancy from repeat rate, scheduling throughput, or
  /// class defaults.
  unsigned getResourceCycles(const SimInst &SI) const override;

  /// Return the immediate operand of S_DELAY_ALU, or zero otherwise.
  unsigned getDelayAluImm(const SimInst &SI) const override;

  /// Decode modeled MachineInstr wait opcodes into counter requirements.
  WaitRequirements getWaitInfo(const SimInst &SI) const override;

  /// Decode the VaVdst target from S_WAITCNT_DEPCTR, or return 15 otherwise.
  unsigned getVaVdstTarget(const SimInst &SI) const override;

  /// Extract normalized WMMA coexecution properties through SIInstrInfo.
  WMMAProperties getWMMAProperties(const SimInst &SI) const override;

  /// Return whether the MachineInstr has an explicit physical SGPR source.
  bool hasSGPROperands(const SimInst &SI) const override;

  /// Return the encoded MachineInstr size reported by SIInstrInfo.
  unsigned getInstBytes(const SimInst &SI) const override;

private:
  /// Return the simulator instruction class for \p MI.
  InstClass classifyInst(const MachineInstr &MI) const;

  /// Return the modeled latency for \p MI classified as \p IC.
  unsigned getLatency(const MachineInstr &MI, InstClass IC) const;
};

} // namespace AMDGPUSim
} // namespace llvm

#endif
