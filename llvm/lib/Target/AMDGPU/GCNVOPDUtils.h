//===- GCNVOPDUtils.h - GCN VOPD Utils  ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains the AMDGPU DAG scheduling
/// mutation to pair VOPD instructions back to back. It also contains
//  subroutines useful in the creation of VOPD instructions
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_VOPDUTILS_H
#define LLVM_LIB_TARGET_AMDGPU_VOPDUTILS_H

#include "llvm/CodeGen/MachineScheduler.h"
#include <optional>

namespace llvm {

class MachineInstr;
class SIInstrInfo;

bool checkVOPDRegConstraints(const SIInstrInfo &TII,
                             const MachineInstr &FirstMI,
                             const MachineInstr &SecondMI, bool IsVOPD3,
                             bool AllowSameVGPR);

/// Describes a matched VOPD pair: which instruction is the X component and
/// which is the Y component, and whether this is a VOPD3 encoding.
struct VOPDMatchInfo {
  MachineInstr *MIX;
  MachineInstr *MIY;
  bool IsVOPD3;
};

/// Check whether FirstMI and SecondMI can be
/// combined into a VOPD instruction.  Returns the match info (X/Y assignment
/// and encoding variant) on success, or std::nullopt if they cannot be paired.
std::optional<VOPDMatchInfo> tryMatchVOPDPair(const SIInstrInfo &TII,
                                              MachineInstr &FirstMI,
                                              MachineInstr &SecondMI);

std::unique_ptr<ScheduleDAGMutation> createVOPDPairingMutation();

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_VOPDUTILS_H
