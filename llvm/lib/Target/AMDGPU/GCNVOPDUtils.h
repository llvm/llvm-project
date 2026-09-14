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

/// Describes a matched VOPD pair.
struct VOPDMatchInfo {
  /// The component instructions in program order.
  MachineInstr *InOrder[2];
  /// Which entry in \p InOrder is the X component.
  unsigned XIdx;
  bool IsVOPD3;

  MachineInstr *getMIX() const { return InOrder[XIdx]; }
  MachineInstr *getMIY() const { return InOrder[1 - XIdx]; }
};

/// Check whether \p FirstMI and \p SecondMI, which are next to each other in
/// program order, can be combined into a VOPD instruction. Returns the match
/// info (program order, X/Y assignment, and encoding variant) on success, or
/// std::nullopt if they cannot be paired.
std::optional<VOPDMatchInfo> tryMatchVOPDPair(const SIInstrInfo &TII,
                                              MachineInstr &FirstMI,
                                              MachineInstr &SecondMI);

std::unique_ptr<ScheduleDAGMutation> createVOPDPairingMutation();

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_VOPDUTILS_H
