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

#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include <optional>

namespace llvm {

class MachineInstr;
class SIInstrInfo;
class MCRegisterClass;

/// A 32-bit immediate which the VOPD encoding cannot hold. The pair only
/// becomes legal after the operand is replaced by a scalar register holding
/// \p Imm.
struct VOPDLiteralFixup {
  /// Component holding the immediate, AMDGPU::VOPD::X or AMDGPU::VOPD::Y.
  unsigned CompIdx;
  /// Index of the immediate operand within that component.
  unsigned OpIdx;
  /// Value which has to be placed in a register.
  int32_t Imm;
  /// Scalar registers the VOPD source slot can read. This is the slot class
  /// narrowed to SGPR_32, so every register in it can be used.
  const MCRegisterClass *SlotRC;
};

/// Describes a matched VOPD pair.
struct VOPDMatchInfo {
  /// The component instructions in program order.
  MachineInstr *InOrder[2];
  /// Which entry in \p InOrder is the X component.
  unsigned XIdx;
  bool IsVOPD3;
  /// Immediates which have to be moved into scalar registers before the pair
  /// can be built. They all have the same 32-bit value, so one register serves
  /// the whole pair. Only a VOPD3 pair can need this.
  SmallVector<VOPDLiteralFixup, 2> LiteralFixups;

  MachineInstr *getMIX() const { return InOrder[XIdx]; }
  MachineInstr *getMIY() const { return InOrder[1 - XIdx]; }
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
