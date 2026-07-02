//===------------ MachineStableHash.h - MIR Stable Hashing Utilities ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Stable hashing for MachineInstr and MachineOperand. Useful or getting a
// hash across runs, modules, etc.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINESTABLEHASH_H
#define LLVM_CODEGEN_MACHINESTABLEHASH_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StableHashing.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
class MachineBasicBlock;
class MachineFunction;
class MachineInstr;
class MachineOperand;

struct MachineBasicBlockStableHashInfo {
  const MachineBasicBlock *MBB = nullptr;
  stable_hash Hash = 0;
};

struct MachineFunctionStableHashInfo {
  stable_hash Hash = 0;
  SmallVector<MachineBasicBlockStableHashInfo, 0> Blocks;
};

LLVM_ABI stable_hash stableHashValue(const MachineOperand &MO);
LLVM_ABI stable_hash stableHashValue(const MachineInstr &MI,
                                     bool HashVRegs = false,
                                     bool HashConstantPoolIndices = false,
                                     bool HashMemOperands = false);
LLVM_ABI stable_hash stableHashValue(const MachineBasicBlock &MBB);
LLVM_ABI stable_hash stableHashValue(const MachineFunction &MF);

/// Diagnostic hash for -print-changed; more permissive than stableHashValue().
LLVM_ABI stable_hash stableHashValueForChangePrinter(const MachineInstr &MI);
LLVM_ABI stable_hash
stableHashValueForChangePrinter(const MachineBasicBlock &MBB);
LLVM_ABI stable_hash stableHashValueForChangePrinter(const MachineFunction &MF);
LLVM_ABI MachineFunctionStableHashInfo
stableHashValueWithDetailsForChangePrinter(const MachineFunction &MF);

} // namespace llvm

#endif
