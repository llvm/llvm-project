//===- llvm/CodeGen/DeadMachineInstructionElim.h ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_DEADMACHINEINSTRUCTIONELIM_H
#define LLVM_CODEGEN_DEADMACHINEINSTRUCTIONELIM_H

#include "llvm/CodeGen/LiveRegUnits.h"
#include "llvm/CodeGen/MachinePassManager.h"

namespace llvm {

class DeadMachineInstructionInfo {
private:
  const MachineRegisterInfo *MRI = nullptr;

public:
  /// Check whether \p MI is dead. If \p LivePhysRegs is provided, it is assumed
  /// to be at the position of MI and will be used to check the Liveness of
  /// physical register defs. If \p LivePhysRegs is not provided, this will
  /// pessimistically assume any PhysReg def is live.
  bool isDead(const MachineInstr *MI,
              LiveRegUnits *LivePhysRegs = nullptr) const;

  /// Do a function walk over \p MF and delete any dead MachineInstrs. Uses
  /// LivePhysRegs to track liveness of PhysRegs.
  bool eliminateDeadMI(MachineFunction &MF);

  DeadMachineInstructionInfo(const MachineRegisterInfo *MRI) : MRI(MRI) {}
};

class DeadMachineInstructionElimPass
    : public PassInfoMixin<DeadMachineInstructionElimPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

} // namespace llvm

#endif // LLVM_CODEGEN_DEADMACHINEINSTRUCTIONELIM_H
