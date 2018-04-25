//===- AMDGPUMCUtils.cpp - MachineIR utils for AMDGPU -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUMCUtils.h"

namespace llvm {

namespace AMDGPU {

static bool isSameReg(const MachineOperand &LHS, const MachineOperand &RHS) {
  return LHS.isReg() &&
         RHS.isReg() &&
         LHS.getReg() == RHS.getReg() &&
         LHS.getSubReg() == RHS.getSubReg();
}

Optional<int64_t> foldToImm(const MachineOperand &Op,
                            const MachineRegisterInfo *MRI, const SIInstrInfo *TII) {
  if (Op.isImm()) {
    return Op.getImm();
  }

  // If this is not immediate then it can be copy of immediate value, e.g.:
  // %1 = S_MOV_B32 255;
  if (Op.isReg()) {
    for (const MachineOperand &Def : MRI->def_operands(Op.getReg())) {
      if (!isSameReg(Op, Def))
        continue;

      const MachineInstr *DefInst = Def.getParent();
      if (!TII->isFoldableCopy(*DefInst))
        return None;

      const MachineOperand &Copied = DefInst->getOperand(1);
      if (!Copied.isImm())
        return None;

      return Copied.getImm();
    }
  }

  return None;
}

} // end namespace AMDGPU
} // end namespace llvm
