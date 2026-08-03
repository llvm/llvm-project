//===-- PISAUtils.cpp ---- PISA Utility Functions -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PISAUtils.h"
#include "MCTargetDesc/PISABaseInfo.h"
#include "PISA.h"
#include "PISAInstrInfo.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/IntrinsicsPISA.h"

namespace llvm::PISA {
std::tuple<MachineInstr *, unsigned>
getDefIgnoringBitcasts(Register Reg, const MachineRegisterInfo &MRI,
                       bool NoVectors) {
  auto *DefMI = MRI.getVRegDef(Reg);
  unsigned Opc = DefMI->getOpcode();
  Register SrcReg = 0;
  while (Opc == TargetOpcode::G_BITCAST ||
         isPreISelGenericOptimizationHint(Opc)) {
    SrcReg = DefMI->getOperand(1).getReg();
    auto SrcTy = MRI.getType(SrcReg);
    if (!SrcTy.isValid())
      break;
    auto DstTy = MRI.getType(DefMI->getOperand(0).getReg());
    if (NoVectors && (DstTy.isVector() || SrcTy.isVector()))
      break;
    DefMI = MRI.getVRegDef(SrcReg);
    Opc = DefMI->getOpcode();
  }
  unsigned RegIdx = 0;
  if (SrcReg != 0) {
    for (unsigned I = 0; I < DefMI->getNumOperands(); I++) {
      if (DefMI->getOperand(I).isReg() &&
          DefMI->getOperand(I).getReg() == SrcReg) {
        RegIdx = I;
        break;
      }
    }
  }
  return std::make_tuple(DefMI, RegIdx);
}

} // namespace llvm::PISA
