//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements utility functions for the PISA backend.
///
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
  MachineInstr *DefMI = MRI.getVRegDef(Reg);
  unsigned Opc = DefMI->getOpcode();
  Register DefReg = Reg;
  while (Opc == TargetOpcode::G_BITCAST ||
         isPreISelGenericOptimizationHint(Opc)) {
    Register SrcReg = DefMI->getOperand(1).getReg();
    LLT SrcTy = MRI.getType(SrcReg);
    if (!SrcTy.isValid())
      break;
    LLT DstTy = MRI.getType(DefMI->getOperand(0).getReg());
    if (NoVectors && (DstTy.isVector() || SrcTy.isVector()))
      break;
    DefReg = SrcReg;
    DefMI = MRI.getVRegDef(DefReg);
    Opc = DefMI->getOpcode();
  }
  int RegIdx = DefMI->findRegisterDefOperandIdx(DefReg, /*TRI=*/nullptr);
  assert(RegIdx >= 0 && "defining register operand not found");
  return std::make_tuple(DefMI, static_cast<unsigned>(RegIdx));
}

} // namespace llvm::PISA
