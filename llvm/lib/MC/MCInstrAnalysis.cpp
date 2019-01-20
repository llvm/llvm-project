//===- MCInstrAnalysis.cpp - InstrDesc target hooks -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCInstrAnalysis.h"

#include "llvm/ADT/APInt.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include <cstdint>

using namespace llvm;

bool MCInstrAnalysis::clearsSuperRegisters(const MCRegisterInfo &MRI,
                                           const MCInst &Inst,
                                           APInt &Writes) const {
  Writes.clearAllBits();
  return false;
}

bool MCInstrAnalysis::evaluateBranch(const MCInst &Inst, uint64_t Addr,
                                     uint64_t Size, uint64_t &Target) const {
  if (Inst.getNumOperands() == 0 ||
      Info->get(Inst.getOpcode()).OpInfo[0].OperandType != MCOI::OPERAND_PCREL)
    return false;

  int64_t Imm = Inst.getOperand(0).getImm();
  Target = Addr+Size+Imm;
  return true;
}
