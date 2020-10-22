//===- MachineCycleInfo.cpp - Cycle Info for Machine IR ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineCfgTraits.h"

#include "llvm/IR/BasicBlock.h"

using namespace llvm;

void MachineCfgTraits::Printer::printValue(raw_ostream &out,
                                           Register value) const {
  out << printReg(value, m_regInfo->getTargetRegisterInfo(), 0, m_regInfo);

  if (value) {
    out << ": ";

    MachineInstr *instr = m_regInfo->getUniqueVRegDef(value);
    instr->print(out);
  }
}

void MachineCfgTraits::Printer::printBlockName(raw_ostream &out,
                                               MachineBasicBlock *block) const {
  block->printName(out);
}
