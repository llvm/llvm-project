//===- MachineSSAContext.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines a specialization of the GenericSSAContext<X>
/// template class for Machine IR.
///
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineSSAContext.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

const Register MachineSSAContext::ValueRefNull{};

void MachineSSAContext::setFunction(MachineFunction &Fn) {
  MF = &Fn;
  RegInfo = &MF->getRegInfo();
}

MachineBasicBlock *MachineSSAContext::getEntryBlock(MachineFunction &F) {
  return &F.front();
}

void MachineSSAContext::appendBlockTerms(
    SmallVectorImpl<const MachineInstr *> &terms,
    const MachineBasicBlock &block) {
  for (auto &T : block.terminators())
    terms.push_back(&T);
}

void MachineSSAContext::appendBlockDefs(SmallVectorImpl<Register> &defs,
                                        const MachineBasicBlock &block) {
  for (const MachineInstr &instr : block.instrs()) {
    for (const MachineOperand &op : instr.operands()) {
      if (op.isReg() && op.isDef())
        defs.push_back(op.getReg());
    }
  }
}

/// Get the defining block of a value.
MachineBasicBlock *MachineSSAContext::getDefBlock(Register value) const {
  if (!value)
    return nullptr;
  return RegInfo->getVRegDef(value)->getParent();
}

bool MachineSSAContext::isConstantValuePhi(const MachineInstr &Phi) {
  return Phi.isConstantValuePHI();
}

Printable MachineSSAContext::print(const MachineBasicBlock *Block) const {
  if (!Block)
    return Printable([](raw_ostream &Out) { Out << "<nullptr>"; });
  return Printable([Block](raw_ostream &Out) { Block->printName(Out); });
}

Printable MachineSSAContext::print(const MachineInstr *I) const {
  return Printable([I](raw_ostream &Out) { I->print(Out); });
}

Printable MachineSSAContext::print(Register Value) const {
  auto *MRI = RegInfo;
  return Printable([MRI, Value](raw_ostream &Out) {
    Out << printReg(Value, MRI->getTargetRegisterInfo(), 0, MRI);

    if (Value) {
      // Try to print the definition.
      if (auto *Instr = MRI->getUniqueVRegDef(Value)) {
        Out << ": ";
        Instr->print(Out);
      }
    }
  });
}
