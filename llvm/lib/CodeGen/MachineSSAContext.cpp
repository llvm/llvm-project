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

MachineBasicBlock *MachineSSAContext::getEntryBlock(MachineFunction &F) {
  return &F.front();
}

void MachineSSAContext::setFunction(MachineFunction &Fn) {
  MF = &Fn;
  RegInfo = &MF->getRegInfo();
}

Printable MachineSSAContext::print(MachineBasicBlock *Block) const {
  return Printable([Block](raw_ostream &Out) { Block->printName(Out); });
}

Printable MachineSSAContext::print(MachineInstr *I) const {
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
