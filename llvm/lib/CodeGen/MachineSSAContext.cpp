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
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

template <>
void MachineSSAContext::appendBlockDefs(SmallVectorImpl<Register> &defs,
                                        const MachineBasicBlock &block) {
  for (auto &instr : block.instrs()) {
    for (auto &op : instr.all_defs())
      defs.push_back(op.getReg());
  }
}

template <>
void MachineSSAContext::appendBlockTerms(SmallVectorImpl<MachineInstr *> &terms,
                                         MachineBasicBlock &block) {
  for (auto &T : block.terminators())
    terms.push_back(&T);
}

template <>
void MachineSSAContext::appendBlockTerms(
    SmallVectorImpl<const MachineInstr *> &terms,
    const MachineBasicBlock &block) {
  for (auto &T : block.terminators())
    terms.push_back(&T);
}

/// Get the defining block of a value.
template <>
const MachineBasicBlock *MachineSSAContext::getDefBlock(Register value) const {
  if (!value)
    return nullptr;
  return F->getRegInfo().getVRegDef(value)->getParent();
}

static bool isUndef(const MachineInstr &MI) {
  return MI.getOpcode() == TargetOpcode::G_IMPLICIT_DEF ||
         MI.getOpcode() == TargetOpcode::IMPLICIT_DEF;
}

/// MachineInstr equivalent of PHINode::hasConstantOrUndefValue() for G_PHI.
template <>
bool MachineSSAContext::isConstantOrUndefValuePhi(const MachineInstr &Phi) {
  if (!Phi.isPHI())
    return false;

  // In later passes PHI may appear with an undef operand, getVRegDef can fail.
  if (Phi.getOpcode() == TargetOpcode::PHI)
    return Phi.isConstantValuePHI();

  // For G_PHI we do equivalent of PHINode::hasConstantOrUndefValue().
  const MachineRegisterInfo &MRI = Phi.getMF()->getRegInfo();
  Register This = Phi.getOperand(0).getReg();
  Register ConstantValue;
  for (unsigned i = 1, e = Phi.getNumOperands(); i < e; i += 2) {
    Register Incoming = Phi.getOperand(i).getReg();
    if (Incoming != This && !isUndef(*MRI.getVRegDef(Incoming))) {
      if (ConstantValue && ConstantValue != Incoming)
        return false;
      ConstantValue = Incoming;
    }
  }
  return true;
}

template <>
Intrinsic::ID MachineSSAContext::getIntrinsicID(const MachineInstr &MI) {
  if (auto *GI = dyn_cast<GIntrinsic>(&MI))
    return GI->getIntrinsicID();
  return Intrinsic::not_intrinsic;
}

template <>
Printable MachineSSAContext::print(const MachineBasicBlock *Block) const {
  if (!Block)
    return Printable([](raw_ostream &Out) { Out << "<nullptr>"; });
  return Printable([Block](raw_ostream &Out) { Block->printName(Out); });
}

template <> Printable MachineSSAContext::print(const MachineInstr *I) const {
  return Printable([I](raw_ostream &Out) { I->print(Out); });
}

template <> Printable MachineSSAContext::print(Register Value) const {
  auto *MRI = &F->getRegInfo();
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

template <>
Printable MachineSSAContext::printAsOperand(const MachineBasicBlock *BB) const {
  return Printable([BB](raw_ostream &Out) { BB->printAsOperand(Out); });
}
