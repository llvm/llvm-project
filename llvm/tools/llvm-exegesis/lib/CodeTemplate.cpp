//===-- CodeTemplate.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CodeTemplate.h"

namespace exegesis {

CodeTemplate::CodeTemplate(CodeTemplate &&) = default;

CodeTemplate &CodeTemplate::operator=(CodeTemplate &&) = default;

InstructionTemplate::InstructionTemplate(const Instruction &Instr)
    : Instr(Instr), VariableValues(Instr.Variables.size()) {}

InstructionTemplate::InstructionTemplate(InstructionTemplate &&) = default;

InstructionTemplate &InstructionTemplate::
operator=(InstructionTemplate &&) = default;

InstructionTemplate::InstructionTemplate(const InstructionTemplate &) = default;

InstructionTemplate &InstructionTemplate::
operator=(const InstructionTemplate &) = default;

unsigned InstructionTemplate::getOpcode() const {
  return Instr.Description->getOpcode();
}

llvm::MCOperand &InstructionTemplate::getValueFor(const Variable &Var) {
  return VariableValues[Var.Index];
}

const llvm::MCOperand &
InstructionTemplate::getValueFor(const Variable &Var) const {
  return VariableValues[Var.Index];
}

llvm::MCOperand &InstructionTemplate::getValueFor(const Operand &Op) {
  assert(Op.VariableIndex >= 0);
  return getValueFor(Instr.Variables[Op.VariableIndex]);
}

const llvm::MCOperand &
InstructionTemplate::getValueFor(const Operand &Op) const {
  assert(Op.VariableIndex >= 0);
  return getValueFor(Instr.Variables[Op.VariableIndex]);
}

bool InstructionTemplate::hasImmediateVariables() const {
  return llvm::any_of(Instr.Variables, [this](const Variable &Var) {
    assert(!Var.TiedOperands.empty());
    const unsigned OpIndex = Var.TiedOperands[0];
    const Operand &Op = Instr.Operands[OpIndex];
    assert(Op.Info);
    return Op.Info->OperandType == llvm::MCOI::OPERAND_IMMEDIATE;
  });
}

llvm::MCInst InstructionTemplate::build() const {
  llvm::MCInst Result;
  Result.setOpcode(Instr.Description->Opcode);
  for (const auto &Op : Instr.Operands)
    if (Op.IsExplicit)
      Result.addOperand(getValueFor(Op));
  return Result;
}

} // namespace exegesis
