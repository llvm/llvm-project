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

// forward declaration.
static void randomize(const Instruction &Instr, const Variable &Var,
                      llvm::MCOperand &AssignedValue,
                      const llvm::BitVector &ForbiddenRegs);

bool InstructionTemplate::hasImmediateVariables() const {
  return llvm::any_of(Instr.Variables, [this](const Variable &Var) {
    assert(!Var.TiedOperands.empty());
    const unsigned OpIndex = Var.TiedOperands[0];
    const Operand &Op = Instr.Operands[OpIndex];
    assert(Op.Info);
    return Op.Info->OperandType == llvm::MCOI::OPERAND_IMMEDIATE;
  });
}

void InstructionTemplate::randomizeUnsetVariables(
    const llvm::BitVector &ForbiddenRegs) {
  for (const Variable &Var : Instr.Variables) {
    llvm::MCOperand &AssignedValue = getValueFor(Var);
    if (!AssignedValue.isValid())
      randomize(Instr, Var, AssignedValue, ForbiddenRegs);
  }
}

llvm::MCInst InstructionTemplate::build() const {
  llvm::MCInst Result;
  Result.setOpcode(Instr.Description->Opcode);
  for (const auto &Op : Instr.Operands)
    if (Op.IsExplicit)
      Result.addOperand(getValueFor(Op));
  return Result;
}

std::mt19937 &randomGenerator() {
  static std::random_device RandomDevice;
  static std::mt19937 RandomGenerator(RandomDevice());
  return RandomGenerator;
}

static size_t randomIndex(size_t Size) {
  assert(Size > 0);
  std::uniform_int_distribution<> Distribution(0, Size - 1);
  return Distribution(randomGenerator());
}

template <typename C>
static auto randomElement(const C &Container) -> decltype(Container[0]) {
  return Container[randomIndex(Container.size())];
}

static void randomize(const Instruction &Instr, const Variable &Var,
                      llvm::MCOperand &AssignedValue,
                      const llvm::BitVector &ForbiddenRegs) {
  assert(!Var.TiedOperands.empty());
  const Operand &Op = Instr.Operands[Var.TiedOperands.front()];
  assert(Op.Info != nullptr);
  const auto &OpInfo = *Op.Info;
  switch (OpInfo.OperandType) {
  case llvm::MCOI::OperandType::OPERAND_IMMEDIATE:
    // FIXME: explore immediate values too.
    AssignedValue = llvm::MCOperand::createImm(1);
    break;
  case llvm::MCOI::OperandType::OPERAND_REGISTER: {
    assert(Op.Tracker);
    auto AllowedRegs = Op.Tracker->sourceBits();
    assert(AllowedRegs.size() == ForbiddenRegs.size());
    for (auto I : ForbiddenRegs.set_bits())
      AllowedRegs.reset(I);
    AssignedValue = llvm::MCOperand::createReg(randomBit(AllowedRegs));
    break;
  }
  default:
    break;
  }
}

static void setRegisterOperandValue(const RegisterOperandAssignment &ROV,
                                    InstructionTemplate &IB) {
  assert(ROV.Op);
  if (ROV.Op->IsExplicit) {
    auto &AssignedValue = IB.getValueFor(*ROV.Op);
    if (AssignedValue.isValid()) {
      assert(AssignedValue.isReg() && AssignedValue.getReg() == ROV.Reg);
      return;
    }
    AssignedValue = llvm::MCOperand::createReg(ROV.Reg);
  } else {
    assert(ROV.Op->ImplicitReg != nullptr);
    assert(ROV.Reg == *ROV.Op->ImplicitReg);
  }
}

size_t randomBit(const llvm::BitVector &Vector) {
  assert(Vector.any());
  auto Itr = Vector.set_bits_begin();
  for (size_t I = randomIndex(Vector.count()); I != 0; --I)
    ++Itr;
  return *Itr;
}

void setRandomAliasing(const AliasingConfigurations &AliasingConfigurations,
                       InstructionTemplate &DefIB, InstructionTemplate &UseIB) {
  assert(!AliasingConfigurations.empty());
  assert(!AliasingConfigurations.hasImplicitAliasing());
  const auto &RandomConf = randomElement(AliasingConfigurations.Configurations);
  setRegisterOperandValue(randomElement(RandomConf.Defs), DefIB);
  setRegisterOperandValue(randomElement(RandomConf.Uses), UseIB);
}

} // namespace exegesis
