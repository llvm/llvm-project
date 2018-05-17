//===-- MCInstrDescView.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCInstrDescView.h"

#include <iterator>
#include <map>
#include <tuple>

#include "llvm/ADT/STLExtras.h"

namespace exegesis {

static void tie(const Operand *FromOperand, llvm::Optional<Variable> &Var) {
  if (!Var)
    Var.emplace();
  Var->TiedOperands.push_back(FromOperand);
}

Instruction::Instruction(const llvm::MCInstrDesc &MCInstrDesc,
                         RegisterAliasingTrackerCache &RATC)
    : Description(MCInstrDesc) {
  unsigned OpIndex = 0;
  for (; OpIndex < MCInstrDesc.getNumOperands(); ++OpIndex) {
    const auto &OpInfo = MCInstrDesc.opInfo_begin()[OpIndex];
    Operand Operand;
    Operand.Index = OpIndex;
    Operand.IsDef = (OpIndex < MCInstrDesc.getNumDefs());
    Operand.IsExplicit = true;
    // TODO(gchatelet): Handle isLookupPtrRegClass.
    if (OpInfo.RegClass >= 0)
      Operand.Tracker = &RATC.getRegisterClass(OpInfo.RegClass);
    Operand.Info = &OpInfo;
    Operands.push_back(Operand);
  }
  for (const llvm::MCPhysReg *MCPhysReg = MCInstrDesc.getImplicitDefs();
       MCPhysReg && *MCPhysReg; ++MCPhysReg, ++OpIndex) {
    Operand Operand;
    Operand.Index = OpIndex;
    Operand.IsDef = true;
    Operand.IsExplicit = false;
    Operand.Tracker = &RATC.getRegister(*MCPhysReg);
    Operand.ImplicitReg = MCPhysReg;
    Operands.push_back(Operand);
  }
  for (const llvm::MCPhysReg *MCPhysReg = MCInstrDesc.getImplicitUses();
       MCPhysReg && *MCPhysReg; ++MCPhysReg, ++OpIndex) {
    Operand Operand;
    Operand.Index = OpIndex;
    Operand.IsDef = false;
    Operand.IsExplicit = false;
    Operand.Tracker = &RATC.getRegister(*MCPhysReg);
    Operand.ImplicitReg = MCPhysReg;
    Operands.push_back(Operand);
  }
  // Set TiedTo for operands.
  for (auto &Op : Operands) {
    if (Op.IsExplicit) {
      const int TiedTo =
          MCInstrDesc.getOperandConstraint(Op.Index, llvm::MCOI::TIED_TO);
      if (TiedTo >= 0) {
        Op.TiedTo = &Operands[TiedTo];
        tie(&Op, Operands[TiedTo].Var);
      } else {
        tie(&Op, Op.Var);
      }
    }
  }
  for (auto &Op : Operands) {
    if (Op.Var) {
      Variables.push_back(&*Op.Var);
    }
  }
  // Processing Aliasing.
  DefRegisters = RATC.emptyRegisters();
  UseRegisters = RATC.emptyRegisters();
  for (const auto &Op : Operands) {
    if (Op.Tracker) {
      auto &Registers = Op.IsDef ? DefRegisters : UseRegisters;
      Registers |= Op.Tracker->aliasedBits();
    }
  }
}

bool RegisterOperandAssignment::
operator==(const RegisterOperandAssignment &Other) const {
  return std::tie(Op, Reg) == std::tie(Other.Op, Other.Reg);
}

bool AliasingRegisterOperands::
operator==(const AliasingRegisterOperands &Other) const {
  return std::tie(Defs, Uses) == std::tie(Other.Defs, Other.Uses);
}

static void addOperandIfAlias(
    const llvm::MCPhysReg Reg, bool SelectDef, llvm::ArrayRef<Operand> Operands,
    llvm::SmallVectorImpl<RegisterOperandAssignment> &OperandValues) {
  for (const auto &Op : Operands) {
    if (Op.Tracker && Op.IsDef == SelectDef) {
      const int SourceReg = Op.Tracker->getOrigin(Reg);
      if (SourceReg >= 0)
        OperandValues.emplace_back(&Op, SourceReg);
    }
  }
}

bool AliasingRegisterOperands::hasImplicitAliasing() const {
  const auto HasImplicit = [](const RegisterOperandAssignment &ROV) {
    return !ROV.Op->IsExplicit;
  };
  return llvm::any_of(Defs, HasImplicit) && llvm::any_of(Uses, HasImplicit);
}

bool AliasingConfigurations::empty() const { return Configurations.empty(); }

bool AliasingConfigurations::hasImplicitAliasing() const {
  return llvm::any_of(Configurations, [](const AliasingRegisterOperands &ARO) {
    return ARO.hasImplicitAliasing();
  });
}

AliasingConfigurations::AliasingConfigurations(
    const Instruction &DefInstruction, const Instruction &UseInstruction)
    : DefInstruction(DefInstruction), UseInstruction(UseInstruction) {
  if (UseInstruction.UseRegisters.anyCommon(DefInstruction.DefRegisters)) {
    auto CommonRegisters = UseInstruction.UseRegisters;
    CommonRegisters &= DefInstruction.DefRegisters;
    for (const llvm::MCPhysReg Reg : CommonRegisters.set_bits()) {
      AliasingRegisterOperands ARO;
      addOperandIfAlias(Reg, true, DefInstruction.Operands, ARO.Defs);
      addOperandIfAlias(Reg, false, UseInstruction.Operands, ARO.Uses);
      if (!ARO.Defs.empty() && !ARO.Uses.empty() &&
          !llvm::is_contained(Configurations, ARO))
        Configurations.push_back(std::move(ARO));
    }
  }
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

static void randomize(Variable &Var) {
  assert(!Var.TiedOperands.empty());
  assert(Var.TiedOperands.front() != nullptr);
  const Operand &Op = *Var.TiedOperands.front();
  assert(Op.Info != nullptr);
  const auto &OpInfo = *Op.Info;
  switch (OpInfo.OperandType) {
  case llvm::MCOI::OperandType::OPERAND_IMMEDIATE:
    // FIXME: explore immediate values too.
    Var.AssignedValue = llvm::MCOperand::createImm(1);
    break;
  case llvm::MCOI::OperandType::OPERAND_REGISTER: {
    assert(Op.Tracker);
    const auto &Registers = Op.Tracker->sourceBits();
    Var.AssignedValue = llvm::MCOperand::createReg(randomBit(Registers));
    break;
  }
  default:
    break;
  }
}

static void setRegisterOperandValue(const RegisterOperandAssignment &ROV) {
  const Operand *Op = ROV.Op->TiedTo ? ROV.Op->TiedTo : ROV.Op;
  assert(Op->Var);
  auto &AssignedValue = Op->Var->AssignedValue;
  if (AssignedValue.isValid()) {
    assert(AssignedValue.isReg() && AssignedValue.getReg() == ROV.Reg);
    return;
  }
  Op->Var->AssignedValue = llvm::MCOperand::createReg(ROV.Reg);
}

size_t randomBit(const llvm::BitVector &Vector) {
  assert(Vector.any());
  auto Itr = Vector.set_bits_begin();
  for (size_t I = randomIndex(Vector.count()); I != 0; --I)
    ++Itr;
  return *Itr;
}

void setRandomAliasing(const AliasingConfigurations &AliasingConfigurations) {
  assert(!AliasingConfigurations.empty());
  assert(!AliasingConfigurations.hasImplicitAliasing());
  const auto &RandomConf = randomElement(AliasingConfigurations.Configurations);
  setRegisterOperandValue(randomElement(RandomConf.Defs));
  setRegisterOperandValue(randomElement(RandomConf.Uses));
}

void randomizeUnsetVariable(const Instruction &Instruction) {
  for (auto *Var : Instruction.Variables)
    if (!Var->AssignedValue.isValid())
      randomize(*Var);
}

void clearVariableAssignments(const Instruction &Instruction) {
  for (auto *Var : Instruction.Variables)
    Var->AssignedValue = llvm::MCOperand();
}

llvm::MCInst build(const Instruction &Instruction) {
  llvm::MCInst Result;
  Result.setOpcode(Instruction.Description.Opcode);
  for (const auto &Op : Instruction.Operands) {
    if (Op.IsExplicit) {
      auto &Var = Op.TiedTo ? Op.TiedTo->Var : Op.Var;
      assert(Var);
      Result.addOperand(Var->AssignedValue);
    }
  }
  return Result;
}

llvm::MCInst randomizeUnsetVariablesAndBuild(const Instruction &Instruction) {
  randomizeUnsetVariable(Instruction);
  return build(Instruction);
}

void DumpMCOperand(const llvm::MCRegisterInfo &MCRegisterInfo,
                   const llvm::MCOperand &Op, llvm::raw_ostream &OS) {
  if (!Op.isValid())
    OS << "Invalid";
  else if (Op.isReg())
    OS << MCRegisterInfo.getName(Op.getReg());
  else if (Op.isImm())
    OS << Op.getImm();
  else if (Op.isFPImm())
    OS << Op.getFPImm();
  else if (Op.isExpr())
    OS << "Expr";
  else if (Op.isInst())
    OS << "SubInst";
}

void DumpMCInst(const llvm::MCRegisterInfo &MCRegisterInfo,
                const llvm::MCInstrInfo &MCInstrInfo,
                const llvm::MCInst &MCInst, llvm::raw_ostream &OS) {
  OS << MCInstrInfo.getName(MCInst.getOpcode());
  for (unsigned I = 0, E = MCInst.getNumOperands(); I < E; ++I) {
    if (I > 0)
      OS << ',';
    OS << ' ';
    DumpMCOperand(MCRegisterInfo, MCInst.getOperand(I), OS);
  }
}

} // namespace exegesis
