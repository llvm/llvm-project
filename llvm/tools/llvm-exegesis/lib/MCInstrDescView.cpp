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

unsigned Variable::getIndex() const {
  assert(Index >= 0);
  return Index;
}
unsigned Variable::getPrimaryOperandIndex() const {
  assert(!TiedOperands.empty());
  return TiedOperands[0];
}

bool Variable::hasTiedOperands() const { return TiedOperands.size() > 1; }

bool Operand::getIndex() const { return Index; }

bool Operand::isExplicit() const { return Info; }

bool Operand::isImplicit() const { return !Info; }

bool Operand::isImplicitReg() const { return ImplicitReg; }

bool Operand::isDef() const { return IsDef; }

bool Operand::isUse() const { return !IsDef; }

bool Operand::isReg() const { return Tracker; }

bool Operand::isTied() const { return TiedToIndex >= 0; }

bool Operand::isVariable() const { return VariableIndex >= 0; }

bool Operand::isMemory() const {
  return isExplicit() &&
         getExplicitOperandInfo().OperandType == llvm::MCOI::OPERAND_MEMORY;
}

bool Operand::isImmediate() const {
  return isExplicit() &&
         getExplicitOperandInfo().OperandType == llvm::MCOI::OPERAND_IMMEDIATE;
}

int Operand::getTiedToIndex() const {
  assert(isTied());
  return TiedToIndex;
}

int Operand::getVariableIndex() const {
  assert(isVariable());
  return VariableIndex;
}

unsigned Operand::getImplicitReg() const {
  assert(ImplicitReg);
  return *ImplicitReg;
}

const RegisterAliasingTracker &Operand::getRegisterAliasing() const {
  assert(Tracker);
  return *Tracker;
}

const llvm::MCOperandInfo &Operand::getExplicitOperandInfo() const {
  assert(Info);
  return *Info;
}

Instruction::Instruction(const llvm::MCInstrDesc &MCInstrDesc,
                         const RegisterAliasingTrackerCache &RATC)
    : Description(&MCInstrDesc) {
  unsigned OpIndex = 0;
  for (; OpIndex < MCInstrDesc.getNumOperands(); ++OpIndex) {
    const auto &OpInfo = MCInstrDesc.opInfo_begin()[OpIndex];
    Operand Operand;
    Operand.Index = OpIndex;
    Operand.IsDef = (OpIndex < MCInstrDesc.getNumDefs());
    // TODO(gchatelet): Handle isLookupPtrRegClass.
    if (OpInfo.RegClass >= 0)
      Operand.Tracker = &RATC.getRegisterClass(OpInfo.RegClass);
    Operand.TiedToIndex =
        MCInstrDesc.getOperandConstraint(OpIndex, llvm::MCOI::TIED_TO);
    Operand.Info = &OpInfo;
    Operands.push_back(Operand);
  }
  for (const llvm::MCPhysReg *MCPhysReg = MCInstrDesc.getImplicitDefs();
       MCPhysReg && *MCPhysReg; ++MCPhysReg, ++OpIndex) {
    Operand Operand;
    Operand.Index = OpIndex;
    Operand.IsDef = true;
    Operand.Tracker = &RATC.getRegister(*MCPhysReg);
    Operand.ImplicitReg = MCPhysReg;
    Operands.push_back(Operand);
  }
  for (const llvm::MCPhysReg *MCPhysReg = MCInstrDesc.getImplicitUses();
       MCPhysReg && *MCPhysReg; ++MCPhysReg, ++OpIndex) {
    Operand Operand;
    Operand.Index = OpIndex;
    Operand.IsDef = false;
    Operand.Tracker = &RATC.getRegister(*MCPhysReg);
    Operand.ImplicitReg = MCPhysReg;
    Operands.push_back(Operand);
  }
  // Assigning Variables to non tied explicit operands.
  Variables.reserve(Operands.size()); // Variables.size() <= Operands.size()
  for (auto &Op : Operands)
    if (Op.isExplicit() && !Op.isTied()) {
      const size_t VariableIndex = Variables.size();
      Op.VariableIndex = VariableIndex;
      Variables.emplace_back();
      Variables.back().Index = VariableIndex;
    }
  // Assigning Variables to tied operands.
  for (auto &Op : Operands)
    if (Op.isTied())
      Op.VariableIndex = Operands[Op.getTiedToIndex()].getVariableIndex();
  // Assigning Operands to Variables.
  for (auto &Op : Operands)
    if (Op.isVariable())
      Variables[Op.getVariableIndex()].TiedOperands.push_back(Op.getIndex());
  // Processing Aliasing.
  ImplDefRegs = RATC.emptyRegisters();
  ImplUseRegs = RATC.emptyRegisters();
  AllDefRegs = RATC.emptyRegisters();
  AllUseRegs = RATC.emptyRegisters();
  for (const auto &Op : Operands) {
    if (Op.isReg()) {
      const auto &AliasingBits = Op.getRegisterAliasing().aliasedBits();
      if (Op.isDef())
        AllDefRegs |= AliasingBits;
      if (Op.isUse())
        AllUseRegs |= AliasingBits;
      if (Op.isDef() && Op.isImplicit())
        ImplDefRegs |= AliasingBits;
      if (Op.isUse() && Op.isImplicit())
        ImplUseRegs |= AliasingBits;
    }
  }
}

const Operand &Instruction::getPrimaryOperand(const Variable &Var) const {
  const auto PrimaryOperandIndex = Var.getPrimaryOperandIndex();
  assert(PrimaryOperandIndex < Operands.size());
  return Operands[PrimaryOperandIndex];
}

bool Instruction::hasMemoryOperands() const {
  return std::any_of(Operands.begin(), Operands.end(), [](const Operand &Op) {
    return Op.isReg() && Op.isExplicit() && Op.isMemory();
  });
}

bool Instruction::hasAliasingImplicitRegisters() const {
  return ImplDefRegs.anyCommon(ImplUseRegs);
}

bool Instruction::hasTiedRegisters() const {
  return llvm::any_of(
      Variables, [](const Variable &Var) { return Var.hasTiedOperands(); });
}

bool Instruction::hasAliasingRegisters() const {
  return AllDefRegs.anyCommon(AllUseRegs);
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
    if (Op.isReg() && Op.isDef() == SelectDef) {
      const int SourceReg = Op.getRegisterAliasing().getOrigin(Reg);
      if (SourceReg >= 0)
        OperandValues.emplace_back(&Op, SourceReg);
    }
  }
}

bool AliasingRegisterOperands::hasImplicitAliasing() const {
  const auto HasImplicit = [](const RegisterOperandAssignment &ROV) {
    return ROV.Op->isImplicit();
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
  if (UseInstruction.AllUseRegs.anyCommon(DefInstruction.AllDefRegs)) {
    auto CommonRegisters = UseInstruction.AllUseRegs;
    CommonRegisters &= DefInstruction.AllDefRegs;
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
