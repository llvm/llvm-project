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

Instruction::Instruction(const llvm::MCInstrDesc &MCInstrDesc,
                         const RegisterAliasingTrackerCache &RATC)
    : Description(&MCInstrDesc) {
  unsigned OpIndex = 0;
  for (; OpIndex < MCInstrDesc.getNumOperands(); ++OpIndex) {
    const auto &OpInfo = MCInstrDesc.opInfo_begin()[OpIndex];
    Operand Operand;
    Operand.Index = OpIndex;
    Operand.IsDef = (OpIndex < MCInstrDesc.getNumDefs());
    Operand.IsMem = OpInfo.OperandType == llvm::MCOI::OPERAND_MEMORY;
    Operand.IsExplicit = true;
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
  // Assigning Variables to non tied explicit operands.
  Variables.reserve(Operands.size()); // Variables.size() <= Operands.size()
  for (auto &Op : Operands)
    if (Op.IsExplicit && Op.TiedToIndex < 0) {
      const size_t VariableIndex = Variables.size();
      Op.VariableIndex = VariableIndex;
      Variables.emplace_back();
      Variables.back().Index = VariableIndex;
    }
  // Assigning Variables to tied operands.
  for (auto &Op : Operands)
    if (Op.TiedToIndex >= 0)
      Op.VariableIndex = Operands[Op.TiedToIndex].VariableIndex;
  // Assigning Operands to Variables.
  for (auto &Op : Operands)
    if (Op.VariableIndex >= 0)
      Variables[Op.VariableIndex].TiedOperands.push_back(Op.Index);
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

bool Instruction::hasMemoryOperands() const {
  return std::any_of(Operands.begin(), Operands.end(),
                     [](const Operand &Op) { return Op.IsMem; });
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
