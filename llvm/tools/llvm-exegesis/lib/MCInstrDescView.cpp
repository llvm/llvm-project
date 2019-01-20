//===-- MCInstrDescView.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCInstrDescView.h"

#include <iterator>
#include <map>
#include <tuple>

#include "llvm/ADT/STLExtras.h"

namespace llvm {
namespace exegesis {

unsigned Variable::getIndex() const {
  assert(Index >= 0 && "Index must be set");
  return Index;
}

unsigned Variable::getPrimaryOperandIndex() const {
  assert(!TiedOperands.empty());
  return TiedOperands[0];
}

bool Variable::hasTiedOperands() const {
  assert(TiedOperands.size() <= 2 &&
         "No more than two operands can be tied together");
  // By definition only Use and Def operands can be tied together.
  // TiedOperands[0] is the Def operand (LLVM stores defs first).
  // TiedOperands[1] is the Use operand.
  return TiedOperands.size() > 1;
}

unsigned Operand::getIndex() const {
  assert(Index >= 0 && "Index must be set");
  return Index;
}

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

unsigned Operand::getTiedToIndex() const {
  assert(isTied() && "Operand must be tied to get the tied index");
  assert(TiedToIndex >= 0 && "TiedToIndex must be set");
  return TiedToIndex;
}

unsigned Operand::getVariableIndex() const {
  assert(isVariable() && "Operand must be variable to get the Variable index");
  assert(VariableIndex >= 0 && "VariableIndex must be set");
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

Instruction::Instruction(const llvm::MCInstrInfo &InstrInfo,
                         const RegisterAliasingTrackerCache &RATC,
                         unsigned Opcode)
    : Description(&InstrInfo.get(Opcode)), Name(InstrInfo.getName(Opcode)) {
  unsigned OpIndex = 0;
  for (; OpIndex < Description->getNumOperands(); ++OpIndex) {
    const auto &OpInfo = Description->opInfo_begin()[OpIndex];
    Operand Operand;
    Operand.Index = OpIndex;
    Operand.IsDef = (OpIndex < Description->getNumDefs());
    // TODO(gchatelet): Handle isLookupPtrRegClass.
    if (OpInfo.RegClass >= 0)
      Operand.Tracker = &RATC.getRegisterClass(OpInfo.RegClass);
    Operand.TiedToIndex =
        Description->getOperandConstraint(OpIndex, llvm::MCOI::TIED_TO);
    Operand.Info = &OpInfo;
    Operands.push_back(Operand);
  }
  for (const llvm::MCPhysReg *MCPhysReg = Description->getImplicitDefs();
       MCPhysReg && *MCPhysReg; ++MCPhysReg, ++OpIndex) {
    Operand Operand;
    Operand.Index = OpIndex;
    Operand.IsDef = true;
    Operand.Tracker = &RATC.getRegister(*MCPhysReg);
    Operand.ImplicitReg = MCPhysReg;
    Operands.push_back(Operand);
  }
  for (const llvm::MCPhysReg *MCPhysReg = Description->getImplicitUses();
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
  return any_of(Operands, [](const Operand &Op) {
    return Op.isReg() && Op.isExplicit() && Op.isMemory();
  });
}

bool Instruction::hasAliasingImplicitRegisters() const {
  return ImplDefRegs.anyCommon(ImplUseRegs);
}

bool Instruction::hasAliasingImplicitRegistersThrough(
    const Instruction &OtherInstr) const {
  return ImplDefRegs.anyCommon(OtherInstr.ImplUseRegs) &&
         OtherInstr.ImplDefRegs.anyCommon(ImplUseRegs);
}

bool Instruction::hasAliasingRegistersThrough(
    const Instruction &OtherInstr) const {
  return AllDefRegs.anyCommon(OtherInstr.AllUseRegs) &&
         OtherInstr.AllDefRegs.anyCommon(AllUseRegs);
}

bool Instruction::hasTiedRegisters() const {
  return llvm::any_of(
      Variables, [](const Variable &Var) { return Var.hasTiedOperands(); });
}

bool Instruction::hasAliasingRegisters() const {
  return AllDefRegs.anyCommon(AllUseRegs);
}

bool Instruction::hasOneUseOrOneDef() const {
  return AllDefRegs.count() || AllUseRegs.count();
}

void Instruction::dump(const llvm::MCRegisterInfo &RegInfo,
                       llvm::raw_ostream &Stream) const {
  Stream << "- " << Name << "\n";
  for (const auto &Op : Operands) {
    Stream << "- Op" << Op.getIndex();
    if (Op.isExplicit())
      Stream << " Explicit";
    if (Op.isImplicit())
      Stream << " Implicit";
    if (Op.isUse())
      Stream << " Use";
    if (Op.isDef())
      Stream << " Def";
    if (Op.isImmediate())
      Stream << " Immediate";
    if (Op.isMemory())
      Stream << " Memory";
    if (Op.isReg()) {
      if (Op.isImplicitReg())
        Stream << " Reg(" << RegInfo.getName(Op.getImplicitReg()) << ")";
      else
        Stream << " RegClass("
               << RegInfo.getRegClassName(
                      &RegInfo.getRegClass(Op.Info->RegClass))
               << ")";
    }
    if (Op.isTied())
      Stream << " TiedToOp" << Op.getTiedToIndex();
    Stream << "\n";
  }
  for (const auto &Var : Variables) {
    Stream << "- Var" << Var.getIndex();
    Stream << " [";
    bool IsFirst = true;
    for (auto OperandIndex : Var.TiedOperands) {
      if (!IsFirst)
        Stream << ",";
      Stream << "Op" << OperandIndex;
      IsFirst = false;
    }
    Stream << "]";
    Stream << "\n";
  }
  if (hasMemoryOperands())
    Stream << "- hasMemoryOperands\n";
  if (hasAliasingImplicitRegisters())
    Stream << "- hasAliasingImplicitRegisters (execution is always serial)\n";
  if (hasTiedRegisters())
    Stream << "- hasTiedRegisters (execution is always serial)\n";
  if (hasAliasingRegisters())
    Stream << "- hasAliasingRegisters\n";
}

InstructionsCache::InstructionsCache(const llvm::MCInstrInfo &InstrInfo,
                                     const RegisterAliasingTrackerCache &RATC)
    : InstrInfo(InstrInfo), RATC(RATC) {}

const Instruction &InstructionsCache::getInstr(unsigned Opcode) const {
  auto &Found = Instructions[Opcode];
  if (!Found)
    Found.reset(new Instruction(InstrInfo, RATC, Opcode));
  return *Found;
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
    const Instruction &DefInstruction, const Instruction &UseInstruction) {
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
} // namespace llvm
