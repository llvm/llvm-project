//===-- MCInstrDescView.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Provide views around LLVM structures to represents an instruction instance,
/// as well as its implicit and explicit arguments in a uniform way.
/// Arguments that are explicit and independant (non tied) also have a Variable
/// associated to them so the instruction can be fully defined by reading its
/// Variables.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_EXEGESIS_MCINSTRDESCVIEW_H
#define LLVM_TOOLS_LLVM_EXEGESIS_MCINSTRDESCVIEW_H

#include <random>

#include "RegisterAliasing.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"

namespace exegesis {

struct Operand; // forward declaration.

// A variable represents the value of an Operand or a set of Operands if they ar
// tied together.
struct Variable {
  llvm::SmallVector<const Operand *, 2> TiedOperands;
  llvm::MCOperand AssignedValue;
};

// MCOperandInfo can only represents Explicit operands. This object gives a
// uniform view of Implicit and Explicit Operands.
//
// - Index: can be used to refer to MCInstrDesc::operands for Explicit operands.
// - Tracker: is set for Register Operands and is used to keep track of possible
// registers and the registers reachable from them (aliasing registers).
// - Info: a shortcut for MCInstrDesc::operands()[Index].
// - TiedTo: a pointer to the Operand holding the value or nullptr.
// - ImplicitReg: a pointer to the register value when Operand is Implicit,
// nullptr otherwise.
// - Variable: The value associated with this Operand. It is only set for
// explicit operands that are not TiedTo.
struct Operand {
  uint8_t Index = 0;
  bool IsDef = false;
  bool IsExplicit = false;
  const RegisterAliasingTracker *Tracker = nullptr; // Set for Register Op.
  const llvm::MCOperandInfo *Info = nullptr;        // Set for Explicit Op.
  const Operand *TiedTo = nullptr;                  // Set for Reg/Explicit Op.
  const llvm::MCPhysReg *ImplicitReg = nullptr;     // Set for Implicit Op.
  mutable llvm::Optional<Variable> Var;             // Set for Explicit Op.
};

// A view over an MCInstrDesc offering a convenient interface to compute
// Register aliasing and assign values to Operands.
struct Instruction {
  Instruction(const llvm::MCInstrDesc &MCInstrDesc,
              RegisterAliasingTrackerCache &ATC);

  const llvm::MCInstrDesc &Description;
  llvm::SmallVector<Operand, 8> Operands;
  llvm::SmallVector<Variable *, 8> Variables;
  llvm::BitVector DefRegisters; // The union of the aliased def registers.
  llvm::BitVector UseRegisters; // The union of the aliased use registers.
};

// Represents the assignment of a Register to an Operand.
struct RegisterOperandAssignment {
  RegisterOperandAssignment(const Operand *Operand, llvm::MCPhysReg Reg)
      : Op(Operand), Reg(Reg) {}

  const Operand *Op; // Pointer to an Explicit Register Operand.
  llvm::MCPhysReg Reg;

  bool operator==(const RegisterOperandAssignment &other) const;
};

// Represents a set of Operands that would alias through the use of some
// Registers.
// There are two reasons why operands would alias:
// - The registers assigned to each of the operands are the same or alias each
//   other (e.g. AX/AL)
// - The operands are tied.
struct AliasingRegisterOperands {
  llvm::SmallVector<RegisterOperandAssignment, 1> Defs; // Unlikely size() > 1.
  llvm::SmallVector<RegisterOperandAssignment, 2> Uses;

  // True is Defs and Use contain an Implicit Operand.
  bool hasImplicitAliasing() const;

  bool operator==(const AliasingRegisterOperands &other) const;
};

// Returns all possible configurations leading Def registers of DefInstruction
// to alias with Use registers of UseInstruction.
struct AliasingConfigurations {
  AliasingConfigurations(const Instruction &DefInstruction,
                         const Instruction &UseInstruction);

  bool empty() const; // True if no aliasing configuration is found.
  bool hasImplicitAliasing() const;
  void setExplicitAliasing() const;

  const Instruction &DefInstruction;
  const Instruction &UseInstruction;
  llvm::SmallVector<AliasingRegisterOperands, 32> Configurations;
};

// A global Random Number Generator to randomize configurations.
// FIXME: Move random number generation into an object and make it seedable for
// unit tests.
std::mt19937 &randomGenerator();

// Picks a random bit among the bits set in Vector and returns its index.
// Precondition: Vector must have at least one bit set.
size_t randomBit(const llvm::BitVector &Vector);

// Picks a random configuration, then select a random def and a random use from
// it and set the target Variables to the selected values.
// FIXME: This function mutates some nested variables in a const object, please
// fix ASAP.
void setRandomAliasing(const AliasingConfigurations &AliasingConfigurations);

// Set all Instruction's Variables AssignedValue to Invalid.
void clearVariableAssignments(const Instruction &Instruction);

// Assigns a Random Value to all Instruction's Variables that are still Invalid.
llvm::MCInst randomizeUnsetVariablesAndBuild(const Instruction &Instruction);

// Writes MCInst to OS.
// This is not assembly but the internal LLVM's name for instructions and
// registers.
void DumpMCInst(const llvm::MCRegisterInfo &MCRegisterInfo,
                const llvm::MCInstrInfo &MCInstrInfo,
                const llvm::MCInst &MCInst, llvm::raw_ostream &OS);

} // namespace exegesis

#endif // LLVM_TOOLS_LLVM_EXEGESIS_MCINSTRDESCVIEW_H
