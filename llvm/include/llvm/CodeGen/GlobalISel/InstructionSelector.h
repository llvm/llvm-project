//==-- llvm/CodeGen/GlobalISel/InstructionSelector.h -------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file This file declares the API for the instruction selector.
/// This class is responsible for selecting machine instructions.
/// It's implemented by the target. It's used by the InstructionSelect pass.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_INSTRUCTIONSELECTOR_H
#define LLVM_CODEGEN_GLOBALISEL_INSTRUCTIONSELECTOR_H

#include "llvm/ADT/Optional.h"
#include <bitset>
#include <cstdint>
#include <functional>

namespace llvm {
class MachineInstr;
class MachineInstrBuilder;
class MachineFunction;
class MachineOperand;
class MachineRegisterInfo;
class RegisterBankInfo;
class TargetInstrInfo;
class TargetRegisterClass;
class TargetRegisterInfo;

/// Container class for CodeGen predicate results.
/// This is convenient because std::bitset does not have a constructor
/// with an initializer list of set bits.
///
/// Each InstructionSelector subclass should define a PredicateBitset class with:
///   const unsigned MAX_SUBTARGET_PREDICATES = 192;
///   using PredicateBitset = PredicateBitsetImpl<MAX_SUBTARGET_PREDICATES>;
/// and updating the constant to suit the target. Tablegen provides a suitable
/// definition for the predicates in use in <Target>GenGlobalISel.inc when
/// GET_GLOBALISEL_PREDICATE_BITSET is defined.
template <std::size_t MaxPredicates>
class PredicateBitsetImpl : public std::bitset<MaxPredicates> {
public:
  // Cannot inherit constructors because it's not supported by VC++..
  PredicateBitsetImpl() = default;

  PredicateBitsetImpl(const std::bitset<MaxPredicates> &B)
      : std::bitset<MaxPredicates>(B) {}

  PredicateBitsetImpl(std::initializer_list<unsigned> Init) {
    for (auto I : Init)
      std::bitset<MaxPredicates>::set(I);
  }
};

/// Provides the logic to select generic machine instructions.
class InstructionSelector {
public:
  virtual ~InstructionSelector() {}

  /// Select the (possibly generic) instruction \p I to only use target-specific
  /// opcodes. It is OK to insert multiple instructions, but they cannot be
  /// generic pre-isel instructions.
  ///
  /// \returns whether selection succeeded.
  /// \pre  I.getParent() && I.getParent()->getParent()
  /// \post
  ///   if returns true:
  ///     for I in all mutated/inserted instructions:
  ///       !isPreISelGenericOpcode(I.getOpcode())
  ///
  virtual bool select(MachineInstr &I) const = 0;

protected:
  typedef std::function<void(MachineInstrBuilder &)> ComplexRendererFn;

  InstructionSelector();

  /// Constrain a register operand of an instruction \p I to a specified
  /// register class. This could involve inserting COPYs before (for uses) or
  /// after (for defs) and may replace the operand of \p I.
  /// \returns whether operand regclass constraining succeeded.
  bool constrainOperandRegToRegClass(MachineInstr &I, unsigned OpIdx,
                                     const TargetRegisterClass &RC,
                                     const TargetInstrInfo &TII,
                                     const TargetRegisterInfo &TRI,
                                     const RegisterBankInfo &RBI) const;

  /// Mutate the newly-selected instruction \p I to constrain its (possibly
  /// generic) virtual register operands to the instruction's register class.
  /// This could involve inserting COPYs before (for uses) or after (for defs).
  /// This requires the number of operands to match the instruction description.
  /// \returns whether operand regclass constraining succeeded.
  ///
  // FIXME: Not all instructions have the same number of operands. We should
  // probably expose a constrain helper per operand and let the target selector
  // constrain individual registers, like fast-isel.
  bool constrainSelectedInstRegOperands(MachineInstr &I,
                                        const TargetInstrInfo &TII,
                                        const TargetRegisterInfo &TRI,
                                        const RegisterBankInfo &RBI) const;

  bool isOperandImmEqual(const MachineOperand &MO, int64_t Value,
                         const MachineRegisterInfo &MRI) const;

  bool isObviouslySafeToFold(MachineInstr &MI) const;
};

} // End namespace llvm.

#endif
