//===- llvm/CodeGen/GlobalISel/InstructionSelector.h ------------*- C++ -*-===//
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

#include "llvm/ADT/SmallVector.h"
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <vector>

namespace llvm {

class LLT;
class MachineInstr;
class MachineInstrBuilder;
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
/// Each InstructionSelector subclass should define a PredicateBitset class
/// with:
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

enum {
  /// Record the specified instruction
  /// - NewInsnID - Instruction ID to define
  /// - InsnID - Instruction ID
  /// - OpIdx - Operand index
  GIM_RecordInsn,

  /// Check the feature bits
  /// - Expected features
  GIM_CheckFeatures,

  /// Check the opcode on the specified instruction
  /// - InsnID - Instruction ID
  /// - Expected opcode
  GIM_CheckOpcode,
  /// Check the instruction has the right number of operands
  /// - InsnID - Instruction ID
  /// - Expected number of operands
  GIM_CheckNumOperands,

  /// Check the type for the specified operand
  /// - InsnID - Instruction ID
  /// - OpIdx - Operand index
  /// - Expected type
  GIM_CheckType,
  /// Check the register bank for the specified operand
  /// - InsnID - Instruction ID
  /// - OpIdx - Operand index
  /// - Expected register bank (specified as a register class)
  GIM_CheckRegBankForClass,
  /// Check the operand matches a complex predicate
  /// - InsnID - Instruction ID
  /// - OpIdx - Operand index
  /// - RendererID - The renderer to hold the result
  /// - Complex predicate ID
  GIM_CheckComplexPattern,
  /// Check the operand is a specific integer
  /// - InsnID - Instruction ID
  /// - OpIdx - Operand index
  /// - Expected integer
  GIM_CheckConstantInt,
  /// Check the operand is a specific literal integer (i.e. MO.isImm() or
  /// MO.isCImm() is true).
  /// - InsnID - Instruction ID
  /// - OpIdx - Operand index
  /// - Expected integer
  GIM_CheckLiteralInt,
  /// Check the operand is a specific intrinsic ID
  /// - InsnID - Instruction ID
  /// - OpIdx - Operand index
  /// - Expected Intrinsic ID
  GIM_CheckIntrinsicID,
  /// Check the specified operand is an MBB
  /// - InsnID - Instruction ID
  /// - OpIdx - Operand index
  GIM_CheckIsMBB,

  /// Check if the specified operand is safe to fold into the current
  /// instruction.
  /// - InsnID - Instruction ID
  GIM_CheckIsSafeToFold,

  //=== Renderers ===

  /// Mutate an instruction
  /// - NewInsnID - Instruction ID to define
  /// - OldInsnID - Instruction ID to mutate
  /// - NewOpcode - The new opcode to use
  GIR_MutateOpcode,
  /// Build a new instruction
  /// - InsnID - Instruction ID to define
  /// - Opcode - The new opcode to use
  GIR_BuildMI,

  /// Copy an operand to the specified instruction
  /// - NewInsnID - Instruction ID to modify
  /// - OldInsnID - Instruction ID to copy from
  /// - OpIdx - The operand to copy
  GIR_Copy,
  /// Copy an operand to the specified instruction
  /// - NewInsnID - Instruction ID to modify
  /// - OldInsnID - Instruction ID to copy from
  /// - OpIdx - The operand to copy
  /// - SubRegIdx - The subregister to copy
  GIR_CopySubReg,
  /// Add an implicit register def to the specified instruction
  /// - InsnID - Instruction ID to modify
  /// - RegNum - The register to add
  GIR_AddImplicitDef,
  /// Add an implicit register use to the specified instruction
  /// - InsnID - Instruction ID to modify
  /// - RegNum - The register to add
  GIR_AddImplicitUse,
  /// Add an register to the specified instruction
  /// - InsnID - Instruction ID to modify
  /// - RegNum - The register to add
  GIR_AddRegister,
  /// Add an immediate to the specified instruction
  /// - InsnID - Instruction ID to modify
  /// - Imm - The immediate to add
  GIR_AddImm,
  /// Render complex operands to the specified instruction
  /// - InsnID - Instruction ID to modify
  /// - RendererID - The renderer to call
  GIR_ComplexRenderer,

  /// Constrain an instruction operand to a register class.
  /// - InsnID - Instruction ID to modify
  /// - OpIdx - Operand index
  /// - RCEnum - Register class enumeration value
  GIR_ConstrainOperandRC,
  /// Constrain an instructions operands according to the instruction
  /// description.
  /// - InsnID - Instruction ID to modify
  GIR_ConstrainSelectedInstOperands,
  /// Merge all memory operands into instruction.
  /// - InsnID - Instruction ID to modify
  GIR_MergeMemOperands,
  /// Erase from parent.
  /// - InsnID - Instruction ID to erase
  GIR_EraseFromParent,

  /// A successful emission
  GIR_Done,
};

/// Provides the logic to select generic machine instructions.
class InstructionSelector {
public:
  virtual ~InstructionSelector() = default;

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
  using ComplexRendererFn = std::function<void(MachineInstrBuilder &)>;
  using RecordedMIVector = SmallVector<MachineInstr *, 4>;
  using NewMIVector = SmallVector<MachineInstrBuilder, 4>;

  struct MatcherState {
    std::vector<ComplexRendererFn> Renderers;
    RecordedMIVector MIs;

    MatcherState(unsigned MaxRenderers);
  };

public:
  template <class PredicateBitset, class ComplexMatcherMemFn>
  struct MatcherInfoTy {
    const LLT *TypeObjects;
    const PredicateBitset *FeatureBitsets;
    const std::vector<ComplexMatcherMemFn> ComplexPredicates;
  };

protected:
  InstructionSelector();

  /// Execute a given matcher table and return true if the match was successful
  /// and false otherwise.
  template <class TgtInstructionSelector, class PredicateBitset,
            class ComplexMatcherMemFn>
  bool executeMatchTable(
      TgtInstructionSelector &ISel, NewMIVector &OutMIs, MatcherState &State,
      const MatcherInfoTy<PredicateBitset, ComplexMatcherMemFn> &MatcherInfo,
      const int64_t *MatchTable, const TargetInstrInfo &TII,
      MachineRegisterInfo &MRI, const TargetRegisterInfo &TRI,
      const RegisterBankInfo &RBI,
      const PredicateBitset &AvailableFeatures) const;

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

} // end namespace llvm

#endif // LLVM_CODEGEN_GLOBALISEL_INSTRUCTIONSELECTOR_H
