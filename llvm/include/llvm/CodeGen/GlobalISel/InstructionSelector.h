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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CodeGenCoverage.h"
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <vector>

namespace llvm {

class APInt;
class APFloat;
class LLT;
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
  /// Begin a try-block to attempt a match and jump to OnFail if it is
  /// unsuccessful.
  /// - OnFail - The MatchTable entry at which to resume if the match fails.
  ///
  /// FIXME: This ought to take an argument indicating the number of try-blocks
  ///        to exit on failure. It's usually one but the last match attempt of
  ///        a block will need more. The (implemented) alternative is to tack a
  ///        GIM_Reject on the end of each try-block which is simpler but
  ///        requires an extra opcode and iteration in the interpreter on each
  ///        failed match.
  GIM_Try,

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
  /// Check an immediate predicate on the specified instruction
  /// - InsnID - Instruction ID
  /// - The predicate to test
  GIM_CheckI64ImmPredicate,
  /// Check an immediate predicate on the specified instruction via an APInt.
  /// - InsnID - Instruction ID
  /// - The predicate to test
  GIM_CheckAPIntImmPredicate,
  /// Check a floating point immediate predicate on the specified instruction.
  /// - InsnID - Instruction ID
  /// - The predicate to test
  GIM_CheckAPFloatImmPredicate,
  /// Check a memory operation has the specified atomic ordering.
  /// - InsnID - Instruction ID
  /// - Ordering - The AtomicOrdering value
  GIM_CheckAtomicOrdering,
  GIM_CheckAtomicOrderingOrStrongerThan,
  GIM_CheckAtomicOrderingWeakerThan,

  /// Check the type for the specified operand
  /// - InsnID - Instruction ID
  /// - OpIdx - Operand index
  /// - Expected type
  GIM_CheckType,
  /// Check the type of a pointer to any address space.
  /// - InsnID - Instruction ID
  /// - OpIdx - Operand index
  /// - SizeInBits - The size of the pointer value in bits.
  GIM_CheckPointerToAny,
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

  /// Check the specified operands are identical.
  /// - InsnID - Instruction ID
  /// - OpIdx - Operand index
  /// - OtherInsnID - Other instruction ID
  /// - OtherOpIdx - Other operand index
  GIM_CheckIsSameOperand,

  /// Fail the current try-block, or completely fail to match if there is no
  /// current try-block.
  GIM_Reject,

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
  /// Copy an operand to the specified instruction or add a zero register if the
  /// operand is a zero immediate.
  /// - NewInsnID - Instruction ID to modify
  /// - OldInsnID - Instruction ID to copy from
  /// - OpIdx - The operand to copy
  /// - ZeroReg - The zero register to use
  GIR_CopyOrAddZeroReg,
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
  /// Add a a temporary register to the specified instruction
  /// - InsnID - Instruction ID to modify
  /// - TempRegID - The temporary register ID to add
  GIR_AddTempRegister,
  /// Add an immediate to the specified instruction
  /// - InsnID - Instruction ID to modify
  /// - Imm - The immediate to add
  GIR_AddImm,
  /// Render complex operands to the specified instruction
  /// - InsnID - Instruction ID to modify
  /// - RendererID - The renderer to call
  GIR_ComplexRenderer,
  /// Render sub-operands of complex operands to the specified instruction
  /// - InsnID - Instruction ID to modify
  /// - RendererID - The renderer to call
  /// - RenderOpID - The suboperand to render.
  GIR_ComplexSubOperandRenderer,

  /// Render a G_CONSTANT operator as a sign-extended immediate.
  /// - NewInsnID - Instruction ID to modify
  /// - OldInsnID - Instruction ID to copy from
  /// The operand index is implicitly 1.
  GIR_CopyConstantAsSImm,

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
  /// - MergeInsnID... - One or more Instruction ID to merge into the result.
  /// - GIU_MergeMemOperands_EndOfList - Terminates the list of instructions to
  ///                                    merge.
  GIR_MergeMemOperands,
  /// Erase from parent.
  /// - InsnID - Instruction ID to erase
  GIR_EraseFromParent,
  /// Create a new temporary register that's not constrained.
  /// - TempRegID - The temporary register ID to initialize.
  /// - Expected type
  GIR_MakeTempReg,

  /// A successful emission
  GIR_Done,

  /// Increment the rule coverage counter.
  /// - RuleID - The ID of the rule that was covered.
  GIR_Coverage,
};

enum {
  /// Indicates the end of the variable-length MergeInsnID list in a
  /// GIR_MergeMemOperands opcode.
  GIU_MergeMemOperands_EndOfList = -1,
};

/// Provides the logic to select generic machine instructions.
class InstructionSelector {
public:
  using I64ImmediatePredicateFn = bool (*)(int64_t);
  using APIntImmediatePredicateFn = bool (*)(const APInt &);
  using APFloatImmediatePredicateFn = bool (*)(const APFloat &);

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
  virtual bool select(MachineInstr &I, CodeGenCoverage &CoverageInfo) const = 0;

protected:
  using ComplexRendererFns =
      Optional<SmallVector<std::function<void(MachineInstrBuilder &)>, 4>>;
  using RecordedMIVector = SmallVector<MachineInstr *, 4>;
  using NewMIVector = SmallVector<MachineInstrBuilder, 4>;

  struct MatcherState {
    std::vector<ComplexRendererFns::value_type> Renderers;
    RecordedMIVector MIs;
    DenseMap<unsigned, unsigned> TempRegisters;

    MatcherState(unsigned MaxRenderers);
  };

public:
  template <class PredicateBitset, class ComplexMatcherMemFn>
  struct MatcherInfoTy {
    const LLT *TypeObjects;
    const PredicateBitset *FeatureBitsets;
    const I64ImmediatePredicateFn *I64ImmPredicateFns;
    const APIntImmediatePredicateFn *APIntImmPredicateFns;
    const APFloatImmediatePredicateFn *APFloatImmPredicateFns;
    const ComplexMatcherMemFn *ComplexPredicates;
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
      const RegisterBankInfo &RBI, const PredicateBitset &AvailableFeatures,
      CodeGenCoverage &CoverageInfo) const;

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

  /// Return true if the specified operand is a G_GEP with a G_CONSTANT on the
  /// right-hand side. GlobalISel's separation of pointer and integer types
  /// means that we don't need to worry about G_OR with equivalent semantics.
  bool isBaseWithConstantOffset(const MachineOperand &Root,
                                const MachineRegisterInfo &MRI) const;

  /// Return true if MI can obviously be folded into IntoMI.
  /// MI and IntoMI do not need to be in the same basic blocks, but MI must
  /// preceed IntoMI.
  bool isObviouslySafeToFold(MachineInstr &MI, MachineInstr &IntoMI) const;
};

} // end namespace llvm

#endif // LLVM_CODEGEN_GLOBALISEL_INSTRUCTIONSELECTOR_H
