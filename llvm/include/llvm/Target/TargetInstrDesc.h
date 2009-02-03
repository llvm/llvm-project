//===-- llvm/Target/TargetInstrDesc.h - Instruction Descriptors -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the TargetOperandInfo and TargetInstrDesc classes, which
// are used to describe target instructions and their operands. 
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETINSTRDESC_H
#define LLVM_TARGET_TARGETINSTRDESC_H

namespace llvm {

class TargetRegisterClass;

//===----------------------------------------------------------------------===//
// Machine Operand Flags and Description
//===----------------------------------------------------------------------===//
  
namespace TOI {
  // Operand constraints: only "tied_to" for now.
  enum OperandConstraint {
    TIED_TO = 0  // Must be allocated the same register as.
  };
  
  /// OperandFlags - These are flags set on operands, but should be considered
  /// private, all access should go through the TargetOperandInfo accessors.
  /// See the accessors for a description of what these are.
  enum OperandFlags {
    LookupPtrRegClass = 0,
    Predicate,
    OptionalDef
  };
}

/// TargetOperandInfo - This holds information about one operand of a machine
/// instruction, indicating the register class for register operands, etc.
///
class TargetOperandInfo {
public:
  /// RegClass - This specifies the register class enumeration of the operand 
  /// if the operand is a register.  If not, this contains 0.
  unsigned short RegClass;
  unsigned short Flags;
  /// Lower 16 bits are used to specify which constraints are set. The higher 16
  /// bits are used to specify the value of constraints (4 bits each).
  unsigned int Constraints;
  /// Currently no other information.
  
  /// isLookupPtrRegClass - Set if this operand is a pointer value and it
  /// requires a callback to look up its register class.
  bool isLookupPtrRegClass() const { return Flags&(1 <<TOI::LookupPtrRegClass);}
  
  /// isPredicate - Set if this is one of the operands that made up of
  /// the predicate operand that controls an isPredicable() instruction.
  bool isPredicate() const { return Flags & (1 << TOI::Predicate); }
  
  /// isOptionalDef - Set if this operand is a optional def.
  ///
  bool isOptionalDef() const { return Flags & (1 << TOI::OptionalDef); }
};

  
//===----------------------------------------------------------------------===//
// Machine Instruction Flags and Description
//===----------------------------------------------------------------------===//

/// TargetInstrDesc flags - These should be considered private to the
/// implementation of the TargetInstrDesc class.  Clients should use the
/// predicate methods on TargetInstrDesc, not use these directly.  These
/// all correspond to bitfields in the TargetInstrDesc::Flags field.
namespace TID {
  enum {
    Variadic = 0,
    HasOptionalDef,
    Return,
    Call,
    Barrier,
    Terminator,
    Branch,
    IndirectBranch,
    Predicable,
    NotDuplicable,
    DelaySlot,
    FoldableAsLoad,
    MayLoad,
    MayStore,
    UnmodeledSideEffects,
    Commutable,
    ConvertibleTo3Addr,
    UsesCustomDAGSchedInserter,
    Rematerializable,
    CheapAsAMove
  };
}

/// TargetInstrDesc - Describe properties that are true of each
/// instruction in the target description file.  This captures information about
/// side effects, register use and many other things.  There is one instance of
/// this struct for each target instruction class, and the MachineInstr class
/// points to this struct directly to describe itself.
class TargetInstrDesc {
public:
  unsigned short  Opcode;        // The opcode number
  unsigned short  NumOperands;   // Num of args (may be more if variable_ops)
  unsigned short  NumDefs;       // Num of args that are definitions
  unsigned short  SchedClass;    // enum identifying instr sched class
  const char *    Name;          // Name of the instruction record in td file
  unsigned        Flags;         // Flags identifying machine instr class
  unsigned        TSFlags;       // Target Specific Flag values
  const unsigned *ImplicitUses;  // Registers implicitly read by this instr
  const unsigned *ImplicitDefs;  // Registers implicitly defined by this instr
  const TargetRegisterClass **RCBarriers; // Reg classes completely "clobbered"
  const TargetOperandInfo *OpInfo; // 'NumOperands' entries about operands

  /// getOperandConstraint - Returns the value of the specific constraint if
  /// it is set. Returns -1 if it is not set.
  int getOperandConstraint(unsigned OpNum,
                           TOI::OperandConstraint Constraint) const {
    if (OpNum < NumOperands &&
        (OpInfo[OpNum].Constraints & (1 << Constraint))) {
      unsigned Pos = 16 + Constraint * 4;
      return (int)(OpInfo[OpNum].Constraints >> Pos) & 0xf;
    }
    return -1;
  }

  /// findTiedToSrcOperand - Returns the operand that is tied to the specified
  /// dest operand. Returns -1 if there isn't one.
  int findTiedToSrcOperand(unsigned OpNum) const;
  
  /// getOpcode - Return the opcode number for this descriptor.
  unsigned getOpcode() const {
    return Opcode;
  }
  
  /// getName - Return the name of the record in the .td file for this
  /// instruction, for example "ADD8ri".
  const char *getName() const {
    return Name;
  }
  
  /// getNumOperands - Return the number of declared MachineOperands for this
  /// MachineInstruction.  Note that variadic (isVariadic() returns true)
  /// instructions may have additional operands at the end of the list, and note
  /// that the machine instruction may include implicit register def/uses as
  /// well.
  unsigned getNumOperands() const {
    return NumOperands;
  }
  
  /// getNumDefs - Return the number of MachineOperands that are register
  /// definitions.  Register definitions always occur at the start of the 
  /// machine operand list.  This is the number of "outs" in the .td file,
  /// and does not include implicit defs.
  unsigned getNumDefs() const {
    return NumDefs;
  }
  
  /// isVariadic - Return true if this instruction can have a variable number of
  /// operands.  In this case, the variable operands will be after the normal
  /// operands but before the implicit definitions and uses (if any are
  /// present).
  bool isVariadic() const {
    return Flags & (1 << TID::Variadic);
  }
  
  /// hasOptionalDef - Set if this instruction has an optional definition, e.g.
  /// ARM instructions which can set condition code if 's' bit is set.
  bool hasOptionalDef() const {
    return Flags & (1 << TID::HasOptionalDef);
  }
  
  /// getImplicitUses - Return a list of registers that are potentially
  /// read by any instance of this machine instruction.  For example, on X86,
  /// the "adc" instruction adds two register operands and adds the carry bit in
  /// from the flags register.  In this case, the instruction is marked as
  /// implicitly reading the flags.  Likewise, the variable shift instruction on
  /// X86 is marked as implicitly reading the 'CL' register, which it always
  /// does.
  ///
  /// This method returns null if the instruction has no implicit uses.
  const unsigned *getImplicitUses() const {
    return ImplicitUses;
  }
  
  /// getImplicitDefs - Return a list of registers that are potentially
  /// written by any instance of this machine instruction.  For example, on X86,
  /// many instructions implicitly set the flags register.  In this case, they
  /// are marked as setting the FLAGS.  Likewise, many instructions always
  /// deposit their result in a physical register.  For example, the X86 divide
  /// instruction always deposits the quotient and remainder in the EAX/EDX
  /// registers.  For that instruction, this will return a list containing the
  /// EAX/EDX/EFLAGS registers.
  ///
  /// This method returns null if the instruction has no implicit defs.
  const unsigned *getImplicitDefs() const {
    return ImplicitDefs;
  }

  /// getRegClassBarriers - Return a list of register classes that are
  /// completely clobbered by this machine instruction. For example, on X86
  /// the call instructions will completely clobber all the registers in the
  /// fp stack and XMM classes.
  ///
  /// This method returns null if the instruction doesn't completely clobber
  /// any register class.
  const TargetRegisterClass **getRegClassBarriers() const {
    return RCBarriers;
  }

  /// getSchedClass - Return the scheduling class for this instruction.  The
  /// scheduling class is an index into the InstrItineraryData table.  This
  /// returns zero if there is no known scheduling information for the
  /// instruction.
  ///
  unsigned getSchedClass() const {
    return SchedClass;
  }
  
  bool isReturn() const {
    return Flags & (1 << TID::Return);
  }
  
  bool isCall() const {
    return Flags & (1 << TID::Call);
  }
  
  /// isBarrier - Returns true if the specified instruction stops control flow
  /// from executing the instruction immediately following it.  Examples include
  /// unconditional branches and return instructions.
  bool isBarrier() const {
    return Flags & (1 << TID::Barrier);
  }
  
  /// isTerminator - Returns true if this instruction part of the terminator for
  /// a basic block.  Typically this is things like return and branch
  /// instructions.
  ///
  /// Various passes use this to insert code into the bottom of a basic block,
  /// but before control flow occurs.
  bool isTerminator() const {
    return Flags & (1 << TID::Terminator);
  }
  
  /// isBranch - Returns true if this is a conditional, unconditional, or
  /// indirect branch.  Predicates below can be used to discriminate between
  /// these cases, and the TargetInstrInfo::AnalyzeBranch method can be used to
  /// get more information.
  bool isBranch() const {
    return Flags & (1 << TID::Branch);
  }

  /// isIndirectBranch - Return true if this is an indirect branch, such as a
  /// branch through a register.
  bool isIndirectBranch() const {
    return Flags & (1 << TID::IndirectBranch);
  }
  
  /// isConditionalBranch - Return true if this is a branch which may fall
  /// through to the next instruction or may transfer control flow to some other
  /// block.  The TargetInstrInfo::AnalyzeBranch method can be used to get more
  /// information about this branch.
  bool isConditionalBranch() const {
    return isBranch() & !isBarrier() & !isIndirectBranch();
  }
  
  /// isUnconditionalBranch - Return true if this is a branch which always
  /// transfers control flow to some other block.  The
  /// TargetInstrInfo::AnalyzeBranch method can be used to get more information
  /// about this branch.
  bool isUnconditionalBranch() const {
    return isBranch() & isBarrier() & !isIndirectBranch();
  }
  
  // isPredicable - Return true if this instruction has a predicate operand that
  // controls execution.  It may be set to 'always', or may be set to other
  /// values.   There are various methods in TargetInstrInfo that can be used to
  /// control and modify the predicate in this instruction.
  bool isPredicable() const {
    return Flags & (1 << TID::Predicable);
  }
  
  /// isNotDuplicable - Return true if this instruction cannot be safely
  /// duplicated.  For example, if the instruction has a unique labels attached
  /// to it, duplicating it would cause multiple definition errors.
  bool isNotDuplicable() const {
    return Flags & (1 << TID::NotDuplicable);
  }
  
  /// hasDelaySlot - Returns true if the specified instruction has a delay slot
  /// which must be filled by the code generator.
  bool hasDelaySlot() const {
    return Flags & (1 << TID::DelaySlot);
  }
  
  /// canFoldAsLoad - Return true for instructions that can be folded as
  /// memory operands in other instructions. The most common use for this
  /// is instructions that are simple loads from memory that don't modify
  /// the loaded value in any way, but it can also be used for instructions
  /// that can be expressed as constant-pool loads, such as V_SETALLONES
  /// on x86, to allow them to be folded when it is beneficial.
  /// This should only be set on instructions that return a value in their
  /// only virtual register definition.
  bool canFoldAsLoad() const {
    return Flags & (1 << TID::FoldableAsLoad);
  }
  
  //===--------------------------------------------------------------------===//
  // Side Effect Analysis
  //===--------------------------------------------------------------------===//

  /// mayLoad - Return true if this instruction could possibly read memory.
  /// Instructions with this flag set are not necessarily simple load
  /// instructions, they may load a value and modify it, for example.
  bool mayLoad() const {
    return Flags & (1 << TID::MayLoad);
  }
  
  
  /// mayStore - Return true if this instruction could possibly modify memory.
  /// Instructions with this flag set are not necessarily simple store
  /// instructions, they may store a modified value based on their operands, or
  /// may not actually modify anything, for example.
  bool mayStore() const {
    return Flags & (1 << TID::MayStore);
  }
  
  /// hasUnmodeledSideEffects - Return true if this instruction has side
  /// effects that are not modeled by other flags.  This does not return true
  /// for instructions whose effects are captured by:
  ///
  ///  1. Their operand list and implicit definition/use list.  Register use/def
  ///     info is explicit for instructions.
  ///  2. Memory accesses.  Use mayLoad/mayStore.
  ///  3. Calling, branching, returning: use isCall/isReturn/isBranch.
  ///
  /// Examples of side effects would be modifying 'invisible' machine state like
  /// a control register, flushing a cache, modifying a register invisible to
  /// LLVM, etc.
  ///
  bool hasUnmodeledSideEffects() const {
    return Flags & (1 << TID::UnmodeledSideEffects);
  }
  
  //===--------------------------------------------------------------------===//
  // Flags that indicate whether an instruction can be modified by a method.
  //===--------------------------------------------------------------------===//
  
  /// isCommutable - Return true if this may be a 2- or 3-address
  /// instruction (of the form "X = op Y, Z, ..."), which produces the same
  /// result if Y and Z are exchanged.  If this flag is set, then the 
  /// TargetInstrInfo::commuteInstruction method may be used to hack on the
  /// instruction.
  ///
  /// Note that this flag may be set on instructions that are only commutable
  /// sometimes.  In these cases, the call to commuteInstruction will fail.
  /// Also note that some instructions require non-trivial modification to
  /// commute them.
  bool isCommutable() const {
    return Flags & (1 << TID::Commutable);
  }
  
  /// isConvertibleTo3Addr - Return true if this is a 2-address instruction
  /// which can be changed into a 3-address instruction if needed.  Doing this
  /// transformation can be profitable in the register allocator, because it
  /// means that the instruction can use a 2-address form if possible, but
  /// degrade into a less efficient form if the source and dest register cannot
  /// be assigned to the same register.  For example, this allows the x86
  /// backend to turn a "shl reg, 3" instruction into an LEA instruction, which
  /// is the same speed as the shift but has bigger code size.
  ///
  /// If this returns true, then the target must implement the
  /// TargetInstrInfo::convertToThreeAddress method for this instruction, which
  /// is allowed to fail if the transformation isn't valid for this specific
  /// instruction (e.g. shl reg, 4 on x86).
  ///
  bool isConvertibleTo3Addr() const {
    return Flags & (1 << TID::ConvertibleTo3Addr);
  }
  
  /// usesCustomDAGSchedInsertionHook - Return true if this instruction requires
  /// custom insertion support when the DAG scheduler is inserting it into a
  /// machine basic block.  If this is true for the instruction, it basically
  /// means that it is a pseudo instruction used at SelectionDAG time that is 
  /// expanded out into magic code by the target when MachineInstrs are formed.
  ///
  /// If this is true, the TargetLoweringInfo::InsertAtEndOfBasicBlock method
  /// is used to insert this into the MachineBasicBlock.
  bool usesCustomDAGSchedInsertionHook() const {
    return Flags & (1 << TID::UsesCustomDAGSchedInserter);
  }
  
  /// isRematerializable - Returns true if this instruction is a candidate for
  /// remat.  This flag is deprecated, please don't use it anymore.  If this
  /// flag is set, the isReallyTriviallyReMaterializable() method is called to
  /// verify the instruction is really rematable.
  bool isRematerializable() const {
    return Flags & (1 << TID::Rematerializable);
  }

  /// isAsCheapAsAMove - Returns true if this instruction has the same cost (or
  /// less) than a move instruction. This is useful during certain types of
  /// rematerializations (e.g., during two-address conversion) where we would
  /// like to remat the instruction, but not if it costs more than moving the
  /// instruction into the appropriate register.
  bool isAsCheapAsAMove() const {
    return Flags & (1 << TID::CheapAsAMove);
  }
};

} // end namespace llvm

#endif
