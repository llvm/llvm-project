//===-- llvm/MC/MCInstrDesc.h - Instruction Descriptors -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MCOperandInfo and MCInstrDesc classes, which
// are used to describe target instructions and their operands.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCINSTRDESC_H
#define LLVM_MC_MCINSTRDESC_H

#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/DataTypes.h"
#include <string>

namespace llvm {
  class MCInst;
  class MCSubtargetInfo;
  class FeatureBitset;

//===----------------------------------------------------------------------===//
// Machine Operand Flags and Description
//===----------------------------------------------------------------------===//

namespace MCOI {
// Operand constraints
enum OperandConstraint {
  TIED_TO = 0,  // Must be allocated the same register as.
  EARLY_CLOBBER // Operand is an early clobber register operand
};

/// \brief These are flags set on operands, but should be considered
/// private, all access should go through the MCOperandInfo accessors.
/// See the accessors for a description of what these are.
enum OperandFlags { LookupPtrRegClass = 0, Predicate, OptionalDef };

/// \brief Operands are tagged with one of the values of this enum.
enum OperandType {
  OPERAND_UNKNOWN = 0,
  OPERAND_IMMEDIATE = 1,
  OPERAND_REGISTER = 2,
  OPERAND_MEMORY = 3,
  OPERAND_PCREL = 4,

  OPERAND_FIRST_GENERIC = 6,
  OPERAND_GENERIC_0 = 6,
  OPERAND_GENERIC_1 = 7,
  OPERAND_GENERIC_2 = 8,
  OPERAND_GENERIC_3 = 9,
  OPERAND_GENERIC_4 = 10,
  OPERAND_GENERIC_5 = 11,
  OPERAND_LAST_GENERIC = 11,

  OPERAND_FIRST_TARGET = 12,
};

enum GenericOperandType {
};

}

/// \brief This holds information about one operand of a machine instruction,
/// indicating the register class for register operands, etc.
class MCOperandInfo {
public:
  /// \brief This specifies the register class enumeration of the operand
  /// if the operand is a register.  If isLookupPtrRegClass is set, then this is
  /// an index that is passed to TargetRegisterInfo::getPointerRegClass(x) to
  /// get a dynamic register class.
  int16_t RegClass;

  /// \brief These are flags from the MCOI::OperandFlags enum.
  uint8_t Flags;

  /// \brief Information about the type of the operand.
  uint8_t OperandType;
  /// \brief The lower 16 bits are used to specify which constraints are set.
  /// The higher 16 bits are used to specify the value of constraints (4 bits
  /// each).
  uint32_t Constraints;

  /// \brief Set if this operand is a pointer value and it requires a callback
  /// to look up its register class.
  bool isLookupPtrRegClass() const {
    return Flags & (1 << MCOI::LookupPtrRegClass);
  }

  /// \brief Set if this is one of the operands that made up of the predicate
  /// operand that controls an isPredicable() instruction.
  bool isPredicate() const { return Flags & (1 << MCOI::Predicate); }

  /// \brief Set if this operand is a optional def.
  bool isOptionalDef() const { return Flags & (1 << MCOI::OptionalDef); }

  bool isGenericType() const {
    return OperandType >= MCOI::OPERAND_FIRST_GENERIC &&
           OperandType <= MCOI::OPERAND_LAST_GENERIC;
  }

  unsigned getGenericTypeIndex() const {
    assert(isGenericType() && "non-generic types don't have an index");
    return OperandType - MCOI::OPERAND_FIRST_GENERIC;
  }
};

//===----------------------------------------------------------------------===//
// Machine Instruction Flags and Description
//===----------------------------------------------------------------------===//

namespace MCID {
/// \brief These should be considered private to the implementation of the
/// MCInstrDesc class.  Clients should use the predicate methods on MCInstrDesc,
/// not use these directly.  These all correspond to bitfields in the
/// MCInstrDesc::Flags field.
enum Flag {
  Variadic = 0,
  HasOptionalDef,
  Pseudo,
  Return,
  Call,
  Barrier,
  Terminator,
  Branch,
  IndirectBranch,
  Compare,
  MoveImm,
  Bitcast,
  Select,
  DelaySlot,
  FoldableAsLoad,
  MayLoad,
  MayStore,
  Predicable,
  NotDuplicable,
  UnmodeledSideEffects,
  Commutable,
  ConvertibleTo3Addr,
  UsesCustomInserter,
  HasPostISelHook,
  Rematerializable,
  CheapAsAMove,
  ExtraSrcRegAllocReq,
  ExtraDefRegAllocReq,
  RegSequence,
  ExtractSubreg,
  InsertSubreg,
  Convergent,
  Add
};
}

/// \brief Describe properties that are true of each instruction in the target
/// description file.  This captures information about side effects, register
/// use and many other things.  There is one instance of this struct for each
/// target instruction class, and the MachineInstr class points to this struct
/// directly to describe itself.
class MCInstrDesc {
public:
  unsigned short Opcode;         // The opcode number
  unsigned short NumOperands;    // Num of args (may be more if variable_ops)
  unsigned char NumDefs;         // Num of args that are definitions
  unsigned char Size;            // Number of bytes in encoding.
  unsigned short SchedClass;     // enum identifying instr sched class
  uint64_t Flags;                // Flags identifying machine instr class
  uint64_t TSFlags;              // Target Specific Flag values
  const MCPhysReg *ImplicitUses; // Registers implicitly read by this instr
  const MCPhysReg *ImplicitDefs; // Registers implicitly defined by this instr
  const MCOperandInfo *OpInfo;   // 'NumOperands' entries about operands
  // Subtarget feature that this is deprecated on, if any
  // -1 implies this is not deprecated by any single feature. It may still be 
  // deprecated due to a "complex" reason, below.
  int64_t DeprecatedFeature;

  // A complex method to determine is a certain is deprecated or not, and return
  // the reason for deprecation.
  bool (*ComplexDeprecationInfo)(MCInst &, const MCSubtargetInfo &,
                                 std::string &);

  /// \brief Returns the value of the specific constraint if
  /// it is set. Returns -1 if it is not set.
  int getOperandConstraint(unsigned OpNum,
                           MCOI::OperandConstraint Constraint) const {
    if (OpNum < NumOperands &&
        (OpInfo[OpNum].Constraints & (1 << Constraint))) {
      unsigned Pos = 16 + Constraint * 4;
      return (int)(OpInfo[OpNum].Constraints >> Pos) & 0xf;
    }
    return -1;
  }

  /// \brief Returns true if a certain instruction is deprecated and if so
  /// returns the reason in \p Info.
  bool getDeprecatedInfo(MCInst &MI, const MCSubtargetInfo &STI,
                         std::string &Info) const;

  /// \brief Return the opcode number for this descriptor.
  unsigned getOpcode() const { return Opcode; }

  /// \brief Return the number of declared MachineOperands for this
  /// MachineInstruction.  Note that variadic (isVariadic() returns true)
  /// instructions may have additional operands at the end of the list, and note
  /// that the machine instruction may include implicit register def/uses as
  /// well.
  unsigned getNumOperands() const { return NumOperands; }

  /// \brief Return the number of MachineOperands that are register
  /// definitions.  Register definitions always occur at the start of the
  /// machine operand list.  This is the number of "outs" in the .td file,
  /// and does not include implicit defs.
  unsigned getNumDefs() const { return NumDefs; }

  /// \brief Return flags of this instruction.
  uint64_t getFlags() const { return Flags; }

  /// \brief Return true if this instruction can have a variable number of
  /// operands.  In this case, the variable operands will be after the normal
  /// operands but before the implicit definitions and uses (if any are
  /// present).
  bool isVariadic() const { return Flags & (1ULL << MCID::Variadic); }

  /// \brief Set if this instruction has an optional definition, e.g.
  /// ARM instructions which can set condition code if 's' bit is set.
  bool hasOptionalDef() const { return Flags & (1ULL << MCID::HasOptionalDef); }

  /// \brief Return true if this is a pseudo instruction that doesn't
  /// correspond to a real machine instruction.
  bool isPseudo() const { return Flags & (1ULL << MCID::Pseudo); }

  /// \brief Return true if the instruction is a return.
  bool isReturn() const { return Flags & (1ULL << MCID::Return); }

  /// \brief Return true if the instruction is an add instruction.
  bool isAdd() const { return Flags & (1ULL << MCID::Add); }

  /// \brief  Return true if the instruction is a call.
  bool isCall() const { return Flags & (1ULL << MCID::Call); }

  /// \brief Returns true if the specified instruction stops control flow
  /// from executing the instruction immediately following it.  Examples include
  /// unconditional branches and return instructions.
  bool isBarrier() const { return Flags & (1ULL << MCID::Barrier); }

  /// \brief Returns true if this instruction part of the terminator for
  /// a basic block.  Typically this is things like return and branch
  /// instructions.
  ///
  /// Various passes use this to insert code into the bottom of a basic block,
  /// but before control flow occurs.
  bool isTerminator() const { return Flags & (1ULL << MCID::Terminator); }

  /// \brief Returns true if this is a conditional, unconditional, or
  /// indirect branch.  Predicates below can be used to discriminate between
  /// these cases, and the TargetInstrInfo::AnalyzeBranch method can be used to
  /// get more information.
  bool isBranch() const { return Flags & (1ULL << MCID::Branch); }

  /// \brief Return true if this is an indirect branch, such as a
  /// branch through a register.
  bool isIndirectBranch() const { return Flags & (1ULL << MCID::IndirectBranch); }

  /// \brief Return true if this is a branch which may fall
  /// through to the next instruction or may transfer control flow to some other
  /// block.  The TargetInstrInfo::AnalyzeBranch method can be used to get more
  /// information about this branch.
  bool isConditionalBranch() const {
    return isBranch() & !isBarrier() & !isIndirectBranch();
  }

  /// \brief Return true if this is a branch which always
  /// transfers control flow to some other block.  The
  /// TargetInstrInfo::AnalyzeBranch method can be used to get more information
  /// about this branch.
  bool isUnconditionalBranch() const {
    return isBranch() & isBarrier() & !isIndirectBranch();
  }

  /// \brief Return true if this is a branch or an instruction which directly
  /// writes to the program counter. Considered 'may' affect rather than
  /// 'does' affect as things like predication are not taken into account.
  bool mayAffectControlFlow(const MCInst &MI, const MCRegisterInfo &RI) const;

  /// \brief Return true if this instruction has a predicate operand
  /// that controls execution. It may be set to 'always', or may be set to other
  /// values. There are various methods in TargetInstrInfo that can be used to
  /// control and modify the predicate in this instruction.
  bool isPredicable() const { return Flags & (1ULL << MCID::Predicable); }

  /// \brief Return true if this instruction is a comparison.
  bool isCompare() const { return Flags & (1ULL << MCID::Compare); }

  /// \brief Return true if this is a select instruction.
  bool isSelect() const { return Flags & (1ULL << MCID::Select); }

  /// \brief Returns true if the specified instruction has a delay slot which
  /// must be filled by the code generator.
  bool hasDelaySlot() const { return Flags & (1ULL << MCID::DelaySlot); }

  //===--------------------------------------------------------------------===//
  // Side Effect Analysis
  //===--------------------------------------------------------------------===//

  /// \brief Return true if this instruction could possibly read memory.
  /// Instructions with this flag set are not necessarily simple load
  /// instructions, they may load a value and modify it, for example.
  bool mayLoad() const { return Flags & (1ULL << MCID::MayLoad); }

  /// \brief Return true if this instruction could possibly modify memory.
  /// Instructions with this flag set are not necessarily simple store
  /// instructions, they may store a modified value based on their operands, or
  /// may not actually modify anything, for example.
  bool mayStore() const { return Flags & (1ULL << MCID::MayStore); }

  //===--------------------------------------------------------------------===//
  // Flags that indicate whether an instruction can be modified by a method.
  //===--------------------------------------------------------------------===//

  /// \brief Return true if this may be a 2- or 3-address instruction (of the
  /// form "X = op Y, Z, ..."), which produces the same result if Y and Z are
  /// exchanged.  If this flag is set, then the
  /// TargetInstrInfo::commuteInstruction method may be used to hack on the
  /// instruction.
  ///
  /// Note that this flag may be set on instructions that are only commutable
  /// sometimes.  In these cases, the call to commuteInstruction will fail.
  /// Also note that some instructions require non-trivial modification to
  /// commute them.
  bool isCommutable() const { return Flags & (1ULL << MCID::Commutable); }

  /// \brief Return true if this instruction requires *adjustment* after
  /// instruction selection by calling a target hook. For example, this can be
  /// used to fill in ARM 's' optional operand depending on whether the
  /// conditional flag register is used.
  bool hasPostISelHook() const { return Flags & (1ULL << MCID::HasPostISelHook); }

  /// \brief Returns true if this instruction is a candidate for remat. This
  /// flag is only used in TargetInstrInfo method isTriviallyRematerializable.
  ///
  /// If this flag is set, the isReallyTriviallyReMaterializable()
  /// or isReallyTriviallyReMaterializableGeneric methods are called to verify
  /// the instruction is really rematable.
  bool isRematerializable() const {
    return Flags & (1ULL << MCID::Rematerializable);
  }

  /// \brief Return a list of registers that are potentially read by any
  /// instance of this machine instruction.  For example, on X86, the "adc"
  /// instruction adds two register operands and adds the carry bit in from the
  /// flags register.  In this case, the instruction is marked as implicitly
  /// reading the flags.  Likewise, the variable shift instruction on X86 is
  /// marked as implicitly reading the 'CL' register, which it always does.
  ///
  /// This method returns null if the instruction has no implicit uses.
  const MCPhysReg *getImplicitUses() const { return ImplicitUses; }

  /// \brief Return the number of implicit uses this instruction has.
  unsigned getNumImplicitUses() const {
    if (!ImplicitUses)
      return 0;
    unsigned i = 0;
    for (; ImplicitUses[i]; ++i) /*empty*/
      ;
    return i;
  }

  /// \brief Return a list of registers that are potentially written by any
  /// instance of this machine instruction.  For example, on X86, many
  /// instructions implicitly set the flags register.  In this case, they are
  /// marked as setting the FLAGS.  Likewise, many instructions always deposit
  /// their result in a physical register.  For example, the X86 divide
  /// instruction always deposits the quotient and remainder in the EAX/EDX
  /// registers.  For that instruction, this will return a list containing the
  /// EAX/EDX/EFLAGS registers.
  ///
  /// This method returns null if the instruction has no implicit defs.
  const MCPhysReg *getImplicitDefs() const { return ImplicitDefs; }

  /// \brief Return the number of implicit defs this instruct has.
  unsigned getNumImplicitDefs() const {
    if (!ImplicitDefs)
      return 0;
    unsigned i = 0;
    for (; ImplicitDefs[i]; ++i) /*empty*/
      ;
    return i;
  }

  /// \brief Return true if this instruction implicitly
  /// defines the specified physical register.
  bool hasImplicitDefOfPhysReg(unsigned Reg,
                               const MCRegisterInfo *MRI = nullptr) const;

  /// \brief Return the scheduling class for this instruction.  The
  /// scheduling class is an index into the InstrItineraryData table.  This
  /// returns zero if there is no known scheduling information for the
  /// instruction.
  unsigned getSchedClass() const { return SchedClass; }

  /// \brief Return the number of bytes in the encoding of this instruction,
  /// or zero if the encoding size cannot be known from the opcode.
  unsigned getSize() const { return Size; }

  /// \brief Find the index of the first operand in the
  /// operand list that is used to represent the predicate. It returns -1 if
  /// none is found.
  int findFirstPredOperandIdx() const {
    if (isPredicable()) {
      for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
        if (OpInfo[i].isPredicate())
          return i;
    }
    return -1;
  }

private:

  /// \brief Return true if this instruction defines the specified physical
  /// register, either explicitly or implicitly.
  bool hasDefOfPhysReg(const MCInst &MI, unsigned Reg,
                       const MCRegisterInfo &RI) const;
};

} // end namespace llvm

#endif
