//===- TargetInstrPredicate.td - ---------------------------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines MCInstPredicate classes and its subclasses.
//
// MCInstPredicate is used to describe constraints on the opcode/operand(s) of
// an instruction. Each MCInstPredicate class has a well-known semantic, and it
// is used by a PredicateExpander to generate code for MachineInstr and/or
// MCInst.
//
// MCInstPredicate definitions can be used to construct MCSchedPredicate
// definitions. An MCSchedPredicate can be used in place of a SchedPredicate
// when defining SchedReadVariant and SchedWriteVariant used by a processor
// scheduling model.
//
// Here is an example of MCInstPredicate definition:
//
// def MCInstPredicateExample : CheckAll<[
//    CheckOpcode<[BLR]>,
//    CheckIsRegOperand<0>,
//    CheckNot<CheckRegOperand<0, LR>>]>;
//
// Predicate `MCInstPredicateExample` checks that the machine instruction in
// input is a BLR, and that operand at index 0 is register `LR`.
//
// That predicate could be used to rewrite the following definition (from
// AArch64SchedExynosM3.td):
//
// def M3BranchLinkFastPred  : SchedPredicate<[{
//    MI->getOpcode() == AArch64::BLR &&
//    MI->getOperand(0).isReg() &&
//    MI->getOperand(0).getReg() != AArch64::LR}]>;
//
// MCInstPredicate definitions are used to construct MCSchedPredicate (see the
// definition of class MCSchedPredicate in llvm/Target/TargetSchedule.td).  An
// MCSchedPredicate can be used by a `SchedVar` to associate a predicate with a
// list of SchedReadWrites. Note that `SchedVar` are used to create SchedVariant
// definitions.
//
// Each MCInstPredicate class has a well known semantic. For example,
// `CheckOpcode` is only used to check the instruction opcode value.
//
// MCInstPredicate classes allow the definition of predicates in a declarative
// way.  These predicates don't require a custom block of C++, and can be used
// to define conditions on instructions without being bound to a particular
// representation (i.e. MachineInstr vs MCInst).
//
// It also means that tablegen backends must know how to parse and expand them
// into code that works on MCInst (or MachineInst).
//
// Instances of class PredicateExpander (see utils/Tablegen/PredicateExpander.h)
// know how to expand a predicate. For each MCInstPredicate class, there must be
// an "expand" method available in the PredicateExpander interface.
//
// For example, a `CheckOpcode` predicate is expanded using method
// `PredicateExpander::expandCheckOpcode()`.
//
// New MCInstPredicate classes must be added to this file. For each new class
// XYZ, an "expandXYZ" method must be added to the PredicateExpander.
//
//===----------------------------------------------------------------------===//

// Forward declarations.
class Instruction;

// A generic machine instruction predicate.
class MCInstPredicate;

class MCTrue  : MCInstPredicate;   // A predicate that always evaluates to True.
class MCFalse : MCInstPredicate;   // A predicate that always evaluates to False.
def TruePred  : MCTrue;
def FalsePred : MCFalse;

// A predicate used to negate the outcome of another predicate.
// It allows to easily express "set difference" operations. For example, it
// makes it easy to describe a check that tests if an opcode is not part of a
// set of opcodes.
class CheckNot<MCInstPredicate P> : MCInstPredicate {
  MCInstPredicate Pred = P;
}

// This class is used as a building block to define predicates on instruction
// operands. It is used to reference a specific machine operand.
class MCOperandPredicate<int Index> : MCInstPredicate {
  int OpIndex = Index;
}

// Return true if machine operand at position `Index` is a register operand.
class CheckIsRegOperand<int Index> : MCOperandPredicate<Index>;

// Return true if machine operand at position `Index` is an immediate operand.
class CheckIsImmOperand<int Index> : MCOperandPredicate<Index>;

// Check if machine operands at index `First` and index `Second` both reference
// the same register.
class CheckSameRegOperand<int First, int Second> : MCInstPredicate {
  int FirstIndex = First;
  int SecondIndex = Second;
}

// Check that the machine register operand at position `Index` references
// register R. This predicate assumes that we already checked that the machine
// operand at position `Index` is a register operand.
class CheckRegOperand<int Index, Register R> : MCOperandPredicate<Index> {
  Register Reg = R;
}

// Check if register operand at index `Index` is the invalid register.
class CheckInvalidRegOperand<int Index> : MCOperandPredicate<Index>;

// Check that the operand at position `Index` is immediate `Imm`.
class CheckImmOperand<int Index, int Imm> : MCOperandPredicate<Index> {
  int ImmVal = Imm;
}

// Similar to CheckImmOperand, however the immediate is not a literal number.
// This is useful when we want to compare the value of an operand against an
// enum value, and we know the actual integer value of that enum.
class CheckImmOperand_s<int Index, string Value> : MCOperandPredicate<Index> {
  string ImmVal = Value;
}

// Check that the operand at position `Index` is immediate value zero.
class CheckZeroOperand<int Index> : CheckImmOperand<Index, 0>;

// Check that the instruction has exactly `Num` operands.
class CheckNumOperands<int Num> : MCInstPredicate {
  int NumOps = Num;
}

// Check that the instruction opcode is one of the opcodes in set `Opcodes`.
// This is a simple set membership query. The easier way to check if an opcode
// is not a member of the set is by using a `CheckNot<CheckOpcode<[...]>>`
// sequence.
class CheckOpcode<list<Instruction> Opcodes> : MCInstPredicate {
  list<Instruction> ValidOpcodes = Opcodes;
}

// Check that the instruction opcode is a pseudo opcode member of the set
// `Opcodes`.  This check is always expanded to "false" if we are generating
// code for MCInst.
class CheckPseudo<list<Instruction> Opcodes> : CheckOpcode<Opcodes>;

// A non-portable predicate. Only to use as a last resort when a block of code
// cannot possibly be converted in a declarative way using other MCInstPredicate
// classes. This check is always expanded to "false" when generating code for
// MCInst.
class CheckNonPortable<string Code> : MCInstPredicate {
  string CodeBlock = Code;
}

// A sequence of predicates. It is used as the base class for CheckAll, and
// CheckAny. It allows to describe compositions of predicates.
class CheckPredicateSequence<list<MCInstPredicate> Preds> : MCInstPredicate {
  list<MCInstPredicate> Predicates = Preds;
}

// Check that all of the predicates in `Preds` evaluate to true.
class CheckAll<list<MCInstPredicate> Sequence>
    : CheckPredicateSequence<Sequence>;

// Check that at least one of the predicates in `Preds` evaluates to true.
class CheckAny<list<MCInstPredicate> Sequence>
    : CheckPredicateSequence<Sequence>;

// Check that a call to method `Name` in class "XXXGenInstrInfo" (where XXX is
// the `Target` name) returns true.
//
// TIIPredicate definitions are used to model calls to the target-specific
// InstrInfo. A TIIPredicate is treated specially by the InstrInfoEmitter
// tablegen backend, which will use it to automatically generate a definition in
// the target specific `GenInstrInfo` class.
class TIIPredicate<string Target, string Name, MCInstPredicate P> : MCInstPredicate {
  string TargetName = Target;
  string FunctionName = Name;
  MCInstPredicate Pred = P;
}

// A function predicate that takes as input a machine instruction, and returns
// a boolean value.
//
// This predicate is expanded into a function call by the PredicateExpander.
// In particular, the PredicateExpander would either expand this predicate into
// a call to `MCInstFn`, or into a call to`MachineInstrFn` depending on whether
// it is lowering predicates for MCInst or MachineInstr.
//
// In this context, `MCInstFn` and `MachineInstrFn` are both function names.
class CheckFunctionPredicate<string MCInstFn, string MachineInstrFn> : MCInstPredicate {
  string MCInstFnName = MCInstFn;
  string MachineInstrFnName = MachineInstrFn;
}
