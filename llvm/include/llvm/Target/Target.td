//===- Target.td - Target Independent TableGen interface ---*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the target-independent interfaces which should be
// implemented by each target which is using a TableGen based code generator.
//
//===----------------------------------------------------------------------===//

// Include all information about LLVM intrinsics.
include "llvm/Intrinsics.td"

//===----------------------------------------------------------------------===//
// Register file description - These classes are used to fill in the target
// description classes.

class RegisterClass; // Forward def

// Register - You should define one instance of this class for each register
// in the target machine.  String n will become the "name" of the register.
class Register<string n> {
  string Namespace = "";
  string AsmName = n;

  // SpillSize - If this value is set to a non-zero value, it is the size in
  // bits of the spill slot required to hold this register.  If this value is
  // set to zero, the information is inferred from any register classes the
  // register belongs to.
  int SpillSize = 0;

  // SpillAlignment - This value is used to specify the alignment required for
  // spilling the register.  Like SpillSize, this should only be explicitly
  // specified if the register is not in a register class.
  int SpillAlignment = 0;

  // Aliases - A list of registers that this register overlaps with.  A read or
  // modification of this register can potentially read or modify the aliased
  // registers.
  list<Register> Aliases = [];
  
  // SubRegs - A list of registers that are parts of this register. Note these
  // are "immediate" sub-registers and the registers within the list do not
  // themselves overlap. e.g. For X86, EAX's SubRegs list contains only [AX],
  // not [AX, AH, AL].
  list<Register> SubRegs = [];

  // DwarfNumbers - Numbers used internally by gcc/gdb to identify the register.
  // These values can be determined by locating the <target>.h file in the
  // directory llvmgcc/gcc/config/<target>/ and looking for REGISTER_NAMES.  The
  // order of these names correspond to the enumeration used by gcc.  A value of
  // -1 indicates that the gcc number is undefined and -2 that register number
  // is invalid for this mode/flavour.
  list<int> DwarfNumbers = [];
}

// RegisterWithSubRegs - This can be used to define instances of Register which
// need to specify sub-registers.
// List "subregs" specifies which registers are sub-registers to this one. This
// is used to populate the SubRegs and AliasSet fields of TargetRegisterDesc.
// This allows the code generator to be careful not to put two values with 
// overlapping live ranges into registers which alias.
class RegisterWithSubRegs<string n, list<Register> subregs> : Register<n> {
  let SubRegs = subregs;
}

// SubRegSet - This can be used to define a specific mapping of registers to
// indices, for use as named subregs of a particular physical register.  Each
// register in 'subregs' becomes an addressable subregister at index 'n' of the
// corresponding register in 'regs'.
class SubRegSet<int n, list<Register> regs, list<Register> subregs> {
  int index = n;
  
  list<Register> From = regs;
  list<Register> To = subregs;
}

// RegisterClass - Now that all of the registers are defined, and aliases
// between registers are defined, specify which registers belong to which
// register classes.  This also defines the default allocation order of
// registers by register allocators.
//
class RegisterClass<string namespace, list<ValueType> regTypes, int alignment,
                    list<Register> regList> {
  string Namespace = namespace;

  // RegType - Specify the list ValueType of the registers in this register
  // class.  Note that all registers in a register class must have the same
  // ValueTypes.  This is a list because some targets permit storing different 
  // types in same register, for example vector values with 128-bit total size,
  // but different count/size of items, like SSE on x86.
  //
  list<ValueType> RegTypes = regTypes;

  // Size - Specify the spill size in bits of the registers.  A default value of
  // zero lets tablgen pick an appropriate size.
  int Size = 0;

  // Alignment - Specify the alignment required of the registers when they are
  // stored or loaded to memory.
  //
  int Alignment = alignment;

  // CopyCost - This value is used to specify the cost of copying a value
  // between two registers in this register class. The default value is one
  // meaning it takes a single instruction to perform the copying. A negative
  // value means copying is extremely expensive or impossible.
  int CopyCost = 1;

  // MemberList - Specify which registers are in this class.  If the
  // allocation_order_* method are not specified, this also defines the order of
  // allocation used by the register allocator.
  //
  list<Register> MemberList = regList;
  
  // SubClassList - Specify which register classes correspond to subregisters
  // of this class. The order should be by subregister set index.
  list<RegisterClass> SubRegClassList = [];

  // MethodProtos/MethodBodies - These members can be used to insert arbitrary
  // code into a generated register class.   The normal usage of this is to 
  // overload virtual methods.
  code MethodProtos = [{}];
  code MethodBodies = [{}];
}


//===----------------------------------------------------------------------===//
// DwarfRegNum - This class provides a mapping of the llvm register enumeration
// to the register numbering used by gcc and gdb.  These values are used by a
// debug information writer (ex. DwarfWriter) to describe where values may be
// located during execution.
class DwarfRegNum<list<int> Numbers> {
  // DwarfNumbers - Numbers used internally by gcc/gdb to identify the register.
  // These values can be determined by locating the <target>.h file in the
  // directory llvmgcc/gcc/config/<target>/ and looking for REGISTER_NAMES.  The
  // order of these names correspond to the enumeration used by gcc.  A value of
  // -1 indicates that the gcc number is undefined and -2 that register number is 
  // invalid for this mode/flavour.
  list<int> DwarfNumbers = Numbers;
}

//===----------------------------------------------------------------------===//
// Pull in the common support for scheduling
//
include "llvm/Target/TargetSchedule.td"

class Predicate; // Forward def

//===----------------------------------------------------------------------===//
// Instruction set description - These classes correspond to the C++ classes in
// the Target/TargetInstrInfo.h file.
//
class Instruction {
  string Namespace = "";

  dag OutOperandList;       // An dag containing the MI def operand list.
  dag InOperandList;        // An dag containing the MI use operand list.
  string AsmString = "";    // The .s format to print the instruction with.

  // Pattern - Set to the DAG pattern for this instruction, if we know of one,
  // otherwise, uninitialized.
  list<dag> Pattern;

  // The follow state will eventually be inferred automatically from the
  // instruction pattern.

  list<Register> Uses = []; // Default to using no non-operand registers
  list<Register> Defs = []; // Default to modifying no non-operand registers

  // Predicates - List of predicates which will be turned into isel matching
  // code.
  list<Predicate> Predicates = [];

  // Code size.
  int CodeSize = 0;

  // Added complexity passed onto matching pattern.
  int AddedComplexity  = 0;

  // These bits capture information about the high-level semantics of the
  // instruction.
  bit isReturn     = 0;     // Is this instruction a return instruction?
  bit isBranch     = 0;     // Is this instruction a branch instruction?
  bit isIndirectBranch = 0; // Is this instruction an indirect branch?
  bit isBarrier    = 0;     // Can control flow fall through this instruction?
  bit isCall       = 0;     // Is this instruction a call instruction?
  bit canFoldAsLoad = 0;    // Can this be folded as a simple memory operand?
  bit mayLoad      = 0;     // Is it possible for this inst to read memory?
  bit mayStore     = 0;     // Is it possible for this inst to write memory?
  bit isTwoAddress = 0;     // Is this a two address instruction?
  bit isConvertibleToThreeAddress = 0;  // Can this 2-addr instruction promote?
  bit isCommutable = 0;     // Is this 3 operand instruction commutable?
  bit isTerminator = 0;     // Is this part of the terminator for a basic block?
  bit isReMaterializable = 0; // Is this instruction re-materializable?
  bit isPredicable = 0;     // Is this instruction predicable?
  bit hasDelaySlot = 0;     // Does this instruction have an delay slot?
  bit usesCustomDAGSchedInserter = 0; // Pseudo instr needing special help.
  bit hasCtrlDep   = 0;     // Does this instruction r/w ctrl-flow chains?
  bit isNotDuplicable = 0;  // Is it unsafe to duplicate this instruction?
  bit isAsCheapAsAMove = 0; // As cheap (or cheaper) than a move instruction.

  // Side effect flags - When set, the flags have these meanings:
  //
  //  hasSideEffects - The instruction has side effects that are not
  //    captured by any operands of the instruction or other flags.
  //
  //  mayHaveSideEffects  - Some instances of the instruction can have side
  //    effects. The virtual method "isReallySideEffectFree" is called to
  //    determine this. Load instructions are an example of where this is
  //    useful. In general, loads always have side effects. However, loads from
  //    constant pools don't. Individual back ends make this determination.
  //
  //  neverHasSideEffects - Set on an instruction with no pattern if it has no
  //    side effects.
  bit hasSideEffects = 0;
  bit mayHaveSideEffects = 0;
  bit neverHasSideEffects = 0;

  InstrItinClass Itinerary = NoItinerary;// Execution steps used for scheduling.

  string Constraints = "";  // OperandConstraint, e.g. $src = $dst.
  
  /// DisableEncoding - List of operand names (e.g. "$op1,$op2") that should not
  /// be encoded into the output machineinstr.
  string DisableEncoding = "";
}

/// Predicates - These are extra conditionals which are turned into instruction
/// selector matching code. Currently each predicate is just a string.
class Predicate<string cond> {
  string CondString = cond;
}

/// NoHonorSignDependentRounding - This predicate is true if support for
/// sign-dependent-rounding is not enabled.
def NoHonorSignDependentRounding
 : Predicate<"!HonorSignDependentRoundingFPMath()">;

class Requires<list<Predicate> preds> {
  list<Predicate> Predicates = preds;
}

/// ops definition - This is just a simple marker used to identify the operands
/// list for an instruction. outs and ins are identical both syntatically and
/// semantically, they are used to define def operands and use operands to
/// improve readibility. This should be used like this:
///     (outs R32:$dst), (ins R32:$src1, R32:$src2) or something similar.
def ops;
def outs;
def ins;

/// variable_ops definition - Mark this instruction as taking a variable number
/// of operands.
def variable_ops;

/// ptr_rc definition - Mark this operand as being a pointer value whose
/// register class is resolved dynamically via a callback to TargetInstrInfo.
/// FIXME: We should probably change this to a class which contain a list of
/// flags. But currently we have but one flag.
def ptr_rc;

/// unknown definition - Mark this operand as being of unknown type, causing
/// it to be resolved by inference in the context it is used.
def unknown;

/// Operand Types - These provide the built-in operand types that may be used
/// by a target.  Targets can optionally provide their own operand types as
/// needed, though this should not be needed for RISC targets.
class Operand<ValueType ty> {
  ValueType Type = ty;
  string PrintMethod = "printOperand";
  dag MIOperandInfo = (ops);
}

def i1imm  : Operand<i1>;
def i8imm  : Operand<i8>;
def i16imm : Operand<i16>;
def i32imm : Operand<i32>;
def i64imm : Operand<i64>;

def f32imm : Operand<f32>;
def f64imm : Operand<f64>;

/// zero_reg definition - Special node to stand for the zero register.
///
def zero_reg;

/// PredicateOperand - This can be used to define a predicate operand for an
/// instruction.  OpTypes specifies the MIOperandInfo for the operand, and
/// AlwaysVal specifies the value of this predicate when set to "always
/// execute".
class PredicateOperand<ValueType ty, dag OpTypes, dag AlwaysVal>
  : Operand<ty> {
  let MIOperandInfo = OpTypes;
  dag DefaultOps = AlwaysVal;
}

/// OptionalDefOperand - This is used to define a optional definition operand
/// for an instruction. DefaultOps is the register the operand represents if none
/// is supplied, e.g. zero_reg.
class OptionalDefOperand<ValueType ty, dag OpTypes, dag defaultops>
  : Operand<ty> {
  let MIOperandInfo = OpTypes;
  dag DefaultOps = defaultops;
}


// InstrInfo - This class should only be instantiated once to provide parameters
// which are global to the the target machine.
//
class InstrInfo {
  // If the target wants to associate some target-specific information with each
  // instruction, it should provide these two lists to indicate how to assemble
  // the target specific information into the 32 bits available.
  //
  list<string> TSFlagsFields = [];
  list<int>    TSFlagsShifts = [];

  // Target can specify its instructions in either big or little-endian formats.
  // For instance, while both Sparc and PowerPC are big-endian platforms, the
  // Sparc manual specifies its instructions in the format [31..0] (big), while
  // PowerPC specifies them using the format [0..31] (little).
  bit isLittleEndianEncoding = 0;
}

// Standard Instructions.
def PHI : Instruction {
  let OutOperandList = (ops);
  let InOperandList = (ops variable_ops);
  let AsmString = "PHINODE";
  let Namespace = "TargetInstrInfo";
}
def INLINEASM : Instruction {
  let OutOperandList = (ops);
  let InOperandList = (ops variable_ops);
  let AsmString = "";
  let Namespace = "TargetInstrInfo";
}
def DBG_LABEL : Instruction {
  let OutOperandList = (ops);
  let InOperandList = (ops i32imm:$id);
  let AsmString = "";
  let Namespace = "TargetInstrInfo";
  let hasCtrlDep = 1;
}
def EH_LABEL : Instruction {
  let OutOperandList = (ops);
  let InOperandList = (ops i32imm:$id);
  let AsmString = "";
  let Namespace = "TargetInstrInfo";
  let hasCtrlDep = 1;
}
def GC_LABEL : Instruction {
  let OutOperandList = (ops);
  let InOperandList = (ops i32imm:$id);
  let AsmString = "";
  let Namespace = "TargetInstrInfo";
  let hasCtrlDep = 1;
}
def DECLARE : Instruction {
  let OutOperandList = (ops);
  let InOperandList = (ops variable_ops);
  let AsmString = "";
  let Namespace = "TargetInstrInfo";
  let hasCtrlDep = 1;
}
def EXTRACT_SUBREG : Instruction {
  let OutOperandList = (ops unknown:$dst);
  let InOperandList = (ops unknown:$supersrc, i32imm:$subidx);
  let AsmString = "";
  let Namespace = "TargetInstrInfo";
  let neverHasSideEffects = 1;
}
def INSERT_SUBREG : Instruction {
  let OutOperandList = (ops unknown:$dst);
  let InOperandList = (ops unknown:$supersrc, unknown:$subsrc, i32imm:$subidx);
  let AsmString = "";
  let Namespace = "TargetInstrInfo";
  let neverHasSideEffects = 1;
  let Constraints = "$supersrc = $dst";
}
def IMPLICIT_DEF : Instruction {
  let OutOperandList = (ops unknown:$dst);
  let InOperandList = (ops);
  let AsmString = "";
  let Namespace = "TargetInstrInfo";
  let neverHasSideEffects = 1;
  let isReMaterializable = 1;
  let isAsCheapAsAMove = 1;
}
def SUBREG_TO_REG : Instruction {
  let OutOperandList = (ops unknown:$dst);
  let InOperandList = (ops unknown:$implsrc, unknown:$subsrc, i32imm:$subidx);
  let AsmString = "";
  let Namespace = "TargetInstrInfo";
  let neverHasSideEffects = 1;
}

//===----------------------------------------------------------------------===//
// AsmWriter - This class can be implemented by targets that need to customize
// the format of the .s file writer.
//
// Subtargets can have multiple different asmwriters (e.g. AT&T vs Intel syntax
// on X86 for example).
//
class AsmWriter {
  // AsmWriterClassName - This specifies the suffix to use for the asmwriter
  // class.  Generated AsmWriter classes are always prefixed with the target
  // name.
  string AsmWriterClassName  = "AsmPrinter";

  // InstFormatName - AsmWriters can specify the name of the format string to
  // print instructions with.
  string InstFormatName = "AsmString";

  // Variant - AsmWriters can be of multiple different variants.  Variants are
  // used to support targets that need to emit assembly code in ways that are
  // mostly the same for different targets, but have minor differences in
  // syntax.  If the asmstring contains {|} characters in them, this integer
  // will specify which alternative to use.  For example "{x|y|z}" with Variant
  // == 1, will expand to "y".
  int Variant = 0;
}
def DefaultAsmWriter : AsmWriter;


//===----------------------------------------------------------------------===//
// Target - This class contains the "global" target information
//
class Target {
  // InstructionSet - Instruction set description for this target.
  InstrInfo InstructionSet;

  // AssemblyWriters - The AsmWriter instances available for this target.
  list<AsmWriter> AssemblyWriters = [DefaultAsmWriter];
}

//===----------------------------------------------------------------------===//
// SubtargetFeature - A characteristic of the chip set.
//
class SubtargetFeature<string n, string a,  string v, string d,
                       list<SubtargetFeature> i = []> {
  // Name - Feature name.  Used by command line (-mattr=) to determine the
  // appropriate target chip.
  //
  string Name = n;
  
  // Attribute - Attribute to be set by feature.
  //
  string Attribute = a;
  
  // Value - Value the attribute to be set to by feature.
  //
  string Value = v;
  
  // Desc - Feature description.  Used by command line (-mattr=) to display help
  // information.
  //
  string Desc = d;

  // Implies - Features that this feature implies are present. If one of those
  // features isn't set, then this one shouldn't be set either.
  //
  list<SubtargetFeature> Implies = i;
}

//===----------------------------------------------------------------------===//
// Processor chip sets - These values represent each of the chip sets supported
// by the scheduler.  Each Processor definition requires corresponding
// instruction itineraries.
//
class Processor<string n, ProcessorItineraries pi, list<SubtargetFeature> f> {
  // Name - Chip set name.  Used by command line (-mcpu=) to determine the
  // appropriate target chip.
  //
  string Name = n;
  
  // ProcItin - The scheduling information for the target processor.
  //
  ProcessorItineraries ProcItin = pi;
  
  // Features - list of 
  list<SubtargetFeature> Features = f;
}

//===----------------------------------------------------------------------===//
// Pull in the common support for calling conventions.
//
include "llvm/Target/TargetCallingConv.td"

//===----------------------------------------------------------------------===//
// Pull in the common support for DAG isel generation.
//
include "llvm/Target/TargetSelectionDAG.td"
