//===- MipsInstrInfo.td - Mips Register defs --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction format superclass
//===----------------------------------------------------------------------===//

include "MipsInstrFormats.td"

//===----------------------------------------------------------------------===//
// Mips profiles and nodes
//===----------------------------------------------------------------------===//

def SDT_MipsRet          : SDTypeProfile<0, 1, [SDTCisInt<0>]>;
def SDT_MipsJmpLink      : SDTypeProfile<0, 1, [SDTCisVT<0, iPTR>]>;
def SDT_MipsSelectCC     : SDTypeProfile<1, 3, [SDTCisSameAs<0, 2>, 
                                         SDTCisSameAs<2, 3>, SDTCisInt<1>]>;
def SDT_MipsCMov         : SDTypeProfile<1, 4, [SDTCisSameAs<0, 1>, 
                                         SDTCisSameAs<1, 2>, SDTCisSameAs<3, 4>,
                                         SDTCisInt<4>]>;
def SDT_MipsCallSeqStart : SDCallSeqStart<[SDTCisVT<0, i32>]>;
def SDT_MipsCallSeqEnd   : SDCallSeqEnd<[SDTCisVT<0, i32>, SDTCisVT<1, i32>]>;

// Call
def MipsJmpLink : SDNode<"MipsISD::JmpLink",SDT_MipsJmpLink, [SDNPHasChain,
                         SDNPOutFlag]>;

// Hi and Lo nodes are used to handle global addresses. Used on 
// MipsISelLowering to lower stuff like GlobalAddress, ExternalSymbol 
// static model. (nothing to do with Mips Registers Hi and Lo)
def MipsHi    : SDNode<"MipsISD::Hi", SDTIntUnaryOp>;
def MipsLo    : SDNode<"MipsISD::Lo", SDTIntUnaryOp>;
def MipsGPRel : SDNode<"MipsISD::GPRel", SDTIntUnaryOp>;

// Return
def MipsRet : SDNode<"MipsISD::Ret", SDT_MipsRet, [SDNPHasChain, 
                     SDNPOptInFlag]>;

// These are target-independent nodes, but have target-specific formats.
def callseq_start : SDNode<"ISD::CALLSEQ_START", SDT_MipsCallSeqStart,
                           [SDNPHasChain, SDNPOutFlag]>;
def callseq_end   : SDNode<"ISD::CALLSEQ_END", SDT_MipsCallSeqEnd,
                           [SDNPHasChain, SDNPOptInFlag, SDNPOutFlag]>;

// Select Condition Code
def MipsSelectCC  : SDNode<"MipsISD::SelectCC", SDT_MipsSelectCC>;

// Conditional Move
def MipsCMov      : SDNode<"MipsISD::CMov", SDT_MipsCMov>;

//===----------------------------------------------------------------------===//
// Mips Instruction Predicate Definitions.
//===----------------------------------------------------------------------===//
def HasSEInReg  : Predicate<"Subtarget.hasSEInReg()">;
def HasBitCount : Predicate<"Subtarget.hasBitCount()">;
def HasSwap     : Predicate<"Subtarget.hasSwap()">;
def HasCondMov  : Predicate<"Subtarget.hasCondMov()">;

//===----------------------------------------------------------------------===//
// Mips Operand, Complex Patterns and Transformations Definitions.
//===----------------------------------------------------------------------===//

// Instruction operand types
def brtarget    : Operand<OtherVT>;
def calltarget  : Operand<i32>;
def simm16      : Operand<i32>;
def shamt       : Operand<i32>;

// Unsigned Operand
def uimm16      : Operand<i32> {
  let PrintMethod = "printUnsignedImm";
}

// Address operand
def mem : Operand<i32> {
  let PrintMethod = "printMemOperand";
  let MIOperandInfo = (ops simm16, CPURegs);
}

// Transformation Function - get the lower 16 bits.
def LO16 : SDNodeXForm<imm, [{
  return getI32Imm((unsigned)N->getZExtValue() & 0xFFFF);
}]>;

// Transformation Function - get the higher 16 bits.
def HI16 : SDNodeXForm<imm, [{
  return getI32Imm((unsigned)N->getZExtValue() >> 16);
}]>;

// Node immediate fits as 16-bit sign extended on target immediate.
// e.g. addi, andi
def immSExt16  : PatLeaf<(imm), [{
  if (N->getValueType(0) == MVT::i32)
    return (int32_t)N->getZExtValue() == (short)N->getZExtValue();
  else
    return (int64_t)N->getZExtValue() == (short)N->getZExtValue();
}]>;

// Node immediate fits as 16-bit zero extended on target immediate.
// The LO16 param means that only the lower 16 bits of the node
// immediate are caught.
// e.g. addiu, sltiu
def immZExt16  : PatLeaf<(imm), [{
  if (N->getValueType(0) == MVT::i32)
    return (uint32_t)N->getZExtValue() == (unsigned short)N->getZExtValue();
  else
    return (uint64_t)N->getZExtValue() == (unsigned short)N->getZExtValue();
}], LO16>;

// shamt field must fit in 5 bits.
def immZExt5 : PatLeaf<(imm), [{
  return N->getZExtValue() == ((N->getZExtValue()) & 0x1f) ;
}]>;

// Mips Address Mode! SDNode frameindex could possibily be a match
// since load and store instructions from stack used it.
def addr : ComplexPattern<i32, 2, "SelectAddr", [frameindex], []>;

//===----------------------------------------------------------------------===//
// Instructions specific format
//===----------------------------------------------------------------------===//

// Arithmetic 3 register operands
let isCommutable = 1 in
class ArithR<bits<6> op, bits<6> func, string instr_asm, SDNode OpNode,
             InstrItinClass itin>:
  FR< op,
      func,
      (outs CPURegs:$dst),
      (ins CPURegs:$b, CPURegs:$c),
      !strconcat(instr_asm, "\t$dst, $b, $c"),
      [(set CPURegs:$dst, (OpNode CPURegs:$b, CPURegs:$c))], itin>;

let isCommutable = 1 in
class ArithOverflowR<bits<6> op, bits<6> func, string instr_asm>:
  FR< op,
      func,
      (outs CPURegs:$dst),
      (ins CPURegs:$b, CPURegs:$c),
      !strconcat(instr_asm, "\t$dst, $b, $c"),
      [], IIAlu>;

// Arithmetic 2 register operands
class ArithI<bits<6> op, string instr_asm, SDNode OpNode,
             Operand Od, PatLeaf imm_type> :
  FI< op,
      (outs CPURegs:$dst),
      (ins CPURegs:$b, Od:$c),
      !strconcat(instr_asm, "\t$dst, $b, $c"),
      [(set CPURegs:$dst, (OpNode CPURegs:$b, imm_type:$c))], IIAlu>;

class ArithOverflowI<bits<6> op, string instr_asm, SDNode OpNode,
             Operand Od, PatLeaf imm_type> :
  FI< op,
      (outs CPURegs:$dst),
      (ins CPURegs:$b, Od:$c),
      !strconcat(instr_asm, "\t$dst, $b, $c"),
      [], IIAlu>;

// Arithmetic Multiply ADD/SUB
let rd=0 in
class MArithR<bits<6> func, string instr_asm> :
  FR< 0x1c,
      func,
      (outs CPURegs:$rs),
      (ins CPURegs:$rt),
      !strconcat(instr_asm, "\t$rs, $rt"),
      [], IIImul>;

//  Logical
class LogicR<bits<6> func, string instr_asm, SDNode OpNode>:
  FR< 0x00,
      func,
      (outs CPURegs:$dst),
      (ins CPURegs:$b, CPURegs:$c),
      !strconcat(instr_asm, "\t$dst, $b, $c"),
      [(set CPURegs:$dst, (OpNode CPURegs:$b, CPURegs:$c))], IIAlu>;

class LogicI<bits<6> op, string instr_asm, SDNode OpNode>:
  FI< op,
      (outs CPURegs:$dst),
      (ins CPURegs:$b, uimm16:$c),
      !strconcat(instr_asm, "\t$dst, $b, $c"),
      [(set CPURegs:$dst, (OpNode CPURegs:$b, immZExt16:$c))], IIAlu>;

class LogicNOR<bits<6> op, bits<6> func, string instr_asm>:
  FR< op,
      func,
      (outs CPURegs:$dst),
      (ins CPURegs:$b, CPURegs:$c),
      !strconcat(instr_asm, "\t$dst, $b, $c"),
      [(set CPURegs:$dst, (not (or CPURegs:$b, CPURegs:$c)))], IIAlu>;

// Shifts
let rt = 0 in
class LogicR_shift_imm<bits<6> func, string instr_asm, SDNode OpNode>:
  FR< 0x00,
      func,
      (outs CPURegs:$dst),
      (ins CPURegs:$b, shamt:$c),
      !strconcat(instr_asm, "\t$dst, $b, $c"),
      [(set CPURegs:$dst, (OpNode CPURegs:$b, immZExt5:$c))], IIAlu>;

class LogicR_shift_reg<bits<6> func, string instr_asm, SDNode OpNode>:
  FR< 0x00,
      func,
      (outs CPURegs:$dst),
      (ins CPURegs:$b, CPURegs:$c),
      !strconcat(instr_asm, "\t$dst, $b, $c"),
      [(set CPURegs:$dst, (OpNode CPURegs:$b, CPURegs:$c))], IIAlu>;

// Load Upper Imediate
class LoadUpper<bits<6> op, string instr_asm>:
  FI< op,
      (outs CPURegs:$dst),
      (ins uimm16:$imm),
      !strconcat(instr_asm, "\t$dst, $imm"),
      [], IIAlu>;

// Memory Load/Store
let canFoldAsLoad = 1, hasDelaySlot = 1 in
class LoadM<bits<6> op, string instr_asm, PatFrag OpNode>:
  FI< op,
      (outs CPURegs:$dst),
      (ins mem:$addr),
      !strconcat(instr_asm, "\t$dst, $addr"),
      [(set CPURegs:$dst, (OpNode addr:$addr))], IILoad>;

class StoreM<bits<6> op, string instr_asm, PatFrag OpNode>:
  FI< op,
      (outs),
      (ins CPURegs:$dst, mem:$addr),
      !strconcat(instr_asm, "\t$dst, $addr"),
      [(OpNode CPURegs:$dst, addr:$addr)], IIStore>;

// Conditional Branch
let isBranch = 1, isTerminator=1, hasDelaySlot = 1 in {
class CBranch<bits<6> op, string instr_asm, PatFrag cond_op>:
  FI< op,
      (outs),
      (ins CPURegs:$a, CPURegs:$b, brtarget:$offset),
      !strconcat(instr_asm, "\t$a, $b, $offset"),
      [(brcond (cond_op CPURegs:$a, CPURegs:$b), bb:$offset)],
      IIBranch>;


class CBranchZero<bits<6> op, string instr_asm, PatFrag cond_op>:
  FI< op,
      (outs),
      (ins CPURegs:$src, brtarget:$offset),
      !strconcat(instr_asm, "\t$src, $offset"),
      [(brcond (cond_op CPURegs:$src, 0), bb:$offset)],
      IIBranch>;
}

// SetCC
class SetCC_R<bits<6> op, bits<6> func, string instr_asm,
      PatFrag cond_op>:
  FR< op,
      func,
      (outs CPURegs:$dst),
      (ins CPURegs:$b, CPURegs:$c),
      !strconcat(instr_asm, "\t$dst, $b, $c"),
      [(set CPURegs:$dst, (cond_op CPURegs:$b, CPURegs:$c))],
      IIAlu>;

class SetCC_I<bits<6> op, string instr_asm, PatFrag cond_op,
      Operand Od, PatLeaf imm_type>:
  FI< op,
      (outs CPURegs:$dst),
      (ins CPURegs:$b, Od:$c),
      !strconcat(instr_asm, "\t$dst, $b, $c"),
      [(set CPURegs:$dst, (cond_op CPURegs:$b, imm_type:$c))],
      IIAlu>;

// Unconditional branch
let isBranch=1, isTerminator=1, isBarrier=1, hasDelaySlot = 1 in
class JumpFJ<bits<6> op, string instr_asm>:
  FJ< op,
      (outs),
      (ins brtarget:$target),
      !strconcat(instr_asm, "\t$target"),
      [(br bb:$target)], IIBranch>;

let isBranch=1, isTerminator=1, isBarrier=1, rd=0, hasDelaySlot = 1 in
class JumpFR<bits<6> op, bits<6> func, string instr_asm>:
  FR< op,
      func,
      (outs),
      (ins CPURegs:$target),
      !strconcat(instr_asm, "\t$target"),
      [(brind CPURegs:$target)], IIBranch>;

// Jump and Link (Call)
let isCall=1, hasDelaySlot=1,
  // All calls clobber the non-callee saved registers...
  Defs = [AT, V0, V1, A0, A1, A2, A3, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, 
          K0, K1, F0, F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13,
          F14, F15, F16, F17, F18, F19], Uses = [GP] in {
  class JumpLink<bits<6> op, string instr_asm>:
    FJ< op,
        (outs),
        (ins calltarget:$target),
        !strconcat(instr_asm, "\t$target"),
        [(MipsJmpLink imm:$target)], IIBranch>;

  let rd=31 in
  class JumpLinkReg<bits<6> op, bits<6> func, string instr_asm>:
    FR< op,
        func,
        (outs),
        (ins CPURegs:$rs),
        !strconcat(instr_asm, "\t$rs"),
        [(MipsJmpLink CPURegs:$rs)], IIBranch>;

  class BranchLink<string instr_asm>:
    FI< 0x1,
        (outs),
        (ins CPURegs:$rs, brtarget:$target),
        !strconcat(instr_asm, "\t$rs, $target"),
        [], IIBranch>;
}

// Mul, Div
class MulDiv<bits<6> func, string instr_asm, InstrItinClass itin>:
  FR< 0x00,
      func,
      (outs),
      (ins CPURegs:$a, CPURegs:$b),
      !strconcat(instr_asm, "\t$a, $b"),
      [], itin>;

// Move from Hi/Lo
class MoveFromLOHI<bits<6> func, string instr_asm>:
  FR< 0x00,
      func,
      (outs CPURegs:$dst),
      (ins),
      !strconcat(instr_asm, "\t$dst"),
      [], IIHiLo>;

class MoveToLOHI<bits<6> func, string instr_asm>:
  FR< 0x00,
      func,
      (outs),
      (ins CPURegs:$src),
      !strconcat(instr_asm, "\t$src"),
      [], IIHiLo>;

class EffectiveAddress<string instr_asm> :
  FI<0x09,
     (outs CPURegs:$dst),
     (ins mem:$addr),
     instr_asm,
     [(set CPURegs:$dst, addr:$addr)], IIAlu>;

// Count Leading Ones/Zeros in Word
class CountLeading<bits<6> func, string instr_asm, SDNode CountOp>:
  FR< 0x1c, func, (outs CPURegs:$dst), (ins CPURegs:$src),
      !strconcat(instr_asm, "\t$dst, $src"), 
      [(set CPURegs:$dst, (CountOp CPURegs:$src))], IIAlu>;

// Sign Extend in Register.
class SignExtInReg<bits<6> func, string instr_asm, ValueType vt>:
  FR< 0x3f, func, (outs CPURegs:$dst), (ins CPURegs:$src),
      !strconcat(instr_asm, "\t$dst, $src"),
      [(set CPURegs:$dst, (sext_inreg CPURegs:$src, vt))], NoItinerary>;

// Byte Swap
class ByteSwap<bits<6> func, string instr_asm>:
  FR< 0x1f, func, (outs CPURegs:$dst), (ins CPURegs:$src),
      !strconcat(instr_asm, "\t$dst, $src"),
      [(set CPURegs:$dst, (bswap CPURegs:$src))], NoItinerary>;

// Conditional Move
class CondMov<bits<6> func, string instr_asm, PatLeaf MovCode>:
  FR< 0x00, func, (outs CPURegs:$dst), (ins CPURegs:$F, CPURegs:$T, 
      CPURegs:$cond), !strconcat(instr_asm, "\t$dst, $T, $cond"), 
      [(set CPURegs:$dst, (MipsCMov CPURegs:$F, CPURegs:$T, 
                           CPURegs:$cond, MovCode))], NoItinerary>;

//===----------------------------------------------------------------------===//
// Pseudo instructions
//===----------------------------------------------------------------------===//

// As stack alignment is always done with addiu, we need a 16-bit immediate
let Defs = [SP], Uses = [SP] in {
def ADJCALLSTACKDOWN : MipsPseudo<(outs), (ins uimm16:$amt),
                                  "!ADJCALLSTACKDOWN $amt",
                                  [(callseq_start timm:$amt)]>;
def ADJCALLSTACKUP   : MipsPseudo<(outs), (ins uimm16:$amt1, uimm16:$amt2),
                                  "!ADJCALLSTACKUP $amt1",
                                  [(callseq_end timm:$amt1, timm:$amt2)]>;
}

// Some assembly macros need to avoid pseudoinstructions and assembler
// automatic reodering, we should reorder ourselves.
def MACRO     : MipsPseudo<(outs), (ins), ".set\tmacro",     []>;
def REORDER   : MipsPseudo<(outs), (ins), ".set\treorder",   []>;
def NOMACRO   : MipsPseudo<(outs), (ins), ".set\tnomacro",   []>;
def NOREORDER : MipsPseudo<(outs), (ins), ".set\tnoreorder", []>;

// When handling PIC code the assembler needs .cpload and .cprestore
// directives. If the real instructions corresponding these directives
// are used, we have the same behavior, but get also a bunch of warnings
// from the assembler.
def CPLOAD : MipsPseudo<(outs), (ins CPURegs:$picreg), ".cpload\t$picreg", []>;
def CPRESTORE : MipsPseudo<(outs), (ins uimm16:$loc), ".cprestore\t$loc\n", []>;

// The supported Mips ISAs dont have any instruction close to the SELECT_CC 
// operation. The solution is to create a Mips pseudo SELECT_CC instruction
// (MipsSelectCC), use LowerSELECT_CC to generate this instruction and finally 
// replace it for real supported nodes into EmitInstrWithCustomInserter
let usesCustomDAGSchedInserter = 1 in {
  class PseudoSelCC<RegisterClass RC, string asmstr>: 
    MipsPseudo<(outs RC:$dst), (ins CPURegs:$CmpRes, RC:$T, RC:$F), asmstr, 
    [(set RC:$dst, (MipsSelectCC CPURegs:$CmpRes, RC:$T, RC:$F))]>;
}

def Select_CC : PseudoSelCC<CPURegs, "# MipsSelect_CC_i32">;

//===----------------------------------------------------------------------===//
// Instruction definition
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MipsI Instructions
//===----------------------------------------------------------------------===//

/// Arithmetic Instructions (ALU Immediate)
def ADDiu   : ArithI<0x09, "addiu", add, simm16, immSExt16>;
def ADDi    : ArithOverflowI<0x08, "addi",  add, simm16, immSExt16>;
def SLTi    : SetCC_I<0x0a, "slti", setlt, simm16, immSExt16>;
def SLTiu   : SetCC_I<0x0b, "sltiu", setult, simm16, immSExt16>;
def ANDi    : LogicI<0x0c, "andi", and>;
def ORi     : LogicI<0x0d, "ori",  or>;
def XORi    : LogicI<0x0e, "xori",  xor>;
def LUi     : LoadUpper<0x0f, "lui">;

/// Arithmetic Instructions (3-Operand, R-Type)
def ADDu    : ArithR<0x00, 0x21, "addu", add, IIAlu>;
def SUBu    : ArithR<0x00, 0x23, "subu", sub, IIAlu>;
def ADD     : ArithOverflowR<0x00, 0x20, "add">;
def SUB     : ArithOverflowR<0x00, 0x22, "sub">;
def SLT     : SetCC_R<0x00, 0x2a, "slt", setlt>;
def SLTu    : SetCC_R<0x00, 0x2b, "sltu", setult>;
def AND     : LogicR<0x24, "and", and>;
def OR      : LogicR<0x25, "or",  or>;
def XOR     : LogicR<0x26, "xor", xor>;
def NOR     : LogicNOR<0x00, 0x27, "nor">;

/// Shift Instructions
def SLL     : LogicR_shift_imm<0x00, "sll", shl>;
def SRL     : LogicR_shift_imm<0x02, "srl", srl>;
def SRA     : LogicR_shift_imm<0x03, "sra", sra>;
def SLLV    : LogicR_shift_reg<0x04, "sllv", shl>;
def SRLV    : LogicR_shift_reg<0x06, "srlv", srl>;
def SRAV    : LogicR_shift_reg<0x07, "srav", sra>;

/// Load and Store Instructions
def LB      : LoadM<0x20, "lb",  sextloadi8>;
def LBu     : LoadM<0x24, "lbu", zextloadi8>;
def LH      : LoadM<0x21, "lh",  sextloadi16>;
def LHu     : LoadM<0x25, "lhu", zextloadi16>;
def LW      : LoadM<0x23, "lw",  load>;
def SB      : StoreM<0x28, "sb", truncstorei8>;
def SH      : StoreM<0x29, "sh", truncstorei16>;
def SW      : StoreM<0x2b, "sw", store>;

/// Jump and Branch Instructions
def J       : JumpFJ<0x02, "j">;
def JR      : JumpFR<0x00, 0x08, "jr">;
def JAL     : JumpLink<0x03, "jal">;
def JALR    : JumpLinkReg<0x00, 0x09, "jalr">;
def BEQ     : CBranch<0x04, "beq", seteq>;
def BNE     : CBranch<0x05, "bne", setne>;

let rt=1 in
  def BGEZ  : CBranchZero<0x01, "bgez", setge>;

let rt=0 in {
  def BGTZ  : CBranchZero<0x07, "bgtz", setgt>;
  def BLEZ  : CBranchZero<0x07, "blez", setle>;
  def BLTZ  : CBranchZero<0x01, "bltz", setlt>;
}

def BGEZAL  : BranchLink<"bgezal">;
def BLTZAL  : BranchLink<"bltzal">;

let isReturn=1, isTerminator=1, hasDelaySlot=1,
    isBarrier=1, hasCtrlDep=1, rs=0, rt=0, shamt=0 in
  def RET : FR <0x00, 0x02, (outs), (ins CPURegs:$target),
                "jr\t$target", [(MipsRet CPURegs:$target)], IIBranch>;

/// Multiply and Divide Instructions. 
let Defs = [HI, LO] in {
  def MULT    : MulDiv<0x18, "mult", IIImul>;
  def MULTu   : MulDiv<0x19, "multu", IIImul>;
  def DIV     : MulDiv<0x1a, "div", IIIdiv>;
  def DIVu    : MulDiv<0x1b, "divu", IIIdiv>;
}

let Defs = [HI] in
  def MTHI  : MoveToLOHI<0x11, "mthi">;
let Defs = [LO] in
  def MTLO  : MoveToLOHI<0x13, "mtlo">;

let Uses = [HI] in
  def MFHI  : MoveFromLOHI<0x10, "mfhi">;
let Uses = [LO] in
  def MFLO  : MoveFromLOHI<0x12, "mflo">;

/// Sign Ext In Register Instructions.
let Predicates = [HasSEInReg] in {
  let shamt = 0x10, rs = 0 in 
    def SEB : SignExtInReg<0x21, "seb", i8>;

  let shamt = 0x18, rs = 0 in 
    def SEH : SignExtInReg<0x20, "seh", i16>;
}

/// Count Leading
let Predicates = [HasBitCount] in {
  let rt = 0 in
    def CLZ : CountLeading<0b010110, "clz", ctlz>;
}

/// Byte Swap
let Predicates = [HasSwap] in {
  let shamt = 0x3, rs = 0 in
    def WSBW : ByteSwap<0x20, "wsbw">;
}

/// Conditional Move
def MIPS_CMOV_ZERO  : PatLeaf<(i32 0)>;
def MIPS_CMOV_NZERO : PatLeaf<(i32 1)>;

let Predicates = [HasCondMov], isTwoAddress = 1 in {
  def MOVN : CondMov<0x0a, "movn", MIPS_CMOV_NZERO>;
  def MOVZ : CondMov<0x0b, "movz", MIPS_CMOV_ZERO>;
}

/// No operation
let addr=0 in
  def NOP   : FJ<0, (outs), (ins), "nop", [], IIAlu>;

// FrameIndexes are legalized when they are operands from load/store
// instructions. The same not happens for stack address copies, so an
// add op with mem ComplexPattern is used and the stack address copy
// can be matched. It's similar to Sparc LEA_ADDRi
def LEA_ADDiu : EffectiveAddress<"addiu\t$dst, ${addr:stackloc}">;

// MADD*/MSUB* are not part of MipsI either.
//def MADD    : MArithR<0x00, "madd">;
//def MADDU   : MArithR<0x01, "maddu">;
//def MSUB    : MArithR<0x04, "msub">;
//def MSUBU   : MArithR<0x05, "msubu">;

// MUL is a assembly macro in the current used ISAs. In recent ISA's
// it is a real instruction.
//def MUL   : ArithR<0x1c, 0x02, "mul", mul, IIImul>;

//===----------------------------------------------------------------------===//
//  Arbitrary patterns that map to one or more instructions
//===----------------------------------------------------------------------===//

// Small immediates
def : Pat<(i32 immSExt16:$in),
          (ADDiu ZERO, imm:$in)>;
def : Pat<(i32 immZExt16:$in),
          (ORi ZERO, imm:$in)>;

// Arbitrary immediates
def : Pat<(i32 imm:$imm),
          (ORi (LUi (HI16 imm:$imm)), (LO16 imm:$imm))>;

// Carry patterns
def : Pat<(subc CPURegs:$lhs, CPURegs:$rhs),
          (SUBu CPURegs:$lhs, CPURegs:$rhs)>;
def : Pat<(addc CPURegs:$lhs, CPURegs:$rhs),
          (ADDu CPURegs:$lhs, CPURegs:$rhs)>;
def : Pat<(addc  CPURegs:$src, imm:$imm),
          (ADDiu CPURegs:$src, imm:$imm)>;

// Call
def : Pat<(MipsJmpLink (i32 tglobaladdr:$dst)),
          (JAL tglobaladdr:$dst)>;
def : Pat<(MipsJmpLink (i32 texternalsym:$dst)),
          (JAL texternalsym:$dst)>;
def : Pat<(MipsJmpLink CPURegs:$dst),
          (JALR CPURegs:$dst)>;

// hi/lo relocs
def : Pat<(MipsHi tglobaladdr:$in), (LUi tglobaladdr:$in)>;
def : Pat<(add CPURegs:$hi, (MipsLo tglobaladdr:$lo)),
          (ADDiu CPURegs:$hi, tglobaladdr:$lo)>;

def : Pat<(MipsHi tjumptable:$in), (LUi tjumptable:$in)>;
def : Pat<(add CPURegs:$hi, (MipsLo tjumptable:$lo)),
          (ADDiu CPURegs:$hi, tjumptable:$lo)>;

def : Pat<(MipsHi tconstpool:$in), (LUi tconstpool:$in)>;
def : Pat<(add CPURegs:$hi, (MipsLo tconstpool:$lo)),
          (ADDiu CPURegs:$hi, tconstpool:$lo)>;

// gp_rel relocs
def : Pat<(add CPURegs:$gp, (MipsGPRel tglobaladdr:$in)), 
          (ADDiu CPURegs:$gp, tglobaladdr:$in)>;
def : Pat<(add CPURegs:$gp, (MipsGPRel tconstpool:$in)), 
          (ADDiu CPURegs:$gp, tconstpool:$in)>;

// Mips does not have "not", so we expand our way
def : Pat<(not CPURegs:$in),
          (NOR CPURegs:$in, ZERO)>;

// extended load and stores
def : Pat<(extloadi1  addr:$src), (LBu addr:$src)>;
def : Pat<(extloadi8  addr:$src), (LBu addr:$src)>;
def : Pat<(extloadi16 addr:$src), (LHu addr:$src)>;

// peepholes
def : Pat<(store (i32 0), addr:$dst), (SW ZERO, addr:$dst)>;

// brcond patterns
def : Pat<(brcond (setne CPURegs:$lhs, 0), bb:$dst),
          (BNE CPURegs:$lhs, ZERO, bb:$dst)>;
def : Pat<(brcond (seteq CPURegs:$lhs, 0), bb:$dst),
          (BEQ CPURegs:$lhs, ZERO, bb:$dst)>;

def : Pat<(brcond (setge CPURegs:$lhs, CPURegs:$rhs), bb:$dst),
          (BEQ (SLT CPURegs:$lhs, CPURegs:$rhs), ZERO, bb:$dst)>;
def : Pat<(brcond (setuge CPURegs:$lhs, CPURegs:$rhs), bb:$dst),
          (BEQ (SLTu CPURegs:$lhs, CPURegs:$rhs), ZERO, bb:$dst)>;
def : Pat<(brcond (setge CPURegs:$lhs, immSExt16:$rhs), bb:$dst),
          (BEQ (SLTi CPURegs:$lhs, immSExt16:$rhs), ZERO, bb:$dst)>;
def : Pat<(brcond (setuge CPURegs:$lhs, immSExt16:$rhs), bb:$dst),
          (BEQ (SLTiu CPURegs:$lhs, immSExt16:$rhs), ZERO, bb:$dst)>;

def : Pat<(brcond (setle CPURegs:$lhs, CPURegs:$rhs), bb:$dst),
          (BEQ (SLT CPURegs:$rhs, CPURegs:$lhs), ZERO, bb:$dst)>;
def : Pat<(brcond (setule CPURegs:$lhs, CPURegs:$rhs), bb:$dst),
          (BEQ (SLTu CPURegs:$rhs, CPURegs:$lhs), ZERO, bb:$dst)>;

def : Pat<(brcond CPURegs:$cond, bb:$dst),
          (BNE CPURegs:$cond, ZERO, bb:$dst)>;

// select patterns
def : Pat<(select (setge CPURegs:$lhs, CPURegs:$rhs), CPURegs:$T, CPURegs:$F),
          (MOVZ CPURegs:$F, CPURegs:$T, (SLT CPURegs:$lhs, CPURegs:$rhs))>;
def : Pat<(select (setuge CPURegs:$lhs, CPURegs:$rhs), CPURegs:$T, CPURegs:$F),
          (MOVZ CPURegs:$F, CPURegs:$T, (SLTu CPURegs:$lhs, CPURegs:$rhs))>;
def : Pat<(select (setge CPURegs:$lhs, immSExt16:$rhs), CPURegs:$T, CPURegs:$F),
          (MOVZ CPURegs:$F, CPURegs:$T, (SLTi CPURegs:$lhs, immSExt16:$rhs))>;
def : Pat<(select (setuge CPURegs:$lh, immSExt16:$rh), CPURegs:$T, CPURegs:$F),
          (MOVZ CPURegs:$F, CPURegs:$T, (SLTiu CPURegs:$lh, immSExt16:$rh))>;

def : Pat<(select (setle CPURegs:$lhs, CPURegs:$rhs), CPURegs:$T, CPURegs:$F),
          (MOVZ CPURegs:$F, CPURegs:$T, (SLT CPURegs:$rhs, CPURegs:$lhs))>;
def : Pat<(select (setule CPURegs:$lhs, CPURegs:$rhs), CPURegs:$T, CPURegs:$F),
          (MOVZ CPURegs:$F, CPURegs:$T, (SLTu CPURegs:$rhs, CPURegs:$lhs))>;

def : Pat<(select (seteq CPURegs:$lhs, CPURegs:$rhs), CPURegs:$T, CPURegs:$F),
          (MOVZ CPURegs:$F, CPURegs:$T, (XOR CPURegs:$lhs, CPURegs:$rhs))>;
def : Pat<(select (setne CPURegs:$lhs, CPURegs:$rhs), CPURegs:$T, CPURegs:$F),
          (MOVN CPURegs:$F, CPURegs:$T, (XOR CPURegs:$lhs, CPURegs:$rhs))>;

def : Pat<(select CPURegs:$cond, CPURegs:$T, CPURegs:$F), 
          (MOVN CPURegs:$F, CPURegs:$T, CPURegs:$cond)>;

// setcc patterns
def : Pat<(seteq CPURegs:$lhs, CPURegs:$rhs),
          (SLTu (XOR CPURegs:$lhs, CPURegs:$rhs), 1)>;
def : Pat<(setne CPURegs:$lhs, CPURegs:$rhs),
          (SLTu ZERO, (XOR CPURegs:$lhs, CPURegs:$rhs))>;

def : Pat<(setle CPURegs:$lhs, CPURegs:$rhs),
          (XORi (SLT CPURegs:$rhs, CPURegs:$lhs), 1)>;
def : Pat<(setule CPURegs:$lhs, CPURegs:$rhs),
          (XORi (SLTu CPURegs:$rhs, CPURegs:$lhs), 1)>;

def : Pat<(setgt CPURegs:$lhs, CPURegs:$rhs),
          (SLT CPURegs:$rhs, CPURegs:$lhs)>;
def : Pat<(setugt CPURegs:$lhs, CPURegs:$rhs),
          (SLTu CPURegs:$rhs, CPURegs:$lhs)>;

def : Pat<(setge CPURegs:$lhs, CPURegs:$rhs),
          (XORi (SLT CPURegs:$lhs, CPURegs:$rhs), 1)>;
def : Pat<(setuge CPURegs:$lhs, CPURegs:$rhs),
          (XORi (SLTu CPURegs:$lhs, CPURegs:$rhs), 1)>;

def : Pat<(setge CPURegs:$lhs, immSExt16:$rhs),
          (XORi (SLTi CPURegs:$lhs, immSExt16:$rhs), 1)>;
def : Pat<(setuge CPURegs:$lhs, immSExt16:$rhs),
          (XORi (SLTiu CPURegs:$lhs, immSExt16:$rhs), 1)>;

//===----------------------------------------------------------------------===//
// Floating Point Support
//===----------------------------------------------------------------------===//

include "MipsInstrFPU.td"

