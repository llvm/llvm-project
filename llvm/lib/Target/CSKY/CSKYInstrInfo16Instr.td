//===-- CSKYInstrInfo16Instr.td - CSKY 16-bit Instruction --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the CSKY 16-bit instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CSKY specific DAG Nodes.
//===----------------------------------------------------------------------===//

// Target-dependent nodes.
def CSKY_NIE : SDNode<"CSKYISD::NIE", SDTNone,
    [SDNPHasChain, SDNPOptInGlue]>;
def CSKY_NIR : SDNode<"CSKYISD::NIR", SDTNone,
    [SDNPHasChain, SDNPOptInGlue]>;

//===----------------------------------------------------------------------===//
// Operand and SDNode transformation definitions.
//===----------------------------------------------------------------------===//

def br_symbol_16bit : Operand<iPTR> {
  let EncoderMethod =
    "getBranchSymbolOpValue<CSKY::fixup_csky_pcrel_imm10_scale2>";
  let ParserMatchClass = CSKYSymbol;
  let DecoderMethod = "decodeSImmOperand<10, 1>";
  let PrintMethod = "printCSKYSymbolOperand";
  let OperandType = "OPERAND_PCREL";
}

def SPOperand : AsmOperandClass {
  let Name = "SPOperand";
  let RenderMethod = "addRegOperands";
  let DiagnosticType = !strconcat("Invalid", Name);
}

def SPOp : RegisterOperand<GPR> {
 let ParserMatchClass = SPOperand;
}

def constpool_symbol_16bit : Operand<iPTR> {
  let ParserMatchClass = Constpool;
  let EncoderMethod =
    "getConstpoolSymbolOpValue<CSKY::fixup_csky_pcrel_uimm7_scale4>";
  let DecoderMethod = "decodeLRW16Imm8";
  let PrintMethod = "printConstpool";
  let OperandType = "OPERAND_PCREL";
}

//===----------------------------------------------------------------------===//
// Instruction Formats
//===----------------------------------------------------------------------===//

include "CSKYInstrFormats16Instr.td"

//===----------------------------------------------------------------------===//
// Instruction definitions.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Basic ALU instructions.
//===----------------------------------------------------------------------===//

let isCommutable = 1, isAdd = 1 in
  def ADDU16 : R16_XYZ<0, "addu16", add>;
let Pattern = [(set mGPR:$rz, (sub mGPR:$rx, mGPR:$ry))] in
  def SUBU16 : R16_XYZ<1, "subu16", sub>;

let isCommutable = 1, isAdd = 1 in
  def ADDC16 : R16_XZ_BINOP_C<0b1000, 0b01, "addc16">;
def SUBC16 : R16_XZ_BINOP_C<0b1000, 0b11, "subc16">;

let isCommutable = 1 in {
  let isAdd = 1 in
  def ADDU16XZ : R16_XZ_BINOP<0b1000, 0b00, "addu16", BinOpFrag<(add node:$LHS, node:$RHS)>>;
  def AND16 : R16_XZ_BINOP<0b1010, 0b00, "and16", BinOpFrag<(and node:$LHS, node:$RHS)>>;
  def OR16 : R16_XZ_BINOP<0b1011, 0b00, "or16", BinOpFrag<(or node:$LHS, node:$RHS)>>;
  def XOR16 : R16_XZ_BINOP<0b1011, 0b01, "xor16", BinOpFrag<(xor node:$LHS, node:$RHS)>>;
  def NOR16 : R16_XZ_BINOP<0b1011, 0b10, "nor16", BinOpFrag<(not (or node:$LHS, node:$RHS))>>;
  let isCodeGenOnly = 1 in
  def NOT16 : R16_XZ_UNOP<0b1011, 0b10, "not16">;
  def MULT16 :  R16_XZ_BINOP<0b1111, 0b00, "mult16", BinOpFrag<(mul node:$LHS, node:$RHS)>>;
}
def SUBU16XZ : R16_XZ_BINOP<0b1000, 0b10, "subu16", BinOpFrag<(sub node:$LHS, node:$RHS)>>;
def ANDN16 : R16_XZ_BINOP<0b1010, 0b01, "andn16", BinOpFrag<(and node:$LHS, (not node:$RHS))>>;
def LSL16 : R16_XZ_BINOP<0b1100, 0b00, "lsl16", BinOpFrag<(shl node:$LHS, node:$RHS)>>;
def LSR16 : R16_XZ_BINOP<0b1100, 0b01, "lsr16", BinOpFrag<(srl node:$LHS, node:$RHS)>>;
def ASR16 : R16_XZ_BINOP<0b1100, 0b10, "asr16", BinOpFrag<(sra node:$LHS, node:$RHS)>>;
def ROTL16 : R16_XZ_BINOP<0b1100, 0b11, "rotl16", BinOpFrag<(rotl node:$LHS, (and node:$RHS, 0x1f))>>;

def MULSH16 : R16_XZ_BINOP_NOPat<0b1111, 0b01, "mulsh16">;

def ZEXTB16 : R16_XZ_UNOP<0b1101, 0b00, "zextb16">;
def ZEXTH16 : R16_XZ_UNOP<0b1101, 0b01, "zexth16">;
def SEXTB16 : R16_XZ_UNOP<0b1101, 0b10, "sextb16">;
def SEXTH16 : R16_XZ_UNOP<0b1101, 0b11, "sexth16">;

let Constraints = "$rZ = $rz", isReMaterializable = 1, isAsCheapAsAMove = 1 in {
  let isAdd = 1, Pattern = [(set mGPR:$rz, (add mGPR:$rZ, oimm8:$imm8))] in
  def ADDI16 : I16_Z_8<0b100, (ins mGPR:$rZ, oimm8:$imm8), "addi16\t$rz, $imm8">;
  let Pattern = [(set mGPR:$rz, (sub mGPR:$rZ, oimm8:$imm8))] in
  def SUBI16 : I16_Z_8<0b101, (ins mGPR:$rZ, oimm8:$imm8), "subi16\t$rz, $imm8">;
}

let isAdd = 1 in
def ADDI16ZSP : I16_Z_8<0b011, (ins SPOp:$sp, uimm8_2:$imm8),
                        "addi16\t$rz, $sp, $imm8">;

let isAdd = 1 in
def ADDI16SPSP : I16_SP_IMM7<0b000,"addi16">;
def SUBI16SPSP : I16_SP_IMM7<0b001,"subi16">;

let isReMaterializable = 1, isAsCheapAsAMove = 1 in {
  def LSLI16 : I16_XZ_IMM5<0, "lsli16", shl>;
  def LSRI16 : I16_XZ_IMM5<1, "lsri16", srl>;
  def ASRI16 : I16_XZ_IMM5<2, "asri16", sra>;
}

let isAdd = 1 in
def ADDI16XZ : I16_XZ_IMM3<0b10, "addi16", add>;
def SUBI16XZ : I16_XZ_IMM3<0b11, "subi16", sub>;

let Size = 4 in
def NEG16 : CSKYPseudo<(outs mGPR:$rd), (ins mGPR:$rx), "neg16 $rd, $rx", []>;

let Size = 4 in
def RSUBI16 : CSKYPseudo<(outs mGPR:$rd),
  (ins mGPR:$rx, uimm8:$imm8), "rsubi16 $rd, $rx, $imm8", []>;

//===----------------------------------------------------------------------===//
// Load & Store instructions.
//===----------------------------------------------------------------------===//

def LD16B : I16_XZ_LDST<AddrMode16B, 0b000, "ld16.b",
  (outs mGPR:$rz), (ins mGPR:$rx, uimm5:$imm)>;
def LD16H : I16_XZ_LDST<AddrMode16H, 0b001, "ld16.h",
  (outs mGPR:$rz), (ins mGPR:$rx, uimm5_1:$imm)>;
def LD16W : I16_XZ_LDST<AddrMode16W, 0b010, "ld16.w",
  (outs mGPR:$rz), (ins mGPR:$rx, uimm5_2:$imm)>;
def ST16B : I16_XZ_LDST<AddrMode16B, 0b100, "st16.b",
  (outs), (ins mGPR:$rz, mGPR:$rx, uimm5:$imm)>;
def ST16H : I16_XZ_LDST<AddrMode16H, 0b101, "st16.h",
  (outs), (ins mGPR:$rz, mGPR:$rx, uimm5_1:$imm)>;
def ST16W : I16_XZ_LDST<AddrMode16W, 0b110, "st16.w",
  (outs), (ins mGPR:$rz, mGPR:$rx, uimm5_2:$imm)>;

def LD16WSP : I16_ZSP_LDST<AddrMode16W, 0b011, "ld16.w",
  (outs mGPR:$rz), (ins SPOp:$sp, uimm8_2:$addr)>;
def ST16WSP : I16_ZSP_LDST<AddrMode16W, 0b111, "st16.w",
  (outs), (ins mGPR:$rz, SPOp:$sp, uimm8_2:$addr)>;

//===----------------------------------------------------------------------===//
// Compare instructions.
//===----------------------------------------------------------------------===//

def CMPHS16 : R16_XY_CMP<0, "cmphs16">;
def CMPLT16 : R16_XY_CMP<1, "cmplt16">;
let isCommutable = 1 in
def CMPNE16 : R16_XY_CMP<2, "cmpne16">;


def CMPHSI16 : I16_X_CMP<0, "cmphsi16", oimm5>;
def CMPLTI16 : I16_X_CMP<1, "cmplti16", oimm5>;
def CMPLEI16 : CSKYPseudo<(outs CARRY:$ca), (ins mGPR:$rx, uimm5:$imm5),
    "cmplei16\t$rx, $imm5", []>;
def CMPNEI16 : I16_X_CMP<2, "cmpnei16", uimm5>;

//===----------------------------------------------------------------------===//
// Data move instructions.
//===----------------------------------------------------------------------===//


def MOVI16 : I16_Z_8<0b110, (ins uimm8:$imm8), "movi16\t$rz, $imm8"> {
  let isReMaterializable = 1;
  let isAsCheapAsAMove = 1;
  let isMoveImm = 1;
  let Pattern = [(set mGPR:$rz, uimm8:$imm8)];
}

def MOV16 : CSKY16Inst<AddrModeNone, (outs sGPR:$rz), (ins sGPR:$rx),
                       "mov16\t$rz, $rx", []> {
  bits<4> rz;
  bits<4> rx;
  let Inst{15,14} = 0b01;
  let Inst{13 - 10} = 0b1011;
  let Inst{9 - 6} = rz;
  let Inst{5 - 2} = rx;
  let Inst{1,0} = 0b11;
}

// MVC16 is not in "cskyv2 instructions reference manul"
def MVCV16 : CSKY16Inst<AddrModeNone,
  (outs sGPR:$rz), (ins CARRY:$ca), "mvcv16\t$rz", []> {
  bits<4> rz;
  let Inst{15,14} = 0b01;
  let Inst{13 - 10} = 0b1001;
  let Inst{9 - 6} = rz;
  let Inst{5 - 2} = 0;
  let Inst{1,0} = 0b11;
}


//===----------------------------------------------------------------------===//
// Branch and call instructions.
//===----------------------------------------------------------------------===//

let isBranch = 1, isTerminator = 1 in {
  let isBarrier = 1, isPredicable = 1 in
    def BR16 : J16<1, "br16", (ins br_symbol_16bit:$offset)>;

  def BT16 : J16_B<2, "bt16">;
  def BF16 : J16_B<3, "bf16">;
}

def JMP16 : R16_X_J<0b11100000, 0b00, "jmp16"> {
  let isBranch = 1;
  let isTerminator = 1;
  let isBarrier = 1;
  let isIndirectBranch = 1;
  let Pattern = [(brind sGPR:$rx)];
}

def JSR16 : R16_X_J<0b11101111, 0b01, "jsr16"> {
  let isCall = 1;
  let Defs = [ R15 ];
}

def RTS16 : CSKY16Inst<AddrModeNone, (outs), (ins), "rts16", [(CSKY_RET)]> {
  let isTerminator = 1;
  let isReturn = 1;
  let isBarrier = 1;
  let Inst = 0b0111100000111100;
  let Uses = [R15];
  let isCodeGenOnly = 1;
}

def JMPIX16 :  CSKY16Inst<AddrModeNone, (outs),
  (ins mGPR:$rx, uimm2_jmpix:$indeximm2), "jmpix16\t$rx, $indeximm2", []> {
  bits<3> rx;
  bits<2> indeximm2;
  let Inst{15,14} = 0b00;
  let Inst{13 - 11} = 0b111;
  let Inst{10 - 8} = rx;
  let Inst{7 - 2} = 0b111000;
  let Inst{1,0} = indeximm2;
  let Predicates = [HasJAVA];
  let Uses = [R30];
}

//===----------------------------------------------------------------------===//
// Symbol address instructions.
//===----------------------------------------------------------------------===//

def LRW16 : CSKY16Inst<AddrModeNone, (outs mGPR:$rz),
  (ins constpool_symbol_16bit:$label), "lrw16\t$rz, $label", []> {
  bits<3> rz;
  bits<8> label;
  let Inst{15 - 13} = 0b000;
  let Inst{12} = label{7};
  let Inst{11,10} = 0b00;
  let Inst{9,8} = label{6,5};
  let Inst{7 - 5} = rz;
  let Inst{4 - 0} = label{4-0};
  let mayLoad = 1;
  let mayStore = 0;
}

def LRW16_Gen : CSKY16Inst<AddrModeNone, (outs mGPR:$rz),
  (ins bare_symbol:$src, constpool_symbol_16bit:$label),
  "lrw16\t$rz, $label", []> {
  bits<3> rz;
  bits<8> label;
  let Inst{15 - 13} = 0b000;
  let Inst{12} = label{7};
  let Inst{11,10} = 0b00;
  let Inst{9,8} = label{6,5};
  let Inst{7 - 5} = rz;
  let Inst{4 - 0} = label{4-0};
  let mayLoad = 1;
  let mayStore = 0;
  let isCodeGenOnly = 1;
}


//===----------------------------------------------------------------------===//
// Other operation instructions.
//===----------------------------------------------------------------------===//

def REVB16 :  R16_XZ_UNOP<0b1110, 0b10, "revb16">;
def REVH16 :  R16_XZ_UNOP<0b1110, 0b11, "revh16">;

let isCodeGenOnly = 1 in
def SETC16 : CSKY16Inst<AddrModeNone,
  (outs CARRY:$ca), (ins), "setc16", []> {
  let Inst{15, 14} = 0b01;
  let Inst{13 - 10} = 0b1001;
  let Inst{9 - 6} = 0;
  let Inst{5 - 2} = 0;
  let Inst{1, 0} = 0;
  let isCompare = 1;
}

let isCodeGenOnly = 1 in
def CLRC16 : CSKY16Inst<AddrModeNone,
  (outs CARRY:$ca), (ins), "clrc16", []> {
  let Inst{15, 14} = 0b01;
  let Inst{13 - 10} = 0b1001;
  let Inst{9 - 6} = 0;
  let Inst{5 - 2} = 0;
  let Inst{1, 0} = 2;
  let isCompare = 1;
}

let Constraints = "$rZ = $rz" in {
  def BCLRI16 : I16_Z_5<0b100, (outs mGPR:$rz), (ins mGPR:$rZ, uimm5:$imm5),
                        "bclri16">;
  def BSETI16 : I16_Z_5<0b101, (outs mGPR:$rz), (ins mGPR:$rZ, uimm5:$imm5),
                        "bseti16">;
}

let Predicates = [HasBTST16] in
  def BTSTI16 : I16_Z_5<0b110, (outs CARRY:$ca), (ins mGPR:$rz, uimm5:$imm5),
                        "btsti16">;

def TST16 : CSKY16Inst<AddrModeNone, (outs CARRY:$ca), (ins sGPR:$rx, sGPR:$ry),
                        "tst16\t$rx, $ry", []> {
  bits<4> ry;
  bits<4> rx;
  let Inst{15,14} = 0b01;
  let Inst{13 - 10} = 0b1010;
  let Inst{9 - 6} = ry;
  let Inst{5 - 2} = rx;
  let Inst{1,0} = 0b10;
  let isCompare = 1;
}

def TSTNBZ16 : CSKY16Inst<AddrModeNone, (outs CARRY:$ca), (ins sGPR:$rx),
                          "tstnbz16\t$rx", []> {
  bits<4> rx;
  let Inst{15,14} = 0b01;
  let Inst{13 - 10} = 0b1010;
  let Inst{9 - 6} = 0b0000;
  let Inst{5 - 2} = rx;
  let Inst{1,0} = 0b11;
  let isCompare = 1;
}

//===----------------------------------------------------------------------===//
// Special instructions.
//===----------------------------------------------------------------------===//

def BKPT : CSKY16Inst<AddrModeNone, (outs), (ins), "bkpt", []> {
  let Inst = 0;
}

let mayStore = 1 in {
def BPUSHH : I16_BPushPop<0b00010100111, 0, (outs), (ins mGPR:$rz), "bpush.h $rz">;
def BPUSHW : I16_BPushPop<0b00010100111, 0b10, (outs), (ins mGPR:$rz), "bpush.w $rz">;
}

let mayLoad = 1 in {
def BPOPH : I16_BPushPop<0b00010100101, 0, (outs mGPR:$rz), (ins),  "bpop.h $rz">;
def BPOPW : I16_BPushPop<0b00010100101, 0b10, (outs mGPR:$rz), (ins), "bpop.w $rz">;
}

def NIE : CSKY16Inst<AddrModeNone, (outs), (ins), "nie", [(CSKY_NIE)]> {
  let Inst = 0b0001010001100000;
}

let isBarrier = 1, isReturn = 1, isTerminator = 1 in
def NIR : CSKY16Inst<AddrModeNone, (outs), (ins), "nir", [(CSKY_NIR)]> {
  let Inst = 0b0001010001100001;
}

def IPUSH16 : CSKY16Inst<AddrModeNone, (outs), (ins), "ipush16", []> {
  let Inst{15- 5} = 0b00010100011;
  let Inst{4-0} = 0b00010;
  let Predicates = [iHasE1];
  let Defs  = [R14];
  let Uses  = [R14, R0, R1, R2, R3, R12, R13];
  let mayStore = 1;
}

def IPOP16 : CSKY16Inst<AddrModeNone, (outs), (ins), "ipop16", []> {
  let Inst{15- 5} = 0b00010100011;
  let Inst{4-0} = 0b00011;
  let Predicates = [iHasE1];
  let Defs  = [R14, R0, R1, R2, R3, R12, R13];
  let Uses  = [R14];
  let mayLoad = 1;
}

def PUSH16 : CSKY16Inst<AddrModeNone, (outs),
  (ins reglist:$regs, variable_ops), "push16 $regs", []> {
  bits<5> regs;

  let Inst{15- 5} = 0b00010100110;
  let Inst{4-0} = regs;
  let Predicates = [iHasE1];
  let Defs  = [R14];
  let Uses  = [R14];
  let mayStore = 1;
}

def POP16 : CSKY16Inst<AddrModeNone, (outs),
  (ins reglist:$regs, variable_ops), "pop16 $regs", []> {
  bits<5> regs;

  let Inst{15- 5} = 0b00010100100;
  let Inst{4-0} = regs;
  let Predicates = [iHasE1];
  let Defs  = [R14];
  let Uses  = [R14];
  let mayLoad = 1;
}

//===----------------------------------------------------------------------===//
// CSKYPseudo
//===----------------------------------------------------------------------===//

let usesCustomInserter = 1 in  {
  def ISEL16 : CSKYPseudo<(outs sGPR:$dst),
    (ins CARRY:$cond, sGPR:$src1, sGPR:$src2),
    "!isel16\t$dst, $src1, src2",
    [(set sGPR:$dst, (select CARRY:$cond, sGPR:$src1, sGPR:$src2))]>;
}

class JBranchPseudo<dag out, dag ins, string opstr> :
  CSKYPseudo<out, ins, opstr, []> {
  let isBranch = 1;
  let isTerminator = 1;
  let isIndirectBranch = 1;
  let mayLoad = 1;
  let Size = 2;
}

let isBarrier = 1 in
def JBR16 : JBranchPseudo<(outs),
  (ins br_symbol_16bit:$src1), "jbr16\t$src1">;
def JBT16 : JBranchPseudo<(outs),
  (ins CARRY:$ca, br_symbol_16bit:$src1), "jbt16\t$src1">;
def JBF16 : JBranchPseudo<(outs),
  (ins CARRY:$ca, br_symbol_16bit:$src1), "jbf16\t$src1">;

let mayLoad = 1, Size = 2, isCodeGenOnly = 0 in
def PseudoLRW16 : CSKYPseudo<(outs mGPR:$rz),
  (ins bare_symbol:$src), "lrw16 $rz, $src", []>;
