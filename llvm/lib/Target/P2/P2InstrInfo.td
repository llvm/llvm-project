//===- P2InstrInfo.td - Target Description for P2 Target -*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the P2 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

include "P2InstrFormats.td"

/*--------------------------------*/
/* General type definitions       */
/*--------------------------------*/

def SDT_P2Call : SDTypeProfile<0, -1, [SDTCisVT<0, iPTR>]>;
def SDT_P2Mov : SDTypeProfile<1, 1, [SDTCisSameAs<0, 1>]>;
def SDT_P2CallSeqStart : SDCallSeqStart<[SDTCisVT<0, i32>, SDTCisVT<1, i32>]>;
def SDT_P2CallSeqEnd : SDCallSeqEnd<[SDTCisVT<0, i32>, SDTCisVT<1, i32>]>;

def jmptarget    : Operand<OtherVT> {
    let EncoderMethod = "getJumpTargetOpValue";
    let OperandType = "OPERAND_PCREL";
}

def jmp9target    : Operand<OtherVT> {
    let EncoderMethod = "getJump9TargetOpValue";
    let DecoderMethod = "decodeJump9Target";
    let OperandType = "OPERAND_PCREL";
}

def calltarget : Operand<iPTR> {
    let EncoderMethod = "encodeCallTarget";
}

def NegateImm : SDNodeXForm<imm, [{return CurDAG->getConstant(-N->getSExtValue(), SDLoc(N), MVT::i32, true);}]>;
def NegSubInlineConst32 : ImmLeaf<i32, [{return Imm < 0;}], NegateImm>;

// def s9imm : Operand<i32>; // 9 bit immediate

def imm9 : PatLeaf<(i32 imm), [{ // select this leaf when the immediate is between 0 and 2^9
    int32_t v = N->getZExtValue();
    return (v >= 0) && (v < (1 << 9));
}]>;

def addr : ComplexPattern<iPTR, 1, "selectAddr", [], []>;

/*--------------------------------*/
/* custom nodes                   */
/*--------------------------------*/
// this is used for loading global addresses
def P2GAWrapper    : SDNode<"P2ISD::GAWRAPPER", SDT_P2Mov>;

def P2callseq_start : SDNode<"ISD::CALLSEQ_START", SDT_P2CallSeqStart, [SDNPHasChain, SDNPOutGlue]>;
def P2callseq_end : SDNode<"ISD::CALLSEQ_END", SDT_P2CallSeqEnd, [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;

def P2RDLUT  : SDNode<"P2ISD::RDLUT", SDTLoad, [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def P2WRLUT  : SDNode<"P2ISD::WRLUT", SDTStore, [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;

def P2RET   : SDNode<"P2ISD::RET", SDTNone, [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;
def P2CALL  : SDNode<"P2ISD::CALL", SDT_P2Call, [SDNPHasChain, SDNPOutGlue, SDNPOptInGlue, SDNPVariadic]>;

def rdlut : PatFrag<(ops node:$ptr), (P2RDLUT node:$ptr)> {
    let IsNonExtLoad = 1;
    let IsUnindexed = 1;
}

def wrlut : PatFrag<(ops node:$val, node:$ptr), (P2WRLUT node:$val, node:$ptr)> {
    let IsTruncStore = 0;
    let IsUnindexed = 1;
}

/*--------------------------------*/
/* basic math and logic           */
/*--------------------------------*/

// place this first so that the below pattern matches before the standard addition pattern matches
let Constraints = "$s1 = $d" in {
    defm SUB : P2InstCZIDSm<0b0001100, (ins P2GPR:$s1), "sub">;
}

// convert add -c to sub c
def : Pat<(sub P2GPR:$src1, P2GPR:$s), (SUBrr P2GPR:$src1, P2GPR:$s, always, noeff)>;
def : Pat<(sub P2GPR:$src1, imm9:$s), (SUBri P2GPR:$src1, s9imm:$s, always, noeff)>;
def : Pat<(add i32:$d, NegSubInlineConst32:$s), (SUBri $d, NegSubInlineConst32:$s, always, noeff)>;

let Constraints = "$s1 = $d" in {
    defm ROR    : P2InstCZIDSm<0b0000000, (ins P2GPR:$s1), "ror">;
    defm ROL    : P2InstCZIDSm<0b0000001, (ins P2GPR:$s1), "rol">;
    defm SHR    : P2InstCZIDSm<0b0000010, (ins P2GPR:$s1), "shr">;
    defm SHL    : P2InstCZIDSm<0b0000011, (ins P2GPR:$s1), "shl">;
    defm RCR    : P2InstCZIDSm<0b0000100, (ins P2GPR:$s1), "rcr">;
    defm RCL    : P2InstCZIDSm<0b0000101, (ins P2GPR:$s1), "rcl">;
    defm SAR    : P2InstCZIDSm<0b0000110, (ins P2GPR:$s1), "sar">;
    defm SAL    : P2InstCZIDSm<0b0000111, (ins P2GPR:$s1), "sal">;
    defm ADD    : P2InstCZIDSm<0b0001000, (ins P2GPR:$s1), "add">;
    defm ADDX   : P2InstCZIDSm<0b0001001, (ins P2GPR:$s1), "addx">;
    defm ADDS   : P2InstCZIDSm<0b0001010, (ins P2GPR:$s1), "adds">;
    defm ADDSX  : P2InstCZIDSm<0b0001011, (ins P2GPR:$s1), "addsx">;
    defm SUBX   : P2InstCZIDSm<0b0001101, (ins P2GPR:$s1), "subx">;
    defm SUBS   : P2InstCZIDSm<0b0001110, (ins P2GPR:$s1), "subs">;
    defm SUBSX  : P2InstCZIDSm<0b0001111, (ins P2GPR:$s1), "subsx">;
    defm AND    : P2InstCZIDSm<0b0101000, (ins P2GPR:$s1), "and">;
    defm OR     : P2InstCZIDSm<0b0101010, (ins P2GPR:$s1), "or">;
    defm XOR    : P2InstCZIDSm<0b0101011, (ins P2GPR:$s1), "xor">;
    defm SIGNX  : P2InstCZIDSm<0b0111011, (ins P2GPR:$s1), "signx">;
    defm MUL    : P2InstZIDSm<0b10100000, (ins P2GPR:$s1), "mul">;
    defm MULS   : P2InstZIDSm<0b10100001, (ins P2GPR:$s1), "muls">;
}

defm NOT    : P2InstCZIDSm<0b0110001, (ins), "not">;
defm ABS    : P2InstCZIDSm<0b0110010, (ins), "abs">;
defm NEG    : P2InstCZIDSm<0b0110011, (ins), "neg">;

let Constraints = "$s = $d" in {
    def REV    : P2InstD<0b1101011, 0b001101001, (outs P2GPR:$d), (ins P2GPR:$s), "rev $d">;
}

defm ENCOD  : P2InstCZIDSm<0b0111100, (ins), "encod">;

def : Pat<(rotl P2GPR:$src1, P2GPR:$s), (ROLrr P2GPR:$src1, P2GPR:$s, always, noeff)>;
def : Pat<(rotl P2GPR:$src1, imm9:$s), (ROLri P2GPR:$src1, imm9:$s, always, noeff)>;
def : Pat<(rotr P2GPR:$src1, P2GPR:$s), (RORrr P2GPR:$src1, P2GPR:$s, always, noeff)>;
def : Pat<(rotr P2GPR:$src1, imm9:$s), (RORri P2GPR:$src1, imm9:$s, always, noeff)>;
def : Pat<(srl P2GPR:$src1, P2GPR:$s), (SHRrr P2GPR:$src1, P2GPR:$s, always, noeff)>;
def : Pat<(srl P2GPR:$src1, imm9:$s), (SHRri P2GPR:$src1, imm9:$s, always, noeff)>;
def : Pat<(shl P2GPR:$src1, P2GPR:$s), (SHLrr P2GPR:$src1, P2GPR:$s, always, noeff)>;
def : Pat<(shl P2GPR:$src1, imm9:$s), (SHLri P2GPR:$src1, imm9:$s, always, noeff)>;
def : Pat<(sra P2GPR:$src1, P2GPR:$s), (SARrr P2GPR:$src1, P2GPR:$s, always, noeff)>;
def : Pat<(sra P2GPR:$src1, imm9:$s), (SARri P2GPR:$src1, imm9:$s, always, noeff)>;
def : Pat<(add P2GPR:$src1, P2GPR:$s), (ADDrr P2GPR:$src1, P2GPR:$s, always, noeff)>;
def : Pat<(add P2GPR:$src1, imm9:$s), (ADDri P2GPR:$src1, imm9:$s, always, noeff)>;
def : Pat<(add P2GPR:$src1, P2GPR:$s), (ADDXrr P2GPR:$src1, P2GPR:$s, always, noeff)>;
def : Pat<(add P2GPR:$src1, imm9:$s), (ADDXri P2GPR:$src1, imm9:$s, always, noeff)>;
def : Pat<(and P2GPR:$src1, P2GPR:$s), (ANDrr P2GPR:$src1, P2GPR:$s, always, noeff)>;
def : Pat<(and P2GPR:$src1, imm9:$s), (ANDri P2GPR:$src1, imm9:$s, always, noeff)>;
def : Pat<(or P2GPR:$src1, P2GPR:$s), (ORrr P2GPR:$src1, P2GPR:$s, always, noeff)>;
def : Pat<(or P2GPR:$src1, imm9:$s), (ORri P2GPR:$src1, imm9:$s, always, noeff)>;
def : Pat<(xor P2GPR:$src1, P2GPR:$s), (XORrr P2GPR:$src1, P2GPR:$s, always, noeff)>;
def : Pat<(xor P2GPR:$src1, imm9:$s), (XORri P2GPR:$src1, imm9:$s, always, noeff)>;
def : Pat<(not P2GPR:$s), (NOTrr P2GPR:$s, always, noeff)>;
def : Pat<(not imm9:$s), (NOTri imm9:$s, always, noeff)>;
def : Pat<(abs P2GPR:$s), (ABSrr P2GPR:$s, always, noeff)>;
def : Pat<(abs imm9:$s), (ABSri imm9:$s, always, noeff)>;
def : Pat<(ineg P2GPR:$s), (NEGrr P2GPR:$s, always, noeff)>;
def : Pat<(ineg imm9:$s), (NEGri imm9:$s, always, noeff)>;
def : Pat<(bitreverse P2GPR:$s), (REV P2GPR:$s, always)>;

def : Pat<(sext_inreg P2GPR:$d, i1), (SIGNXri P2GPR:$d, 0, always, noeff)>;
def : Pat<(sext_inreg P2GPR:$d, i8), (SIGNXri P2GPR:$d, 7, always, noeff)>;
def : Pat<(sext_inreg P2GPR:$d, i16), (SIGNXri P2GPR:$d, 15, always, noeff)>;

/*--------------------------------*/
/* More complicated math          */
/*--------------------------------*/

let DecoderMethod = "DecodeGetQInstruction" in {
    def GETQX   : P2InstCZD<0b1101011, 0b000011000, (outs P2GPR:$d), (ins P2Implicit:$qx), "getqx $d">;
    def GETQY   : P2InstCZD<0b1101011, 0b000011001, (outs P2GPR:$d), (ins P2Implicit:$qy), "getqy $d">;
}

let DecoderMethod = "DecodeCordicInstruction" in {
    defm QMUL   : P2InstLIDSm<0b11010000, "qmul">;
    defm QDIV   : P2InstLIDSm<0b11010001, "qdiv">;
}

def QUDIV   : Pseudo<(outs P2GPR:$dst), (ins P2GPR:$src1, P2GPR:$src2), "udiv $dst, $src1, $src2">;
def QUREM   : Pseudo<(outs P2GPR:$dst), (ins P2GPR:$src1, P2GPR:$src2), "urem $dst, $src1, $src2">;

def : Pat<(udiv P2GPR:$src1, P2GPR:$src2), (QUDIV P2GPR:$src1, P2GPR:$src2)>;
def : Pat<(urem P2GPR:$src1, P2GPR:$src2), (QUREM P2GPR:$src1, P2GPR:$src2)>;

/*--------------------------------*/
/* branching and conditionals     */
/*--------------------------------*/

// how branching will work:
// define branch instruction for various compare branches (equal, not equal, etc), they will probably be all subclasses of common branch with different effects stuck to it.
// define a pattern that matched a compare instruction output inputting to one of the branch instructions for various IR conditional branch patterns

let DecoderMethod = "DecodeCmpInstruction" in {
    // don't use the multiclass for this because these don't output to d
    def CMPrr    : P2InstCZIDS<0b0010000, 0b0, (outs P2Implicit:$cmp), (ins P2GPR:$d, P2GPR:$s), "cmp $d, $s">;
    def CMPri    : P2InstCZIDS<0b0010000, 0b1, (outs P2Implicit:$cmp), (ins P2GPR:$d, s9imm:$s), "cmp $d, #$s">;
    def CMPRrr   : P2InstCZIDS<0b0010100, 0b0, (outs P2Implicit:$cmp), (ins P2GPR:$d, P2GPR:$s), "cmpr $d, $s">;
    def CMPRri   : P2InstCZIDS<0b0010100, 0b1, (outs P2Implicit:$cmp), (ins P2GPR:$d, s9imm:$s), "cmpr $d, #$s">;

    def CMPSrr    : P2InstCZIDS<0b0010010, 0b0, (outs P2Implicit:$cmp), (ins P2GPR:$d, P2GPR:$s), "cmps $d, $s">;
    def CMPSri    : P2InstCZIDS<0b0010010, 0b1, (outs P2Implicit:$cmp), (ins P2GPR:$d, s9imm:$s), "cmps $d, #$s">;
}

def JMPr        : P2InstCZD<0b1101011, 0b000101100, (outs), (ins P2GPR:$d), "jmp $d">;

let DecoderMethod = "DecodeJumpInstruction", isBarrier = 1, isBranch = 1, isTerminator = 1 in {
    def JMP     : P2InstRA<0b1101100, 0b1, (outs), (ins jmptarget:$d, P2Implicit:$cmp), "jmp #$d">; // relative address jump
    def JMPa    : P2InstRA<0b1101100, 0b0, (outs), (ins jmptarget:$d, P2Implicit:$cmp), "jmp /#$d">; // absolute address jump (llvm doesn't support using \ so use /)
}

let DecoderMethod = "DecodeTJInstruction", isBranch = 1, isBarrier = 1, isTerminator = 1 in {
    // don't use multiclass here because we don't want to use the s9imm operand type
    def TJZrr   : P2InstIDS<0b101110010, 0b0, (outs), (ins P2GPR:$d, P2GPR:$s), "tjz $d, $s">;
    def TJZri   : P2InstIDS<0b101110010, 0b1, (outs), (ins P2GPR:$d, jmp9target:$s), "tjz $d, #$s">;

    def TJNZrr  : P2InstIDS<0b101110011, 0b0, (outs), (ins P2GPR:$d, P2GPR:$s), "tjnz $d, $s">;
    def TJNZri  : P2InstIDS<0b101110011, 0b1, (outs), (ins P2GPR:$d, jmp9target:$s), "tjnz $d, #$s">;
}

let Constraints = "$s1 = $d" in {
    // don't use multiclass here because we don't want to use the s9imm operand type
    def DJNZrr  : P2InstIDS<0b101101101, 0b0, (outs P2GPR:$d), (ins P2GPR:$s1, P2GPR:$s), "djnz $d, $s">;
    def DJNZri  : P2InstIDS<0b101101101, 0b1, (outs P2GPR:$d), (ins P2GPR:$s1, jmp9target:$s), "djnz $d, #$s">;
}

defm REP    : P2InstLIDSm<0b11001101, "rep">;

defm TEST   : P2InstCZIDSm<0b0111110, (ins), "test">;

// Patterns for various branch conditions and types of branch instructions
// unsigned comparison
def : Pat<(brcc SETUEQ, P2GPR:$lhs, P2GPR:$rhs, bb:$target), (JMP bb:$target, (CMPrr P2GPR:$lhs, P2GPR:$rhs, always, wcz), if_z)>;
def : Pat<(brcc SETUEQ, P2GPR:$lhs, imm9:$rhs, bb:$target), (JMP bb:$target, (CMPri P2GPR:$lhs, imm9:$rhs, always, wcz), if_z)>;

def : Pat<(brcc SETUNE, P2GPR:$lhs, P2GPR:$rhs, bb:$target), (JMP bb:$target, (CMPrr P2GPR:$lhs, P2GPR:$rhs, always, wcz), if_nz)>;
def : Pat<(brcc SETUNE, P2GPR:$lhs, imm9:$rhs, bb:$target), (JMP bb:$target, (CMPri P2GPR:$lhs, imm9:$rhs, always, wcz), if_nz)>;

def : Pat<(brcc SETULT, P2GPR:$lhs, P2GPR:$rhs, bb:$target), (JMP bb:$target, (CMPrr P2GPR:$lhs, P2GPR:$rhs, always, wcz), if_c)>;
def : Pat<(brcc SETULT, P2GPR:$lhs, imm9:$rhs, bb:$target), (JMP bb:$target, (CMPri P2GPR:$lhs, imm9:$rhs, always, wcz), if_c)>;

def : Pat<(brcc SETUGT, P2GPR:$lhs, P2GPR:$rhs, bb:$target), (JMP bb:$target, (CMPrr P2GPR:$lhs, P2GPR:$rhs, always, wcz), if_nc_and_nz)>;
def : Pat<(brcc SETUGT, P2GPR:$lhs, imm9:$rhs, bb:$target), (JMP bb:$target, (CMPri P2GPR:$lhs, imm9:$rhs, always, wcz), if_nc_and_nz)>;

def : Pat<(brcc SETULE, P2GPR:$lhs, P2GPR:$rhs, bb:$target), (JMP bb:$target, (CMPrr P2GPR:$lhs, P2GPR:$rhs, always, wcz), if_c_or_z)>;
def : Pat<(brcc SETULE, P2GPR:$lhs, imm9:$rhs, bb:$target), (JMP bb:$target, (CMPri P2GPR:$lhs, imm9:$rhs, always, wcz), if_c_or_z)>;

def : Pat<(brcc SETUGE, P2GPR:$lhs, P2GPR:$rhs, bb:$target), (JMP bb:$target, (CMPrr P2GPR:$lhs, P2GPR:$rhs, always, wcz), if_nc)>;
def : Pat<(brcc SETUGE, P2GPR:$lhs, imm9:$rhs, bb:$target), (JMP bb:$target, (CMPri P2GPR:$lhs, imm9:$rhs, always, wcz), if_nc)>;

// signed comparison
def : Pat<(brcc SETEQ, P2GPR:$lhs, P2GPR:$rhs, bb:$target), (JMP bb:$target, (CMPSrr P2GPR:$lhs, P2GPR:$rhs, always, wcz), if_z)>;
def : Pat<(brcc SETEQ, P2GPR:$lhs, imm9:$rhs, bb:$target), (JMP bb:$target, (CMPSri P2GPR:$lhs, imm9:$rhs, always, wcz), if_z)>;

def : Pat<(brcc SETNE, P2GPR:$lhs, P2GPR:$rhs, bb:$target), (JMP bb:$target, (CMPSrr P2GPR:$lhs, P2GPR:$rhs, always, wcz), if_nz)>;
def : Pat<(brcc SETNE, P2GPR:$lhs, imm9:$rhs, bb:$target), (JMP bb:$target, (CMPSri P2GPR:$lhs, imm9:$rhs, always, wcz), if_nz)>;

def : Pat<(brcc SETLT, P2GPR:$lhs, P2GPR:$rhs, bb:$target), (JMP bb:$target, (CMPSrr P2GPR:$lhs, P2GPR:$rhs, always, wcz), if_c)>;
def : Pat<(brcc SETLT, P2GPR:$lhs, imm9:$rhs, bb:$target), (JMP bb:$target, (CMPSri P2GPR:$lhs, imm9:$rhs, always, wcz), if_c)>;

def : Pat<(brcc SETGE, P2GPR:$lhs, P2GPR:$rhs, bb:$target), (JMP bb:$target, (CMPSrr P2GPR:$lhs, P2GPR:$rhs, always, wcz), if_nc)>;
def : Pat<(brcc SETGE, P2GPR:$lhs, imm9:$rhs, bb:$target), (JMP bb:$target, (CMPSri P2GPR:$lhs, imm9:$rhs, always, wcz), if_nc)>;

def : Pat<(brcc SETGT, P2GPR:$lhs, P2GPR:$rhs, bb:$target), (JMP bb:$target, (CMPSrr P2GPR:$lhs, P2GPR:$rhs, always, wcz), if_nc_and_nz)>;
def : Pat<(brcc SETGT, P2GPR:$lhs, imm9:$rhs, bb:$target), (JMP bb:$target, (CMPSri P2GPR:$lhs, imm9:$rhs, always, wcz), if_nc_and_nz)>;

def : Pat<(brcc SETLE, P2GPR:$lhs, P2GPR:$rhs, bb:$target), (JMP bb:$target, (CMPSrr P2GPR:$lhs, P2GPR:$rhs, always, wcz), if_c_or_z)>;
def : Pat<(brcc SETLE, P2GPR:$lhs, imm9:$rhs, bb:$target), (JMP bb:$target, (CMPSri P2GPR:$lhs, imm9:$rhs, always, wcz), if_c_or_z)>;

// if (c) branch => jump if not 0
def : Pat<(brcond P2GPR:$c, bb:$target), (TJNZri P2GPR:$c, bb:$target, always)>;

// unconditional jump
def : Pat<(br bb:$target), (JMP bb:$target, 1, always)>;

// indirect branch
def : Pat<(brind P2GPR:$d), (JMPr P2GPR:$d, noeff, always)>;

// these patterns break jump analysis, so might need to do it as a pass later to replace, or update the jump analysis
// to select these instructions when appropriate
// test and jump if 0
// def : Pat<(brcc SETEQ, P2GPR:$lhs, 0, bb:$target), (TJZri P2GPR:$lhs, bb:$target, always)>;
// test and jump if not 0
// def : Pat<(brcc SETNE, P2GPR:$lhs, 0, bb:$target), (TJNZri P2GPR:$lhs, bb:$target, always)>;

/*--------------------------------*/
/* moving data                    */
/*--------------------------------*/

def SELECTCC    : Pseudo<(outs P2GPR:$d), (ins P2GPR:$lhs, s9imm:$rhs, s9imm:$t, s9imm:$f, i8imm:$cc), "selectcc $d, $lhs, $rhs, $t, $f, $cc">;

multiclass sel_cc<CondCode c, int cc> {
    def : Pat<(selectcc P2GPR:$lhs, imm9:$rhs, imm9:$t, imm9:$f, c), (SELECTCC P2GPR:$lhs, imm9:$rhs, imm9:$t, imm9:$f, cc)>;
    def : Pat<(selectcc P2GPR:$lhs, imm9:$rhs, P2GPR:$t, imm9:$f, c), (SELECTCC P2GPR:$lhs, imm9:$rhs, P2GPR:$t, imm9:$f, cc)>;
    def : Pat<(selectcc P2GPR:$lhs, imm9:$rhs, P2GPR:$t, imm9:$f, c), (SELECTCC P2GPR:$lhs, imm9:$rhs, P2GPR:$t, imm9:$f, cc)>;
    def : Pat<(selectcc P2GPR:$lhs, imm9:$rhs, P2GPR:$t, P2GPR:$f, c), (SELECTCC P2GPR:$lhs, imm9:$rhs, P2GPR:$t, P2GPR:$f, cc)>;
    def : Pat<(selectcc P2GPR:$lhs, P2GPR:$rhs, imm9:$t, imm9:$f, c), (SELECTCC P2GPR:$lhs, P2GPR:$rhs, imm9:$t, imm9:$f, cc)>;
    def : Pat<(selectcc P2GPR:$lhs, P2GPR:$rhs, imm9:$t, P2GPR:$f, c), (SELECTCC P2GPR:$lhs, P2GPR:$rhs, imm9:$t, P2GPR:$f, cc)>;
    def : Pat<(selectcc P2GPR:$lhs, P2GPR:$rhs, P2GPR:$t, imm9:$f, c), (SELECTCC P2GPR:$lhs, P2GPR:$rhs, P2GPR:$t, imm9:$f, cc)>;
    def : Pat<(selectcc P2GPR:$lhs, P2GPR:$rhs, P2GPR:$t, P2GPR:$f, c), (SELECTCC P2GPR:$lhs, P2GPR:$rhs, P2GPR:$t, P2GPR:$f, cc)>;
}

defm SELECTueq   : sel_cc<SETUEQ, 0>;
defm SELECTune   : sel_cc<SETUNE, 1>;
defm SELECTule   : sel_cc<SETULE, 2>;
defm SELECTult   : sel_cc<SETULT, 3>;
defm SELECTugt   : sel_cc<SETUGT, 4>;
defm SELECTuge   : sel_cc<SETUGE, 5>;
defm SELECTeq   : sel_cc<SETEQ, 6>;
defm SELECTne   : sel_cc<SETNE, 7>;
defm SELECTle   : sel_cc<SETLE, 8>;
defm SELECTlt   : sel_cc<SETLT, 9>;
defm SELECTgt   : sel_cc<SETGT, 10>;
defm SELECTge   : sel_cc<SETGE, 11>;

def : Pat<(select P2GPR:$c, P2GPR:$t, P2GPR:$f), (SELECTCC P2GPR:$c, 0, P2GPR:$t, P2GPR:$f, 1)>;
def : Pat<(select P2GPR:$c, P2GPR:$t, imm9:$f), (SELECTCC P2GPR:$c, 0, P2GPR:$t, imm9:$f, 1)>;
def : Pat<(select P2GPR:$c, imm9:$t, P2GPR:$f), (SELECTCC P2GPR:$c, 0, imm9:$t, P2GPR:$f, 1)>;
def : Pat<(select P2GPR:$c, imm9:$t, imm9:$f), (SELECTCC P2GPR:$c, 0, imm9:$t, imm9:$f, 1)>;

// for immediate moves, use a psuedo instruction that will insert an AUGS. Later we will write an optimizer pass
// that will replace the pattern of augs #imm32, mov <r> #imm9, <any instruction operating on r> to remove the
// mov <r> call, change augs to s or d as needed, and change the final instruction to use the #imm9 instead.
// a second pass should remove any augs/d #0 calls that are not used in relocation calls

def MOVri32     : Pseudo<(outs P2GPR:$d), (ins i32imm:$s), "mov $d, ##s">;

defm MOV        : P2InstCZIDSm<0b0110000, (ins), "mov">;

def : Pat<(imm:$s), (MOVri32 imm:$s)>;
def : Pat<(imm9:$s), (MOVri imm9:$s, always, noeff)>;

defm RDBYTE     : P2InstCZIDSm<0b1010110, (ins), "rdbyte">;
defm RDWORD     : P2InstCZIDSm<0b1010111, (ins), "rdword">;
defm RDLONG     : P2InstCZIDSm<0b1011000, (ins), "rdlong">;

defm WRBYTE     : P2InstLIDSm<0b11000100, "wrbyte">;
defm WRWORD     : P2InstLIDSm<0b11000101, "wrword">;
defm WRLONG     : P2InstLIDSm<0b11000110, "wrlong">;

def RFBYTE      : P2InstCZD<0b1101011, 0b000010000, (outs P2GPR:$d), (ins), "rfbyte $d">;
def RFWORD      : P2InstCZD<0b1101011, 0b000010001, (outs P2GPR:$d), (ins), "rfword $d">;
def RFLONG      : P2InstCZD<0b1101011, 0b000010010, (outs P2GPR:$d), (ins), "rflong $d">;

defm SETQ       : P2InstLDm<0b1101011, 0b000101000, "setq">;

defm RDLUT      : P2InstCZIDSm<0b1010101, (ins), "rdlut">;
defm WRLUT      : P2InstLIDSm<0b11000011, "wrlut">;

defm RDFAST     : P2InstLIDSm<0b11000111, "rdfast">;
defm WRFAST     : P2InstLIDSm<0b11001000, "wrfast">;

// these have an output and no input, leave without multiclass for now
def WFBYTEr     : P2InstLD<0b1101011, 0b000010101, 0b0, (outs P2GPR:$d), (ins), "wfbyte $d">;
def WFBYTEi     : P2InstLD<0b1101011, 0b000010101, 0b1, (outs s9imm:$d), (ins), "wfbyte #$d">;
def WFWORDr     : P2InstLD<0b1101011, 0b000010110, 0b0, (outs P2GPR:$d), (ins), "wfword $d">;
def WFWORDi     : P2InstLD<0b1101011, 0b000010110, 0b1, (outs s9imm:$d), (ins), "wfword #$d">;
def WFLONGr     : P2InstLD<0b1101011, 0b000010111, 0b0, (outs P2GPR:$d), (ins), "wflong $d">;
def WFLONGi     : P2InstLD<0b1101011, 0b000010111, 0b1, (outs s9imm:$d), (ins), "wflong #$d">;

def : Pat<(zextloadi8 imm9:$s), (RDBYTEri imm9:$s, always, noeff)>;
def : Pat<(zextloadi8 addr:$s), (RDBYTErr P2GPR:$s, always, noeff)>;
def : Pat<(zextloadi16 imm9:$s), (RDWORDri imm9:$s, always, noeff)>;
def : Pat<(zextloadi16 addr:$s), (RDWORDrr P2GPR:$s, always, noeff)>;
def : Pat<(load imm9:$s), (RDLONGri imm9:$s, always, noeff)>;
def : Pat<(load addr:$s), (RDLONGrr P2GPR:$s, always, noeff)>;

def : Pat<(truncstorei8 P2GPR:$d, addr:$s), (WRBYTErr P2GPR:$d, P2GPR:$s, always)>;
def : Pat<(truncstorei8 imm9:$d, addr:$s), (WRBYTEir imm9:$d, P2GPR:$s, always)>;
def : Pat<(truncstorei16 P2GPR:$d, addr:$s), (WRWORDrr P2GPR:$d, P2GPR:$s, always)>;
def : Pat<(truncstorei16 imm9:$d, addr:$s), (WRWORDir imm9:$d, P2GPR:$s, always)>;
def : Pat<(store P2GPR:$d, imm9:$s), (WRLONGri P2GPR:$d, imm9:$s, always)>;
def : Pat<(store imm9:$d, imm9:$s), (WRLONGii imm9:$d, imm9:$s, always)>;
def : Pat<(store P2GPR:$d, addr:$s), (WRLONGrr P2GPR:$d, P2GPR:$s, always)>;
def : Pat<(store imm9:$d, addr:$s), (WRLONGir imm9:$d, P2GPR:$s, always)>;

def : Pat<(extloadi8 addr:$s), (RDBYTErr P2GPR:$s, always, noeff)>;
def : Pat<(extloadi16 addr:$s), (RDWORDrr P2GPR:$s, always, noeff)>;

def : Pat<(sextloadi1 addr:$s), (SIGNXri (RDBYTErr P2GPR:$s, always, noeff), 1, always, noeff)>;
def : Pat<(sextloadi8 addr:$s), (SIGNXri (RDBYTErr P2GPR:$s, always, noeff), 7, always, noeff)>;
def : Pat<(sextloadi16 addr:$s), (SIGNXri (RDWORDrr P2GPR:$s, always, noeff), 15, always, noeff)>;

/*--------------------------------*/
/* Register indirection           */
/*--------------------------------*/

let hasSideEffects = 1 in {
    defm ALTS   : P2InstIDSm<0b100110010, "alts">;
}

/*--------------------------------*/
/* function calling and globals   */
/*--------------------------------*/

let isTerminator = 1, isReturn = 1, isBarrier = 1 in {
    def RET     : P2InstCZ<0b11010111, 0b000000000, 0b000101101, (outs), (ins), "ret">;
    def RETA    : P2InstCZ<0b11010111, 0b000000000, 0b000101110, (outs), (ins), "reta">; // primarily use this for returns
}

let isCall = 1, Uses = [PTRA], DecoderMethod = "DecodeCallInstruction" in {
    def CALL    : P2InstRA<0b1101101, 0b1, (outs), (ins calltarget:$a), "call #$a">; // use this for relative function calls in cog mode.
    def CALLa   : P2InstRA<0b1101101, 0b0, (outs), (ins calltarget:$a), "call /#$a">; // use this for absolute function calls in cog mode.
    def CALLAa   : P2InstRA<0b1101110, 0b0, (outs), (ins calltarget:$a), "calla #$a">; // mainly use this for direct function calls
    def CALLr    : P2InstCZD<0b1101011, 0b000101101, (outs), (ins P2GPR:$d), "call $d">;
    def CALLAr   : P2InstCZD<0b1101011, 0b000101110, (outs), (ins P2GPR:$d), "calla $d">; // mainly use this for function calls with register input
}

let Defs = [PTRA], Uses = [PTRA] in {
    def ADJCALLSTACKUP      : Pseudo<(outs), (ins i32imm:$amt1, i32imm:$amt2), "ADJCALLSTACKUP">;
    def ADJCALLSTACKDOWN    : Pseudo<(outs), (ins i32imm:$amt, i32imm:$amt2), "ADJCALLSTACKDOWN">;
}

def : Pat<(P2RET), (RETA always, noeff)>;
def : Pat<(P2callseq_start timm:$amt1, timm:$amt2), (ADJCALLSTACKUP i32imm:$amt1, i32imm:$amt2)>;
def : Pat<(P2callseq_end timm:$amt1, timm:$amt2), (ADJCALLSTACKDOWN i32imm:$amt1, i32imm:$amt2)>;

def : Pat<(P2GAWrapper tglobaladdr:$addr), (MOVri32 tglobaladdr:$addr)>;
def : Pat<(P2GAWrapper tjumptable:$addr), (MOVri32 tjumptable:$addr)>;

def : Pat<(P2CALL (i32 tglobaladdr:$dst)), (CALLAa tglobaladdr:$dst, always)>;
def : Pat<(P2CALL (i32 texternalsym:$dst)), (CALLAa texternalsym:$dst, always)>;
def : Pat<(P2CALL (i32 P2GPR:$d)), (CALLAr P2GPR:$d, always, noeff)>;

/*--------------------------------*/
/* I/O and hardware control       */
/*--------------------------------*/

let DecoderMethod = "DecodeIOInstruction" in {

    defm DIRL       : P2InstCZLDm<0b1101011, 0b001000000, "dirl">;
    defm DIRH       : P2InstCZLDm<0b1101011, 0b001000001, "dirh">;
    defm OUTL       : P2InstCZLDm<0b1101011, 0b001001000, "outl">;
    defm OUTH       : P2InstCZLDm<0b1101011, 0b001001001, "outh">;
    defm OUTNOT     : P2InstCZLDm<0b1101011, 0b001001111, "outnot">;
    defm DRVL       : P2InstCZLDm<0b1101011, 0b001011000, "drvl">;
    defm DRVH       : P2InstCZLDm<0b1101011, 0b001011001, "drvh">;
    defm DRVNOT     : P2InstCZLDm<0b1101011, 0b001011111, "drvnot">;
    let cz = 0b10 in defm TESTPc      : P2InstCZLDm<0b1101011, 0b001000000, "testp">;
    let cz = 0b01 in defm TESTPz      : P2InstCZLDm<0b1101011, 0b001000000, "testp">;
}

// no multiclass, since this is an output instead of an input
def COGIDr      : P2InstCLD<0b1101011, 0b000000001, 0b0, (outs P2GPR:$d), (ins), "cogid $d">;
def GETCT       : P2InstCZD<0b1101011, 0b000011010, (outs P2GPR:$d), (ins), "getct $d">;

defm COGINIT    : P2InstCLIDSm<0b1100111, "coginit">;
defm WAITX      : P2InstLDm<0b1101011, 0b000011111, "waitx">;
defm HUBSET     : P2InstLDm<0b1101011, 0b000000000, "hubset">;

defm WRPIN      : P2InstLIDSm<0b11000000, "wrpin">;
defm WXPIN      : P2InstLIDSm<0b11000001, "wxpin">;
defm WYPIN      : P2InstLIDSm<0b11000010, "wypin">;

// no multiclass, since there's an output
def RQPINrr     : P2InstCIDS<0b10101000, 0b0, (outs P2GPR:$d), (ins P2GPR:$s), "rqpin $d, $s">;
def RQPINri     : P2InstCIDS<0b10101000, 0b1, (outs P2GPR:$d), (ins s9imm:$s), "rqpin $d, #$s">;
def RDPINrr     : P2InstCIDS<0b10101001, 0b0, (outs P2GPR:$d), (ins P2GPR:$s), "rdpin $d, $s">;
def RDPINri     : P2InstCIDS<0b10101001, 0b1, (outs P2GPR:$d), (ins s9imm:$s), "rdpin $d, #$s">;

def LOCKNEW     : P2InstCLD<0b1101011, 0b000000100, 0b0, (outs P2GPR:$d), (ins), "locknew $d">; // this instruction has no L=1 variant
defm LOCKRET    : P2InstLDm<0b1101011, 0b000000101, "lockret">;
defm LOCKTRY    : P2InstCLDm<0b1101011, 0b000000110, "locktry">;
defm LOCKREL    : P2InstCLDm<0b1101011, 0b000000111, "lockrel">;

defm SETXFRQ    : P2InstLDm<0b1101011, 0b000011101, "setxfrq">;

defm XINIT      : P2InstLIDSm<0b11001010, "xinit">;
defm XCONT      : P2InstLIDSm<0b11001100, "xcont">;

/*--------------------------------*/
/* Interrupts and events          */
/*--------------------------------*/

defm ADDCT1     : P2InstIDSm<0b101001100, "addct1">;
defm ADDCT2     : P2InstIDSm<0b101001101, "addct2">;
defm ADDCT3     : P2InstIDSm<0b101001110, "addct3">;

def WAITCT1     : P2InstCZ<0b11010110, 0b000010001, 0b000100100, (outs), (ins), "waitct1">;
def WAITCT2     : P2InstCZ<0b11010110, 0b000010010, 0b000100100, (outs), (ins), "waitct2">;
def WAITCT3     : P2InstCZ<0b11010110, 0b000010011, 0b000100100, (outs), (ins), "waitct3">;

def GETBRK      : P2InstCZD<0b1101011, 0b000110101, (outs P2GPR:$d), (ins), "getbrk $d">;
defm COGBRK     : P2InstLDm<0b1101011, 0b000110101, "cogbrk">;
defm BRK        : P2InstLDm<0b1101011, 0b000110110, "brk">;

/*--------------------------------*/
/* misc instructions              */
/*--------------------------------*/

let Uses = [SW] in {
    def WRC         : P2InstD<0b1101011, 0b001101100, (outs P2GPR:$d), (ins), "wrc $d">;
    def WRNC        : P2InstD<0b1101011, 0b001101101, (outs P2GPR:$d), (ins), "wrnc $d">;
    def WRZ         : P2InstD<0b1101011, 0b001101110, (outs P2GPR:$d), (ins), "wrz $d">;
    def WRNZ        : P2InstD<0b1101011, 0b001101111, (outs P2GPR:$d), (ins), "wrnz $d">;
}

def NOP         : P2InstNOP;
def AUGS        : P2InstN<0b11110, (outs), (ins i32imm:$n), "augs #$n">;
def AUGD        : P2InstN<0b11111, (outs), (ins i32imm:$n), "augd #$n">;

def FRMIDX      : Pseudo<(outs P2GPR:$dst), (ins P2GPR:$src), "frmidx $dst, $src">;

/*--------------------------------*/
/* Optimization patterns and such */
/*--------------------------------*/