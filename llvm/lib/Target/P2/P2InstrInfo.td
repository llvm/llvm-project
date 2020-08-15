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

// TODO: create multiclasses to reduce the number of repetitions when definition immediate versions of instructions

include "P2InstrFormats.td"

/*--------------------------------*/
/* General type definitions       */
/*--------------------------------*/

def SDT_P2Call : SDTypeProfile<0, -1, [SDTCisVT<0, iPTR>]>;
def SDT_P2Mov : SDTypeProfile<1, 1, [SDTCisSameAs<0, 1>]>;
//def SDT_P2GETQ : SDTypeProfile<1, 1, []>;
//def SDT_P2QMUL : SDTypeProfile<0, 2, []>;
def SDT_P2CallSeqStart : SDCallSeqStart<[SDTCisVT<0, i32>, SDTCisVT<1, i32>]>;
def SDT_P2CallSeqEnd : SDCallSeqEnd<[SDTCisVT<0, i32>, SDTCisVT<1, i32>]>;

def jmptarget    : Operand<OtherVT> {
    let EncoderMethod = "getJumpTargetOpValue";
    let OperandType = "OPERAND_PCREL";
}

def calltarget : Operand<iPTR> {
    let EncoderMethod = "encodeCallTarget";
}

def NegateImm : SDNodeXForm<imm, [{return CurDAG->getConstant(-N->getSExtValue(), SDLoc(N), MVT::i32, true);}]>;
def NegSubInlineConst32 : ImmLeaf<i32, [{return Imm < 0;}], NegateImm>;

def s9imm : Operand<i32>; // 9 bit immediate

def imm9 : PatLeaf<(i32 imm), [{ // select this leaf when the immediate is between 0 and 2^9
    int32_t v = N->getZExtValue();
    return (v >= 0) && (v < (1 << 9));
}]>;

def P2MemAsmOperand : AsmOperandClass {
  let Name = "Mem";
}

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
let Constraints = "$src1 = $d" in {
    def SUBrr   : P2InstCZIDS<0b0001100, 0b000, always, (outs P2GPR:$d), (ins P2GPR:$src1, P2GPR:$s), "sub $d, $s", [(set P2GPR:$d, (sub P2GPR:$src1, P2GPR:$s))]>;
    def SUBri   : P2InstCZIDS<0b0001100, 0b001, always, (outs P2GPR:$d), (ins P2GPR:$src1, s9imm:$s), "sub $d, #$s", [(set P2GPR:$d, (sub P2GPR:$src1, imm9:$s))]>;
}

// convert add -c to sub c
def : Pat<(add i32:$d, NegSubInlineConst32:$s), (SUBri $d, NegSubInlineConst32:$s)>;

let Constraints = "$src1 = $d" in {
    def RORrr   : P2InstCZIDS<0b0000000, 0b000, always, (outs P2GPR:$d), (ins P2GPR:$src1, P2GPR:$s), "ror $d, $s", []>;
    def RORri   : P2InstCZIDS<0b0000000, 0b001, always, (outs P2GPR:$d), (ins P2GPR:$src1, s9imm:$s), "ror $d, #$s", []>;

    def ROLrr   : P2InstCZIDS<0b0000001, 0b000, always, (outs P2GPR:$d), (ins P2GPR:$src1, P2GPR:$s), "rol $d, $s", []>;
    def ROLri   : P2InstCZIDS<0b0000001, 0b001, always, (outs P2GPR:$d), (ins P2GPR:$src1, s9imm:$s), "rol $d, #$s", []>;

    def SHRrr   : P2InstCZIDS<0b0000010, 0b000, always, (outs P2GPR:$d), (ins P2GPR:$src1, P2GPR:$s), "shr $d, $s", [(set P2GPR:$d, (srl P2GPR:$src1, P2GPR:$s))]>;
    def SHRri   : P2InstCZIDS<0b0000010, 0b001, always, (outs P2GPR:$d), (ins P2GPR:$src1, s9imm:$s), "shr $d, #$s", [(set P2GPR:$d, (srl P2GPR:$src1, imm9:$s))]>;

    def SHLrr   : P2InstCZIDS<0b0000011, 0b000, always, (outs P2GPR:$d), (ins P2GPR:$src1, P2GPR:$s), "shl $d, $s", [(set P2GPR:$d, (shl P2GPR:$src1, P2GPR:$s))]>;
    def SHLri   : P2InstCZIDS<0b0000011, 0b001, always, (outs P2GPR:$d), (ins P2GPR:$src1, s9imm:$s), "shl $d, #$s", [(set P2GPR:$d, (shl P2GPR:$src1, imm9:$s))]>;

    def RCRrr   : P2InstCZIDS<0b0000100, 0b000, always, (outs P2GPR:$d), (ins P2GPR:$src1, P2GPR:$s), "rcr $d, $s", []>;
    def RCRri   : P2InstCZIDS<0b0000100, 0b001, always, (outs P2GPR:$d), (ins P2GPR:$src1, s9imm:$s), "rcr $d, #$s", []>;

    def RCLrr   : P2InstCZIDS<0b0000101, 0b000, always, (outs P2GPR:$d), (ins P2GPR:$src1, P2GPR:$s), "rcl $d, $s", []>;
    def RCLri   : P2InstCZIDS<0b0000101, 0b001, always, (outs P2GPR:$d), (ins P2GPR:$src1, s9imm:$s), "rcl $d, #$s", []>;

    def SARrr   : P2InstCZIDS<0b0000110, 0b000, always, (outs P2GPR:$d), (ins P2GPR:$src1, P2GPR:$s), "sar $d, $s", [(set P2GPR:$d, (sra P2GPR:$src1, P2GPR:$s))]>;
    def SARri   : P2InstCZIDS<0b0000110, 0b001, always, (outs P2GPR:$d), (ins P2GPR:$src1, s9imm:$s), "sar $d, #$s", [(set P2GPR:$d, (sra P2GPR:$src1, imm9:$s))]>;

    def SALrr   : P2InstCZIDS<0b0000111, 0b000, always, (outs P2GPR:$d), (ins P2GPR:$src1, P2GPR:$s), "sal $d, $s", []>;
    def SALri   : P2InstCZIDS<0b0000111, 0b001, always, (outs P2GPR:$d), (ins P2GPR:$src1, s9imm:$s), "sal $d, #$s", []>;

    def ADDrr   : P2InstCZIDS<0b0001000, 0b000, always, (outs P2GPR:$d), (ins P2GPR:$src1, P2GPR:$s), "add $d, $s", [(set P2GPR:$d, (add P2GPR:$src1, P2GPR:$s))]>;
    def ADDri   : P2InstCZIDS<0b0001000, 0b001, always, (outs P2GPR:$d), (ins P2GPR:$src1, s9imm:$s), "add $d, #$s", [(set P2GPR:$d, (add P2GPR:$src1, imm9:$s))]>;

    def ADDXrr  : P2InstCZIDS<0b0001001, 0b000, always, (outs P2GPR:$d), (ins P2GPR:$src1, P2GPR:$s), "addx $d, $s", [(set P2GPR:$d, (add P2GPR:$src1, P2GPR:$s))]>;
    def ADDXri  : P2InstCZIDS<0b0001001, 0b001, always, (outs P2GPR:$d), (ins P2GPR:$src1, s9imm:$s), "addx $d, #$s", [(set P2GPR:$d, (add P2GPR:$src1, imm9:$s))]>;

    def ADDSrr  : P2InstCZIDS<0b0001010, 0b000, always, (outs P2GPR:$d), (ins P2GPR:$src1, P2GPR:$s), "adds $d, $s", []>;
    def ADDSri  : P2InstCZIDS<0b0001010, 0b001, always, (outs P2GPR:$d), (ins P2GPR:$src1, s9imm:$s), "adds $d, #$s", [(set P2GPR:$d, (add P2GPR:$src1, imm9:$s))]>;

    def ADDSXrr : P2InstCZIDS<0b0001011, 0b000, always, (outs P2GPR:$d), (ins P2GPR:$src1, P2GPR:$s), "addsx $d, $s", []>;
    def ADDSXri : P2InstCZIDS<0b0001011, 0b001, always, (outs P2GPR:$d), (ins P2GPR:$src1, s9imm:$s), "addsx $d, #$s", []>;

    def SUBXrr  : P2InstCZIDS<0b0001101, 0b000, always, (outs P2GPR:$d), (ins P2GPR:$src1, P2GPR:$s), "subx $d, $s", []>;
    def SUBXri  : P2InstCZIDS<0b0001101, 0b001, always, (outs P2GPR:$d), (ins P2GPR:$src1, s9imm:$s), "subx $d, #$s", []>;

    def SUBSrr  : P2InstCZIDS<0b0001110, 0b000, always, (outs P2GPR:$d), (ins P2GPR:$src1, P2GPR:$s), "subs $d, $s", []>;
    def SUBSri  : P2InstCZIDS<0b0001110, 0b001, always, (outs P2GPR:$d), (ins P2GPR:$src1, s9imm:$s), "subs $d, #$s", []>;

    def SUBSXrr : P2InstCZIDS<0b0001111, 0b000, always, (outs P2GPR:$d), (ins P2GPR:$src1, P2GPR:$s), "subsx $d, $s", []>;
    def SUBSXri : P2InstCZIDS<0b0001111, 0b001, always, (outs P2GPR:$d), (ins P2GPR:$src1, s9imm:$s), "subsx $d, #$s", []>;

    def ANDrr   : P2InstCZIDS<0b0101000, 0b000, always, (outs P2GPR:$d), (ins P2GPR:$src1, P2GPR:$s), "and $d, $s", [(set P2GPR:$d, (and P2GPR:$src1, P2GPR:$s))]>;
    def ANDri   : P2InstCZIDS<0b0101000, 0b001, always, (outs P2GPR:$d), (ins P2GPR:$src1, s9imm:$s), "and $d, #$s", [(set P2GPR:$d, (and P2GPR:$src1, imm9:$s))]>;

    def ORrr    : P2InstCZIDS<0b0101010, 0b000, always, (outs P2GPR:$d), (ins P2GPR:$src1, P2GPR:$s), "or $d, $s", [(set P2GPR:$d, (or P2GPR:$src1, P2GPR:$s))]>;
    def ORri    : P2InstCZIDS<0b0101010, 0b001, always, (outs P2GPR:$d), (ins P2GPR:$src1, s9imm:$s), "or $d, #$s", [(set P2GPR:$d, (or P2GPR:$src1, imm9:$s))]>;

    def XORrr   : P2InstCZIDS<0b0101011, 0b000, always, (outs P2GPR:$d), (ins P2GPR:$src1, P2GPR:$s), "xor $d, $s", [(set P2GPR:$d, (xor P2GPR:$src1, P2GPR:$s))]>;
    def XORri   : P2InstCZIDS<0b0101011, 0b001, always, (outs P2GPR:$d), (ins P2GPR:$src1, s9imm:$s), "xor $d, #$s", [(set P2GPR:$d, (xor P2GPR:$src1, imm9:$s))]>;

    def MULrr   : P2InstZIDS<0b10100000, 0b00, always, (outs P2GPR:$d), (ins P2GPR:$src1, P2GPR:$s), "mul $d, $s", []>;
    def MULri   : P2InstZIDS<0b10100000, 0b01, always, (outs P2GPR:$d), (ins P2GPR:$src1, s9imm:$s), "mul $d, #$s", []>;

    def MULSrr   : P2InstZIDS<0b10100001, 0b00, always, (outs P2GPR:$d), (ins P2GPR:$src1, P2GPR:$s), "muls $d, $s", []>;
    def MULSri   : P2InstZIDS<0b10100001, 0b01, always, (outs P2GPR:$d), (ins P2GPR:$src1, s9imm:$s), "muls $d, #$s", []>;

    def SIGNXrr : P2InstCZIDS<0b0111011, 0b000, always, (outs P2GPR:$d), (ins P2GPR:$src1, P2GPR:$s), "signx $d, $s", []>;
    def SIGNXri : P2InstCZIDS<0b0111011, 0b001, always, (outs P2GPR:$d), (ins P2GPR:$src1, s9imm:$s), "signx $d, #$s", []>;
}

def NOTrr   : P2InstCZIDS<0b0110001, 0b000, always, (outs P2GPR:$d), (ins P2GPR:$s), "not $d, $s", [(set P2GPR:$d, (not P2GPR:$s))]>;
def NOTri   : P2InstCZIDS<0b0110001, 0b001, always, (outs P2GPR:$d), (ins s9imm:$s), "not $d, #$s", [(set P2GPR:$d, (not imm9:$s))]>;

def ABSrr   : P2InstCZIDS<0b0110010, 0b000, always, (outs P2GPR:$d), (ins P2GPR:$s), "abs $d, $s", [(set P2GPR:$d, (abs P2GPR:$s))]>;
def ABSri   : P2InstCZIDS<0b0110010, 0b001, always, (outs P2GPR:$d), (ins s9imm:$s), "abs $d, #$s", [(set P2GPR:$d, (abs imm9:$s))]>;

def NEGrr   : P2InstCZIDS<0b0110011, 0b000, always, (outs P2GPR:$d), (ins P2GPR:$s), "neg $d, $s", [(set P2GPR:$d, (ineg P2GPR:$s))]>;
def NEGri   : P2InstCZIDS<0b0110011, 0b001, always, (outs P2GPR:$d), (ins s9imm:$s), "neg $d, #$s", [(set P2GPR:$d, (ineg imm9:$s))]>;

def REVr    : P2InstD<0b1101011, 0b001101001, always, (outs P2GPR:$d), (ins P2GPR:$s), "rev $d", [(set P2GPR:$d, (bitreverse P2GPR:$s))]>;

def ENCODrr : P2InstCZIDS<0b0111100, 0b000, always, (outs P2GPR:$d), (ins P2GPR:$s), "encod $d, $s", []>;

def : Pat<(sext_inreg P2GPR:$d, i1), (SIGNXri P2GPR:$d, 0)>;
def : Pat<(sext_inreg P2GPR:$d, i8), (SIGNXri P2GPR:$d, 7)>;
def : Pat<(sext_inreg P2GPR:$d, i16), (SIGNXri P2GPR:$d, 15)>;

/*--------------------------------*/
/* More complicated math          */
/*--------------------------------*/

def GETQX   : P2InstCZD<0b1101011, 0b000011000, 0b00, always, (outs P2GPR:$d), (ins P2Implicit:$qx), "getqx $d", []>;
def GETQY   : P2InstCZD<0b1101011, 0b000011001, 0b00, always, (outs P2GPR:$d), (ins P2Implicit:$qy), "getqy $d", []>;

let DecoderMethod = "DecodeCordicInstruction" in {
    def QMULrr    : P2InstLIDS<0b11010000, 0b00, always, (outs), (ins P2GPR:$d, P2GPR:$s), "qmul $d, $s", []>;
    def QMULri    : P2InstLIDS<0b11010000, 0b01, always, (outs), (ins P2GPR:$d, s9imm:$s), "qmul $d, #$s", []>;
    def QMULir    : P2InstLIDS<0b11010000, 0b10, always, (outs), (ins s9imm:$d, P2GPR:$s), "qmul #$d, $s", []>;
    def QMULii    : P2InstLIDS<0b11010000, 0b11, always, (outs), (ins s9imm:$d, s9imm:$s), "qmul #$d, #$s", []>;

    def QDIVrr    : P2InstLIDS<0b11010001, 0b00, always, (outs), (ins P2GPR:$d, P2GPR:$s), "qdiv $d, $s", []>;
    def QDIVri    : P2InstLIDS<0b11010001, 0b01, always, (outs), (ins P2GPR:$d, s9imm:$s), "qdiv $d, #$s", []>;
    def QDIVir    : P2InstLIDS<0b11010001, 0b10, always, (outs), (ins s9imm:$d, P2GPR:$s), "qdiv #$d, $s", []>;
    def QDIVii    : P2InstLIDS<0b11010001, 0b11, always, (outs), (ins s9imm:$d, s9imm:$s), "qdiv #$d, #$s", []>;
}

def QUDIV   : Pseudo<(outs P2GPR:$dst), (ins P2GPR:$src1, P2GPR:$src2), "udiv $dst, $src1, $src2", [(set P2GPR:$dst, (udiv P2GPR:$src1, P2GPR:$src2))]>;
def QUREM   : Pseudo<(outs P2GPR:$dst), (ins P2GPR:$src1, P2GPR:$src2), "urem $dst, $src1, $src2", [(set P2GPR:$dst, (urem P2GPR:$src1, P2GPR:$src2))]>;

/*--------------------------------*/
/* branching and conditionals     */
/*--------------------------------*/

// how branching will work:
// define branch instruction for various compare branches (equal, not equal, etc), they will probably be all subclasses of common branch with different effects stuck to it.
// define a pattern that matched a compare instruction output inputting to one of the branch instructions for various IR conditional branch patterns

let DecoderMethod = "DecodeCmpInstruction" in {
    def CMPrr    : P2InstCZIDS<0b0010000, 0b110, always, (outs P2Implicit:$dst), (ins P2GPR:$d, P2GPR:$s), "cmp $d, $s\twcz", []>;
    def CMPri    : P2InstCZIDS<0b0010000, 0b111, always, (outs P2Implicit:$dst), (ins P2GPR:$d, s9imm:$s), "cmp $d, #$s\twcz", []>;
    def CMPRrr   : P2InstCZIDS<0b0010100, 0b110, always, (outs P2Implicit:$dst), (ins P2GPR:$d, P2GPR:$s), "cmpr $d, $s\twcz", []>;
    def CMPRri   : P2InstCZIDS<0b0010100, 0b111, always, (outs P2Implicit:$dst), (ins P2GPR:$d, s9imm:$s), "cmpr $d, #$s\twcz", []>;

    def CMPSrr    : P2InstCZIDS<0b0010010, 0b110, always, (outs P2Implicit:$dst), (ins P2GPR:$d, P2GPR:$s), "cmp $d, $s\twcz", []>;
    def CMPSri    : P2InstCZIDS<0b0010010, 0b111, always, (outs P2Implicit:$dst), (ins P2GPR:$d, s9imm:$s), "cmp $d, #$s\twcz", []>;
    // dont' have a signed compare with reversed operands, so need to get creative
}

def JMPr    : P2InstCZD<0b1101011, 0b000101100, 0b00, always, (outs), (ins P2GPR:$d), "jmp $d", [(brind P2GPR:$d)]>;

let DecoderMethod = "DecodeJumpInstruction" in {
    def JMP     : P2BranchInst<(outs), (ins jmptarget:$target), always, [(br bb:$target)]>;
    def JMPeq   : P2BranchInst<(outs), (ins jmptarget:$target, P2Implicit:$cmp), if_z, []>;
    def JMPne   : P2BranchInst<(outs), (ins jmptarget:$target, P2Implicit:$cmp), if_nz, []>;
    def JMPlt   : P2BranchInst<(outs), (ins jmptarget:$target, P2Implicit:$cmp), if_c, []>;
    def JMPgte  : P2BranchInst<(outs), (ins jmptarget:$target, P2Implicit:$cmp), if_nc, []>;
    def JMPgt   : P2BranchInst<(outs), (ins jmptarget:$target, P2Implicit:$cmp), if_nc_and_nz, []>;
    def JMPlte  : P2BranchInst<(outs), (ins jmptarget:$target, P2Implicit:$cmp), if_c_or_z, []>;
}

def TJZrr   : P2InstIDS<0b101110010, 0b0, always, (outs), (ins P2GPR:$d, P2GPR:$s), "tjz $d, $s", []>;
def TJZri   : P2InstIDS<0b101110010, 0b1, always, (outs), (ins P2GPR:$d, jmptarget:$s), "tjz $d, #$s", []>;

def TJNZrr   : P2InstIDS<0b101110011, 0b0, always, (outs), (ins P2GPR:$d, P2GPR:$s), "tjnz $d, $s", []>;
def TJNZri   : P2InstIDS<0b101110011, 0b1, always, (outs), (ins P2GPR:$d, jmptarget:$s), "tjnz $d, #$s", []>;

// Patterns for various branch conditions and types of branch instructions
// unsigned comparison
def : Pat<(brcc SETUEQ, P2GPR:$lhs, P2GPR:$rhs, bb:$target), (JMPeq bb:$target, (CMPrr P2GPR:$lhs, P2GPR:$rhs))>;
def : Pat<(brcc SETUEQ, P2GPR:$lhs, imm9:$rhs, bb:$target), (JMPeq bb:$target, (CMPri P2GPR:$lhs, imm9:$rhs))>;

def : Pat<(brcc SETUNE, P2GPR:$lhs, P2GPR:$rhs, bb:$target), (JMPne bb:$target, (CMPrr P2GPR:$lhs, P2GPR:$rhs))>;
def : Pat<(brcc SETUNE, P2GPR:$lhs, imm9:$rhs, bb:$target), (JMPne bb:$target, (CMPri P2GPR:$lhs, imm9:$rhs))>;

def : Pat<(brcc SETULT, P2GPR:$lhs, P2GPR:$rhs, bb:$target), (JMPlt bb:$target, (CMPrr P2GPR:$lhs, P2GPR:$rhs))>;
def : Pat<(brcc SETULT, P2GPR:$lhs, imm9:$rhs, bb:$target), (JMPlt bb:$target, (CMPri P2GPR:$lhs, imm9:$rhs))>;

def : Pat<(brcc SETUGT, P2GPR:$lhs, P2GPR:$rhs, bb:$target), (JMPgt bb:$target, (CMPrr P2GPR:$lhs, P2GPR:$rhs))>;
def : Pat<(brcc SETUGT, P2GPR:$lhs, imm9:$rhs, bb:$target), (JMPgt bb:$target, (CMPri P2GPR:$lhs, imm9:$rhs))>;

def : Pat<(brcc SETULE, P2GPR:$lhs, P2GPR:$rhs, bb:$target), (JMPlte bb:$target, (CMPrr P2GPR:$lhs, P2GPR:$rhs))>;
def : Pat<(brcc SETULE, P2GPR:$lhs, imm9:$rhs, bb:$target), (JMPlte bb:$target, (CMPri P2GPR:$lhs, imm9:$rhs))>;

def : Pat<(brcc SETUGE, P2GPR:$lhs, P2GPR:$rhs, bb:$target), (JMPgte bb:$target, (CMPrr P2GPR:$lhs, P2GPR:$rhs))>;
def : Pat<(brcc SETUGE, P2GPR:$lhs, imm9:$rhs, bb:$target), (JMPgte bb:$target, (CMPri P2GPR:$lhs, imm9:$rhs))>;

// signed comparison
def : Pat<(brcc SETEQ, P2GPR:$lhs, P2GPR:$rhs, bb:$target), (JMPeq bb:$target, (CMPSrr P2GPR:$lhs, P2GPR:$rhs))>;
def : Pat<(brcc SETEQ, P2GPR:$lhs, imm9:$rhs, bb:$target), (JMPeq bb:$target, (CMPSri P2GPR:$lhs, imm9:$rhs))>;

def : Pat<(brcc SETNE, P2GPR:$lhs, P2GPR:$rhs, bb:$target), (JMPne bb:$target, (CMPSrr P2GPR:$lhs, P2GPR:$rhs))>;
def : Pat<(brcc SETNE, P2GPR:$lhs, imm9:$rhs, bb:$target), (JMPne bb:$target, (CMPSri P2GPR:$lhs, imm9:$rhs))>;

def : Pat<(brcc SETLT, P2GPR:$lhs, P2GPR:$rhs, bb:$target), (JMPlt bb:$target, (CMPSrr P2GPR:$lhs, P2GPR:$rhs))>;
def : Pat<(brcc SETLT, P2GPR:$lhs, imm9:$rhs, bb:$target), (JMPlt bb:$target, (CMPSri P2GPR:$lhs, imm9:$rhs))>;

def : Pat<(brcc SETGE, P2GPR:$lhs, P2GPR:$rhs, bb:$target), (JMPgte bb:$target, (CMPSrr P2GPR:$lhs, P2GPR:$rhs))>;
def : Pat<(brcc SETGE, P2GPR:$lhs, imm9:$rhs, bb:$target), (JMPgte bb:$target, (CMPSri P2GPR:$lhs, imm9:$rhs))>;

def : Pat<(brcc SETGT, P2GPR:$lhs, P2GPR:$rhs, bb:$target), (JMPgt bb:$target, (CMPSrr P2GPR:$lhs, P2GPR:$rhs))>;
def : Pat<(brcc SETGT, P2GPR:$lhs, imm9:$rhs, bb:$target), (JMPgt bb:$target, (CMPSri P2GPR:$lhs, imm9:$rhs))>;

def : Pat<(brcc SETLE, P2GPR:$lhs, P2GPR:$rhs, bb:$target), (JMPlte bb:$target, (CMPSrr P2GPR:$lhs, P2GPR:$rhs))>;
def : Pat<(brcc SETLE, P2GPR:$lhs, imm9:$rhs, bb:$target), (JMPlte bb:$target, (CMPSri P2GPR:$lhs, imm9:$rhs))>;

// if (c) branch => jump if not 0
def : Pat<(brcond P2GPR:$c, bb:$target), (TJNZri P2GPR:$c, bb:$target)>;

/*--------------------------------*/
/* moving data                    */
/*--------------------------------*/

// so we need to define one of these instructions for every condition we want to support, for every iteration of possible operand types
// which comes out to 8*number of conditions. expansion of this pseudo will actually dictate which compare command to use. ugh
// don't need a version of this where lhs is an immediate, since that compare instruction doesn't exist
multiclass sel_cc <CondCode c> {
    def iii     : Pseudo<(outs P2GPR:$d), (ins P2GPR:$lhs, s9imm:$rhs, s9imm:$t, s9imm:$f), "selectcc $d, $lhs, #$rhs, #$t, #$f",
                        [(set P2GPR:$d, (selectcc P2GPR:$lhs, imm9:$rhs, imm9:$t, imm9:$f, c))]>;

    def iir     : Pseudo<(outs P2GPR:$d), (ins P2GPR:$lhs, s9imm:$rhs, s9imm:$t, P2GPR:$f), "selectcc $d, $lhs, #$rhs, #$t, $f",
                        [(set P2GPR:$d, (selectcc P2GPR:$lhs, imm9:$rhs, imm9:$t, P2GPR:$f, c))]>;

    def iri     : Pseudo<(outs P2GPR:$d), (ins P2GPR:$lhs, s9imm:$rhs, P2GPR:$t, s9imm:$f), "selectcc $d, $lhs, #$rhs, $t, #$f",
                        [(set P2GPR:$d, (selectcc P2GPR:$lhs, imm9:$rhs, P2GPR:$t, imm9:$f, c))]>;

    def irr     : Pseudo<(outs P2GPR:$d), (ins P2GPR:$lhs, s9imm:$rhs, P2GPR:$t, P2GPR:$f), "selectcc $d, $lhs, #$rhs, $t, $f",
                        [(set P2GPR:$d, (selectcc P2GPR:$lhs, imm9:$rhs, P2GPR:$t, P2GPR:$f, c))]>;

    def rii     : Pseudo<(outs P2GPR:$d), (ins P2GPR:$lhs, P2GPR:$rhs, s9imm:$t, s9imm:$f), "selectcc $d, $lhs, $rhs, #$t, #$f",
                        [(set P2GPR:$d, (selectcc P2GPR:$lhs, P2GPR:$rhs, imm9:$t, imm9:$f, c))]>;

    def rir     : Pseudo<(outs P2GPR:$d), (ins P2GPR:$lhs, P2GPR:$rhs, s9imm:$t, P2GPR:$f), "selectcc $d, $lhs, $rhs, #$t, $f",
                        [(set P2GPR:$d, (selectcc P2GPR:$lhs, P2GPR:$rhs, imm9:$t, P2GPR:$f, c))]>;

    def rri     : Pseudo<(outs P2GPR:$d), (ins P2GPR:$lhs, P2GPR:$rhs, P2GPR:$t, s9imm:$f), "selectcc $d, $lhs, $rhs, $t, #$f",
                        [(set P2GPR:$d, (selectcc P2GPR:$lhs, P2GPR:$rhs, P2GPR:$t, imm9:$f, c))]>;

    def rrr     : Pseudo<(outs P2GPR:$d), (ins P2GPR:$lhs, P2GPR:$rhs, P2GPR:$t, P2GPR:$f), "selectcc $d, $lhs, $rhs, $t, $f",
                        [(set P2GPR:$d, (selectcc P2GPR:$lhs, P2GPR:$rhs, P2GPR:$t, P2GPR:$f, c))]>;
}

defm SELECTueq   : sel_cc<SETUEQ>;
defm SELECTune   : sel_cc<SETUNE>;
defm SELECTule   : sel_cc<SETULE>;
defm SELECTult   : sel_cc<SETULT>;
defm SELECTugt   : sel_cc<SETUGT>;
defm SELECTuge   : sel_cc<SETUGE>;
defm SELECTeq   : sel_cc<SETEQ>;
defm SELECTne   : sel_cc<SETNE>;
defm SELECTle   : sel_cc<SETLE>;
defm SELECTlt   : sel_cc<SETLT>;
defm SELECTgt   : sel_cc<SETGT>;
defm SELECTge   : sel_cc<SETGE>;

def : Pat<(select P2GPR:$c, P2GPR:$t, P2GPR:$f), (SELECTuneirr P2GPR:$c, 0, P2GPR:$t, P2GPR:$f)>;
def : Pat<(select P2GPR:$c, P2GPR:$t, imm9:$f), (SELECTuneiri P2GPR:$c, 0, P2GPR:$t, imm9:$f)>;
def : Pat<(select P2GPR:$c, imm9:$t, P2GPR:$f), (SELECTuneiir P2GPR:$c, 0, imm9:$t, P2GPR:$f)>;
def : Pat<(select P2GPR:$c, imm9:$t, imm9:$f), (SELECTuneiii P2GPR:$c, 0, imm9:$t, imm9:$f)>;

// for immediate moves, use a psuedo instruction that will insert an AUGS. Later we will write an optimizer pass
// that will replace the pattern of augs #imm32, mov <r> #imm9, <any instruction operating on r> to remove the
// mov <r> call, change augs to s or d as needed, and change the final instruction to use the #imm9 instead.
// a second pass should remove any augs/d #0 calls that are not used in relocation calls

def MOVri32     : Pseudo<(outs P2GPR:$d), (ins i32imm:$s), "mov $d, ##s", [(set P2GPR:$d, imm:$s)]>;
def MOVri       : P2InstCZIDS<0b0110000, 0b001, always, (outs P2GPR:$d), (ins s9imm:$s), "mov $d, #$s", [(set P2GPR:$d, imm9:$s)]>;
def MOVrr       : P2InstCZIDS<0b0110000, 0b000, always, (outs P2GPR:$d), (ins P2GPR:$s), "mov $d, $s", [(set P2GPR:$d, P2GPR:$s)]>;

def MOVeqri     : P2InstCZIDS<0b0110000, 0b001, if_z, (outs P2GPR:$d), (ins s9imm:$s), "mov $d, #$s", []>;
def MOVeqrr     : P2InstCZIDS<0b0110000, 0b000, if_z, (outs P2GPR:$d), (ins P2GPR:$s), "mov $d, $s", []>;

def MOVneri     : P2InstCZIDS<0b0110000, 0b001, if_nz, (outs P2GPR:$d), (ins s9imm:$s), "mov $d, #$s", []>;
def MOVnerr     : P2InstCZIDS<0b0110000, 0b000, if_nz, (outs P2GPR:$d), (ins P2GPR:$s), "mov $d, $s", []>;

def MOVltri     : P2InstCZIDS<0b0110000, 0b001, if_c, (outs P2GPR:$d), (ins s9imm:$s), "mov $d, #$s", []>;
def MOVltrr     : P2InstCZIDS<0b0110000, 0b000, if_c, (outs P2GPR:$d), (ins P2GPR:$s), "mov $d, $s", []>;

def MOVgteri     : P2InstCZIDS<0b0110000, 0b001, if_nc, (outs P2GPR:$d), (ins s9imm:$s), "mov $d, #$s", []>;
def MOVgterr     : P2InstCZIDS<0b0110000, 0b000, if_nc, (outs P2GPR:$d), (ins P2GPR:$s), "mov $d, $s", []>;

def MOVgtri     : P2InstCZIDS<0b0110000, 0b001, if_nc_and_nz, (outs P2GPR:$d), (ins s9imm:$s), "mov $d, #$s", []>;
def MOVgtrr     : P2InstCZIDS<0b0110000, 0b000, if_nc_and_nz, (outs P2GPR:$d), (ins P2GPR:$s), "mov $d, $s", []>;

def MOVlteri     : P2InstCZIDS<0b0110000, 0b001, if_c_or_z, (outs P2GPR:$d), (ins s9imm:$s), "mov $d, #$s", []>;
def MOVlterr     : P2InstCZIDS<0b0110000, 0b000, if_c_or_z, (outs P2GPR:$d), (ins P2GPR:$s), "mov $d, $s", []>;

def RDBYTEri    : P2InstCZIDS<0b1010110, 0b001, always, (outs P2GPR:$d), (ins s9imm:$s), "rdbyte $d, #$s", [(set P2GPR:$d, (zextloadi8 imm9:$s))]>;
def RDBYTErr    : P2InstCZIDS<0b1010110, 0b000, always, (outs P2GPR:$d), (ins P2GPR:$s), "rdbyte $d, $s", [(set P2GPR:$d, (zextloadi8 addr:$s))]>;

def RDWORDri    : P2InstCZIDS<0b1010111, 0b001, always, (outs P2GPR:$d), (ins s9imm:$s), "rdword $d, #$s", [(set P2GPR:$d, (zextloadi16 imm9:$s))]>;
def RDWORDrr    : P2InstCZIDS<0b1010111, 0b000, always, (outs P2GPR:$d), (ins P2GPR:$s), "rdword $d, $s", [(set P2GPR:$d, (zextloadi16 addr:$s))]>;

def RDLONGri    : P2InstCZIDS<0b1011000, 0b001, always, (outs P2GPR:$d), (ins s9imm:$s), "rdlong $d, #$s", [(set P2GPR:$d, (load imm9:$s))]>;
def RDLONGrr    : P2InstCZIDS<0b1011000, 0b000, always, (outs P2GPR:$d), (ins P2GPR:$s), "rdlong $d, $s", [(set P2GPR:$d, (load addr:$s))]>;

def WRBYTErr    : P2InstLIDS<0b11000100, 0b00, always, (outs), (ins P2GPR:$d, P2GPR:$s), "wrbyte $d, $s", [(truncstorei8 P2GPR:$d, addr:$s)]>;
def WRBYTEir    : P2InstLIDS<0b11000100, 0b10, always, (outs), (ins s9imm:$d, P2GPR:$s), "wrbyte #$d, $s", [(truncstorei8 imm9:$d, addr:$s)]>;

def WRWORDrr    : P2InstLIDS<0b11000101, 0b00, always, (outs), (ins P2GPR:$d, P2GPR:$s), "wrword $d, $s", [(truncstorei16 P2GPR:$d, addr:$s)]>;
def WRWORDir    : P2InstLIDS<0b11000101, 0b10, always, (outs), (ins s9imm:$d, P2GPR:$s), "wrword #$d, $s", [(truncstorei16 imm9:$d, addr:$s)]>;

def WRLONGri    : P2InstLIDS<0b11000110, 0b01, always, (outs), (ins P2GPR:$d, s9imm:$s), "wrlong $d, #$s", [(store P2GPR:$d, imm9:$s)]>;
def WRLONGii    : P2InstLIDS<0b11000110, 0b11, always, (outs), (ins s9imm:$d, s9imm:$s), "wrlong #$d, #$s", [(store imm9:$d, imm9:$s)]>;

def WRLONGrr    : P2InstLIDS<0b11000110, 0b00, always, (outs), (ins P2GPR:$d, P2GPR:$s), "wrlong $d, $s", [(store P2GPR:$d, addr:$s)]>;
def WRLONGir    : P2InstLIDS<0b11000110, 0b10, always, (outs), (ins s9imm:$d, P2GPR:$s), "wrlong #$d, $s", [(store imm9:$d, addr:$s)]>;

def SETQr       : P2InstLD<0b1101011, 0b000101000, 0b0, always, (outs), (ins P2GPR:$d), "setq $d", []>;
def SETQi       : P2InstLD<0b1101011, 0b000101000, 0b1, always, (outs), (ins s9imm:$d), "setq #$d", []>;

def RDLUTrr     : P2InstCZIDS<0b1010101, 0b000, always, (outs P2GPR:$d), (ins P2GPR:$s), "rdlut %d, %s", []>;
def RDLUTri     : P2InstCZIDS<0b1010101, 0b001, always, (outs P2GPR:$d), (ins s9imm:$s), "rdlut %d, #%s", []>;

def WRLUTrr     : P2InstLIDS<0b11000011, 0b00, always, (outs), (ins P2GPR:$d, P2GPR:$s), "wrlut %d, %s", []>;
def WRLUTri     : P2InstLIDS<0b11000011, 0b01, always, (outs), (ins P2GPR:$d, s9imm:$s), "wrlut %d, #%s", []>;
def WRLUTir     : P2InstLIDS<0b11000011, 0b10, always, (outs), (ins s9imm:$d, P2GPR:$s), "wrlut #%d, %s", []>;
def WRLUTii     : P2InstLIDS<0b11000011, 0b11, always, (outs), (ins s9imm:$d, s9imm:$s), "wrlut #%d, #%s", []>;

def : Pat<(extloadi8 addr:$s), (RDBYTErr P2GPR:$s)>;
def : Pat<(extloadi16 addr:$s), (RDWORDrr P2GPR:$s)>;

def : Pat<(sextloadi8 addr:$s), (SIGNXri (RDBYTErr P2GPR:$s), 7)>;
def : Pat<(sextloadi16 addr:$s), (SIGNXri (RDWORDrr P2GPR:$s), 15)>;

/*--------------------------------*/
/* function calling and globals   */
/*--------------------------------*/

let isTerminator = 1, isReturn = 1, isBarrier = 1 in {
    def RET     : P2InstCZ<0b11010111, 0b000000000, 0b000101101, 0b00, always, (outs), (ins), "ret", []>;
    def RETA    : P2InstCZ<0b11010111, 0b000000000, 0b000101110, 0b00, always, (outs), (ins), "reta", [(P2RET)]>; // primarily use this for returns
}

// FIXME: add CALLA and use a PTRA for the stack pointer
let isCall = 1, Uses = [PTRA] in {
    def CALLa    : P2InstRA<0b1101101, 0b0, always, (outs), (ins calltarget:$a), "call #$a", []>;
    def CALLAa   : P2InstRA<0b1101110, 0b0, always, (outs), (ins calltarget:$a), "calla #$a", []>; // mainly use this for direct function calls
    def CALLr    : P2InstCZD<0b1101011, 0b000101101, 0b00, always, (outs), (ins P2GPR:$d), "call $d", []>;
    def CALLAr   : P2InstCZD<0b1101011, 0b000101110, 0b00, always, (outs), (ins P2GPR:$d), "calla $d", []>; // mainly use this for function calls with register input
}

let Defs = [PTRA], Uses = [PTRA] in {
    def ADJCALLSTACKUP      : Pseudo<(outs), (ins i32imm:$amt1, i32imm:$amt2), "ADJCALLSTACKUP", [(P2callseq_start timm:$amt1, timm:$amt2)]>;
    def ADJCALLSTACKDOWN    : Pseudo<(outs), (ins i32imm:$amt, i32imm:$amt2), "ADJCALLSTACKDOWN", [(P2callseq_end timm:$amt, timm:$amt2)]>;
}

def : Pat<(P2GAWrapper tglobaladdr:$addr), (MOVri32 tglobaladdr:$addr)>;
def : Pat<(P2GAWrapper tjumptable:$addr), (MOVri32 tjumptable:$addr)>;
def : Pat<(P2CALL (i32 tglobaladdr:$dst)), (CALLAa tglobaladdr:$dst)>;
def : Pat<(P2CALL (i32 texternalsym:$dst)), (CALLAa texternalsym:$dst)>;
def : Pat<(P2CALL (i32 P2GPR:$d)), (CALLAr P2GPR:$d)>;

/*--------------------------------*/
/* I/O and hardware control       */
/*--------------------------------*/

let DecoderMethod = "DecodeIOInstruction" in {
    def DIRLr       : P2InstCZLD<0b1101011, 0b001000000, 0b000, always, (outs P2GPR:$dir), (ins P2GPR:$d), "dirl $d", []>;
    def DIRLi       : P2InstCZLD<0b1101011, 0b001000000, 0b001, always, (outs P2GPR:$dir), (ins i32imm:$d), "dirl #$d", []>;

    def DIRHr       : P2InstCZLD<0b1101011, 0b001000001, 0b000, always, (outs P2GPR:$dir), (ins P2GPR:$d), "dirh $d", []>;
    def DIRHi       : P2InstCZLD<0b1101011, 0b001000001, 0b001, always, (outs P2GPR:$dir), (ins i32imm:$d), "dirh #$d", []>;

    def OUTLr       : P2InstCZLD<0b1101011, 0b001001000, 0b000, always, (outs P2GPR:$dir), (ins P2GPR:$d), "outl $d", []>;
    def OUTLi       : P2InstCZLD<0b1101011, 0b001001000, 0b001, always, (outs P2GPR:$dir), (ins i32imm:$d), "outl #$d", []>;

    def OUTHr       : P2InstCZLD<0b1101011, 0b001001001, 0b000, always, (outs P2GPR:$dir), (ins P2GPR:$d), "outh $d", []>;
    def OUTHi       : P2InstCZLD<0b1101011, 0b001001001, 0b001, always, (outs P2GPR:$dir), (ins i32imm:$d), "outh #$d", []>;

    def DRVLr       : P2InstCZLD<0b1101011, 0b001011000, 0b000, always, (outs P2GPR:$dir), (ins P2GPR:$d), "drvl $d", []>;
    def DRVLi       : P2InstCZLD<0b1101011, 0b001011000, 0b001, always, (outs P2GPR:$dir), (ins i32imm:$d), "drvl #$d", []>;

    def DRVHr       : P2InstCZLD<0b1101011, 0b001011001, 0b000, always, (outs P2GPR:$dir), (ins P2GPR:$d), "drvh $d", []>;
    def DRVHi       : P2InstCZLD<0b1101011, 0b001011001, 0b001, always, (outs P2GPR:$dir), (ins i32imm:$d), "drvh #$d", []>;
}

def COGIDr      : P2InstCLD<0b1101011, 0b000000001, 0b00, always, (outs P2GPR:$d), (ins), "cogid $d", []>;

def WAITXr      : P2InstLD<0b1101011, 0b000011111, 0b0, always, (outs), (ins P2GPR:$d), "waitx $d", []>;
def WAITXi      : P2InstLD<0b1101011, 0b000011111, 0b1, always, (outs), (ins i32imm:$d), "waitx #$d", []>;

def GETCTlor    : P2InstCZD<0b1101011, 0b000011010, 0b00, always, (outs P2GPR:$d), (ins), "getct $d", []>;
def GETCThir    : P2InstCZD<0b1101011, 0b000011010, 0b10, always, (outs P2GPR:$d), (ins), "getct $d wc", []>;

def COGINITrr   : P2InstCLIDS<0b1100111, 0b000, always, (outs), (ins P2GPR:$d, P2GPR:$s), "coginit $d, $s", []>;
def COGINITir   : P2InstCLIDS<0b1100111, 0b010, always, (outs), (ins s9imm:$d, P2GPR:$s), "coginit #$d, $s", []>;
def COGINITri   : P2InstCLIDS<0b1100111, 0b001, always, (outs), (ins P2GPR:$d, s9imm:$s), "coginit $d, #$s", []>;
def COGINITii   : P2InstCLIDS<0b1100111, 0b011, always, (outs), (ins s9imm:$d, s9imm:$s), "coginit #$d, #$s", []>;

def HUBSETrr    : P2InstLD<0b1101011, 0b000000000, 0b0, always, (outs), (ins P2GPR:$d), "hubset $d", []>;
def HUBSETri    : P2InstLD<0b1101011, 0b000000000, 0b1, always, (outs), (ins s9imm:$d), "hubset #$d", []>;

def WRPINrr     : P2InstLIDS<0b11000000, 0b00, always, (outs), (ins P2GPR:$d, P2GPR:$s), "wrpin $d, $s", []>;
def WRPINii     : P2InstLIDS<0b11000000, 0b11, always, (outs), (ins s9imm:$d, s9imm:$s), "wrpin #$d, #$s", []>;
def WRPINri     : P2InstLIDS<0b11000000, 0b01, always, (outs), (ins P2GPR:$d, s9imm:$s), "wrpin $d, #$s", []>;
def WRPINir     : P2InstLIDS<0b11000000, 0b10, always, (outs), (ins s9imm:$d, P2GPR:$s), "wrpin #$d, $s", []>;

def WXPINrr     : P2InstLIDS<0b11000001, 0b00, always, (outs), (ins P2GPR:$d, P2GPR:$s), "wxpin $d, $s", []>;
def WXPINii     : P2InstLIDS<0b11000001, 0b11, always, (outs), (ins s9imm:$d, s9imm:$s), "wxpin #$d, #$s", []>;
def WXPINri     : P2InstLIDS<0b11000001, 0b01, always, (outs), (ins P2GPR:$d, s9imm:$s), "wxpin $d, #$s", []>;
def WXPINir     : P2InstLIDS<0b11000001, 0b10, always, (outs), (ins s9imm:$d, P2GPR:$s), "wxpin #$d, $s", []>;

def WYPINrr     : P2InstLIDS<0b11000010, 0b00, always, (outs), (ins P2GPR:$d, P2GPR:$s), "wypin $d, $s", []>;
def WYPINii     : P2InstLIDS<0b11000010, 0b11, always, (outs), (ins s9imm:$d, s9imm:$s), "wypin #$d, #$s", []>;
def WYPINri     : P2InstLIDS<0b11000010, 0b01, always, (outs), (ins P2GPR:$d, s9imm:$s), "wypin $d, #$s", []>;
def WYPINir     : P2InstLIDS<0b11000010, 0b10, always, (outs), (ins s9imm:$d, P2GPR:$s), "wypin #$d, $s", []>;

/*--------------------------------*/
/* Interrupts and events          */
/*--------------------------------*/

def ADDCT1rr    : P2InstIDS<0b101001100, 0b0, always, (outs), (ins P2GPR:$d, P2GPR:$s), "addct1 $d, $s", []>;
def ADDCT1ri    : P2InstIDS<0b101001100, 0b1, always, (outs), (ins P2GPR:$d, s9imm:$s), "addct1 $d, #$s", []>;

def ADDCT2rr    : P2InstIDS<0b101001101, 0b0, always, (outs), (ins P2GPR:$d, P2GPR:$s), "addct2 $d, $s", []>;
def ADDCT2ri    : P2InstIDS<0b101001101, 0b1, always, (outs), (ins P2GPR:$d, s9imm:$s), "addct2 $d, #$s", []>;

def ADDCT3rr    : P2InstIDS<0b101001110, 0b0, always, (outs), (ins P2GPR:$d, P2GPR:$s), "addct3 $d, $s", []>;
def ADDCT3ri    : P2InstIDS<0b101001110, 0b1, always, (outs), (ins P2GPR:$d, s9imm:$s), "addct3 $d, #$s", []>;

def WAITCT1     : P2InstCZ<0b11010110, 0b000010001, 0b000100100, 0b00, always, (outs), (ins), "waitct1", []>;
def WAITCT2     : P2InstCZ<0b11010110, 0b000010010, 0b000100100, 0b00, always, (outs), (ins), "waitct2", []>;
def WAITCT3     : P2InstCZ<0b11010110, 0b000010011, 0b000100100, 0b00, always, (outs), (ins), "waitct3", []>;

/*--------------------------------*/
/* misc instructions              */
/*--------------------------------*/
def NOP         : P2InstNOP;
def AUGS        : P2InstN<0b11110, always, (outs), (ins i32imm:$n), "augs #$n", []>;
def AUGD        : P2InstN<0b11111, always, (outs), (ins i32imm:$n), "augd #$n", []>;

def FRMIDX      : Pseudo<(outs P2GPR:$dst), (ins P2GPR:$src), "frmidx $dst, $src", []>;

/*--------------------------------*/
/* Optimization patterns and such */
/*--------------------------------*/