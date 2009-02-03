//===- XCoreInstrInfo.td - Target Description for XCore ----*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the XCore instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

// Uses of CP, DP are not currently reflected in the patterns, since
// having a physical register as an operand prevents loop hoisting and
// since the value of these registers never changes during the life of the
// function.

//===----------------------------------------------------------------------===//
// Instruction format superclass.
//===----------------------------------------------------------------------===//

include "XCoreInstrFormats.td"

//===----------------------------------------------------------------------===//
// Feature predicates.
//===----------------------------------------------------------------------===//

// HasXS1A - This predicate is true when the target processor supports XS1A
// instructions.
def HasXS1A   : Predicate<"Subtarget.isXS1A()">;

// HasXS1B - This predicate is true when the target processor supports XS1B
// instructions.
def HasXS1B : Predicate<"Subtarget.isXS1B()">;

//===----------------------------------------------------------------------===//
// XCore specific DAG Nodes.
//

// Call
def SDT_XCoreBranchLink : SDTypeProfile<0, 1, [SDTCisPtrTy<0>]>;
def XCoreBranchLink     : SDNode<"XCoreISD::BL",SDT_XCoreBranchLink,
                            [SDNPHasChain, SDNPOptInFlag, SDNPOutFlag]>;

def XCoreRetsp       : SDNode<"XCoreISD::RETSP", SDTNone,
                         [SDNPHasChain, SDNPOptInFlag]>;

def SDT_XCoreAddress    : SDTypeProfile<1, 1,
                            [SDTCisSameAs<0, 1>, SDTCisPtrTy<0>]>;

def pcrelwrapper : SDNode<"XCoreISD::PCRelativeWrapper", SDT_XCoreAddress,
	                   []>;

def dprelwrapper : SDNode<"XCoreISD::DPRelativeWrapper", SDT_XCoreAddress,
	                   []>;

def cprelwrapper : SDNode<"XCoreISD::CPRelativeWrapper", SDT_XCoreAddress,
	                   []>;

def SDT_XCoreStwsp    : SDTypeProfile<0, 2, [SDTCisInt<1>]>;
def XCoreStwsp         : SDNode<"XCoreISD::STWSP", SDT_XCoreStwsp,
	                   [SDNPHasChain]>;

// These are target-independent nodes, but have target-specific formats.
def SDT_XCoreCallSeqStart : SDCallSeqStart<[ SDTCisVT<0, i32> ]>;
def SDT_XCoreCallSeqEnd   : SDCallSeqEnd<[ SDTCisVT<0, i32>,
                                        SDTCisVT<1, i32> ]>;

def callseq_start : SDNode<"ISD::CALLSEQ_START", SDT_XCoreCallSeqStart,
                           [SDNPHasChain, SDNPOutFlag]>;
def callseq_end   : SDNode<"ISD::CALLSEQ_END",   SDT_XCoreCallSeqEnd,
                           [SDNPHasChain, SDNPOptInFlag, SDNPOutFlag]>;

//===----------------------------------------------------------------------===//
// Instruction Pattern Stuff
//===----------------------------------------------------------------------===//

def div4_xform : SDNodeXForm<imm, [{
  // Transformation function: imm/4
  assert(N->getZExtValue() % 4 == 0);
  return getI32Imm(N->getZExtValue()/4);
}]>;

def msksize_xform : SDNodeXForm<imm, [{
  // Transformation function: get the size of a mask
  assert(isMask_32(N->getZExtValue()));
  // look for the first non-zero bit
  return getI32Imm(32 - CountLeadingZeros_32(N->getZExtValue()));
}]>;

def neg_xform : SDNodeXForm<imm, [{
  // Transformation function: -imm
  uint32_t value = N->getZExtValue();
  return getI32Imm(-value);
}]>;

def div4neg_xform : SDNodeXForm<imm, [{
  // Transformation function: -imm/4
  uint32_t value = N->getZExtValue();
  assert(-value % 4 == 0);
  return getI32Imm(-value/4);
}]>;

def immUs4Neg : PatLeaf<(imm), [{
  uint32_t value = (uint32_t)N->getZExtValue();
  return (-value)%4 == 0 && (-value)/4 <= 11;
}]>;

def immUs4 : PatLeaf<(imm), [{
  uint32_t value = (uint32_t)N->getZExtValue();
  return value%4 == 0 && value/4 <= 11;
}]>;

def immUsNeg : PatLeaf<(imm), [{
  return -((uint32_t)N->getZExtValue()) <= 11;
}]>;

def immUs : PatLeaf<(imm), [{
  return (uint32_t)N->getZExtValue() <= 11;
}]>;

def immU6 : PatLeaf<(imm), [{
  return (uint32_t)N->getZExtValue() < (1 << 6);
}]>;

def immU10 : PatLeaf<(imm), [{
  return (uint32_t)N->getZExtValue() < (1 << 10);
}]>;

def immU16 : PatLeaf<(imm), [{
  return (uint32_t)N->getZExtValue() < (1 << 16);
}]>;

def immU20 : PatLeaf<(imm), [{
  return (uint32_t)N->getZExtValue() < (1 << 20);
}]>;

// FIXME check subtarget. Currently we check if the immediate
// is in the common subset of legal immediate values for both
// XS1A and XS1B.
def immMskBitp : PatLeaf<(imm), [{
  uint32_t value = (uint32_t)N->getZExtValue();
  if (!isMask_32(value)) {
    return false;
  }
  int msksize = 32 - CountLeadingZeros_32(value);
  return (msksize >= 1 && msksize <= 8)
          || msksize == 16
          || msksize == 24
          || msksize == 32;
}]>;

// FIXME check subtarget. Currently we check if the immediate
// is in the common subset of legal immediate values for both
// XS1A and XS1B.
def immBitp : PatLeaf<(imm), [{
  uint32_t value = (uint32_t)N->getZExtValue();
  return (value >= 1 && value <= 8)
          || value == 16
          || value == 24
          || value == 32;
}]>;

def lda16f : PatFrag<(ops node:$addr, node:$offset),
                     (add node:$addr, (shl node:$offset, 1))>;
def lda16b : PatFrag<(ops node:$addr, node:$offset),
                     (sub node:$addr, (shl node:$offset, 1))>;
def ldawf : PatFrag<(ops node:$addr, node:$offset),
                     (add node:$addr, (shl node:$offset, 2))>;
def ldawb : PatFrag<(ops node:$addr, node:$offset),
                     (sub node:$addr, (shl node:$offset, 2))>;

// Instruction operand types
def calltarget  : Operand<i32>;
def brtarget : Operand<OtherVT>;
def pclabel : Operand<i32>;

// Addressing modes
def ADDRspii : ComplexPattern<i32, 2, "SelectADDRspii", [add, frameindex], []>;
def ADDRdpii : ComplexPattern<i32, 2, "SelectADDRdpii", [add, dprelwrapper],
                 []>;
def ADDRcpii : ComplexPattern<i32, 2, "SelectADDRcpii", [add, cprelwrapper],
                 []>;

// Address operands
def MEMii : Operand<i32> {
  let PrintMethod = "printMemOperand";
  let MIOperandInfo = (ops i32imm, i32imm);
}

//===----------------------------------------------------------------------===//
// Instruction Class Templates
//===----------------------------------------------------------------------===//

// Three operand short

multiclass F3R_2RUS<string OpcStr, SDNode OpNode> {
  def _3r: _F3R<
                 (outs GRRegs:$dst), (ins GRRegs:$b, GRRegs:$c),
                 !strconcat(OpcStr, " $dst, $b, $c"),
                 [(set GRRegs:$dst, (OpNode GRRegs:$b, GRRegs:$c))]>;
  def _2rus : _F2RUS<
                 (outs GRRegs:$dst), (ins GRRegs:$b, i32imm:$c),
                 !strconcat(OpcStr, " $dst, $b, $c"),
                 [(set GRRegs:$dst, (OpNode GRRegs:$b, immUs:$c))]>;
}

multiclass F3R_2RUS_np<string OpcStr> {
  def _3r: _F3R<
                 (outs GRRegs:$dst), (ins GRRegs:$b, GRRegs:$c),
                 !strconcat(OpcStr, " $dst, $b, $c"),
                 []>;
  def _2rus : _F2RUS<
                 (outs GRRegs:$dst), (ins GRRegs:$b, i32imm:$c),
                 !strconcat(OpcStr, " $dst, $b, $c"),
                 []>;
}

multiclass F3R_2RBITP<string OpcStr, SDNode OpNode> {
  def _3r: _F3R<
                 (outs GRRegs:$dst), (ins GRRegs:$b, GRRegs:$c),
                 !strconcat(OpcStr, " $dst, $b, $c"),
                 [(set GRRegs:$dst, (OpNode GRRegs:$b, GRRegs:$c))]>;
  def _2rus : _F2RUS<
                 (outs GRRegs:$dst), (ins GRRegs:$b, i32imm:$c),
                 !strconcat(OpcStr, " $dst, $b, $c"),
                 [(set GRRegs:$dst, (OpNode GRRegs:$b, immBitp:$c))]>;
}

class F3R<string OpcStr, SDNode OpNode> : _F3R<
                 (outs GRRegs:$dst), (ins GRRegs:$b, GRRegs:$c),
                 !strconcat(OpcStr, " $dst, $b, $c"),
                 [(set GRRegs:$dst, (OpNode GRRegs:$b, GRRegs:$c))]>;

class F3R_np<string OpcStr> : _F3R<
                 (outs GRRegs:$dst), (ins GRRegs:$b, GRRegs:$c),
                 !strconcat(OpcStr, " $dst, $b, $c"),
                 []>;
// Three operand long

/// FL3R_L2RUS multiclass - Define a normal FL3R/FL2RUS pattern in one shot.
multiclass FL3R_L2RUS<string OpcStr, SDNode OpNode> {
  def _l3r: _FL3R<
                 (outs GRRegs:$dst), (ins GRRegs:$b, GRRegs:$c),
                 !strconcat(OpcStr, " $dst, $b, $c"),
                 [(set GRRegs:$dst, (OpNode GRRegs:$b, GRRegs:$c))]>;
  def _l2rus : _FL2RUS<
                 (outs GRRegs:$dst), (ins GRRegs:$b, i32imm:$c),
                 !strconcat(OpcStr, " $dst, $b, $c"),
                 [(set GRRegs:$dst, (OpNode GRRegs:$b, immUs:$c))]>;
}

/// FL3R_L2RUS multiclass - Define a normal FL3R/FL2RUS pattern in one shot.
multiclass FL3R_L2RBITP<string OpcStr, SDNode OpNode> {
  def _l3r: _FL3R<
                 (outs GRRegs:$dst), (ins GRRegs:$b, GRRegs:$c),
                 !strconcat(OpcStr, " $dst, $b, $c"),
                 [(set GRRegs:$dst, (OpNode GRRegs:$b, GRRegs:$c))]>;
  def _l2rus : _FL2RUS<
                 (outs GRRegs:$dst), (ins GRRegs:$b, i32imm:$c),
                 !strconcat(OpcStr, " $dst, $b, $c"),
                 [(set GRRegs:$dst, (OpNode GRRegs:$b, immBitp:$c))]>;
}

class FL3R<string OpcStr, SDNode OpNode> : _FL3R<
                 (outs GRRegs:$dst), (ins GRRegs:$b, GRRegs:$c),
                 !strconcat(OpcStr, " $dst, $b, $c"),
                 [(set GRRegs:$dst, (OpNode GRRegs:$b, GRRegs:$c))]>;

// Register - U6
// Operand register - U6
multiclass FRU6_LRU6_branch<string OpcStr> {
  def _ru6: _FRU6<
                 (outs), (ins GRRegs:$cond, brtarget:$dest),
                 !strconcat(OpcStr, " $cond, $dest"),
                 []>;
  def _lru6: _FLRU6<
                 (outs), (ins GRRegs:$cond, brtarget:$dest),
                 !strconcat(OpcStr, " $cond, $dest"),
                 []>;
}

multiclass FRU6_LRU6_cp<string OpcStr> {
  def _ru6: _FRU6<
                 (outs GRRegs:$dst), (ins i32imm:$a),
                 !strconcat(OpcStr, " $dst, cp[$a]"),
                 []>;
  def _lru6: _FLRU6<
                 (outs GRRegs:$dst), (ins i32imm:$a),
                 !strconcat(OpcStr, " $dst, cp[$a]"),
                 []>;
}

// U6
multiclass FU6_LU6<string OpcStr, SDNode OpNode> {
  def _u6: _FU6<
                 (outs), (ins i32imm:$b),
                 !strconcat(OpcStr, " $b"),
                 [(OpNode immU6:$b)]>;
  def _lu6: _FLU6<
                 (outs), (ins i32imm:$b),
                 !strconcat(OpcStr, " $b"),
                 [(OpNode immU16:$b)]>;
}

multiclass FU6_LU6_np<string OpcStr> {
  def _u6: _FU6<
                 (outs), (ins i32imm:$b),
                 !strconcat(OpcStr, " $b"),
                 []>;
  def _lu6: _FLU6<
                 (outs), (ins i32imm:$b),
                 !strconcat(OpcStr, " $b"),
                 []>;
}

// U10
multiclass FU10_LU10_np<string OpcStr> {
  def _u10: _FU10<
                 (outs), (ins i32imm:$b),
                 !strconcat(OpcStr, " $b"),
                 []>;
  def _lu10: _FLU10<
                 (outs), (ins i32imm:$b),
                 !strconcat(OpcStr, " $b"),
                 []>;
}

// Two operand short

class F2R_np<string OpcStr> : _F2R<
                 (outs GRRegs:$dst), (ins GRRegs:$b),
                 !strconcat(OpcStr, " $dst, $b"),
                 []>;

// Two operand long

//===----------------------------------------------------------------------===//
// Pseudo Instructions
//===----------------------------------------------------------------------===//

let Defs = [SP], Uses = [SP] in {
def ADJCALLSTACKDOWN : PseudoInstXCore<(outs), (ins i32imm:$amt),
                               "${:comment} ADJCALLSTACKDOWN $amt",
                               [(callseq_start timm:$amt)]>;
def ADJCALLSTACKUP : PseudoInstXCore<(outs), (ins i32imm:$amt1, i32imm:$amt2),
                            "${:comment} ADJCALLSTACKUP $amt1",
                            [(callseq_end timm:$amt1, timm:$amt2)]>;
}

def LDWFI : PseudoInstXCore<(outs GRRegs:$dst), (ins MEMii:$addr),
                             "${:comment} LDWFI $dst, $addr",
                             [(set GRRegs:$dst, (load ADDRspii:$addr))]>;

def LDAWFI : PseudoInstXCore<(outs GRRegs:$dst), (ins MEMii:$addr),
                             "${:comment} LDAWFI $dst, $addr",
                             [(set GRRegs:$dst, ADDRspii:$addr)]>;

def STWFI : PseudoInstXCore<(outs), (ins GRRegs:$src, MEMii:$addr),
                            "${:comment} STWFI $src, $addr",
                            [(store GRRegs:$src, ADDRspii:$addr)]>;

// SELECT_CC_* - Used to implement the SELECT_CC DAG operation.  Expanded by the
// scheduler into a branch sequence.
let usesCustomDAGSchedInserter = 1 in {
  def SELECT_CC : PseudoInstXCore<(outs GRRegs:$dst),
                              (ins GRRegs:$cond, GRRegs:$T, GRRegs:$F),
                              "${:comment} SELECT_CC PSEUDO!",
                              [(set GRRegs:$dst,
                                 (select GRRegs:$cond, GRRegs:$T, GRRegs:$F))]>;
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

// Three operand short
defm ADD : F3R_2RUS<"add", add>;
defm SUB : F3R_2RUS<"sub", sub>;
let neverHasSideEffects = 1 in {
defm EQ : F3R_2RUS_np<"eq">;
def LSS_3r : F3R_np<"lss">;
def LSU_3r : F3R_np<"lsu">;
}
def AND_3r : F3R<"and", and>;
def OR_3r : F3R<"or", or>;

let mayLoad=1 in {
def LDW_3r : _F3R<(outs GRRegs:$dst), (ins GRRegs:$addr, GRRegs:$offset),
                  "ldw $dst, $addr[$offset]",
                  []>;

def LDW_2rus : _F2RUS<(outs GRRegs:$dst), (ins GRRegs:$addr, i32imm:$offset),
                  "ldw $dst, $addr[$offset]",
                  []>;

def LD16S_3r :  _F3R<(outs GRRegs:$dst), (ins GRRegs:$addr, GRRegs:$offset),
                  "ld16s $dst, $addr[$offset]",
                  []>;

def LD8U_3r :  _F3R<(outs GRRegs:$dst), (ins GRRegs:$addr, GRRegs:$offset),
                  "ld8u $dst, $addr[$offset]",
                  []>;
}

let mayStore=1 in {
def STW_3r : _F3R<(outs), (ins GRRegs:$val, GRRegs:$addr, GRRegs:$offset),
                  "stw $val, $addr[$offset]",
                  []>;

def STW_2rus : _F2RUS<(outs), (ins GRRegs:$val, GRRegs:$addr, i32imm:$offset),
                  "stw $val, $addr[$offset]",
                  []>;
}

defm SHL : F3R_2RBITP<"shl", shl>;
defm SHR : F3R_2RBITP<"shr", srl>;
// TODO tsetr

// Three operand long
def LDAWF_l3r : _FL3R<(outs GRRegs:$dst), (ins GRRegs:$addr, GRRegs:$offset),
                  "ldaw $dst, $addr[$offset]",
                  [(set GRRegs:$dst, (ldawf GRRegs:$addr, GRRegs:$offset))]>;

let neverHasSideEffects = 1 in
def LDAWF_l2rus : _FL2RUS<(outs GRRegs:$dst),
                    (ins GRRegs:$addr, i32imm:$offset),
                    "ldaw $dst, $addr[$offset]",
                    []>;

def LDAWB_l3r : _FL3R<(outs GRRegs:$dst), (ins GRRegs:$addr, GRRegs:$offset),
                  "ldaw $dst, $addr[-$offset]",
                  [(set GRRegs:$dst, (ldawb GRRegs:$addr, GRRegs:$offset))]>;

let neverHasSideEffects = 1 in
def LDAWB_l2rus : _FL2RUS<(outs GRRegs:$dst),
                    (ins GRRegs:$addr, i32imm:$offset),
                    "ldaw $dst, $addr[-$offset]",
                    []>;

def LDA16F_l3r : _FL3R<(outs GRRegs:$dst), (ins GRRegs:$addr, GRRegs:$offset),
                  "lda16 $dst, $addr[$offset]",
                  [(set GRRegs:$dst, (lda16f GRRegs:$addr, GRRegs:$offset))]>;

def LDA16B_l3r : _FL3R<(outs GRRegs:$dst), (ins GRRegs:$addr, GRRegs:$offset),
                  "lda16 $dst, $addr[-$offset]",
                  [(set GRRegs:$dst, (lda16b GRRegs:$addr, GRRegs:$offset))]>;

def MUL_l3r : FL3R<"mul", mul>;
// Instructions which may trap are marked as side effecting.
let hasSideEffects = 1 in {
def DIVS_l3r : FL3R<"divs", sdiv>;
def DIVU_l3r : FL3R<"divu", udiv>;
def REMS_l3r : FL3R<"rems", srem>;
def REMU_l3r : FL3R<"remu", urem>;
}
def XOR_l3r : FL3R<"xor", xor>;
defm ASHR : FL3R_L2RBITP<"ashr", sra>;
// TODO crc32, crc8, inpw, outpw
let mayStore=1 in {
def ST16_l3r : _FL3R<(outs), (ins GRRegs:$val, GRRegs:$addr, GRRegs:$offset),
                "st16 $val, $addr[$offset]",
                []>;

def ST8_l3r : _FL3R<(outs), (ins GRRegs:$val, GRRegs:$addr, GRRegs:$offset),
                "st8 $val, $addr[$offset]",
                []>;
}

// Four operand long
let Predicates = [HasXS1B], Constraints = "$src1 = $dst1,$src2 = $dst2" in {
def MACCU_l4r : _L4R<(outs GRRegs:$dst1, GRRegs:$dst2),
                    (ins GRRegs:$src1, GRRegs:$src2, GRRegs:$src3,
                      GRRegs:$src4),
                    "maccu $dst1, $dst2, $src3, $src4",
                    []>;

def MACCS_l4r : _L4R<(outs GRRegs:$dst1, GRRegs:$dst2),
                    (ins GRRegs:$src1, GRRegs:$src2, GRRegs:$src3,
                      GRRegs:$src4),
                    "maccs $dst1, $dst2, $src3, $src4",
                    []>;
}

// Five operand long

let Predicates = [HasXS1B] in {
def LADD_l5r : _L5R<(outs GRRegs:$dst1, GRRegs:$dst2),
                    (ins GRRegs:$src1, GRRegs:$src2, GRRegs:$src3),
                    "ladd $dst1, $dst2, $src1, $src2, $src3",
                    []>;

def LSUB_l5r : _L5R<(outs GRRegs:$dst1, GRRegs:$dst2),
                    (ins GRRegs:$src1, GRRegs:$src2, GRRegs:$src3),
                    "lsub $dst1, $dst2, $src1, $src2, $src3",
                    []>;

def LDIV_l5r : _L5R<(outs GRRegs:$dst1, GRRegs:$dst2),
                    (ins GRRegs:$src1, GRRegs:$src2, GRRegs:$src3),
                    "ldiv $dst1, $dst2, $src1, $src2, $src3",
                    []>;
}

// Six operand long

def LMUL_l6r : _L6R<(outs GRRegs:$dst1, GRRegs:$dst2),
                    (ins GRRegs:$src1, GRRegs:$src2, GRRegs:$src3,
                      GRRegs:$src4),
                    "lmul $dst1, $dst2, $src1, $src2, $src3, $src4",
                    []>;

let Predicates = [HasXS1A] in
def MACC_l6r : _L6R<(outs GRRegs:$dst1, GRRegs:$dst2),
                    (ins GRRegs:$src1, GRRegs:$src2, GRRegs:$src3,
                      GRRegs:$src4),
                    "macc $dst1, $dst2, $src1, $src2, $src3, $src4",
                    []>;

// Register - U6

//let Uses = [DP] in ...
let neverHasSideEffects = 1, isReMaterializable = 1 in
def LDAWDP_ru6: _FRU6<(outs GRRegs:$dst), (ins MEMii:$a),
                    "ldaw $dst, dp[$a]",
                    []>;

let isReMaterializable = 1 in                    
def LDAWDP_lru6: _FLRU6<
                    (outs GRRegs:$dst), (ins MEMii:$a),
                    "ldaw $dst, dp[$a]",
                    [(set GRRegs:$dst, ADDRdpii:$a)]>;

let mayLoad=1 in
def LDWDP_ru6: _FRU6<(outs GRRegs:$dst), (ins MEMii:$a),
                    "ldw $dst, dp[$a]",
                    []>;
                    
def LDWDP_lru6: _FLRU6<
                    (outs GRRegs:$dst), (ins MEMii:$a),
                    "ldw $dst, dp[$a]",
                    [(set GRRegs:$dst, (load ADDRdpii:$a))]>;

let mayStore=1 in
def STWDP_ru6 : _FRU6<(outs), (ins GRRegs:$val, MEMii:$addr),
                  "stw $val, dp[$addr]",
                  []>;

def STWDP_lru6 : _FLRU6<(outs), (ins GRRegs:$val, MEMii:$addr),
                  "stw $val, dp[$addr]",
                  [(store GRRegs:$val, ADDRdpii:$addr)]>;

//let Uses = [CP] in ..
let mayLoad = 1, isReMaterializable = 1 in
defm LDWCP : FRU6_LRU6_cp<"ldw">;

let Uses = [SP] in {
let mayStore=1 in {
def STWSP_ru6 : _FRU6<
                 (outs), (ins GRRegs:$val, i32imm:$index),
                 "stw $val, sp[$index]",
                 [(XCoreStwsp GRRegs:$val, immU6:$index)]>;

def STWSP_lru6 : _FLRU6<
                 (outs), (ins GRRegs:$val, i32imm:$index),
                 "stw $val, sp[$index]",
                 [(XCoreStwsp GRRegs:$val, immU16:$index)]>;
}

let mayLoad=1 in {
def LDWSP_ru6 : _FRU6<
                 (outs GRRegs:$dst), (ins i32imm:$b),
                 "ldw $dst, sp[$b]",
                 []>;

def LDWSP_lru6 : _FLRU6<
                 (outs GRRegs:$dst), (ins i32imm:$b),
                 "ldw $dst, sp[$b]",
                 []>;
}

let neverHasSideEffects = 1 in {
def LDAWSP_ru6 : _FRU6<
                 (outs GRRegs:$dst), (ins i32imm:$b),
                 "ldaw $dst, sp[$b]",
                 []>;

def LDAWSP_lru6 : _FLRU6<
                 (outs GRRegs:$dst), (ins i32imm:$b),
                 "ldaw $dst, sp[$b]",
                 []>;

def LDAWSP_ru6_RRegs : _FRU6<
                 (outs RRegs:$dst), (ins i32imm:$b),
                 "ldaw $dst, sp[$b]",
                 []>;

def LDAWSP_lru6_RRegs : _FLRU6<
                 (outs RRegs:$dst), (ins i32imm:$b),
                 "ldaw $dst, sp[$b]",
                 []>;
}
}

let isReMaterializable = 1 in {
def LDC_ru6 : _FRU6<
                 (outs GRRegs:$dst), (ins i32imm:$b),
                 "ldc $dst, $b",
                 [(set GRRegs:$dst, immU6:$b)]>;

def LDC_lru6 : _FLRU6<
                 (outs GRRegs:$dst), (ins i32imm:$b),
                 "ldc $dst, $b",
                 [(set GRRegs:$dst, immU16:$b)]>;
}

// Operand register - U6
// TODO setc
let isBranch = 1, isTerminator = 1 in {
defm BRFT: FRU6_LRU6_branch<"bt">;
defm BRBT: FRU6_LRU6_branch<"bt">;
defm BRFF: FRU6_LRU6_branch<"bf">;
defm BRBF: FRU6_LRU6_branch<"bf">;
}

// U6
let Defs = [SP], Uses = [SP] in {
let neverHasSideEffects = 1 in
defm EXTSP : FU6_LU6_np<"extsp">;
let mayStore = 1 in
defm ENTSP : FU6_LU6_np<"entsp">;

let isReturn = 1, isTerminator = 1, mayLoad = 1 in {
defm RETSP : FU6_LU6<"retsp", XCoreRetsp>;
}
}

// TODO extdp, kentsp, krestsp, blat, setsr
// clrsr, getsr, kalli
let isBranch = 1, isTerminator = 1 in {
def BRBU_u6 : _FU6<
                 (outs),
                 (ins brtarget:$target),
                 "bu $target",
                 []>;

def BRBU_lu6 : _FLU6<
                 (outs),
                 (ins brtarget:$target),
                 "bu $target",
                 []>;

def BRFU_u6 : _FU6<
                 (outs),
                 (ins brtarget:$target),
                 "bu $target",
                 []>;

def BRFU_lu6 : _FLU6<
                 (outs),
                 (ins brtarget:$target),
                 "bu $target",
                 []>;
}

//let Uses = [CP] in ...
let Predicates = [HasXS1B], Defs = [R11], neverHasSideEffects = 1,
  isReMaterializable = 1 in
def LDAWCP_u6: _FRU6<(outs), (ins MEMii:$a),
                    "ldaw r11, cp[$a]",
                    []>;

let Predicates = [HasXS1B], Defs = [R11], isReMaterializable = 1 in
def LDAWCP_lu6: _FLRU6<
                    (outs), (ins MEMii:$a),
                    "ldaw r11, cp[$a]",
                    [(set R11, ADDRcpii:$a)]>;

// U10
// TODO ldwcpl, blacp

let Defs = [R11], isReMaterializable = 1, neverHasSideEffects = 1 in
def LDAP_u10 : _FU10<
                  (outs),
                  (ins i32imm:$addr),
                  "ldap r11, $addr",
                  []>;

let Defs = [R11], isReMaterializable = 1 in
def LDAP_lu10 : _FLU10<
                  (outs),
                  (ins i32imm:$addr),
                  "ldap r11, $addr",
                  [(set R11, (pcrelwrapper tglobaladdr:$addr))]>;

let isCall=1,
// All calls clobber the the link register and the non-callee-saved registers:
Defs = [R0, R1, R2, R3, R11, LR] in {
def BL_u10 : _FU10<
                  (outs),
                  (ins calltarget:$target, variable_ops),
                  "bl $target",
                  [(XCoreBranchLink immU10:$target)]>;

def BL_lu10 : _FLU10<
                  (outs),
                  (ins calltarget:$target, variable_ops),
                  "bl $target",
                  [(XCoreBranchLink immU20:$target)]>;
}

// Two operand short
// TODO getr, getst
def NOT : _F2R<(outs GRRegs:$dst), (ins GRRegs:$b),
                 "not $dst, $b",
                 [(set GRRegs:$dst, (not GRRegs:$b))]>;

def NEG : _F2R<(outs GRRegs:$dst), (ins GRRegs:$b),
                 "neg $dst, $b",
                 [(set GRRegs:$dst, (ineg GRRegs:$b))]>;

// TODO setd, eet, eef, getts, setpt, outct, inct, chkct, outt, intt, out,
// in, outshr, inshr, testct, testwct, tinitpc, tinitdp, tinitsp, tinitcp,
// tsetmr, sext (reg), zext (reg)
let isTwoAddress = 1 in {
let neverHasSideEffects = 1 in
def SEXT_rus : _FRUS<(outs GRRegs:$dst), (ins GRRegs:$src1, i32imm:$src2),
                 "sext $dst, $src2",
                 []>;

let neverHasSideEffects = 1 in
def ZEXT_rus : _FRUS<(outs GRRegs:$dst), (ins GRRegs:$src1, i32imm:$src2),
                 "zext $dst, $src2",
                 []>;

def ANDNOT_2r : _F2R<(outs GRRegs:$dst), (ins GRRegs:$src1, GRRegs:$src2),
                 "andnot $dst, $src2",
                 [(set GRRegs:$dst, (and GRRegs:$src1, (not GRRegs:$src2)))]>;
}

let isReMaterializable = 1, neverHasSideEffects = 1 in
def MKMSK_rus : _FRUS<(outs GRRegs:$dst), (ins i32imm:$size),
                 "mkmsk $dst, $size",
                 []>;

def MKMSK_2r : _FRUS<(outs GRRegs:$dst), (ins GRRegs:$size),
                 "mkmsk $dst, $size",
                 [(set GRRegs:$dst, (add (shl 1, GRRegs:$size), 0xffffffff))]>;

// Two operand long
// TODO settw, setclk, setrdy, setpsc, endin, peek,
// getd, testlcl, tinitlr, getps, setps
def BITREV_l2r : _FL2R<(outs GRRegs:$dst), (ins GRRegs:$src),
                 "bitrev $dst, $src",
                 [(set GRRegs:$dst, (int_xcore_bitrev GRRegs:$src))]>;

def BYTEREV_l2r : _FL2R<(outs GRRegs:$dst), (ins GRRegs:$src),
                 "byterev $dst, $src",
                 [(set GRRegs:$dst, (bswap GRRegs:$src))]>;

def CLZ_l2r : _FL2R<(outs GRRegs:$dst), (ins GRRegs:$src),
                 "clz $dst, $src",
                 [(set GRRegs:$dst, (ctlz GRRegs:$src))]>;

// One operand short
// TODO edu, eeu, waitet, waitef, freer, tstart, msync, mjoin, syncr, clrtp
// bru, setdp, setcp, setv, setev, kcall
// dgetreg
let isBranch=1, isIndirectBranch=1, isTerminator=1 in
def BAU_1r : _F1R<(outs), (ins GRRegs:$addr),
                 "bau $addr",
                 [(brind GRRegs:$addr)]>;

let Defs=[SP], neverHasSideEffects=1 in
def SETSP_1r : _F1R<(outs), (ins GRRegs:$src),
                 "set sp, $src",
                 []>;

let isBarrier = 1, hasCtrlDep = 1 in 
def ECALLT_1r : _F1R<(outs), (ins GRRegs:$src),
                 "ecallt $src",
                 []>;

let isBarrier = 1, hasCtrlDep = 1 in 
def ECALLF_1r : _F1R<(outs), (ins GRRegs:$src),
                 "ecallf $src",
                 []>;

let isCall=1, 
// All calls clobber the the link register and the non-callee-saved registers:
Defs = [R0, R1, R2, R3, R11, LR] in {
def BLA_1r : _F1R<(outs), (ins GRRegs:$addr, variable_ops),
                 "bla $addr",
                 [(XCoreBranchLink GRRegs:$addr)]>;
}

// Zero operand short
// TODO waiteu, clre, ssync, freet, ldspc, stspc, ldssr, stssr, ldsed, stsed,
// stet, geted, getet, getkep, getksp, setkep, getid, kret, dcall, dret,
// dentsp, drestsp

let Defs = [R11] in
def GETID_0R : _F0R<(outs), (ins),
                 "get r11, id",
                 [(set R11, (int_xcore_getid))]>;

//===----------------------------------------------------------------------===//
// Non-Instruction Patterns
//===----------------------------------------------------------------------===//

def : Pat<(XCoreBranchLink tglobaladdr:$addr), (BL_lu10 tglobaladdr:$addr)>;
def : Pat<(XCoreBranchLink texternalsym:$addr), (BL_lu10 texternalsym:$addr)>;

/// sext_inreg
def : Pat<(sext_inreg GRRegs:$b, i1), (SEXT_rus GRRegs:$b, 1)>;
def : Pat<(sext_inreg GRRegs:$b, i8), (SEXT_rus GRRegs:$b, 8)>;
def : Pat<(sext_inreg GRRegs:$b, i16), (SEXT_rus GRRegs:$b, 16)>;

/// loads
def : Pat<(zextloadi8 (add GRRegs:$addr, GRRegs:$offset)),
          (LD8U_3r GRRegs:$addr, GRRegs:$offset)>;
def : Pat<(zextloadi8 GRRegs:$addr), (LD8U_3r GRRegs:$addr, (LDC_ru6 0))>;

def : Pat<(zextloadi16 (lda16f GRRegs:$addr, GRRegs:$offset)),
          (LD16S_3r GRRegs:$addr, GRRegs:$offset)>;
def : Pat<(sextloadi16 GRRegs:$addr), (LD16S_3r GRRegs:$addr, (LDC_ru6 0))>;

def : Pat<(load (ldawf GRRegs:$addr, GRRegs:$offset)),
          (LDW_3r GRRegs:$addr, GRRegs:$offset)>;
def : Pat<(load (add GRRegs:$addr, immUs4:$offset)),
          (LDW_2rus GRRegs:$addr, (div4_xform immUs4:$offset))>;
def : Pat<(load GRRegs:$addr), (LDW_2rus GRRegs:$addr, 0)>;

/// anyext
def : Pat<(extloadi8 (add GRRegs:$addr, GRRegs:$offset)),
          (LD8U_3r GRRegs:$addr, GRRegs:$offset)>;
def : Pat<(extloadi8 GRRegs:$addr), (LD8U_3r GRRegs:$addr, (LDC_ru6 0))>;
def : Pat<(extloadi16 (lda16f GRRegs:$addr, GRRegs:$offset)),
          (LD16S_3r GRRegs:$addr, GRRegs:$offset)>;
def : Pat<(extloadi16 GRRegs:$addr), (LD16S_3r GRRegs:$addr, (LDC_ru6 0))>;

/// stores
def : Pat<(truncstorei8 GRRegs:$val, (add GRRegs:$addr, GRRegs:$offset)),
          (ST8_l3r GRRegs:$val, GRRegs:$addr, GRRegs:$offset)>;
def : Pat<(truncstorei8 GRRegs:$val, GRRegs:$addr),
          (ST8_l3r GRRegs:$val, GRRegs:$addr, (LDC_ru6 0))>;
          
def : Pat<(truncstorei16 GRRegs:$val, (lda16f GRRegs:$addr, GRRegs:$offset)),
          (ST16_l3r GRRegs:$val, GRRegs:$addr, GRRegs:$offset)>;
def : Pat<(truncstorei16 GRRegs:$val, GRRegs:$addr),
          (ST16_l3r GRRegs:$val, GRRegs:$addr, (LDC_ru6 0))>;

def : Pat<(store GRRegs:$val, (ldawf GRRegs:$addr, GRRegs:$offset)),
          (STW_3r GRRegs:$val, GRRegs:$addr, GRRegs:$offset)>;
def : Pat<(store GRRegs:$val, (add GRRegs:$addr, immUs4:$offset)),
          (STW_2rus GRRegs:$val, GRRegs:$addr, (div4_xform immUs4:$offset))>;
def : Pat<(store GRRegs:$val, GRRegs:$addr),
          (STW_2rus GRRegs:$val, GRRegs:$addr, 0)>;

/// cttz
def : Pat<(cttz GRRegs:$src), (CLZ_l2r (BITREV_l2r GRRegs:$src))>;

/// trap
def : Pat<(trap), (ECALLF_1r (LDC_ru6 0))>;

///
/// branch patterns
///

// unconditional branch
def : Pat<(br bb:$addr), (BRFU_lu6 bb:$addr)>;

// direct match equal/notequal zero brcond
def : Pat<(brcond (setne GRRegs:$lhs, 0), bb:$dst),
          (BRFT_lru6 GRRegs:$lhs, bb:$dst)>;
def : Pat<(brcond (seteq GRRegs:$lhs, 0), bb:$dst),
          (BRFF_lru6 GRRegs:$lhs, bb:$dst)>;

def : Pat<(brcond (setle GRRegs:$lhs, GRRegs:$rhs), bb:$dst),
          (BRFF_lru6 (LSS_3r GRRegs:$rhs, GRRegs:$lhs), bb:$dst)>;
def : Pat<(brcond (setule GRRegs:$lhs, GRRegs:$rhs), bb:$dst),
          (BRFF_lru6 (LSU_3r GRRegs:$rhs, GRRegs:$lhs), bb:$dst)>;
def : Pat<(brcond (setge GRRegs:$lhs, GRRegs:$rhs), bb:$dst),
          (BRFF_lru6 (LSS_3r GRRegs:$lhs, GRRegs:$rhs), bb:$dst)>;
def : Pat<(brcond (setuge GRRegs:$lhs, GRRegs:$rhs), bb:$dst),
          (BRFF_lru6 (LSU_3r GRRegs:$lhs, GRRegs:$rhs), bb:$dst)>;
def : Pat<(brcond (setne GRRegs:$lhs, GRRegs:$rhs), bb:$dst),
          (BRFF_lru6 (EQ_3r GRRegs:$lhs, GRRegs:$rhs), bb:$dst)>;
def : Pat<(brcond (setne GRRegs:$lhs, immUs:$rhs), bb:$dst),
          (BRFF_lru6 (EQ_2rus GRRegs:$lhs, immUs:$rhs), bb:$dst)>;

// generic brcond pattern
def : Pat<(brcond GRRegs:$cond, bb:$addr), (BRFT_lru6 GRRegs:$cond, bb:$addr)>;


///
/// Select patterns
///

// direct match equal/notequal zero select
def : Pat<(select (setne GRRegs:$lhs, 0), GRRegs:$T, GRRegs:$F),
        (SELECT_CC GRRegs:$lhs, GRRegs:$T, GRRegs:$F)>;

def : Pat<(select (seteq GRRegs:$lhs, 0), GRRegs:$T, GRRegs:$F),
        (SELECT_CC GRRegs:$lhs, GRRegs:$F, GRRegs:$T)>;

def : Pat<(select (setle GRRegs:$lhs, GRRegs:$rhs), GRRegs:$T, GRRegs:$F),
          (SELECT_CC (LSS_3r GRRegs:$rhs, GRRegs:$lhs), GRRegs:$F, GRRegs:$T)>;
def : Pat<(select (setule GRRegs:$lhs, GRRegs:$rhs), GRRegs:$T, GRRegs:$F),
          (SELECT_CC (LSU_3r GRRegs:$rhs, GRRegs:$lhs), GRRegs:$F, GRRegs:$T)>;
def : Pat<(select (setge GRRegs:$lhs, GRRegs:$rhs), GRRegs:$T, GRRegs:$F),
          (SELECT_CC (LSS_3r GRRegs:$lhs, GRRegs:$rhs), GRRegs:$F, GRRegs:$T)>;
def : Pat<(select (setuge GRRegs:$lhs, GRRegs:$rhs), GRRegs:$T, GRRegs:$F),
          (SELECT_CC (LSU_3r GRRegs:$lhs, GRRegs:$rhs), GRRegs:$F, GRRegs:$T)>;
def : Pat<(select (setne GRRegs:$lhs, GRRegs:$rhs), GRRegs:$T, GRRegs:$F),
          (SELECT_CC (EQ_3r GRRegs:$lhs, GRRegs:$rhs), GRRegs:$F, GRRegs:$T)>;
def : Pat<(select (setne GRRegs:$lhs, immUs:$rhs), GRRegs:$T, GRRegs:$F),
          (SELECT_CC (EQ_2rus GRRegs:$lhs, immUs:$rhs), GRRegs:$F, GRRegs:$T)>;

///
/// setcc patterns, only matched when none of the above brcond
/// patterns match
///

// setcc 2 register operands
def : Pat<(setle GRRegs:$lhs, GRRegs:$rhs),
          (EQ_2rus (LSS_3r GRRegs:$rhs, GRRegs:$lhs), 0)>;
def : Pat<(setule GRRegs:$lhs, GRRegs:$rhs),
          (EQ_2rus (LSU_3r GRRegs:$rhs, GRRegs:$lhs), 0)>;

def : Pat<(setgt GRRegs:$lhs, GRRegs:$rhs),
          (LSS_3r GRRegs:$rhs, GRRegs:$lhs)>;
def : Pat<(setugt GRRegs:$lhs, GRRegs:$rhs),
          (LSU_3r GRRegs:$rhs, GRRegs:$lhs)>;

def : Pat<(setge GRRegs:$lhs, GRRegs:$rhs),
          (EQ_2rus (LSS_3r GRRegs:$lhs, GRRegs:$rhs), 0)>;
def : Pat<(setuge GRRegs:$lhs, GRRegs:$rhs),
          (EQ_2rus (LSU_3r GRRegs:$lhs, GRRegs:$rhs), 0)>;

def : Pat<(setlt GRRegs:$lhs, GRRegs:$rhs),
          (LSS_3r GRRegs:$lhs, GRRegs:$rhs)>;
def : Pat<(setult GRRegs:$lhs, GRRegs:$rhs),
          (LSU_3r GRRegs:$lhs, GRRegs:$rhs)>;

def : Pat<(setne GRRegs:$lhs, GRRegs:$rhs),
          (EQ_2rus (EQ_3r GRRegs:$lhs, GRRegs:$rhs), 0)>;

def : Pat<(seteq GRRegs:$lhs, GRRegs:$rhs),
          (EQ_3r GRRegs:$lhs, GRRegs:$rhs)>;

// setcc reg/imm operands
def : Pat<(seteq GRRegs:$lhs, immUs:$rhs),
          (EQ_2rus GRRegs:$lhs, immUs:$rhs)>;
def : Pat<(setne GRRegs:$lhs, immUs:$rhs),
          (EQ_2rus (EQ_2rus GRRegs:$lhs, immUs:$rhs), 0)>;

// misc
def : Pat<(add GRRegs:$addr, immUs4:$offset),
          (LDAWF_l2rus GRRegs:$addr, (div4_xform immUs4:$offset))>;

def : Pat<(sub GRRegs:$addr, immUs4:$offset),
          (LDAWB_l2rus GRRegs:$addr, (div4_xform immUs4:$offset))>;

def : Pat<(and GRRegs:$val, immMskBitp:$mask),
          (ZEXT_rus GRRegs:$val, (msksize_xform immMskBitp:$mask))>;

// (sub X, imm) gets canonicalized to (add X, -imm).  Match this form.
def : Pat<(add GRRegs:$src1, immUsNeg:$src2),
          (SUB_2rus GRRegs:$src1, (neg_xform immUsNeg:$src2))>;

def : Pat<(add GRRegs:$src1, immUs4Neg:$src2),
          (LDAWB_l2rus GRRegs:$src1, (div4neg_xform immUs4Neg:$src2))>;

///
/// Some peepholes
///

def : Pat<(mul GRRegs:$src, 3),
          (LDA16F_l3r GRRegs:$src, GRRegs:$src)>;

def : Pat<(mul GRRegs:$src, 5),
          (LDAWF_l3r GRRegs:$src, GRRegs:$src)>;

def : Pat<(mul GRRegs:$src, -3),
          (LDAWB_l3r GRRegs:$src, GRRegs:$src)>;

// ashr X, 32 is equivalent to ashr X, 31 on the XCore.
def : Pat<(sra GRRegs:$src, 31),
          (ASHR_l2rus GRRegs:$src, 32)>;

