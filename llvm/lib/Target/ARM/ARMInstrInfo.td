//===- ARMInstrInfo.td - Target Description for ARM Target -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the ARM instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ARM specific DAG Nodes.
//

// Type profiles.
def SDT_ARMCallSeqStart : SDCallSeqStart<[ SDTCisVT<0, i32> ]>;
def SDT_ARMCallSeqEnd   : SDCallSeqEnd<[ SDTCisVT<0, i32>, SDTCisVT<1, i32> ]>;

def SDT_ARMSaveCallPC : SDTypeProfile<0, 1, []>;

def SDT_ARMcall    : SDTypeProfile<0, -1, [SDTCisInt<0>]>;

def SDT_ARMCMov    : SDTypeProfile<1, 3,
                                   [SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>,
                                    SDTCisVT<3, i32>]>;

def SDT_ARMBrcond  : SDTypeProfile<0, 2,
                                   [SDTCisVT<0, OtherVT>, SDTCisVT<1, i32>]>;

def SDT_ARMBrJT    : SDTypeProfile<0, 3,
                                  [SDTCisPtrTy<0>, SDTCisVT<1, i32>,
                                   SDTCisVT<2, i32>]>;

def SDT_ARMCmp     : SDTypeProfile<0, 2, [SDTCisSameAs<0, 1>]>;

def SDT_ARMPICAdd  : SDTypeProfile<1, 2, [SDTCisSameAs<0, 1>,
                                          SDTCisPtrTy<1>, SDTCisVT<2, i32>]>;

def SDT_ARMThreadPointer : SDTypeProfile<1, 0, [SDTCisPtrTy<0>]>;

// Node definitions.
def ARMWrapper       : SDNode<"ARMISD::Wrapper",     SDTIntUnaryOp>;
def ARMWrapperJT     : SDNode<"ARMISD::WrapperJT",   SDTIntBinOp>;

def ARMcallseq_start : SDNode<"ISD::CALLSEQ_START", SDT_ARMCallSeqStart,
                              [SDNPHasChain, SDNPOutFlag]>;
def ARMcallseq_end   : SDNode<"ISD::CALLSEQ_END",   SDT_ARMCallSeqEnd,
                              [SDNPHasChain, SDNPOptInFlag, SDNPOutFlag]>;

def ARMcall          : SDNode<"ARMISD::CALL", SDT_ARMcall,
                              [SDNPHasChain, SDNPOptInFlag, SDNPOutFlag]>;
def ARMcall_pred    : SDNode<"ARMISD::CALL_PRED", SDT_ARMcall,
                              [SDNPHasChain, SDNPOptInFlag, SDNPOutFlag]>;
def ARMcall_nolink   : SDNode<"ARMISD::CALL_NOLINK", SDT_ARMcall,
                              [SDNPHasChain, SDNPOptInFlag, SDNPOutFlag]>;

def ARMretflag       : SDNode<"ARMISD::RET_FLAG", SDTNone,
                              [SDNPHasChain, SDNPOptInFlag]>;

def ARMcmov          : SDNode<"ARMISD::CMOV", SDT_ARMCMov,
                              [SDNPInFlag]>;
def ARMcneg          : SDNode<"ARMISD::CNEG", SDT_ARMCMov,
                              [SDNPInFlag]>;

def ARMbrcond        : SDNode<"ARMISD::BRCOND", SDT_ARMBrcond,
                              [SDNPHasChain, SDNPInFlag, SDNPOutFlag]>;

def ARMbrjt          : SDNode<"ARMISD::BR_JT", SDT_ARMBrJT,
                              [SDNPHasChain]>;

def ARMcmp           : SDNode<"ARMISD::CMP", SDT_ARMCmp,
                              [SDNPOutFlag]>;

def ARMcmpNZ         : SDNode<"ARMISD::CMPNZ", SDT_ARMCmp,
                              [SDNPOutFlag]>;

def ARMpic_add       : SDNode<"ARMISD::PIC_ADD", SDT_ARMPICAdd>;

def ARMsrl_flag      : SDNode<"ARMISD::SRL_FLAG", SDTIntUnaryOp, [SDNPOutFlag]>;
def ARMsra_flag      : SDNode<"ARMISD::SRA_FLAG", SDTIntUnaryOp, [SDNPOutFlag]>;
def ARMrrx           : SDNode<"ARMISD::RRX"     , SDTIntUnaryOp, [SDNPInFlag ]>;

def ARMthread_pointer: SDNode<"ARMISD::THREAD_POINTER", SDT_ARMThreadPointer>;

//===----------------------------------------------------------------------===//
// ARM Instruction Predicate Definitions.
//
def HasV5T  : Predicate<"Subtarget->hasV5TOps()">;
def HasV5TE : Predicate<"Subtarget->hasV5TEOps()">;
def HasV6   : Predicate<"Subtarget->hasV6Ops()">;
def IsThumb : Predicate<"Subtarget->isThumb()">;
def IsARM   : Predicate<"!Subtarget->isThumb()">;

//===----------------------------------------------------------------------===//
// ARM Flag Definitions.

class RegConstraint<string C> {
  string Constraints = C;
}

//===----------------------------------------------------------------------===//
//  ARM specific transformation functions and pattern fragments.
//

// so_imm_XFORM - Return a so_imm value packed into the format described for
// so_imm def below.
def so_imm_XFORM : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(ARM_AM::getSOImmVal(N->getZExtValue()),
                                   MVT::i32);
}]>;

// so_imm_neg_XFORM - Return a so_imm value packed into the format described for
// so_imm_neg def below.
def so_imm_neg_XFORM : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(ARM_AM::getSOImmVal(-(int)N->getZExtValue()),
                                   MVT::i32);
}]>;

// so_imm_not_XFORM - Return a so_imm value packed into the format described for
// so_imm_not def below.
def so_imm_not_XFORM : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(ARM_AM::getSOImmVal(~(int)N->getZExtValue()),
                                   MVT::i32);
}]>;

// rot_imm predicate - True if the 32-bit immediate is equal to 8, 16, or 24.
def rot_imm : PatLeaf<(i32 imm), [{
  int32_t v = (int32_t)N->getZExtValue();
  return v == 8 || v == 16 || v == 24;
}]>;

/// imm1_15 predicate - True if the 32-bit immediate is in the range [1,15].
def imm1_15 : PatLeaf<(i32 imm), [{
  return (int32_t)N->getZExtValue() >= 1 && (int32_t)N->getZExtValue() < 16;
}]>;

/// imm16_31 predicate - True if the 32-bit immediate is in the range [16,31].
def imm16_31 : PatLeaf<(i32 imm), [{
  return (int32_t)N->getZExtValue() >= 16 && (int32_t)N->getZExtValue() < 32;
}]>;

def so_imm_neg : 
  PatLeaf<(imm), [{
    return ARM_AM::getSOImmVal(-(int)N->getZExtValue()) != -1;
  }], so_imm_neg_XFORM>;

def so_imm_not :
  PatLeaf<(imm), [{
    return ARM_AM::getSOImmVal(~(int)N->getZExtValue()) != -1;
  }], so_imm_not_XFORM>;

// sext_16_node predicate - True if the SDNode is sign-extended 16 or more bits.
def sext_16_node : PatLeaf<(i32 GPR:$a), [{
  return CurDAG->ComputeNumSignBits(SDValue(N,0)) >= 17;
}]>;

class BinOpFrag<dag res> : PatFrag<(ops node:$LHS, node:$RHS), res>;
class UnOpFrag <dag res> : PatFrag<(ops node:$Src), res>;

//===----------------------------------------------------------------------===//
// Operand Definitions.
//

// Branch target.
def brtarget : Operand<OtherVT>;

// A list of registers separated by comma. Used by load/store multiple.
def reglist : Operand<i32> {
  let PrintMethod = "printRegisterList";
}

// An operand for the CONSTPOOL_ENTRY pseudo-instruction.
def cpinst_operand : Operand<i32> {
  let PrintMethod = "printCPInstOperand";
}

def jtblock_operand : Operand<i32> {
  let PrintMethod = "printJTBlockOperand";
}

// Local PC labels.
def pclabel : Operand<i32> {
  let PrintMethod = "printPCLabel";
}

// shifter_operand operands: so_reg and so_imm.
def so_reg : Operand<i32>,    // reg reg imm
            ComplexPattern<i32, 3, "SelectShifterOperandReg",
                            [shl,srl,sra,rotr]> {
  let PrintMethod = "printSORegOperand";
  let MIOperandInfo = (ops GPR, GPR, i32imm);
}

// so_imm - Match a 32-bit shifter_operand immediate operand, which is an
// 8-bit immediate rotated by an arbitrary number of bits.  so_imm values are
// represented in the imm field in the same 12-bit form that they are encoded
// into so_imm instructions: the 8-bit immediate is the least significant bits
// [bits 0-7], the 4-bit shift amount is the next 4 bits [bits 8-11].
def so_imm : Operand<i32>,
             PatLeaf<(imm),
                     [{ return ARM_AM::getSOImmVal(N->getZExtValue()) != -1; }],
                     so_imm_XFORM> {
  let PrintMethod = "printSOImmOperand";
}

// Break so_imm's up into two pieces.  This handles immediates with up to 16
// bits set in them.  This uses so_imm2part to match and so_imm2part_[12] to
// get the first/second pieces.
def so_imm2part : Operand<i32>,
                  PatLeaf<(imm), [{
      return ARM_AM::isSOImmTwoPartVal((unsigned)N->getZExtValue());
    }]> {
  let PrintMethod = "printSOImm2PartOperand";
}

def so_imm2part_1 : SDNodeXForm<imm, [{
  unsigned V = ARM_AM::getSOImmTwoPartFirst((unsigned)N->getZExtValue());
  return CurDAG->getTargetConstant(ARM_AM::getSOImmVal(V), MVT::i32);
}]>;

def so_imm2part_2 : SDNodeXForm<imm, [{
  unsigned V = ARM_AM::getSOImmTwoPartSecond((unsigned)N->getZExtValue());
  return CurDAG->getTargetConstant(ARM_AM::getSOImmVal(V), MVT::i32);
}]>;


// Define ARM specific addressing modes.

// addrmode2 := reg +/- reg shop imm
// addrmode2 := reg +/- imm12
//
def addrmode2 : Operand<i32>,
                ComplexPattern<i32, 3, "SelectAddrMode2", []> {
  let PrintMethod = "printAddrMode2Operand";
  let MIOperandInfo = (ops GPR:$base, GPR:$offsreg, i32imm:$offsimm);
}

def am2offset : Operand<i32>,
                ComplexPattern<i32, 2, "SelectAddrMode2Offset", []> {
  let PrintMethod = "printAddrMode2OffsetOperand";
  let MIOperandInfo = (ops GPR, i32imm);
}

// addrmode3 := reg +/- reg
// addrmode3 := reg +/- imm8
//
def addrmode3 : Operand<i32>,
                ComplexPattern<i32, 3, "SelectAddrMode3", []> {
  let PrintMethod = "printAddrMode3Operand";
  let MIOperandInfo = (ops GPR:$base, GPR:$offsreg, i32imm:$offsimm);
}

def am3offset : Operand<i32>,
                ComplexPattern<i32, 2, "SelectAddrMode3Offset", []> {
  let PrintMethod = "printAddrMode3OffsetOperand";
  let MIOperandInfo = (ops GPR, i32imm);
}

// addrmode4 := reg, <mode|W>
//
def addrmode4 : Operand<i32>,
                ComplexPattern<i32, 2, "", []> {
  let PrintMethod = "printAddrMode4Operand";
  let MIOperandInfo = (ops GPR, i32imm);
}

// addrmode5 := reg +/- imm8*4
//
def addrmode5 : Operand<i32>,
                ComplexPattern<i32, 2, "SelectAddrMode5", []> {
  let PrintMethod = "printAddrMode5Operand";
  let MIOperandInfo = (ops GPR, i32imm);
}

// addrmodepc := pc + reg
//
def addrmodepc : Operand<i32>,
                 ComplexPattern<i32, 2, "SelectAddrModePC", []> {
  let PrintMethod = "printAddrModePCOperand";
  let MIOperandInfo = (ops GPR, i32imm);
}

// ARM Predicate operand. Default to 14 = always (AL). Second part is CC
// register whose default is 0 (no register).
def pred : PredicateOperand<OtherVT, (ops i32imm, CCR),
                                     (ops (i32 14), (i32 zero_reg))> {
  let PrintMethod = "printPredicateOperand";
}

// Conditional code result for instructions whose 's' bit is set, e.g. subs.
//
def cc_out : OptionalDefOperand<OtherVT, (ops CCR), (ops (i32 zero_reg))> {
  let PrintMethod = "printSBitModifierOperand";
}

//===----------------------------------------------------------------------===//
// ARM Instruction flags.  These need to match ARMInstrInfo.h.
//

// Addressing mode.
class AddrMode<bits<4> val> {
  bits<4> Value = val;
}
def AddrModeNone : AddrMode<0>;
def AddrMode1    : AddrMode<1>;
def AddrMode2    : AddrMode<2>;
def AddrMode3    : AddrMode<3>;
def AddrMode4    : AddrMode<4>;
def AddrMode5    : AddrMode<5>;
def AddrModeT1   : AddrMode<6>;
def AddrModeT2   : AddrMode<7>;
def AddrModeT4   : AddrMode<8>;
def AddrModeTs   : AddrMode<9>;

// Instruction size.
class SizeFlagVal<bits<3> val> {
  bits<3> Value = val;
}
def SizeInvalid  : SizeFlagVal<0>;  // Unset.
def SizeSpecial  : SizeFlagVal<1>;  // Pseudo or special.
def Size8Bytes   : SizeFlagVal<2>;
def Size4Bytes   : SizeFlagVal<3>;
def Size2Bytes   : SizeFlagVal<4>;

// Load / store index mode.
class IndexMode<bits<2> val> {
  bits<2> Value = val;
}
def IndexModeNone : IndexMode<0>;
def IndexModePre  : IndexMode<1>;
def IndexModePost : IndexMode<2>;

//===----------------------------------------------------------------------===//

include "ARMInstrFormats.td"

//===----------------------------------------------------------------------===//
// Multiclass helpers...
//

/// AsI1_bin_irs - Defines a set of (op r, {so_imm|r|so_reg}) patterns for a
/// binop that produces a value.
multiclass AsI1_bin_irs<bits<4> opcod, string opc, PatFrag opnode> {
  def ri : AsI1<opcod, (outs GPR:$dst), (ins GPR:$a, so_imm:$b), DPRIm,
               opc, " $dst, $a, $b",
               [(set GPR:$dst, (opnode GPR:$a, so_imm:$b))]>;
  def rr : AsI1<opcod, (outs GPR:$dst), (ins GPR:$a, GPR:$b), DPRReg,
               opc, " $dst, $a, $b",
               [(set GPR:$dst, (opnode GPR:$a, GPR:$b))]>;
  def rs : AsI1<opcod, (outs GPR:$dst), (ins GPR:$a, so_reg:$b), DPRSoReg,
               opc, " $dst, $a, $b",
               [(set GPR:$dst, (opnode GPR:$a, so_reg:$b))]>;
}

/// ASI1_bin_s_irs - Similar to AsI1_bin_irs except it sets the 's' bit so the
/// instruction modifies the CSPR register.
let Defs = [CPSR] in {
multiclass ASI1_bin_s_irs<bits<4> opcod, string opc, PatFrag opnode> {
  def ri : AI1<opcod, (outs GPR:$dst), (ins GPR:$a, so_imm:$b), DPRImS,
               opc, "s $dst, $a, $b",
               [(set GPR:$dst, (opnode GPR:$a, so_imm:$b))]>;
  def rr : AI1<opcod, (outs GPR:$dst), (ins GPR:$a, GPR:$b), DPRRegS,
               opc, "s $dst, $a, $b",
               [(set GPR:$dst, (opnode GPR:$a, GPR:$b))]>;
  def rs : AI1<opcod, (outs GPR:$dst), (ins GPR:$a, so_reg:$b), DPRSoRegS,
               opc, "s $dst, $a, $b",
               [(set GPR:$dst, (opnode GPR:$a, so_reg:$b))]>;
}
}

/// AI1_cmp_irs - Defines a set of (op r, {so_imm|r|so_reg}) cmp / test
/// patterns. Similar to AsI1_bin_irs except the instruction does not produce
/// a explicit result, only implicitly set CPSR.
let Defs = [CPSR] in {
multiclass AI1_cmp_irs<bits<4> opcod, string opc, PatFrag opnode> {
  def ri : AI1<opcod, (outs), (ins GPR:$a, so_imm:$b), DPRnIm,
               opc, " $a, $b",
               [(opnode GPR:$a, so_imm:$b)]>;
  def rr : AI1<opcod, (outs), (ins GPR:$a, GPR:$b), DPRnReg,
               opc, " $a, $b",
               [(opnode GPR:$a, GPR:$b)]>;
  def rs : AI1<opcod, (outs), (ins GPR:$a, so_reg:$b), DPRnSoReg,
               opc, " $a, $b",
               [(opnode GPR:$a, so_reg:$b)]>;
}
}

/// AI_unary_rrot - A unary operation with two forms: one whose operand is a
/// register and one whose operand is a register rotated by 8/16/24.
multiclass AI_unary_rrot<bits<4> opcod, string opc, PatFrag opnode> {
  def r     : AI<opcod, (outs GPR:$dst), (ins GPR:$Src), Pseudo,
                 opc, " $dst, $Src",
                 [(set GPR:$dst, (opnode GPR:$Src))]>, Requires<[IsARM, HasV6]>;
  def r_rot : AI<opcod, (outs GPR:$dst), (ins GPR:$Src, i32imm:$rot), Pseudo,
                 opc, " $dst, $Src, ror $rot",
                 [(set GPR:$dst, (opnode (rotr GPR:$Src, rot_imm:$rot)))]>,
              Requires<[IsARM, HasV6]>;
}

/// AI_bin_rrot - A binary operation with two forms: one whose operand is a
/// register and one whose operand is a register rotated by 8/16/24.
multiclass AI_bin_rrot<bits<4> opcod, string opc, PatFrag opnode> {
  def rr     : AI<opcod, (outs GPR:$dst), (ins GPR:$LHS, GPR:$RHS),
                  Pseudo, opc, " $dst, $LHS, $RHS",
                  [(set GPR:$dst, (opnode GPR:$LHS, GPR:$RHS))]>,
                  Requires<[IsARM, HasV6]>;
  def rr_rot : AI<opcod, (outs GPR:$dst), (ins GPR:$LHS, GPR:$RHS, i32imm:$rot),
                  Pseudo, opc, " $dst, $LHS, $RHS, ror $rot",
                  [(set GPR:$dst, (opnode GPR:$LHS,
                                          (rotr GPR:$RHS, rot_imm:$rot)))]>,
                  Requires<[IsARM, HasV6]>;
}

/// AsXI1_bin_c_irs - Same as AsI1_bin_irs but without the predicate operand and
/// setting carry bit. But it can optionally set CPSR.
let Uses = [CPSR] in {
multiclass AsXI1_bin_c_irs<bits<4> opcod, string opc, PatFrag opnode> {
  def ri : AXI1<opcod, (outs GPR:$dst), (ins GPR:$a, so_imm:$b, cc_out:$s),
                DPRIm, !strconcat(opc, "${s} $dst, $a, $b"),
               [(set GPR:$dst, (opnode GPR:$a, so_imm:$b))]>;
  def rr : AXI1<opcod, (outs GPR:$dst), (ins GPR:$a, GPR:$b, cc_out:$s),
                DPRReg, !strconcat(opc, "${s} $dst, $a, $b"),
               [(set GPR:$dst, (opnode GPR:$a, GPR:$b))]>;
  def rs : AXI1<opcod, (outs GPR:$dst), (ins GPR:$a, so_reg:$b, cc_out:$s),
                DPRSoReg, !strconcat(opc, "${s} $dst, $a, $b"),
               [(set GPR:$dst, (opnode GPR:$a, so_reg:$b))]>;
}
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Miscellaneous Instructions.
//

/// CONSTPOOL_ENTRY - This instruction represents a floating constant pool in
/// the function.  The first operand is the ID# for this instruction, the second
/// is the index into the MachineConstantPool that this is, the third is the
/// size in bytes of this constant pool entry.
let isNotDuplicable = 1 in
def CONSTPOOL_ENTRY :
PseudoInst<(outs), (ins cpinst_operand:$instid, cpinst_operand:$cpidx,
                        i32imm:$size),
           "${instid:label} ${cpidx:cpentry}", []>;

let Defs = [SP], Uses = [SP] in {
def ADJCALLSTACKUP :
PseudoInst<(outs), (ins i32imm:$amt1, i32imm:$amt2, pred:$p),
           "@ ADJCALLSTACKUP $amt1",
           [(ARMcallseq_end timm:$amt1, timm:$amt2)]>;

def ADJCALLSTACKDOWN : 
PseudoInst<(outs), (ins i32imm:$amt, pred:$p),
           "@ ADJCALLSTACKDOWN $amt",
           [(ARMcallseq_start timm:$amt)]>;
}

def DWARF_LOC :
PseudoInst<(outs), (ins i32imm:$line, i32imm:$col, i32imm:$file),
           ".loc $file, $line, $col",
           [(dwarf_loc (i32 imm:$line), (i32 imm:$col), (i32 imm:$file))]>;

let isNotDuplicable = 1 in {
def PICADD : AXI1<0x0, (outs GPR:$dst), (ins GPR:$a, pclabel:$cp, pred:$p),
                  Pseudo, "$cp:\n\tadd$p $dst, pc, $a",
                   [(set GPR:$dst, (ARMpic_add GPR:$a, imm:$cp))]>;

let AddedComplexity = 10 in {
let isSimpleLoad = 1 in
def PICLD   : AXI2ldw<0x0, (outs GPR:$dst), (ins addrmodepc:$addr, pred:$p),
                  Pseudo, "${addr:label}:\n\tldr$p $dst, $addr",
                  [(set GPR:$dst, (load addrmodepc:$addr))]>;

def PICLDZH : AXI3ldh<0x0, (outs GPR:$dst), (ins addrmodepc:$addr, pred:$p),
                  Pseudo, "${addr:label}:\n\tldr${p}h $dst, $addr",
                  [(set GPR:$dst, (zextloadi16 addrmodepc:$addr))]>;

def PICLDZB : AXI2ldb<0x0, (outs GPR:$dst), (ins addrmodepc:$addr, pred:$p),
                  Pseudo, "${addr:label}:\n\tldr${p}b $dst, $addr",
                  [(set GPR:$dst, (zextloadi8 addrmodepc:$addr))]>;

def PICLDH  : AXI3ldh<0x0, (outs GPR:$dst), (ins addrmodepc:$addr, pred:$p),
                  Pseudo, "${addr:label}:\n\tldr${p}h $dst, $addr",
                  [(set GPR:$dst, (extloadi16 addrmodepc:$addr))]>;

def PICLDB  : AXI2ldb<0x0, (outs GPR:$dst), (ins addrmodepc:$addr, pred:$p),
                  Pseudo, "${addr:label}:\n\tldr${p}b $dst, $addr",
                  [(set GPR:$dst, (extloadi8 addrmodepc:$addr))]>;

def PICLDSH : AXI3ldsh<0x0, (outs GPR:$dst), (ins addrmodepc:$addr, pred:$p),
                  Pseudo, "${addr:label}:\n\tldr${p}sh $dst, $addr",
                  [(set GPR:$dst, (sextloadi16 addrmodepc:$addr))]>;

def PICLDSB : AXI3ldsb<0x0, (outs GPR:$dst), (ins addrmodepc:$addr, pred:$p),
                  Pseudo, "${addr:label}:\n\tldr${p}sb $dst, $addr",
                  [(set GPR:$dst, (sextloadi8 addrmodepc:$addr))]>;
}
let AddedComplexity = 10 in {
def PICSTR  : AXI2stw<0x0, (outs), (ins GPR:$src, addrmodepc:$addr, pred:$p),
               Pseudo, "${addr:label}:\n\tstr$p $src, $addr",
               [(store GPR:$src, addrmodepc:$addr)]>;

def PICSTRH : AXI3sth<0x0, (outs), (ins GPR:$src, addrmodepc:$addr, pred:$p),
               Pseudo, "${addr:label}:\n\tstr${p}h $src, $addr",
               [(truncstorei16 GPR:$src, addrmodepc:$addr)]>;

def PICSTRB : AXI2stb<0x0, (outs), (ins GPR:$src, addrmodepc:$addr, pred:$p),
               Pseudo, "${addr:label}:\n\tstr${p}b $src, $addr",
               [(truncstorei8 GPR:$src, addrmodepc:$addr)]>;
}
}

//===----------------------------------------------------------------------===//
//  Control Flow Instructions.
//

let isReturn = 1, isTerminator = 1 in
  def BX_RET : AI<0x0, (outs), (ins), BranchMisc, "bx", " lr", [(ARMretflag)]> {
  let Inst{4-7}   = {1,0,0,0};
  let Inst{8-19}  = {1,1,1,1,1,1,1,1,1,1,1,1};
  let Inst{20-27} = {0,1,0,0,1,0,0,0};
}

// FIXME: remove when we have a way to marking a MI with these properties.
// FIXME: $dst1 should be a def. But the extra ops must be in the end of the
// operand list.
let isReturn = 1, isTerminator = 1 in
  def LDM_RET : AXI4ldpc<0x0, (outs),
                    (ins addrmode4:$addr, pred:$p, reglist:$dst1, variable_ops),
                    LdFrm, "ldm${p}${addr:submode} $addr, $dst1",
                    []>;

let isCall = 1,
  Defs = [R0, R1, R2, R3, R12, LR,
          D0, D1, D2, D3, D4, D5, D6, D7, CPSR] in {
  def BL  : ABLI<0xB, (outs), (ins i32imm:$func, variable_ops), Branch,
                "bl ${func:call}",
                [(ARMcall tglobaladdr:$func)]>;

  def BL_pred : ABLpredI<0xB, (outs), (ins i32imm:$func, variable_ops), Branch,
                   "bl", " ${func:call}",
                   [(ARMcall_pred tglobaladdr:$func)]>;

  // ARMv5T and above
  def BLX : AXI<0x0, (outs), (ins GPR:$func, variable_ops), BranchMisc,
                "blx $func",
                [(ARMcall GPR:$func)]>, Requires<[IsARM, HasV5T]> {
    let Inst{4-7}   = {1,1,0,0};
    let Inst{8-19}  = {1,1,1,1,1,1,1,1,1,1,1,1};
    let Inst{20-27} = {0,1,0,0,1,0,0,0};
  }

  let Uses = [LR] in {
    // ARMv4T
    def BX : AXIx2<0x0, (outs), (ins GPR:$func, variable_ops),
                   BranchMisc, "mov lr, pc\n\tbx $func",
                   [(ARMcall_nolink GPR:$func)]>;
  }
}

let isBranch = 1, isTerminator = 1 in {
  // B is "predicable" since it can be xformed into a Bcc.
  let isBarrier = 1 in {
    let isPredicable = 1 in
    def B : AXI<0xA, (outs), (ins brtarget:$target), Branch, "b $target",
                [(br bb:$target)]>;

  let isNotDuplicable = 1, isIndirectBranch = 1 in {
  def BR_JTr : JTI<0x0, (outs), (ins GPR:$target, jtblock_operand:$jt, i32imm:$id),
                    "mov pc, $target \n$jt",
                    [(ARMbrjt GPR:$target, tjumptable:$jt, imm:$id)]>;
  def BR_JTm : JTI2<0x0, (outs), (ins addrmode2:$target, jtblock_operand:$jt, i32imm:$id),
                     "ldr pc, $target \n$jt",
                     [(ARMbrjt (i32 (load addrmode2:$target)), tjumptable:$jt,
                       imm:$id)]>;
  def BR_JTadd : JTI1<0x0, (outs), (ins GPR:$target, GPR:$idx, jtblock_operand:$jt,
                       i32imm:$id),
                       "add pc, $target, $idx \n$jt",
                       [(ARMbrjt (add GPR:$target, GPR:$idx), tjumptable:$jt,
                         imm:$id)]>;
  }
  }

  // FIXME: should be able to write a pattern for ARMBrcond, but can't use
  // a two-value operand where a dag node expects two operands. :( 
  def Bcc : ABccI<0xA, (outs), (ins brtarget:$target), Branch,
               "b", " $target",
               [/*(ARMbrcond bb:$target, imm:$cc, CCR:$ccr)*/]>;
}

//===----------------------------------------------------------------------===//
//  Load / store Instructions.
//

// Load
let isSimpleLoad = 1 in 
def LDR  : AI2ldw<0x0, (outs GPR:$dst), (ins addrmode2:$addr), LdFrm,
               "ldr", " $dst, $addr",
               [(set GPR:$dst, (load addrmode2:$addr))]>;

// Special LDR for loads from non-pc-relative constpools.
let isSimpleLoad = 1, mayLoad = 1, isReMaterializable = 1 in
def LDRcp : AI2ldw<0x0, (outs GPR:$dst), (ins addrmode2:$addr), LdFrm,
                 "ldr", " $dst, $addr", []>;

// Loads with zero extension
def LDRH  : AI3ldh<0xB, (outs GPR:$dst), (ins addrmode3:$addr), LdFrm,
                 "ldr", "h $dst, $addr",
                [(set GPR:$dst, (zextloadi16 addrmode3:$addr))]>;

def LDRB  : AI2ldb<0x1, (outs GPR:$dst), (ins addrmode2:$addr), LdFrm,
                 "ldr", "b $dst, $addr",
                [(set GPR:$dst, (zextloadi8 addrmode2:$addr))]>;

// Loads with sign extension
def LDRSH : AI3ldsh<0xE, (outs GPR:$dst), (ins addrmode3:$addr), LdFrm,
                 "ldr", "sh $dst, $addr",
                [(set GPR:$dst, (sextloadi16 addrmode3:$addr))]>;

def LDRSB : AI3ldsb<0xD, (outs GPR:$dst), (ins addrmode3:$addr), LdFrm,
                 "ldr", "sb $dst, $addr",
                [(set GPR:$dst, (sextloadi8 addrmode3:$addr))]>;

let mayLoad = 1 in {
// Load doubleword
def LDRD  : AI3ldd<0xD, (outs GPR:$dst), (ins addrmode3:$addr), LdFrm,
                 "ldr", "d $dst, $addr",
                []>, Requires<[IsARM, HasV5T]>;

// Indexed loads
def LDR_PRE  : AI2ldwpr<0x0, (outs GPR:$dst, GPR:$base_wb),
                     (ins addrmode2:$addr), LdFrm,
                     "ldr", " $dst, $addr!", "$addr.base = $base_wb", []>;

def LDR_POST : AI2ldwpo<0x0, (outs GPR:$dst, GPR:$base_wb),
                     (ins GPR:$base, am2offset:$offset), LdFrm,
                     "ldr", " $dst, [$base], $offset", "$base = $base_wb", []>;

def LDRH_PRE  : AI3ldhpr<0xB, (outs GPR:$dst, GPR:$base_wb),
                     (ins addrmode3:$addr), LdFrm,
                     "ldr", "h $dst, $addr!", "$addr.base = $base_wb", []>;

def LDRH_POST : AI3ldhpo<0xB, (outs GPR:$dst, GPR:$base_wb),
                     (ins GPR:$base,am3offset:$offset), LdFrm,
                     "ldr", "h $dst, [$base], $offset", "$base = $base_wb", []>;

def LDRB_PRE  : AI2ldbpr<0x1, (outs GPR:$dst, GPR:$base_wb),
                     (ins addrmode2:$addr), LdFrm,
                     "ldr", "b $dst, $addr!", "$addr.base = $base_wb", []>;

def LDRB_POST : AI2ldbpo<0x1, (outs GPR:$dst, GPR:$base_wb),
                     (ins GPR:$base,am2offset:$offset), LdFrm,
                     "ldr", "b $dst, [$base], $offset", "$base = $base_wb", []>;

def LDRSH_PRE : AI3ldshpr<0xE, (outs GPR:$dst, GPR:$base_wb),
                      (ins addrmode3:$addr), LdFrm,
                      "ldr", "sh $dst, $addr!", "$addr.base = $base_wb", []>;

def LDRSH_POST: AI3ldshpo<0xE, (outs GPR:$dst, GPR:$base_wb),
                      (ins GPR:$base,am3offset:$offset), LdFrm,
                      "ldr", "sh $dst, [$base], $offset", "$base = $base_wb", []>;

def LDRSB_PRE : AI3ldsbpr<0xD, (outs GPR:$dst, GPR:$base_wb),
                      (ins addrmode3:$addr), LdFrm,
                      "ldr", "sb $dst, $addr!", "$addr.base = $base_wb", []>;

def LDRSB_POST: AI3ldsbpo<0xD, (outs GPR:$dst, GPR:$base_wb),
                      (ins GPR:$base,am3offset:$offset), LdFrm,
                      "ldr", "sb $dst, [$base], $offset", "$base = $base_wb", []>;
}

// Store
def STR  : AI2stw<0x0, (outs), (ins GPR:$src, addrmode2:$addr), StFrm,
               "str", " $src, $addr",
               [(store GPR:$src, addrmode2:$addr)]>;

// Stores with truncate
def STRH : AI3sth<0xB, (outs), (ins GPR:$src, addrmode3:$addr), StFrm,
               "str", "h $src, $addr",
               [(truncstorei16 GPR:$src, addrmode3:$addr)]>;

def STRB : AI2stb<0x1, (outs), (ins GPR:$src, addrmode2:$addr), StFrm,
               "str", "b $src, $addr",
               [(truncstorei8 GPR:$src, addrmode2:$addr)]>;

// Store doubleword
let mayStore = 1 in
def STRD : AI3std<0xF, (outs), (ins GPR:$src, addrmode3:$addr), StFrm,
               "str", "d $src, $addr",
               []>, Requires<[IsARM, HasV5T]>;

// Indexed stores
def STR_PRE  : AI2stwpr<0x0, (outs GPR:$base_wb),
                     (ins GPR:$src, GPR:$base, am2offset:$offset), StFrm,
                    "str", " $src, [$base, $offset]!", "$base = $base_wb",
                    [(set GPR:$base_wb,
                      (pre_store GPR:$src, GPR:$base, am2offset:$offset))]>;

def STR_POST : AI2stwpo<0x0, (outs GPR:$base_wb),
                     (ins GPR:$src, GPR:$base,am2offset:$offset), StFrm,
                    "str", " $src, [$base], $offset", "$base = $base_wb",
                    [(set GPR:$base_wb,
                      (post_store GPR:$src, GPR:$base, am2offset:$offset))]>;

def STRH_PRE : AI3sthpr<0xB, (outs GPR:$base_wb),
                     (ins GPR:$src, GPR:$base,am3offset:$offset), StFrm,
                     "str", "h $src, [$base, $offset]!", "$base = $base_wb",
                    [(set GPR:$base_wb,
                      (pre_truncsti16 GPR:$src, GPR:$base,am3offset:$offset))]>;

def STRH_POST: AI3sthpo<0xB, (outs GPR:$base_wb),
                     (ins GPR:$src, GPR:$base,am3offset:$offset), StFrm,
                     "str", "h $src, [$base], $offset", "$base = $base_wb",
                    [(set GPR:$base_wb, (post_truncsti16 GPR:$src,
                                         GPR:$base, am3offset:$offset))]>;

def STRB_PRE : AI2stbpr<0x1, (outs GPR:$base_wb),
                     (ins GPR:$src, GPR:$base,am2offset:$offset), StFrm,
                     "str", "b $src, [$base, $offset]!", "$base = $base_wb",
                    [(set GPR:$base_wb, (pre_truncsti8 GPR:$src,
                                         GPR:$base, am2offset:$offset))]>;

def STRB_POST: AI2stbpo<0x1, (outs GPR:$base_wb),
                     (ins GPR:$src, GPR:$base,am2offset:$offset), StFrm,
                     "str", "b $src, [$base], $offset", "$base = $base_wb",
                    [(set GPR:$base_wb, (post_truncsti8 GPR:$src,
                                         GPR:$base, am2offset:$offset))]>;

//===----------------------------------------------------------------------===//
//  Load / store multiple Instructions.
//

// FIXME: $dst1 should be a def.
let mayLoad = 1 in
def LDM : AXI4ld<0x0, (outs),
               (ins addrmode4:$addr, pred:$p, reglist:$dst1, variable_ops),
               LdFrm, "ldm${p}${addr:submode} $addr, $dst1",
               []>;

let mayStore = 1 in
def STM : AXI4st<0x0, (outs),
               (ins addrmode4:$addr, pred:$p, reglist:$src1, variable_ops),
               StFrm, "stm${p}${addr:submode} $addr, $src1",
               []>;

//===----------------------------------------------------------------------===//
//  Move Instructions.
//

def MOVr : AsI1<{1,0,1,1}, (outs GPR:$dst), (ins GPR:$src), DPRdReg,
                 "mov", " $dst, $src", []>;
def MOVs : AsI1<{1,0,1,1}, (outs GPR:$dst), (ins so_reg:$src), DPRdSoReg,
                 "mov", " $dst, $src", [(set GPR:$dst, so_reg:$src)]>;

let isReMaterializable = 1 in
def MOVi : AsI1<{1,0,1,1}, (outs GPR:$dst), (ins so_imm:$src), DPRdIm,
                 "mov", " $dst, $src", [(set GPR:$dst, so_imm:$src)]>;

def MOVrx : AsI1<{1,0,1,1}, (outs GPR:$dst), (ins GPR:$src), DPRdMisc,
                 "mov", " $dst, $src, rrx",
                 [(set GPR:$dst, (ARMrrx GPR:$src))]>;

// These aren't really mov instructions, but we have to define them this way
// due to flag operands.

let Defs = [CPSR] in {
def MOVsrl_flag : AI1<{1,0,1,1}, (outs GPR:$dst), (ins GPR:$src), DPRdMisc,
                      "mov", "s $dst, $src, lsr #1",
                      [(set GPR:$dst, (ARMsrl_flag GPR:$src))]>;
def MOVsra_flag : AI1<{1,0,1,1}, (outs GPR:$dst), (ins GPR:$src), DPRdMisc,
                      "mov", "s $dst, $src, asr #1",
                      [(set GPR:$dst, (ARMsra_flag GPR:$src))]>;
}

//===----------------------------------------------------------------------===//
//  Extend Instructions.
//

// Sign extenders

defm SXTB  : AI_unary_rrot<0x0, "sxtb", UnOpFrag<(sext_inreg node:$Src, i8)>>;
defm SXTH  : AI_unary_rrot<0x0, "sxth", UnOpFrag<(sext_inreg node:$Src, i16)>>;

defm SXTAB : AI_bin_rrot<0x0, "sxtab",
                        BinOpFrag<(add node:$LHS, (sext_inreg node:$RHS, i8))>>;
defm SXTAH : AI_bin_rrot<0x0, "sxtah",
                        BinOpFrag<(add node:$LHS, (sext_inreg node:$RHS,i16))>>;

// TODO: SXT(A){B|H}16

// Zero extenders

let AddedComplexity = 16 in {
defm UXTB   : AI_unary_rrot<0x0, "uxtb"  , UnOpFrag<(and node:$Src, 0x000000FF)>>;
defm UXTH   : AI_unary_rrot<0x0, "uxth"  , UnOpFrag<(and node:$Src, 0x0000FFFF)>>;
defm UXTB16 : AI_unary_rrot<0x0, "uxtb16", UnOpFrag<(and node:$Src, 0x00FF00FF)>>;

def : ARMV6Pat<(and (shl GPR:$Src, 8), 0xFF00FF),
               (UXTB16r_rot GPR:$Src, 24)>;
def : ARMV6Pat<(and (srl GPR:$Src, 8), 0xFF00FF),
               (UXTB16r_rot GPR:$Src, 8)>;

defm UXTAB : AI_bin_rrot<0x0, "uxtab",
                        BinOpFrag<(add node:$LHS, (and node:$RHS, 0x00FF))>>;
defm UXTAH : AI_bin_rrot<0x0, "uxtah",
                        BinOpFrag<(add node:$LHS, (and node:$RHS, 0xFFFF))>>;
}

// This isn't safe in general, the add is two 16-bit units, not a 32-bit add.
//defm UXTAB16 : xxx<"uxtab16", 0xff00ff>;

// TODO: UXT(A){B|H}16

//===----------------------------------------------------------------------===//
//  Arithmetic Instructions.
//

defm ADD  : AsI1_bin_irs<{0,0,1,0}, "add",
                         BinOpFrag<(add  node:$LHS, node:$RHS)>>;
defm SUB  : AsI1_bin_irs<{0,1,0,0}, "sub",
                         BinOpFrag<(sub  node:$LHS, node:$RHS)>>;

// ADD and SUB with 's' bit set.
defm ADDS : ASI1_bin_s_irs<{0,0,1,0}, "add",
                           BinOpFrag<(addc node:$LHS, node:$RHS)>>;
defm SUBS : ASI1_bin_s_irs<{0,1,0,0}, "sub",
                           BinOpFrag<(subc node:$LHS, node:$RHS)>>;

// FIXME: Do not allow ADC / SBC to be predicated for now.
defm ADC  : AsXI1_bin_c_irs<{1,0,1,0}, "adc",
                            BinOpFrag<(adde node:$LHS, node:$RHS)>>;
defm SBC  : AsXI1_bin_c_irs<{0,1,1,0}, "sbc",
                            BinOpFrag<(sube node:$LHS, node:$RHS)>>;

// These don't define reg/reg forms, because they are handled above.
def RSBri : AsI1<{1,1,0,0}, (outs GPR:$dst), (ins GPR:$a, so_imm:$b), DPRIm,
                  "rsb", " $dst, $a, $b",
                  [(set GPR:$dst, (sub so_imm:$b, GPR:$a))]>;

def RSBrs : AsI1<{1,1,0,0}, (outs GPR:$dst), (ins GPR:$a, so_reg:$b), DPRSoReg,
                  "rsb", " $dst, $a, $b",
                  [(set GPR:$dst, (sub so_reg:$b, GPR:$a))]>;

// RSB with 's' bit set.
let Defs = [CPSR] in {
def RSBSri : AI1<{1,1,0,0}, (outs GPR:$dst), (ins GPR:$a, so_imm:$b), DPRIm,
                 "rsb", "s $dst, $a, $b",
                 [(set GPR:$dst, (subc so_imm:$b, GPR:$a))]>;
def RSBSrs : AI1<{1,1,0,0}, (outs GPR:$dst), (ins GPR:$a, so_reg:$b), DPRSoReg,
                 "rsb", "s $dst, $a, $b",
                 [(set GPR:$dst, (subc so_reg:$b, GPR:$a))]>;
}

// FIXME: Do not allow RSC to be predicated for now. But they can set CPSR.
let Uses = [CPSR] in {
def RSCri : AXI1<{1,1,1,0}, (outs GPR:$dst), (ins GPR:$a, so_imm:$b, cc_out:$s),
                 DPRIm, "rsc${s} $dst, $a, $b",
                 [(set GPR:$dst, (sube so_imm:$b, GPR:$a))]>;
def RSCrs : AXI1<{1,1,1,0}, (outs GPR:$dst), (ins GPR:$a, so_reg:$b, cc_out:$s),
                 DPRSoReg, "rsc${s} $dst, $a, $b",
                 [(set GPR:$dst, (sube so_reg:$b, GPR:$a))]>;
}

// (sub X, imm) gets canonicalized to (add X, -imm).  Match this form.
def : ARMPat<(add    GPR:$src, so_imm_neg:$imm),
             (SUBri  GPR:$src, so_imm_neg:$imm)>;

//def : ARMPat<(addc   GPR:$src, so_imm_neg:$imm),
//             (SUBSri GPR:$src, so_imm_neg:$imm)>;
//def : ARMPat<(adde   GPR:$src, so_imm_neg:$imm),
//             (SBCri  GPR:$src, so_imm_neg:$imm)>;

// Note: These are implemented in C++ code, because they have to generate
// ADD/SUBrs instructions, which use a complex pattern that a xform function
// cannot produce.
// (mul X, 2^n+1) -> (add (X << n), X)
// (mul X, 2^n-1) -> (rsb X, (X << n))


//===----------------------------------------------------------------------===//
//  Bitwise Instructions.
//

defm AND   : AsI1_bin_irs<{0,0,0,0}, "and",
                          BinOpFrag<(and node:$LHS, node:$RHS)>>;
defm ORR   : AsI1_bin_irs<{0,0,1,1}, "orr",
                          BinOpFrag<(or  node:$LHS, node:$RHS)>>;
defm EOR   : AsI1_bin_irs<{1,0,0,0}, "eor",
                          BinOpFrag<(xor node:$LHS, node:$RHS)>>;
defm BIC   : AsI1_bin_irs<{0,1,1,1}, "bic",
                          BinOpFrag<(and node:$LHS, (not node:$RHS))>>;

def  MVNr  : AsI1<{1,1,1,1}, (outs GPR:$dst), (ins GPR:$src), DPRdReg,
                  "mvn", " $dst, $src", [(set GPR:$dst, (not GPR:$src))]>;
def  MVNs  : AsI1<{1,1,1,1}, (outs GPR:$dst), (ins so_reg:$src), DPRdSoReg,
                  "mvn", " $dst, $src", [(set GPR:$dst, (not so_reg:$src))]>;
let isReMaterializable = 1 in
def  MVNi  : AsI1<{1,1,1,1}, (outs GPR:$dst), (ins so_imm:$imm), DPRdIm,
                  "mvn", " $dst, $imm", [(set GPR:$dst, so_imm_not:$imm)]>;

def : ARMPat<(and   GPR:$src, so_imm_not:$imm),
             (BICri GPR:$src, so_imm_not:$imm)>;

//===----------------------------------------------------------------------===//
//  Multiply Instructions.
//

def MUL  : AsI<0x0, (outs GPR:$dst), (ins GPR:$a, GPR:$b), MulFrm,
               "mul", " $dst, $a, $b",
               [(set GPR:$dst, (mul GPR:$a, GPR:$b))]>;

def MLA  : AsI<0x2, (outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$c),
               MulFrm, "mla", " $dst, $a, $b, $c",
               [(set GPR:$dst, (add (mul GPR:$a, GPR:$b), GPR:$c))]>;

// Extra precision multiplies with low / high results
def SMULL : AsI<0xC, (outs GPR:$ldst, GPR:$hdst), (ins GPR:$a, GPR:$b),
                MulFrm, "smull", " $ldst, $hdst, $a, $b", []>;

def UMULL : AsI<0x8, (outs GPR:$ldst, GPR:$hdst), (ins GPR:$a, GPR:$b),
                MulFrm, "umull", " $ldst, $hdst, $a, $b", []>;

// Multiply + accumulate
def SMLAL : AsI<0xE, (outs GPR:$ldst, GPR:$hdst), (ins GPR:$a, GPR:$b),
                MulFrm, "smlal", " $ldst, $hdst, $a, $b", []>;

def UMLAL : AsI<0xA, (outs GPR:$ldst, GPR:$hdst), (ins GPR:$a, GPR:$b),
                MulFrm, "umlal", " $ldst, $hdst, $a, $b", []>;

def UMAAL : AI<0x0, (outs GPR:$ldst, GPR:$hdst), (ins GPR:$a, GPR:$b), MulFrm,
               "umaal", " $ldst, $hdst, $a, $b", []>,
            Requires<[IsARM, HasV6]>;

// Most significant word multiply
def SMMUL : AI<0x0, (outs GPR:$dst), (ins GPR:$a, GPR:$b), MulFrm,
               "smmul", " $dst, $a, $b",
               [(set GPR:$dst, (mulhs GPR:$a, GPR:$b))]>,
            Requires<[IsARM, HasV6]>;

def SMMLA : AI<0x0, (outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$c), MulFrm,
               "smmla", " $dst, $a, $b, $c",
               [(set GPR:$dst, (add (mulhs GPR:$a, GPR:$b), GPR:$c))]>,
            Requires<[IsARM, HasV6]>;


def SMMLS : AI<0x0, (outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$c), MulFrm,
               "smmls", " $dst, $a, $b, $c",
               [(set GPR:$dst, (sub GPR:$c, (mulhs GPR:$a, GPR:$b)))]>,
               Requires<[IsARM, HasV6]>;

multiclass AI_smul<string opc, PatFrag opnode> {
  def BB : AI<0x8, (outs GPR:$dst), (ins GPR:$a, GPR:$b), MulSMUL,
              !strconcat(opc, "bb"), " $dst, $a, $b",
              [(set GPR:$dst, (opnode (sext_inreg GPR:$a, i16),
                                      (sext_inreg GPR:$b, i16)))]>,
           Requires<[IsARM, HasV5TE]>;

  def BT : AI<0xC, (outs GPR:$dst), (ins GPR:$a, GPR:$b), MulSMUL,
              !strconcat(opc, "bt"), " $dst, $a, $b",
              [(set GPR:$dst, (opnode (sext_inreg GPR:$a, i16),
                                      (sra GPR:$b, 16)))]>,
           Requires<[IsARM, HasV5TE]>;

  def TB : AI<0xA, (outs GPR:$dst), (ins GPR:$a, GPR:$b), MulSMUL,
              !strconcat(opc, "tb"), " $dst, $a, $b",
              [(set GPR:$dst, (opnode (sra GPR:$a, 16),
                                      (sext_inreg GPR:$b, i16)))]>,
           Requires<[IsARM, HasV5TE]>;

  def TT : AI<0xE, (outs GPR:$dst), (ins GPR:$a, GPR:$b), MulSMUL,
              !strconcat(opc, "tt"), " $dst, $a, $b",
              [(set GPR:$dst, (opnode (sra GPR:$a, 16),
                                      (sra GPR:$b, 16)))]>,
            Requires<[IsARM, HasV5TE]>;

  def WB : AI<0xA, (outs GPR:$dst), (ins GPR:$a, GPR:$b), MulSMULW,
              !strconcat(opc, "wb"), " $dst, $a, $b",
              [(set GPR:$dst, (sra (opnode GPR:$a,
                                    (sext_inreg GPR:$b, i16)), 16))]>,
           Requires<[IsARM, HasV5TE]>;

  def WT : AI<0xE, (outs GPR:$dst), (ins GPR:$a, GPR:$b), MulSMULW,
              !strconcat(opc, "wt"), " $dst, $a, $b",
              [(set GPR:$dst, (sra (opnode GPR:$a,
                                    (sra GPR:$b, 16)), 16))]>,
            Requires<[IsARM, HasV5TE]>;
}


multiclass AI_smla<string opc, PatFrag opnode> {
  def BB : AI<0x8, (outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$acc), MulSMLA,
              !strconcat(opc, "bb"), " $dst, $a, $b, $acc",
              [(set GPR:$dst, (add GPR:$acc,
                               (opnode (sext_inreg GPR:$a, i16),
                                       (sext_inreg GPR:$b, i16))))]>,
           Requires<[IsARM, HasV5TE]>;

  def BT : AI<0xC, (outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$acc), MulSMLA,
              !strconcat(opc, "bt"), " $dst, $a, $b, $acc",
              [(set GPR:$dst, (add GPR:$acc, (opnode (sext_inreg GPR:$a, i16),
                                                     (sra GPR:$b, 16))))]>,
           Requires<[IsARM, HasV5TE]>;

  def TB : AI<0xA, (outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$acc), MulSMLA,
              !strconcat(opc, "tb"), " $dst, $a, $b, $acc",
              [(set GPR:$dst, (add GPR:$acc, (opnode (sra GPR:$a, 16),
                                                 (sext_inreg GPR:$b, i16))))]>,
           Requires<[IsARM, HasV5TE]>;

  def TT : AI<0xE, (outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$acc), MulSMLA,
              !strconcat(opc, "tt"), " $dst, $a, $b, $acc",
              [(set GPR:$dst, (add GPR:$acc, (opnode (sra GPR:$a, 16),
                                                     (sra GPR:$b, 16))))]>,
            Requires<[IsARM, HasV5TE]>;

  def WB : AI<0xA, (outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$acc), MulSMLAW,
              !strconcat(opc, "wb"), " $dst, $a, $b, $acc",
              [(set GPR:$dst, (add GPR:$acc, (sra (opnode GPR:$a,
                                            (sext_inreg GPR:$b, i16)), 16)))]>,
           Requires<[IsARM, HasV5TE]>;

  def WT : AI<0xE, (outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$acc), MulSMLAW,
              !strconcat(opc, "wt"), " $dst, $a, $b, $acc",
              [(set GPR:$dst, (add GPR:$acc, (sra (opnode GPR:$a,
                                                   (sra GPR:$b, 16)), 16)))]>,
            Requires<[IsARM, HasV5TE]>;
}

defm SMUL : AI_smul<"smul", BinOpFrag<(mul node:$LHS, node:$RHS)>>;
defm SMLA : AI_smla<"smla", BinOpFrag<(mul node:$LHS, node:$RHS)>>;

// TODO: Halfword multiple accumulate long: SMLAL<x><y>
// TODO: Dual halfword multiple: SMUAD, SMUSD, SMLAD, SMLSD, SMLALD, SMLSLD

//===----------------------------------------------------------------------===//
//  Misc. Arithmetic Instructions.
//

def CLZ  : AI<0x0, (outs GPR:$dst), (ins GPR:$src), ArithMisc,
              "clz", " $dst, $src",
              [(set GPR:$dst, (ctlz GPR:$src))]>, Requires<[IsARM, HasV5T]>;

def REV  : AI<0x0, (outs GPR:$dst), (ins GPR:$src), ArithMisc,
              "rev", " $dst, $src",
              [(set GPR:$dst, (bswap GPR:$src))]>, Requires<[IsARM, HasV6]>;

def REV16 : AI<0x0, (outs GPR:$dst), (ins GPR:$src), ArithMisc,
               "rev16", " $dst, $src",
               [(set GPR:$dst,
                   (or (and (srl GPR:$src, 8), 0xFF),
                       (or (and (shl GPR:$src, 8), 0xFF00),
                           (or (and (srl GPR:$src, 8), 0xFF0000),
                               (and (shl GPR:$src, 8), 0xFF000000)))))]>,
               Requires<[IsARM, HasV6]>;

def REVSH : AI<0x0, (outs GPR:$dst), (ins GPR:$src), ArithMisc,
               "revsh", " $dst, $src",
               [(set GPR:$dst,
                  (sext_inreg
                    (or (srl (and GPR:$src, 0xFF00), 8),
                        (shl GPR:$src, 8)), i16))]>,
               Requires<[IsARM, HasV6]>;

def PKHBT : AI<0x0, (outs GPR:$dst), (ins GPR:$src1, GPR:$src2, i32imm:$shamt),
               Pseudo, "pkhbt", " $dst, $src1, $src2, LSL $shamt",
               [(set GPR:$dst, (or (and GPR:$src1, 0xFFFF),
                                   (and (shl GPR:$src2, (i32 imm:$shamt)),
                                        0xFFFF0000)))]>,
               Requires<[IsARM, HasV6]>;

// Alternate cases for PKHBT where identities eliminate some nodes.
def : ARMV6Pat<(or (and GPR:$src1, 0xFFFF), (and GPR:$src2, 0xFFFF0000)),
               (PKHBT GPR:$src1, GPR:$src2, 0)>;
def : ARMV6Pat<(or (and GPR:$src1, 0xFFFF), (shl GPR:$src2, imm16_31:$shamt)),
               (PKHBT GPR:$src1, GPR:$src2, imm16_31:$shamt)>;


def PKHTB : AI<0x0, (outs GPR:$dst), (ins GPR:$src1, GPR:$src2, i32imm:$shamt),
               Pseudo, "pkhtb", " $dst, $src1, $src2, ASR $shamt",
               [(set GPR:$dst, (or (and GPR:$src1, 0xFFFF0000),
                                   (and (sra GPR:$src2, imm16_31:$shamt),
                                        0xFFFF)))]>, Requires<[IsARM, HasV6]>;

// Alternate cases for PKHTB where identities eliminate some nodes.  Note that
// a shift amount of 0 is *not legal* here, it is PKHBT instead.
def : ARMV6Pat<(or (and GPR:$src1, 0xFFFF0000), (srl GPR:$src2, 16)),
               (PKHTB GPR:$src1, GPR:$src2, 16)>;
def : ARMV6Pat<(or (and GPR:$src1, 0xFFFF0000),
                   (and (srl GPR:$src2, imm1_15:$shamt), 0xFFFF)),
               (PKHTB GPR:$src1, GPR:$src2, imm1_15:$shamt)>;


//===----------------------------------------------------------------------===//
//  Comparison Instructions...
//

defm CMP  : AI1_cmp_irs<0xA, "cmp",
                        BinOpFrag<(ARMcmp node:$LHS, node:$RHS)>>;
defm CMN  : AI1_cmp_irs<0xB, "cmn",
                        BinOpFrag<(ARMcmp node:$LHS,(ineg node:$RHS))>>;

// Note that TST/TEQ don't set all the same flags that CMP does!
defm TST  : AI1_cmp_irs<0x8, "tst",
                        BinOpFrag<(ARMcmpNZ (and node:$LHS, node:$RHS), 0)>>;
defm TEQ  : AI1_cmp_irs<0x9, "teq",
                        BinOpFrag<(ARMcmpNZ (xor node:$LHS, node:$RHS), 0)>>;

defm CMPnz : AI1_cmp_irs<0xA, "cmp",
                         BinOpFrag<(ARMcmpNZ node:$LHS, node:$RHS)>>;
defm CMNnz : AI1_cmp_irs<0xA, "cmn",
                         BinOpFrag<(ARMcmpNZ node:$LHS,(ineg node:$RHS))>>;

def : ARMPat<(ARMcmp GPR:$src, so_imm_neg:$imm),
             (CMNri  GPR:$src, so_imm_neg:$imm)>;

def : ARMPat<(ARMcmpNZ GPR:$src, so_imm_neg:$imm),
             (CMNri  GPR:$src, so_imm_neg:$imm)>;


// Conditional moves
// FIXME: should be able to write a pattern for ARMcmov, but can't use
// a two-value operand where a dag node expects two operands. :( 
def MOVCCr : AI<0xD, (outs GPR:$dst), (ins GPR:$false, GPR:$true),
                DPRdReg, "mov", " $dst, $true",
      [/*(set GPR:$dst, (ARMcmov GPR:$false, GPR:$true, imm:$cc, CCR:$ccr))*/]>,
                RegConstraint<"$false = $dst">;

def MOVCCs : AI<0xD, (outs GPR:$dst), (ins GPR:$false, so_reg:$true),
                DPRdSoReg, "mov", " $dst, $true",
   [/*(set GPR:$dst, (ARMcmov GPR:$false, so_reg:$true, imm:$cc, CCR:$ccr))*/]>,
                RegConstraint<"$false = $dst">;

def MOVCCi : AI<0xD, (outs GPR:$dst), (ins GPR:$false, so_imm:$true),
                DPRdIm, "mov", " $dst, $true",
   [/*(set GPR:$dst, (ARMcmov GPR:$false, so_imm:$true, imm:$cc, CCR:$ccr))*/]>,
                RegConstraint<"$false = $dst">;


// LEApcrel - Load a pc-relative address into a register without offending the
// assembler.
def LEApcrel : AXI1<0x0, (outs GPR:$dst), (ins i32imm:$label, pred:$p), Pseudo,
                   !strconcat(!strconcat(".set PCRELV${:uid}, ($label-(",
                                         "${:private}PCRELL${:uid}+8))\n"),
                              !strconcat("${:private}PCRELL${:uid}:\n\t",
                                         "add$p $dst, pc, #PCRELV${:uid}")),
                   []>;

def LEApcrelJT : AXI1<0x0, (outs GPR:$dst), (ins i32imm:$label, i32imm:$id, pred:$p),
          Pseudo,
          !strconcat(!strconcat(".set PCRELV${:uid}, (${label}_${id:no_hash}-(",
                                         "${:private}PCRELL${:uid}+8))\n"),
                              !strconcat("${:private}PCRELL${:uid}:\n\t",
                                         "add$p $dst, pc, #PCRELV${:uid}")),
                   []>;

//===----------------------------------------------------------------------===//
// TLS Instructions
//

// __aeabi_read_tp preserves the registers r1-r3.
let isCall = 1,
  Defs = [R0, R12, LR, CPSR] in {
  def TPsoft : AXI<0x0, (outs), (ins), BranchMisc,
               "bl __aeabi_read_tp",
               [(set R0, ARMthread_pointer)]>;
}

//===----------------------------------------------------------------------===//
// Non-Instruction Patterns
//

// ConstantPool, GlobalAddress, and JumpTable
def : ARMPat<(ARMWrapper  tglobaladdr :$dst), (LEApcrel tglobaladdr :$dst)>;
def : ARMPat<(ARMWrapper  tconstpool  :$dst), (LEApcrel tconstpool  :$dst)>;
def : ARMPat<(ARMWrapperJT tjumptable:$dst, imm:$id),
             (LEApcrelJT tjumptable:$dst, imm:$id)>;

// Large immediate handling.

// Two piece so_imms.
let isReMaterializable = 1 in
def MOVi2pieces : AI1x2<0x0, (outs GPR:$dst), (ins so_imm2part:$src), DPRdMisc,
                         "mov", " $dst, $src",
                         [(set GPR:$dst, so_imm2part:$src)]>;

def : ARMPat<(or GPR:$LHS, so_imm2part:$RHS),
              (ORRri (ORRri GPR:$LHS, (so_imm2part_1 imm:$RHS)),
                     (so_imm2part_2 imm:$RHS))>;
def : ARMPat<(xor GPR:$LHS, so_imm2part:$RHS),
              (EORri (EORri GPR:$LHS, (so_imm2part_1 imm:$RHS)),
                     (so_imm2part_2 imm:$RHS))>;

// TODO: add,sub,and, 3-instr forms?


// Direct calls
def : ARMPat<(ARMcall texternalsym:$func), (BL texternalsym:$func)>;

// zextload i1 -> zextload i8
def : ARMPat<(zextloadi1 addrmode2:$addr),  (LDRB addrmode2:$addr)>;

// extload -> zextload
def : ARMPat<(extloadi1  addrmode2:$addr),  (LDRB addrmode2:$addr)>;
def : ARMPat<(extloadi8  addrmode2:$addr),  (LDRB addrmode2:$addr)>;
def : ARMPat<(extloadi16 addrmode3:$addr),  (LDRH addrmode3:$addr)>;

// smul* and smla*
def : ARMV5TEPat<(mul (sra (shl GPR:$a, 16), 16), (sra (shl GPR:$b, 16), 16)),
                 (SMULBB GPR:$a, GPR:$b)>;
def : ARMV5TEPat<(mul sext_16_node:$a, sext_16_node:$b),
                 (SMULBB GPR:$a, GPR:$b)>;
def : ARMV5TEPat<(mul (sra (shl GPR:$a, 16), 16), (sra GPR:$b, 16)),
                 (SMULBT GPR:$a, GPR:$b)>;
def : ARMV5TEPat<(mul sext_16_node:$a, (sra GPR:$b, 16)),
                 (SMULBT GPR:$a, GPR:$b)>;
def : ARMV5TEPat<(mul (sra GPR:$a, 16), (sra (shl GPR:$b, 16), 16)),
                 (SMULTB GPR:$a, GPR:$b)>;
def : ARMV5TEPat<(mul (sra GPR:$a, 16), sext_16_node:$b),
                (SMULTB GPR:$a, GPR:$b)>;
def : ARMV5TEPat<(sra (mul GPR:$a, (sra (shl GPR:$b, 16), 16)), 16),
                 (SMULWB GPR:$a, GPR:$b)>;
def : ARMV5TEPat<(sra (mul GPR:$a, sext_16_node:$b), 16),
                 (SMULWB GPR:$a, GPR:$b)>;

def : ARMV5TEPat<(add GPR:$acc,
                      (mul (sra (shl GPR:$a, 16), 16),
                           (sra (shl GPR:$b, 16), 16))),
                 (SMLABB GPR:$a, GPR:$b, GPR:$acc)>;
def : ARMV5TEPat<(add GPR:$acc,
                      (mul sext_16_node:$a, sext_16_node:$b)),
                 (SMLABB GPR:$a, GPR:$b, GPR:$acc)>;
def : ARMV5TEPat<(add GPR:$acc,
                      (mul (sra (shl GPR:$a, 16), 16), (sra GPR:$b, 16))),
                 (SMLABT GPR:$a, GPR:$b, GPR:$acc)>;
def : ARMV5TEPat<(add GPR:$acc,
                      (mul sext_16_node:$a, (sra GPR:$b, 16))),
                 (SMLABT GPR:$a, GPR:$b, GPR:$acc)>;
def : ARMV5TEPat<(add GPR:$acc,
                      (mul (sra GPR:$a, 16), (sra (shl GPR:$b, 16), 16))),
                 (SMLATB GPR:$a, GPR:$b, GPR:$acc)>;
def : ARMV5TEPat<(add GPR:$acc,
                      (mul (sra GPR:$a, 16), sext_16_node:$b)),
                 (SMLATB GPR:$a, GPR:$b, GPR:$acc)>;
def : ARMV5TEPat<(add GPR:$acc,
                      (sra (mul GPR:$a, (sra (shl GPR:$b, 16), 16)), 16)),
                 (SMLAWB GPR:$a, GPR:$b, GPR:$acc)>;
def : ARMV5TEPat<(add GPR:$acc,
                      (sra (mul GPR:$a, sext_16_node:$b), 16)),
                 (SMLAWB GPR:$a, GPR:$b, GPR:$acc)>;

//===----------------------------------------------------------------------===//
// Thumb Support
//

include "ARMInstrThumb.td"

//===----------------------------------------------------------------------===//
// Floating Point Support
//

include "ARMInstrVFP.td"
