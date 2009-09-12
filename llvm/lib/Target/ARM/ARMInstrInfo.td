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

def SDT_ARMBr2JT   : SDTypeProfile<0, 4,
                                  [SDTCisPtrTy<0>, SDTCisVT<1, i32>,
                                   SDTCisVT<2, i32>, SDTCisVT<3, i32>]>;

def SDT_ARMCmp     : SDTypeProfile<0, 2, [SDTCisSameAs<0, 1>]>;

def SDT_ARMPICAdd  : SDTypeProfile<1, 2, [SDTCisSameAs<0, 1>,
                                          SDTCisPtrTy<1>, SDTCisVT<2, i32>]>;

def SDT_ARMThreadPointer : SDTypeProfile<1, 0, [SDTCisPtrTy<0>]>;
def SDT_ARMEH_SJLJ_Setjmp : SDTypeProfile<1, 1, [SDTCisInt<0>, SDTCisPtrTy<1>]>;

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
def ARMbr2jt         : SDNode<"ARMISD::BR2_JT", SDT_ARMBr2JT,
                              [SDNPHasChain]>;

def ARMcmp           : SDNode<"ARMISD::CMP", SDT_ARMCmp,
                              [SDNPOutFlag]>;

def ARMcmpZ          : SDNode<"ARMISD::CMPZ", SDT_ARMCmp,
                              [SDNPOutFlag,SDNPCommutative]>;

def ARMpic_add       : SDNode<"ARMISD::PIC_ADD", SDT_ARMPICAdd>;

def ARMsrl_flag      : SDNode<"ARMISD::SRL_FLAG", SDTIntUnaryOp, [SDNPOutFlag]>;
def ARMsra_flag      : SDNode<"ARMISD::SRA_FLAG", SDTIntUnaryOp, [SDNPOutFlag]>;
def ARMrrx           : SDNode<"ARMISD::RRX"     , SDTIntUnaryOp, [SDNPInFlag ]>;

def ARMthread_pointer: SDNode<"ARMISD::THREAD_POINTER", SDT_ARMThreadPointer>;
def ARMeh_sjlj_setjmp: SDNode<"ARMISD::EH_SJLJ_SETJMP", SDT_ARMEH_SJLJ_Setjmp>;

//===----------------------------------------------------------------------===//
// ARM Instruction Predicate Definitions.
//
def HasV5T    : Predicate<"Subtarget->hasV5TOps()">;
def HasV5TE   : Predicate<"Subtarget->hasV5TEOps()">;
def HasV6     : Predicate<"Subtarget->hasV6Ops()">;
def HasV6T2   : Predicate<"Subtarget->hasV6T2Ops()">;
def HasV7     : Predicate<"Subtarget->hasV7Ops()">;
def HasVFP2   : Predicate<"Subtarget->hasVFP2()">;
def HasVFP3   : Predicate<"Subtarget->hasVFP3()">;
def HasNEON   : Predicate<"Subtarget->hasNEON()">;
def UseNEONForFP : Predicate<"Subtarget->useNEONForSinglePrecisionFP()">;
def DontUseNEONForFP : Predicate<"!Subtarget->useNEONForSinglePrecisionFP()">;
def IsThumb   : Predicate<"Subtarget->isThumb()">;
def IsThumb1Only : Predicate<"Subtarget->isThumb1Only()">;
def IsThumb2  : Predicate<"Subtarget->isThumb2()">;
def IsARM     : Predicate<"!Subtarget->isThumb()">;
def IsDarwin    : Predicate<"Subtarget->isTargetDarwin()">;
def IsNotDarwin : Predicate<"!Subtarget->isTargetDarwin()">;
def CarryDefIsUnused : Predicate<"!N.getNode()->hasAnyUseOfValue(1)">;
def CarryDefIsUsed   : Predicate<"N.getNode()->hasAnyUseOfValue(1)">;

//===----------------------------------------------------------------------===//
// ARM Flag Definitions.

class RegConstraint<string C> {
  string Constraints = C;
}

//===----------------------------------------------------------------------===//
//  ARM specific transformation functions and pattern fragments.
//

// so_imm_neg_XFORM - Return a so_imm value packed into the format described for
// so_imm_neg def below.
def so_imm_neg_XFORM : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(-(int)N->getZExtValue(), MVT::i32);
}]>;

// so_imm_not_XFORM - Return a so_imm value packed into the format described for
// so_imm_not def below.
def so_imm_not_XFORM : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(~(int)N->getZExtValue(), MVT::i32);
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

/// bf_inv_mask_imm predicate - An AND mask to clear an arbitrary width bitfield
/// e.g., 0xf000ffff
def bf_inv_mask_imm : Operand<i32>,
                      PatLeaf<(imm), [{ 
  uint32_t v = (uint32_t)N->getZExtValue();
  if (v == 0xffffffff)
    return 0;
  // there can be 1's on either or both "outsides", all the "inside"
  // bits must be 0's
  unsigned int lsb = 0, msb = 31;
  while (v & (1 << msb)) --msb;
  while (v & (1 << lsb)) ++lsb;
  for (unsigned int i = lsb; i <= msb; ++i) {
    if (v & (1 << i))
      return 0;
  }
  return 1;
}] > {
  let PrintMethod = "printBitfieldInvMaskImmOperand";
}

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
def jt2block_operand : Operand<i32> {
  let PrintMethod = "printJT2BlockOperand";
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
             PatLeaf<(imm), [{
      return ARM_AM::getSOImmVal(N->getZExtValue()) != -1;
    }]> {
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
  return CurDAG->getTargetConstant(V, MVT::i32);
}]>;

def so_imm2part_2 : SDNodeXForm<imm, [{
  unsigned V = ARM_AM::getSOImmTwoPartSecond((unsigned)N->getZExtValue());
  return CurDAG->getTargetConstant(V, MVT::i32);
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
                ComplexPattern<i32, 2, "SelectAddrMode4", []> {
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

// addrmode6 := reg with optional writeback
//
def addrmode6 : Operand<i32>,
                ComplexPattern<i32, 3, "SelectAddrMode6", []> {
  let PrintMethod = "printAddrMode6Operand";
  let MIOperandInfo = (ops GPR:$addr, GPR:$upd, i32imm);
}

// addrmodepc := pc + reg
//
def addrmodepc : Operand<i32>,
                 ComplexPattern<i32, 2, "SelectAddrModePC", []> {
  let PrintMethod = "printAddrModePCOperand";
  let MIOperandInfo = (ops GPR, i32imm);
}

def nohash_imm : Operand<i32> {
  let PrintMethod = "printNoHashImmediate";
}

//===----------------------------------------------------------------------===//

include "ARMInstrFormats.td"

//===----------------------------------------------------------------------===//
// Multiclass helpers...
//

/// AsI1_bin_irs - Defines a set of (op r, {so_imm|r|so_reg}) patterns for a
/// binop that produces a value.
multiclass AsI1_bin_irs<bits<4> opcod, string opc, PatFrag opnode,
                        bit Commutable = 0> {
  def ri : AsI1<opcod, (outs GPR:$dst), (ins GPR:$a, so_imm:$b), DPFrm,
               IIC_iALUi, opc, " $dst, $a, $b",
               [(set GPR:$dst, (opnode GPR:$a, so_imm:$b))]> {
    let Inst{25} = 1;
  }
  def rr : AsI1<opcod, (outs GPR:$dst), (ins GPR:$a, GPR:$b), DPFrm,
               IIC_iALUr, opc, " $dst, $a, $b",
               [(set GPR:$dst, (opnode GPR:$a, GPR:$b))]> {
    let Inst{25} = 0;
    let isCommutable = Commutable;
  }
  def rs : AsI1<opcod, (outs GPR:$dst), (ins GPR:$a, so_reg:$b), DPSoRegFrm,
               IIC_iALUsr, opc, " $dst, $a, $b",
               [(set GPR:$dst, (opnode GPR:$a, so_reg:$b))]> {
    let Inst{25} = 0;
  }
}

/// AI1_bin_s_irs - Similar to AsI1_bin_irs except it sets the 's' bit so the
/// instruction modifies the CSPR register.
let Defs = [CPSR] in {
multiclass AI1_bin_s_irs<bits<4> opcod, string opc, PatFrag opnode,
                         bit Commutable = 0> {
  def ri : AI1<opcod, (outs GPR:$dst), (ins GPR:$a, so_imm:$b), DPFrm,
               IIC_iALUi, opc, "s $dst, $a, $b",
               [(set GPR:$dst, (opnode GPR:$a, so_imm:$b))]> {
    let Inst{25} = 1;
  }
  def rr : AI1<opcod, (outs GPR:$dst), (ins GPR:$a, GPR:$b), DPFrm,
               IIC_iALUr, opc, "s $dst, $a, $b",
               [(set GPR:$dst, (opnode GPR:$a, GPR:$b))]> {
    let isCommutable = Commutable;
	let Inst{25} = 0;
  }
  def rs : AI1<opcod, (outs GPR:$dst), (ins GPR:$a, so_reg:$b), DPSoRegFrm,
               IIC_iALUsr, opc, "s $dst, $a, $b",
               [(set GPR:$dst, (opnode GPR:$a, so_reg:$b))]> {
    let Inst{25} = 0;
  }
}
}

/// AI1_cmp_irs - Defines a set of (op r, {so_imm|r|so_reg}) cmp / test
/// patterns. Similar to AsI1_bin_irs except the instruction does not produce
/// a explicit result, only implicitly set CPSR.
let Defs = [CPSR] in {
multiclass AI1_cmp_irs<bits<4> opcod, string opc, PatFrag opnode,
                       bit Commutable = 0> {
  def ri : AI1<opcod, (outs), (ins GPR:$a, so_imm:$b), DPFrm, IIC_iCMPi,
               opc, " $a, $b",
               [(opnode GPR:$a, so_imm:$b)]> {
    let Inst{25} = 1;
  }
  def rr : AI1<opcod, (outs), (ins GPR:$a, GPR:$b), DPFrm, IIC_iCMPr,
               opc, " $a, $b",
               [(opnode GPR:$a, GPR:$b)]> {
    let Inst{25} = 0;
    let isCommutable = Commutable;
  }
  def rs : AI1<opcod, (outs), (ins GPR:$a, so_reg:$b), DPSoRegFrm, IIC_iCMPsr,
               opc, " $a, $b",
               [(opnode GPR:$a, so_reg:$b)]> {
    let Inst{25} = 0;
  }
}
}

/// AI_unary_rrot - A unary operation with two forms: one whose operand is a
/// register and one whose operand is a register rotated by 8/16/24.
/// FIXME: Remove the 'r' variant. Its rot_imm is zero.
multiclass AI_unary_rrot<bits<8> opcod, string opc, PatFrag opnode> {
  def r     : AExtI<opcod, (outs GPR:$dst), (ins GPR:$src),
                 IIC_iUNAr, opc, " $dst, $src",
                 [(set GPR:$dst, (opnode GPR:$src))]>,
              Requires<[IsARM, HasV6]> {
                let Inst{19-16} = 0b1111;
              }
  def r_rot : AExtI<opcod, (outs GPR:$dst), (ins GPR:$src, i32imm:$rot),
                 IIC_iUNAsi, opc, " $dst, $src, ror $rot",
                 [(set GPR:$dst, (opnode (rotr GPR:$src, rot_imm:$rot)))]>,
              Requires<[IsARM, HasV6]> {
                let Inst{19-16} = 0b1111;
              }
}

/// AI_bin_rrot - A binary operation with two forms: one whose operand is a
/// register and one whose operand is a register rotated by 8/16/24.
multiclass AI_bin_rrot<bits<8> opcod, string opc, PatFrag opnode> {
  def rr     : AExtI<opcod, (outs GPR:$dst), (ins GPR:$LHS, GPR:$RHS),
                  IIC_iALUr, opc, " $dst, $LHS, $RHS",
                  [(set GPR:$dst, (opnode GPR:$LHS, GPR:$RHS))]>,
                  Requires<[IsARM, HasV6]>;
  def rr_rot : AExtI<opcod, (outs GPR:$dst), (ins GPR:$LHS, GPR:$RHS, i32imm:$rot),
                  IIC_iALUsi, opc, " $dst, $LHS, $RHS, ror $rot",
                  [(set GPR:$dst, (opnode GPR:$LHS,
                                          (rotr GPR:$RHS, rot_imm:$rot)))]>,
                  Requires<[IsARM, HasV6]>;
}

/// AI1_adde_sube_irs - Define instructions and patterns for adde and sube.
let Uses = [CPSR] in {
multiclass AI1_adde_sube_irs<bits<4> opcod, string opc, PatFrag opnode,
                             bit Commutable = 0> {
  def ri : AsI1<opcod, (outs GPR:$dst), (ins GPR:$a, so_imm:$b),
                DPFrm, IIC_iALUi, opc, " $dst, $a, $b",
               [(set GPR:$dst, (opnode GPR:$a, so_imm:$b))]>,
               Requires<[IsARM, CarryDefIsUnused]> {
    let Inst{25} = 1;
  }
  def rr : AsI1<opcod, (outs GPR:$dst), (ins GPR:$a, GPR:$b),
                DPFrm, IIC_iALUr, opc, " $dst, $a, $b",
               [(set GPR:$dst, (opnode GPR:$a, GPR:$b))]>,
               Requires<[IsARM, CarryDefIsUnused]> {
    let isCommutable = Commutable;
    let Inst{25} = 0;
  }
  def rs : AsI1<opcod, (outs GPR:$dst), (ins GPR:$a, so_reg:$b),
                DPSoRegFrm, IIC_iALUsr, opc, " $dst, $a, $b",
               [(set GPR:$dst, (opnode GPR:$a, so_reg:$b))]>,
               Requires<[IsARM, CarryDefIsUnused]> {
    let Inst{25} = 0;
  }
  // Carry setting variants
  def Sri : AXI1<opcod, (outs GPR:$dst), (ins GPR:$a, so_imm:$b),
                DPFrm, IIC_iALUi, !strconcat(opc, "s $dst, $a, $b"),
               [(set GPR:$dst, (opnode GPR:$a, so_imm:$b))]>,
               Requires<[IsARM, CarryDefIsUsed]> {
    let Defs = [CPSR];
    let Inst{25} = 1;
  }
  def Srr : AXI1<opcod, (outs GPR:$dst), (ins GPR:$a, GPR:$b),
                DPFrm, IIC_iALUr, !strconcat(opc, "s $dst, $a, $b"),
               [(set GPR:$dst, (opnode GPR:$a, GPR:$b))]>,
               Requires<[IsARM, CarryDefIsUsed]> {
    let Defs = [CPSR];
    let Inst{25} = 0;
  }
  def Srs : AXI1<opcod, (outs GPR:$dst), (ins GPR:$a, so_reg:$b),
                DPSoRegFrm, IIC_iALUsr, !strconcat(opc, "s $dst, $a, $b"),
               [(set GPR:$dst, (opnode GPR:$a, so_reg:$b))]>,
               Requires<[IsARM, CarryDefIsUsed]> {
    let Defs = [CPSR];
    let Inst{25} = 0;
  }
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
let neverHasSideEffects = 1, isNotDuplicable = 1 in
def CONSTPOOL_ENTRY :
PseudoInst<(outs), (ins cpinst_operand:$instid, cpinst_operand:$cpidx,
                    i32imm:$size), NoItinerary,
           "${instid:label} ${cpidx:cpentry}", []>;

let Defs = [SP], Uses = [SP] in {
def ADJCALLSTACKUP :
PseudoInst<(outs), (ins i32imm:$amt1, i32imm:$amt2, pred:$p), NoItinerary,
           "@ ADJCALLSTACKUP $amt1",
           [(ARMcallseq_end timm:$amt1, timm:$amt2)]>;

def ADJCALLSTACKDOWN : 
PseudoInst<(outs), (ins i32imm:$amt, pred:$p), NoItinerary,
           "@ ADJCALLSTACKDOWN $amt",
           [(ARMcallseq_start timm:$amt)]>;
}

def DWARF_LOC :
PseudoInst<(outs), (ins i32imm:$line, i32imm:$col, i32imm:$file), NoItinerary,
           ".loc $file, $line, $col",
           [(dwarf_loc (i32 imm:$line), (i32 imm:$col), (i32 imm:$file))]>;


// Address computation and loads and stores in PIC mode.
let isNotDuplicable = 1 in {
def PICADD : AXI1<0b0100, (outs GPR:$dst), (ins GPR:$a, pclabel:$cp, pred:$p),
                  Pseudo, IIC_iALUr, "$cp:\n\tadd$p $dst, pc, $a",
                   [(set GPR:$dst, (ARMpic_add GPR:$a, imm:$cp))]>;

let AddedComplexity = 10 in {
let canFoldAsLoad = 1 in
def PICLDR  : AXI2ldw<(outs GPR:$dst), (ins addrmodepc:$addr, pred:$p),
                  Pseudo, IIC_iLoadr, "${addr:label}:\n\tldr$p $dst, $addr",
                  [(set GPR:$dst, (load addrmodepc:$addr))]>;

def PICLDRH : AXI3ldh<(outs GPR:$dst), (ins addrmodepc:$addr, pred:$p),
                  Pseudo, IIC_iLoadr, "${addr:label}:\n\tldr${p}h $dst, $addr",
                  [(set GPR:$dst, (zextloadi16 addrmodepc:$addr))]>;

def PICLDRB : AXI2ldb<(outs GPR:$dst), (ins addrmodepc:$addr, pred:$p),
                  Pseudo, IIC_iLoadr, "${addr:label}:\n\tldr${p}b $dst, $addr",
                  [(set GPR:$dst, (zextloadi8 addrmodepc:$addr))]>;

def PICLDRSH : AXI3ldsh<(outs GPR:$dst), (ins addrmodepc:$addr, pred:$p),
                  Pseudo, IIC_iLoadr, "${addr:label}:\n\tldr${p}sh $dst, $addr",
                  [(set GPR:$dst, (sextloadi16 addrmodepc:$addr))]>;

def PICLDRSB : AXI3ldsb<(outs GPR:$dst), (ins addrmodepc:$addr, pred:$p),
                  Pseudo, IIC_iLoadr, "${addr:label}:\n\tldr${p}sb $dst, $addr",
                  [(set GPR:$dst, (sextloadi8 addrmodepc:$addr))]>;
}
let AddedComplexity = 10 in {
def PICSTR  : AXI2stw<(outs), (ins GPR:$src, addrmodepc:$addr, pred:$p),
               Pseudo, IIC_iStorer, "${addr:label}:\n\tstr$p $src, $addr",
               [(store GPR:$src, addrmodepc:$addr)]>;

def PICSTRH : AXI3sth<(outs), (ins GPR:$src, addrmodepc:$addr, pred:$p),
               Pseudo, IIC_iStorer, "${addr:label}:\n\tstr${p}h $src, $addr",
               [(truncstorei16 GPR:$src, addrmodepc:$addr)]>;

def PICSTRB : AXI2stb<(outs), (ins GPR:$src, addrmodepc:$addr, pred:$p),
               Pseudo, IIC_iStorer, "${addr:label}:\n\tstr${p}b $src, $addr",
               [(truncstorei8 GPR:$src, addrmodepc:$addr)]>;
}
} // isNotDuplicable = 1


// LEApcrel - Load a pc-relative address into a register without offending the
// assembler.
def LEApcrel : AXI1<0x0, (outs GPR:$dst), (ins i32imm:$label, pred:$p),
                    Pseudo, IIC_iALUi,
            !strconcat(!strconcat(".set ${:private}PCRELV${:uid}, ($label-(",
                                  "${:private}PCRELL${:uid}+8))\n"),
                       !strconcat("${:private}PCRELL${:uid}:\n\t",
                                  "add$p $dst, pc, #${:private}PCRELV${:uid}")),
                   []>;

def LEApcrelJT : AXI1<0x0, (outs GPR:$dst),
                           (ins i32imm:$label, nohash_imm:$id, pred:$p),
          Pseudo, IIC_iALUi,
   !strconcat(!strconcat(".set ${:private}PCRELV${:uid}, "
                         "(${label}_${id}-(",
                                  "${:private}PCRELL${:uid}+8))\n"),
                       !strconcat("${:private}PCRELL${:uid}:\n\t",
                                  "add$p $dst, pc, #${:private}PCRELV${:uid}")),
                   []> {
    let Inst{25} = 1;
}

//===----------------------------------------------------------------------===//
//  Control Flow Instructions.
//

let isReturn = 1, isTerminator = 1 in
  def BX_RET : AI<(outs), (ins), BrMiscFrm, IIC_Br, 
                  "bx", " lr", [(ARMretflag)]> {
  let Inst{7-4}   = 0b0001;
  let Inst{19-8}  = 0b111111111111;
  let Inst{27-20} = 0b00010010;
}

// FIXME: remove when we have a way to marking a MI with these properties.
// FIXME: $dst1 should be a def. But the extra ops must be in the end of the
// operand list.
// FIXME: Should pc be an implicit operand like PICADD, etc?
let isReturn = 1, isTerminator = 1, mayLoad = 1 in
  def LDM_RET : AXI4ld<(outs),
                    (ins addrmode4:$addr, pred:$p, reglist:$dst1, variable_ops),
                    LdStMulFrm, IIC_Br, "ldm${p}${addr:submode} $addr, $dst1",
                    []>;

// On non-Darwin platforms R9 is callee-saved.
let isCall = 1,
  Defs = [R0,  R1,  R2,  R3,  R12, LR,
          D0,  D1,  D2,  D3,  D4,  D5,  D6,  D7,
          D16, D17, D18, D19, D20, D21, D22, D23,
          D24, D25, D26, D27, D28, D29, D30, D31, CPSR] in {
  def BL  : ABXI<0b1011, (outs), (ins i32imm:$func, variable_ops),
                IIC_Br, "bl ${func:call}",
                [(ARMcall tglobaladdr:$func)]>,
            Requires<[IsARM, IsNotDarwin]>;

  def BL_pred : ABI<0b1011, (outs), (ins i32imm:$func, variable_ops),
                   IIC_Br, "bl", " ${func:call}",
                   [(ARMcall_pred tglobaladdr:$func)]>,
                Requires<[IsARM, IsNotDarwin]>;

  // ARMv5T and above
  def BLX : AXI<(outs), (ins GPR:$func, variable_ops), BrMiscFrm,
                IIC_Br, "blx $func",
                [(ARMcall GPR:$func)]>,
            Requires<[IsARM, HasV5T, IsNotDarwin]> {
    let Inst{7-4}   = 0b0011;
    let Inst{19-8}  = 0b111111111111;
    let Inst{27-20} = 0b00010010;
  }

  // ARMv4T
  def BX : ABXIx2<(outs), (ins GPR:$func, variable_ops),
                  IIC_Br, "mov lr, pc\n\tbx $func",
                  [(ARMcall_nolink GPR:$func)]>,
           Requires<[IsARM, IsNotDarwin]> {
    let Inst{7-4}   = 0b0001;
    let Inst{19-8}  = 0b111111111111;
    let Inst{27-20} = 0b00010010;
  }
}

// On Darwin R9 is call-clobbered.
let isCall = 1,
  Defs = [R0,  R1,  R2,  R3,  R9,  R12, LR,
          D0,  D1,  D2,  D3,  D4,  D5,  D6,  D7,
          D16, D17, D18, D19, D20, D21, D22, D23,
          D24, D25, D26, D27, D28, D29, D30, D31, CPSR] in {
  def BLr9  : ABXI<0b1011, (outs), (ins i32imm:$func, variable_ops),
                IIC_Br, "bl ${func:call}",
                [(ARMcall tglobaladdr:$func)]>, Requires<[IsARM, IsDarwin]>;

  def BLr9_pred : ABI<0b1011, (outs), (ins i32imm:$func, variable_ops),
                   IIC_Br, "bl", " ${func:call}",
                   [(ARMcall_pred tglobaladdr:$func)]>,
                  Requires<[IsARM, IsDarwin]>;

  // ARMv5T and above
  def BLXr9 : AXI<(outs), (ins GPR:$func, variable_ops), BrMiscFrm,
                IIC_Br, "blx $func",
                [(ARMcall GPR:$func)]>, Requires<[IsARM, HasV5T, IsDarwin]> {
    let Inst{7-4}   = 0b0011;
    let Inst{19-8}  = 0b111111111111;
    let Inst{27-20} = 0b00010010;
  }

  // ARMv4T
  def BXr9 : ABXIx2<(outs), (ins GPR:$func, variable_ops),
                  IIC_Br, "mov lr, pc\n\tbx $func",
                  [(ARMcall_nolink GPR:$func)]>, Requires<[IsARM, IsDarwin]> {
    let Inst{7-4}   = 0b0001;
    let Inst{19-8}  = 0b111111111111;
    let Inst{27-20} = 0b00010010;
  }
}

let isBranch = 1, isTerminator = 1 in {
  // B is "predicable" since it can be xformed into a Bcc.
  let isBarrier = 1 in {
    let isPredicable = 1 in
    def B : ABXI<0b1010, (outs), (ins brtarget:$target), IIC_Br,
                "b $target", [(br bb:$target)]>;

  let isNotDuplicable = 1, isIndirectBranch = 1 in {
  def BR_JTr : JTI<(outs), (ins GPR:$target, jtblock_operand:$jt, i32imm:$id),
                    IIC_Br, "mov pc, $target \n$jt",
                    [(ARMbrjt GPR:$target, tjumptable:$jt, imm:$id)]> {
    let Inst{20}    = 0; // S Bit
    let Inst{24-21} = 0b1101;
    let Inst{27-25} = 0b000;
  }
  def BR_JTm : JTI<(outs),
                   (ins addrmode2:$target, jtblock_operand:$jt, i32imm:$id),
                   IIC_Br, "ldr pc, $target \n$jt",
                   [(ARMbrjt (i32 (load addrmode2:$target)), tjumptable:$jt,
                     imm:$id)]> {
    let Inst{20}    = 1; // L bit
    let Inst{21}    = 0; // W bit
    let Inst{22}    = 0; // B bit
    let Inst{24}    = 1; // P bit
    let Inst{27-25} = 0b011;
  }
  def BR_JTadd : JTI<(outs),
                   (ins GPR:$target, GPR:$idx, jtblock_operand:$jt, i32imm:$id),
                    IIC_Br, "add pc, $target, $idx \n$jt",
                    [(ARMbrjt (add GPR:$target, GPR:$idx), tjumptable:$jt,
                      imm:$id)]> {
    let Inst{20}    = 0; // S bit
    let Inst{24-21} = 0b0100;
    let Inst{27-25} = 0b000;
  }
  } // isNotDuplicable = 1, isIndirectBranch = 1
  } // isBarrier = 1

  // FIXME: should be able to write a pattern for ARMBrcond, but can't use
  // a two-value operand where a dag node expects two operands. :( 
  def Bcc : ABI<0b1010, (outs), (ins brtarget:$target),
               IIC_Br, "b", " $target",
               [/*(ARMbrcond bb:$target, imm:$cc, CCR:$ccr)*/]>;
}

//===----------------------------------------------------------------------===//
//  Load / store Instructions.
//

// Load
let canFoldAsLoad = 1 in 
def LDR  : AI2ldw<(outs GPR:$dst), (ins addrmode2:$addr), LdFrm, IIC_iLoadr,
               "ldr", " $dst, $addr",
               [(set GPR:$dst, (load addrmode2:$addr))]>;

// Special LDR for loads from non-pc-relative constpools.
let canFoldAsLoad = 1, mayLoad = 1, isReMaterializable = 1 in
def LDRcp : AI2ldw<(outs GPR:$dst), (ins addrmode2:$addr), LdFrm, IIC_iLoadr,
                 "ldr", " $dst, $addr", []>;

// Loads with zero extension
def LDRH  : AI3ldh<(outs GPR:$dst), (ins addrmode3:$addr), LdMiscFrm,
                  IIC_iLoadr, "ldr", "h $dst, $addr",
                  [(set GPR:$dst, (zextloadi16 addrmode3:$addr))]>;

def LDRB  : AI2ldb<(outs GPR:$dst), (ins addrmode2:$addr), LdFrm, 
                  IIC_iLoadr, "ldr", "b $dst, $addr",
                  [(set GPR:$dst, (zextloadi8 addrmode2:$addr))]>;

// Loads with sign extension
def LDRSH : AI3ldsh<(outs GPR:$dst), (ins addrmode3:$addr), LdMiscFrm,
                   IIC_iLoadr, "ldr", "sh $dst, $addr",
                   [(set GPR:$dst, (sextloadi16 addrmode3:$addr))]>;

def LDRSB : AI3ldsb<(outs GPR:$dst), (ins addrmode3:$addr), LdMiscFrm,
                   IIC_iLoadr, "ldr", "sb $dst, $addr",
                   [(set GPR:$dst, (sextloadi8 addrmode3:$addr))]>;

let mayLoad = 1 in {
// Load doubleword
def LDRD : AI3ldd<(outs GPR:$dst1, GPR:$dst2), (ins addrmode3:$addr), LdMiscFrm,
                 IIC_iLoadr, "ldr", "d $dst1, $addr",
                 []>, Requires<[IsARM, HasV5T]>;

// Indexed loads
def LDR_PRE  : AI2ldwpr<(outs GPR:$dst, GPR:$base_wb),
                     (ins addrmode2:$addr), LdFrm, IIC_iLoadru,
                     "ldr", " $dst, $addr!", "$addr.base = $base_wb", []>;

def LDR_POST : AI2ldwpo<(outs GPR:$dst, GPR:$base_wb),
                     (ins GPR:$base, am2offset:$offset), LdFrm, IIC_iLoadru,
                     "ldr", " $dst, [$base], $offset", "$base = $base_wb", []>;

def LDRH_PRE  : AI3ldhpr<(outs GPR:$dst, GPR:$base_wb),
                     (ins addrmode3:$addr), LdMiscFrm, IIC_iLoadru,
                     "ldr", "h $dst, $addr!", "$addr.base = $base_wb", []>;

def LDRH_POST : AI3ldhpo<(outs GPR:$dst, GPR:$base_wb),
                     (ins GPR:$base,am3offset:$offset), LdMiscFrm, IIC_iLoadru,
                     "ldr", "h $dst, [$base], $offset", "$base = $base_wb", []>;

def LDRB_PRE  : AI2ldbpr<(outs GPR:$dst, GPR:$base_wb),
                     (ins addrmode2:$addr), LdFrm, IIC_iLoadru,
                     "ldr", "b $dst, $addr!", "$addr.base = $base_wb", []>;

def LDRB_POST : AI2ldbpo<(outs GPR:$dst, GPR:$base_wb),
                     (ins GPR:$base,am2offset:$offset), LdFrm, IIC_iLoadru,
                     "ldr", "b $dst, [$base], $offset", "$base = $base_wb", []>;

def LDRSH_PRE : AI3ldshpr<(outs GPR:$dst, GPR:$base_wb),
                      (ins addrmode3:$addr), LdMiscFrm, IIC_iLoadru,
                      "ldr", "sh $dst, $addr!", "$addr.base = $base_wb", []>;

def LDRSH_POST: AI3ldshpo<(outs GPR:$dst, GPR:$base_wb),
                      (ins GPR:$base,am3offset:$offset), LdMiscFrm, IIC_iLoadru,
                    "ldr", "sh $dst, [$base], $offset", "$base = $base_wb", []>;

def LDRSB_PRE : AI3ldsbpr<(outs GPR:$dst, GPR:$base_wb),
                      (ins addrmode3:$addr), LdMiscFrm, IIC_iLoadru,
                      "ldr", "sb $dst, $addr!", "$addr.base = $base_wb", []>;

def LDRSB_POST: AI3ldsbpo<(outs GPR:$dst, GPR:$base_wb),
                      (ins GPR:$base,am3offset:$offset), LdMiscFrm, IIC_iLoadru,
                    "ldr", "sb $dst, [$base], $offset", "$base = $base_wb", []>;
}

// Store
def STR  : AI2stw<(outs), (ins GPR:$src, addrmode2:$addr), StFrm, IIC_iStorer,
               "str", " $src, $addr",
               [(store GPR:$src, addrmode2:$addr)]>;

// Stores with truncate
def STRH : AI3sth<(outs), (ins GPR:$src, addrmode3:$addr), StMiscFrm, IIC_iStorer,
               "str", "h $src, $addr",
               [(truncstorei16 GPR:$src, addrmode3:$addr)]>;

def STRB : AI2stb<(outs), (ins GPR:$src, addrmode2:$addr), StFrm, IIC_iStorer,
               "str", "b $src, $addr",
               [(truncstorei8 GPR:$src, addrmode2:$addr)]>;

// Store doubleword
let mayStore = 1 in
def STRD : AI3std<(outs), (ins GPR:$src1, GPR:$src2, addrmode3:$addr),
               StMiscFrm, IIC_iStorer,
               "str", "d $src1, $addr", []>, Requires<[IsARM, HasV5T]>;

// Indexed stores
def STR_PRE  : AI2stwpr<(outs GPR:$base_wb),
                     (ins GPR:$src, GPR:$base, am2offset:$offset), 
                     StFrm, IIC_iStoreru,
                    "str", " $src, [$base, $offset]!", "$base = $base_wb",
                    [(set GPR:$base_wb,
                      (pre_store GPR:$src, GPR:$base, am2offset:$offset))]>;

def STR_POST : AI2stwpo<(outs GPR:$base_wb),
                     (ins GPR:$src, GPR:$base,am2offset:$offset), 
                     StFrm, IIC_iStoreru,
                    "str", " $src, [$base], $offset", "$base = $base_wb",
                    [(set GPR:$base_wb,
                      (post_store GPR:$src, GPR:$base, am2offset:$offset))]>;

def STRH_PRE : AI3sthpr<(outs GPR:$base_wb),
                     (ins GPR:$src, GPR:$base,am3offset:$offset), 
                     StMiscFrm, IIC_iStoreru,
                     "str", "h $src, [$base, $offset]!", "$base = $base_wb",
                    [(set GPR:$base_wb,
                      (pre_truncsti16 GPR:$src, GPR:$base,am3offset:$offset))]>;

def STRH_POST: AI3sthpo<(outs GPR:$base_wb),
                     (ins GPR:$src, GPR:$base,am3offset:$offset), 
                     StMiscFrm, IIC_iStoreru,
                     "str", "h $src, [$base], $offset", "$base = $base_wb",
                    [(set GPR:$base_wb, (post_truncsti16 GPR:$src,
                                         GPR:$base, am3offset:$offset))]>;

def STRB_PRE : AI2stbpr<(outs GPR:$base_wb),
                     (ins GPR:$src, GPR:$base,am2offset:$offset), 
                     StFrm, IIC_iStoreru,
                     "str", "b $src, [$base, $offset]!", "$base = $base_wb",
                    [(set GPR:$base_wb, (pre_truncsti8 GPR:$src,
                                         GPR:$base, am2offset:$offset))]>;

def STRB_POST: AI2stbpo<(outs GPR:$base_wb),
                     (ins GPR:$src, GPR:$base,am2offset:$offset), 
                     StFrm, IIC_iStoreru,
                     "str", "b $src, [$base], $offset", "$base = $base_wb",
                    [(set GPR:$base_wb, (post_truncsti8 GPR:$src,
                                         GPR:$base, am2offset:$offset))]>;

//===----------------------------------------------------------------------===//
//  Load / store multiple Instructions.
//

// FIXME: $dst1 should be a def.
let mayLoad = 1 in
def LDM : AXI4ld<(outs),
               (ins addrmode4:$addr, pred:$p, reglist:$dst1, variable_ops),
               LdStMulFrm, IIC_iLoadm, "ldm${p}${addr:submode} $addr, $dst1",
               []>;

let mayStore = 1 in
def STM : AXI4st<(outs),
               (ins addrmode4:$addr, pred:$p, reglist:$src1, variable_ops),
               LdStMulFrm, IIC_iStorem, "stm${p}${addr:submode} $addr, $src1",
               []>;

//===----------------------------------------------------------------------===//
//  Move Instructions.
//

let neverHasSideEffects = 1 in
def MOVr : AsI1<0b1101, (outs GPR:$dst), (ins GPR:$src), DPFrm, IIC_iMOVr,
                 "mov", " $dst, $src", []>, UnaryDP;
def MOVs : AsI1<0b1101, (outs GPR:$dst), (ins so_reg:$src), 
                 DPSoRegFrm, IIC_iMOVsr,
                 "mov", " $dst, $src", [(set GPR:$dst, so_reg:$src)]>, UnaryDP;

let isReMaterializable = 1, isAsCheapAsAMove = 1 in
def MOVi : AsI1<0b1101, (outs GPR:$dst), (ins so_imm:$src), DPFrm, IIC_iMOVi,
                 "mov", " $dst, $src", [(set GPR:$dst, so_imm:$src)]>, UnaryDP {
    let Inst{25} = 1;
}

def MOVrx : AsI1<0b1101, (outs GPR:$dst), (ins GPR:$src), Pseudo, IIC_iMOVsi,
                 "mov", " $dst, $src, rrx",
                 [(set GPR:$dst, (ARMrrx GPR:$src))]>, UnaryDP;

// These aren't really mov instructions, but we have to define them this way
// due to flag operands.

let Defs = [CPSR] in {
def MOVsrl_flag : AI1<0b1101, (outs GPR:$dst), (ins GPR:$src), Pseudo, 
                      IIC_iMOVsi, "mov", "s $dst, $src, lsr #1",
                      [(set GPR:$dst, (ARMsrl_flag GPR:$src))]>, UnaryDP;
def MOVsra_flag : AI1<0b1101, (outs GPR:$dst), (ins GPR:$src), Pseudo,
                      IIC_iMOVsi, "mov", "s $dst, $src, asr #1",
                      [(set GPR:$dst, (ARMsra_flag GPR:$src))]>, UnaryDP;
}

//===----------------------------------------------------------------------===//
//  Extend Instructions.
//

// Sign extenders

defm SXTB  : AI_unary_rrot<0b01101010,
                           "sxtb", UnOpFrag<(sext_inreg node:$Src, i8)>>;
defm SXTH  : AI_unary_rrot<0b01101011,
                           "sxth", UnOpFrag<(sext_inreg node:$Src, i16)>>;

defm SXTAB : AI_bin_rrot<0b01101010,
               "sxtab", BinOpFrag<(add node:$LHS, (sext_inreg node:$RHS, i8))>>;
defm SXTAH : AI_bin_rrot<0b01101011,
               "sxtah", BinOpFrag<(add node:$LHS, (sext_inreg node:$RHS,i16))>>;

// TODO: SXT(A){B|H}16

// Zero extenders

let AddedComplexity = 16 in {
defm UXTB   : AI_unary_rrot<0b01101110,
                            "uxtb"  , UnOpFrag<(and node:$Src, 0x000000FF)>>;
defm UXTH   : AI_unary_rrot<0b01101111,
                            "uxth"  , UnOpFrag<(and node:$Src, 0x0000FFFF)>>;
defm UXTB16 : AI_unary_rrot<0b01101100,
                            "uxtb16", UnOpFrag<(and node:$Src, 0x00FF00FF)>>;

def : ARMV6Pat<(and (shl GPR:$Src, (i32 8)), 0xFF00FF),
               (UXTB16r_rot GPR:$Src, 24)>;
def : ARMV6Pat<(and (srl GPR:$Src, (i32 8)), 0xFF00FF),
               (UXTB16r_rot GPR:$Src, 8)>;

defm UXTAB : AI_bin_rrot<0b01101110, "uxtab",
                        BinOpFrag<(add node:$LHS, (and node:$RHS, 0x00FF))>>;
defm UXTAH : AI_bin_rrot<0b01101111, "uxtah",
                        BinOpFrag<(add node:$LHS, (and node:$RHS, 0xFFFF))>>;
}

// This isn't safe in general, the add is two 16-bit units, not a 32-bit add.
//defm UXTAB16 : xxx<"uxtab16", 0xff00ff>;

// TODO: UXT(A){B|H}16

//===----------------------------------------------------------------------===//
//  Arithmetic Instructions.
//

defm ADD  : AsI1_bin_irs<0b0100, "add",
                         BinOpFrag<(add  node:$LHS, node:$RHS)>, 1>;
defm SUB  : AsI1_bin_irs<0b0010, "sub",
                         BinOpFrag<(sub  node:$LHS, node:$RHS)>>;

// ADD and SUB with 's' bit set.
defm ADDS : AI1_bin_s_irs<0b0100, "add",
                          BinOpFrag<(addc node:$LHS, node:$RHS)>>;
defm SUBS : AI1_bin_s_irs<0b0010, "sub",
                          BinOpFrag<(subc node:$LHS, node:$RHS)>>;

defm ADC : AI1_adde_sube_irs<0b0101, "adc",
                             BinOpFrag<(adde node:$LHS, node:$RHS)>, 1>;
defm SBC : AI1_adde_sube_irs<0b0110, "sbc",
                             BinOpFrag<(sube node:$LHS, node:$RHS)>>;

// These don't define reg/reg forms, because they are handled above.
def RSBri : AsI1<0b0011, (outs GPR:$dst), (ins GPR:$a, so_imm:$b), DPFrm,
                  IIC_iALUi, "rsb", " $dst, $a, $b",
                  [(set GPR:$dst, (sub so_imm:$b, GPR:$a))]> {
    let Inst{25} = 1;
}

def RSBrs : AsI1<0b0011, (outs GPR:$dst), (ins GPR:$a, so_reg:$b), DPSoRegFrm,
                  IIC_iALUsr, "rsb", " $dst, $a, $b",
                  [(set GPR:$dst, (sub so_reg:$b, GPR:$a))]>;

// RSB with 's' bit set.
let Defs = [CPSR] in {
def RSBSri : AI1<0b0011, (outs GPR:$dst), (ins GPR:$a, so_imm:$b), DPFrm,
                 IIC_iALUi, "rsb", "s $dst, $a, $b",
                 [(set GPR:$dst, (subc so_imm:$b, GPR:$a))]> {
    let Inst{25} = 1;
}
def RSBSrs : AI1<0b0011, (outs GPR:$dst), (ins GPR:$a, so_reg:$b), DPSoRegFrm,
                 IIC_iALUsr, "rsb", "s $dst, $a, $b",
                 [(set GPR:$dst, (subc so_reg:$b, GPR:$a))]>;
}

let Uses = [CPSR] in {
def RSCri : AsI1<0b0111, (outs GPR:$dst), (ins GPR:$a, so_imm:$b),
                 DPFrm, IIC_iALUi, "rsc", " $dst, $a, $b",
                 [(set GPR:$dst, (sube so_imm:$b, GPR:$a))]>,
                 Requires<[IsARM, CarryDefIsUnused]> {
    let Inst{25} = 1;
}
def RSCrs : AsI1<0b0111, (outs GPR:$dst), (ins GPR:$a, so_reg:$b),
                 DPSoRegFrm, IIC_iALUsr, "rsc", " $dst, $a, $b",
                 [(set GPR:$dst, (sube so_reg:$b, GPR:$a))]>,
                 Requires<[IsARM, CarryDefIsUnused]>;
}

// FIXME: Allow these to be predicated.
let Defs = [CPSR], Uses = [CPSR] in {
def RSCSri : AXI1<0b0111, (outs GPR:$dst), (ins GPR:$a, so_imm:$b),
                  DPFrm, IIC_iALUi, "rscs $dst, $a, $b",
                  [(set GPR:$dst, (sube so_imm:$b, GPR:$a))]>,
                  Requires<[IsARM, CarryDefIsUnused]> {
    let Inst{25} = 1;
}
def RSCSrs : AXI1<0b0111, (outs GPR:$dst), (ins GPR:$a, so_reg:$b),
                  DPSoRegFrm, IIC_iALUsr, "rscs $dst, $a, $b",
                  [(set GPR:$dst, (sube so_reg:$b, GPR:$a))]>,
                  Requires<[IsARM, CarryDefIsUnused]>;
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

defm AND   : AsI1_bin_irs<0b0000, "and",
                          BinOpFrag<(and node:$LHS, node:$RHS)>, 1>;
defm ORR   : AsI1_bin_irs<0b1100, "orr",
                          BinOpFrag<(or  node:$LHS, node:$RHS)>, 1>;
defm EOR   : AsI1_bin_irs<0b0001, "eor",
                          BinOpFrag<(xor node:$LHS, node:$RHS)>, 1>;
defm BIC   : AsI1_bin_irs<0b1110, "bic",
                          BinOpFrag<(and node:$LHS, (not node:$RHS))>>;

def BFC    : I<(outs GPR:$dst), (ins GPR:$src, bf_inv_mask_imm:$imm),
               AddrMode1, Size4Bytes, IndexModeNone, DPFrm, IIC_iALUi,
               "bfc", " $dst, $imm", "$src = $dst",
               [(set GPR:$dst, (and GPR:$src, bf_inv_mask_imm:$imm))]>,
               Requires<[IsARM, HasV6T2]> {
  let Inst{27-21} = 0b0111110;
  let Inst{6-0}   = 0b0011111;
}

def  MVNr  : AsI1<0b1111, (outs GPR:$dst), (ins GPR:$src), DPFrm, IIC_iMOVr,
                  "mvn", " $dst, $src",
                  [(set GPR:$dst, (not GPR:$src))]>, UnaryDP;
def  MVNs  : AsI1<0b1111, (outs GPR:$dst), (ins so_reg:$src), DPSoRegFrm,
                  IIC_iMOVsr, "mvn", " $dst, $src",
                  [(set GPR:$dst, (not so_reg:$src))]>, UnaryDP;
let isReMaterializable = 1, isAsCheapAsAMove = 1 in
def  MVNi  : AsI1<0b1111, (outs GPR:$dst), (ins so_imm:$imm), DPFrm, 
                  IIC_iMOVi, "mvn", " $dst, $imm",
                  [(set GPR:$dst, so_imm_not:$imm)]>,UnaryDP {
    let Inst{25} = 1;
}

def : ARMPat<(and   GPR:$src, so_imm_not:$imm),
             (BICri GPR:$src, so_imm_not:$imm)>;

//===----------------------------------------------------------------------===//
//  Multiply Instructions.
//

let isCommutable = 1 in
def MUL   : AsMul1I<0b0000000, (outs GPR:$dst), (ins GPR:$a, GPR:$b),
                   IIC_iMUL32, "mul", " $dst, $a, $b",
                   [(set GPR:$dst, (mul GPR:$a, GPR:$b))]>;

def MLA   : AsMul1I<0b0000001, (outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$c),
                    IIC_iMAC32, "mla", " $dst, $a, $b, $c",
                   [(set GPR:$dst, (add (mul GPR:$a, GPR:$b), GPR:$c))]>;

def MLS   : AMul1I<0b0000011, (outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$c),
                   IIC_iMAC32, "mls", " $dst, $a, $b, $c",
                   [(set GPR:$dst, (sub GPR:$c, (mul GPR:$a, GPR:$b)))]>,
                   Requires<[IsARM, HasV6T2]>;

// Extra precision multiplies with low / high results
let neverHasSideEffects = 1 in {
let isCommutable = 1 in {
def SMULL : AsMul1I<0b0000110, (outs GPR:$ldst, GPR:$hdst),
                               (ins GPR:$a, GPR:$b), IIC_iMUL64,
                    "smull", " $ldst, $hdst, $a, $b", []>;

def UMULL : AsMul1I<0b0000100, (outs GPR:$ldst, GPR:$hdst),
                               (ins GPR:$a, GPR:$b), IIC_iMUL64,
                    "umull", " $ldst, $hdst, $a, $b", []>;
}

// Multiply + accumulate
def SMLAL : AsMul1I<0b0000111, (outs GPR:$ldst, GPR:$hdst),
                               (ins GPR:$a, GPR:$b), IIC_iMAC64,
                    "smlal", " $ldst, $hdst, $a, $b", []>;

def UMLAL : AsMul1I<0b0000101, (outs GPR:$ldst, GPR:$hdst),
                               (ins GPR:$a, GPR:$b), IIC_iMAC64,
                    "umlal", " $ldst, $hdst, $a, $b", []>;

def UMAAL : AMul1I <0b0000010, (outs GPR:$ldst, GPR:$hdst),
                               (ins GPR:$a, GPR:$b), IIC_iMAC64,
                    "umaal", " $ldst, $hdst, $a, $b", []>,
                    Requires<[IsARM, HasV6]>;
} // neverHasSideEffects

// Most significant word multiply
def SMMUL : AMul2I <0b0111010, (outs GPR:$dst), (ins GPR:$a, GPR:$b),
               IIC_iMUL32, "smmul", " $dst, $a, $b",
               [(set GPR:$dst, (mulhs GPR:$a, GPR:$b))]>,
            Requires<[IsARM, HasV6]> {
  let Inst{7-4}   = 0b0001;
  let Inst{15-12} = 0b1111;
}

def SMMLA : AMul2I <0b0111010, (outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$c),
               IIC_iMAC32, "smmla", " $dst, $a, $b, $c",
               [(set GPR:$dst, (add (mulhs GPR:$a, GPR:$b), GPR:$c))]>,
            Requires<[IsARM, HasV6]> {
  let Inst{7-4}   = 0b0001;
}


def SMMLS : AMul2I <0b0111010, (outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$c),
               IIC_iMAC32, "smmls", " $dst, $a, $b, $c",
               [(set GPR:$dst, (sub GPR:$c, (mulhs GPR:$a, GPR:$b)))]>,
            Requires<[IsARM, HasV6]> {
  let Inst{7-4}   = 0b1101;
}

multiclass AI_smul<string opc, PatFrag opnode> {
  def BB : AMulxyI<0b0001011, (outs GPR:$dst), (ins GPR:$a, GPR:$b),
              IIC_iMUL32, !strconcat(opc, "bb"), " $dst, $a, $b",
              [(set GPR:$dst, (opnode (sext_inreg GPR:$a, i16),
                                      (sext_inreg GPR:$b, i16)))]>,
           Requires<[IsARM, HasV5TE]> {
             let Inst{5} = 0;
             let Inst{6} = 0;
           }

  def BT : AMulxyI<0b0001011, (outs GPR:$dst), (ins GPR:$a, GPR:$b),
              IIC_iMUL32, !strconcat(opc, "bt"), " $dst, $a, $b",
              [(set GPR:$dst, (opnode (sext_inreg GPR:$a, i16),
                                      (sra GPR:$b, (i32 16))))]>,
           Requires<[IsARM, HasV5TE]> {
             let Inst{5} = 0;
             let Inst{6} = 1;
           }

  def TB : AMulxyI<0b0001011, (outs GPR:$dst), (ins GPR:$a, GPR:$b),
              IIC_iMUL32, !strconcat(opc, "tb"), " $dst, $a, $b",
              [(set GPR:$dst, (opnode (sra GPR:$a, (i32 16)),
                                      (sext_inreg GPR:$b, i16)))]>,
           Requires<[IsARM, HasV5TE]> {
             let Inst{5} = 1;
             let Inst{6} = 0;
           }

  def TT : AMulxyI<0b0001011, (outs GPR:$dst), (ins GPR:$a, GPR:$b),
              IIC_iMUL32, !strconcat(opc, "tt"), " $dst, $a, $b",
              [(set GPR:$dst, (opnode (sra GPR:$a, (i32 16)),
                                      (sra GPR:$b, (i32 16))))]>,
            Requires<[IsARM, HasV5TE]> {
             let Inst{5} = 1;
             let Inst{6} = 1;
           }

  def WB : AMulxyI<0b0001001, (outs GPR:$dst), (ins GPR:$a, GPR:$b),
              IIC_iMUL16, !strconcat(opc, "wb"), " $dst, $a, $b",
              [(set GPR:$dst, (sra (opnode GPR:$a,
                                    (sext_inreg GPR:$b, i16)), (i32 16)))]>,
           Requires<[IsARM, HasV5TE]> {
             let Inst{5} = 1;
             let Inst{6} = 0;
           }

  def WT : AMulxyI<0b0001001, (outs GPR:$dst), (ins GPR:$a, GPR:$b),
              IIC_iMUL16, !strconcat(opc, "wt"), " $dst, $a, $b",
              [(set GPR:$dst, (sra (opnode GPR:$a,
                                    (sra GPR:$b, (i32 16))), (i32 16)))]>,
            Requires<[IsARM, HasV5TE]> {
             let Inst{5} = 1;
             let Inst{6} = 1;
           }
}


multiclass AI_smla<string opc, PatFrag opnode> {
  def BB : AMulxyI<0b0001000, (outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$acc),
              IIC_iMAC16, !strconcat(opc, "bb"), " $dst, $a, $b, $acc",
              [(set GPR:$dst, (add GPR:$acc,
                               (opnode (sext_inreg GPR:$a, i16),
                                       (sext_inreg GPR:$b, i16))))]>,
           Requires<[IsARM, HasV5TE]> {
             let Inst{5} = 0;
             let Inst{6} = 0;
           }

  def BT : AMulxyI<0b0001000, (outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$acc),
              IIC_iMAC16, !strconcat(opc, "bt"), " $dst, $a, $b, $acc",
              [(set GPR:$dst, (add GPR:$acc, (opnode (sext_inreg GPR:$a, i16),
                                                     (sra GPR:$b, (i32 16)))))]>,
           Requires<[IsARM, HasV5TE]> {
             let Inst{5} = 0;
             let Inst{6} = 1;
           }

  def TB : AMulxyI<0b0001000, (outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$acc),
              IIC_iMAC16, !strconcat(opc, "tb"), " $dst, $a, $b, $acc",
              [(set GPR:$dst, (add GPR:$acc, (opnode (sra GPR:$a, (i32 16)),
                                                 (sext_inreg GPR:$b, i16))))]>,
           Requires<[IsARM, HasV5TE]> {
             let Inst{5} = 1;
             let Inst{6} = 0;
           }

  def TT : AMulxyI<0b0001000, (outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$acc),
              IIC_iMAC16, !strconcat(opc, "tt"), " $dst, $a, $b, $acc",
              [(set GPR:$dst, (add GPR:$acc, (opnode (sra GPR:$a, (i32 16)),
                                                     (sra GPR:$b, (i32 16)))))]>,
            Requires<[IsARM, HasV5TE]> {
             let Inst{5} = 1;
             let Inst{6} = 1;
           }

  def WB : AMulxyI<0b0001001, (outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$acc),
              IIC_iMAC16, !strconcat(opc, "wb"), " $dst, $a, $b, $acc",
              [(set GPR:$dst, (add GPR:$acc, (sra (opnode GPR:$a,
                                       (sext_inreg GPR:$b, i16)), (i32 16))))]>,
           Requires<[IsARM, HasV5TE]> {
             let Inst{5} = 0;
             let Inst{6} = 0;
           }

  def WT : AMulxyI<0b0001001, (outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$acc),
              IIC_iMAC16, !strconcat(opc, "wt"), " $dst, $a, $b, $acc",
              [(set GPR:$dst, (add GPR:$acc, (sra (opnode GPR:$a,
                                         (sra GPR:$b, (i32 16))), (i32 16))))]>,
            Requires<[IsARM, HasV5TE]> {
             let Inst{5} = 0;
             let Inst{6} = 1;
           }
}

defm SMUL : AI_smul<"smul", BinOpFrag<(mul node:$LHS, node:$RHS)>>;
defm SMLA : AI_smla<"smla", BinOpFrag<(mul node:$LHS, node:$RHS)>>;

// TODO: Halfword multiple accumulate long: SMLAL<x><y>
// TODO: Dual halfword multiple: SMUAD, SMUSD, SMLAD, SMLSD, SMLALD, SMLSLD

//===----------------------------------------------------------------------===//
//  Misc. Arithmetic Instructions.
//

def CLZ  : AMiscA1I<0b000010110, (outs GPR:$dst), (ins GPR:$src), IIC_iUNAr,
              "clz", " $dst, $src",
              [(set GPR:$dst, (ctlz GPR:$src))]>, Requires<[IsARM, HasV5T]> {
  let Inst{7-4}   = 0b0001;
  let Inst{11-8}  = 0b1111;
  let Inst{19-16} = 0b1111;
}

def REV  : AMiscA1I<0b01101011, (outs GPR:$dst), (ins GPR:$src), IIC_iUNAr,
              "rev", " $dst, $src",
              [(set GPR:$dst, (bswap GPR:$src))]>, Requires<[IsARM, HasV6]> {
  let Inst{7-4}   = 0b0011;
  let Inst{11-8}  = 0b1111;
  let Inst{19-16} = 0b1111;
}

def REV16 : AMiscA1I<0b01101011, (outs GPR:$dst), (ins GPR:$src), IIC_iUNAr,
               "rev16", " $dst, $src",
               [(set GPR:$dst,
                   (or (and (srl GPR:$src, (i32 8)), 0xFF),
                       (or (and (shl GPR:$src, (i32 8)), 0xFF00),
                           (or (and (srl GPR:$src, (i32 8)), 0xFF0000),
                               (and (shl GPR:$src, (i32 8)), 0xFF000000)))))]>,
               Requires<[IsARM, HasV6]> {
  let Inst{7-4}   = 0b1011;
  let Inst{11-8}  = 0b1111;
  let Inst{19-16} = 0b1111;
}

def REVSH : AMiscA1I<0b01101111, (outs GPR:$dst), (ins GPR:$src), IIC_iUNAr,
               "revsh", " $dst, $src",
               [(set GPR:$dst,
                  (sext_inreg
                    (or (srl (and GPR:$src, 0xFF00), (i32 8)),
                        (shl GPR:$src, (i32 8))), i16))]>,
               Requires<[IsARM, HasV6]> {
  let Inst{7-4}   = 0b1011;
  let Inst{11-8}  = 0b1111;
  let Inst{19-16} = 0b1111;
}

def PKHBT : AMiscA1I<0b01101000, (outs GPR:$dst),
                                 (ins GPR:$src1, GPR:$src2, i32imm:$shamt),
               IIC_iALUsi, "pkhbt", " $dst, $src1, $src2, LSL $shamt",
               [(set GPR:$dst, (or (and GPR:$src1, 0xFFFF),
                                   (and (shl GPR:$src2, (i32 imm:$shamt)),
                                        0xFFFF0000)))]>,
               Requires<[IsARM, HasV6]> {
  let Inst{6-4} = 0b001;
}

// Alternate cases for PKHBT where identities eliminate some nodes.
def : ARMV6Pat<(or (and GPR:$src1, 0xFFFF), (and GPR:$src2, 0xFFFF0000)),
               (PKHBT GPR:$src1, GPR:$src2, 0)>;
def : ARMV6Pat<(or (and GPR:$src1, 0xFFFF), (shl GPR:$src2, imm16_31:$shamt)),
               (PKHBT GPR:$src1, GPR:$src2, imm16_31:$shamt)>;


def PKHTB : AMiscA1I<0b01101000, (outs GPR:$dst),
                                 (ins GPR:$src1, GPR:$src2, i32imm:$shamt),
               IIC_iALUsi, "pkhtb", " $dst, $src1, $src2, ASR $shamt",
               [(set GPR:$dst, (or (and GPR:$src1, 0xFFFF0000),
                                   (and (sra GPR:$src2, imm16_31:$shamt),
                                        0xFFFF)))]>, Requires<[IsARM, HasV6]> {
  let Inst{6-4} = 0b101;
}

// Alternate cases for PKHTB where identities eliminate some nodes.  Note that
// a shift amount of 0 is *not legal* here, it is PKHBT instead.
def : ARMV6Pat<(or (and GPR:$src1, 0xFFFF0000), (srl GPR:$src2, (i32 16))),
               (PKHTB GPR:$src1, GPR:$src2, 16)>;
def : ARMV6Pat<(or (and GPR:$src1, 0xFFFF0000),
                   (and (srl GPR:$src2, imm1_15:$shamt), 0xFFFF)),
               (PKHTB GPR:$src1, GPR:$src2, imm1_15:$shamt)>;

//===----------------------------------------------------------------------===//
//  Comparison Instructions...
//

defm CMP  : AI1_cmp_irs<0b1010, "cmp",
                        BinOpFrag<(ARMcmp node:$LHS, node:$RHS)>>;
defm CMN  : AI1_cmp_irs<0b1011, "cmn",
                        BinOpFrag<(ARMcmp node:$LHS,(ineg node:$RHS))>>;

// Note that TST/TEQ don't set all the same flags that CMP does!
defm TST  : AI1_cmp_irs<0b1000, "tst",
                        BinOpFrag<(ARMcmpZ (and node:$LHS, node:$RHS), 0)>, 1>;
defm TEQ  : AI1_cmp_irs<0b1001, "teq",
                        BinOpFrag<(ARMcmpZ (xor node:$LHS, node:$RHS), 0)>, 1>;

defm CMPz  : AI1_cmp_irs<0b1010, "cmp",
                         BinOpFrag<(ARMcmpZ node:$LHS, node:$RHS)>>;
defm CMNz  : AI1_cmp_irs<0b1011, "cmn",
                         BinOpFrag<(ARMcmpZ node:$LHS,(ineg node:$RHS))>>;

def : ARMPat<(ARMcmp GPR:$src, so_imm_neg:$imm),
             (CMNri  GPR:$src, so_imm_neg:$imm)>;

def : ARMPat<(ARMcmpZ GPR:$src, so_imm_neg:$imm),
             (CMNri  GPR:$src, so_imm_neg:$imm)>;


// Conditional moves
// FIXME: should be able to write a pattern for ARMcmov, but can't use
// a two-value operand where a dag node expects two operands. :( 
def MOVCCr : AI1<0b1101, (outs GPR:$dst), (ins GPR:$false, GPR:$true), DPFrm,
                IIC_iCMOVr, "mov", " $dst, $true",
      [/*(set GPR:$dst, (ARMcmov GPR:$false, GPR:$true, imm:$cc, CCR:$ccr))*/]>,
                RegConstraint<"$false = $dst">, UnaryDP;

def MOVCCs : AI1<0b1101, (outs GPR:$dst),
                        (ins GPR:$false, so_reg:$true), DPSoRegFrm, IIC_iCMOVsr,
                "mov", " $dst, $true",
   [/*(set GPR:$dst, (ARMcmov GPR:$false, so_reg:$true, imm:$cc, CCR:$ccr))*/]>,
                RegConstraint<"$false = $dst">, UnaryDP;

def MOVCCi : AI1<0b1101, (outs GPR:$dst),
                        (ins GPR:$false, so_imm:$true), DPFrm, IIC_iCMOVi,
                "mov", " $dst, $true",
   [/*(set GPR:$dst, (ARMcmov GPR:$false, so_imm:$true, imm:$cc, CCR:$ccr))*/]>,
                RegConstraint<"$false = $dst">, UnaryDP {
    let Inst{25} = 1;
}


//===----------------------------------------------------------------------===//
// TLS Instructions
//

// __aeabi_read_tp preserves the registers r1-r3.
let isCall = 1,
  Defs = [R0, R12, LR, CPSR] in {
  def TPsoft : ABXI<0b1011, (outs), (ins), IIC_Br,
               "bl __aeabi_read_tp",
               [(set R0, ARMthread_pointer)]>;
}

//===----------------------------------------------------------------------===//
// SJLJ Exception handling intrinsics
//   eh_sjlj_setjmp() is an instruction sequence to store the return
//   address and save #0 in R0 for the non-longjmp case.
//   Since by its nature we may be coming from some other function to get
//   here, and we're using the stack frame for the containing function to
//   save/restore registers, we can't keep anything live in regs across
//   the eh_sjlj_setjmp(), else it will almost certainly have been tromped upon
//   when we get here from a longjmp(). We force everthing out of registers
//   except for our own input by listing the relevant registers in Defs. By
//   doing so, we also cause the prologue/epilogue code to actively preserve
//   all of the callee-saved resgisters, which is exactly what we want.
let Defs = 
  [ R0,  R1,  R2,  R3,  R4,  R5,  R6,  R7,  R8,  R9,  R10, R11, R12, LR,  D0,
    D1,  D2,  D3,  D4,  D5,  D6,  D7,  D8,  D9,  D10, D11, D12, D13, D14, D15,
    D16, D17, D18, D19, D20, D21, D22, D23, D24, D25, D26, D27, D28, D29, D30,
    D31 ] in {
  def Int_eh_sjlj_setjmp : XI<(outs), (ins GPR:$src),
                               AddrModeNone, SizeSpecial, IndexModeNone,
                               Pseudo, NoItinerary,
                               "str sp, [$src, #+8] @ eh_setjmp begin\n\t"
                               "add r12, pc, #8\n\t"
                               "str r12, [$src, #+4]\n\t"
                               "mov r0, #0\n\t"
                               "add pc, pc, #0\n\t"
                               "mov r0, #1 @ eh_setjmp end", "",
                               [(set R0, (ARMeh_sjlj_setjmp GPR:$src))]>;
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
def MOVi2pieces : AI1x2<(outs GPR:$dst), (ins so_imm2part:$src), 
                         Pseudo, IIC_iMOVi,
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
def : ARMPat<(ARMcall texternalsym:$func), (BL texternalsym:$func)>,
      Requires<[IsARM, IsNotDarwin]>;
def : ARMPat<(ARMcall texternalsym:$func), (BLr9 texternalsym:$func)>,
      Requires<[IsARM, IsDarwin]>;

// zextload i1 -> zextload i8
def : ARMPat<(zextloadi1 addrmode2:$addr),  (LDRB addrmode2:$addr)>;

// extload -> zextload
def : ARMPat<(extloadi1  addrmode2:$addr),  (LDRB addrmode2:$addr)>;
def : ARMPat<(extloadi8  addrmode2:$addr),  (LDRB addrmode2:$addr)>;
def : ARMPat<(extloadi16 addrmode3:$addr),  (LDRH addrmode3:$addr)>;

def : ARMPat<(extloadi8  addrmodepc:$addr), (PICLDRB addrmodepc:$addr)>;
def : ARMPat<(extloadi16 addrmodepc:$addr), (PICLDRH addrmodepc:$addr)>;

// smul* and smla*
def : ARMV5TEPat<(mul (sra (shl GPR:$a, (i32 16)), (i32 16)),
                      (sra (shl GPR:$b, (i32 16)), (i32 16))),
                 (SMULBB GPR:$a, GPR:$b)>;
def : ARMV5TEPat<(mul sext_16_node:$a, sext_16_node:$b),
                 (SMULBB GPR:$a, GPR:$b)>;
def : ARMV5TEPat<(mul (sra (shl GPR:$a, (i32 16)), (i32 16)),
                      (sra GPR:$b, (i32 16))),
                 (SMULBT GPR:$a, GPR:$b)>;
def : ARMV5TEPat<(mul sext_16_node:$a, (sra GPR:$b, (i32 16))),
                 (SMULBT GPR:$a, GPR:$b)>;
def : ARMV5TEPat<(mul (sra GPR:$a, (i32 16)),
                      (sra (shl GPR:$b, (i32 16)), (i32 16))),
                 (SMULTB GPR:$a, GPR:$b)>;
def : ARMV5TEPat<(mul (sra GPR:$a, (i32 16)), sext_16_node:$b),
                (SMULTB GPR:$a, GPR:$b)>;
def : ARMV5TEPat<(sra (mul GPR:$a, (sra (shl GPR:$b, (i32 16)), (i32 16))),
                      (i32 16)),
                 (SMULWB GPR:$a, GPR:$b)>;
def : ARMV5TEPat<(sra (mul GPR:$a, sext_16_node:$b), (i32 16)),
                 (SMULWB GPR:$a, GPR:$b)>;

def : ARMV5TEPat<(add GPR:$acc,
                      (mul (sra (shl GPR:$a, (i32 16)), (i32 16)),
                           (sra (shl GPR:$b, (i32 16)), (i32 16)))),
                 (SMLABB GPR:$a, GPR:$b, GPR:$acc)>;
def : ARMV5TEPat<(add GPR:$acc,
                      (mul sext_16_node:$a, sext_16_node:$b)),
                 (SMLABB GPR:$a, GPR:$b, GPR:$acc)>;
def : ARMV5TEPat<(add GPR:$acc,
                      (mul (sra (shl GPR:$a, (i32 16)), (i32 16)),
                           (sra GPR:$b, (i32 16)))),
                 (SMLABT GPR:$a, GPR:$b, GPR:$acc)>;
def : ARMV5TEPat<(add GPR:$acc,
                      (mul sext_16_node:$a, (sra GPR:$b, (i32 16)))),
                 (SMLABT GPR:$a, GPR:$b, GPR:$acc)>;
def : ARMV5TEPat<(add GPR:$acc,
                      (mul (sra GPR:$a, (i32 16)),
                           (sra (shl GPR:$b, (i32 16)), (i32 16)))),
                 (SMLATB GPR:$a, GPR:$b, GPR:$acc)>;
def : ARMV5TEPat<(add GPR:$acc,
                      (mul (sra GPR:$a, (i32 16)), sext_16_node:$b)),
                 (SMLATB GPR:$a, GPR:$b, GPR:$acc)>;
def : ARMV5TEPat<(add GPR:$acc,
                      (sra (mul GPR:$a, (sra (shl GPR:$b, (i32 16)), (i32 16))),
                           (i32 16))),
                 (SMLAWB GPR:$a, GPR:$b, GPR:$acc)>;
def : ARMV5TEPat<(add GPR:$acc,
                      (sra (mul GPR:$a, sext_16_node:$b), (i32 16))),
                 (SMLAWB GPR:$a, GPR:$b, GPR:$acc)>;

//===----------------------------------------------------------------------===//
// Thumb Support
//

include "ARMInstrThumb.td"

//===----------------------------------------------------------------------===//
// Thumb2 Support
//

include "ARMInstrThumb2.td"

//===----------------------------------------------------------------------===//
// Floating Point Support
//

include "ARMInstrVFP.td"

//===----------------------------------------------------------------------===//
// Advanced SIMD (NEON) Support
//

include "ARMInstrNEON.td"
