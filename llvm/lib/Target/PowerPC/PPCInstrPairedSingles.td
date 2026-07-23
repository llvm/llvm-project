//===-- PPCInstrPairedSingles.td - PPC750CL Paired Singles -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the paired-singles instructions of the IBM PPC750CL
// family of processors.
//
// It's important to note that Paired Singles aren't actually independent
// registers. Instead each floating point register is actually like a pair
// of 32 bit registers. Regular floating point operations write to both, and
// ignore the second one. Only Paired Singles operations actually "see" the
// second one. It's like a 64+32-bit register, or 64+64-bit when it's a
// rename register.
//
//===----------------------------------------------------------------------===//

class PSForm_QD<bits<6> opcode, dag OOL, dag IOL, string asmstr,
                InstrItinClass itin, list<dag> pattern>
  : I<opcode, OOL, IOL, asmstr, itin>, MemriOp {
  bits<5>  FRT;
  bits<12> D;
  bits<5>  RA;
  bits<1>  W;
  bits<3>  QI;

  let Pattern = pattern;

  let Inst{6...10}  = FRT;
  let Inst{11...15} = RA;
  let Inst{16}      = W;
  let Inst{17...19} = QI;
  let Inst{20...31} = D;
}

class PSForm_QX<bits<6> xo, dag OOL, dag IOL, string asmstr,
                InstrItinClass itin, list<dag> pattern>
  : I<4, OOL, IOL, asmstr, itin> {
  bits<5> FRT;
  bits<5> RA;
  bits<5> RB;
  bits<1> W;
  bits<3> QI;

  let Pattern = pattern;

  let Inst{6...10}  = FRT;
  let Inst{11...15} = RA;
  let Inst{16...20} = RB;
  let Inst{21}      = W;
  let Inst{22...24} = QI;
  let Inst{25...30} = xo;
  let Inst{31}      = 0;
}

// NOTE: this is differrent than `dcbzl`, *this is* `dcbz_l` which is 750CL specific,
// has separate encodings, slightly different semantics, and requires a locked cache.
class PSForm_DCBZL<bits<10> xo, dag OOL, dag IOL, string asmstr,
                   InstrItinClass itin, list<dag> pattern>
  : I<4, OOL, IOL, asmstr, itin> {
  bits<5> RA;
  bits<5> RB;

  let Pattern = pattern;

  let Inst{6...10}  = 0;
  let Inst{11...15} = RA;
  let Inst{16...20} = RB;
  let Inst{21...30} = xo;
  let Inst{31}      = 0;
}

let DecoderNamespace = "PairedSingles", Predicates = [HasPairedSingles] in {

let mayLoad = 1, mayStore = 0, hasSideEffects = 0 in {
def PSQ_L   : PSForm_QD<56, (outs f8rc:$FRT),
                        (ins (psqdisp $D, $RA):$addr, u1imm:$W, u3imm:$QI),
                        "psq_l $FRT, $addr, $W, $QI", IIC_LdStLFD, []>;
def PSQ_LU  : PSForm_QD<57, (outs f8rc:$FRT, ptr_rc_nor0:$ea_result),
                        (ins (psqdisp $D, $RA):$addr, u1imm:$W, u3imm:$QI),
                        "psq_lu $FRT, $addr, $W, $QI", IIC_LdStLFDU, []>,
              RegConstraint<"$addr.reg = $ea_result">;
def PSQ_LX  : PSForm_QX<6, (outs f8rc:$FRT),
                        (ins (memrr $RA, $RB):$addr, u1imm:$W, u3imm:$QI),
                        "psq_lx $FRT, $addr, $W, $QI", IIC_LdStLFD, []>;
def PSQ_LUX : PSForm_QX<38, (outs f8rc:$FRT, ptr_rc_nor0:$ea_result),
                        (ins (memrr $RA, $RB):$addr, u1imm:$W, u3imm:$QI),
                        "psq_lux $FRT, $addr, $W, $QI", IIC_LdStLFDU, []>,
              RegConstraint<"$addr.ptrreg = $ea_result">;
}

let mayLoad = 0, mayStore = 1, hasSideEffects = 0 in {
def PSQ_ST  : PSForm_QD<60, (outs),
                        (ins f8rc:$FRT, (psqdisp $D, $RA):$dst, u1imm:$W,
                             u3imm:$QI),
                        "psq_st $FRT, $dst, $W, $QI", IIC_LdStSTFD, []>;
def PSQ_STU : PSForm_QD<61, (outs ptr_rc_nor0:$ea_res),
                        (ins f8rc:$FRT, (psqdisp $D, $RA):$dst, u1imm:$W,
                             u3imm:$QI),
                        "psq_stu $FRT, $dst, $W, $QI", IIC_LdStSTFDU, []>,
              RegConstraint<"$dst.reg = $ea_res">;
def PSQ_STX : PSForm_QX<7, (outs),
                        (ins f8rc:$FRT, (memrr $RA, $RB):$dst, u1imm:$W,
                             u3imm:$QI),
                        "psq_stx $FRT, $dst, $W, $QI", IIC_LdStSTFD, []>;
def PSQ_STUX : PSForm_QX<39, (outs ptr_rc_nor0:$ea_res),
                         (ins f8rc:$FRT, (memrr $RA, $RB):$dst, u1imm:$W,
                              u3imm:$QI),
                         "psq_stux $FRT, $dst, $W, $QI", IIC_LdStSTFDU, []>,
               RegConstraint<"$dst.ptrreg = $ea_res">;
}

let mayRaiseFPException = 1, hasSideEffects = 0 in {
let isCommutable = 1 in {
defm PS_ADD    : AForm_2r<4, 21, (outs f8rc:$FRT),
                          (ins f8rc:$FRA, f8rc:$FRB),
                          "ps_add", "$FRT, $FRA, $FRB", IIC_FPGeneral, []>;
defm PS_MUL    : AForm_3r<4, 25, (outs f8rc:$FRT),
                          (ins f8rc:$FRA, f8rc:$FRC),
                          "ps_mul", "$FRT, $FRA, $FRC", IIC_FPGeneral, []>;
defm PS_MADD   : AForm_1r<4, 29, (outs f8rc:$FRT),
                          (ins f8rc:$FRA, f8rc:$FRC, f8rc:$FRB),
                          "ps_madd", "$FRT, $FRA, $FRC, $FRB", IIC_FPFused,
                          []>;
defm PS_NMADD  : AForm_1r<4, 31, (outs f8rc:$FRT),
                          (ins f8rc:$FRA, f8rc:$FRC, f8rc:$FRB),
                          "ps_nmadd", "$FRT, $FRA, $FRC, $FRB", IIC_FPFused,
                          []>;
}
defm PS_SUB    : AForm_2r<4, 20, (outs f8rc:$FRT),
                          (ins f8rc:$FRA, f8rc:$FRB),
                          "ps_sub", "$FRT, $FRA, $FRB", IIC_FPGeneral, []>;
defm PS_DIV    : AForm_2r<4, 18, (outs f8rc:$FRT),
                          (ins f8rc:$FRA, f8rc:$FRB),
                          "ps_div", "$FRT, $FRA, $FRB", IIC_FPDivS, []>;
defm PS_MSUB   : AForm_1r<4, 28, (outs f8rc:$FRT),
                          (ins f8rc:$FRA, f8rc:$FRC, f8rc:$FRB),
                          "ps_msub", "$FRT, $FRA, $FRC, $FRB", IIC_FPFused,
                          []>;
defm PS_NMSUB  : AForm_1r<4, 30, (outs f8rc:$FRT),
                          (ins f8rc:$FRA, f8rc:$FRC, f8rc:$FRB),
                          "ps_nmsub", "$FRT, $FRA, $FRC, $FRB", IIC_FPFused,
                          []>;
defm PS_RES    : XForm_26r<4, 24, (outs f8rc:$RST), (ins f8rc:$RB),
                           "ps_res", "$RST, $RB", IIC_FPGeneral, []>;
defm PS_RSQRTE : XForm_26r<4, 26, (outs f8rc:$RST), (ins f8rc:$RB),
                           "ps_rsqrte", "$RST, $RB", IIC_FPGeneral, []>;

defm PS_MULS0  : AForm_3r<4, 12, (outs f8rc:$FRT),
                          (ins f8rc:$FRA, f8rc:$FRC),
                          "ps_muls0", "$FRT, $FRA, $FRC", IIC_FPGeneral, []>;
defm PS_MULS1  : AForm_3r<4, 13, (outs f8rc:$FRT),
                          (ins f8rc:$FRA, f8rc:$FRC),
                          "ps_muls1", "$FRT, $FRA, $FRC", IIC_FPGeneral, []>;
defm PS_MADDS0 : AForm_1r<4, 14, (outs f8rc:$FRT),
                          (ins f8rc:$FRA, f8rc:$FRC, f8rc:$FRB),
                          "ps_madds0", "$FRT, $FRA, $FRC, $FRB", IIC_FPFused,
                          []>;
defm PS_MADDS1 : AForm_1r<4, 15, (outs f8rc:$FRT),
                          (ins f8rc:$FRA, f8rc:$FRC, f8rc:$FRB),
                          "ps_madds1", "$FRT, $FRA, $FRC, $FRB", IIC_FPFused,
                          []>;
defm PS_SUM0   : AForm_1r<4, 10, (outs f8rc:$FRT),
                          (ins f8rc:$FRA, f8rc:$FRC, f8rc:$FRB),
                          "ps_sum0", "$FRT, $FRA, $FRC, $FRB", IIC_FPGeneral,
                          []>;
defm PS_SUM1   : AForm_1r<4, 11, (outs f8rc:$FRT),
                          (ins f8rc:$FRA, f8rc:$FRC, f8rc:$FRB),
                          "ps_sum1", "$FRT, $FRA, $FRC, $FRB", IIC_FPGeneral,
                          []>;
}

let hasSideEffects = 0 in {
defm PS_SEL     : AForm_1r<4, 23, (outs f8rc:$FRT),
                           (ins f8rc:$FRA, f8rc:$FRC, f8rc:$FRB),
                           "ps_sel", "$FRT, $FRA, $FRC, $FRB", IIC_FPGeneral,
                           []>;
defm PS_NEG     : XForm_26r<4, 40, (outs f8rc:$RST), (ins f8rc:$RB),
                            "ps_neg", "$RST, $RB", IIC_FPGeneral, []>;
defm PS_MR      : XForm_26r<4, 72, (outs f8rc:$RST), (ins f8rc:$RB),
                            "ps_mr", "$RST, $RB", IIC_FPGeneral, []>;
defm PS_NABS    : XForm_26r<4, 136, (outs f8rc:$RST), (ins f8rc:$RB),
                            "ps_nabs", "$RST, $RB", IIC_FPGeneral, []>;
defm PS_ABS     : XForm_26r<4, 264, (outs f8rc:$RST), (ins f8rc:$RB),
                            "ps_abs", "$RST, $RB", IIC_FPGeneral, []>;
defm PS_MERGE00 : XForm_28r<4, 528, (outs f8rc:$RST),
                            (ins f8rc:$RA, f8rc:$RB),
                            "ps_merge00", "$RST, $RA, $RB", IIC_FPGeneral, []>;
defm PS_MERGE01 : XForm_28r<4, 560, (outs f8rc:$RST),
                            (ins f8rc:$RA, f8rc:$RB),
                            "ps_merge01", "$RST, $RA, $RB", IIC_FPGeneral, []>;
defm PS_MERGE10 : XForm_28r<4, 592, (outs f8rc:$RST),
                            (ins f8rc:$RA, f8rc:$RB),
                            "ps_merge10", "$RST, $RA, $RB", IIC_FPGeneral, []>;
defm PS_MERGE11 : XForm_28r<4, 624, (outs f8rc:$RST),
                            (ins f8rc:$RA, f8rc:$RB),
                            "ps_merge11", "$RST, $RA, $RB", IIC_FPGeneral, []>;
}

let mayRaiseFPException = 1, hasSideEffects = 0 in {
def PS_CMPU0 : XForm_17<4, 0, (outs crrc:$BF), (ins f8rc:$RA, f8rc:$RB),
                        "ps_cmpu0 $BF, $RA, $RB", IIC_FPCompare>;
def PS_CMPO0 : XForm_17<4, 32, (outs crrc:$BF), (ins f8rc:$RA, f8rc:$RB),
                        "ps_cmpo0 $BF, $RA, $RB", IIC_FPCompare>;
def PS_CMPU1 : XForm_17<4, 64, (outs crrc:$BF), (ins f8rc:$RA, f8rc:$RB),
                        "ps_cmpu1 $BF, $RA, $RB", IIC_FPCompare>;
def PS_CMPO1 : XForm_17<4, 96, (outs crrc:$BF), (ins f8rc:$RA, f8rc:$RB),
                        "ps_cmpo1 $BF, $RA, $RB", IIC_FPCompare>;
}

let hasSideEffects = 1 in {
def DCBZ_L : PSForm_DCBZL<1014, (outs), (ins (memrr $RA, $RB):$addr),
                          "dcbz_l $addr", IIC_LdStDCBF, []>;
}

} // DecoderNamespace = "PairedSingles", Predicates = [HasPairedSingles]
