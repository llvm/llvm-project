//===-- PPCInstrDFP.td - PowerPC Decimal Floating Point ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the PowerPC Decimal Floating Point (DFP) instructions.
//
//===----------------------------------------------------------------------===//

// We provide no scheduling info for the DFP instructions.
// While they are not pseudo instructions we don't intend on scheduling them.
let hasNoSchedulingInfo = 1 in {
let mayRaiseFPException = 1, hasSideEffects = 0 in {

let isCommutable = 1 in {
defm DADD : XForm_28r<59, 2, (outs f8rc:$RST), (ins f8rc:$RA, f8rc:$RB),
                      "dadd",  "$RST, $RA, $RB", IIC_FPGeneral, []>;

defm DADDQ : XForm_28r<63, 2, (outs fpairrc:$RST), (ins fpairrc:$RA, fpairrc:$RB),
                       "daddq",  "$RST, $RA, $RB", IIC_FPGeneral, []>;
}

defm DSUB : XForm_28r<59, 514, (outs f8rc:$RST), (ins f8rc:$RA, f8rc:$RB),
                      "dsub",  "$RST, $RA, $RB", IIC_FPGeneral, []>;

defm DSUBQ : XForm_28r<63, 514, (outs fpairrc:$RST), (ins fpairrc:$RA, fpairrc:$RB),
                       "dsubq",  "$RST, $RA, $RB", IIC_FPGeneral, []>;

let isCommutable = 1 in {
defm DMUL : XForm_28r<59, 34, (outs f8rc:$RST), (ins f8rc:$RA, f8rc:$RB),
                      "dmul",  "$RST, $RA, $RB", IIC_FPGeneral, []>;

defm DMULQ : XForm_28r<63, 34, (outs fpairrc:$RST), (ins fpairrc:$RA, fpairrc:$RB),
                       "dmulq",  "$RST, $RA, $RB", IIC_FPGeneral, []>;
}

defm DDIV : XForm_28r<59, 546, (outs f8rc:$RST), (ins f8rc:$RA, f8rc:$RB),
                      "ddiv",  "$RST, $RA, $RB", IIC_FPGeneral, []>;

defm DDIVQ : XForm_28r<63, 546, (outs fpairrc:$RST), (ins fpairrc:$RA, fpairrc:$RB),
                       "ddivq",  "$RST, $RA, $RB", IIC_FPGeneral, []>;

let isCompare = 1 in {
  def DCMPU : XForm_17<59, 642, (outs crrc:$BF), (ins f8rc:$RA, f8rc:$RB),
                       "dcmpu $BF, $RA, $RB", IIC_FPCompare>;

  def DCMPUQ : XForm_17<63, 642, (outs crrc:$BF), (ins fpairrc:$RA, fpairrc:$RB),
                        "dcmpuq $BF, $RA, $RB", IIC_FPCompare>;

  def DCMPO : XForm_17<59, 130, (outs crrc:$BF), (ins f8rc:$RA, f8rc:$RB),
                       "dcmpo $BF, $RA, $RB", IIC_FPCompare>;

  def DCMPOQ : XForm_17<63, 130, (outs crrc:$BF), (ins fpairrc:$RA, fpairrc:$RB),
                        "dcmpoq $BF, $RA, $RB", IIC_FPCompare>;
}

// 5.6.4 DFP Quantum Adjustment Instructions
defm DQUAI: Z23Form_TE5_FRTB5_RMC2r<59, 67, (outs f8rc:$FRT),
                                    (ins s5imm:$TE, f8rc:$FRB, u2imm:$RMC),
                                    "dquai", "$TE, $FRT, $FRB, $RMC", []>;
defm DQUAIQ: Z23Form_TE5_FRTB5_RMC2r<63, 67, (outs fpairrc:$FRT),
                                     (ins s5imm:$TE, fpairrc:$FRB, u2imm:$RMC),
                                     "dquaiq", "$TE, $FRT, $FRB, $RMC", []>;
defm DQUA: Z23Form_FRTAB5_RMC2r<59, 3, (outs f8rc:$FRT),
                                (ins f8rc:$FRA, f8rc:$FRB, u2imm:$RMC),
                                "dqua", "$FRT, $FRA, $FRB, $RMC", []>;
defm DQUAQ: Z23Form_FRTAB5_RMC2r<63, 3, (outs fpairrc:$FRT),
                                 (ins fpairrc:$FRA, fpairrc:$FRB, u2imm:$RMC),
                                 "dquaq", "$FRT, $FRA, $FRB, $RMC", []>;
defm DRRND: Z23Form_FRTAB5_RMC2r<59, 35, (outs f8rc:$FRT),
                                (ins f8rc:$FRA, f8rc:$FRB, u2imm:$RMC),
                                "drrnd", "$FRT, $FRA, $FRB, $RMC", []>;
defm DRRNDQ: Z23Form_FRTAB5_RMC2r<63, 35, (outs fpairrc:$FRT),
                                 (ins f8rc:$FRA, fpairrc:$FRB, u2imm:$RMC),
                                 "drrndq", "$FRT, $FRA, $FRB, $RMC", []>;
defm DRINTX: Z23Form_FRTB5_R1_RMC2r<59, 99, (outs f8rc:$FRT),
                                    (ins u1imm:$R, f8rc:$FRB, u2imm:$RMC),
                                    "drintx", "$R, $FRT, $FRB, $RMC", []>;
defm DRINTXQ: Z23Form_FRTB5_R1_RMC2r<63, 99, (outs fpairrc:$FRT),
                                     (ins u1imm:$R, fpairrc:$FRB, u2imm:$RMC),
                                     "drintxq", "$R, $FRT, $FRB, $RMC", []>;
defm DRINTN: Z23Form_FRTB5_R1_RMC2r<59, 227, (outs f8rc:$FRT),
                                    (ins u1imm:$R, f8rc:$FRB, u2imm:$RMC),
                                    "drintn", "$R, $FRT, $FRB, $RMC", []>;
defm DRINTNQ: Z23Form_FRTB5_R1_RMC2r<63, 227, (outs fpairrc:$FRT),
                                     (ins u1imm:$R, fpairrc:$FRB, u2imm:$RMC),
                                     "drintnq", "$R, $FRT, $FRB, $RMC", []>;

} // mayRaiseFPException
} // hasNoSchedulingInfo

