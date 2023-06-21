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
}
}

