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
defm DADD : XForm_28r<59, 2, (outs f8rc:$RST), (ins f8rc:$RA, f8rc:$RB),
                      "dadd",  "$RST, $RA, $RB", IIC_FPGeneral, []>;

defm DADDQ : XForm_28r<63, 2, (outs fpairrc:$RST), (ins fpairrc:$RA, fpairrc:$RB),
                       "daddq",  "$RST, $RA, $RB", IIC_FPGeneral, []>;

defm DSUB : XForm_28r<59, 514, (outs f8rc:$RST), (ins f8rc:$RA, f8rc:$RB),
                      "dsub",  "$RST, $RA, $RB", IIC_FPGeneral, []>;

defm DSUBQ : XForm_28r<63, 514, (outs fpairrc:$RST), (ins fpairrc:$RA, fpairrc:$RB),
                       "dsubq",  "$RST, $RA, $RB", IIC_FPGeneral, []>;
}

