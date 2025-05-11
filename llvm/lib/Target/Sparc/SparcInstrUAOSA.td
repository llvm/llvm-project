//=== SparcInstrUAOSA.td - UltraSPARC/Oracle SPARC Architecture extensions ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains instruction formats, definitions and patterns needed for
// UA 2005 and UA 2007 instructions on SPARC.
//===----------------------------------------------------------------------===//

class UA2005RegWin<string asmstr, bits<5> fcn>
    : F3_1<2, 0b110001, (outs), (ins), asmstr, []> {
    let rd = fcn;
    let rs1 = 0;
    let rs2 = 0;
}

// Convenience template for 4-operand instructions
class FourOp<string OpcStr, bits<6> op3val, bits<4> op5val,
        RegisterClass RC>
      : F3_4<op3val, op5val, (outs RC:$rd), (ins RC:$rs1, RC:$rs2, RC:$rs3),
        !strconcat(OpcStr, " $rs1, $rs2, $rs3, $rd")>;

// UltraSPARC Architecture 2005 Instructions
let Predicates = [HasUA2005] in {
let hasSideEffects = 1 in {
def ALLCLEAN : UA2005RegWin<"allclean", 0b00010>;
def INVALW   : UA2005RegWin<"invalw", 0b00101>;
def NORMALW  : UA2005RegWin<"normalw", 0b00100>;
def OTHERW   : UA2005RegWin<"otherw", 0b00011>;
}
} // Predicates = [HasUA2005]

// UltraSPARC Architecture 2007 Instructions
let Predicates = [HasUA2007] in {
def FMADDS : FourOp<"fmadds", 0b110111, 0b0001, FPRegs>;
def FMADDD : FourOp<"fmaddd", 0b110111, 0b0010, DFPRegs>;
def FMSUBS : FourOp<"fmsubs", 0b110111, 0b0101, FPRegs>;
def FMSUBD : FourOp<"fmsubd", 0b110111, 0b0110, DFPRegs>;

def FNMADDS : FourOp<"fnmadds", 0b110111, 0b1101, FPRegs>;
def FNMADDD : FourOp<"fnmaddd", 0b110111, 0b1110, DFPRegs>;
def FNMSUBS : FourOp<"fnmsubs", 0b110111, 0b1001, FPRegs>;
def FNMSUBD : FourOp<"fnmsubd", 0b110111, 0b1010, DFPRegs>;
} // Predicates = [HasUA2007]
