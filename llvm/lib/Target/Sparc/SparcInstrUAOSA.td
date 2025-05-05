//=== SparcInstrUAOSA.td - UltraSPARC/Oracle SPARC Architecture extensions ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains instruction formats, definitions and patterns needed for
// UA 2005 instructions on SPARC.
//===----------------------------------------------------------------------===//

class UA2005RegWin<string asmstr, bits<5> fcn>
    : F3_1<2, 0b110001, (outs), (ins), asmstr, []> {
    let rd = fcn;
    let rs1 = 0;
    let rs2 = 0;
}

// UltraSPARC Architecture 2005 Instructions
let Predicates = [HasUA2005] in {
let hasSideEffects = 1 in {
def ALLCLEAN : UA2005RegWin<"allclean", 0b00010>;
def INVALW   : UA2005RegWin<"invalw", 0b00101>;
def NORMALW  : UA2005RegWin<"normalw", 0b00100>;
def OTHERW   : UA2005RegWin<"otherw", 0b00011>;
}
} // Predicates = [HasUA2005]
