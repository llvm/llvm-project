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

// UltraSPARC Architecture 2005 Instructions
let Predicates = [HasUA2005] in {
let hasSideEffects = 1 in
def ALLCLEAN : InstSP<(outs), (ins), "allclean", []> {
    let op = 2;
    let Inst{29-19} = 0b00010110001;
    let Inst{18-0} = 0;
}
def INVALW : InstSP<(outs), (ins), "invalw", []> {
    let op = 2;
    let Inst{29-19} = 0b00101110001;
    let Inst{18-0} = 0;
}
def NORMALW : InstSP<(outs), (ins), "normalw", []> {
    let op = 2;
    let Inst{29-19} = 0b00100110001;
    let Inst{18-0} = 0;
}
def OTHERW : InstSP<(outs), (ins), "otherw", []> {
    let op = 2;
    let Inst{29-19} = 0b00011110001;
    let Inst{18-0} = 0;
}
} // Predicates = [HasUA2005]
