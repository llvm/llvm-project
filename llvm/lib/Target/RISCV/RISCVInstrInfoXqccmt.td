//===-- RISCVInstrInfoXqccmt.td ----------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes Qualcomm's Xqccmt extension.
//
// Xqccmt is broadly equivalent to (and incompatible with) Zcmt except the
// following change:
//
// - qc.cm.jalt uses bit 0 of the jump table entry as metadata to select the
//   link register: bit 0 = 0 saves the return address in ra (x1), bit 0 = 1
//   saves it in t0 (x5), the alternate link register. The jump target always
//   has bit 0 cleared.
//
// The instruction encoding is identical to cm.jt / cm.jalt from Zcmt.
// Xqccmt and Zcmt are mutually exclusive and cannot both be enabled.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let DecoderNamespace = "Xqccmt", Predicates = [HasVendorXqccmt],
    hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {

def QC_CM_JT : RVInst16CJ<0b101, OPC_C2, (outs), (ins uimm5:$index),
                          "qc.cm.jt", "$index"> {
  bits<5> index;

  let Inst{12-7} = 0b000000;
  let Inst{6-2} = index;
}

// qc.cm.jalt may write to either ra (x1) or t0 (x5) at runtime, depending
// on bit 0 of the jump table entry. Both registers are listed as Defs for
// conservative correctness.
let Defs = [X1, X5] in
def QC_CM_JALT : RVInst16CJ<0b101, OPC_C2, (outs), (ins uimm8ge32:$index),
                            "qc.cm.jalt", "$index"> {
  bits<8> index;

  let Inst{12-10} = 0b000;
  let Inst{9-2} = index;
}

} // DecoderNamespace = "Xqccmt", Predicates = [HasVendorXqccmt]
