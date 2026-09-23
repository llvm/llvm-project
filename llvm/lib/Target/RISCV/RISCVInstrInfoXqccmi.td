//===-- RISCVInstrInfoXqccmi.td ---------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes Qualcomm's Xqccmi extension.
//
// Xqccmi provides qc.cm.ilut, a compressed instruction which executes one or two
// instructions from an instruction lookup table. The encoding overlaps with
// c.fld, so Xqccmi and Zcd are mutually exclusive.
//
//===----------------------------------------------------------------------===//

let DecoderNamespace = "Xqccmi", Predicates = [HasVendorXqccmi, IsRV32],
    hasSideEffects = 1, mayLoad = 1, mayStore = 0 in {

def QC_CM_ILUT : RVInst16<(outs), (ins uimm11:$ilut_index),
                          "qc.cm.ilut", "$ilut_index", [],
                          InstFormatOther> {
  bits<11> ilut_index;

  let Inst{15-13} = 0b001;
  let Inst{12-2} = ilut_index;
  let Inst{1-0} = 0b00;
}

} // DecoderNamespace = "Xqccmi", Predicates = [HasVendorXqccmi, IsRV32]
