//===---------------- RISCVInstrInfoXQci.td ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the vendor extensions defined by QUALCOMM.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operand and SDNode transformation definitions.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction Formats
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction Class Templates
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in {
class QCILoad_ScaleIdx<bits<4> func4, string opcodestr>
    : RVInstRBase<0b111, OPC_CUSTOM_0,
                  (outs GPR:$rd), (ins GPRMem:$rs1, GPRNoX0:$rs2, uimm3:$shamt),
                  opcodestr, "$rd, $rs1, $rs2, $shamt"> {
  bits<3> shamt;
  let Inst{31-28} = func4;
  let Inst{27-25} = shamt;
}
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in {
// rd corresponds to the source for the store 'rs3' described in the spec.
class QCIStore_ScaleIdx<bits<4> func4, string opcodestr>
    : RVInstRBase<0b110, OPC_CUSTOM_1, (outs),
                  (ins GPR:$rd, GPRMem:$rs1, GPRNoX0:$rs2, uimm3:$shamt),
                  opcodestr, "$rd, $rs1, $rs2, $shamt"> {
  bits<3> shamt;
  let Inst{31-28} = func4;
  let Inst{27-25} = shamt;
}
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasVendorXqcicsr, IsRV32], DecoderNamespace = "Xqcicsr" in {
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
  def QC_CSRRWR : RVInstR<0b1000110, 0b000, OPC_SYSTEM, (outs GPR:$rd),
                          (ins GPR:$rs1, GPRNoX0:$rs2), "qc.csrrwr",
                          "$rd, $rs1, $rs2">;

  def QC_CSRRWRI : RVInstR<0b1000111, 0b000, OPC_SYSTEM, (outs GPR:$rd),
                           (ins uimm5:$rs1, GPRNoX0:$rs2), "qc.csrrwri",
                           "$rd, $rs1, $rs2">;
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0
} // Predicates = [HasVendorXqcicsr, IsRV32], DecoderNamespace = "Xqcicsr"

let Predicates = [HasVendorXqcisls, IsRV32], DecoderNamespace = "Xqcisls" in {
  def  QC_LRB  : QCILoad_ScaleIdx<0b1000, "qc.lrb">;
  def  QC_LRH  : QCILoad_ScaleIdx<0b1001, "qc.lrh">;
  def  QC_LRW  : QCILoad_ScaleIdx<0b1010, "qc.lrw">;
  def  QC_LRBU : QCILoad_ScaleIdx<0b1011, "qc.lrbu">;
  def  QC_LRHU : QCILoad_ScaleIdx<0b1100, "qc.lrhu">;

  def  QC_SRB  : QCIStore_ScaleIdx<0b1101, "qc.srb">;
  def  QC_SRH  : QCIStore_ScaleIdx<0b1110, "qc.srh">;
  def  QC_SRW  : QCIStore_ScaleIdx<0b1111, "qc.srw">;
} // Predicates = [HasVendorXqcisls, IsRV32], DecoderNamespace = "Xqcisls"
