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

def uimm11 : RISCVUImmLeafOp<11>;

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

class QCIRVInstR<bits<4> func4, string opcodestr>
    : RVInstR<{0b000, func4}, 0b011, OPC_CUSTOM_0, (outs GPRNoX0:$rd),
              (ins GPRNoX0:$rs1), opcodestr, "$rd, $rs1"> {
  let rs2 = 0;
}

class QCIRVInstRR<bits<5> func5, DAGOperand InTyRs1, string opcodestr>
    : RVInstR<{0b00, func5}, 0b011, OPC_CUSTOM_0, (outs GPRNoX0:$rd),
              (ins InTyRs1:$rs1, GPRNoX0:$rs2), opcodestr, "$rd, $rs1, $rs2">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class QCISELECTIICC<bits<3> funct3, string opcodestr>
    : RVInstR4<0b00, funct3, OPC_CUSTOM_2, (outs GPRNoX0:$rd_wb),
               (ins GPRNoX0:$rd, GPRNoX0:$rs1, simm5:$simm1, simm5:$simm2),
               opcodestr, "$rd, $rs1, $simm1, $simm2"> {
  let Constraints = "$rd = $rd_wb";
  bits<5> simm1;
  bits<5> simm2;

  let rs3 = simm2;
  let rs2 = simm1;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class QCISELECTICC<bits<3> funct3, string opcodestr>
    : RVInstR4<0b01, funct3, OPC_CUSTOM_2, (outs GPRNoX0:$rd_wb),
               (ins GPRNoX0:$rd, GPRNoX0:$rs1, GPRNoX0:$rs2, simm5:$simm2),
               opcodestr, "$rd, $rs1, $rs2, $simm2"> {
  let Constraints = "$rd = $rd_wb";
  bits<5> simm2;

  let rs3 = simm2;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class QCISELECTCCI<bits<3> funct3, string opcodestr>
    : RVInstR4<0b10, funct3, OPC_CUSTOM_2, (outs GPRNoX0:$rd_wb),
               (ins GPRNoX0:$rd, simm5:$imm, GPRNoX0:$rs2, GPRNoX0:$rs3),
               opcodestr, "$rd, $imm, $rs2, $rs3"> {
  let Constraints = "$rd = $rd_wb";
  bits<5> imm;

  let rs1 = imm;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class QCISELECTICCI<bits<3> funct3, string opcodestr>
    : RVInstR4<0b11, funct3, OPC_CUSTOM_2, (outs GPRNoX0:$rd_wb),
               (ins GPRNoX0:$rd, simm5:$imm, GPRNoX0:$rs2, simm5:$simm2),
               opcodestr, "$rd, $imm, $rs2, $simm2"> {
  let Constraints = "$rd = $rd_wb";
  bits<5> imm;
  bits<5> simm2;

  let rs3 = simm2;
  let rs1 = imm;
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasVendorXqcicsr, IsRV32], DecoderNamespace = "Xqcicsr" in {
let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in {
  def QC_CSRRWR : RVInstR<0b1000110, 0b000, OPC_SYSTEM, (outs GPR:$rd),
                          (ins GPR:$rs1, GPRNoX0:$rs2), "qc.csrrwr",
                          "$rd, $rs1, $rs2">;

  def QC_CSRRWRI : RVInstR<0b1000111, 0b000, OPC_SYSTEM, (outs GPR:$rd),
                           (ins uimm5:$rs1, GPRNoX0:$rs2), "qc.csrrwri",
                           "$rd, $rs1, $rs2">;
} // hasSideEffects = 1, mayLoad = 0, mayStore = 0
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

let Predicates = [HasVendorXqcia, IsRV32], DecoderNamespace = "Xqcia" in {
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
  def QC_SLASAT : QCIRVInstRR<0b01010, GPRNoX0, "qc.slasat">;
  def QC_SLLSAT : QCIRVInstRR<0b01100, GPRNoX0, "qc.sllsat">;
  def QC_ADDSAT : QCIRVInstRR<0b01110, GPRNoX0, "qc.addsat">;
  def QC_ADDUSAT : QCIRVInstRR<0b01111, GPRNoX0, "qc.addusat">;
  def QC_SUBSAT : QCIRVInstRR<0b10000, GPRNoX0, "qc.subsat">;
  def QC_SUBUSAT : QCIRVInstRR<0b10001, GPRNoX0, "qc.subusat">;

  def QC_WRAP : QCIRVInstRR<0b10010, GPR, "qc.wrap">;
  def QC_WRAPI : RVInstI<0b000, OPC_CUSTOM_0, (outs GPRNoX0:$rd),
                         (ins GPRNoX0:$rs1, uimm11:$imm11), "qc.wrapi",
                         "$rd, $rs1, $imm11"> {
    bits<11> imm11;

    let imm12 = {0b0, imm11};
  }

  def QC_NORM : QCIRVInstR<0b0111, "qc.norm">;
  def QC_NORMU : QCIRVInstR<0b1000, "qc.normu">;
  def QC_NORMEU : QCIRVInstR<0b1001, "qc.normeu">;
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0
} // Predicates = [HasVendorXqcia, IsRV32], DecoderNamespace = "Xqcia"

let Predicates = [HasVendorXqcics, IsRV32], DecoderNamespace = "Xqcics" in {
  def QC_SELECTIIEQ : QCISELECTIICC <0b010, "qc.selectiieq">;
  def QC_SELECTIINE : QCISELECTIICC <0b011, "qc.selectiine">;
  def QC_SELECTIEQ  : QCISELECTICC  <0b010, "qc.selectieq">;
  def QC_SELECTINE  : QCISELECTICC  <0b011, "qc.selectine">;
  def QC_SELECTEQI  : QCISELECTCCI  <0b010, "qc.selecteqi">;
  def QC_SELECTNEI  : QCISELECTCCI  <0b011, "qc.selectnei">;
  def QC_SELECTIEQI : QCISELECTICCI <0b010, "qc.selectieqi">;
  def QC_SELECTINEI : QCISELECTICCI <0b011, "qc.selectinei">;
} // Predicates = [HasVendorXqcics, IsRV32], DecoderNamespace = "Xqcics"
