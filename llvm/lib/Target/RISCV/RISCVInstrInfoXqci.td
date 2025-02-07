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

def uimm5nonzero : RISCVOp<XLenVT>,
                   ImmLeaf<XLenVT, [{return (Imm != 0) && isUInt<5>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<5, "NonZero">;
  let DecoderMethod = "decodeUImmNonZeroOperand<5>";
  let OperandType = "OPERAND_UIMM5_NONZERO";
}

def uimm5gt3 : RISCVOp<XLenVT>, ImmLeaf<XLenVT,
  [{return (Imm > 3) && isUInt<5>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<5, "GT3">;
  let DecoderMethod = "decodeUImmOperand<5>";
  let OperandType = "OPERAND_UIMM5_GT3";
}

def uimm10 : RISCVUImmLeafOp<10>;

def uimm11 : RISCVUImmLeafOp<11>;

def simm26 : RISCVSImmLeafOp<26>;

//===----------------------------------------------------------------------===//
// Instruction Formats
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction Class Templates
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in {
class QCILoad_ScaleIdx<bits<4> funct4, string opcodestr>
    : RVInstRBase<0b111, OPC_CUSTOM_0,
                  (outs GPR:$rd), (ins GPRMem:$rs1, GPRNoX0:$rs2, uimm3:$shamt),
                  opcodestr, "$rd, $rs1, $rs2, $shamt"> {
  bits<3> shamt;
  let Inst{31-28} = funct4;
  let Inst{27-25} = shamt;
}
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in {
// rd corresponds to the source for the store 'rs3' described in the spec.
class QCIStore_ScaleIdx<bits<4> funct4, string opcodestr>
    : RVInstRBase<0b110, OPC_CUSTOM_1, (outs),
                  (ins GPR:$rd, GPRMem:$rs1, GPRNoX0:$rs2, uimm3:$shamt),
                  opcodestr, "$rd, $rs1, $rs2, $shamt"> {
  bits<3> shamt;
  let Inst{31-28} = funct4;
  let Inst{27-25} = shamt;
}
}

class QCIRVInstR<bits<4> funct4, string opcodestr>
    : RVInstR<{0b000, funct4}, 0b011, OPC_CUSTOM_0, (outs GPRNoX0:$rd),
              (ins GPRNoX0:$rs1), opcodestr, "$rd, $rs1"> {
  let rs2 = 0;
}

class QCIRVInstRR<bits<5> funct5, DAGOperand InTyRs1, string opcodestr>
    : RVInstR<{0b00, funct5}, 0b011, OPC_CUSTOM_0, (outs GPRNoX0:$rd),
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

let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in
class QCILoadMultiple<bits<2> funct2, DAGOperand InTyRs2, string opcodestr>
    : RVInstRBase<0b111, OPC_CUSTOM_0, (outs GPRNoX0:$rd),
                  (ins GPR:$rs1, InTyRs2:$rs2, uimm7_lsb00:$imm),
                  opcodestr, "$rd, $rs2, ${imm}(${rs1})"> {
  bits<7> imm;
  let Inst{31-25} = {funct2, imm{6-2}};
}


// rd corresponds to the source for the store 'rs3' described in the spec.
let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in
class QCIStoreMultiple<bits<2> funct2, DAGOperand InTyRs2, string opcodestr>
    : RVInstRBase<0b111, OPC_CUSTOM_1, (outs),
                  (ins GPR:$rd, GPR:$rs1, InTyRs2:$rs2, uimm7_lsb00:$imm),
                  opcodestr, "$rd, $rs2, ${imm}(${rs1})"> {
  bits<7> imm;
  let Inst{31-25} = {funct2, imm{6-2}};
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class QCILICC<bits<3> funct3, bits<2> funct2, DAGOperand InTyRs2, string opcodestr>
    : RVInstRBase<funct3, OPC_CUSTOM_2, (outs GPRNoX0:$rd_wb),
                  (ins GPRNoX0:$rd, GPRNoX0:$rs1, InTyRs2:$rs2, simm5:$simm),
                  opcodestr, "$rd, $rs1, $rs2, $simm"> {
  let Constraints = "$rd = $rd_wb";
  bits<5> simm;

  let Inst{31-25} = {simm, funct2};
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class QCIMVCC<bits<3> funct3, string opcodestr>
    : RVInstR4<0b00, funct3, OPC_CUSTOM_2, (outs GPRNoX0:$rd),
               (ins GPRNoX0:$rs1, GPRNoX0:$rs2, GPRNoX0:$rs3),
               opcodestr, "$rd, $rs1, $rs2, $rs3">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class QCIMVCCI<bits<3> funct3, string opcodestr, DAGOperand immType>
    : RVInstR4<0b10, funct3, OPC_CUSTOM_2, (outs GPRNoX0:$rd),
               (ins GPRNoX0:$rs1, immType:$imm, GPRNoX0:$rs3),
               opcodestr, "$rd, $rs1, $imm, $rs3"> {
  bits<5> imm;

  let rs2 = imm;
}

let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in
class QCIRVInst16CI_RS1<bits<5> funct5, string OpcodeStr>
    : RVInst16CI<0b000, 0b10, (outs), (ins GPRNoX0:$rs1), OpcodeStr, "$rs1"> {
  bits<5> rs1;

  let Inst{12} = 0b1;
  let Inst{11-7} = rs1;
  let Inst{6-2} = funct5{4-0};
}

let hasSideEffects = 1 in
class QCIRVInst16CI_NONE<bits<5> funct5, string OpcodeStr>
    : RVInst16CI<0b000, 0b10, (outs), (ins), OpcodeStr, ""> {
  let Inst{12} = 0b1;
  let Inst{11-7} = funct5;
  let Inst{6-2} = 0b00100;
}

let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in
class QCIInt_IMM<bits<1> funct1, string opcodestr>
    : RVInstIBase<0b000, OPC_SYSTEM, (outs), (ins uimm10:$imm10), opcodestr,
                  "$imm10"> {
  bits<10> imm10;

  let rd = 0;
  let rs1 = imm10{4-0};
  let Inst{31-25} = {0b110011, funct1};
  let Inst{24-20} = imm10{9-5};
}

class QCIRVInstEIBase<bits<3> funct3, bits<2> funct2, dag outs,
                      dag ins, string opcodestr, string argstr>
    : RVInst48<outs, ins, opcodestr, argstr, [], InstFormatOther> {
  bits<5> rd;
  bits<5> rs1;
  bits<26> imm;

  let Inst{47-32} = imm{25-10};
  let Inst{31-30} = funct2;
  let Inst{29-20} = imm{9-0};
  let Inst{19-15} = rs1;
  let Inst{14-12} = funct3;
  let Inst{11-7} = rd;
  let Inst{6-0} = 0b0011111;
}

let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in
class QCIRVInstEILoad<bits<3> funct3, bits<2> funct2, string opcodestr>
    : QCIRVInstEIBase<funct3, funct2, (outs GPR:$rd),
                      (ins GPRMem:$rs1, simm26:$imm), opcodestr,
                      "$rd, ${imm}(${rs1})">;

class QCIRVInstESBase<bits<3> funct3, bits<2> funct2, dag outs,
                      dag ins, string opcodestr, string argstr>
    : RVInst48<outs, ins, opcodestr, argstr, [], InstFormatOther> {
  bits<5> rs1;
  bits<5> rs2;
  bits<26> imm;

  let Inst{47-32} = imm{25-10};
  let Inst{31-30} = funct2;
  let Inst{29-25} = imm{9-5};
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = funct3;
  let Inst{11-7} = imm{4-0};
  let Inst{6-0} = 0b0011111;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in
class QCIRVInstESStore<bits<3> funct3, bits<2> funct2, string opcodestr>
    : QCIRVInstESBase<funct3, funct2, (outs),
                      (ins GPRMem:$rs2, GPR:$rs1, simm26:$imm),
                      opcodestr, "$rs2, ${imm}(${rs1})">;

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

let Predicates = [HasVendorXqciac, IsRV32], DecoderNamespace = "Xqciac" in {
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
  def QC_C_MULIADD : RVInst16CL<0b001, 0b10, (outs GPRC:$rd_wb),
                               (ins GPRC:$rd, GPRC:$rs1, uimm5:$uimm),
                               "qc.c.muliadd", "$rd, $rs1, $uimm"> {
    let Constraints = "$rd = $rd_wb";
    bits<5> uimm;

    let Inst{12-10} = uimm{3-1};
    let Inst{6} = uimm{0};
    let Inst{5} = uimm{4};
  }

  def QC_MULIADD : RVInstI<0b110, OPC_CUSTOM_0, (outs GPRNoX0:$rd_wb),
                           (ins GPRNoX0:$rd, GPRNoX0:$rs1, simm12:$imm12),
                           "qc.muliadd", "$rd, $rs1, $imm12"> {
    let Constraints = "$rd = $rd_wb";
  }

  def QC_SHLADD : RVInstRBase<0b011, OPC_CUSTOM_0, (outs GPRNoX0:$rd),
                              (ins GPRNoX0:$rs1, GPRNoX0:$rs2, uimm5gt3:$shamt),
                              "qc.shladd", "$rd, $rs1, $rs2, $shamt"> {
    bits<5> shamt;

    let Inst{31-30} = 0b01;
    let Inst{29-25} = shamt;
  }

} // hasSideEffects = 0, mayLoad = 0, mayStore = 0
} // Predicates = [HasVendorXqciac, IsRV32], DecoderNamespace = "Xqciac"

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

let Predicates = [HasVendorXqcilsm, IsRV32], DecoderNamespace = "Xqcilsm" in {
    def QC_SWM : QCIStoreMultiple<0b00, GPRNoX0, "qc.swm">;
    def QC_SWMI : QCIStoreMultiple<0b01, uimm5nonzero, "qc.swmi">;
    def QC_SETWM : QCIStoreMultiple<0b10, GPRNoX0, "qc.setwm">;
    def QC_SETWMI : QCIStoreMultiple<0b11, uimm5nonzero, "qc.setwmi">;

    def QC_LWM : QCILoadMultiple<0b00, GPRNoX0, "qc.lwm">;
    def QC_LWMI : QCILoadMultiple<0b01, uimm5nonzero, "qc.lwmi">;
} // Predicates = [HasVendorXqcilsm, IsRV32], DecoderNamespace = "Xqcilsm"

let Predicates = [HasVendorXqcicli, IsRV32], DecoderNamespace = "Xqcicli" in {
  def QC_LIEQ    : QCILICC<0b000, 0b01, GPRNoX0, "qc.lieq">;
  def QC_LINE    : QCILICC<0b001, 0b01, GPRNoX0, "qc.line">;
  def QC_LILT    : QCILICC<0b100, 0b01, GPRNoX0, "qc.lilt">;
  def QC_LIGE    : QCILICC<0b101, 0b01, GPRNoX0, "qc.lige">;
  def QC_LILTU   : QCILICC<0b110, 0b01, GPRNoX0, "qc.liltu">;
  def QC_LIGEU   : QCILICC<0b111, 0b01, GPRNoX0, "qc.ligeu">;

  def QC_LIEQI   : QCILICC<0b000, 0b11, simm5, "qc.lieqi">;
  def QC_LINEI   : QCILICC<0b001, 0b11, simm5, "qc.linei">;
  def QC_LILTI   : QCILICC<0b100, 0b11, simm5, "qc.lilti">;
  def QC_LIGEI   : QCILICC<0b101, 0b11, simm5, "qc.ligei">;
  def QC_LILTUI  : QCILICC<0b110, 0b11, uimm5, "qc.liltui">;
  def QC_LIGEUI  : QCILICC<0b111, 0b11, uimm5, "qc.ligeui">;
} // Predicates = [HasVendorXqcicli, IsRV32], DecoderNamespace = "Xqcicli"

let Predicates = [HasVendorXqcicm, IsRV32], DecoderNamespace = "Xqcicm" in {
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
  def QC_C_MVEQZ   : RVInst16CL<0b101, 0b10, (outs GPRC:$rd_wb),
                              (ins GPRC:$rd, GPRC:$rs1),
                              "qc.c.mveqz", "$rd, $rs1"> {
    let Constraints = "$rd = $rd_wb";

    let Inst{12-10} = 0b011;
    let Inst{6-5} = 0b00;
  }

  def QC_MVEQ    : QCIMVCC<0b000, "qc.mveq">;
  def QC_MVNE    : QCIMVCC<0b001, "qc.mvne">;
  def QC_MVLT    : QCIMVCC<0b100, "qc.mvlt">;
  def QC_MVGE    : QCIMVCC<0b101, "qc.mvge">;
  def QC_MVLTU   : QCIMVCC<0b110, "qc.mvltu">;
  def QC_MVGEU   : QCIMVCC<0b111, "qc.mvgeu">;

  def QC_MVEQI   : QCIMVCCI<0b000, "qc.mveqi", simm5>;
  def QC_MVNEI   : QCIMVCCI<0b001, "qc.mvnei", simm5>;
  def QC_MVLTI   : QCIMVCCI<0b100, "qc.mvlti", simm5>;
  def QC_MVGEI   : QCIMVCCI<0b101, "qc.mvgei", simm5>;
  def QC_MVLTUI  : QCIMVCCI<0b110, "qc.mvltui", uimm5>;
  def QC_MVGEUI  : QCIMVCCI<0b111, "qc.mvgeui", uimm5>;
} // Predicates = [HasVendorXqcicm, IsRV32], DecoderNamespace = "Xqcicm"

let Predicates = [HasVendorXqciint, IsRV32], DecoderNamespace = "Xqciint" in {
  let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in
  def QC_C_DIR : RVInst16CI<0b000, 0b10, (outs GPRNoX0:$rd), (ins),
                            "qc.c.dir", "$rd"> {
    bits<5> rd;

    let Inst{12} = 0b1;
    let Inst{11-7} = rd;
    let Inst{6-2} = 0b00000;
  }

  def QC_SETINTI : QCIInt_IMM<0b0, "qc.setinti">;
  def QC_CLRINTI : QCIInt_IMM<0b1, "qc.clrinti">;

  def QC_C_EIR    : QCIRVInst16CI_RS1<0b00001, "qc.c.eir">;
  def QC_C_SETINT : QCIRVInst16CI_RS1<0b00010, "qc.c.setint">;
  def QC_C_CLRINT : QCIRVInst16CI_RS1<0b00011, "qc.c.clrint">;

  let mayLoad = 0, mayStore = 0 in {
  def QC_C_DI     : QCIRVInst16CI_NONE<0b10110, "qc.c.di">;
  def QC_C_EI     : QCIRVInst16CI_NONE<0b10111, "qc.c.ei">;
  } // mayLoad =0, mayStore = 0

  let mayLoad = 1, mayStore = 1 in {
  def QC_C_MIENTER      : QCIRVInst16CI_NONE<0b10000, "qc.c.mienter">;
  def QC_C_MIENTER_NEST : QCIRVInst16CI_NONE<0b10001, "qc.c.mienter.nest">;
  } // mayLoad = 1, mayStore = 1

  let mayLoad = 1, mayStore = 1, isReturn = 1, isTerminator = 1 in
  def QC_C_MILEAVERET   : QCIRVInst16CI_NONE<0b10100, "qc.c.mileaveret">;
} // Predicates = [HasVendorXqciint, IsRV32], DecoderNamespace = "Xqciint"

let Predicates = [HasVendorXqcilo, IsRV32], DecoderNamespace = "Xqcilo" in {
  def QC_E_LB    : QCIRVInstEILoad<0b101, 0b00, "qc.e.lb">;
  def QC_E_LBU   : QCIRVInstEILoad<0b101, 0b01, "qc.e.lbu">;
  def QC_E_LH    : QCIRVInstEILoad<0b101, 0b10, "qc.e.lh">;
  def QC_E_LHU   : QCIRVInstEILoad<0b101, 0b11, "qc.e.lhu">;
  def QC_E_LW    : QCIRVInstEILoad<0b110, 0b00, "qc.e.lw">;

  def QC_E_SB    : QCIRVInstESStore<0b110, 0b01, "qc.e.sb">;
  def QC_E_SH    : QCIRVInstESStore<0b110, 0b10, "qc.e.sh">;
  def QC_E_SW    : QCIRVInstESStore<0b110, 0b11, "qc.e.sw">;
} // Predicates = [HasVendorXqcilo, IsRV32], DecoderNamespace = "Xqcilo"

//===----------------------------------------------------------------------===//
// Aliases
//===----------------------------------------------------------------------===//

let Predicates = [HasVendorXqcilsm, IsRV32] in {
let EmitPriority = 0 in {
  def : InstAlias<"qc.swm $rs3, $rs2, (${rs1})",
                  (QC_SWM GPR:$rs3, GPR:$rs1, GPRNoX0:$rs2, 0)>;
  def : InstAlias<"qc.swmi $rs3, $length, (${rs1})",
                  (QC_SWMI GPR:$rs3, GPR:$rs1, uimm5nonzero:$length, 0)>;
  def : InstAlias<"qc.setwm $rs3, $rs2, (${rs1})",
                  (QC_SETWM GPR:$rs3, GPR:$rs1, GPRNoX0:$rs2, 0)>;
  def : InstAlias<"qc.setwmi $rs3, $length, (${rs1})",
                  (QC_SETWMI GPR:$rs3, GPR:$rs1, uimm5nonzero:$length, 0)>;
  def : InstAlias<"qc.lwm $rd, $rs2, (${rs1})",
                  (QC_LWM GPRNoX0:$rd, GPR:$rs1, GPRNoX0:$rs2, 0)>;
  def : InstAlias<"qc.lwmi $rd, $length, (${rs1})",
                  (QC_LWMI GPRNoX0:$rd, GPR:$rs1, uimm5nonzero:$length, 0)>;
} // EmitPriority = 0
} // Predicates = [HasVendorXqcilsm, IsRV32]

let Predicates = [HasVendorXqcilo, IsRV32] in {
let EmitPriority = 0 in {
  def  : InstAlias<"qc.e.lb  $rd, (${rs1})",
                   (QC_E_LB  GPR:$rd, GPR:$rs1, 0)>;
  def  : InstAlias<"qc.e.lbu $rd, (${rs1})",
                   (QC_E_LBU GPR:$rd, GPR:$rs1, 0)>;
  def  : InstAlias<"qc.e.lh  $rd, (${rs1})",
                   (QC_E_LH  GPR:$rd, GPR:$rs1, 0)>;
  def  : InstAlias<"qc.e.lhu $rd, (${rs1})",
                   (QC_E_LHU GPR:$rd, GPR:$rs1, 0)>;
  def  : InstAlias<"qc.e.lw  $rd, (${rs1})",
                   (QC_E_LW  GPR:$rd, GPR:$rs1, 0)>;
  def  : InstAlias<"qc.e.sb  $rs2, (${rs1})",
                   (QC_E_SB GPR:$rs2, GPR:$rs1, 0)>;
  def  : InstAlias<"qc.e.sh  $rs2, (${rs1})",
                   (QC_E_SH GPR:$rs2, GPR:$rs1, 0)>;
  def  : InstAlias<"qc.e.sw  $rs2, (${rs1})",
                   (QC_E_SW GPR:$rs2, GPR:$rs1, 0)>;
} // EmitPriority = 0
} // Predicates = [HasVendorXqcilo, IsRV32]
