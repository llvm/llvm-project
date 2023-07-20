//===-- RISCVInstrInfoXCV.td - CORE-V instructions ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the vendor extensions defined by Core-V extensions.
//
//===----------------------------------------------------------------------===//

let DecoderNamespace = "XCVbitmanip" in {
  class CVInstBitManipRII<bits<2> funct2, bits<3> funct3, dag outs, dag ins,
                      string opcodestr, string argstr>
      : RVInstI<funct3, OPC_CUSTOM_2, outs, ins, opcodestr, argstr> {
    bits<5> is3;
    bits<5> is2;
    let imm12 = {funct2, is3, is2};
  }

  class CVBitManipRII<bits<2> funct2, bits<3> funct3, string opcodestr,
                      Operand i3type = uimm5>
      : CVInstBitManipRII<funct2, funct3, (outs GPR:$rd),
                          (ins GPR:$rs1, i3type:$is3, uimm5:$is2),
                          opcodestr, "$rd, $rs1, $is3, $is2">;

  class CVBitManipRR<bits<7> funct7, string opcodestr>
      : RVInstR<funct7, 0b011, OPC_CUSTOM_1, (outs GPR:$rd),
                (ins GPR:$rs1, GPR:$rs2), opcodestr, "$rd, $rs1, $rs2">;

  class CVBitManipR<bits<7> funct7, string opcodestr>
      : RVInstR<funct7, 0b011, OPC_CUSTOM_1, (outs GPR:$rd),
                (ins GPR:$rs1, GPR:$rs2), opcodestr, "$rd, $rs1"> {
    let rs2 = 0b00000;
  }
}

let Predicates = [HasVendorXCVbitmanip, IsRV32],
    hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
  def CV_EXTRACT : CVBitManipRII<0b00, 0b000, "cv.extract">;
  def CV_EXTRACTU : CVBitManipRII<0b01, 0b000, "cv.extractu">;

  def CV_BCLR : CVBitManipRII<0b00, 0b001, "cv.bclr">;
  def CV_BSET : CVBitManipRII<0b01, 0b001, "cv.bset">;
  def CV_BITREV : CVBitManipRII<0b11, 0b001, "cv.bitrev", uimm2>;

  def CV_EXTRACTR : CVBitManipRR<0b0011000, "cv.extractr">;
  def CV_EXTRACTUR : CVBitManipRR<0b0011001, "cv.extractur">;

  let Constraints = "$rd = $rd_wb" in {
    def CV_INSERT : CVInstBitManipRII<0b10, 0b000, (outs GPR:$rd_wb),
                             (ins GPR:$rd, GPR:$rs1, uimm5:$is3, uimm5:$is2),
                             "cv.insert", "$rd, $rs1, $is3, $is2">;
    def CV_INSERTR : RVInstR<0b0011010, 0b011, OPC_CUSTOM_1, (outs GPR:$rd_wb),
                             (ins GPR:$rd, GPR:$rs1, GPR:$rs2),
                             "cv.insertr", "$rd, $rs1, $rs2">;
  }

  def CV_BCLRR : CVBitManipRR<0b0011100, "cv.bclrr">;
  def CV_BSETR : CVBitManipRR<0b0011101, "cv.bsetr">;

  def CV_ROR : CVBitManipRR<0b0100000, "cv.ror">;
  def CV_FF1 : CVBitManipR<0b0100001, "cv.ff1">;
  def CV_FL1 : CVBitManipR<0b0100010, "cv.fl1">;
  def CV_CLB : CVBitManipR<0b0100011, "cv.clb">;
  def CV_CNT : CVBitManipR<0b0100100, "cv.cnt">;
}

class CVInstMac<bits<7> funct7, bits<3> funct3, dag outs, dag ins,
                string opcodestr, string argstr, list<dag> pattern>
    : RVInst<outs, ins, opcodestr, argstr, pattern, InstFormatOther> {
  bits<5> rs2;
  bits<5> rs1;
  bits<5> rd;

  let Inst{31-25} = funct7;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = funct3;
  let Inst{11-7} = rd;
  let Inst{6-0} = OPC_CUSTOM_1.Value;
  let DecoderNamespace = "XCVmac";
}

class CVInstMac16I<bits<2> funct2, bits<3> funct3, dag outs, dag ins,
                   string opcodestr, string argstr, list<dag> pattern>
    : RVInst<outs, ins, opcodestr, argstr, pattern, InstFormatOther> {
  bits<5> imm5;
  bits<5> rs2;
  bits<5> rs1;
  bits<5> rd;

  let Inst{31-30} = funct2;
  let Inst{29-25} = imm5;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = funct3;
  let Inst{11-7} = rd;
  let Inst{6-0} = OPC_CUSTOM_2.Value;
  let DecoderNamespace = "XCVmac";
}

let Predicates = [HasVendorXCVmac, IsRV32], hasSideEffects = 0, mayLoad = 0,
    mayStore = 0, Constraints = "$rd = $rd_wb" in {
  // 32x32 bit macs
  def CV_MAC      : CVInstMac<0b1001000, 0b011, (outs GPR:$rd_wb),
                              (ins GPR:$rd, GPR:$rs1, GPR:$rs2),
                              "cv.mac", "$rd, $rs1, $rs2", []>,
                    Sched<[]>;
  def CV_MSU      : CVInstMac<0b1001001, 0b011, (outs GPR:$rd_wb),
                              (ins GPR:$rd, GPR:$rs1, GPR:$rs2),
                              "cv.msu", "$rd, $rs1, $rs2", []>,
                    Sched<[]>;

  // Signed 16x16 bit macs with imm
  def CV_MACSN    : CVInstMac16I<0b00, 0b110, (outs GPR:$rd_wb),
                                 (ins GPR:$rd, GPR:$rs1, GPR:$rs2, uimm5:$imm5),
                                 "cv.macsn", "$rd, $rs1, $rs2, $imm5", []>,
                    Sched<[]>;
  def CV_MACHHSN  : CVInstMac16I<0b01, 0b110, (outs GPR:$rd_wb),
                                 (ins GPR:$rd, GPR:$rs1, GPR:$rs2, uimm5:$imm5),
                                 "cv.machhsn", "$rd, $rs1, $rs2, $imm5", []>,
                    Sched<[]>;
  def CV_MACSRN   : CVInstMac16I<0b10, 0b110, (outs GPR:$rd_wb),
                                 (ins GPR:$rd, GPR:$rs1, GPR:$rs2, uimm5:$imm5),
                                 "cv.macsrn", "$rd, $rs1, $rs2, $imm5", []>,
                    Sched<[]>;
  def CV_MACHHSRN : CVInstMac16I<0b11, 0b110, (outs GPR:$rd_wb),
                                 (ins GPR:$rd, GPR:$rs1, GPR:$rs2, uimm5:$imm5),
                                 "cv.machhsrn", "$rd, $rs1, $rs2, $imm5", []>,
                    Sched<[]>;

  // Unsigned 16x16 bit macs with imm
  def CV_MACUN    : CVInstMac16I<0b00, 0b111, (outs GPR:$rd_wb),
                                 (ins GPR:$rd, GPR:$rs1, GPR:$rs2, uimm5:$imm5),
                                 "cv.macun", "$rd, $rs1, $rs2, $imm5", []>,
                    Sched<[]>;
  def CV_MACHHUN  : CVInstMac16I<0b01, 0b111, (outs GPR:$rd_wb),
                                 (ins GPR:$rd, GPR:$rs1, GPR:$rs2, uimm5:$imm5),
                                 "cv.machhun", "$rd, $rs1, $rs2, $imm5", []>,
                    Sched<[]>;
  def CV_MACURN   : CVInstMac16I<0b10, 0b111, (outs GPR:$rd_wb),
                                 (ins GPR:$rd, GPR:$rs1, GPR:$rs2, uimm5:$imm5),
                                 "cv.macurn", "$rd, $rs1, $rs2, $imm5", []>,
                    Sched<[]>;
  def CV_MACHHURN : CVInstMac16I<0b11, 0b111, (outs GPR:$rd_wb),
                                 (ins GPR:$rd, GPR:$rs1, GPR:$rs2, uimm5:$imm5),
                                 "cv.machhurn", "$rd, $rs1, $rs2, $imm5", []>,
                    Sched<[]>;
} // Predicates = [HasVendorXCVmac, IsRV32], hasSideEffects = 0, mayLoad = 0...

let Predicates = [HasVendorXCVmac, IsRV32], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
  // Signed 16x16 bit muls with imm
  def CV_MULSN    : CVInstMac16I<0b00, 0b100, (outs GPR:$rd),
                                 (ins GPR:$rs1, GPR:$rs2, uimm5:$imm5),
                                 "cv.mulsn", "$rd, $rs1, $rs2, $imm5", []>,
                    Sched<[]>;
  def CV_MULHHSN  : CVInstMac16I<0b01, 0b100, (outs GPR:$rd),
                                 (ins GPR:$rs1, GPR:$rs2, uimm5:$imm5),
                                 "cv.mulhhsn", "$rd, $rs1, $rs2, $imm5", []>,
                    Sched<[]>;
  def CV_MULSRN   : CVInstMac16I<0b10, 0b100, (outs GPR:$rd),
                                 (ins GPR:$rs1, GPR:$rs2, uimm5:$imm5),
                                 "cv.mulsrn", "$rd, $rs1, $rs2, $imm5", []>,
                    Sched<[]>;
  def CV_MULHHSRN : CVInstMac16I<0b11, 0b100, (outs GPR:$rd),
                                 (ins GPR:$rs1, GPR:$rs2, uimm5:$imm5),
                                 "cv.mulhhsrn", "$rd, $rs1, $rs2, $imm5", []>,
                    Sched<[]>;


  // Unsigned 16x16 bit muls with imm
  def CV_MULUN    : CVInstMac16I<0b00, 0b101, (outs GPR:$rd),
                                 (ins GPR:$rs1, GPR:$rs2, uimm5:$imm5),
                                 "cv.mulun", "$rd, $rs1, $rs2, $imm5", []>,
                    Sched<[]>;
  def CV_MULHHUN  : CVInstMac16I<0b01, 0b101, (outs GPR:$rd),
                                 (ins GPR:$rs1, GPR:$rs2, uimm5:$imm5),
                                 "cv.mulhhun", "$rd, $rs1, $rs2, $imm5", []>,
                    Sched<[]>;
  def CV_MULURN   : CVInstMac16I<0b10, 0b101, (outs GPR:$rd),
                                 (ins GPR:$rs1, GPR:$rs2, uimm5:$imm5),
                                 "cv.mulurn", "$rd, $rs1, $rs2, $imm5", []>,
                    Sched<[]>;
  def CV_MULHHURN : CVInstMac16I<0b11, 0b101, (outs GPR:$rd),
                                 (ins GPR:$rs1, GPR:$rs2, uimm5:$imm5),
                                 "cv.mulhhurn", "$rd, $rs1, $rs2, $imm5", []>,
                    Sched<[]>;
} // Predicates = [HasVendorXCVmac, IsRV32], hasSideEffects = 0, mayLoad = 0...

let Predicates = [HasVendorXCVmac, IsRV32] in {
  // Xcvmac Pseudo Instructions
  // Signed 16x16 bit muls
  def : InstAlias<"cv.muls $rd1, $rs1, $rs2",
                  (CV_MULSN GPR:$rd1,   GPR:$rs1, GPR:$rs2, 0)>;
  def : InstAlias<"cv.mulhhs $rd1, $rs1, $rs2",
                  (CV_MULHHSN GPR:$rd1, GPR:$rs1, GPR:$rs2, 0)>;

  // Unsigned 16x16 bit muls
  def : InstAlias<"cv.mulu $rd1, $rs1, $rs2",
                  (CV_MULUN GPR:$rd1,   GPR:$rs1, GPR:$rs2, 0)>;
  def : InstAlias<"cv.mulhhu $rd1, $rs1, $rs2",
                  (CV_MULHHUN GPR:$rd1, GPR:$rs1, GPR:$rs2, 0)>;
} // Predicates = [HasVendorXCVmac, IsRV32]
