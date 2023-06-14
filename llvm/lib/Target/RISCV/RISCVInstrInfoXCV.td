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
  class RVInstBitManipRII<bits<2> funct2, bits<3> funct3, dag outs, dag ins,
                      string opcodestr, string argstr>
      : RVInstI<funct3, OPC_CUSTOM_2, outs, ins, opcodestr, argstr> {
    bits<5> is3;
    bits<5> is2;
    let imm12 = {funct2, is3, is2};
  }

  class CVBitManipRII<bits<2> funct2, bits<3> funct3, string opcodestr, 
                      Operand i3type = uimm5>
      : RVInstBitManipRII<funct2, funct3, (outs GPR:$rd), 
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
    def CV_INSERT : RVInstBitManipRII<0b10, 0b000, (outs GPR:$rd_wb),
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
