//===-- RISCVInstrInfoXAndes.td ----------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the vendor extensions defined by Andes Technology.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operand and SDNode transformation definitions.
//===----------------------------------------------------------------------===//

// A 11-bit signed immediate where the least significant bit is zero.
def simm11_lsb0 : Operand<OtherVT> {
  let ParserMatchClass = SImmAsmOperand<11, "Lsb0">;
  let PrintMethod = "printBranchOperand";
  let EncoderMethod = "getImmOpValueAsrN<1>";
  let DecoderMethod = "decodeSImmOperandAndLslN<11, 1>";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (!MCOp.evaluateAsConstantImm(Imm))
      return false;
    return isShiftedInt<10, 1>(Imm);
  }];
  let OperandType = "OPERAND_PCREL";
}

def simm18 : Operand<XLenVT> {
  let ParserMatchClass = SImmAsmOperand<18>;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeSImmOperand<18>";
}

def simm18_lsb0 : Operand<XLenVT> {
  let ParserMatchClass = SImmAsmOperand<18, "Lsb0">;
  let EncoderMethod = "getImmOpValueAsrN<1>";
  let DecoderMethod = "decodeSImmOperandAndLslN<18, 1>";
}

def simm19_lsb00 : Operand<XLenVT> {
  let ParserMatchClass = SImmAsmOperand<19, "Lsb00">;
  let EncoderMethod = "getImmOpValueAsrN<2>";
  let DecoderMethod = "decodeSImmOperandAndLslN<19, 2>";
}

def simm20_lsb000 : Operand<XLenVT> {
  let ParserMatchClass = SImmAsmOperand<20, "Lsb000">;
  let EncoderMethod = "getImmOpValueAsrN<3>";
  let DecoderMethod = "decodeSImmOperandAndLslN<20, 3>";
}

//===----------------------------------------------------------------------===//
// Instruction Class Templates
//===----------------------------------------------------------------------===//

class NDSRVInstBB<bit cs, string opcodestr>
    : RVInst<(outs), (ins GPR:$rs1, uimmlog2xlen:$cimm, simm11_lsb0:$imm10),
             opcodestr, "$rs1, $cimm, $imm10", [], InstFormatOther>,
      Sched<[WriteJmp, ReadIALU]> {
  bits<10> imm10;
  bits<5> rs1;
  bits<6> cimm;

  let Inst{31} = imm10{9};
  let Inst{30} = cs;
  let Inst{29-25} = imm10{8-4};
  let Inst{24-20} = cimm{4-0};
  let Inst{19-15} = rs1;
  let Inst{14-12} = 0b111;
  let Inst{11-8} = imm10{3-0};
  let Inst{7} = cimm{5};
  let Inst{6-0} = OPC_CUSTOM_2.Value;
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
  let isBranch = 1;
  let isTerminator = 1;
}

class NDSRVInstBC<bits<3> funct3, string opcodestr>
    : RVInst<(outs), (ins GPR:$rs1, uimm7:$cimm, simm11_lsb0:$imm10),
             opcodestr, "$rs1, $cimm, $imm10", [], InstFormatOther>,
      Sched<[WriteJmp, ReadIALU]> {
  bits<10> imm10;
  bits<5> rs1;
  bits<7> cimm;

  let Inst{31} = imm10{9};
  let Inst{30} = cimm{6};
  let Inst{29-25} = imm10{8-4};
  let Inst{24-20} = cimm{4-0};
  let Inst{19-15} = rs1;
  let Inst{14-12} = funct3;
  let Inst{11-8} = imm10{3-0};
  let Inst{7} = cimm{5};
  let Inst{6-0} = OPC_CUSTOM_2.Value;
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
  let isBranch = 1;
  let isTerminator = 1;
}

class NDSRVInstBFO<bits<3> funct3, string opcodestr>
    : RVInst<(outs GPR:$rd), (ins GPR:$rs1, uimmlog2xlen:$msb, uimmlog2xlen:$lsb),
             opcodestr, "$rd, $rs1, $msb, $lsb", [], InstFormatOther>,
      Sched<[WriteIALU, ReadIALU]> {
  bits<5> rd;
  bits<5> rs1;
  bits<6> msb;
  bits<6> lsb;

  let Inst{31-26} = msb;
  let Inst{25-20} = lsb;
  let Inst{19-15} = rs1;
  let Inst{14-12} = funct3;
  let Inst{11-7} = rd;
  let Inst{6-0} = OPC_CUSTOM_2.Value;
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
}

class NDSRVInstRR<bits<7> funct7, string opcodestr>
    : RVInstR<funct7, 0b000, OPC_CUSTOM_2,
              (outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
              opcodestr, "$rd, $rs1, $rs2">,
      Sched<[WriteIALU, ReadIALU, ReadIALU]> {
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
}

// GP: ADDI, LB, LBU
class NDSRVInstLBGP<bits<2> funct2, string opcodestr>
    : RVInst<(outs GPR:$rd), (ins simm18:$imm18),
             opcodestr, "$rd, ${imm18}", [], InstFormatOther> {
  bits<18> imm18;
  bits<5> rd;

  let Inst{31} = imm18{17};
  let Inst{30-21} = imm18{10-1};
  let Inst{20} = imm18{11};
  let Inst{19-17} = imm18{14-12};
  let Inst{16-15} = imm18{16-15};
  let Inst{14} = imm18{0};
  let Inst{13-12} = funct2;
  let Inst{11-7} = rd;
  let Inst{6-0} = OPC_CUSTOM_0.Value;
  let hasSideEffects = 0;
  let mayLoad = 1;
  let mayStore = 0;
}

// GP: LH, LHU
class NDSRVInstLHGP<bits<3> funct3, string opcodestr>
    : RVInst<(outs GPR:$rd), (ins simm18_lsb0:$imm17),
             opcodestr, "$rd, ${imm17}", [], InstFormatOther> {
  bits<17> imm17;
  bits<5> rd;

  let Inst{31} = imm17{16};
  let Inst{30-21} = imm17{9-0};
  let Inst{20} = imm17{10};
  let Inst{19-17} = imm17{13-11};
  let Inst{16-15} = imm17{15-14};
  let Inst{14-12} = funct3;
  let Inst{11-7} = rd;
  let Inst{6-0} = OPC_CUSTOM_1.Value;
  let hasSideEffects = 0;
  let mayLoad = 1;
  let mayStore = 0;
}

// GP: LW, LWU
class NDSRVInstLWGP<bits<3> funct3, string opcodestr>
    : RVInst<(outs GPR:$rd), (ins simm19_lsb00:$imm17),
             opcodestr, "$rd, ${imm17}", [], InstFormatOther> {
  bits<17> imm17;
  bits<5> rd;

  let Inst{31} = imm17{16};
  let Inst{30-22} = imm17{8-0};
  let Inst{21} = imm17{15};
  let Inst{20} = imm17{9};
  let Inst{19-17} = imm17{12-10};
  let Inst{16-15} = imm17{14-13};
  let Inst{14-12} = funct3;
  let Inst{11-7} = rd;
  let Inst{6-0} = OPC_CUSTOM_1.Value;
  let hasSideEffects = 0;
  let mayLoad = 1;
  let mayStore = 0;
}

// GP: LD
class NDSRVInstLDGP<bits<3> funct3, string opcodestr>
    : RVInst<(outs GPR:$rd), (ins simm20_lsb000:$imm17),
             opcodestr, "$rd, ${imm17}", [], InstFormatOther> {
  bits<17> imm17;
  bits<5> rd;

  let Inst{31} = imm17{16};
  let Inst{30-23} = imm17{7-0};
  let Inst{22-21} = imm17{15-14};
  let Inst{20} = imm17{8};
  let Inst{19-17} = imm17{11-9};
  let Inst{16-15} = imm17{13-12};
  let Inst{14-12} = funct3;
  let Inst{11-7} = rd;
  let Inst{6-0} = OPC_CUSTOM_1.Value;
  let hasSideEffects = 0;
  let mayLoad = 1;
  let mayStore = 0;
}

// GP: SB
class NDSRVInstSBGP<bits<2> funct2, string opcodestr>
    : RVInst<(outs), (ins GPR:$rs2, simm18:$imm18),
             opcodestr, "$rs2, ${imm18}", [], InstFormatOther> {
  bits<18> imm18;
  bits<5> rs2;

  let Inst{31} = imm18{17};
  let Inst{30-25} = imm18{10-5};
  let Inst{24-20} = rs2;
  let Inst{19-17} = imm18{14-12};
  let Inst{16-15} = imm18{16-15};
  let Inst{14} = imm18{0};
  let Inst{13-12} = funct2;
  let Inst{11-8} = imm18{4-1};
  let Inst{7} = imm18{11};
  let Inst{6-0} = OPC_CUSTOM_0.Value;
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 1;
}

// GP: SH
class NDSRVInstSHGP<bits<3> funct3, string opcodestr>
    : RVInst<(outs), (ins GPR:$rs2, simm18_lsb0:$imm17),
             opcodestr, "$rs2, ${imm17}", [], InstFormatOther> {
  bits<17> imm17;
  bits<5> rs2;

  let Inst{31} = imm17{16};
  let Inst{30-25} = imm17{9-4};
  let Inst{24-20} = rs2;
  let Inst{19-17} = imm17{13-11};
  let Inst{16-15} = imm17{15-14};
  let Inst{14-12} = funct3;
  let Inst{11-8} = imm17{3-0};
  let Inst{7} = imm17{10};
  let Inst{6-0} = OPC_CUSTOM_1.Value;
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 1;
}

// GP: SW
class NDSRVInstSWGP<bits<3> funct3, string opcodestr>
    : RVInst<(outs), (ins GPR:$rs2, simm19_lsb00:$imm17),
             opcodestr, "$rs2, ${imm17}", [], InstFormatOther> {
  bits<17> imm17;
  bits<5> rs2;

  let Inst{31} = imm17{16};
  let Inst{30-25} = imm17{8-3};
  let Inst{24-20} = rs2;
  let Inst{19-17} = imm17{12-10};
  let Inst{16-15} = imm17{14-13};
  let Inst{14-12} = funct3;
  let Inst{11-9} = imm17{2-0};
  let Inst{8} = imm17{15};
  let Inst{7} = imm17{9};
  let Inst{6-0} = OPC_CUSTOM_1.Value;
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 1;
}

// GP: SD
class NDSRVInstSDGP<bits<3> funct3, string opcodestr>
    : RVInst<(outs), (ins GPR:$rs2, simm20_lsb000:$imm17),
             opcodestr, "$rs2, ${imm17}", [], InstFormatOther> {
  bits<17> imm17;
  bits<5> rs2;

  let Inst{31} = imm17{16};
  let Inst{30-25} = imm17{7-2};
  let Inst{24-20} = rs2;
  let Inst{19-17} = imm17{11-9};
  let Inst{16-15} = imm17{13-12};
  let Inst{14-12} = funct3;
  let Inst{11-10} = imm17{1-0};
  let Inst{9-8} = imm17{15-14};
  let Inst{7} = imm17{8};
  let Inst{6-0} = OPC_CUSTOM_1.Value;
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 1;
}

class NDSRVInstVFPMAD<bits<6> funct6, string opcodestr>
    : RVInst<(outs VR:$vd), (ins VR:$vs2, FPR32:$rs1, VMaskOp:$vm),
             opcodestr # "." # "vf", "$vd, $rs1, $vs2$vm", [], InstFormatR>,
      SchedBinaryMC<"WriteVFALUF", "ReadVFALUV", "ReadVFALUF"> {
  bits<5> vs2;
  bits<5> rs1;
  bits<5> vd;
  bit vm;

  let Inst{31-26} = funct6;
  let Inst{25} = vm;
  let Inst{24-20} = vs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = 0b100;
  let Inst{11-7} = vd;
  let Inst{6-0} = OPC_CUSTOM_2.Value;
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;

  let RVVConstraint = VMConstraint;
}

//===----------------------------------------------------------------------===//
// XAndesPerf
//===----------------------------------------------------------------------===//

let DecoderNamespace = "XAndes" in {

let Predicates = [HasVendorXAndesPerf] in {
def NDS_BBC : NDSRVInstBB<0, "nds.bbc">;
def NDS_BBS : NDSRVInstBB<1, "nds.bbs">;

def NDS_BEQC : NDSRVInstBC<0b101, "nds.beqc">;
def NDS_BNEC : NDSRVInstBC<0b110, "nds.bnec">;

def NDS_BFOS : NDSRVInstBFO<0b011, "nds.bfos">;
def NDS_BFOZ : NDSRVInstBFO<0b010, "nds.bfoz">;

def NDS_LEA_H : NDSRVInstRR<0b0000101, "nds.lea.h">;
def NDS_LEA_W : NDSRVInstRR<0b0000110, "nds.lea.w">;
def NDS_LEA_D : NDSRVInstRR<0b0000111, "nds.lea.d">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def NDS_ADDIGP : NDSRVInstLBGP<0b01, "nds.addigp">;

def NDS_LBGP  : NDSRVInstLBGP<0b00, "nds.lbgp">;
def NDS_LBUGP : NDSRVInstLBGP<0b10, "nds.lbugp">;
def NDS_LHGP  : NDSRVInstLHGP<0b001, "nds.lhgp">;
def NDS_LHUGP : NDSRVInstLHGP<0b101, "nds.lhugp">;
def NDS_LWGP  : NDSRVInstLWGP<0b010, "nds.lwgp">;

def NDS_SBGP  : NDSRVInstSBGP<0b11, "nds.sbgp">;
def NDS_SHGP  : NDSRVInstSHGP<0b000, "nds.shgp">;
def NDS_SWGP  : NDSRVInstSWGP<0b100, "nds.swgp">;

def NDS_FFB     : NDSRVInstRR<0b0010000, "nds.ffb">;
def NDS_FFZMISM : NDSRVInstRR<0b0010001, "nds.ffzmism">;
def NDS_FFMISM  : NDSRVInstRR<0b0010010, "nds.ffmism">;
def NDS_FLMISM  : NDSRVInstRR<0b0010011, "nds.flmism">;
} // Predicates = [HasVendorXAndesPerf]

let Predicates = [HasVendorXAndesPerf, IsRV64] in {
def NDS_LEA_B_ZE : NDSRVInstRR<0b0001000, "nds.lea.b.ze">;
def NDS_LEA_H_ZE : NDSRVInstRR<0b0001001, "nds.lea.h.ze">;
def NDS_LEA_W_ZE : NDSRVInstRR<0b0001010, "nds.lea.w.ze">;
def NDS_LEA_D_ZE : NDSRVInstRR<0b0001011, "nds.lea.d.ze">;

def NDS_LWUGP : NDSRVInstLWGP<0b110, "nds.lwugp">;
def NDS_LDGP  : NDSRVInstLDGP<0b011, "nds.ldgp">;

def NDS_SDGP  : NDSRVInstSDGP<0b111, "nds.sdgp">;
} // Predicates = [HasVendorXAndesPerf, IsRV64]

//===----------------------------------------------------------------------===//
// XAndesVPackFPH
//===----------------------------------------------------------------------===//

let Predicates = [HasVendorXAndesVPackFPH],
    Uses = [FRM, VL, VTYPE], mayRaiseFPException = true in {
def NDS_VFPMADT_VF : NDSRVInstVFPMAD<0b000010, "nds.vfpmadt">;
def NDS_VFPMADB_VF : NDSRVInstVFPMAD<0b000011, "nds.vfpmadb">;
}
} // DecoderNamespace = "XAndes"
