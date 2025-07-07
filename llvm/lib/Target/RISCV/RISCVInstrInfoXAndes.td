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
def bare_simm11_lsb0 : Operand<OtherVT> {
  let ParserMatchClass = BareSImmNLsb0AsmOperand<11>;
  let PrintMethod = "printBranchOperand";
  let EncoderMethod = "getImmOpValueAsrN<1>";
  let DecoderMethod = "decodeSImmOperandAndLslN<11, 1>";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (MCOp.evaluateAsConstantImm(Imm))
      return isShiftedInt<10, 1>(Imm);
    return MCOp.isBareSymbolRef();
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

// Predicate: True if immediate is a power of 2.
def PowerOf2 : PatLeaf<(imm), [{
  return isPowerOf2_64(N->getZExtValue());
}]>;

// Transformation function: Get log2 of immediate.
def Log2 : SDNodeXForm<imm, [{
  uint64_t Imm = Log2_64(N->getZExtValue());
  return CurDAG->getTargetConstant(Imm, SDLoc(N), N->getValueType(0));
}]>;

//===----------------------------------------------------------------------===//
// Instruction Class Templates
//===----------------------------------------------------------------------===//

class NDSRVInstBB<bit cs, string opcodestr>
    : RVInst<(outs),
             (ins GPR:$rs1, uimmlog2xlen:$cimm, bare_simm11_lsb0:$imm10),
             opcodestr, "$rs1, $cimm, $imm10", [], InstFormatNDS_BRANCH_10>,
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
    : RVInst<(outs), (ins GPR:$rs1, uimm7:$cimm, bare_simm11_lsb0:$imm10),
             opcodestr, "$rs1, $cimm, $imm10", [], InstFormatNDS_BRANCH_10>,
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
    : RVInst<(outs GPR:$rd),
             (ins GPR:$rs1, uimmlog2xlen:$msb, uimmlog2xlen:$lsb),
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

class NDSRVInstLEA<bits<7> funct7, string opcodestr>
    : RVInstR<funct7, 0b000, OPC_CUSTOM_2,
              (outs GPR:$rd), (ins GPR:$rs2, GPR:$rs1),
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
      SchedBinaryMC<"WriteVFMulAddF", "ReadVFMulAddV", "ReadVFMulAddF"> {
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

class NDSRVInstVD4DOT<bits<6> funct6, string opcodestr>
    : RVInst<(outs VR:$vd), (ins VR:$vs1, VR:$vs2, VMaskOp:$vm),
             opcodestr # "." # "vv", "$vd, $vs1, $vs2$vm", [], InstFormatR>,
      SchedBinaryMC<"WriteVIMulAddV", "ReadVIMulAddV", "ReadVIMulAddV"> {
  bits<5> vs2;
  bits<5> vs1;
  bits<5> vd;
  bit vm;

  let Inst{31-26} = funct6;
  let Inst{25} = vm;
  let Inst{24-20} = vs2;
  let Inst{19-15} = vs1;
  let Inst{14-12} = 0b100;
  let Inst{11-7} = vd;
  let Inst{6-0} = OPC_CUSTOM_2.Value;
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;

  let RVVConstraint = VMConstraint;
}

class NDSRVInstVBFHCvt<bits<5> vs1, string opcodestr>
    : RVInst<(outs VR:$vd), (ins VR:$vs2, VMaskOp:$vm),
             opcodestr, "$vd, $vs2", [], InstFormatR> {
  bits<5> vs2;
  bits<5> vd;

  let Inst{31-25} = 0b0000000;
  let Inst{24-20} = vs2;
  let Inst{19-15} = vs1;
  let Inst{14-12} = 0b100;
  let Inst{11-7} = vd;
  let Inst{6-0} = OPC_CUSTOM_2.Value;
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;

  let Uses = [VL, VTYPE];
}

//===----------------------------------------------------------------------===//
// Multiclass
//===----------------------------------------------------------------------===//

multiclass VPseudoVWCVT_S_BF16 {
  defvar constraint = "@earlyclobber $rd";
  foreach m = MxListFW in {
    let VLMul = m.value, SEW=16 in
    def "_" # m.MX : VPseudoUnaryNoMask<m.wvrclass, m.vrclass, constraint>,
                     SchedUnary<"WriteVFWCvtIToFV", "ReadVFWCvtIToFV", m.MX, 16,
                                forcePassthruRead=true>;
  }
}

multiclass VPseudoVNCVT_BF16_S {
  defvar constraint = "@earlyclobber $rd";
  foreach m = MxListFW in {
    let VLMul = m.value, SEW=16 in
    def "_" # m.MX : VPseudoUnaryNoMaskRoundingMode<m.vrclass, m.wvrclass,
                                                    constraint>,
                     SchedUnary<"WriteVFNCvtFToFV", "ReadVFNCvtFToFV", m.MX, 16,
                                forcePassthruRead=true>;
  }
}

multiclass VPatConversionS_BF16<string intrinsic, string instruction> {
  foreach fvtiToFWti = AllWidenableBFloatToFloatVectors in {
    defvar fvti = fvtiToFWti.Vti;
    defvar fwti = fvtiToFWti.Wti;
    let Predicates = [HasVendorXAndesVBFHCvt] in
    def : VPatUnaryNoMask<intrinsic, instruction, "BF16",
                          fwti.Vector, fvti.Vector,
                          fvti.Log2SEW, fvti.LMul,
                          fwti.RegClass, fvti.RegClass>;
  }
}

multiclass VPatConversionBF16_S<string intrinsic, string instruction> {
  foreach fvtiToFWti = AllWidenableBFloatToFloatVectors in {
    defvar fvti = fvtiToFWti.Vti;
    defvar fwti = fvtiToFWti.Wti;
    let Predicates = [HasVendorXAndesVBFHCvt] in
    def : VPatUnaryNoMaskRoundingMode<intrinsic, instruction, "S",
                                      fvti.Vector, fwti.Vector,
                                      fvti.Log2SEW, fvti.LMul,
                                      fvti.RegClass, fwti.RegClass>;
  }
}

let fprclass = !cast<RegisterClass>("FPR32") in
def SCALAR_F16_FPR32 : FPR_Info<16>;

let hasSideEffects = 0 in
multiclass VPseudoVFPMAD_VF_RM {
  foreach m = SCALAR_F16_FPR32.MxList in {
    defm "" : VPseudoBinaryV_VF_RM<m, SCALAR_F16_FPR32, 0>,
              SchedBinary<"WriteVFMulAddF", "ReadVFMulAddV", "ReadVFMulAddF",
                          m.MX,  SCALAR_F16_FPR32.SEW, forcePassthruRead=true>;
  }
}

multiclass VPatVFPMADBinaryV_VX_RM<string intrinsic, string instruction,
                                   list<VTypeInfo> vtilist> {
  foreach vti = vtilist in {
    defvar kind = "V"#vti.ScalarSuffix;
    defm : VPatBinaryRoundingMode<intrinsic,
                                  instruction#"_"#kind#"_"#vti.LMul.MX,
                                  vti.Vector, vti.Vector, f32, vti.Mask,
                                  vti.Log2SEW, vti.RegClass,
                                  vti.RegClass, FPR32>;
  }
}

multiclass VPseudoVD4DOT_VV {
  foreach m = [V_MF2, V_M1, V_M2, V_M4, V_M8] in {
    defm "" : VPseudoBinaryV_VV<m>,
              SchedBinary<"WriteVIALUV", "ReadVIALUV", "ReadVIALUV", m.MX,
                          forcePassthruRead=true>;
  }
}

multiclass VPatTernaryVD4DOT_VV<string intrinsic, string instruction,
                                list<VTypeInfoToWide> vtilist> {
  foreach vtiToWti = vtilist in {
    defvar vti = vtiToWti.Vti;
    defvar wti = vtiToWti.Wti;
    let Predicates = GetVTypePredicates<wti>.Predicates in
    defm : VPatTernaryWithPolicy<intrinsic, instruction, "VV",
                                 wti.Vector, vti.Vector, vti.Vector,
                                 wti.Mask, wti.Log2SEW, vti.LMul,
                                 wti.RegClass, vti.RegClass, vti.RegClass>;
  }
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

def NDS_LEA_H : NDSRVInstLEA<0b0000101, "nds.lea.h">;
def NDS_LEA_W : NDSRVInstLEA<0b0000110, "nds.lea.w">;
def NDS_LEA_D : NDSRVInstLEA<0b0000111, "nds.lea.d">;

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
def NDS_LEA_B_ZE : NDSRVInstLEA<0b0001000, "nds.lea.b.ze">;
def NDS_LEA_H_ZE : NDSRVInstLEA<0b0001001, "nds.lea.h.ze">;
def NDS_LEA_W_ZE : NDSRVInstLEA<0b0001010, "nds.lea.w.ze">;
def NDS_LEA_D_ZE : NDSRVInstLEA<0b0001011, "nds.lea.d.ze">;

def NDS_LWUGP : NDSRVInstLWGP<0b110, "nds.lwugp">;
def NDS_LDGP  : NDSRVInstLDGP<0b011, "nds.ldgp">;

def NDS_SDGP  : NDSRVInstSDGP<0b111, "nds.sdgp">;
} // Predicates = [HasVendorXAndesPerf, IsRV64]

//===----------------------------------------------------------------------===//
// XAndesVBFHCvt
//===----------------------------------------------------------------------===//

let Predicates = [HasVendorXAndesVBFHCvt], Constraints = "@earlyclobber $vd",
    mayRaiseFPException = true in {
let RVVConstraint = VS2Constraint, DestEEW = EEWSEWx2 in
def NDS_VFWCVT_S_BF16 : NDSRVInstVBFHCvt<0b00000, "nds.vfwcvt.s.bf16">;
let Uses = [FRM, VL, VTYPE] in
def NDS_VFNCVT_BF16_S : NDSRVInstVBFHCvt<0b00001, "nds.vfncvt.bf16.s">;
}

//===----------------------------------------------------------------------===//
// XAndesVPackFPH
//===----------------------------------------------------------------------===//

let Predicates = [HasVendorXAndesVPackFPH],
    Uses = [FRM, VL, VTYPE], mayRaiseFPException = true in {
def NDS_VFPMADT_VF : NDSRVInstVFPMAD<0b000010, "nds.vfpmadt">;
def NDS_VFPMADB_VF : NDSRVInstVFPMAD<0b000011, "nds.vfpmadb">;
}

//===----------------------------------------------------------------------===//
// XAndesVDot
//===----------------------------------------------------------------------===//

let Predicates = [HasVendorXAndesVDot], Uses = [VL, VTYPE] in {
def NDS_VD4DOTS_VV  : NDSRVInstVD4DOT<0b000100, "nds.vd4dots">;
def NDS_VD4DOTU_VV  : NDSRVInstVD4DOT<0b000111, "nds.vd4dotu">;
def NDS_VD4DOTSU_VV : NDSRVInstVD4DOT<0b000101, "nds.vd4dotsu">;
}
} // DecoderNamespace = "XAndes"

//===----------------------------------------------------------------------===//
// Pseudo-instructions and codegen patterns
//===----------------------------------------------------------------------===//

class NDS_BBPat<CondCode Cond, NDSRVInstBB Inst>
    : Pat<(riscv_brcc (and(XLenVT GPR:$rs1), PowerOf2:$mask), 0, Cond,
              bb:$imm10),
          (Inst GPR:$rs1, (Log2 PowerOf2:$mask), bare_simm11_lsb0:$imm10)>;

class NDS_BCPat<CondCode Cond, NDSRVInstBC Inst>
    : Pat<(riscv_brcc (XLenVT GPR:$rs1), uimm7:$cimm, Cond, bb:$imm10),
          (Inst GPR:$rs1, uimm7:$cimm, bare_simm11_lsb0:$imm10)>;

defm CC_UImmLog2XLen_NDS : SelectCC_GPR_riirr<GPR, uimmlog2xlen>;
defm CC_UImm7_NDS        : SelectCC_GPR_riirr<GPR, uimm7>;

class SelectNDS_BB<CondCode Cond>
    : Pat<(riscv_selectcc_frag:$cc (and(XLenVT GPR:$lhs), PowerOf2:$mask), 0,
              Cond, (XLenVT GPR:$truev), GPR:$falsev),
          (Select_GPR_Using_CC_UImmLog2XLen_NDS GPR:$lhs, (Log2 PowerOf2:$mask),
              (IntCCtoRISCVCC $cc), GPR:$truev, GPR:$falsev)>;

class SelectNDS_BC<CondCode Cond>
    : Pat<(riscv_selectcc_frag:$cc (XLenVT GPR:$lhs), uimm7:$cimm, Cond,
              (XLenVT GPR:$truev), GPR:$falsev),
          (Select_GPR_Using_CC_UImm7_NDS GPR:$lhs, uimm7:$cimm,
              (IntCCtoRISCVCC $cc), GPR:$truev, GPR:$falsev)>;

let Predicates = [HasVendorXAndesPerf] in {

def : NDS_BBPat<SETEQ, NDS_BBC>;
def : NDS_BBPat<SETNE, NDS_BBS>;

def : SelectNDS_BB<SETEQ>;
def : SelectNDS_BB<SETNE>;

def : NDS_BCPat<SETEQ, NDS_BEQC>;
def : NDS_BCPat<SETNE, NDS_BNEC>;

def : SelectNDS_BC<SETEQ>;
def : SelectNDS_BC<SETNE>;

def : Pat<(sext_inreg (XLenVT GPR:$rs1), i16), (NDS_BFOS GPR:$rs1, 15, 0)>;
def : Pat<(sext_inreg (XLenVT GPR:$rs1), i8), (NDS_BFOS GPR:$rs1, 7, 0)>;
def : Pat<(sext_inreg (XLenVT GPR:$rs1), i1), (NDS_BFOS GPR:$rs1, 0, 0)>;

defm : ShxAddPat<1, NDS_LEA_H>;
defm : ShxAddPat<2, NDS_LEA_W>;
defm : ShxAddPat<3, NDS_LEA_D>;

def : CSImm12MulBy4Pat<NDS_LEA_W>;
def : CSImm12MulBy8Pat<NDS_LEA_D>;
} // Predicates = [HasVendorXAndesPerf]

let Predicates = [HasVendorXAndesPerf, IsRV64] in {

defm : ADD_UWPat<NDS_LEA_B_ZE>;

defm : ShxAdd_UWPat<1, NDS_LEA_H_ZE>;
defm : ShxAdd_UWPat<2, NDS_LEA_W_ZE>;
defm : ShxAdd_UWPat<3, NDS_LEA_D_ZE>;

defm : Sh1Add_UWPat<NDS_LEA_H_ZE>;
defm : Sh2Add_UWPat<NDS_LEA_W_ZE>;
defm : Sh3Add_UWPat<NDS_LEA_D_ZE>;

def : Sh1AddPat<NDS_LEA_H_ZE>;
def : Sh2AddPat<NDS_LEA_W_ZE>;
def : Sh3AddPat<NDS_LEA_D_ZE>;
} // Predicates = [HasVendorXAndesPerf, IsRV64]

let Predicates = [HasVendorXAndesVBFHCvt] in {
defm PseudoNDS_VFWCVT_S_BF16 : VPseudoVWCVT_S_BF16;
defm PseudoNDS_VFNCVT_BF16_S : VPseudoVNCVT_BF16_S;
} // Predicates = [HasVendorXAndesVBFHCvt]

defm : VPatConversionS_BF16<"int_riscv_nds_vfwcvt_s_bf16",
                            "PseudoNDS_VFWCVT_S">;
defm : VPatConversionBF16_S<"int_riscv_nds_vfncvt_bf16_s",
                            "PseudoNDS_VFNCVT_BF16">;

let Predicates = [HasVendorXAndesVPackFPH],
    mayRaiseFPException = true in {
defm PseudoNDS_VFPMADT : VPseudoVFPMAD_VF_RM;
defm PseudoNDS_VFPMADB : VPseudoVFPMAD_VF_RM;
} // Predicates = [HasVendorXAndesVPackFPH]

defm : VPatVFPMADBinaryV_VX_RM<"int_riscv_nds_vfpmadt", "PseudoNDS_VFPMADT",
                               AllFP16Vectors>;
defm : VPatVFPMADBinaryV_VX_RM<"int_riscv_nds_vfpmadb", "PseudoNDS_VFPMADB",
                               AllFP16Vectors>;

let Predicates = [HasVendorXAndesVDot] in {
defm PseudoNDS_VD4DOTS  : VPseudoVD4DOT_VV;
defm PseudoNDS_VD4DOTU  : VPseudoVD4DOT_VV;
defm PseudoNDS_VD4DOTSU : VPseudoVD4DOT_VV;
}

defset list<VTypeInfoToWide> AllQuadWidenableVD4DOTVectors = {
  def : VTypeInfoToWide<VI8MF2,  VI32MF2>;
  def : VTypeInfoToWide<VI8M1,   VI32M1>;
  def : VTypeInfoToWide<VI8M2,   VI32M2>;
  def : VTypeInfoToWide<VI8M4,   VI32M4>;
  def : VTypeInfoToWide<VI8M8,   VI32M8>;
  def : VTypeInfoToWide<VI16M1,  VI64M1>;
  def : VTypeInfoToWide<VI16M2,  VI64M2>;
  def : VTypeInfoToWide<VI16M4,  VI64M4>;
  def : VTypeInfoToWide<VI16M8,  VI64M8>;
}

defm : VPatTernaryVD4DOT_VV<"int_riscv_nds_vd4dots", "PseudoNDS_VD4DOTS",
                            AllQuadWidenableVD4DOTVectors>;
defm : VPatTernaryVD4DOT_VV<"int_riscv_nds_vd4dotu", "PseudoNDS_VD4DOTU",
                            AllQuadWidenableVD4DOTVectors>;
defm : VPatTernaryVD4DOT_VV<"int_riscv_nds_vd4dotsu", "PseudoNDS_VD4DOTSU",
                            AllQuadWidenableVD4DOTVectors>;

//===----------------------------------------------------------------------===//
// Pseudo-instructions for SFB (Short Forward Branch)
//===----------------------------------------------------------------------===//

let Predicates = [HasShortForwardBranchOpt], hasSideEffects = 0,
    mayLoad = 0, mayStore = 0, Size = 8, Constraints = "$dst = $falsev" in {
def PseudoCCNDS_BFOS : Pseudo<(outs GPR:$dst),
                              (ins GPR:$lhs, GPR:$rhs, ixlenimm:$cc,
                               GPR:$falsev, GPR:$rs1,
                               uimmlog2xlen:$msb, uimmlog2xlen:$lsb), []>,
                       Sched<[WriteSFB, ReadSFBJmp, ReadSFBJmp, ReadSFBALU,
                              ReadSFBALU]>;
def PseudoCCNDS_BFOZ : Pseudo<(outs GPR:$dst),
                              (ins GPR:$lhs, GPR:$rhs, ixlenimm:$cc,
                               GPR:$falsev, GPR:$rs1,
                               uimmlog2xlen:$msb, uimmlog2xlen:$lsb), []>,
                       Sched<[WriteSFB, ReadSFBJmp, ReadSFBJmp, ReadSFBALU,
                              ReadSFBALU]>;
}
