//===-- RISCVInstrInfoXTHead.td ----------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the vendor extensions defined by T-Head of Alibaba.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// T-HEAD specific DAG Nodes.
//===----------------------------------------------------------------------===//

def SDT_LoadPair : SDTypeProfile<2, 2,
  [SDTCisSameAs<0, 1>, SDTCisSameAs<1, 3>, SDTCisPtrTy<2>, SDTCisVT<3, XLenVT>]>;
def SDT_StorePair : SDTypeProfile<0, 4,
  [SDTCisSameAs<0, 1>, SDTCisSameAs<1, 3>, SDTCisPtrTy<2>, SDTCisVT<3, XLenVT>]>;

def th_lwud : SDNode<"RISCVISD::TH_LWUD", SDT_LoadPair,
  [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def th_lwd : SDNode<"RISCVISD::TH_LWD", SDT_LoadPair,
  [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def th_ldd : SDNode<"RISCVISD::TH_LDD", SDT_LoadPair,
  [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def th_swd : SDNode<"RISCVISD::TH_SWD", SDT_StorePair,
  [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def th_sdd : SDNode<"RISCVISD::TH_SDD", SDT_StorePair,
  [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;

//===----------------------------------------------------------------------===//
// Instruction class templates
//===----------------------------------------------------------------------===//
class THInstVdotVV<bits<6> funct6, RISCVVFormat opv, dag outs, dag ins,
                   string opcodestr, string argstr>
    : RVInstVV<funct6, opv, outs, ins, opcodestr, argstr> {
  let Inst{26} = 0;
  let Opcode = OPC_CUSTOM_0.Value;
  let DecoderNamespace = "THeadV";
}

class THInstVdotVX<bits<6> funct6, RISCVVFormat opv, dag outs, dag ins,
                   string opcodestr, string argstr>
    : RVInstVX<funct6, opv, outs, ins, opcodestr, argstr> {
  let Inst{26} = 1;
  let Opcode = OPC_CUSTOM_0.Value;
  let DecoderNamespace = "THeadV";
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
// op vd, vs1, vs2, vm (reverse the order of vs1 and vs2)
class THVdotALUrVV<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : THInstVdotVV<funct6, opv, (outs VR:$vd),
                   (ins VR:$vs1, VR:$vs2, VMaskOp:$vm),
                   opcodestr, "$vd, $vs1, $vs2$vm">;

// op vd, rs1, vs2, vm (reverse the order of rs1 and vs2)
class THVdotALUrVX<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : THInstVdotVX<funct6, opv, (outs VR:$vd),
                   (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm),
                   opcodestr, "$vd, $rs1, $vs2$vm">;
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0

let Predicates = [HasVendorXTHeadBa], DecoderNamespace = "THeadBa",
  hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class THShiftALU_rri<bits<3> funct3, string opcodestr>
    : RVInstR<0, funct3, OPC_CUSTOM_0, (outs GPR:$rd),
              (ins GPR:$rs1, GPR:$rs2, uimm2:$uimm2),
	      opcodestr, "$rd, $rs1, $rs2, $uimm2"> {
  bits<2> uimm2;
  let Inst{31-27} = 0;
  let Inst{26-25} = uimm2;
}

let Predicates = [HasVendorXTHeadBb], DecoderNamespace = "THeadBb",
  hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
class THShift_ri<bits<5> funct5, bits<3> funct3, string opcodestr>
    : RVInstIShift<funct5, funct3, OPC_CUSTOM_0, (outs GPR:$rd),
                   (ins GPR:$rs1, uimmlog2xlen:$shamt),
		   opcodestr, "$rd, $rs1, $shamt">;

class THBitfieldExtract_rii<bits<3> funct3, string opcodestr>
    : RVInstI<funct3, OPC_CUSTOM_0, (outs GPR:$rd),
              (ins GPR:$rs1, uimmlog2xlen:$msb, uimmlog2xlen:$lsb),
	      opcodestr, "$rd, $rs1, $msb, $lsb"> {
  bits<6> msb;
  bits<6> lsb;
  let Inst{31-26} = msb;
  let Inst{25-20} = lsb;
}

class THRev_r<bits<5> funct5, bits<2> funct2, string opcodestr>
    : RVInstR4<funct2, 0b001, OPC_CUSTOM_0, (outs GPR:$rd), (ins GPR:$rs1),
               opcodestr, "$rd, $rs1"> {
  let rs3 = funct5;
  let rs2 = 0;
}
}

let Predicates = [HasVendorXTHeadBb, IsRV64], DecoderNamespace = "THeadBb",
  hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class THShiftW_ri<bits<7> funct7, bits<3> funct3, string opcodestr>
    : RVInstIShiftW<funct7, funct3, OPC_CUSTOM_0, (outs GPR:$rd),
                    (ins GPR:$rs1, uimm5:$shamt),
		    opcodestr, "$rd, $rs1, $shamt">;


let Predicates = [HasVendorXTHeadMac], DecoderNamespace = "THeadMac",
    hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCommutable = 1 in
class THMulAccumulate_rr<bits<7> funct7, string opcodestr>
    : RVInstR<funct7, 0b001, OPC_CUSTOM_0, (outs GPR:$rd_wb),
              (ins GPR:$rd, GPR:$rs1, GPR:$rs2),
              opcodestr, "$rd, $rs1, $rs2"> {
  let Constraints = "$rd_wb = $rd";
}

let Predicates = [HasVendorXTHeadMemPair], DecoderNamespace = "THeadMemPair",
  hasSideEffects = 0, mayLoad = 1, mayStore = 0 in
class THLoadPair<bits<5> funct5, string opcodestr>
  : RVInstR<!shl(funct5, 2), 0b100, OPC_CUSTOM_0,
            (outs GPR:$rd, GPR:$rs2),
            (ins GPR:$rs1, uimm2:$uimm2, uimm7:$const3or4),
             opcodestr, "$rd, $rs2, (${rs1}), $uimm2, $const3or4"> {
  bits<2> uimm2;
  let Inst{26-25} = uimm2;
  let DecoderMethod = "decodeXTHeadMemPair";
  let Constraints = "@earlyclobber $rd,@earlyclobber $rs2";
}

let Predicates = [HasVendorXTHeadMemPair], DecoderNamespace = "THeadMemPair",
  hasSideEffects = 0, mayLoad = 0, mayStore = 1 in
class THStorePair<bits<5> funct5, string opcodestr>
  : RVInstR<!shl(funct5, 2), 0b101, OPC_CUSTOM_0,
            (outs),
            (ins GPR:$rd, GPR:$rs2, GPR:$rs1, uimm2:$uimm2, uimm7:$const3or4),
            opcodestr, "$rd, $rs2, (${rs1}), $uimm2, $const3or4"> {
  bits<2> uimm2;
  let Inst{26-25} = uimm2;
  let DecoderMethod = "decodeXTHeadMemPair";
}


//===----------------------------------------------------------------------===//
// Combination of instruction classes.
// Use these multiclasses to define instructions more easily.
//===----------------------------------------------------------------------===//
multiclass THVdotVMAQA_VX<string opcodestr, bits<6> funct6> {
  def _VX : THVdotALUrVX<funct6, OPMVX, opcodestr # ".vx">;
}

multiclass THVdotVMAQA<string opcodestr, bits<6> funct6> {
  def _VV   : THVdotALUrVV<funct6, OPMVX, opcodestr # ".vv">;
  defm ""   : THVdotVMAQA_VX<opcodestr, funct6>;
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//
let Predicates = [HasVendorXTHeadBa] in {
def TH_ADDSL : THShiftALU_rri<0b001, "th.addsl">,
               Sched<[WriteSHXADD, ReadSHXADD, ReadSHXADD]>;
} // Predicates = [HasVendorXTHeadBa]

let Predicates = [HasVendorXTHeadBb] in {
def TH_SRRI : THShift_ri<0b00010, 0b001, "th.srri">;
def TH_EXT : THBitfieldExtract_rii<0b010, "th.ext">;
def TH_EXTU : THBitfieldExtract_rii<0b011, "th.extu">;
def TH_FF0 : THRev_r<0b10000, 0b10, "th.ff0">;
def TH_FF1 : THRev_r<0b10000, 0b11, "th.ff1">;
def TH_REV : THRev_r<0b10000, 0b01, "th.rev">;
def TH_TSTNBZ : THRev_r<0b10000, 0b00, "th.tstnbz">;
} // Predicates = [HasVendorXTHeadBb]

let Predicates = [HasVendorXTHeadBb, IsRV64], IsSignExtendingOpW = 1 in {
def TH_SRRIW : THShiftW_ri<0b0001010, 0b001, "th.srriw">;
def TH_REVW : THRev_r<0b10010, 0b00, "th.revw">;
} // Predicates = [HasVendorXTHeadBb, IsRV64]

let Predicates = [HasVendorXTHeadBs], DecoderNamespace = "THeadBs" in {
let IsSignExtendingOpW = 1 in
def TH_TST : RVBShift_ri<0b10001, 0b001, OPC_CUSTOM_0, "th.tst">,
             Sched<[WriteSingleBitImm, ReadSingleBitImm]>;
} // Predicates = [HasVendorXTHeadBs]

let Predicates = [HasVendorXTHeadMac] in {
def TH_MULA  : THMulAccumulate_rr<0b0010000, "th.mula">;
def TH_MULS  : THMulAccumulate_rr<0b0010001, "th.muls">;
}  // Predicates = [HasVendorXTHeadMac]

let Predicates = [HasVendorXTHeadMac], IsSignExtendingOpW = 1 in {
def TH_MULAH : THMulAccumulate_rr<0b0010100, "th.mulah">;
def TH_MULSH : THMulAccumulate_rr<0b0010101, "th.mulsh">;
} // Predicates = [HasVendorXTHeadMac], IsSignExtendingOpW = 1

let Predicates = [HasVendorXTHeadMac, IsRV64], IsSignExtendingOpW = 1 in {
def TH_MULAW : THMulAccumulate_rr<0b0010010, "th.mulaw">;
def TH_MULSW : THMulAccumulate_rr<0b0010011, "th.mulsw">;
} // Predicates = [HasVendorXTHeadMac, IsRV64]

let Predicates = [HasVendorXTHeadMemPair] in {
def TH_LWUD : THLoadPair<0b11110, "th.lwud">,
              Sched<[WriteLDW, WriteLDW, ReadMemBase]>;
def TH_SWD  : THStorePair<0b11100, "th.swd">,
              Sched<[WriteSTW, WriteSTW, ReadStoreData, ReadMemBase]>;
let IsSignExtendingOpW = 1 in
def TH_LWD  : THLoadPair<0b11100, "th.lwd">,
              Sched<[WriteLDW, WriteLDW, ReadMemBase]>;
}

let Predicates = [HasVendorXTHeadMemPair, IsRV64] in {
def TH_LDD  : THLoadPair<0b11111, "th.ldd">,
              Sched<[WriteLDD, WriteLDD, ReadMemBase]>;
def TH_SDD  : THStorePair<0b11111, "th.sdd">,
              Sched<[WriteSTD, WriteSTD, ReadStoreData, ReadMemBase]>;
}

let Predicates = [HasVendorXTHeadVdot],
    Constraints = "@earlyclobber $vd",
    RVVConstraint = WidenV in {
defm THVdotVMAQA      : THVdotVMAQA<"th.vmaqa",     0b100000>;
defm THVdotVMAQAU     : THVdotVMAQA<"th.vmaqau",    0b100010>;
defm THVdotVMAQASU    : THVdotVMAQA<"th.vmaqasu",   0b100100>;
defm THVdotVMAQAUS    : THVdotVMAQA_VX<"th.vmaqaus",0b100110>;
}

// Associate LMUL with tablegen records of register classes.
def THVdotV_M1  : LMULInfo<0b000,  8,   VR, VR,   VR,   VR,   VR, "M1">;
def THVdotV_M2  : LMULInfo<0b001, 16, VRM2, VRM2, VR,   VR,   VR, "M2">;
def THVdotV_M4  : LMULInfo<0b010, 32, VRM4, VRM4, VRM2, VR,   VR, "M4">;
def THVdotV_M8  : LMULInfo<0b011, 64, VRM8, VRM8, VRM4, VRM2, VR, "M8">;

defvar MxListTHVdot = [V_MF2, THVdotV_M1, THVdotV_M2, THVdotV_M4, THVdotV_M8];

defset list<VTypeInfoToWide> AllQuadWidenableInt8NoVLMulVectors = {
  def : VTypeInfoToWide<VI8MF2,  VI32MF2>;
  def : VTypeInfoToWide<VI8M1,   VI32M1>;
  def : VTypeInfoToWide<VI8M2,   VI32M2>;
  def : VTypeInfoToWide<VI8M4,   VI32M4>;
  def : VTypeInfoToWide<VI8M8,   VI32M8>;
}

//===----------------------------------------------------------------------===//
// Combination of instruction classes.
// Use these multiclasses to define instructions more easily.
//===----------------------------------------------------------------------===//
multiclass VPseudoVMAQA_VV_VX {
  foreach m = MxListTHVdot in {
    defm "" : VPseudoTernaryW_VV<m>;
    defm "" : VPseudoTernaryW_VX<m>;
  }
}

multiclass VPseudoVMAQA_VX {
  foreach m = MxListTHVdot in {
    defm "" : VPseudoTernaryW_VX<m>;
  }
}

multiclass VPatTernaryVMAQA_VV<string intrinsic, string instruction,
                               list<VTypeInfoToWide> vtilist> {
  foreach vtiToWti = vtilist in {
    defvar vti = vtiToWti.Vti;
    defvar wti = vtiToWti.Wti;
    defm : VPatTernaryWithPolicy<intrinsic, instruction, "VV",
                                 wti.Vector, vti.Vector, vti.Vector,
                                 vti.Mask, wti.Log2SEW, vti.LMul,
                                 wti.RegClass, vti.RegClass, vti.RegClass>;
  }
}

multiclass VPatTernaryVMAQA_VX<string intrinsic, string instruction,
                               list<VTypeInfoToWide> vtilist> {
  foreach vtiToWti = vtilist in {
    defvar vti = vtiToWti.Vti;
    defvar wti = vtiToWti.Wti;
    defm : VPatTernaryWithPolicy<intrinsic, instruction,
                                 "V"#vti.ScalarSuffix,
                                 wti.Vector, vti.Scalar, vti.Vector,
                                 vti.Mask, wti.Log2SEW, vti.LMul,
                                 wti.RegClass, vti.ScalarRegClass, vti.RegClass>;
  }
}

multiclass VPatTernaryVMAQA_VV_VX<string intrinsic, string instruction,
                                  list<VTypeInfoToWide> vtilist>
    : VPatTernaryVMAQA_VV<intrinsic, instruction, vtilist>,
      VPatTernaryVMAQA_VX<intrinsic, instruction, vtilist>;

//===----------------------------------------------------------------------===//
// Pseudo-instructions and codegen patterns
//===----------------------------------------------------------------------===//
let Predicates = [HasVendorXTHeadBa] in {
def : Pat<(add GPR:$rs1, (shl GPR:$rs2, uimm2:$uimm2)),
          (TH_ADDSL GPR:$rs1, GPR:$rs2, uimm2:$uimm2)>;

// Reuse complex patterns from StdExtZba
def : Pat<(add sh1add_op:$rs1, non_imm12:$rs2),
          (TH_ADDSL GPR:$rs2, sh1add_op:$rs1, 1)>;
def : Pat<(add sh2add_op:$rs1, non_imm12:$rs2),
          (TH_ADDSL GPR:$rs2, sh2add_op:$rs1, 2)>;
def : Pat<(add sh3add_op:$rs1, non_imm12:$rs2),
          (TH_ADDSL GPR:$rs2, sh3add_op:$rs1, 3)>;

def : Pat<(add (mul_oneuse GPR:$rs1, (XLenVT 6)), GPR:$rs2),
          (TH_ADDSL GPR:$rs2, (TH_ADDSL GPR:$rs1, GPR:$rs1, 1), 1)>;
def : Pat<(add (mul_oneuse GPR:$rs1, (XLenVT 10)), GPR:$rs2),
          (TH_ADDSL GPR:$rs2, (TH_ADDSL GPR:$rs1, GPR:$rs1, 2), 1)>;
def : Pat<(add (mul_oneuse GPR:$rs1, (XLenVT 18)), GPR:$rs2),
          (TH_ADDSL GPR:$rs2, (TH_ADDSL GPR:$rs1, GPR:$rs1, 3), 1)>;
def : Pat<(add (mul_oneuse GPR:$rs1, (XLenVT 12)), GPR:$rs2),
          (TH_ADDSL GPR:$rs2, (TH_ADDSL GPR:$rs1, GPR:$rs1, 1), 2)>;
def : Pat<(add (mul_oneuse GPR:$rs1, (XLenVT 20)), GPR:$rs2),
          (TH_ADDSL GPR:$rs2, (TH_ADDSL GPR:$rs1, GPR:$rs1, 2), 2)>;
def : Pat<(add (mul_oneuse GPR:$rs1, (XLenVT 36)), GPR:$rs2),
          (TH_ADDSL GPR:$rs2, (TH_ADDSL GPR:$rs1, GPR:$rs1, 3), 2)>;
def : Pat<(add (mul_oneuse GPR:$rs1, (XLenVT 24)), GPR:$rs2),
          (TH_ADDSL GPR:$rs2, (TH_ADDSL GPR:$rs1, GPR:$rs1, 1), 3)>;
def : Pat<(add (mul_oneuse GPR:$rs1, (XLenVT 40)), GPR:$rs2),
          (TH_ADDSL GPR:$rs2, (TH_ADDSL GPR:$rs1, GPR:$rs1, 2), 3)>;
def : Pat<(add (mul_oneuse GPR:$rs1, (XLenVT 72)), GPR:$rs2),
          (TH_ADDSL GPR:$rs2, (TH_ADDSL GPR:$rs1, GPR:$rs1, 3), 3)>;

def : Pat<(add GPR:$r, CSImm12MulBy4:$i),
          (TH_ADDSL GPR:$r, (ADDI X0, (SimmShiftRightBy2XForm CSImm12MulBy4:$i)), 2)>;
def : Pat<(add GPR:$r, CSImm12MulBy8:$i),
          (TH_ADDSL GPR:$r, (ADDI X0, (SimmShiftRightBy3XForm CSImm12MulBy8:$i)), 3)>;

def : Pat<(mul GPR:$r, C3LeftShift:$i),
          (SLLI (TH_ADDSL GPR:$r, GPR:$r, 1),
                (TrailingZeros C3LeftShift:$i))>;
def : Pat<(mul GPR:$r, C5LeftShift:$i),
          (SLLI (TH_ADDSL GPR:$r, GPR:$r, 2),
                (TrailingZeros C5LeftShift:$i))>;
def : Pat<(mul GPR:$r, C9LeftShift:$i),
          (SLLI (TH_ADDSL GPR:$r, GPR:$r, 3),
                (TrailingZeros C9LeftShift:$i))>;

def : Pat<(mul_const_oneuse GPR:$r, (XLenVT 11)),
	  (TH_ADDSL GPR:$r, (TH_ADDSL GPR:$r, GPR:$r, 2), 1)>;
def : Pat<(mul_const_oneuse GPR:$r, (XLenVT 19)),
	  (TH_ADDSL GPR:$r, (TH_ADDSL GPR:$r, GPR:$r, 3), 1)>;
def : Pat<(mul_const_oneuse GPR:$r, (XLenVT 13)),
	  (TH_ADDSL GPR:$r, (TH_ADDSL GPR:$r, GPR:$r, 1), 2)>;
def : Pat<(mul_const_oneuse GPR:$r, (XLenVT 21)),
	  (TH_ADDSL GPR:$r, (TH_ADDSL GPR:$r, GPR:$r, 2), 2)>;
def : Pat<(mul_const_oneuse GPR:$r, (XLenVT 37)),
	  (TH_ADDSL GPR:$r, (TH_ADDSL GPR:$r, GPR:$r, 3), 2)>;
def : Pat<(mul_const_oneuse GPR:$r, (XLenVT 25)),
	  (TH_ADDSL (TH_ADDSL GPR:$r, GPR:$r, 2), (TH_ADDSL GPR:$r, GPR:$r, 2), 2)>;
def : Pat<(mul_const_oneuse GPR:$r, (XLenVT 41)),
	  (TH_ADDSL GPR:$r, (TH_ADDSL GPR:$r, GPR:$r, 2), 3)>;
def : Pat<(mul_const_oneuse GPR:$r, (XLenVT 73)),
	  (TH_ADDSL GPR:$r, (TH_ADDSL GPR:$r, GPR:$r, 3), 3)>;
def : Pat<(mul_const_oneuse GPR:$r, (XLenVT 27)),
	  (TH_ADDSL (TH_ADDSL GPR:$r, GPR:$r, 3), (TH_ADDSL GPR:$r, GPR:$r, 3), 1)>;
def : Pat<(mul_const_oneuse GPR:$r, (XLenVT 45)),
	  (TH_ADDSL (TH_ADDSL GPR:$r, GPR:$r, 3), (TH_ADDSL GPR:$r, GPR:$r, 3), 2)>;
def : Pat<(mul_const_oneuse GPR:$r, (XLenVT 81)),
	  (TH_ADDSL (TH_ADDSL GPR:$r, GPR:$r, 3), (TH_ADDSL GPR:$r, GPR:$r, 3), 3)>;

def : Pat<(mul_const_oneuse GPR:$r, (XLenVT 200)),
	  (SLLI (TH_ADDSL (TH_ADDSL GPR:$r, GPR:$r, 2),
		       (TH_ADDSL GPR:$r, GPR:$r, 2), 2), 3)>;
} // Predicates = [HasVendorXTHeadBa]

let Predicates = [HasVendorXTHeadBb] in {
def : PatGprImm<rotr, TH_SRRI, uimmlog2xlen>;
// There's no encoding for a rotate-left-immediate in X-THead-Bb, as
// it can be implemented with th.srri by negating the immediate.
def : Pat<(rotl GPR:$rs1, uimmlog2xlen:$shamt),
          (TH_SRRI GPR:$rs1, (ImmSubFromXLen uimmlog2xlen:$shamt))>;
def : Pat<(rotr GPR:$rs1, GPR:$rs2),
           (OR (SRL GPR:$rs1, GPR:$rs2),
               (SLL GPR:$rs1, (SUB X0, GPR:$rs2)))>;
def : Pat<(rotl GPR:$rs1, GPR:$rs2),
           (OR (SLL GPR:$rs1, GPR:$rs2),
               (SRL GPR:$rs1, (SUB X0, GPR:$rs2)))>;
def : Pat<(sext_inreg GPR:$rs1, i32), (TH_EXT GPR:$rs1, 31, 0)>;
def : Pat<(sext_inreg GPR:$rs1, i16), (TH_EXT GPR:$rs1, 15, 0)>;
def : Pat<(sext_inreg GPR:$rs1, i8), (TH_EXT GPR:$rs1, 7, 0)>;
def : Pat<(sext_inreg GPR:$rs1, i1), (TH_EXT GPR:$rs1, 0, 0)>;
def : PatGpr<ctlz, TH_FF1>;
def : Pat<(ctlz (xor GPR:$rs1, -1)), (TH_FF0 GPR:$rs1)>;
def : PatGpr<bswap, TH_REV>;
} // Predicates = [HasVendorXTHeadBb]

let Predicates = [HasVendorXTHeadBb, IsRV64] in {
def : PatGprImm<riscv_rorw, TH_SRRIW, uimm5>;
def : Pat<(riscv_rolw GPR:$rs1, uimm5:$rs2),
          (TH_SRRIW GPR:$rs1, (ImmSubFrom32 uimm5:$rs2))>;
def : Pat<(riscv_rorw i64:$rs1, i64:$rs2),
          (OR (SRLW i64:$rs1, i64:$rs2),
              (SLLW i64:$rs1, (SUB X0, i64:$rs2)))>;
def : Pat<(riscv_rolw i64:$rs1, i64:$rs2),
          (OR (SLLW i64:$rs1, i64:$rs2),
              (SRLW i64:$rs1, (SUB X0, i64:$rs2)))>;
def : Pat<(sra (bswap i64:$rs1), (i64 32)),
          (TH_REVW i64:$rs1)>;
def : Pat<(binop_allwusers<srl> (bswap i64:$rs1), (i64 32)),
          (TH_REVW i64:$rs1)>;
def : Pat<(riscv_clzw i64:$rs1),
          (TH_FF0 (SLLI (XORI i64:$rs1, -1), 32))>;
} // Predicates = [HasVendorXTHeadBb, IsRV64]

let Predicates = [HasVendorXTHeadBs] in {
def : Pat<(and (srl GPR:$rs1, uimmlog2xlen:$shamt), 1),
          (TH_TST GPR:$rs1, uimmlog2xlen:$shamt)>;
def : Pat<(seteq (and GPR:$rs1, SingleBitSetMask:$mask), 0),
          (TH_TST (XORI GPR:$rs1, -1), SingleBitSetMask:$mask)>;
} // Predicates = [HasVendorXTHeadBs]

let Predicates = [HasVendorXTHeadMac] in {
def : Pat<(add GPR:$rd, (mul GPR:$rs1, GPR:$rs2)), (TH_MULA GPR:$rd, GPR:$rs1, GPR:$rs2)>;
def : Pat<(sub GPR:$rd, (mul GPR:$rs1, GPR:$rs2)), (TH_MULS GPR:$rd, GPR:$rs1, GPR:$rs2)>;
} // Predicates = [HasVendorXTHeadMac]

let Predicates = [HasVendorXTHeadMac, IsRV64] in {
// mulaw, mulsw are available only in RV64.
def : Pat<(binop_allwusers<add> GPR:$rd, (mul GPR:$rs1, GPR:$rs2)),
          (TH_MULAW GPR:$rd, GPR:$rs1, GPR:$rs2)>;
def : Pat<(binop_allwusers<sub> GPR:$rd, (mul GPR:$rs1, GPR:$rs2)),
          (TH_MULSW GPR:$rd, GPR:$rs1, GPR:$rs2)>;
// mulah, mulsh produce a sign-extended result.
def : Pat<(binop_allwusers<add> GPR:$rd, (mul
            (sexti16 GPR:$rs1),
            (sexti16 GPR:$rs2))),
          (TH_MULAH GPR:$rd, GPR:$rs1, GPR:$rs2)>;
def : Pat<(binop_allwusers<sub> GPR:$rd, (mul
            (sexti16 GPR:$rs1),
            (sexti16 GPR:$rs2))),
          (TH_MULSH GPR:$rd, GPR:$rs1, GPR:$rs2)>;
} // Predicates = [HasVendorXTHeadMac, IsRV64]

let Predicates = [HasVendorXTHeadMac, IsRV32] in {
def : Pat<(add GPR:$rd, (mul (sexti16 GPR:$rs1), (sexti16 GPR:$rs2))),
	  (TH_MULAH GPR:$rd, GPR:$rs1, GPR:$rs2)>;
def : Pat<(sub GPR:$rd, (mul (sexti16 GPR:$rs1), (sexti16 GPR:$rs2))),
	  (TH_MULSH GPR:$rd, GPR:$rs1, GPR:$rs2)>;
} // Predicates = [HasVendorXTHeadMac, IsRV32]

defm PseudoTHVdotVMAQA      : VPseudoVMAQA_VV_VX;
defm PseudoTHVdotVMAQAU     : VPseudoVMAQA_VV_VX;
defm PseudoTHVdotVMAQASU    : VPseudoVMAQA_VV_VX;
defm PseudoTHVdotVMAQAUS    : VPseudoVMAQA_VX;

let Predicates = [HasVendorXTHeadVdot] in {
defm : VPatTernaryVMAQA_VV_VX<"int_riscv_th_vmaqa",  "PseudoTHVdotVMAQA",  AllQuadWidenableInt8NoVLMulVectors>;
defm : VPatTernaryVMAQA_VV_VX<"int_riscv_th_vmaqau", "PseudoTHVdotVMAQAU", AllQuadWidenableInt8NoVLMulVectors>;
defm : VPatTernaryVMAQA_VV_VX<"int_riscv_th_vmaqasu","PseudoTHVdotVMAQASU",AllQuadWidenableInt8NoVLMulVectors>;
defm : VPatTernaryVMAQA_VX<"int_riscv_th_vmaqaus",   "PseudoTHVdotVMAQAUS",AllQuadWidenableInt8NoVLMulVectors>;
}

def uimm2_3_XFORM : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant((N->getZExtValue() >> 3) & 0x3,
                                   SDLoc(N), Subtarget->getXLenVT());
}]>;

def uimm2_3 : Operand<XLenVT>, ImmLeaf<XLenVT, [{
  return isShiftedUInt<2, 3>(Imm);
}], uimm2_3_XFORM>;

def uimm2_4_XFORM : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant((N->getZExtValue() >> 4) & 0x3,
                                   SDLoc(N), Subtarget->getXLenVT());
}]>;

def uimm2_4 : Operand<XLenVT>, ImmLeaf<XLenVT, [{
  return isShiftedUInt<2, 4>(Imm);
}], uimm2_4_XFORM>;

let Predicates = [HasVendorXTHeadMemPair, IsRV64] in {
def : Pat<(th_lwud i64:$rs1, uimm2_3:$uimm2_3), (TH_LWUD i64:$rs1, uimm2_3:$uimm2_3, 3)>;
def : Pat<(th_ldd i64:$rs1, uimm2_4:$uimm2_4), (TH_LDD i64:$rs1, uimm2_4:$uimm2_4, 4)>;

def : Pat<(th_sdd i64:$rd1, i64:$rd2, i64:$rs1, uimm2_4:$uimm2_4),
          (TH_SDD i64:$rd1, i64:$rd2, i64:$rs1, uimm2_4:$uimm2_4, 4)>;
}

let Predicates = [HasVendorXTHeadMemPair] in {
  def : Pat<(th_lwd GPR:$rs1, uimm2_3:$uimm2_3), (TH_LWD GPR:$rs1, uimm2_3:$uimm2_3, 3)>;
  def : Pat<(th_swd GPR:$rd1, GPR:$rd2, GPR:$rs1, uimm2_3:$uimm2_3),
            (TH_SWD GPR:$rd1, GPR:$rd2, GPR:$rs1, uimm2_3:$uimm2_3, 3)>;
}
