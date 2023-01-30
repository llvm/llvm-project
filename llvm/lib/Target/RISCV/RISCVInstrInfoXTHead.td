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

let Predicates = [HasVendorXTHeadBs], DecoderNamespace = "THeadBs" in {
let IsSignExtendingOpW = 1 in
def TH_TST : RVBShift_ri<0b10001, 0b001, OPC_CUSTOM_0, "th.tst">,
             Sched<[WriteSingleBitImm, ReadSingleBitImm]>;
} // Predicates = [HasVendorXTHeadBs]

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
} // Predicates = [HasVendorXTHeadBa]

let Predicates = [HasVendorXTHeadBs] in {
def : Pat<(and (srl GPR:$rs1, uimmlog2xlen:$shamt), 1),
          (TH_TST GPR:$rs1, uimmlog2xlen:$shamt)>;
def : Pat<(seteq (and GPR:$rs1, SingleBitSetMask:$mask), 0),
          (TH_TST (XORI GPR:$rs1, -1), SingleBitSetMask:$mask)>;
} // Predicates = [HasVendorXTHeadBs]


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
