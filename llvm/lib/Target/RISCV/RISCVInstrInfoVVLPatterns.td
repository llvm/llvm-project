//===- RISCVInstrInfoVVLPatterns.td - RVV VL patterns ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file contains the required infrastructure and VL patterns to
/// support code generation for the standard 'V' (Vector) extension, version
/// 0.10.  This version is still experimental as the 'V' extension hasn't been
/// ratified yet.
///
/// This file is included from and depends upon RISCVInstrInfoVPseudos.td
///
/// Note: the patterns for RVV intrinsics are found in
/// RISCVInstrInfoVPseudos.td.
///
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Helpers to define the VL patterns.
//===----------------------------------------------------------------------===//

def SDT_RISCVVLE_VL : SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisPtrTy<1>,
                                           SDTCisVT<2, XLenVT>]>;
def SDT_RISCVVSE_VL : SDTypeProfile<0, 3, [SDTCisVec<0>, SDTCisPtrTy<1>,
                                           SDTCisVT<2, XLenVT>]>;

def SDT_RISCVIntBinOp_VL : SDTypeProfile<1, 4, [SDTCisSameAs<0, 1>,
                                                SDTCisSameAs<0, 2>,
                                                SDTCisVec<0>, SDTCisInt<0>,
                                                SDTCVecEltisVT<3, i1>,
                                                SDTCisSameNumEltsAs<0, 3>,
                                                SDTCisVT<4, XLenVT>]>;

def SDT_RISCVFPUnOp_VL : SDTypeProfile<1, 3, [SDTCisSameAs<0, 1>,
                                              SDTCisVec<0>, SDTCisFP<0>,
                                              SDTCVecEltisVT<2, i1>,
                                              SDTCisSameNumEltsAs<0, 2>,
                                              SDTCisVT<3, XLenVT>]>;
def SDT_RISCVFPBinOp_VL : SDTypeProfile<1, 4, [SDTCisSameAs<0, 1>,
                                               SDTCisSameAs<0, 2>,
                                               SDTCisVec<0>, SDTCisFP<0>,
                                               SDTCVecEltisVT<3, i1>,
                                               SDTCisSameNumEltsAs<0, 3>,
                                               SDTCisVT<4, XLenVT>]>;

def riscv_vmv_v_x_vl : SDNode<"RISCVISD::VMV_V_X_VL",
                              SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisInt<0>,
                                                   SDTCisVT<1, XLenVT>,
                                                   SDTCisVT<2, XLenVT>]>>;
def riscv_vfmv_v_f_vl : SDNode<"RISCVISD::VFMV_V_F_VL",
                               SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisFP<0>,
                                                    SDTCisEltOfVec<1, 0>,
                                                    SDTCisVT<2, XLenVT>]>>;

def riscv_vle_vl : SDNode<"RISCVISD::VLE_VL", SDT_RISCVVLE_VL,
                          [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_vse_vl : SDNode<"RISCVISD::VSE_VL", SDT_RISCVVSE_VL,
                          [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;

def riscv_add_vl   : SDNode<"RISCVISD::ADD_VL",   SDT_RISCVIntBinOp_VL, [SDNPCommutative]>;
def riscv_sub_vl   : SDNode<"RISCVISD::SUB_VL",   SDT_RISCVIntBinOp_VL>;
def riscv_mul_vl   : SDNode<"RISCVISD::MUL_VL",   SDT_RISCVIntBinOp_VL, [SDNPCommutative]>;
def riscv_and_vl   : SDNode<"RISCVISD::AND_VL",   SDT_RISCVIntBinOp_VL, [SDNPCommutative]>;
def riscv_or_vl    : SDNode<"RISCVISD::OR_VL",    SDT_RISCVIntBinOp_VL, [SDNPCommutative]>;
def riscv_xor_vl   : SDNode<"RISCVISD::XOR_VL",   SDT_RISCVIntBinOp_VL, [SDNPCommutative]>;
def riscv_sdiv_vl  : SDNode<"RISCVISD::SDIV_VL",  SDT_RISCVIntBinOp_VL>;
def riscv_srem_vl  : SDNode<"RISCVISD::SREM_VL",  SDT_RISCVIntBinOp_VL>;
def riscv_udiv_vl  : SDNode<"RISCVISD::UDIV_VL",  SDT_RISCVIntBinOp_VL>;
def riscv_urem_vl  : SDNode<"RISCVISD::UREM_VL",  SDT_RISCVIntBinOp_VL>;
def riscv_shl_vl   : SDNode<"RISCVISD::SHL_VL",   SDT_RISCVIntBinOp_VL>;
def riscv_sra_vl   : SDNode<"RISCVISD::SRA_VL",   SDT_RISCVIntBinOp_VL>;
def riscv_srl_vl   : SDNode<"RISCVISD::SRL_VL",   SDT_RISCVIntBinOp_VL>;
def riscv_smin_vl  : SDNode<"RISCVISD::SMIN_VL",  SDT_RISCVIntBinOp_VL>;
def riscv_smax_vl  : SDNode<"RISCVISD::SMAX_VL",  SDT_RISCVIntBinOp_VL>;
def riscv_umin_vl  : SDNode<"RISCVISD::UMIN_VL",  SDT_RISCVIntBinOp_VL>;
def riscv_umax_vl  : SDNode<"RISCVISD::UMAX_VL",  SDT_RISCVIntBinOp_VL>;
def riscv_fadd_vl  : SDNode<"RISCVISD::FADD_VL",  SDT_RISCVFPBinOp_VL, [SDNPCommutative]>;
def riscv_fsub_vl  : SDNode<"RISCVISD::FSUB_VL",  SDT_RISCVFPBinOp_VL>;
def riscv_fmul_vl  : SDNode<"RISCVISD::FMUL_VL",  SDT_RISCVFPBinOp_VL, [SDNPCommutative]>;
def riscv_fdiv_vl  : SDNode<"RISCVISD::FDIV_VL",  SDT_RISCVFPBinOp_VL>;
def riscv_fneg_vl  : SDNode<"RISCVISD::FNEG_VL",  SDT_RISCVFPUnOp_VL>;
def riscv_fabs_vl  : SDNode<"RISCVISD::FABS_VL",  SDT_RISCVFPUnOp_VL>;
def riscv_fsqrt_vl : SDNode<"RISCVISD::FSQRT_VL", SDT_RISCVFPUnOp_VL>;

def SDT_RISCVVecFMA_VL : SDTypeProfile<1, 5, [SDTCisSameAs<0, 1>,
                                              SDTCisSameAs<0, 2>,
                                              SDTCisSameAs<0, 3>,
                                              SDTCisVec<0>, SDTCisFP<0>,
                                              SDTCVecEltisVT<4, i1>,
                                              SDTCisSameNumEltsAs<0, 4>,
                                              SDTCisVT<5, XLenVT>]>;
def riscv_fma_vl : SDNode<"RISCVISD::FMA_VL", SDT_RISCVVecFMA_VL>;

def riscv_setcc_vl : SDNode<"RISCVISD::SETCC_VL",
                            SDTypeProfile<1, 5, [SDTCisVec<0>,
                                                 SDTCVecEltisVT<0, i1>,
                                                 SDTCisVec<1>,
                                                 SDTCisSameNumEltsAs<0, 1>,
                                                 SDTCisSameAs<1, 2>,
                                                 SDTCisVT<3, OtherVT>,
                                                 SDTCisSameAs<0, 4>,
                                                 SDTCisVT<5, XLenVT>]>>;

def riscv_vrgather_vx_vl : SDNode<"RISCVISD::VRGATHER_VX_VL",
                                  SDTypeProfile<1, 4, [SDTCisVec<0>,
                                                       SDTCisSameAs<0, 1>,
                                                       SDTCisVT<2, XLenVT>,
                                                       SDTCVecEltisVT<3, i1>,
                                                       SDTCisSameNumEltsAs<0, 3>,
                                                       SDTCisVT<4, XLenVT>]>>;

def SDT_RISCVVMSETCLR_VL : SDTypeProfile<1, 1, [SDTCisVec<0>,
                                                SDTCVecEltisVT<0, i1>,
                                                SDTCisVT<1, XLenVT>]>;
def riscv_vmclr_vl : SDNode<"RISCVISD::VMCLR_VL", SDT_RISCVVMSETCLR_VL>;
def riscv_vmset_vl : SDNode<"RISCVISD::VMSET_VL", SDT_RISCVVMSETCLR_VL>;

def true_mask : PatLeaf<(riscv_vmset_vl (XLenVT srcvalue))>;

// Ignore the vl operand.
def SplatFPOp : PatFrag<(ops node:$op),
                        (riscv_vfmv_v_f_vl node:$op, srcvalue)>;

def sew8simm5  : ComplexPattern<XLenVT, 1, "selectRVVSimm5<8>",  []>;
def sew16simm5 : ComplexPattern<XLenVT, 1, "selectRVVSimm5<16>", []>;
def sew32simm5 : ComplexPattern<XLenVT, 1, "selectRVVSimm5<32>", []>;
def sew64simm5 : ComplexPattern<XLenVT, 1, "selectRVVSimm5<64>", []>;

def sew8uimm5  : ComplexPattern<XLenVT, 1, "selectRVVUimm5<8>",  []>;
def sew16uimm5 : ComplexPattern<XLenVT, 1, "selectRVVUimm5<16>", []>;
def sew32uimm5 : ComplexPattern<XLenVT, 1, "selectRVVUimm5<32>", []>;
def sew64uimm5 : ComplexPattern<XLenVT, 1, "selectRVVUimm5<64>", []>;

class VPatBinaryVL_VV<SDNode vop,
                      string instruction_name,
                      ValueType result_type,
                      ValueType op_type,
                      ValueType mask_type,
                      int sew,
                      LMULInfo vlmul,
                      VReg RetClass,
                      VReg op_reg_class> :
    Pat<(result_type (vop
                     (op_type op_reg_class:$rs1),
                     (op_type op_reg_class:$rs2),
                     (mask_type true_mask),
                     (XLenVT (VLOp GPR:$vl)))),
        (!cast<Instruction>(instruction_name#"_VV_"# vlmul.MX)
                     op_reg_class:$rs1,
                     op_reg_class:$rs2,
                     GPR:$vl, sew)>;

class VPatBinaryVL_XI<SDNode vop,
                      string instruction_name,
                      string suffix,
                      ValueType result_type,
                      ValueType vop_type,
                      ValueType mask_type,
                      int sew,
                      LMULInfo vlmul,
                      VReg RetClass,
                      VReg vop_reg_class,
                      ComplexPattern SplatPatKind,
                      DAGOperand xop_kind> :
    Pat<(result_type (vop
                     (vop_type vop_reg_class:$rs1),
                     (vop_type (SplatPatKind xop_kind:$rs2)),
                     (mask_type true_mask),
                     (XLenVT (VLOp GPR:$vl)))),
        (!cast<Instruction>(instruction_name#_#suffix#_# vlmul.MX)
                     vop_reg_class:$rs1,
                     xop_kind:$rs2,
                     GPR:$vl, sew)>;

multiclass VPatBinaryVL_VV_VX<SDNode vop, string instruction_name> {
  foreach vti = AllIntegerVectors in {
    def : VPatBinaryVL_VV<vop, instruction_name,
                          vti.Vector, vti.Vector, vti.Mask, vti.SEW,
                          vti.LMul, vti.RegClass, vti.RegClass>;
    def : VPatBinaryVL_XI<vop, instruction_name, "VX",
                          vti.Vector, vti.Vector, vti.Mask, vti.SEW,
                          vti.LMul, vti.RegClass, vti.RegClass,
                          SplatPat, GPR>;
  }
}

multiclass VPatBinaryVL_VV_VX_VI<SDNode vop, string instruction_name,
                                 Operand ImmType = simm5> {
  foreach vti = AllIntegerVectors in {
    def : VPatBinaryVL_VV<vop, instruction_name,
                          vti.Vector, vti.Vector, vti.Mask, vti.SEW,
                          vti.LMul, vti.RegClass, vti.RegClass>;
    def : VPatBinaryVL_XI<vop, instruction_name, "VX",
                          vti.Vector, vti.Vector, vti.Mask, vti.SEW,
                          vti.LMul, vti.RegClass, vti.RegClass,
                          SplatPat, GPR>;
    def : VPatBinaryVL_XI<vop, instruction_name, "VI",
                          vti.Vector, vti.Vector, vti.Mask, vti.SEW,
                          vti.LMul, vti.RegClass, vti.RegClass,
                          !cast<ComplexPattern>(SplatPat#_#ImmType),
                          ImmType>;
  }
}

class VPatBinaryVL_VF<SDNode vop,
                      string instruction_name,
                      ValueType result_type,
                      ValueType vop_type,
                      ValueType mask_type,
                      int sew,
                      LMULInfo vlmul,
                      VReg RetClass,
                      VReg vop_reg_class,
                      RegisterClass scalar_reg_class> :
    Pat<(result_type (vop (vop_type vop_reg_class:$rs1),
                          (vop_type (SplatFPOp scalar_reg_class:$rs2)),
                          (mask_type true_mask),
                          (XLenVT (VLOp GPR:$vl)))),
        (!cast<Instruction>(instruction_name#"_"#vlmul.MX)
                     vop_reg_class:$rs1,
                     scalar_reg_class:$rs2,
                     GPR:$vl, sew)>;

multiclass VPatBinaryFPVL_VV_VF<SDNode vop, string instruction_name> {
  foreach vti = AllFloatVectors in {
    def : VPatBinaryVL_VV<vop, instruction_name,
                          vti.Vector, vti.Vector, vti.Mask, vti.SEW,
                          vti.LMul, vti.RegClass, vti.RegClass>;
    def : VPatBinaryVL_VF<vop, instruction_name#"_V"#vti.ScalarSuffix,
                          vti.Vector, vti.Vector, vti.Mask, vti.SEW,
                          vti.LMul, vti.RegClass, vti.RegClass,
                          vti.ScalarRegClass>;
  }
}

multiclass VPatBinaryFPVL_R_VF<SDNode vop, string instruction_name> {
  foreach fvti = AllFloatVectors in
    def : Pat<(fvti.Vector (vop (SplatFPOp fvti.ScalarRegClass:$rs2),
                                fvti.RegClass:$rs1,
                                (fvti.Mask true_mask),
                                (XLenVT (VLOp GPR:$vl)))),
              (!cast<Instruction>(instruction_name#"_V"#fvti.ScalarSuffix#"_"#fvti.LMul.MX)
                           fvti.RegClass:$rs1, fvti.ScalarRegClass:$rs2,
                           GPR:$vl, fvti.SEW)>;
}

multiclass VPatIntegerSetCCVL_VV<VTypeInfo vti, string instruction_name,
                                 CondCode cc> {
  def : Pat<(vti.Mask (riscv_setcc_vl (vti.Vector vti.RegClass:$rs1),
                                      vti.RegClass:$rs2, cc,
                                      (vti.Mask true_mask),
                                      (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>(instruction_name#"_VV_"#vti.LMul.MX)
                         vti.RegClass:$rs1, vti.RegClass:$rs2, GPR:$vl,
                         vti.SEW)>;
}

// Inherits from VPatIntegerSetCCVL_VV and adds a pattern with operands swapped.
multiclass VPatIntegerSetCCVL_VV_Swappable<VTypeInfo vti, string instruction_name,
                                           CondCode cc, CondCode invcc> :
  VPatIntegerSetCCVL_VV<vti, instruction_name, cc> {
  def : Pat<(vti.Mask (riscv_setcc_vl (vti.Vector vti.RegClass:$rs2),
                                      vti.RegClass:$rs1, invcc,
                                      (vti.Mask true_mask),
                                      (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>(instruction_name#"_VV_"#vti.LMul.MX)
                         vti.RegClass:$rs1, vti.RegClass:$rs2, GPR:$vl,
                         vti.SEW)>;
}

multiclass VPatIntegerSetCCVL_VX_Swappable<VTypeInfo vti, string instruction_name,
                                           CondCode cc, CondCode invcc> {
  defvar instruction = !cast<Instruction>(instruction_name#"_VX_"#vti.LMul.MX);
  def : Pat<(vti.Mask (riscv_setcc_vl (vti.Vector vti.RegClass:$rs1),
                                      (SplatPat GPR:$rs2), cc,
                                      (vti.Mask true_mask),
                                      (XLenVT (VLOp GPR:$vl)))),
            (instruction vti.RegClass:$rs1, GPR:$rs2, GPR:$vl, vti.SEW)>;
  def : Pat<(vti.Mask (riscv_setcc_vl (SplatPat GPR:$rs2),
                                      (vti.Vector vti.RegClass:$rs1), invcc,
                                      (vti.Mask true_mask),
                                      (XLenVT (VLOp GPR:$vl)))),
            (instruction vti.RegClass:$rs1, GPR:$rs2, GPR:$vl, vti.SEW)>;
}

multiclass VPatIntegerSetCCVL_VI_Swappable<VTypeInfo vti, string instruction_name,
                                           CondCode cc, CondCode invcc> {
  defvar instruction = !cast<Instruction>(instruction_name#"_VI_"#vti.LMul.MX);
  defvar ImmPat = !cast<ComplexPattern>("sew"#vti.SEW#"simm5");
  def : Pat<(vti.Mask (riscv_setcc_vl (vti.Vector vti.RegClass:$rs1),
                                      (SplatPat_simm5 simm5:$rs2), cc,
                                      (vti.Mask true_mask),
                                      (XLenVT (VLOp GPR:$vl)))),
            (instruction vti.RegClass:$rs1, XLenVT:$rs2, GPR:$vl, vti.SEW)>;
  def : Pat<(vti.Mask (riscv_setcc_vl (SplatPat_simm5 simm5:$rs2),
                                      (vti.Vector vti.RegClass:$rs1), invcc,
                                      (vti.Mask true_mask),
                                      (XLenVT (VLOp GPR:$vl)))),
            (instruction vti.RegClass:$rs1, simm5:$rs2, GPR:$vl, vti.SEW)>;
}

//===----------------------------------------------------------------------===//
// Patterns.
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtV] in {

// 7.4. Vector Unit-Stride Instructions
foreach vti = AllVectors in {
  defvar load_instr = !cast<Instruction>("PseudoVLE"#vti.SEW#"_V_"#vti.LMul.MX);
  defvar store_instr = !cast<Instruction>("PseudoVSE"#vti.SEW#"_V_"#vti.LMul.MX);
  // Load
  def : Pat<(vti.Vector (riscv_vle_vl BaseAddr:$rs1, (XLenVT (VLOp GPR:$vl)))),
            (load_instr BaseAddr:$rs1, GPR:$vl, vti.SEW)>;
  // Store
  def : Pat<(riscv_vse_vl (vti.Vector vti.RegClass:$rs2), BaseAddr:$rs1,
                          (XLenVT (VLOp GPR:$vl))),
            (store_instr vti.RegClass:$rs2, BaseAddr:$rs1, GPR:$vl, vti.SEW)>;
}

foreach mti = AllMasks in {
  defvar load_instr = !cast<Instruction>("PseudoVLE1_V_"#mti.BX);
  defvar store_instr = !cast<Instruction>("PseudoVSE1_V_"#mti.BX);
  def : Pat<(mti.Mask (riscv_vle_vl BaseAddr:$rs1, (XLenVT (VLOp GPR:$vl)))),
            (load_instr BaseAddr:$rs1, GPR:$vl, mti.SEW)>;
  def : Pat<(riscv_vse_vl (mti.Mask VR:$rs2), BaseAddr:$rs1,
                          (XLenVT (VLOp GPR:$vl))),
            (store_instr VR:$rs2, BaseAddr:$rs1, GPR:$vl, mti.SEW)>;
}

// 12.1. Vector Single-Width Integer Add and Subtract
defm "" : VPatBinaryVL_VV_VX_VI<riscv_add_vl, "PseudoVADD">;
defm "" : VPatBinaryVL_VV_VX<riscv_sub_vl, "PseudoVSUB">;
// Handle VRSUB specially since it's the only integer binary op with reversed
// pattern operands
foreach vti = AllIntegerVectors in {
  def : Pat<(riscv_sub_vl (vti.Vector (SplatPat GPR:$rs2)),
                          (vti.Vector vti.RegClass:$rs1), (vti.Mask true_mask),
                          (XLenVT (VLOp GPR:$vl))),
            (!cast<Instruction>("PseudoVRSUB_VX_"# vti.LMul.MX)
                 vti.RegClass:$rs1, GPR:$rs2, GPR:$vl, vti.SEW)>;
  def : Pat<(riscv_sub_vl (vti.Vector (SplatPat_simm5 simm5:$rs2)),
                          (vti.Vector vti.RegClass:$rs1), (vti.Mask true_mask),
                          (XLenVT (VLOp GPR:$vl))),
            (!cast<Instruction>("PseudoVRSUB_VI_"# vti.LMul.MX)
                 vti.RegClass:$rs1, simm5:$rs2, GPR:$vl, vti.SEW)>;
}

// 12.5. Vector Bitwise Logical Instructions
defm "" : VPatBinaryVL_VV_VX_VI<riscv_and_vl, "PseudoVAND">;
defm "" : VPatBinaryVL_VV_VX_VI<riscv_or_vl,  "PseudoVOR">;
defm "" : VPatBinaryVL_VV_VX_VI<riscv_xor_vl, "PseudoVXOR">;

// 12.6. Vector Single-Width Bit Shift Instructions
defm "" : VPatBinaryVL_VV_VX_VI<riscv_shl_vl, "PseudoVSLL", uimm5>;
defm "" : VPatBinaryVL_VV_VX_VI<riscv_srl_vl, "PseudoVSRL", uimm5>;
defm "" : VPatBinaryVL_VV_VX_VI<riscv_sra_vl, "PseudoVSRA", uimm5>;

// 12.8. Vector Integer Comparison Instructions
foreach vti = AllIntegerVectors in {
  defm "" : VPatIntegerSetCCVL_VV<vti, "PseudoVMSEQ", SETEQ>;
  defm "" : VPatIntegerSetCCVL_VV<vti, "PseudoVMSNE", SETNE>;

  defm "" : VPatIntegerSetCCVL_VV_Swappable<vti, "PseudoVMSLT",  SETLT,  SETGT>;
  defm "" : VPatIntegerSetCCVL_VV_Swappable<vti, "PseudoVMSLTU", SETULT, SETUGT>;
  defm "" : VPatIntegerSetCCVL_VV_Swappable<vti, "PseudoVMSLE",  SETLE,  SETGE>;
  defm "" : VPatIntegerSetCCVL_VV_Swappable<vti, "PseudoVMSLEU", SETULE, SETUGE>;

  defm "" : VPatIntegerSetCCVL_VX_Swappable<vti, "PseudoVMSEQ",  SETEQ,  SETEQ>;
  defm "" : VPatIntegerSetCCVL_VX_Swappable<vti, "PseudoVMSNE",  SETNE,  SETNE>;
  defm "" : VPatIntegerSetCCVL_VX_Swappable<vti, "PseudoVMSLT",  SETLT,  SETGT>;
  defm "" : VPatIntegerSetCCVL_VX_Swappable<vti, "PseudoVMSLTU", SETULT, SETUGT>;
  defm "" : VPatIntegerSetCCVL_VX_Swappable<vti, "PseudoVMSLE",  SETLE,  SETGE>;
  defm "" : VPatIntegerSetCCVL_VX_Swappable<vti, "PseudoVMSLEU", SETULE, SETUGE>;
  defm "" : VPatIntegerSetCCVL_VX_Swappable<vti, "PseudoVMSGT",  SETGT,  SETLT>;
  defm "" : VPatIntegerSetCCVL_VX_Swappable<vti, "PseudoVMSGTU", SETUGT, SETULT>;
  // There is no VMSGE(U)_VX instruction

  // FIXME: Support immediate forms of these by choosing SGT and decrementing
  // the immediate
  defm "" : VPatIntegerSetCCVL_VI_Swappable<vti, "PseudoVMSEQ",  SETEQ,  SETEQ>;
  defm "" : VPatIntegerSetCCVL_VI_Swappable<vti, "PseudoVMSNE",  SETNE,  SETNE>;
  defm "" : VPatIntegerSetCCVL_VI_Swappable<vti, "PseudoVMSLE",  SETLE,  SETGE>;
  defm "" : VPatIntegerSetCCVL_VI_Swappable<vti, "PseudoVMSLEU", SETULE, SETUGE>;
} // foreach vti = AllIntegerVectors

// 12.9. Vector Integer Min/Max Instructions
defm "" : VPatBinaryVL_VV_VX<riscv_umin_vl, "PseudoVMINU">;
defm "" : VPatBinaryVL_VV_VX<riscv_smin_vl, "PseudoVMIN">;
defm "" : VPatBinaryVL_VV_VX<riscv_umax_vl, "PseudoVMAXU">;
defm "" : VPatBinaryVL_VV_VX<riscv_smax_vl, "PseudoVMAX">;

// 12.10. Vector Single-Width Integer Multiply Instructions
defm "" : VPatBinaryVL_VV_VX<riscv_mul_vl, "PseudoVMUL">;

// 12.11. Vector Integer Divide Instructions
defm "" : VPatBinaryVL_VV_VX<riscv_udiv_vl, "PseudoVDIVU">;
defm "" : VPatBinaryVL_VV_VX<riscv_sdiv_vl, "PseudoVDIV">;
defm "" : VPatBinaryVL_VV_VX<riscv_urem_vl, "PseudoVREMU">;
defm "" : VPatBinaryVL_VV_VX<riscv_srem_vl, "PseudoVREM">;

// 12.17. Vector Integer Move Instructions
foreach vti = AllIntegerVectors in {
  def : Pat<(vti.Vector (riscv_vmv_v_x_vl GPR:$rs2, (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVMV_V_X_"#vti.LMul.MX)
             $rs2, GPR:$vl, vti.SEW)>;
  defvar ImmPat = !cast<ComplexPattern>("sew"#vti.SEW#"simm5");
  def : Pat<(vti.Vector (riscv_vmv_v_x_vl (ImmPat XLenVT:$imm5),
                                          (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVMV_V_I_"#vti.LMul.MX)
             XLenVT:$imm5, GPR:$vl, vti.SEW)>;
}

} // Predicates = [HasStdExtV]

let Predicates = [HasStdExtV, HasStdExtF] in {

// 14.2. Vector Single-Width Floating-Point Add/Subtract Instructions
defm "" : VPatBinaryFPVL_VV_VF<riscv_fadd_vl, "PseudoVFADD">;
defm "" : VPatBinaryFPVL_VV_VF<riscv_fsub_vl, "PseudoVFSUB">;
defm "" : VPatBinaryFPVL_R_VF<riscv_fsub_vl, "PseudoVFRSUB">;

// 14.4. Vector Single-Width Floating-Point Multiply/Divide Instructions
defm "" : VPatBinaryFPVL_VV_VF<riscv_fmul_vl, "PseudoVFMUL">;
defm "" : VPatBinaryFPVL_VV_VF<riscv_fdiv_vl, "PseudoVFDIV">;
defm "" : VPatBinaryFPVL_R_VF<riscv_fdiv_vl, "PseudoVFRDIV">;

// 14.6 Vector Single-Width Floating-Point Fused Multiply-Add Instructions.
foreach vti = AllFloatVectors in {
  // NOTE: We choose VFMADD because it has the most commuting freedom. So it
  // works best with how TwoAddressInstructionPass tries commuting.
  defvar suffix = vti.LMul.MX # "_COMMUTABLE";
  def : Pat<(vti.Vector (riscv_fma_vl vti.RegClass:$rd, vti.RegClass:$rs1,
                                      vti.RegClass:$rs2, (vti.Mask true_mask),
                                      (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVFMADD_VV_"# suffix)
                 vti.RegClass:$rd, vti.RegClass:$rs1, vti.RegClass:$rs2,
                 GPR:$vl, vti.SEW)>;

  // The choice of VFMADD here is arbitrary, vfmadd.vf and vfmacc.vf are equally
  // commutable.
  def : Pat<(vti.Vector (riscv_fma_vl (SplatFPOp vti.ScalarRegClass:$rs1),
                                       vti.RegClass:$rd, vti.RegClass:$rs2,
                                       (vti.Mask true_mask),
                                       (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVFMADD_V" # vti.ScalarSuffix # "_" # suffix)
                 vti.RegClass:$rd, vti.ScalarRegClass:$rs1, vti.RegClass:$rs2,
                 GPR:$vl, vti.SEW)>;
}

foreach vti = AllFloatVectors in {
  // 14.8. Vector Floating-Point Square-Root Instruction
  def : Pat<(riscv_fsqrt_vl (vti.Vector vti.RegClass:$rs2), (vti.Mask true_mask),
                            (XLenVT (VLOp GPR:$vl))),
            (!cast<Instruction>("PseudoVFSQRT_V_"# vti.LMul.MX)
                 vti.RegClass:$rs2, GPR:$vl, vti.SEW)>;

  // 14.12. Vector Floating-Point Sign-Injection Instructions
  def : Pat<(riscv_fabs_vl (vti.Vector vti.RegClass:$rs), (vti.Mask true_mask),
                           (XLenVT (VLOp GPR:$vl))),
            (!cast<Instruction>("PseudoVFSGNJX_VV_"# vti.LMul.MX)
                 vti.RegClass:$rs, vti.RegClass:$rs, GPR:$vl, vti.SEW)>;
  // Handle fneg with VFSGNJN using the same input for both operands.
  def : Pat<(riscv_fneg_vl (vti.Vector vti.RegClass:$rs), (vti.Mask true_mask),
                           (XLenVT (VLOp GPR:$vl))),
            (!cast<Instruction>("PseudoVFSGNJN_VV_"# vti.LMul.MX)
                 vti.RegClass:$rs, vti.RegClass:$rs, GPR:$vl, vti.SEW)>;
}

// 14.16. Vector Floating-Point Move Instruction
foreach fvti = AllFloatVectors in {
  // If we're splatting fpimm0, use vmv.v.x vd, x0.
  def : Pat<(fvti.Vector (riscv_vfmv_v_f_vl
                         (fvti.Scalar (fpimm0)), (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVMV_V_I_"#fvti.LMul.MX)
             0, GPR:$vl, fvti.SEW)>;

  def : Pat<(fvti.Vector (riscv_vfmv_v_f_vl
                         (fvti.Scalar fvti.ScalarRegClass:$rs2), (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVFMV_V_" # fvti.ScalarSuffix # "_" #
                                fvti.LMul.MX)
             (fvti.Scalar fvti.ScalarRegClass:$rs2),
             GPR:$vl, fvti.SEW)>;
}

} // Predicates = [HasStdExtV, HasStdExtF]

// 16.1 Vector Mask-Register Logical Instructions
let Predicates = [HasStdExtV] in {

foreach mti = AllMasks in {
  def : Pat<(mti.Mask (riscv_vmset_vl (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVMSET_M_" # mti.BX) GPR:$vl, mti.SEW)>;
  def : Pat<(mti.Mask (riscv_vmclr_vl (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVMCLR_M_" # mti.BX) GPR:$vl, mti.SEW)>;
}

} // Predicates = [HasStdExtV]

// 17.4. Vector Register GAther Instruction
let Predicates = [HasStdExtV] in {

foreach vti = AllIntegerVectors in {
  def : Pat<(vti.Vector (riscv_vrgather_vx_vl vti.RegClass:$rs2, GPR:$rs1,
                                              (vti.Mask true_mask),
                                              (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVRGATHER_VX_"# vti.LMul.MX)
                 vti.RegClass:$rs2, GPR:$rs1, GPR:$vl, vti.SEW)>;
  def : Pat<(vti.Vector (riscv_vrgather_vx_vl vti.RegClass:$rs2, uimm5:$imm,
                                              (vti.Mask true_mask),
                                              (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVRGATHER_VI_"# vti.LMul.MX)
                 vti.RegClass:$rs2, uimm5:$imm, GPR:$vl, vti.SEW)>;
}

} // Predicates = [HasStdExtV]

let Predicates = [HasStdExtV, HasStdExtF] in {

foreach vti = AllFloatVectors in {
  def : Pat<(vti.Vector (riscv_vrgather_vx_vl vti.RegClass:$rs2, GPR:$rs1,
                                              (vti.Mask true_mask),
                                              (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVRGATHER_VX_"# vti.LMul.MX)
                 vti.RegClass:$rs2, GPR:$rs1, GPR:$vl, vti.SEW)>;
  def : Pat<(vti.Vector (riscv_vrgather_vx_vl vti.RegClass:$rs2, uimm5:$imm,
                                              (vti.Mask true_mask),
                                              (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVRGATHER_VI_"# vti.LMul.MX)
                 vti.RegClass:$rs2, uimm5:$imm, GPR:$vl, vti.SEW)>;
}

} // Predicates = [HasStdExtV, HasStdExtF]

//===----------------------------------------------------------------------===//
// Miscellaneous RISCVISD SDNodes
//===----------------------------------------------------------------------===//

def riscv_vid_vl : SDNode<"RISCVISD::VID_VL", SDTypeProfile<1, 2,
                          [SDTCisVec<0>, SDTCisVec<1>, SDTCVecEltisVT<1, i1>,
                           SDTCisSameNumEltsAs<0, 1>, SDTCisVT<2, XLenVT>]>, []>;

def SDTRVVSlide : SDTypeProfile<1, 5, [
  SDTCisVec<0>, SDTCisSameAs<1, 0>, SDTCisSameAs<2, 0>, SDTCisVT<3, XLenVT>,
  SDTCVecEltisVT<4, i1>, SDTCisSameNumEltsAs<0, 4>, SDTCisVT<5, XLenVT>
]>;

def riscv_slideup_vl   : SDNode<"RISCVISD::VSLIDEUP_VL", SDTRVVSlide, []>;
def riscv_slidedown_vl : SDNode<"RISCVISD::VSLIDEDOWN_VL", SDTRVVSlide, []>;

let Predicates = [HasStdExtV] in {

foreach vti = AllIntegerVectors in
  def : Pat<(vti.Vector (riscv_vid_vl (vti.Mask true_mask),
                                      (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVID_V_"#vti.LMul.MX) GPR:$vl, vti.SEW)>;

foreach vti = !listconcat(AllIntegerVectors, AllFloatVectors) in {
  def : Pat<(vti.Vector (riscv_slideup_vl (vti.Vector vti.RegClass:$rs3),
                                          (vti.Vector vti.RegClass:$rs1),
                                          uimm5:$rs2, (vti.Mask true_mask),
                                          (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVSLIDEUP_VI_"#vti.LMul.MX)
                vti.RegClass:$rs3, vti.RegClass:$rs1, uimm5:$rs2,
                GPR:$vl, vti.SEW)>;

  def : Pat<(vti.Vector (riscv_slideup_vl (vti.Vector vti.RegClass:$rs3),
                                          (vti.Vector vti.RegClass:$rs1),
                                          GPR:$rs2, (vti.Mask true_mask),
                                          (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVSLIDEUP_VX_"#vti.LMul.MX)
                vti.RegClass:$rs3, vti.RegClass:$rs1, GPR:$rs2,
                GPR:$vl, vti.SEW)>;

  def : Pat<(vti.Vector (riscv_slidedown_vl (vti.Vector vti.RegClass:$rs3),
                                            (vti.Vector vti.RegClass:$rs1),
                                            uimm5:$rs2, (vti.Mask true_mask),
                                            (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVSLIDEDOWN_VI_"#vti.LMul.MX)
                vti.RegClass:$rs3, vti.RegClass:$rs1, uimm5:$rs2,
                GPR:$vl, vti.SEW)>;

  def : Pat<(vti.Vector (riscv_slidedown_vl (vti.Vector vti.RegClass:$rs3),
                                            (vti.Vector vti.RegClass:$rs1),
                                            GPR:$rs2, (vti.Mask true_mask),
                                            (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVSLIDEDOWN_VX_"#vti.LMul.MX)
                vti.RegClass:$rs3, vti.RegClass:$rs1, GPR:$rs2,
                GPR:$vl, vti.SEW)>;
}

} // Predicates = [HasStdExtV]
