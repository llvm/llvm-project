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
def riscv_vmv_s_xf_vl : SDNode<"RISCVISD::VMV_S_XF_VL",
                               SDTypeProfile<1, 3, [SDTCisSameAs<0, 1>,
                                                    SDTCisVT<3, XLenVT>]>>;

def riscv_vle_vl : SDNode<"RISCVISD::VLE_VL", SDT_RISCVVLE_VL,
                          [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_vse_vl : SDNode<"RISCVISD::VSE_VL", SDT_RISCVVSE_VL,
                          [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;

def riscv_add_vl   : SDNode<"RISCVISD::ADD_VL",   SDT_RISCVIntBinOp_VL, [SDNPCommutative]>;
def riscv_sub_vl   : SDNode<"RISCVISD::SUB_VL",   SDT_RISCVIntBinOp_VL>;
def riscv_mul_vl   : SDNode<"RISCVISD::MUL_VL",   SDT_RISCVIntBinOp_VL, [SDNPCommutative]>;
def riscv_mulhs_vl : SDNode<"RISCVISD::MULHS_VL", SDT_RISCVIntBinOp_VL, [SDNPCommutative]>;
def riscv_mulhu_vl : SDNode<"RISCVISD::MULHU_VL", SDT_RISCVIntBinOp_VL, [SDNPCommutative]>;
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
def riscv_fcopysign_vl : SDNode<"RISCVISD::FCOPYSIGN_VL",  SDT_RISCVFPBinOp_VL>;

def SDT_RISCVVecFMA_VL : SDTypeProfile<1, 5, [SDTCisSameAs<0, 1>,
                                              SDTCisSameAs<0, 2>,
                                              SDTCisSameAs<0, 3>,
                                              SDTCisVec<0>, SDTCisFP<0>,
                                              SDTCVecEltisVT<4, i1>,
                                              SDTCisSameNumEltsAs<0, 4>,
                                              SDTCisVT<5, XLenVT>]>;
def riscv_fma_vl : SDNode<"RISCVISD::FMA_VL", SDT_RISCVVecFMA_VL, [SDNPCommutative]>;

def SDT_RISCVFPRoundOp_VL  : SDTypeProfile<1, 3, [
  SDTCisFP<0>, SDTCisFP<1>, SDTCisOpSmallerThanOp<0, 1>, SDTCisSameNumEltsAs<0, 1>,
  SDTCVecEltisVT<2, i1>, SDTCisSameNumEltsAs<1, 2>, SDTCisVT<3, XLenVT>
]>;
def SDT_RISCVFPExtendOp_VL  : SDTypeProfile<1, 3, [
  SDTCisFP<0>, SDTCisFP<1>, SDTCisOpSmallerThanOp<1, 0>, SDTCisSameNumEltsAs<0, 1>,
  SDTCVecEltisVT<2, i1>, SDTCisSameNumEltsAs<1, 2>, SDTCisVT<3, XLenVT>
]>;

def riscv_fpround_vl : SDNode<"RISCVISD::FP_ROUND_VL", SDT_RISCVFPRoundOp_VL>;
def riscv_fpextend_vl : SDNode<"RISCVISD::FP_EXTEND_VL", SDT_RISCVFPExtendOp_VL>;
def riscv_fncvt_rod_vl : SDNode<"RISCVISD::VFNCVT_ROD_VL", SDT_RISCVFPRoundOp_VL>;

def SDT_RISCVFP2IOp_VL  : SDTypeProfile<1, 3, [
  SDTCisInt<0>, SDTCisFP<1>, SDTCisSameNumEltsAs<0, 1>,
  SDTCVecEltisVT<2, i1>, SDTCisSameNumEltsAs<1, 2>, SDTCisVT<3, XLenVT>
]>;
def SDT_RISCVI2FPOp_VL  : SDTypeProfile<1, 3, [
  SDTCisFP<0>, SDTCisInt<1>, SDTCisSameNumEltsAs<0, 1>,
  SDTCVecEltisVT<2, i1>, SDTCisSameNumEltsAs<1, 2>, SDTCisVT<3, XLenVT>
]>;

def riscv_fp_to_sint_vl : SDNode<"RISCVISD::FP_TO_SINT_VL", SDT_RISCVFP2IOp_VL>;
def riscv_fp_to_uint_vl : SDNode<"RISCVISD::FP_TO_UINT_VL", SDT_RISCVFP2IOp_VL>;
def riscv_sint_to_fp_vl : SDNode<"RISCVISD::SINT_TO_FP_VL", SDT_RISCVI2FPOp_VL>;
def riscv_uint_to_fp_vl : SDNode<"RISCVISD::UINT_TO_FP_VL", SDT_RISCVI2FPOp_VL>;

def riscv_setcc_vl : SDNode<"RISCVISD::SETCC_VL",
                            SDTypeProfile<1, 5, [SDTCVecEltisVT<0, i1>,
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
def riscv_vrgather_vv_vl : SDNode<"RISCVISD::VRGATHER_VV_VL",
                                  SDTypeProfile<1, 4, [SDTCisVec<0>,
                                                       SDTCisSameAs<0, 1>,
                                                       SDTCisInt<2>,
                                                       SDTCisSameNumEltsAs<0, 2>,
                                                       SDTCisSameSizeAs<0, 2>,
                                                       SDTCVecEltisVT<3, i1>,
                                                       SDTCisSameNumEltsAs<0, 3>,
                                                       SDTCisVT<4, XLenVT>]>>;
def riscv_vrgatherei16_vv_vl : SDNode<"RISCVISD::VRGATHEREI16_VV_VL",
                                      SDTypeProfile<1, 4, [SDTCisVec<0>,
                                                           SDTCisSameAs<0, 1>,
                                                           SDTCisInt<2>,
                                                           SDTCVecEltisVT<2, i16>,
                                                           SDTCisSameNumEltsAs<0, 2>,
                                                           SDTCVecEltisVT<3, i1>,
                                                           SDTCisSameNumEltsAs<0, 3>,
                                                           SDTCisVT<4, XLenVT>]>>;

def riscv_vselect_vl : SDNode<"RISCVISD::VSELECT_VL",
                              SDTypeProfile<1, 4, [SDTCisVec<0>,
                                                   SDTCisVec<1>,
                                                   SDTCisSameNumEltsAs<0, 1>,
                                                   SDTCVecEltisVT<1, i1>,
                                                   SDTCisSameAs<0, 2>,
                                                   SDTCisSameAs<2, 3>,
                                                   SDTCisVT<4, XLenVT>]>>;

def SDT_RISCVMaskBinOp_VL : SDTypeProfile<1, 3, [SDTCisSameAs<0, 1>,
                                                 SDTCisSameAs<0, 2>,
                                                 SDTCVecEltisVT<0, i1>,
                                                 SDTCisVT<3, XLenVT>]>;
def riscv_vmand_vl : SDNode<"RISCVISD::VMAND_VL", SDT_RISCVMaskBinOp_VL, [SDNPCommutative]>;
def riscv_vmor_vl  : SDNode<"RISCVISD::VMOR_VL",  SDT_RISCVMaskBinOp_VL, [SDNPCommutative]>;
def riscv_vmxor_vl : SDNode<"RISCVISD::VMXOR_VL", SDT_RISCVMaskBinOp_VL, [SDNPCommutative]>;

def SDT_RISCVVMSETCLR_VL : SDTypeProfile<1, 1, [SDTCVecEltisVT<0, i1>,
                                                SDTCisVT<1, XLenVT>]>;
def riscv_vmclr_vl : SDNode<"RISCVISD::VMCLR_VL", SDT_RISCVVMSETCLR_VL>;
def riscv_vmset_vl : SDNode<"RISCVISD::VMSET_VL", SDT_RISCVVMSETCLR_VL>;

def true_mask : PatLeaf<(riscv_vmset_vl (XLenVT srcvalue))>;

def riscv_vmnot_vl : PatFrag<(ops node:$rs, node:$vl),
                             (riscv_vmxor_vl node:$rs, true_mask, node:$vl)>;

def SDT_RISCVVEXTEND_VL : SDTypeProfile<1, 3, [SDTCisVec<0>,
                                               SDTCisSameNumEltsAs<0, 1>,
                                               SDTCisSameNumEltsAs<1, 2>,
                                               SDTCVecEltisVT<2, i1>,
                                               SDTCisVT<3, XLenVT>]>;
def riscv_sext_vl : SDNode<"RISCVISD::VSEXT_VL", SDT_RISCVVEXTEND_VL>;
def riscv_zext_vl : SDNode<"RISCVISD::VZEXT_VL", SDT_RISCVVEXTEND_VL>;

def riscv_trunc_vector_vl : SDNode<"RISCVISD::TRUNCATE_VECTOR_VL",
                                   SDTypeProfile<1, 3, [SDTCisVec<0>,
                                                        SDTCisVec<1>,
                                                        SDTCisSameNumEltsAs<0, 2>,
                                                        SDTCVecEltisVT<2, i1>,
                                                        SDTCisVT<3, XLenVT>]>>;

def SDTRVVVecReduce : SDTypeProfile<1, 4, [
  SDTCisVec<0>, SDTCisVec<1>, SDTCisSameAs<0, 2>, SDTCVecEltisVT<3, i1>,
  SDTCisSameNumEltsAs<1, 3>, SDTCisVT<4, XLenVT>
]>;

foreach kind = ["ADD", "UMAX", "SMAX", "UMIN", "SMIN", "AND", "OR", "XOR",
                "FADD", "SEQ_FADD"] in
  def rvv_vecreduce_#kind#_vl : SDNode<"RISCVISD::VECREDUCE_"#kind#"_VL", SDTRVVVecReduce>;

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

multiclass VPatFPSetCCVL_VV_VF_FV<CondCode cc,
                                  string inst_name,
                                  string swapped_op_inst_name> {
  foreach fvti = AllFloatVectors in {
    def : Pat<(fvti.Mask (riscv_setcc_vl (fvti.Vector fvti.RegClass:$rs1),
                                         fvti.RegClass:$rs2,
                                         cc,
                                         (fvti.Mask true_mask),
                                         (XLenVT (VLOp GPR:$vl)))),
              (!cast<Instruction>(inst_name#"_VV_"#fvti.LMul.MX)
                  fvti.RegClass:$rs1, fvti.RegClass:$rs2, GPR:$vl, fvti.SEW)>;
    def : Pat<(fvti.Mask (riscv_setcc_vl (fvti.Vector fvti.RegClass:$rs1),
                                         (SplatFPOp fvti.ScalarRegClass:$rs2),
                                         cc,
                                         (fvti.Mask true_mask),
                                         (XLenVT (VLOp GPR:$vl)))),
              (!cast<Instruction>(inst_name#"_V"#fvti.ScalarSuffix#"_"#fvti.LMul.MX)
                  fvti.RegClass:$rs1, fvti.ScalarRegClass:$rs2,
                  GPR:$vl, fvti.SEW)>;
    def : Pat<(fvti.Mask (riscv_setcc_vl (SplatFPOp fvti.ScalarRegClass:$rs2),
                                         (fvti.Vector fvti.RegClass:$rs1),
                                         cc,
                                         (fvti.Mask true_mask),
                                         (XLenVT (VLOp GPR:$vl)))),
              (!cast<Instruction>(swapped_op_inst_name#"_V"#fvti.ScalarSuffix#"_"#fvti.LMul.MX)
                  fvti.RegClass:$rs1, fvti.ScalarRegClass:$rs2,
                  GPR:$vl, fvti.SEW)>;
  }
}

multiclass VPatExtendSDNode_V_VL<SDNode vop, string inst_name, string suffix,
                                 list <VTypeInfoToFraction> fraction_list> {
  foreach vtiTofti = fraction_list in {
    defvar vti = vtiTofti.Vti;
    defvar fti = vtiTofti.Fti;
    def : Pat<(vti.Vector (vop (fti.Vector fti.RegClass:$rs2),
                               true_mask, (XLenVT (VLOp GPR:$vl)))),
              (!cast<Instruction>(inst_name#"_"#suffix#"_"#vti.LMul.MX)
                  fti.RegClass:$rs2, GPR:$vl, vti.SEW)>;
  }
}

multiclass VPatConvertFP2ISDNode_V_VL<SDNode vop, string instruction_name> {
  foreach fvti = AllFloatVectors in {
    defvar ivti = GetIntVTypeInfo<fvti>.Vti;
    def : Pat<(ivti.Vector (vop (fvti.Vector fvti.RegClass:$rs1),
                                (fvti.Mask true_mask),
                                (XLenVT (VLOp GPR:$vl)))),
              (!cast<Instruction>(instruction_name#"_"#ivti.LMul.MX)
                  fvti.RegClass:$rs1, GPR:$vl, ivti.SEW)>;
  }
}

multiclass VPatConvertI2FPSDNode_V_VL<SDNode vop, string instruction_name> {
  foreach fvti = AllFloatVectors in {
    defvar ivti = GetIntVTypeInfo<fvti>.Vti;
    def : Pat<(fvti.Vector (vop (ivti.Vector ivti.RegClass:$rs1),
                                (ivti.Mask true_mask),
                                (XLenVT (VLOp GPR:$vl)))),
              (!cast<Instruction>(instruction_name#"_"#fvti.LMul.MX)
                  ivti.RegClass:$rs1, GPR:$vl, fvti.SEW)>;
  }
}

multiclass VPatWConvertFP2ISDNode_V_VL<SDNode vop, string instruction_name> {
  foreach fvtiToFWti = AllWidenableFloatVectors in {
    defvar fvti = fvtiToFWti.Vti;
    defvar iwti = GetIntVTypeInfo<fvtiToFWti.Wti>.Vti;
    def : Pat<(iwti.Vector (vop (fvti.Vector fvti.RegClass:$rs1),
                                (fvti.Mask true_mask),
                                (XLenVT (VLOp GPR:$vl)))),
              (!cast<Instruction>(instruction_name#"_"#fvti.LMul.MX)
                  fvti.RegClass:$rs1, GPR:$vl, fvti.SEW)>;
  }
}

multiclass VPatWConvertI2FPSDNode_V_VL<SDNode vop, string instruction_name> {
  foreach vtiToWti = AllWidenableIntToFloatVectors in {
    defvar ivti = vtiToWti.Vti;
    defvar fwti = vtiToWti.Wti;
    def : Pat<(fwti.Vector (vop (ivti.Vector ivti.RegClass:$rs1),
                                (ivti.Mask true_mask),
                                (XLenVT (VLOp GPR:$vl)))),
              (!cast<Instruction>(instruction_name#"_"#ivti.LMul.MX)
                  ivti.RegClass:$rs1, GPR:$vl, ivti.SEW)>;
  }
}

multiclass VPatNConvertFP2ISDNode_V_VL<SDNode vop, string instruction_name> {
  foreach vtiToWti = AllWidenableIntToFloatVectors in {
    defvar vti = vtiToWti.Vti;
    defvar fwti = vtiToWti.Wti;
    def : Pat<(vti.Vector (vop (fwti.Vector fwti.RegClass:$rs1),
                               (fwti.Mask true_mask),
                               (XLenVT (VLOp GPR:$vl)))),
              (!cast<Instruction>(instruction_name#"_"#vti.LMul.MX)
                  fwti.RegClass:$rs1, GPR:$vl, vti.SEW)>;
  }
}

multiclass VPatNConvertI2FPSDNode_V_VL<SDNode vop, string instruction_name> {
  foreach fvtiToFWti = AllWidenableFloatVectors in {
    defvar fvti = fvtiToFWti.Vti;
    defvar iwti = GetIntVTypeInfo<fvtiToFWti.Wti>.Vti;
    def : Pat<(fvti.Vector (vop (iwti.Vector iwti.RegClass:$rs1),
                                (iwti.Mask true_mask),
                                (XLenVT (VLOp GPR:$vl)))),
              (!cast<Instruction>(instruction_name#"_"#fvti.LMul.MX)
                  iwti.RegClass:$rs1, GPR:$vl, fvti.SEW)>;
  }
}

multiclass VPatReductionVL<SDNode vop, string instruction_name, bit is_float> {
  foreach vti = !if(is_float, AllFloatVectors, AllIntegerVectors) in {
    defvar vti_m1 = !cast<VTypeInfo>(!if(is_float, "VF", "VI") # vti.SEW # "M1");
    def: Pat<(vti_m1.Vector (vop (vti.Vector vti.RegClass:$rs1), VR:$rs2,
                                 (vti.Mask true_mask),
                                 (XLenVT (VLOp GPR:$vl)))),
        (!cast<Instruction>(instruction_name#"_VS_"#vti.LMul.MX)
            (vti_m1.Vector (IMPLICIT_DEF)),
            (vti.Vector vti.RegClass:$rs1),
            (vti_m1.Vector VR:$rs2),
            GPR:$vl, vti.SEW)>;
  }
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

// 12.3. Vector Integer Extension
defm "" : VPatExtendSDNode_V_VL<riscv_zext_vl, "PseudoVZEXT", "VF2",
                                AllFractionableVF2IntVectors>;
defm "" : VPatExtendSDNode_V_VL<riscv_sext_vl, "PseudoVSEXT", "VF2",
                                AllFractionableVF2IntVectors>;
defm "" : VPatExtendSDNode_V_VL<riscv_zext_vl, "PseudoVZEXT", "VF4",
                                AllFractionableVF4IntVectors>;
defm "" : VPatExtendSDNode_V_VL<riscv_sext_vl, "PseudoVSEXT", "VF4",
                                AllFractionableVF4IntVectors>;
defm "" : VPatExtendSDNode_V_VL<riscv_zext_vl, "PseudoVZEXT", "VF8",
                                AllFractionableVF8IntVectors>;
defm "" : VPatExtendSDNode_V_VL<riscv_sext_vl, "PseudoVSEXT", "VF8",
                                AllFractionableVF8IntVectors>;

// 12.5. Vector Bitwise Logical Instructions
defm "" : VPatBinaryVL_VV_VX_VI<riscv_and_vl, "PseudoVAND">;
defm "" : VPatBinaryVL_VV_VX_VI<riscv_or_vl,  "PseudoVOR">;
defm "" : VPatBinaryVL_VV_VX_VI<riscv_xor_vl, "PseudoVXOR">;

// 12.6. Vector Single-Width Bit Shift Instructions
defm "" : VPatBinaryVL_VV_VX_VI<riscv_shl_vl, "PseudoVSLL", uimm5>;
defm "" : VPatBinaryVL_VV_VX_VI<riscv_srl_vl, "PseudoVSRL", uimm5>;
defm "" : VPatBinaryVL_VV_VX_VI<riscv_sra_vl, "PseudoVSRA", uimm5>;

// 12.7. Vector Narrowing Integer Right Shift Instructions
foreach vtiTofti = AllFractionableVF2IntVectors in {
  defvar vti = vtiTofti.Vti;
  defvar fti = vtiTofti.Fti;
  def : Pat<(fti.Vector (riscv_trunc_vector_vl (vti.Vector vti.RegClass:$rs1),
                                               (vti.Mask true_mask),
                                               (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVNSRL_WI_"#fti.LMul.MX)
                vti.RegClass:$rs1, 0, GPR:$vl, fti.SEW)>;
}

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
defm "" : VPatBinaryVL_VV_VX<riscv_mulhs_vl, "PseudoVMULH">;
defm "" : VPatBinaryVL_VV_VX<riscv_mulhu_vl, "PseudoVMULHU">;

// 12.11. Vector Integer Divide Instructions
defm "" : VPatBinaryVL_VV_VX<riscv_udiv_vl, "PseudoVDIVU">;
defm "" : VPatBinaryVL_VV_VX<riscv_sdiv_vl, "PseudoVDIV">;
defm "" : VPatBinaryVL_VV_VX<riscv_urem_vl, "PseudoVREMU">;
defm "" : VPatBinaryVL_VV_VX<riscv_srem_vl, "PseudoVREM">;

// 12.15. Vector Integer Merge Instructions
foreach vti = AllIntegerVectors in {
  def : Pat<(vti.Vector (riscv_vselect_vl (vti.Mask VMV0:$vm),
                                          vti.RegClass:$rs1,
                                          vti.RegClass:$rs2,
                                          (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVMERGE_VVM_"#vti.LMul.MX)
                 vti.RegClass:$rs2, vti.RegClass:$rs1, VMV0:$vm,
                 GPR:$vl, vti.SEW)>;

  def : Pat<(vti.Vector (riscv_vselect_vl (vti.Mask VMV0:$vm),
                                          (SplatPat XLenVT:$rs1),
                                          vti.RegClass:$rs2,
                                          (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVMERGE_VXM_"#vti.LMul.MX)
                 vti.RegClass:$rs2, GPR:$rs1, VMV0:$vm, GPR:$vl, vti.SEW)>;

  def : Pat<(vti.Vector (riscv_vselect_vl (vti.Mask VMV0:$vm),
                                          (SplatPat_simm5 simm5:$rs1),
                                          vti.RegClass:$rs2,
                                          (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVMERGE_VIM_"#vti.LMul.MX)
                 vti.RegClass:$rs2, simm5:$rs1, VMV0:$vm, GPR:$vl, vti.SEW)>;
}

// 12.16. Vector Integer Move Instructions
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

// 15.1. Vector Single-Width Integer Reduction Instructions
let Predicates = [HasStdExtV] in {
defm "" : VPatReductionVL<rvv_vecreduce_ADD_vl,  "PseudoVREDSUM", /*is_float*/0>;
defm "" : VPatReductionVL<rvv_vecreduce_UMAX_vl, "PseudoVREDMAXU", /*is_float*/0>;
defm "" : VPatReductionVL<rvv_vecreduce_SMAX_vl, "PseudoVREDMAX", /*is_float*/0>;
defm "" : VPatReductionVL<rvv_vecreduce_UMIN_vl, "PseudoVREDMINU", /*is_float*/0>;
defm "" : VPatReductionVL<rvv_vecreduce_SMIN_vl, "PseudoVREDMIN", /*is_float*/0>;
defm "" : VPatReductionVL<rvv_vecreduce_AND_vl,  "PseudoVREDAND", /*is_float*/0>;
defm "" : VPatReductionVL<rvv_vecreduce_OR_vl,   "PseudoVREDOR", /*is_float*/0>;
defm "" : VPatReductionVL<rvv_vecreduce_XOR_vl,  "PseudoVREDXOR", /*is_float*/0>;
} // Predicates = [HasStdExtV]

// 15.3. Vector Single-Width Floating-Point Reduction Instructions
let Predicates = [HasStdExtV, HasStdExtF] in {
defm "" : VPatReductionVL<rvv_vecreduce_SEQ_FADD_vl, "PseudoVFREDOSUM", /*is_float*/1>;
defm "" : VPatReductionVL<rvv_vecreduce_FADD_vl,     "PseudoVFREDSUM", /*is_float*/1>;
} // Predicates = [HasStdExtV, HasStdExtF]

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
  def : Pat<(vti.Vector (riscv_fma_vl vti.RegClass:$rs1, vti.RegClass:$rd,
                                      vti.RegClass:$rs2, (vti.Mask true_mask),
                                      (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVFMADD_VV_"# suffix)
                 vti.RegClass:$rd, vti.RegClass:$rs1, vti.RegClass:$rs2,
                 GPR:$vl, vti.SEW)>;
  def : Pat<(vti.Vector (riscv_fma_vl vti.RegClass:$rs1, vti.RegClass:$rd,
                                      (riscv_fneg_vl vti.RegClass:$rs2,
                                                     (vti.Mask true_mask),
                                                     (XLenVT (VLOp GPR:$vl))),
                                      (vti.Mask true_mask),
                                      (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVFMSUB_VV_"# suffix)
                 vti.RegClass:$rd, vti.RegClass:$rs1, vti.RegClass:$rs2,
                 GPR:$vl, vti.SEW)>;
  def : Pat<(vti.Vector (riscv_fma_vl (riscv_fneg_vl vti.RegClass:$rs1,
                                                     (vti.Mask true_mask),
                                                     (XLenVT (VLOp GPR:$vl))),
                                      vti.RegClass:$rd,
                                      (riscv_fneg_vl vti.RegClass:$rs2,
                                                     (vti.Mask true_mask),
                                                     (XLenVT (VLOp GPR:$vl))),
                                      (vti.Mask true_mask),
                                      (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVFNMADD_VV_"# suffix)
                 vti.RegClass:$rd, vti.RegClass:$rs1, vti.RegClass:$rs2,
                 GPR:$vl, vti.SEW)>;
  def : Pat<(vti.Vector (riscv_fma_vl (riscv_fneg_vl vti.RegClass:$rs1,
                                                     (vti.Mask true_mask),
                                                     (XLenVT (VLOp GPR:$vl))),
                                      vti.RegClass:$rd, vti.RegClass:$rs2,
                                      (vti.Mask true_mask),
                                      (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVFNMSUB_VV_"# suffix)
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
  def : Pat<(vti.Vector (riscv_fma_vl (SplatFPOp vti.ScalarRegClass:$rs1),
                                       vti.RegClass:$rd,
                                       (riscv_fneg_vl vti.RegClass:$rs2,
                                                      (vti.Mask true_mask),
                                                      (XLenVT (VLOp GPR:$vl))),
                                       (vti.Mask true_mask),
                                       (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVFMSUB_V" # vti.ScalarSuffix # "_" # suffix)
                 vti.RegClass:$rd, vti.ScalarRegClass:$rs1, vti.RegClass:$rs2,
                 GPR:$vl, vti.SEW)>;
  def : Pat<(vti.Vector (riscv_fma_vl (SplatFPOp vti.ScalarRegClass:$rs1),
                                       (riscv_fneg_vl vti.RegClass:$rd,
                                                      (vti.Mask true_mask),
                                                      (XLenVT (VLOp GPR:$vl))),
                                       (riscv_fneg_vl vti.RegClass:$rs2,
                                                      (vti.Mask true_mask),
                                                      (XLenVT (VLOp GPR:$vl))),
                                       (vti.Mask true_mask),
                                       (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVFNMADD_V" # vti.ScalarSuffix # "_" # suffix)
                 vti.RegClass:$rd, vti.ScalarRegClass:$rs1, vti.RegClass:$rs2,
                 GPR:$vl, vti.SEW)>;
  def : Pat<(vti.Vector (riscv_fma_vl (SplatFPOp vti.ScalarRegClass:$rs1),
                                       (riscv_fneg_vl vti.RegClass:$rd,
                                                      (vti.Mask true_mask),
                                                      (XLenVT (VLOp GPR:$vl))),
                                       vti.RegClass:$rs2,
                                       (vti.Mask true_mask),
                                       (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVFNMSUB_V" # vti.ScalarSuffix # "_" # suffix)
                 vti.RegClass:$rd, vti.ScalarRegClass:$rs1, vti.RegClass:$rs2,
                 GPR:$vl, vti.SEW)>;

  // The splat might be negated.
  def : Pat<(vti.Vector (riscv_fma_vl (riscv_fneg_vl (SplatFPOp vti.ScalarRegClass:$rs1),
                                                     (vti.Mask true_mask),
                                                     (XLenVT (VLOp GPR:$vl))),
                                       vti.RegClass:$rd,
                                       (riscv_fneg_vl vti.RegClass:$rs2,
                                                      (vti.Mask true_mask),
                                                      (XLenVT (VLOp GPR:$vl))),
                                       (vti.Mask true_mask),
                                       (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVFNMADD_V" # vti.ScalarSuffix # "_" # suffix)
                 vti.RegClass:$rd, vti.ScalarRegClass:$rs1, vti.RegClass:$rs2,
                 GPR:$vl, vti.SEW)>;
  def : Pat<(vti.Vector (riscv_fma_vl (riscv_fneg_vl (SplatFPOp vti.ScalarRegClass:$rs1),
                                                     (vti.Mask true_mask),
                                                     (XLenVT (VLOp GPR:$vl))),
                                       vti.RegClass:$rd, vti.RegClass:$rs2,
                                       (vti.Mask true_mask),
                                       (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVFNMSUB_V" # vti.ScalarSuffix # "_" # suffix)
                 vti.RegClass:$rd, vti.ScalarRegClass:$rs1, vti.RegClass:$rs2,
                 GPR:$vl, vti.SEW)>;
}

// 14.13. Vector Floating-Point Compare Instructions
defm "" : VPatFPSetCCVL_VV_VF_FV<SETEQ,  "PseudoVMFEQ", "PseudoVMFEQ">;
defm "" : VPatFPSetCCVL_VV_VF_FV<SETOEQ, "PseudoVMFEQ", "PseudoVMFEQ">;

defm "" : VPatFPSetCCVL_VV_VF_FV<SETNE,  "PseudoVMFNE", "PseudoVMFNE">;
defm "" : VPatFPSetCCVL_VV_VF_FV<SETUNE, "PseudoVMFNE", "PseudoVMFNE">;

defm "" : VPatFPSetCCVL_VV_VF_FV<SETLT,  "PseudoVMFLT", "PseudoVMFGT">;
defm "" : VPatFPSetCCVL_VV_VF_FV<SETOLT, "PseudoVMFLT", "PseudoVMFGT">;

defm "" : VPatFPSetCCVL_VV_VF_FV<SETLE,  "PseudoVMFLE", "PseudoVMFGE">;
defm "" : VPatFPSetCCVL_VV_VF_FV<SETOLE, "PseudoVMFLE", "PseudoVMFGE">;

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
  def : Pat<(riscv_fcopysign_vl (vti.Vector vti.RegClass:$rs1),
                                (vti.Vector vti.RegClass:$rs2),
                                (vti.Mask true_mask),
                                (XLenVT (VLOp GPR:$vl))),
            (!cast<Instruction>("PseudoVFSGNJ_VV_"# vti.LMul.MX)
                 vti.RegClass:$rs1, vti.RegClass:$rs2, GPR:$vl, vti.SEW)>;
  def : Pat<(riscv_fcopysign_vl (vti.Vector vti.RegClass:$rs1),
                                (riscv_fneg_vl vti.RegClass:$rs2,
                                               (vti.Mask true_mask),
                                               (XLenVT (VLOp GPR:$vl))),
                                (vti.Mask true_mask),
                                (XLenVT (VLOp GPR:$vl))),
            (!cast<Instruction>("PseudoVFSGNJN_VV_"# vti.LMul.MX)
                 vti.RegClass:$rs1, vti.RegClass:$rs2, GPR:$vl, vti.SEW)>;

  def : Pat<(riscv_fcopysign_vl (vti.Vector vti.RegClass:$rs1),
                                (SplatFPOp vti.ScalarRegClass:$rs2),
                                (vti.Mask true_mask),
                                (XLenVT (VLOp GPR:$vl))),
            (!cast<Instruction>("PseudoVFSGNJ_V"#vti.ScalarSuffix#"_"# vti.LMul.MX)
                 vti.RegClass:$rs1, vti.ScalarRegClass:$rs2, GPR:$vl, vti.SEW)>;
}

foreach fvti = AllFloatVectors in {
  // Floating-point vselects:
  // 12.15. Vector Integer Merge Instructions
  // 14.15. Vector Floating-Point Merge Instruction
  def : Pat<(fvti.Vector (riscv_vselect_vl (fvti.Mask VMV0:$vm),
                                           fvti.RegClass:$rs1,
                                           fvti.RegClass:$rs2,
                                           (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVMERGE_VVM_"#fvti.LMul.MX)
                 fvti.RegClass:$rs2, fvti.RegClass:$rs1, VMV0:$vm,
                 GPR:$vl, fvti.SEW)>;

  def : Pat<(fvti.Vector (riscv_vselect_vl (fvti.Mask VMV0:$vm),
                                           (SplatFPOp fvti.ScalarRegClass:$rs1),
                                           fvti.RegClass:$rs2,
                                           (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVFMERGE_V"#fvti.ScalarSuffix#"M_"#fvti.LMul.MX)
                 fvti.RegClass:$rs2,
                 (fvti.Scalar fvti.ScalarRegClass:$rs1),
                 VMV0:$vm, GPR:$vl, fvti.SEW)>;

  def : Pat<(fvti.Vector (riscv_vselect_vl (fvti.Mask VMV0:$vm),
                                           (SplatFPOp (fvti.Scalar fpimm0)),
                                           fvti.RegClass:$rs2,
                                           (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVMERGE_VIM_"#fvti.LMul.MX)
                 fvti.RegClass:$rs2, 0, VMV0:$vm, GPR:$vl, fvti.SEW)>;

  // 14.16. Vector Floating-Point Move Instruction
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

  // 14.17. Vector Single-Width Floating-Point/Integer Type-Convert Instructions
  defm "" : VPatConvertFP2ISDNode_V_VL<riscv_fp_to_sint_vl, "PseudoVFCVT_RTZ_X_F_V">;
  defm "" : VPatConvertFP2ISDNode_V_VL<riscv_fp_to_uint_vl, "PseudoVFCVT_RTZ_XU_F_V">;
  defm "" : VPatConvertI2FPSDNode_V_VL<riscv_sint_to_fp_vl, "PseudoVFCVT_F_X_V">;
  defm "" : VPatConvertI2FPSDNode_V_VL<riscv_uint_to_fp_vl, "PseudoVFCVT_F_XU_V">;

  // 14.18. Widening Floating-Point/Integer Type-Convert Instructions
  defm "" : VPatWConvertFP2ISDNode_V_VL<riscv_fp_to_sint_vl, "PseudoVFWCVT_RTZ_X_F_V">;
  defm "" : VPatWConvertFP2ISDNode_V_VL<riscv_fp_to_uint_vl, "PseudoVFWCVT_RTZ_XU_F_V">;
  defm "" : VPatWConvertI2FPSDNode_V_VL<riscv_sint_to_fp_vl, "PseudoVFWCVT_F_X_V">;
  defm "" : VPatWConvertI2FPSDNode_V_VL<riscv_uint_to_fp_vl, "PseudoVFWCVT_F_XU_V">;
  foreach fvtiToFWti = AllWidenableFloatVectors in {
    defvar fvti = fvtiToFWti.Vti;
    defvar fwti = fvtiToFWti.Wti;
    def : Pat<(fwti.Vector (riscv_fpextend_vl (fvti.Vector fvti.RegClass:$rs1),
                                              (fvti.Mask true_mask),
                                              (XLenVT (VLOp GPR:$vl)))),
              (!cast<Instruction>("PseudoVFWCVT_F_F_V_"#fvti.LMul.MX)
                  fvti.RegClass:$rs1, GPR:$vl, fvti.SEW)>;
  }

  // 14.19 Narrowing Floating-Point/Integer Type-Convert Instructions
  defm "" : VPatNConvertFP2ISDNode_V_VL<riscv_fp_to_sint_vl, "PseudoVFNCVT_RTZ_X_F_W">;
  defm "" : VPatNConvertFP2ISDNode_V_VL<riscv_fp_to_uint_vl, "PseudoVFNCVT_RTZ_XU_F_W">;
  defm "" : VPatNConvertI2FPSDNode_V_VL<riscv_sint_to_fp_vl, "PseudoVFNCVT_F_X_W">;
  defm "" : VPatNConvertI2FPSDNode_V_VL<riscv_uint_to_fp_vl, "PseudoVFNCVT_F_XU_W">;
  foreach fvtiToFWti = AllWidenableFloatVectors in {
    defvar fvti = fvtiToFWti.Vti;
    defvar fwti = fvtiToFWti.Wti;
    def : Pat<(fvti.Vector (riscv_fpround_vl (fwti.Vector fwti.RegClass:$rs1),
                                             (fwti.Mask true_mask),
                                             (XLenVT (VLOp GPR:$vl)))),
              (!cast<Instruction>("PseudoVFNCVT_F_F_W_"#fvti.LMul.MX)
                  fwti.RegClass:$rs1, GPR:$vl, fvti.SEW)>;

    def : Pat<(fvti.Vector (riscv_fncvt_rod_vl (fwti.Vector fwti.RegClass:$rs1),
                                               (fwti.Mask true_mask),
                                               (XLenVT (VLOp GPR:$vl)))),
              (!cast<Instruction>("PseudoVFNCVT_ROD_F_F_W_"#fvti.LMul.MX)
                  fwti.RegClass:$rs1, GPR:$vl, fvti.SEW)>;
  }
}

} // Predicates = [HasStdExtV, HasStdExtF]

// 16.1 Vector Mask-Register Logical Instructions
let Predicates = [HasStdExtV] in {

foreach mti = AllMasks in {
  def : Pat<(mti.Mask (riscv_vmset_vl (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVMSET_M_" # mti.BX) GPR:$vl, mti.SEW)>;
  def : Pat<(mti.Mask (riscv_vmclr_vl (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVMCLR_M_" # mti.BX) GPR:$vl, mti.SEW)>;

  def : Pat<(mti.Mask (riscv_vmand_vl VR:$rs1, VR:$rs2, (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVMAND_MM_" # mti.LMul.MX)
                 VR:$rs1, VR:$rs2, GPR:$vl, mti.SEW)>;
  def : Pat<(mti.Mask (riscv_vmor_vl VR:$rs1, VR:$rs2, (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVMOR_MM_" # mti.LMul.MX)
                 VR:$rs1, VR:$rs2, GPR:$vl, mti.SEW)>;
  def : Pat<(mti.Mask (riscv_vmxor_vl VR:$rs1, VR:$rs2, (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVMXOR_MM_" # mti.LMul.MX)
                 VR:$rs1, VR:$rs2, GPR:$vl, mti.SEW)>;

  def : Pat<(mti.Mask (riscv_vmand_vl (riscv_vmnot_vl VR:$rs1,
                                                      (XLenVT (VLOp GPR:$vl))),
                                      VR:$rs2, (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVMANDNOT_MM_" # mti.LMul.MX)
                 VR:$rs1, VR:$rs2, GPR:$vl, mti.SEW)>;
  def : Pat<(mti.Mask (riscv_vmor_vl (riscv_vmnot_vl VR:$rs1,
                                                     (XLenVT (VLOp GPR:$vl))),
                                     VR:$rs2, (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVMORNOT_MM_" # mti.LMul.MX)
                 VR:$rs1, VR:$rs2, GPR:$vl, mti.SEW)>;
  // XOR is associative so we need 2 patterns for VMXNOR.
  def : Pat<(mti.Mask (riscv_vmxor_vl (riscv_vmnot_vl VR:$rs1,
                                                      (XLenVT (VLOp GPR:$vl))),
                                     VR:$rs2, (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVMXNOR_MM_" # mti.LMul.MX)
                 VR:$rs1, VR:$rs2, GPR:$vl, mti.SEW)>;

  def : Pat<(mti.Mask (riscv_vmnot_vl (riscv_vmand_vl VR:$rs1, VR:$rs2,
                                                      (XLenVT (VLOp GPR:$vl))),
                                      (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVMNAND_MM_" # mti.LMul.MX)
                 VR:$rs1, VR:$rs2, GPR:$vl, mti.SEW)>;
  def : Pat<(mti.Mask (riscv_vmnot_vl (riscv_vmor_vl VR:$rs1, VR:$rs2,
                                                     (XLenVT (VLOp GPR:$vl))),
                                      (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVMNOR_MM_" # mti.LMul.MX)
                 VR:$rs1, VR:$rs2, GPR:$vl, mti.SEW)>;
  def : Pat<(mti.Mask (riscv_vmnot_vl (riscv_vmxor_vl VR:$rs1, VR:$rs2,
                                                      (XLenVT (VLOp GPR:$vl))),
                                      (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVMXNOR_MM_" # mti.LMul.MX)
                 VR:$rs1, VR:$rs2, GPR:$vl, mti.SEW)>;

  // Match the not idiom to the vnot.mm pseudo.
  def : Pat<(mti.Mask (riscv_vmnot_vl VR:$rs, (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVMNAND_MM_" # mti.LMul.MX)
                 VR:$rs, VR:$rs, GPR:$vl, mti.SEW)>;
}

} // Predicates = [HasStdExtV]

let Predicates = [HasStdExtV] in {
// 17.1. Integer Scalar Move Instructions
// 17.4. Vector Register Gather Instruction
foreach vti = AllIntegerVectors in {
  def : Pat<(vti.Vector (riscv_vmv_s_xf_vl (vti.Vector vti.RegClass:$merge),
                                           (XLenVT vti.ScalarRegClass:$rs1),
                                           (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVMV_S_X_"#vti.LMul.MX)
                vti.RegClass:$merge,
                (vti.Scalar vti.ScalarRegClass:$rs1), GPR:$vl, vti.SEW)>;
  def : Pat<(vti.Vector (riscv_vrgather_vv_vl vti.RegClass:$rs2,
                                              (vti.Vector vti.RegClass:$rs1),
                                              (vti.Mask true_mask),
                                              (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVRGATHER_VV_"# vti.LMul.MX)
                 vti.RegClass:$rs2, vti.RegClass:$rs1, GPR:$vl, vti.SEW)>;
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

  // emul = lmul * 16 / sew
  defvar vlmul = vti.LMul;
  defvar octuple_lmul = octuple_from_str<vlmul.MX>.ret;
  defvar octuple_emul = !srl(!mul(octuple_lmul, 16), shift_amount<vti.SEW>.val);
  if !and(!ge(octuple_emul, 1), !le(octuple_emul, 64)) then {
    defvar emul_str = octuple_to_str<octuple_emul>.ret;
    defvar ivti = !cast<VTypeInfo>("VI16" # emul_str);
    defvar inst = "PseudoVRGATHEREI16_VV_" # vti.LMul.MX # "_" # emul_str;
    def : Pat<(vti.Vector (riscv_vrgatherei16_vv_vl vti.RegClass:$rs2,
                                                    (ivti.Vector ivti.RegClass:$rs1),
                                                    (vti.Mask true_mask),
                                                    (XLenVT (VLOp GPR:$vl)))),
              (!cast<Instruction>(inst)
                   vti.RegClass:$rs2, ivti.RegClass:$rs1, GPR:$vl, vti.SEW)>;
  }
}

} // Predicates = [HasStdExtV]

let Predicates = [HasStdExtV, HasStdExtF] in {

// 17.2. Floating-Point Scalar Move Instructions
foreach vti = AllFloatVectors in {
  def : Pat<(vti.Vector (riscv_vmv_s_xf_vl (vti.Vector vti.RegClass:$merge),
                                           vti.ScalarRegClass:$rs1,
                                           (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVFMV_S_"#vti.ScalarSuffix#"_"#vti.LMul.MX)
                vti.RegClass:$merge,
                (vti.Scalar vti.ScalarRegClass:$rs1), GPR:$vl, vti.SEW)>;
  defvar ivti = GetIntVTypeInfo<vti>.Vti;
  def : Pat<(vti.Vector (riscv_vrgather_vv_vl vti.RegClass:$rs2,
                                              (ivti.Vector vti.RegClass:$rs1),
                                              (vti.Mask true_mask),
                                              (XLenVT (VLOp GPR:$vl)))),
            (!cast<Instruction>("PseudoVRGATHER_VV_"# vti.LMul.MX)
                 vti.RegClass:$rs2, vti.RegClass:$rs1, GPR:$vl, vti.SEW)>;
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

  defvar vlmul = vti.LMul;
  defvar octuple_lmul = octuple_from_str<vlmul.MX>.ret;
  defvar octuple_emul = !srl(!mul(octuple_lmul, 16), shift_amount<vti.SEW>.val);
  if !and(!ge(octuple_emul, 1), !le(octuple_emul, 64)) then {
    defvar emul_str = octuple_to_str<octuple_emul>.ret;
    defvar ivti = !cast<VTypeInfo>("VI16" # emul_str);
    defvar inst = "PseudoVRGATHEREI16_VV_" # vti.LMul.MX # "_" # emul_str;
    def : Pat<(vti.Vector (riscv_vrgatherei16_vv_vl vti.RegClass:$rs2,
                                                    (ivti.Vector ivti.RegClass:$rs1),
                                                    (vti.Mask true_mask),
                                                    (XLenVT (VLOp GPR:$vl)))),
              (!cast<Instruction>(inst)
                   vti.RegClass:$rs2, ivti.RegClass:$rs1, GPR:$vl, vti.SEW)>;
  }
}

} // Predicates = [HasStdExtV, HasStdExtF]

//===----------------------------------------------------------------------===//
// Miscellaneous RISCVISD SDNodes
//===----------------------------------------------------------------------===//

def riscv_vid_vl : SDNode<"RISCVISD::VID_VL", SDTypeProfile<1, 2,
                          [SDTCisVec<0>, SDTCVecEltisVT<1, i1>,
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
