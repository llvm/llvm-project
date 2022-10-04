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
/// version 1.0.
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

def SDT_RISCVIntBinOp_VL : SDTypeProfile<1, 5, [SDTCisSameAs<0, 1>,
                                                SDTCisSameAs<0, 2>,
                                                SDTCisVec<0>, SDTCisInt<0>,
                                                SDTCisSameAs<0, 3>,
                                                SDTCVecEltisVT<4, i1>,
                                                SDTCisSameNumEltsAs<0, 4>,
                                                SDTCisVT<5, XLenVT>]>;

def SDT_RISCVFPUnOp_VL : SDTypeProfile<1, 3, [SDTCisSameAs<0, 1>,
                                              SDTCisVec<0>, SDTCisFP<0>,
                                              SDTCVecEltisVT<2, i1>,
                                              SDTCisSameNumEltsAs<0, 2>,
                                              SDTCisVT<3, XLenVT>]>;
def SDT_RISCVFPBinOp_VL : SDTypeProfile<1, 5, [SDTCisSameAs<0, 1>,
                                               SDTCisSameAs<0, 2>,
                                               SDTCisVec<0>, SDTCisFP<0>,
                                               SDTCisSameAs<0, 3>,
                                               SDTCVecEltisVT<4, i1>,
                                               SDTCisSameNumEltsAs<0, 4>,
                                               SDTCisVT<5, XLenVT>]>;

def SDT_RISCVCopySign_VL : SDTypeProfile<1, 5, [SDTCisSameAs<0, 1>,
                                                SDTCisSameAs<0, 2>,
                                                SDTCisVec<0>, SDTCisFP<0>,
                                                SDTCisSameAs<0, 3>,
                                                SDTCVecEltisVT<4, i1>,
                                                SDTCisSameNumEltsAs<0, 4>,
                                                SDTCisVT<5, XLenVT>]>;

def riscv_vmv_v_x_vl : SDNode<"RISCVISD::VMV_V_X_VL",
                              SDTypeProfile<1, 3, [SDTCisVec<0>, SDTCisInt<0>,
                                                   SDTCisSameAs<0, 1>,
                                                   SDTCisVT<2, XLenVT>,
                                                   SDTCisVT<3, XLenVT>]>>;
def riscv_vfmv_v_f_vl : SDNode<"RISCVISD::VFMV_V_F_VL",
                               SDTypeProfile<1, 3, [SDTCisVec<0>, SDTCisFP<0>,
                                                    SDTCisSameAs<0, 1>,
                                                    SDTCisEltOfVec<2, 0>,
                                                    SDTCisVT<3, XLenVT>]>>;
def riscv_vmv_s_x_vl : SDNode<"RISCVISD::VMV_S_X_VL",
                              SDTypeProfile<1, 3, [SDTCisSameAs<0, 1>,
                                                   SDTCisInt<0>,
                                                   SDTCisVT<2, XLenVT>,
                                                   SDTCisVT<3, XLenVT>]>>;
def riscv_vfmv_s_f_vl : SDNode<"RISCVISD::VFMV_S_F_VL",
                               SDTypeProfile<1, 3, [SDTCisSameAs<0, 1>,
                                                    SDTCisFP<0>,
                                                    SDTCisEltOfVec<2, 0>,
                                                    SDTCisVT<3, XLenVT>]>>;

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
def riscv_smin_vl  : SDNode<"RISCVISD::SMIN_VL",  SDT_RISCVIntBinOp_VL, [SDNPCommutative]>;
def riscv_smax_vl  : SDNode<"RISCVISD::SMAX_VL",  SDT_RISCVIntBinOp_VL, [SDNPCommutative]>;
def riscv_umin_vl  : SDNode<"RISCVISD::UMIN_VL",  SDT_RISCVIntBinOp_VL, [SDNPCommutative]>;
def riscv_umax_vl  : SDNode<"RISCVISD::UMAX_VL",  SDT_RISCVIntBinOp_VL, [SDNPCommutative]>;

def riscv_saddsat_vl   : SDNode<"RISCVISD::SADDSAT_VL", SDT_RISCVIntBinOp_VL, [SDNPCommutative]>;
def riscv_uaddsat_vl   : SDNode<"RISCVISD::UADDSAT_VL", SDT_RISCVIntBinOp_VL, [SDNPCommutative]>;
def riscv_ssubsat_vl   : SDNode<"RISCVISD::SSUBSAT_VL", SDT_RISCVIntBinOp_VL>;
def riscv_usubsat_vl   : SDNode<"RISCVISD::USUBSAT_VL", SDT_RISCVIntBinOp_VL>;

def riscv_fadd_vl  : SDNode<"RISCVISD::FADD_VL",  SDT_RISCVFPBinOp_VL, [SDNPCommutative]>;
def riscv_fsub_vl  : SDNode<"RISCVISD::FSUB_VL",  SDT_RISCVFPBinOp_VL>;
def riscv_fmul_vl  : SDNode<"RISCVISD::FMUL_VL",  SDT_RISCVFPBinOp_VL, [SDNPCommutative]>;
def riscv_fdiv_vl  : SDNode<"RISCVISD::FDIV_VL",  SDT_RISCVFPBinOp_VL>;
def riscv_fneg_vl  : SDNode<"RISCVISD::FNEG_VL",  SDT_RISCVFPUnOp_VL>;
def riscv_fabs_vl  : SDNode<"RISCVISD::FABS_VL",  SDT_RISCVFPUnOp_VL>;
def riscv_fsqrt_vl : SDNode<"RISCVISD::FSQRT_VL", SDT_RISCVFPUnOp_VL>;
def riscv_fcopysign_vl : SDNode<"RISCVISD::FCOPYSIGN_VL", SDT_RISCVCopySign_VL>;
def riscv_fminnum_vl   : SDNode<"RISCVISD::FMINNUM_VL",  SDT_RISCVFPBinOp_VL, [SDNPCommutative]>;
def riscv_fmaxnum_vl   : SDNode<"RISCVISD::FMAXNUM_VL",  SDT_RISCVFPBinOp_VL, [SDNPCommutative]>;

def SDT_RISCVVecFMA_VL : SDTypeProfile<1, 5, [SDTCisSameAs<0, 1>,
                                              SDTCisSameAs<0, 2>,
                                              SDTCisSameAs<0, 3>,
                                              SDTCisVec<0>, SDTCisFP<0>,
                                              SDTCVecEltisVT<4, i1>,
                                              SDTCisSameNumEltsAs<0, 4>,
                                              SDTCisVT<5, XLenVT>]>;
def riscv_vfmadd_vl : SDNode<"RISCVISD::VFMADD_VL", SDT_RISCVVecFMA_VL, [SDNPCommutative]>;
def riscv_vfnmadd_vl : SDNode<"RISCVISD::VFNMADD_VL", SDT_RISCVVecFMA_VL, [SDNPCommutative]>;
def riscv_vfmsub_vl : SDNode<"RISCVISD::VFMSUB_VL", SDT_RISCVVecFMA_VL, [SDNPCommutative]>;
def riscv_vfnmsub_vl : SDNode<"RISCVISD::VFNMSUB_VL", SDT_RISCVVecFMA_VL, [SDNPCommutative]>;

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

def riscv_vfcvt_rtz_x_f_vl  : SDNode<"RISCVISD::VFCVT_RTZ_X_F_VL",  SDT_RISCVFP2IOp_VL>;
def riscv_vfcvt_rtz_xu_f_vl : SDNode<"RISCVISD::VFCVT_RTZ_XU_F_VL", SDT_RISCVFP2IOp_VL>;
def riscv_sint_to_fp_vl : SDNode<"RISCVISD::SINT_TO_FP_VL", SDT_RISCVI2FPOp_VL>;
def riscv_uint_to_fp_vl : SDNode<"RISCVISD::UINT_TO_FP_VL", SDT_RISCVI2FPOp_VL>;

def SDT_RISCVVecCvtX2FOp_VL : SDTypeProfile<1, 4, [
  SDTCisInt<0>, SDTCisFP<1>, SDTCisSameNumEltsAs<0, 1>,
  SDTCVecEltisVT<2, i1>, SDTCisSameNumEltsAs<1, 2>, SDTCisVT<3, XLenVT>,
  SDTCisVT<4, XLenVT>
]>;

def riscv_vfcvt_x_f_vl : SDNode<"RISCVISD::VFCVT_X_F_VL", SDT_RISCVVecCvtX2FOp_VL>;

def riscv_setcc_vl : SDNode<"RISCVISD::SETCC_VL",
                            SDTypeProfile<1, 6, [SDTCVecEltisVT<0, i1>,
                                                 SDTCisVec<1>,
                                                 SDTCisSameNumEltsAs<0, 1>,
                                                 SDTCisSameAs<1, 2>,
                                                 SDTCisVT<3, OtherVT>,
                                                 SDTCisSameAs<0, 4>,
                                                 SDTCisSameAs<0, 5>,
                                                 SDTCisVT<6, XLenVT>]>>;

def riscv_vrgather_vx_vl : SDNode<"RISCVISD::VRGATHER_VX_VL",
                                  SDTypeProfile<1, 5, [SDTCisVec<0>,
                                                       SDTCisSameAs<0, 1>,
                                                       SDTCisVT<2, XLenVT>,
                                                       SDTCisSameAs<0, 3>,
                                                       SDTCVecEltisVT<4, i1>,
                                                       SDTCisSameNumEltsAs<0, 4>,
                                                       SDTCisVT<5, XLenVT>]>>;
def riscv_vrgather_vv_vl : SDNode<"RISCVISD::VRGATHER_VV_VL",
                                  SDTypeProfile<1, 5, [SDTCisVec<0>,
                                                       SDTCisSameAs<0, 1>,
                                                       SDTCisInt<2>,
                                                       SDTCisSameNumEltsAs<0, 2>,
                                                       SDTCisSameSizeAs<0, 2>,
                                                       SDTCisSameAs<0, 3>,
                                                       SDTCVecEltisVT<4, i1>,
                                                       SDTCisSameNumEltsAs<0, 4>,
                                                       SDTCisVT<5, XLenVT>]>>;
def riscv_vrgatherei16_vv_vl : SDNode<"RISCVISD::VRGATHEREI16_VV_VL",
                                      SDTypeProfile<1, 5, [SDTCisVec<0>,
                                                           SDTCisSameAs<0, 1>,
                                                           SDTCisInt<2>,
                                                           SDTCVecEltisVT<2, i16>,
                                                           SDTCisSameNumEltsAs<0, 2>,
                                                           SDTCisSameAs<0, 3>,
                                                           SDTCVecEltisVT<4, i1>,
                                                           SDTCisSameNumEltsAs<0, 4>,
                                                           SDTCisVT<5, XLenVT>]>>;

def SDT_RISCVSelect_VL  : SDTypeProfile<1, 4, [
  SDTCisVec<0>, SDTCisVec<1>, SDTCisSameNumEltsAs<0, 1>, SDTCVecEltisVT<1, i1>,
  SDTCisSameAs<0, 2>, SDTCisSameAs<2, 3>, SDTCisVT<4, XLenVT>
]>;

def riscv_vselect_vl  : SDNode<"RISCVISD::VSELECT_VL", SDT_RISCVSelect_VL>;
def riscv_vp_merge_vl : SDNode<"RISCVISD::VP_MERGE_VL", SDT_RISCVSelect_VL>;

def SDT_RISCVVMSETCLR_VL : SDTypeProfile<1, 1, [SDTCVecEltisVT<0, i1>,
                                                SDTCisVT<1, XLenVT>]>;
def riscv_vmclr_vl : SDNode<"RISCVISD::VMCLR_VL", SDT_RISCVVMSETCLR_VL>;
def riscv_vmset_vl : SDNode<"RISCVISD::VMSET_VL", SDT_RISCVVMSETCLR_VL>;

def SDT_RISCVMaskBinOp_VL : SDTypeProfile<1, 3, [SDTCisSameAs<0, 1>,
                                                 SDTCisSameAs<0, 2>,
                                                 SDTCVecEltisVT<0, i1>,
                                                 SDTCisVT<3, XLenVT>]>;
def riscv_vmand_vl : SDNode<"RISCVISD::VMAND_VL", SDT_RISCVMaskBinOp_VL, [SDNPCommutative]>;
def riscv_vmor_vl  : SDNode<"RISCVISD::VMOR_VL",  SDT_RISCVMaskBinOp_VL, [SDNPCommutative]>;
def riscv_vmxor_vl : SDNode<"RISCVISD::VMXOR_VL", SDT_RISCVMaskBinOp_VL, [SDNPCommutative]>;

def true_mask : PatLeaf<(riscv_vmset_vl (XLenVT srcvalue))>;

def riscv_vmnot_vl : PatFrag<(ops node:$rs, node:$vl),
                             (riscv_vmxor_vl node:$rs, true_mask, node:$vl)>;

def riscv_vcpop_vl : SDNode<"RISCVISD::VCPOP_VL",
                            SDTypeProfile<1, 3, [SDTCisVT<0, XLenVT>,
                                                 SDTCisVec<1>, SDTCisInt<1>,
                                                 SDTCVecEltisVT<2, i1>,
                                                 SDTCisSameNumEltsAs<1, 2>,
                                                 SDTCisVT<3, XLenVT>]>>;

def SDT_RISCVVEXTEND_VL : SDTypeProfile<1, 3, [SDTCisVec<0>,
                                               SDTCisSameNumEltsAs<0, 1>,
                                               SDTCisSameNumEltsAs<1, 2>,
                                               SDTCVecEltisVT<2, i1>,
                                               SDTCisVT<3, XLenVT>]>;
def riscv_sext_vl : SDNode<"RISCVISD::VSEXT_VL", SDT_RISCVVEXTEND_VL>;
def riscv_zext_vl : SDNode<"RISCVISD::VZEXT_VL", SDT_RISCVVEXTEND_VL>;

def riscv_trunc_vector_vl : SDNode<"RISCVISD::TRUNCATE_VECTOR_VL",
                                   SDTypeProfile<1, 3, [SDTCisVec<0>,
                                                        SDTCisSameNumEltsAs<0, 1>,
                                                        SDTCisSameNumEltsAs<0, 2>,
                                                        SDTCVecEltisVT<2, i1>,
                                                        SDTCisVT<3, XLenVT>]>>;

def SDT_RISCVVWBinOp_VL : SDTypeProfile<1, 5, [SDTCisVec<0>,
                                               SDTCisSameNumEltsAs<0, 1>,
                                               SDTCisSameAs<1, 2>,
                                               SDTCisSameAs<0, 3>,
                                               SDTCisSameNumEltsAs<1, 4>,
                                               SDTCVecEltisVT<4, i1>,
                                               SDTCisVT<5, XLenVT>]>;
def riscv_vwmul_vl  : SDNode<"RISCVISD::VWMUL_VL",  SDT_RISCVVWBinOp_VL, [SDNPCommutative]>;
def riscv_vwmulu_vl : SDNode<"RISCVISD::VWMULU_VL", SDT_RISCVVWBinOp_VL, [SDNPCommutative]>;
def riscv_vwmulsu_vl : SDNode<"RISCVISD::VWMULSU_VL", SDT_RISCVVWBinOp_VL>;
def riscv_vwadd_vl :  SDNode<"RISCVISD::VWADD_VL",  SDT_RISCVVWBinOp_VL, [SDNPCommutative]>;
def riscv_vwaddu_vl : SDNode<"RISCVISD::VWADDU_VL", SDT_RISCVVWBinOp_VL, [SDNPCommutative]>;
def riscv_vwsub_vl :  SDNode<"RISCVISD::VWSUB_VL",  SDT_RISCVVWBinOp_VL, []>;
def riscv_vwsubu_vl : SDNode<"RISCVISD::VWSUBU_VL", SDT_RISCVVWBinOp_VL, []>;

def SDT_RISCVVNBinOp_VL : SDTypeProfile<1, 5, [SDTCisVec<0>,
                                               SDTCisSameNumEltsAs<0, 1>,
                                               SDTCisOpSmallerThanOp<0, 1>,
                                               SDTCisSameAs<0, 2>,
                                               SDTCisSameAs<0, 3>,
                                               SDTCisSameNumEltsAs<0, 4>,
                                               SDTCVecEltisVT<4, i1>,
                                               SDTCisVT<5, XLenVT>]>;
def riscv_vnsrl_vl : SDNode<"RISCVISD::VNSRL_VL", SDT_RISCVVNBinOp_VL>;

def SDT_RISCVVWBinOpW_VL : SDTypeProfile<1, 5, [SDTCisVec<0>,
                                                SDTCisSameAs<0, 1>,
                                                SDTCisSameNumEltsAs<1, 2>,
                                                SDTCisOpSmallerThanOp<2, 1>,
                                                SDTCisSameAs<0, 3>,
                                                SDTCisSameNumEltsAs<1, 4>,
                                                SDTCVecEltisVT<4, i1>,
                                                SDTCisVT<5, XLenVT>]>;
def riscv_vwadd_w_vl :  SDNode<"RISCVISD::VWADD_W_VL",  SDT_RISCVVWBinOpW_VL>;
def riscv_vwaddu_w_vl : SDNode<"RISCVISD::VWADDU_W_VL", SDT_RISCVVWBinOpW_VL>;
def riscv_vwsub_w_vl :  SDNode<"RISCVISD::VWSUB_W_VL",  SDT_RISCVVWBinOpW_VL>;
def riscv_vwsubu_w_vl : SDNode<"RISCVISD::VWSUBU_W_VL", SDT_RISCVVWBinOpW_VL>;

def SDTRVVVecReduce : SDTypeProfile<1, 5, [
  SDTCisVec<0>, SDTCisVec<1>, SDTCisVec<2>, SDTCisSameAs<0, 3>,
  SDTCVecEltisVT<4, i1>, SDTCisSameNumEltsAs<2, 4>, SDTCisVT<5, XLenVT>
]>;

def riscv_mul_vl_oneuse : PatFrag<(ops node:$A, node:$B, node:$C, node:$D,
                                       node:$E),
                                  (riscv_mul_vl node:$A, node:$B, node:$C,
                                                node:$D, node:$E), [{
  return N->hasOneUse();
}]>;

def riscv_vwmul_vl_oneuse : PatFrag<(ops node:$A, node:$B, node:$C, node:$D,
                                         node:$E),
                                    (riscv_vwmul_vl node:$A, node:$B, node:$C,
                                                    node:$D, node:$E), [{
  return N->hasOneUse();
}]>;

def riscv_vwmulu_vl_oneuse : PatFrag<(ops node:$A, node:$B, node:$C, node:$D,
                                          node:$E),
                                     (riscv_vwmulu_vl node:$A, node:$B, node:$C,
                                                      node:$D, node:$E), [{
  return N->hasOneUse();
}]>;

def riscv_vwmulsu_vl_oneuse : PatFrag<(ops node:$A, node:$B, node:$C, node:$D,
                                           node:$E),
                                      (riscv_vwmulsu_vl node:$A, node:$B, node:$C,
                                                        node:$D, node:$E), [{
  return N->hasOneUse();
}]>;

def riscv_sext_vl_oneuse : PatFrag<(ops node:$A, node:$B, node:$C),
                           (riscv_sext_vl node:$A, node:$B, node:$C), [{
  return N->hasOneUse();
}]>;

def riscv_zext_vl_oneuse : PatFrag<(ops node:$A, node:$B, node:$C),
                           (riscv_zext_vl node:$A, node:$B, node:$C), [{
  return N->hasOneUse();
}]>;

def riscv_fpextend_vl_oneuse : PatFrag<(ops node:$A, node:$B, node:$C),
                           (riscv_fpextend_vl node:$A, node:$B, node:$C), [{
  return N->hasOneUse();
}]>;

foreach kind = ["ADD", "UMAX", "SMAX", "UMIN", "SMIN", "AND", "OR", "XOR",
                "FADD", "SEQ_FADD", "FMIN", "FMAX"] in
  def rvv_vecreduce_#kind#_vl : SDNode<"RISCVISD::VECREDUCE_"#kind#"_VL", SDTRVVVecReduce>;

// Give explicit Complexity to prefer simm5/uimm5.
def SplatPat       : ComplexPattern<vAny, 1, "selectVSplat",      [], [], 1>;
def SplatPat_simm5 : ComplexPattern<vAny, 1, "selectVSplatSimm5", [], [], 2>;
def SplatPat_uimm5 : ComplexPattern<vAny, 1, "selectVSplatUimm5", [], [], 2>;
def SplatPat_simm5_plus1
    : ComplexPattern<vAny, 1, "selectVSplatSimm5Plus1", [], [], 2>;
def SplatPat_simm5_plus1_nonzero
    : ComplexPattern<vAny, 1, "selectVSplatSimm5Plus1NonZero", [], [], 2>;

// Ignore the vl operand.
def SplatFPOp : PatFrag<(ops node:$op),
                        (riscv_vfmv_v_f_vl undef, node:$op, srcvalue)>;

def sew8simm5  : ComplexPattern<XLenVT, 1, "selectRVVSimm5<8>",  []>;
def sew16simm5 : ComplexPattern<XLenVT, 1, "selectRVVSimm5<16>", []>;
def sew32simm5 : ComplexPattern<XLenVT, 1, "selectRVVSimm5<32>", []>;
def sew64simm5 : ComplexPattern<XLenVT, 1, "selectRVVSimm5<64>", []>;

multiclass VPatBinaryVL_V<SDNode vop,
                          string instruction_name,
                          string suffix,
                          ValueType result_type,
                          ValueType op1_type,
                          ValueType op2_type,
                          ValueType mask_type,
                          int sew,
                          LMULInfo vlmul,
                          VReg result_reg_class,
                          VReg op1_reg_class,
                          VReg op2_reg_class> {
  def : Pat<(result_type (vop
                         (op1_type op1_reg_class:$rs1),
                         (op2_type op2_reg_class:$rs2),
                         (result_type result_reg_class:$merge),
                         (mask_type V0),
                         VLOpFrag)),
        (!cast<Instruction>(instruction_name#"_"#suffix#"_"# vlmul.MX#"_MASK")
                     result_reg_class:$merge,
                     op1_reg_class:$rs1,
                     op2_reg_class:$rs2,
                     (mask_type V0), GPR:$vl, sew, TAIL_AGNOSTIC)>;
}

multiclass VPatTiedBinaryNoMaskVL_V<SDNode vop,
                                    string instruction_name,
                                    string suffix,
                                    ValueType result_type,
                                    ValueType op2_type,
                                    int sew,
                                    LMULInfo vlmul,
                                    VReg result_reg_class,
                                    VReg op2_reg_class> {
  def : Pat<(result_type (vop
                         (result_type result_reg_class:$rs1),
                         (op2_type op2_reg_class:$rs2),
                         srcvalue,
                         true_mask,
                         VLOpFrag)),
        (!cast<Instruction>(instruction_name#"_"#suffix#"_"# vlmul.MX#"_TIED")
                     result_reg_class:$rs1,
                     op2_reg_class:$rs2,
                     GPR:$vl, sew, TAIL_AGNOSTIC)>;
}

multiclass VPatBinaryVL_XI<SDNode vop,
                           string instruction_name,
                           string suffix,
                           ValueType result_type,
                           ValueType vop1_type,
                           ValueType vop2_type,
                           ValueType mask_type,
                           int sew,
                           LMULInfo vlmul,
                           VReg result_reg_class,
                           VReg vop_reg_class,
                           ComplexPattern SplatPatKind,
                           DAGOperand xop_kind> {
  def : Pat<(result_type (vop
                     (vop1_type vop_reg_class:$rs1),
                     (vop2_type (SplatPatKind (XLenVT xop_kind:$rs2))),
                     (result_type result_reg_class:$merge),
                     (mask_type V0),
                     VLOpFrag)),
        (!cast<Instruction>(instruction_name#_#suffix#_# vlmul.MX#"_MASK")
                     result_reg_class:$merge,
                     vop_reg_class:$rs1,
                     xop_kind:$rs2,
                     (mask_type V0), GPR:$vl, sew, TAIL_AGNOSTIC)>;
}

multiclass VPatBinaryVL_VV_VX<SDNode vop, string instruction_name> {
  foreach vti = AllIntegerVectors in {
    defm : VPatBinaryVL_V<vop, instruction_name, "VV",
                           vti.Vector, vti.Vector, vti.Vector, vti.Mask,
                           vti.Log2SEW, vti.LMul, vti.RegClass, vti.RegClass,
                           vti.RegClass>;
    defm : VPatBinaryVL_XI<vop, instruction_name, "VX",
                           vti.Vector, vti.Vector, vti.Vector, vti.Mask,
                           vti.Log2SEW, vti.LMul, vti.RegClass, vti.RegClass,
                           SplatPat, GPR>;
  }
}

multiclass VPatBinaryVL_VV_VX_VI<SDNode vop, string instruction_name,
                                 Operand ImmType = simm5>
    : VPatBinaryVL_VV_VX<vop, instruction_name> {
  foreach vti = AllIntegerVectors in {
    defm : VPatBinaryVL_XI<vop, instruction_name, "VI",
                           vti.Vector, vti.Vector, vti.Vector, vti.Mask,
                           vti.Log2SEW, vti.LMul, vti.RegClass, vti.RegClass,
                           !cast<ComplexPattern>(SplatPat#_#ImmType),
                           ImmType>;
  }
}

multiclass VPatBinaryWVL_VV_VX<SDNode vop, string instruction_name> {
  foreach VtiToWti = AllWidenableIntVectors in {
    defvar vti = VtiToWti.Vti;
    defvar wti = VtiToWti.Wti;
    defm : VPatBinaryVL_V<vop, instruction_name, "VV",
                          wti.Vector, vti.Vector, vti.Vector, vti.Mask,
                          vti.Log2SEW, vti.LMul, wti.RegClass, vti.RegClass,
                          vti.RegClass>;
    defm : VPatBinaryVL_XI<vop, instruction_name, "VX",
                           wti.Vector, vti.Vector, vti.Vector, vti.Mask,
                           vti.Log2SEW, vti.LMul, wti.RegClass, vti.RegClass,
                           SplatPat, GPR>;
  }
}
multiclass VPatBinaryWVL_VV_VX_WV_WX<SDNode vop, SDNode vop_w,
                                     string instruction_name>
    : VPatBinaryWVL_VV_VX<vop, instruction_name> {
  foreach VtiToWti = AllWidenableIntVectors in {
    defvar vti = VtiToWti.Vti;
    defvar wti = VtiToWti.Wti;
    defm : VPatTiedBinaryNoMaskVL_V<vop_w, instruction_name, "WV",
                                    wti.Vector, vti.Vector, vti.Log2SEW,
                                    vti.LMul, wti.RegClass, vti.RegClass>;
    defm : VPatBinaryVL_V<vop_w, instruction_name, "WV",
                          wti.Vector, wti.Vector, vti.Vector, vti.Mask,
                          vti.Log2SEW, vti.LMul, wti.RegClass, wti.RegClass,
                          vti.RegClass>;
    defm : VPatBinaryVL_XI<vop_w, instruction_name, "WX",
                           wti.Vector, wti.Vector, vti.Vector, vti.Mask,
                           vti.Log2SEW, vti.LMul, wti.RegClass, wti.RegClass,
                           SplatPat, GPR>;
  }
}

multiclass VPatBinaryNVL_WV_WX_WI<SDNode vop, string instruction_name> {
  foreach VtiToWti = AllWidenableIntVectors in {
    defvar vti = VtiToWti.Vti;
    defvar wti = VtiToWti.Wti;
    defm : VPatBinaryVL_V<vop, instruction_name, "WV",
                          vti.Vector, wti.Vector, vti.Vector, vti.Mask,
                          vti.Log2SEW, vti.LMul, vti.RegClass, wti.RegClass,
                          vti.RegClass>;
    defm : VPatBinaryVL_XI<vop, instruction_name, "WX",
                           vti.Vector, wti.Vector, vti.Vector, vti.Mask,
                           vti.Log2SEW, vti.LMul, vti.RegClass, wti.RegClass,
                           SplatPat, GPR>;
    defm : VPatBinaryVL_XI<vop, instruction_name, "WI",
                           vti.Vector, wti.Vector, vti.Vector, vti.Mask,
                           vti.Log2SEW, vti.LMul, vti.RegClass, wti.RegClass,
                           !cast<ComplexPattern>(SplatPat#_#uimm5),
                           uimm5>;
  }
}

multiclass VPatBinaryVL_VF<SDNode vop,
                           string instruction_name,
                           ValueType result_type,
                           ValueType vop_type,
                           ValueType mask_type,
                           int sew,
                           LMULInfo vlmul,
                           VReg result_reg_class,
                           VReg vop_reg_class,
                           RegisterClass scalar_reg_class> {
  def : Pat<(result_type (vop (vop_type vop_reg_class:$rs1),
                         (vop_type (SplatFPOp scalar_reg_class:$rs2)),
                         (result_type result_reg_class:$merge),
                         (mask_type V0),
                         VLOpFrag)),
        (!cast<Instruction>(instruction_name#"_"#vlmul.MX#"_MASK")
                     result_reg_class:$merge,
                     vop_reg_class:$rs1,
                     scalar_reg_class:$rs2,
                     (mask_type V0), GPR:$vl, sew, TAIL_AGNOSTIC)>;
}

multiclass VPatBinaryFPVL_VV_VF<SDNode vop, string instruction_name> {
  foreach vti = AllFloatVectors in {
    defm : VPatBinaryVL_V<vop, instruction_name, "VV",
                          vti.Vector, vti.Vector, vti.Vector, vti.Mask,
                          vti.Log2SEW, vti.LMul, vti.RegClass, vti.RegClass,
                          vti.RegClass>;
    defm : VPatBinaryVL_VF<vop, instruction_name#"_V"#vti.ScalarSuffix,
                           vti.Vector, vti.Vector, vti.Mask, vti.Log2SEW,
                           vti.LMul, vti.RegClass, vti.RegClass,
                           vti.ScalarRegClass>;
  }
}

multiclass VPatBinaryFPVL_R_VF<SDNode vop, string instruction_name> {
  foreach fvti = AllFloatVectors in {
    def : Pat<(fvti.Vector (vop (SplatFPOp fvti.ScalarRegClass:$rs2),
                                fvti.RegClass:$rs1,
                                (fvti.Vector fvti.RegClass:$merge),
                                (fvti.Mask V0),
                                VLOpFrag)),
              (!cast<Instruction>(instruction_name#"_V"#fvti.ScalarSuffix#"_"#fvti.LMul.MX#"_MASK")
                           fvti.RegClass:$merge,
                           fvti.RegClass:$rs1, fvti.ScalarRegClass:$rs2,
                           (fvti.Mask V0), GPR:$vl, fvti.Log2SEW, TAIL_AGNOSTIC)>;
  }
}

multiclass VPatIntegerSetCCVL_VV<VTypeInfo vti, string instruction_name,
                                 CondCode cc> {
  def : Pat<(vti.Mask (riscv_setcc_vl (vti.Vector vti.RegClass:$rs1),
                                      vti.RegClass:$rs2, cc,
                                      VR:$merge,
                                      (vti.Mask V0),
                                      VLOpFrag)),
            (!cast<Instruction>(instruction_name#"_VV_"#vti.LMul.MX#"_MASK")
                         VR:$merge,
                         vti.RegClass:$rs1,
                         vti.RegClass:$rs2,
                         (vti.Mask V0), GPR:$vl, vti.Log2SEW)>;
}

// Inherits from VPatIntegerSetCCVL_VV and adds a pattern with operands swapped.
multiclass VPatIntegerSetCCVL_VV_Swappable<VTypeInfo vti, string instruction_name,
                                           CondCode cc, CondCode invcc>
    : VPatIntegerSetCCVL_VV<vti, instruction_name, cc> {
  def : Pat<(vti.Mask (riscv_setcc_vl (vti.Vector vti.RegClass:$rs2),
                                      vti.RegClass:$rs1, invcc,
                                      VR:$merge,
                                      (vti.Mask V0),
                                      VLOpFrag)),
            (!cast<Instruction>(instruction_name#"_VV_"#vti.LMul.MX#"_MASK")
                         VR:$merge, vti.RegClass:$rs1,
                         vti.RegClass:$rs2, (vti.Mask V0), GPR:$vl, vti.Log2SEW)>;
}

multiclass VPatIntegerSetCCVL_VX_Swappable<VTypeInfo vti, string instruction_name,
                                           CondCode cc, CondCode invcc> {
  defvar instruction_masked = !cast<Instruction>(instruction_name#"_VX_"#vti.LMul.MX#"_MASK");
  def : Pat<(vti.Mask (riscv_setcc_vl (vti.Vector vti.RegClass:$rs1),
                                      (SplatPat (XLenVT GPR:$rs2)), cc,
                                      VR:$merge,
                                      (vti.Mask V0),
                                      VLOpFrag)),
            (instruction_masked VR:$merge, vti.RegClass:$rs1,
                                GPR:$rs2, (vti.Mask V0), GPR:$vl, vti.Log2SEW)>;
  def : Pat<(vti.Mask (riscv_setcc_vl (SplatPat (XLenVT GPR:$rs2)),
                                      (vti.Vector vti.RegClass:$rs1), invcc,
                                      VR:$merge,
                                      (vti.Mask V0),
                                      VLOpFrag)),
            (instruction_masked VR:$merge, vti.RegClass:$rs1,
                                GPR:$rs2, (vti.Mask V0), GPR:$vl, vti.Log2SEW)>;
}

multiclass VPatIntegerSetCCVL_VI_Swappable<VTypeInfo vti, string instruction_name,
                                           CondCode cc, CondCode invcc> {
  defvar instruction_masked = !cast<Instruction>(instruction_name#"_VI_"#vti.LMul.MX#"_MASK");
  def : Pat<(vti.Mask (riscv_setcc_vl (vti.Vector vti.RegClass:$rs1),
                                      (SplatPat_simm5 simm5:$rs2), cc,
                                      VR:$merge,
                                      (vti.Mask V0),
                                      VLOpFrag)),
            (instruction_masked VR:$merge, vti.RegClass:$rs1,
                                XLenVT:$rs2, (vti.Mask V0), GPR:$vl,
                                vti.Log2SEW)>;

  // FIXME: Can do some canonicalization to remove these patterns.
  def : Pat<(vti.Mask (riscv_setcc_vl (SplatPat_simm5 simm5:$rs2),
                                      (vti.Vector vti.RegClass:$rs1), invcc,
                                      VR:$merge,
                                      (vti.Mask V0),
                                      VLOpFrag)),
            (instruction_masked VR:$merge, vti.RegClass:$rs1,
                                simm5:$rs2, (vti.Mask V0), GPR:$vl,
                                vti.Log2SEW)>;
}

multiclass VPatIntegerSetCCVL_VIPlus1_Swappable<VTypeInfo vti,
                                                string instruction_name,
                                                CondCode cc, CondCode invcc,
                                                ComplexPattern splatpat_kind> {
  defvar instruction_masked = !cast<Instruction>(instruction_name#"_VI_"#vti.LMul.MX#"_MASK");
  def : Pat<(vti.Mask (riscv_setcc_vl (vti.Vector vti.RegClass:$rs1),
                                      (splatpat_kind simm5:$rs2), cc,
                                      VR:$merge,
                                      (vti.Mask V0),
                                      VLOpFrag)),
            (instruction_masked VR:$merge, vti.RegClass:$rs1,
                                (DecImm simm5:$rs2), (vti.Mask V0), GPR:$vl,
                                vti.Log2SEW)>;

  // FIXME: Can do some canonicalization to remove these patterns.
  def : Pat<(vti.Mask (riscv_setcc_vl (splatpat_kind simm5:$rs2),
                                      (vti.Vector vti.RegClass:$rs1), invcc,
                                      VR:$merge,
                                      (vti.Mask V0),
                                      VLOpFrag)),
            (instruction_masked VR:$merge, vti.RegClass:$rs1,
                                (DecImm simm5:$rs2), (vti.Mask V0), GPR:$vl,
                                vti.Log2SEW)>;
}

multiclass VPatFPSetCCVL_VV_VF_FV<CondCode cc,
                                  string inst_name,
                                  string swapped_op_inst_name> {
  foreach fvti = AllFloatVectors in {
    def : Pat<(fvti.Mask (riscv_setcc_vl (fvti.Vector fvti.RegClass:$rs1),
                                         fvti.RegClass:$rs2,
                                         cc,
                                         VR:$merge,
                                         (fvti.Mask V0),
                                         VLOpFrag)),
              (!cast<Instruction>(inst_name#"_VV_"#fvti.LMul.MX#"_MASK")
                  VR:$merge, fvti.RegClass:$rs1,
                  fvti.RegClass:$rs2, (fvti.Mask V0),
                  GPR:$vl, fvti.Log2SEW)>;
    def : Pat<(fvti.Mask (riscv_setcc_vl (fvti.Vector fvti.RegClass:$rs1),
                                         (SplatFPOp fvti.ScalarRegClass:$rs2),
                                         cc,
                                         VR:$merge,
                                         (fvti.Mask V0),
                                         VLOpFrag)),
              (!cast<Instruction>(inst_name#"_V"#fvti.ScalarSuffix#"_"#fvti.LMul.MX#"_MASK")
                  VR:$merge, fvti.RegClass:$rs1,
                  fvti.ScalarRegClass:$rs2, (fvti.Mask V0),
                  GPR:$vl, fvti.Log2SEW)>;
    def : Pat<(fvti.Mask (riscv_setcc_vl (SplatFPOp fvti.ScalarRegClass:$rs2),
                                         (fvti.Vector fvti.RegClass:$rs1),
                                         cc,
                                         VR:$merge,
                                         (fvti.Mask V0),
                                         VLOpFrag)),
              (!cast<Instruction>(swapped_op_inst_name#"_V"#fvti.ScalarSuffix#"_"#fvti.LMul.MX#"_MASK")
                  VR:$merge, fvti.RegClass:$rs1,
                  fvti.ScalarRegClass:$rs2, (fvti.Mask V0),
                  GPR:$vl, fvti.Log2SEW)>;
  }
}

multiclass VPatExtendVL_V<SDNode vop, string inst_name, string suffix,
                          list <VTypeInfoToFraction> fraction_list> {
  foreach vtiTofti = fraction_list in {
    defvar vti = vtiTofti.Vti;
    defvar fti = vtiTofti.Fti;
    def : Pat<(vti.Vector (vop (fti.Vector fti.RegClass:$rs2),
                               (fti.Mask V0), VLOpFrag)),
              (!cast<Instruction>(inst_name#"_"#suffix#"_"#vti.LMul.MX#"_MASK")
                  (vti.Vector (IMPLICIT_DEF)),
                  fti.RegClass:$rs2,
                  (fti.Mask V0), GPR:$vl, vti.Log2SEW, TAIL_AGNOSTIC)>;
  }
}

multiclass VPatConvertFP2IVL_V<SDNode vop, string instruction_name> {
  foreach fvti = AllFloatVectors in {
    defvar ivti = GetIntVTypeInfo<fvti>.Vti;
    def : Pat<(ivti.Vector (vop (fvti.Vector fvti.RegClass:$rs1),
                                (fvti.Mask V0),
                                VLOpFrag)),
              (!cast<Instruction>(instruction_name#"_"#ivti.LMul.MX#"_MASK")
                  (ivti.Vector (IMPLICIT_DEF)), fvti.RegClass:$rs1,
                  (fvti.Mask V0), GPR:$vl, ivti.Log2SEW, TAIL_AGNOSTIC)>;
  }
}

multiclass VPatConvertFP2I_RM_VL_V<SDNode vop, string instruction_name> {
  foreach fvti = AllFloatVectors in {
    defvar ivti = GetIntVTypeInfo<fvti>.Vti;
    def : Pat<(ivti.Vector (vop (fvti.Vector fvti.RegClass:$rs1),
                                (fvti.Mask V0), (XLenVT timm:$frm),
                                VLOpFrag)),
              (!cast<Instruction>(instruction_name#"_"#ivti.LMul.MX#"_MASK")
                  (ivti.Vector (IMPLICIT_DEF)), fvti.RegClass:$rs1,
                  (fvti.Mask V0), timm:$frm, GPR:$vl, ivti.Log2SEW,
                  TAIL_AGNOSTIC)>;
  }
}

multiclass VPatConvertI2FPVL_V<SDNode vop, string instruction_name> {
  foreach fvti = AllFloatVectors in {
    defvar ivti = GetIntVTypeInfo<fvti>.Vti;
    def : Pat<(fvti.Vector (vop (ivti.Vector ivti.RegClass:$rs1),
                                (ivti.Mask V0),
                                VLOpFrag)),
              (!cast<Instruction>(instruction_name#"_"#fvti.LMul.MX#"_MASK")
                  (fvti.Vector (IMPLICIT_DEF)), ivti.RegClass:$rs1,
                  (ivti.Mask V0), GPR:$vl, fvti.Log2SEW, TAIL_AGNOSTIC)>;
  }
}

multiclass VPatWConvertFP2IVL_V<SDNode vop, string instruction_name> {
  foreach fvtiToFWti = AllWidenableFloatVectors in {
    defvar fvti = fvtiToFWti.Vti;
    defvar iwti = GetIntVTypeInfo<fvtiToFWti.Wti>.Vti;
    def : Pat<(iwti.Vector (vop (fvti.Vector fvti.RegClass:$rs1),
                                (fvti.Mask V0),
                                VLOpFrag)),
              (!cast<Instruction>(instruction_name#"_"#fvti.LMul.MX#"_MASK")
                  (iwti.Vector (IMPLICIT_DEF)), fvti.RegClass:$rs1,
                  (fvti.Mask V0), GPR:$vl, fvti.Log2SEW, TAIL_AGNOSTIC)>;
  }
}

multiclass VPatWConvertI2FPVL_V<SDNode vop, string instruction_name> {
  foreach vtiToWti = AllWidenableIntToFloatVectors in {
    defvar ivti = vtiToWti.Vti;
    defvar fwti = vtiToWti.Wti;
    def : Pat<(fwti.Vector (vop (ivti.Vector ivti.RegClass:$rs1),
                                (ivti.Mask V0),
                                VLOpFrag)),
              (!cast<Instruction>(instruction_name#"_"#ivti.LMul.MX#"_MASK")
                  (fwti.Vector (IMPLICIT_DEF)), ivti.RegClass:$rs1,
                  (ivti.Mask V0), GPR:$vl, ivti.Log2SEW, TAIL_AGNOSTIC)>;
  }
}

multiclass VPatNConvertFP2IVL_V<SDNode vop, string instruction_name> {
  foreach vtiToWti = AllWidenableIntToFloatVectors in {
    defvar vti = vtiToWti.Vti;
    defvar fwti = vtiToWti.Wti;
    def : Pat<(vti.Vector (vop (fwti.Vector fwti.RegClass:$rs1),
                               (fwti.Mask V0),
                               VLOpFrag)),
              (!cast<Instruction>(instruction_name#"_"#vti.LMul.MX#"_MASK")
                  (vti.Vector (IMPLICIT_DEF)), fwti.RegClass:$rs1,
                  (fwti.Mask V0), GPR:$vl, vti.Log2SEW, TAIL_AGNOSTIC)>;
  }
}

multiclass VPatNConvertI2FPVL_V<SDNode vop, string instruction_name> {
  foreach fvtiToFWti = AllWidenableFloatVectors in {
    defvar fvti = fvtiToFWti.Vti;
    defvar iwti = GetIntVTypeInfo<fvtiToFWti.Wti>.Vti;
    def : Pat<(fvti.Vector (vop (iwti.Vector iwti.RegClass:$rs1),
                                (iwti.Mask V0),
                                VLOpFrag)),
              (!cast<Instruction>(instruction_name#"_"#fvti.LMul.MX#"_MASK")
                  (fvti.Vector (IMPLICIT_DEF)), iwti.RegClass:$rs1,
                  (iwti.Mask V0), GPR:$vl, fvti.Log2SEW, TAIL_AGNOSTIC)>;
  }
}

multiclass VPatReductionVL<SDNode vop, string instruction_name, bit is_float> {
  foreach vti = !if(is_float, AllFloatVectors, AllIntegerVectors) in {
    defvar vti_m1 = !cast<VTypeInfo>(!if(is_float, "VF", "VI") # vti.SEW # "M1");
    def: Pat<(vti_m1.Vector (vop (vti_m1.Vector VR:$merge), (vti.Vector vti.RegClass:$rs1), VR:$rs2,
                                 (vti.Mask true_mask),
                                 VLOpFrag)),
        (!cast<Instruction>(instruction_name#"_VS_"#vti.LMul.MX)
            (vti_m1.Vector VR:$merge),
            (vti.Vector vti.RegClass:$rs1),
            (vti_m1.Vector VR:$rs2),
            GPR:$vl, vti.Log2SEW)>;

    def: Pat<(vti_m1.Vector (vop (vti_m1.Vector VR:$merge), (vti.Vector vti.RegClass:$rs1), VR:$rs2,
                                 (vti.Mask V0), VLOpFrag)),
        (!cast<Instruction>(instruction_name#"_VS_"#vti.LMul.MX#"_MASK")
            (vti_m1.Vector VR:$merge),
            (vti.Vector vti.RegClass:$rs1),
            (vti_m1.Vector VR:$rs2),
            (vti.Mask V0), GPR:$vl, vti.Log2SEW)>;
  }
}

multiclass VPatBinaryExtVL_WV_WX<SDNode op, PatFrags extop, string instruction_name> {
  foreach vtiToWti = AllWidenableIntVectors in {
    defvar vti = vtiToWti.Vti;
    defvar wti = vtiToWti.Wti;
    def : Pat<
      (vti.Vector
        (riscv_trunc_vector_vl
          (op (wti.Vector wti.RegClass:$rs2),
              (wti.Vector (extop (vti.Vector vti.RegClass:$rs1)))),
          (vti.Mask true_mask),
          VLOpFrag)),
      (!cast<Instruction>(instruction_name#"_WV_"#vti.LMul.MX)
        wti.RegClass:$rs2, vti.RegClass:$rs1, GPR:$vl, vti.Log2SEW)>;
    def : Pat<
      (vti.Vector
        (riscv_trunc_vector_vl
          (op (wti.Vector wti.RegClass:$rs2),
              (wti.Vector (extop (vti.Vector (SplatPat GPR:$rs1))))),
          (vti.Mask true_mask),
          VLOpFrag)),
      (!cast<Instruction>(instruction_name#"_WX_"#vti.LMul.MX)
        wti.RegClass:$rs2, GPR:$rs1, GPR:$vl, vti.Log2SEW)>;
  }
}

multiclass VPatBinaryVL_WV_WX_WI<SDNode op, string instruction_name> {
  defm : VPatBinaryExtVL_WV_WX<op, sext_oneuse, instruction_name>;
  defm : VPatBinaryExtVL_WV_WX<op, zext_oneuse, instruction_name>;
  foreach vtiToWti = AllWidenableIntVectors in {
    defvar vti = vtiToWti.Vti;
    defvar wti = vtiToWti.Wti;
    def : Pat<
      (vti.Vector
        (riscv_trunc_vector_vl
          (op (wti.Vector wti.RegClass:$rs2),
              (wti.Vector (SplatPat_uimm5 uimm5:$rs1))), (vti.Mask true_mask),
          VLOpFrag)),
      (!cast<Instruction>(instruction_name#"_WI_"#vti.LMul.MX)
        wti.RegClass:$rs2, uimm5:$rs1, GPR:$vl, vti.Log2SEW)>;
  }
}

multiclass VPatWidenReductionVL<SDNode vop, PatFrags extop, string instruction_name, bit is_float> {
  foreach vtiToWti = !if(is_float, AllWidenableFloatVectors, AllWidenableIntVectors) in {
    defvar vti = vtiToWti.Vti;
    defvar wti = vtiToWti.Wti;
    defvar wti_m1 = !cast<VTypeInfo>(!if(is_float, "VF", "VI") # wti.SEW # "M1");
    def: Pat<(wti_m1.Vector (vop (wti_m1.Vector VR:$merge),
                                 (wti.Vector (extop (vti.Vector vti.RegClass:$rs1))),
                                 VR:$rs2, (vti.Mask true_mask), VLOpFrag)),
             (!cast<Instruction>(instruction_name#"_VS_"#vti.LMul.MX)
                (wti_m1.Vector VR:$merge), (vti.Vector vti.RegClass:$rs1),
                (wti_m1.Vector VR:$rs2), GPR:$vl, vti.Log2SEW)>;
    def: Pat<(wti_m1.Vector (vop (wti_m1.Vector VR:$merge),
                                 (wti.Vector (extop (vti.Vector vti.RegClass:$rs1))),
                                 VR:$rs2, (vti.Mask V0), VLOpFrag)),
             (!cast<Instruction>(instruction_name#"_VS_"#vti.LMul.MX#"_MASK")
                (wti_m1.Vector VR:$merge), (vti.Vector vti.RegClass:$rs1),
                (wti_m1.Vector VR:$rs2), (vti.Mask V0), GPR:$vl, vti.Log2SEW)>;
  }
}

multiclass VPatWidenReductionVL_Ext_VL<SDNode vop, PatFrags extop, string instruction_name, bit is_float> {
  foreach vtiToWti = !if(is_float, AllWidenableFloatVectors, AllWidenableIntVectors) in {
    defvar vti = vtiToWti.Vti;
    defvar wti = vtiToWti.Wti;
    defvar wti_m1 = !cast<VTypeInfo>(!if(is_float, "VF", "VI") # wti.SEW # "M1");
    def: Pat<(wti_m1.Vector (vop (wti_m1.Vector VR:$merge),
                                 (wti.Vector (extop (vti.Vector vti.RegClass:$rs1), (vti.Mask true_mask), VLOpFrag)),
                                 VR:$rs2, (vti.Mask true_mask), VLOpFrag)),
             (!cast<Instruction>(instruction_name#"_VS_"#vti.LMul.MX)
                (wti_m1.Vector VR:$merge), (vti.Vector vti.RegClass:$rs1),
                (wti_m1.Vector VR:$rs2), GPR:$vl, vti.Log2SEW)>;
    def: Pat<(wti_m1.Vector (vop (wti_m1.Vector VR:$merge),
                                 (wti.Vector (extop (vti.Vector vti.RegClass:$rs1), (vti.Mask true_mask), VLOpFrag)),
                                 VR:$rs2, (vti.Mask V0), VLOpFrag)),
             (!cast<Instruction>(instruction_name#"_VS_"#vti.LMul.MX#"_MASK")
                (wti_m1.Vector VR:$merge), (vti.Vector vti.RegClass:$rs1),
                (wti_m1.Vector VR:$rs2), (vti.Mask V0), GPR:$vl, vti.Log2SEW)>;
  }
}

multiclass VPatWidenBinaryFPVL_VV_VF<SDNode op, PatFrags extop, string instruction_name> {
  foreach fvtiToFWti = AllWidenableFloatVectors in {
    defvar fvti = fvtiToFWti.Vti;
    defvar fwti = fvtiToFWti.Wti;
    def : Pat<(fwti.Vector (op (fwti.Vector (extop (fvti.Vector fvti.RegClass:$rs2),
                                                   (fvti.Mask true_mask), VLOpFrag)),
                               (fwti.Vector (extop (fvti.Vector fvti.RegClass:$rs1),
                                                   (fvti.Mask true_mask), VLOpFrag)),
                               srcvalue, (fwti.Mask true_mask), VLOpFrag)),
              (!cast<Instruction>(instruction_name#"_VV_"#fvti.LMul.MX)
                 fvti.RegClass:$rs2, fvti.RegClass:$rs1,
                 GPR:$vl, fvti.Log2SEW)>;
    def : Pat<(fwti.Vector (op (fwti.Vector (extop (fvti.Vector fvti.RegClass:$rs2),
                                                   (fvti.Mask true_mask), VLOpFrag)),
                               (fwti.Vector (extop (fvti.Vector (SplatFPOp fvti.ScalarRegClass:$rs1)),
                                                   (fvti.Mask true_mask), VLOpFrag)),
                               srcvalue, (fwti.Mask true_mask), VLOpFrag)),
              (!cast<Instruction>(instruction_name#"_V"#fvti.ScalarSuffix#"_"#fvti.LMul.MX)
                 fvti.RegClass:$rs2, fvti.ScalarRegClass:$rs1,
                 GPR:$vl, fvti.Log2SEW)>;
  }
}

multiclass VPatWidenBinaryFPVL_WV_WF<SDNode op, PatFrags extop, string instruction_name> {
  foreach fvtiToFWti = AllWidenableFloatVectors in {
    defvar fvti = fvtiToFWti.Vti;
    defvar fwti = fvtiToFWti.Wti;
    def : Pat<(fwti.Vector (op (fwti.Vector fwti.RegClass:$rs2),
                               (fwti.Vector (extop (fvti.Vector fvti.RegClass:$rs1),
                                                   (fvti.Mask true_mask), VLOpFrag)),
                               srcvalue, (fwti.Mask true_mask), VLOpFrag)),
              (!cast<Instruction>(instruction_name#"_WV_"#fvti.LMul.MX#"_TIED")
                 fwti.RegClass:$rs2, fvti.RegClass:$rs1,
                 GPR:$vl, fvti.Log2SEW, TAIL_AGNOSTIC)>;
    def : Pat<(fwti.Vector (op (fwti.Vector fwti.RegClass:$rs2),
                               (fwti.Vector (extop (fvti.Vector (SplatFPOp fvti.ScalarRegClass:$rs1)),
                                                   (fvti.Mask true_mask), VLOpFrag)),
                               srcvalue, (fwti.Mask true_mask), VLOpFrag)),
              (!cast<Instruction>(instruction_name#"_W"#fvti.ScalarSuffix#"_"#fvti.LMul.MX)
                 fwti.RegClass:$rs2, fvti.ScalarRegClass:$rs1,
                 GPR:$vl, fvti.Log2SEW)>;
  }
}

multiclass VPatWidenBinaryFPVL_VV_VF_WV_WF<SDNode op, string instruction_name> {
  defm : VPatWidenBinaryFPVL_VV_VF<op, riscv_fpextend_vl_oneuse, instruction_name>;
  defm : VPatWidenBinaryFPVL_WV_WF<op, riscv_fpextend_vl_oneuse, instruction_name>;
}

multiclass VPatNarrowShiftSplatExt_WX<SDNode op, PatFrags extop, string instruction_name> {
  foreach vtiToWti = AllWidenableIntVectors in {
    defvar vti = vtiToWti.Vti;
    defvar wti = vtiToWti.Wti;
    def : Pat<
      (vti.Vector
        (riscv_trunc_vector_vl
          (op (wti.Vector wti.RegClass:$rs2),
              (wti.Vector (extop (vti.Vector (SplatPat GPR:$rs1)),
                                 (vti.Mask true_mask), VLOpFrag)),
          srcvalue, (wti.Mask true_mask), VLOpFrag),
        (vti.Mask true_mask), VLOpFrag)),
      (!cast<Instruction>(instruction_name#"_WX_"#vti.LMul.MX)
        wti.RegClass:$rs2, GPR:$rs1, GPR:$vl, vti.Log2SEW)>;
  }
}

multiclass VPatMultiplyAddVL_VV_VX<SDNode op, string instruction_name> {
  foreach vti = AllIntegerVectors in {
    defvar suffix = vti.LMul.MX;
    // NOTE: We choose VMADD because it has the most commuting freedom. So it
    // works best with how TwoAddressInstructionPass tries commuting.
    def : Pat<(vti.Vector
             (op vti.RegClass:$rs2,
                 (riscv_mul_vl_oneuse vti.RegClass:$rs1,
                                      vti.RegClass:$rd,
                                      srcvalue, (vti.Mask true_mask), VLOpFrag),
                           srcvalue, (vti.Mask true_mask), VLOpFrag)),
            (!cast<Instruction>(instruction_name#"_VV_"# suffix)
                 vti.RegClass:$rd, vti.RegClass:$rs1, vti.RegClass:$rs2,
                 GPR:$vl, vti.Log2SEW, TAIL_AGNOSTIC)>;
    // The choice of VMADD here is arbitrary, vmadd.vx and vmacc.vx are equally
    // commutable.
    def : Pat<(vti.Vector
             (op vti.RegClass:$rs2,
                 (riscv_mul_vl_oneuse (SplatPat XLenVT:$rs1),
                                       vti.RegClass:$rd,
                                       srcvalue, (vti.Mask true_mask), VLOpFrag),
                           srcvalue, (vti.Mask true_mask), VLOpFrag)),
            (!cast<Instruction>(instruction_name#"_VX_" # suffix)
                 vti.RegClass:$rd, vti.ScalarRegClass:$rs1, vti.RegClass:$rs2,
                 GPR:$vl, vti.Log2SEW, TAIL_AGNOSTIC)>;
  }
}

multiclass VPatWidenMultiplyAddVL_VV_VX<PatFrag op1, string instruction_name> {
  foreach vtiTowti = AllWidenableIntVectors in {
    defvar vti = vtiTowti.Vti;
    defvar wti = vtiTowti.Wti;
    def : Pat<(wti.Vector
             (riscv_add_vl wti.RegClass:$rd,
                           (op1 vti.RegClass:$rs1,
                                (vti.Vector vti.RegClass:$rs2),
                                srcvalue, (vti.Mask true_mask), VLOpFrag),
                          srcvalue, (vti.Mask true_mask), VLOpFrag)),
            (!cast<Instruction>(instruction_name#"_VV_" # vti.LMul.MX)
                 wti.RegClass:$rd, vti.RegClass:$rs1, vti.RegClass:$rs2,
                 GPR:$vl, vti.Log2SEW, TAIL_AGNOSTIC)>;
    def : Pat<(wti.Vector
             (riscv_add_vl wti.RegClass:$rd,
                          (op1 (SplatPat XLenVT:$rs1),
                               (vti.Vector vti.RegClass:$rs2),
                               srcvalue, (vti.Mask true_mask), VLOpFrag),
                           srcvalue, (vti.Mask true_mask), VLOpFrag)),
            (!cast<Instruction>(instruction_name#"_VX_" # vti.LMul.MX)
                 wti.RegClass:$rd, vti.ScalarRegClass:$rs1, vti.RegClass:$rs2,
                 GPR:$vl, vti.Log2SEW, TAIL_AGNOSTIC)>;
  }
}

multiclass VPatNarrowShiftSplat_WX_WI<SDNode op, string instruction_name> {
  foreach vtiTowti = AllWidenableIntVectors in {
    defvar vti = vtiTowti.Vti;
    defvar wti = vtiTowti.Wti;
    def : Pat<(vti.Vector (riscv_trunc_vector_vl
              (wti.Vector (op wti.RegClass:$rs1, (SplatPat XLenVT:$rs2),
                              srcvalue, true_mask, VLOpFrag)), true_mask, VLOpFrag)),
              (!cast<Instruction>(instruction_name#"_WX_"#vti.LMul.MX)
                   wti.RegClass:$rs1, GPR:$rs2, GPR:$vl, vti.Log2SEW)>;
    def : Pat<(vti.Vector (riscv_trunc_vector_vl
              (wti.Vector (op wti.RegClass:$rs1, (SplatPat_uimm5 uimm5:$rs2),
                              srcvalue, true_mask, VLOpFrag)), true_mask, VLOpFrag)),
              (!cast<Instruction>(instruction_name#"_WI_"#vti.LMul.MX)
                   wti.RegClass:$rs1, uimm5:$rs2, GPR:$vl, vti.Log2SEW)>;
  }
}

multiclass VPatFPMulAddVL_VV_VF<SDNode vop, string instruction_name> {
  foreach vti = AllFloatVectors in {
  defvar suffix = vti.LMul.MX;
  def : Pat<(vti.Vector (vop vti.RegClass:$rs1, vti.RegClass:$rd,
                             vti.RegClass:$rs2, (vti.Mask true_mask),
                             VLOpFrag)),
            (!cast<Instruction>(instruction_name#"_VV_"# suffix)
                 vti.RegClass:$rd, vti.RegClass:$rs1, vti.RegClass:$rs2,
                 GPR:$vl, vti.Log2SEW, TAIL_AGNOSTIC)>;
  def : Pat<(vti.Vector (vop vti.RegClass:$rs1, vti.RegClass:$rd,
                             vti.RegClass:$rs2, (vti.Mask V0),
                             VLOpFrag)),
            (!cast<Instruction>(instruction_name#"_VV_"# suffix #"_MASK")
                 vti.RegClass:$rd, vti.RegClass:$rs1, vti.RegClass:$rs2,
                 (vti.Mask V0), GPR:$vl, vti.Log2SEW, TAIL_AGNOSTIC)>;

  def : Pat<(vti.Vector (vop (SplatFPOp vti.ScalarRegClass:$rs1),
                             vti.RegClass:$rd, vti.RegClass:$rs2,
                             (vti.Mask true_mask),
                             VLOpFrag)),
            (!cast<Instruction>(instruction_name#"_V" # vti.ScalarSuffix # "_" # suffix)
                 vti.RegClass:$rd, vti.ScalarRegClass:$rs1, vti.RegClass:$rs2,
                 GPR:$vl, vti.Log2SEW, TAIL_AGNOSTIC)>;
  def : Pat<(vti.Vector (vop (SplatFPOp vti.ScalarRegClass:$rs1),
                             vti.RegClass:$rd, vti.RegClass:$rs2,
                             (vti.Mask V0),
                             VLOpFrag)),
            (!cast<Instruction>(instruction_name#"_V" # vti.ScalarSuffix # "_" # suffix # "_MASK")
                 vti.RegClass:$rd, vti.ScalarRegClass:$rs1, vti.RegClass:$rs2,
                 (vti.Mask V0), GPR:$vl, vti.Log2SEW, TAIL_AGNOSTIC)>;
  }
}

multiclass VPatWidenFPMulAccVL_VV_VF<SDNode vop, string instruction_name> {
  foreach vtiToWti = AllWidenableFloatVectors in {
    defvar vti = vtiToWti.Vti;
    defvar wti = vtiToWti.Wti;
    def : Pat<(vop
                   (wti.Vector (riscv_fpextend_vl_oneuse
                                    (vti.Vector vti.RegClass:$rs1),
                                    (vti.Mask true_mask), VLOpFrag)),
                   (wti.Vector (riscv_fpextend_vl_oneuse
                                    (vti.Vector vti.RegClass:$rs2),
                                    (vti.Mask true_mask), VLOpFrag)),
                   (wti.Vector wti.RegClass:$rd), (vti.Mask true_mask),
                   VLOpFrag),
              (!cast<Instruction>(instruction_name#"_VV_"#vti.LMul.MX)
                 wti.RegClass:$rd, vti.RegClass:$rs1, vti.RegClass:$rs2,
                 GPR:$vl, vti.Log2SEW, TAIL_AGNOSTIC)>;
    def : Pat<(vop
                   (wti.Vector (riscv_fpextend_vl_oneuse
                                    (vti.Vector (SplatFPOp vti.ScalarRegClass:$rs1)),
                                    (vti.Mask true_mask), VLOpFrag)),
                   (wti.Vector (riscv_fpextend_vl_oneuse
                                    (vti.Vector vti.RegClass:$rs2),
                                    (vti.Mask true_mask), VLOpFrag)),
                   (wti.Vector wti.RegClass:$rd), (vti.Mask true_mask),
                   VLOpFrag),
              (!cast<Instruction>(instruction_name#"_V"#vti.ScalarSuffix#"_"#vti.LMul.MX)
                 wti.RegClass:$rd, vti.ScalarRegClass:$rs1, vti.RegClass:$rs2,
                 GPR:$vl, vti.Log2SEW, TAIL_AGNOSTIC)>;
  }
}

//===----------------------------------------------------------------------===//
// Patterns.
//===----------------------------------------------------------------------===//

let Predicates = [HasVInstructions] in {

// 12.1. Vector Single-Width Integer Add and Subtract
defm : VPatBinaryVL_VV_VX_VI<riscv_add_vl, "PseudoVADD">;
defm : VPatBinaryVL_VV_VX<riscv_sub_vl, "PseudoVSUB">;
// Handle VRSUB specially since it's the only integer binary op with reversed
// pattern operands
foreach vti = AllIntegerVectors in {
  def : Pat<(riscv_sub_vl (vti.Vector (SplatPat (XLenVT GPR:$rs2))),
                          (vti.Vector vti.RegClass:$rs1),
                          vti.RegClass:$merge, (vti.Mask V0), VLOpFrag),
            (!cast<Instruction>("PseudoVRSUB_VX_"# vti.LMul.MX#"_MASK")
                 vti.RegClass:$merge, vti.RegClass:$rs1, GPR:$rs2,
                 (vti.Mask V0), GPR:$vl, vti.Log2SEW, TAIL_AGNOSTIC)>;
  def : Pat<(riscv_sub_vl (vti.Vector (SplatPat_simm5 simm5:$rs2)),
                          (vti.Vector vti.RegClass:$rs1),
                          vti.RegClass:$merge, (vti.Mask V0), VLOpFrag),
            (!cast<Instruction>("PseudoVRSUB_VI_"# vti.LMul.MX#"_MASK")
                 vti.RegClass:$merge, vti.RegClass:$rs1, simm5:$rs2,
                 (vti.Mask V0), GPR:$vl, vti.Log2SEW, TAIL_AGNOSTIC)>;
}

// 12.2. Vector Widening Integer Add/Subtract
defm : VPatBinaryWVL_VV_VX_WV_WX<riscv_vwadd_vl,  riscv_vwadd_w_vl,  "PseudoVWADD">;
defm : VPatBinaryWVL_VV_VX_WV_WX<riscv_vwaddu_vl, riscv_vwaddu_w_vl, "PseudoVWADDU">;
defm : VPatBinaryWVL_VV_VX_WV_WX<riscv_vwsub_vl,  riscv_vwsub_w_vl,  "PseudoVWSUB">;
defm : VPatBinaryWVL_VV_VX_WV_WX<riscv_vwsubu_vl, riscv_vwsubu_w_vl, "PseudoVWSUBU">;

// 12.3. Vector Integer Extension
defm : VPatExtendVL_V<riscv_zext_vl, "PseudoVZEXT", "VF2",
                      AllFractionableVF2IntVectors>;
defm : VPatExtendVL_V<riscv_sext_vl, "PseudoVSEXT", "VF2",
                      AllFractionableVF2IntVectors>;
defm : VPatExtendVL_V<riscv_zext_vl, "PseudoVZEXT", "VF4",
                      AllFractionableVF4IntVectors>;
defm : VPatExtendVL_V<riscv_sext_vl, "PseudoVSEXT", "VF4",
                      AllFractionableVF4IntVectors>;
defm : VPatExtendVL_V<riscv_zext_vl, "PseudoVZEXT", "VF8",
                      AllFractionableVF8IntVectors>;
defm : VPatExtendVL_V<riscv_sext_vl, "PseudoVSEXT", "VF8",
                      AllFractionableVF8IntVectors>;

// 12.5. Vector Bitwise Logical Instructions
defm : VPatBinaryVL_VV_VX_VI<riscv_and_vl, "PseudoVAND">;
defm : VPatBinaryVL_VV_VX_VI<riscv_or_vl,  "PseudoVOR">;
defm : VPatBinaryVL_VV_VX_VI<riscv_xor_vl, "PseudoVXOR">;

// 12.6. Vector Single-Width Bit Shift Instructions
defm : VPatBinaryVL_VV_VX_VI<riscv_shl_vl, "PseudoVSLL", uimm5>;
defm : VPatBinaryVL_VV_VX_VI<riscv_srl_vl, "PseudoVSRL", uimm5>;
defm : VPatBinaryVL_VV_VX_VI<riscv_sra_vl, "PseudoVSRA", uimm5>;

foreach vti = AllIntegerVectors in {
  // Emit shift by 1 as an add since it might be faster.
  def : Pat<(riscv_shl_vl (vti.Vector vti.RegClass:$rs1),
                          (riscv_vmv_v_x_vl (vti.Vector undef), 1, (XLenVT srcvalue)),
                          srcvalue, (vti.Mask true_mask), VLOpFrag),
            (!cast<Instruction>("PseudoVADD_VV_"# vti.LMul.MX)
                 vti.RegClass:$rs1, vti.RegClass:$rs1, GPR:$vl, vti.Log2SEW)>;
}

// 12.7. Vector Narrowing Integer Right Shift Instructions
defm : VPatBinaryVL_WV_WX_WI<srl, "PseudoVNSRL">;
defm : VPatBinaryVL_WV_WX_WI<sra, "PseudoVNSRA">;

defm : VPatNarrowShiftSplat_WX_WI<riscv_sra_vl, "PseudoVNSRA">;
defm : VPatNarrowShiftSplat_WX_WI<riscv_srl_vl, "PseudoVNSRL">;
defm : VPatNarrowShiftSplatExt_WX<riscv_sra_vl, riscv_sext_vl_oneuse, "PseudoVNSRA">;
defm : VPatNarrowShiftSplatExt_WX<riscv_sra_vl, riscv_zext_vl_oneuse, "PseudoVNSRA">;
defm : VPatNarrowShiftSplatExt_WX<riscv_srl_vl, riscv_sext_vl_oneuse, "PseudoVNSRL">;
defm : VPatNarrowShiftSplatExt_WX<riscv_srl_vl, riscv_zext_vl_oneuse, "PseudoVNSRL">;

defm : VPatBinaryNVL_WV_WX_WI<riscv_vnsrl_vl, "PseudoVNSRL">;

foreach vtiTowti = AllWidenableIntVectors in {
  defvar vti = vtiTowti.Vti;
  defvar wti = vtiTowti.Wti;
  def : Pat<(vti.Vector (riscv_trunc_vector_vl (wti.Vector wti.RegClass:$rs1),
                                               (vti.Mask V0),
                                               VLOpFrag)),
            (!cast<Instruction>("PseudoVNSRL_WI_"#vti.LMul.MX#"_MASK")
                (vti.Vector (IMPLICIT_DEF)), wti.RegClass:$rs1, 0,
                (vti.Mask V0), GPR:$vl, vti.Log2SEW, TAIL_AGNOSTIC)>;
}

// 12.8. Vector Integer Comparison Instructions
foreach vti = AllIntegerVectors in {
  defm : VPatIntegerSetCCVL_VV<vti, "PseudoVMSEQ", SETEQ>;
  defm : VPatIntegerSetCCVL_VV<vti, "PseudoVMSNE", SETNE>;

  defm : VPatIntegerSetCCVL_VV_Swappable<vti, "PseudoVMSLT",  SETLT,  SETGT>;
  defm : VPatIntegerSetCCVL_VV_Swappable<vti, "PseudoVMSLTU", SETULT, SETUGT>;
  defm : VPatIntegerSetCCVL_VV_Swappable<vti, "PseudoVMSLE",  SETLE,  SETGE>;
  defm : VPatIntegerSetCCVL_VV_Swappable<vti, "PseudoVMSLEU", SETULE, SETUGE>;

  defm : VPatIntegerSetCCVL_VX_Swappable<vti, "PseudoVMSEQ",  SETEQ,  SETEQ>;
  defm : VPatIntegerSetCCVL_VX_Swappable<vti, "PseudoVMSNE",  SETNE,  SETNE>;
  defm : VPatIntegerSetCCVL_VX_Swappable<vti, "PseudoVMSLT",  SETLT,  SETGT>;
  defm : VPatIntegerSetCCVL_VX_Swappable<vti, "PseudoVMSLTU", SETULT, SETUGT>;
  defm : VPatIntegerSetCCVL_VX_Swappable<vti, "PseudoVMSLE",  SETLE,  SETGE>;
  defm : VPatIntegerSetCCVL_VX_Swappable<vti, "PseudoVMSLEU", SETULE, SETUGE>;
  defm : VPatIntegerSetCCVL_VX_Swappable<vti, "PseudoVMSGT",  SETGT,  SETLT>;
  defm : VPatIntegerSetCCVL_VX_Swappable<vti, "PseudoVMSGTU", SETUGT, SETULT>;
  // There is no VMSGE(U)_VX instruction

  defm : VPatIntegerSetCCVL_VI_Swappable<vti, "PseudoVMSEQ",  SETEQ,  SETEQ>;
  defm : VPatIntegerSetCCVL_VI_Swappable<vti, "PseudoVMSNE",  SETNE,  SETNE>;
  defm : VPatIntegerSetCCVL_VI_Swappable<vti, "PseudoVMSLE",  SETLE,  SETGE>;
  defm : VPatIntegerSetCCVL_VI_Swappable<vti, "PseudoVMSLEU", SETULE, SETUGE>;
  defm : VPatIntegerSetCCVL_VI_Swappable<vti, "PseudoVMSGT",  SETGT,  SETLT>;
  defm : VPatIntegerSetCCVL_VI_Swappable<vti, "PseudoVMSGTU", SETUGT, SETULT>;

  defm : VPatIntegerSetCCVL_VIPlus1_Swappable<vti, "PseudoVMSLE",  SETLT, SETGT,
                                              SplatPat_simm5_plus1_nonzero>;
  defm : VPatIntegerSetCCVL_VIPlus1_Swappable<vti, "PseudoVMSLEU", SETULT, SETUGT,
                                              SplatPat_simm5_plus1_nonzero>;
  defm : VPatIntegerSetCCVL_VIPlus1_Swappable<vti, "PseudoVMSGT",  SETGE, SETLE,
                                              SplatPat_simm5_plus1>;
  defm : VPatIntegerSetCCVL_VIPlus1_Swappable<vti, "PseudoVMSGTU", SETUGE, SETULE,
                                              SplatPat_simm5_plus1_nonzero>;
} // foreach vti = AllIntegerVectors

// 12.9. Vector Integer Min/Max Instructions
defm : VPatBinaryVL_VV_VX<riscv_umin_vl, "PseudoVMINU">;
defm : VPatBinaryVL_VV_VX<riscv_smin_vl, "PseudoVMIN">;
defm : VPatBinaryVL_VV_VX<riscv_umax_vl, "PseudoVMAXU">;
defm : VPatBinaryVL_VV_VX<riscv_smax_vl, "PseudoVMAX">;

// 12.10. Vector Single-Width Integer Multiply Instructions
defm : VPatBinaryVL_VV_VX<riscv_mul_vl, "PseudoVMUL">;
defm : VPatBinaryVL_VV_VX<riscv_mulhs_vl, "PseudoVMULH">;
defm : VPatBinaryVL_VV_VX<riscv_mulhu_vl, "PseudoVMULHU">;

// 12.11. Vector Integer Divide Instructions
defm : VPatBinaryVL_VV_VX<riscv_udiv_vl, "PseudoVDIVU">;
defm : VPatBinaryVL_VV_VX<riscv_sdiv_vl, "PseudoVDIV">;
defm : VPatBinaryVL_VV_VX<riscv_urem_vl, "PseudoVREMU">;
defm : VPatBinaryVL_VV_VX<riscv_srem_vl, "PseudoVREM">;

// 12.12. Vector Widening Integer Multiply Instructions
defm : VPatBinaryWVL_VV_VX<riscv_vwmul_vl, "PseudoVWMUL">;
defm : VPatBinaryWVL_VV_VX<riscv_vwmulu_vl, "PseudoVWMULU">;
defm : VPatBinaryWVL_VV_VX<riscv_vwmulsu_vl, "PseudoVWMULSU">;

// 12.13 Vector Single-Width Integer Multiply-Add Instructions
defm : VPatMultiplyAddVL_VV_VX<riscv_add_vl, "PseudoVMADD">;
defm : VPatMultiplyAddVL_VV_VX<riscv_sub_vl, "PseudoVNMSUB">;

// 12.14. Vector Widening Integer Multiply-Add Instructions
defm : VPatWidenMultiplyAddVL_VV_VX<riscv_vwmul_vl_oneuse, "PseudoVWMACC">;
defm : VPatWidenMultiplyAddVL_VV_VX<riscv_vwmulu_vl_oneuse, "PseudoVWMACCU">;
defm : VPatWidenMultiplyAddVL_VV_VX<riscv_vwmulsu_vl_oneuse, "PseudoVWMACCSU">;
foreach vtiTowti = AllWidenableIntVectors in {
  defvar vti = vtiTowti.Vti;
  defvar wti = vtiTowti.Wti;
  def : Pat<(wti.Vector
             (riscv_add_vl wti.RegClass:$rd,
                           (riscv_vwmulsu_vl_oneuse (vti.Vector vti.RegClass:$rs1),
                                                    (SplatPat XLenVT:$rs2),
                                                    srcvalue,
                                                    (vti.Mask true_mask),
                                                    VLOpFrag),
                           srcvalue, (vti.Mask true_mask),VLOpFrag)),
            (!cast<Instruction>("PseudoVWMACCUS_VX_" # vti.LMul.MX)
                 wti.RegClass:$rd, vti.ScalarRegClass:$rs2, vti.RegClass:$rs1,
                 GPR:$vl, vti.Log2SEW, TAIL_AGNOSTIC)>;
}

// 12.15. Vector Integer Merge Instructions
foreach vti = AllIntegerVectors in {
  def : Pat<(vti.Vector (riscv_vselect_vl (vti.Mask V0),
                                          vti.RegClass:$rs1,
                                          vti.RegClass:$rs2,
                                          VLOpFrag)),
            (!cast<Instruction>("PseudoVMERGE_VVM_"#vti.LMul.MX)
                 vti.RegClass:$rs2, vti.RegClass:$rs1, (vti.Mask V0),
                 GPR:$vl, vti.Log2SEW)>;

  def : Pat<(vti.Vector (riscv_vselect_vl (vti.Mask V0),
                                          (SplatPat XLenVT:$rs1),
                                          vti.RegClass:$rs2,
                                          VLOpFrag)),
            (!cast<Instruction>("PseudoVMERGE_VXM_"#vti.LMul.MX)
                 vti.RegClass:$rs2, GPR:$rs1, (vti.Mask V0), GPR:$vl, vti.Log2SEW)>;

  def : Pat<(vti.Vector (riscv_vselect_vl (vti.Mask V0),
                                          (SplatPat_simm5 simm5:$rs1),
                                          vti.RegClass:$rs2,
                                          VLOpFrag)),
            (!cast<Instruction>("PseudoVMERGE_VIM_"#vti.LMul.MX)
                 vti.RegClass:$rs2, simm5:$rs1, (vti.Mask V0), GPR:$vl, vti.Log2SEW)>;

  def : Pat<(vti.Vector (riscv_vp_merge_vl (vti.Mask V0),
                                           vti.RegClass:$rs1,
                                           vti.RegClass:$rs2,
                                           VLOpFrag)),
            (!cast<Instruction>("PseudoVMERGE_VVM_"#vti.LMul.MX#"_TU")
                 vti.RegClass:$rs2, vti.RegClass:$rs2, vti.RegClass:$rs1,
                 (vti.Mask V0), GPR:$vl, vti.Log2SEW)>;

  def : Pat<(vti.Vector (riscv_vp_merge_vl (vti.Mask V0),
                                           (SplatPat XLenVT:$rs1),
                                           vti.RegClass:$rs2,
                                           VLOpFrag)),
            (!cast<Instruction>("PseudoVMERGE_VXM_"#vti.LMul.MX#"_TU")
                 vti.RegClass:$rs2, vti.RegClass:$rs2, GPR:$rs1,
                 (vti.Mask V0), GPR:$vl, vti.Log2SEW)>;

  def : Pat<(vti.Vector (riscv_vp_merge_vl (vti.Mask V0),
                                           (SplatPat_simm5 simm5:$rs1),
                                           vti.RegClass:$rs2,
                                           VLOpFrag)),
            (!cast<Instruction>("PseudoVMERGE_VIM_"#vti.LMul.MX#"_TU")
                 vti.RegClass:$rs2, vti.RegClass:$rs2, simm5:$rs1,
                 (vti.Mask V0), GPR:$vl, vti.Log2SEW)>;
}

// 12.16. Vector Integer Move Instructions
foreach vti = AllIntegerVectors in {
  def : Pat<(vti.Vector (riscv_vmv_v_x_vl (vti.Vector undef), GPR:$rs2, VLOpFrag)),
            (!cast<Instruction>("PseudoVMV_V_X_"#vti.LMul.MX)
             $rs2, GPR:$vl, vti.Log2SEW)>;
  def : Pat<(vti.Vector (riscv_vmv_v_x_vl vti.Vector:$passthru, GPR:$rs2, VLOpFrag)),
            (!cast<Instruction>("PseudoVMV_V_X_"#vti.LMul.MX#"_TU")
             $passthru, $rs2, GPR:$vl, vti.Log2SEW)>;
  defvar ImmPat = !cast<ComplexPattern>("sew"#vti.SEW#"simm5");
  def : Pat<(vti.Vector (riscv_vmv_v_x_vl (vti.Vector undef), (ImmPat XLenVT:$imm5),
                                          VLOpFrag)),
            (!cast<Instruction>("PseudoVMV_V_I_"#vti.LMul.MX)
             XLenVT:$imm5, GPR:$vl, vti.Log2SEW)>;
  def : Pat<(vti.Vector (riscv_vmv_v_x_vl vti.Vector:$passthru, (ImmPat XLenVT:$imm5),
                                              VLOpFrag)),
            (!cast<Instruction>("PseudoVMV_V_I_"#vti.LMul.MX#"_TU")
             $passthru, XLenVT:$imm5, GPR:$vl, vti.Log2SEW)>;
}

// 12.1. Vector Single-Width Saturating Add and Subtract
defm : VPatBinaryVL_VV_VX_VI<riscv_saddsat_vl, "PseudoVSADD">;
defm : VPatBinaryVL_VV_VX_VI<riscv_uaddsat_vl, "PseudoVSADDU">;
defm : VPatBinaryVL_VV_VX<riscv_ssubsat_vl, "PseudoVSSUB">;
defm : VPatBinaryVL_VV_VX<riscv_usubsat_vl, "PseudoVSSUBU">;

} // Predicates = [HasVInstructions]

// 15.1. Vector Single-Width Integer Reduction Instructions
let Predicates = [HasVInstructions] in {
defm : VPatReductionVL<rvv_vecreduce_ADD_vl,  "PseudoVREDSUM", /*is_float*/0>;
defm : VPatReductionVL<rvv_vecreduce_UMAX_vl, "PseudoVREDMAXU", /*is_float*/0>;
defm : VPatReductionVL<rvv_vecreduce_SMAX_vl, "PseudoVREDMAX", /*is_float*/0>;
defm : VPatReductionVL<rvv_vecreduce_UMIN_vl, "PseudoVREDMINU", /*is_float*/0>;
defm : VPatReductionVL<rvv_vecreduce_SMIN_vl, "PseudoVREDMIN", /*is_float*/0>;
defm : VPatReductionVL<rvv_vecreduce_AND_vl,  "PseudoVREDAND", /*is_float*/0>;
defm : VPatReductionVL<rvv_vecreduce_OR_vl,   "PseudoVREDOR", /*is_float*/0>;
defm : VPatReductionVL<rvv_vecreduce_XOR_vl,  "PseudoVREDXOR", /*is_float*/0>;

// 15.2. Vector Widening Integer Reduction Instructions
defm : VPatWidenReductionVL<rvv_vecreduce_ADD_vl, anyext_oneuse, "PseudoVWREDSUMU", /*is_float*/0>;
defm : VPatWidenReductionVL<rvv_vecreduce_ADD_vl, zext_oneuse, "PseudoVWREDSUMU", /*is_float*/0>;
defm : VPatWidenReductionVL_Ext_VL<rvv_vecreduce_ADD_vl, riscv_zext_vl_oneuse, "PseudoVWREDSUMU", /*is_float*/0>;
defm : VPatWidenReductionVL<rvv_vecreduce_ADD_vl, sext_oneuse, "PseudoVWREDSUM", /*is_float*/0>;
defm : VPatWidenReductionVL_Ext_VL<rvv_vecreduce_ADD_vl, riscv_sext_vl_oneuse, "PseudoVWREDSUM", /*is_float*/0>;
} // Predicates = [HasVInstructions]

// 15.3. Vector Single-Width Floating-Point Reduction Instructions
let Predicates = [HasVInstructionsAnyF] in {
defm : VPatReductionVL<rvv_vecreduce_SEQ_FADD_vl, "PseudoVFREDOSUM", /*is_float*/1>;
defm : VPatReductionVL<rvv_vecreduce_FADD_vl,     "PseudoVFREDUSUM", /*is_float*/1>;
defm : VPatReductionVL<rvv_vecreduce_FMIN_vl,     "PseudoVFREDMIN", /*is_float*/1>;
defm : VPatReductionVL<rvv_vecreduce_FMAX_vl,     "PseudoVFREDMAX", /*is_float*/1>;

// 15.4. Vector Widening Floating-Point Reduction Instructions
defm : VPatWidenReductionVL<rvv_vecreduce_SEQ_FADD_vl, fpext_oneuse, "PseudoVFWREDOSUM", /*is_float*/1>;
defm : VPatWidenReductionVL_Ext_VL<rvv_vecreduce_SEQ_FADD_vl, riscv_fpextend_vl_oneuse, "PseudoVFWREDOSUM", /*is_float*/1>;
defm : VPatWidenReductionVL<rvv_vecreduce_FADD_vl, fpext_oneuse, "PseudoVFWREDUSUM", /*is_float*/1>;
defm : VPatWidenReductionVL_Ext_VL<rvv_vecreduce_FADD_vl, riscv_fpextend_vl_oneuse, "PseudoVFWREDUSUM", /*is_float*/1>;
} // Predicates = [HasVInstructionsAnyF]

let Predicates = [HasVInstructionsAnyF] in {

// 14.2. Vector Single-Width Floating-Point Add/Subtract Instructions
defm : VPatBinaryFPVL_VV_VF<riscv_fadd_vl, "PseudoVFADD">;
defm : VPatBinaryFPVL_VV_VF<riscv_fsub_vl, "PseudoVFSUB">;
defm : VPatBinaryFPVL_R_VF<riscv_fsub_vl, "PseudoVFRSUB">;

// 14.3. Vector Widening Floating-Point Add/Subtract Instructions
defm : VPatWidenBinaryFPVL_VV_VF_WV_WF<riscv_fadd_vl, "PseudoVFWADD">;
defm : VPatWidenBinaryFPVL_VV_VF_WV_WF<riscv_fsub_vl, "PseudoVFWSUB">;

// 14.4. Vector Single-Width Floating-Point Multiply/Divide Instructions
defm : VPatBinaryFPVL_VV_VF<riscv_fmul_vl, "PseudoVFMUL">;
defm : VPatBinaryFPVL_VV_VF<riscv_fdiv_vl, "PseudoVFDIV">;
defm : VPatBinaryFPVL_R_VF<riscv_fdiv_vl, "PseudoVFRDIV">;

// 14.5. Vector Widening Floating-Point Multiply Instructions
defm : VPatWidenBinaryFPVL_VV_VF<riscv_fmul_vl, riscv_fpextend_vl_oneuse, "PseudoVFWMUL">;

// 14.6 Vector Single-Width Floating-Point Fused Multiply-Add Instructions.
defm : VPatFPMulAddVL_VV_VF<riscv_vfmadd_vl,  "PseudoVFMADD">;
defm : VPatFPMulAddVL_VV_VF<riscv_vfmsub_vl,  "PseudoVFMSUB">;
defm : VPatFPMulAddVL_VV_VF<riscv_vfnmadd_vl, "PseudoVFNMADD">;
defm : VPatFPMulAddVL_VV_VF<riscv_vfnmsub_vl, "PseudoVFNMSUB">;

// 14.7. Vector Widening Floating-Point Fused Multiply-Add Instructions
defm : VPatWidenFPMulAccVL_VV_VF<riscv_vfmadd_vl, "PseudoVFWMACC">;
defm : VPatWidenFPMulAccVL_VV_VF<riscv_vfnmadd_vl, "PseudoVFWNMACC">;
defm : VPatWidenFPMulAccVL_VV_VF<riscv_vfmsub_vl, "PseudoVFWMSAC">;
defm : VPatWidenFPMulAccVL_VV_VF<riscv_vfnmsub_vl, "PseudoVFWNMSAC">;

// 14.11. Vector Floating-Point MIN/MAX Instructions
defm : VPatBinaryFPVL_VV_VF<riscv_fminnum_vl, "PseudoVFMIN">;
defm : VPatBinaryFPVL_VV_VF<riscv_fmaxnum_vl, "PseudoVFMAX">;

// 14.13. Vector Floating-Point Compare Instructions
defm : VPatFPSetCCVL_VV_VF_FV<SETEQ,  "PseudoVMFEQ", "PseudoVMFEQ">;
defm : VPatFPSetCCVL_VV_VF_FV<SETOEQ, "PseudoVMFEQ", "PseudoVMFEQ">;

defm : VPatFPSetCCVL_VV_VF_FV<SETNE,  "PseudoVMFNE", "PseudoVMFNE">;
defm : VPatFPSetCCVL_VV_VF_FV<SETUNE, "PseudoVMFNE", "PseudoVMFNE">;

defm : VPatFPSetCCVL_VV_VF_FV<SETLT,  "PseudoVMFLT", "PseudoVMFGT">;
defm : VPatFPSetCCVL_VV_VF_FV<SETOLT, "PseudoVMFLT", "PseudoVMFGT">;

defm : VPatFPSetCCVL_VV_VF_FV<SETLE,  "PseudoVMFLE", "PseudoVMFGE">;
defm : VPatFPSetCCVL_VV_VF_FV<SETOLE, "PseudoVMFLE", "PseudoVMFGE">;

foreach vti = AllFloatVectors in {
  // 14.8. Vector Floating-Point Square-Root Instruction
  def : Pat<(riscv_fsqrt_vl (vti.Vector vti.RegClass:$rs2), (vti.Mask V0),
                            VLOpFrag),
            (!cast<Instruction>("PseudoVFSQRT_V_"# vti.LMul.MX #"_MASK")
                 (vti.Vector (IMPLICIT_DEF)), vti.RegClass:$rs2,
                 (vti.Mask V0), GPR:$vl, vti.Log2SEW, TAIL_AGNOSTIC)>;

  // 14.12. Vector Floating-Point Sign-Injection Instructions
  def : Pat<(riscv_fabs_vl (vti.Vector vti.RegClass:$rs), (vti.Mask V0),
                           VLOpFrag),
            (!cast<Instruction>("PseudoVFSGNJX_VV_"# vti.LMul.MX #"_MASK")
                 (vti.Vector (IMPLICIT_DEF)), vti.RegClass:$rs,
                 vti.RegClass:$rs, (vti.Mask V0), GPR:$vl, vti.Log2SEW,
                 TAIL_AGNOSTIC)>;
  // Handle fneg with VFSGNJN using the same input for both operands.
  def : Pat<(riscv_fneg_vl (vti.Vector vti.RegClass:$rs), (vti.Mask V0),
                           VLOpFrag),
            (!cast<Instruction>("PseudoVFSGNJN_VV_"# vti.LMul.MX #"_MASK")
                 (vti.Vector (IMPLICIT_DEF)), vti.RegClass:$rs,
                 vti.RegClass:$rs, (vti.Mask V0), GPR:$vl, vti.Log2SEW,
                 TAIL_AGNOSTIC)>;

  def : Pat<(riscv_fcopysign_vl (vti.Vector vti.RegClass:$rs1),
                                (vti.Vector vti.RegClass:$rs2),
                                vti.RegClass:$merge,
                                (vti.Mask V0),
                                VLOpFrag),
            (!cast<Instruction>("PseudoVFSGNJ_VV_"# vti.LMul.MX#"_MASK")
                 vti.RegClass:$merge, vti.RegClass:$rs1,
                 vti.RegClass:$rs2, (vti.Mask V0), GPR:$vl, vti.Log2SEW,
                 TAIL_AGNOSTIC)>;

  def : Pat<(riscv_fcopysign_vl (vti.Vector vti.RegClass:$rs1),
                                (riscv_fneg_vl vti.RegClass:$rs2,
                                               (vti.Mask true_mask),
                                               VLOpFrag),
                                srcvalue,
                                (vti.Mask true_mask),
                                VLOpFrag),
            (!cast<Instruction>("PseudoVFSGNJN_VV_"# vti.LMul.MX)
                 vti.RegClass:$rs1, vti.RegClass:$rs2, GPR:$vl, vti.Log2SEW)>;

  def : Pat<(riscv_fcopysign_vl (vti.Vector vti.RegClass:$rs1),
                                (SplatFPOp vti.ScalarRegClass:$rs2),
                                vti.RegClass:$merge,
                                (vti.Mask V0),
                                VLOpFrag),
            (!cast<Instruction>("PseudoVFSGNJ_V"#vti.ScalarSuffix#"_"# vti.LMul.MX#"_MASK")
                 vti.RegClass:$merge, vti.RegClass:$rs1,
                 vti.ScalarRegClass:$rs2, (vti.Mask V0), GPR:$vl, vti.Log2SEW,
                 TAIL_AGNOSTIC)>;
}

foreach fvti = AllFloatVectors in {
  // Floating-point vselects:
  // 12.15. Vector Integer Merge Instructions
  // 14.15. Vector Floating-Point Merge Instruction
  def : Pat<(fvti.Vector (riscv_vselect_vl (fvti.Mask V0),
                                           fvti.RegClass:$rs1,
                                           fvti.RegClass:$rs2,
                                           VLOpFrag)),
            (!cast<Instruction>("PseudoVMERGE_VVM_"#fvti.LMul.MX)
                 fvti.RegClass:$rs2, fvti.RegClass:$rs1, (fvti.Mask V0),
                 GPR:$vl, fvti.Log2SEW)>;

  def : Pat<(fvti.Vector (riscv_vselect_vl (fvti.Mask V0),
                                           (SplatFPOp fvti.ScalarRegClass:$rs1),
                                           fvti.RegClass:$rs2,
                                           VLOpFrag)),
            (!cast<Instruction>("PseudoVFMERGE_V"#fvti.ScalarSuffix#"M_"#fvti.LMul.MX)
                 fvti.RegClass:$rs2,
                 (fvti.Scalar fvti.ScalarRegClass:$rs1),
                 (fvti.Mask V0), GPR:$vl, fvti.Log2SEW)>;

  def : Pat<(fvti.Vector (riscv_vselect_vl (fvti.Mask V0),
                                           (SplatFPOp (fvti.Scalar fpimm0)),
                                           fvti.RegClass:$rs2,
                                           VLOpFrag)),
            (!cast<Instruction>("PseudoVMERGE_VIM_"#fvti.LMul.MX)
                 fvti.RegClass:$rs2, 0, (fvti.Mask V0), GPR:$vl, fvti.Log2SEW)>;

  def : Pat<(fvti.Vector (riscv_vp_merge_vl (fvti.Mask V0),
                                            fvti.RegClass:$rs1,
                                            fvti.RegClass:$rs2,
                                            VLOpFrag)),
            (!cast<Instruction>("PseudoVMERGE_VVM_"#fvti.LMul.MX#"_TU")
                 fvti.RegClass:$rs2, fvti.RegClass:$rs2, fvti.RegClass:$rs1, (fvti.Mask V0),
                 GPR:$vl, fvti.Log2SEW)>;

  def : Pat<(fvti.Vector (riscv_vp_merge_vl (fvti.Mask V0),
                                            (SplatFPOp fvti.ScalarRegClass:$rs1),
                                            fvti.RegClass:$rs2,
                                            VLOpFrag)),
            (!cast<Instruction>("PseudoVFMERGE_V"#fvti.ScalarSuffix#"M_"#fvti.LMul.MX#"_TU")
                 fvti.RegClass:$rs2, fvti.RegClass:$rs2,
                 (fvti.Scalar fvti.ScalarRegClass:$rs1),
                 (fvti.Mask V0), GPR:$vl, fvti.Log2SEW)>;

  def : Pat<(fvti.Vector (riscv_vp_merge_vl (fvti.Mask V0),
                                            (SplatFPOp (fvti.Scalar fpimm0)),
                                            fvti.RegClass:$rs2,
                                            VLOpFrag)),
            (!cast<Instruction>("PseudoVMERGE_VIM_"#fvti.LMul.MX#"_TU")
                 fvti.RegClass:$rs2, fvti.RegClass:$rs2, 0, (fvti.Mask V0),
                 GPR:$vl, fvti.Log2SEW)>;

  // 14.16. Vector Floating-Point Move Instruction
  // If we're splatting fpimm0, use vmv.v.x vd, x0.
  def : Pat<(fvti.Vector (riscv_vfmv_v_f_vl
                         (fvti.Vector undef), (fvti.Scalar (fpimm0)), VLOpFrag)),
            (!cast<Instruction>("PseudoVMV_V_I_"#fvti.LMul.MX)
             0, GPR:$vl, fvti.Log2SEW)>;
  def : Pat<(fvti.Vector (riscv_vfmv_v_f_vl
                         fvti.Vector:$passthru, (fvti.Scalar (fpimm0)), VLOpFrag)),
            (!cast<Instruction>("PseudoVMV_V_I_"#fvti.LMul.MX#"_TU")
             $passthru, 0, GPR:$vl, fvti.Log2SEW)>;

  def : Pat<(fvti.Vector (riscv_vfmv_v_f_vl
                         (fvti.Vector undef), (fvti.Scalar fvti.ScalarRegClass:$rs2), VLOpFrag)),
            (!cast<Instruction>("PseudoVFMV_V_" # fvti.ScalarSuffix # "_" #
                                fvti.LMul.MX)
             (fvti.Scalar fvti.ScalarRegClass:$rs2),
             GPR:$vl, fvti.Log2SEW)>;
  def : Pat<(fvti.Vector (riscv_vfmv_v_f_vl
                         fvti.Vector:$passthru, (fvti.Scalar fvti.ScalarRegClass:$rs2), VLOpFrag)),
            (!cast<Instruction>("PseudoVFMV_V_" # fvti.ScalarSuffix # "_" #
                                fvti.LMul.MX # "_TU")
             $passthru, (fvti.Scalar fvti.ScalarRegClass:$rs2),
             GPR:$vl, fvti.Log2SEW)>;

  // 14.17. Vector Single-Width Floating-Point/Integer Type-Convert Instructions
  defm : VPatConvertFP2I_RM_VL_V<riscv_vfcvt_x_f_vl,  "PseudoVFCVT_RM_X_F_V">;
  defm : VPatConvertFP2IVL_V<riscv_vfcvt_rtz_x_f_vl, "PseudoVFCVT_RTZ_X_F_V">;
  defm : VPatConvertFP2IVL_V<riscv_vfcvt_rtz_xu_f_vl, "PseudoVFCVT_RTZ_XU_F_V">;
  defm : VPatConvertI2FPVL_V<riscv_sint_to_fp_vl, "PseudoVFCVT_F_X_V">;
  defm : VPatConvertI2FPVL_V<riscv_uint_to_fp_vl, "PseudoVFCVT_F_XU_V">;

  // 14.18. Widening Floating-Point/Integer Type-Convert Instructions
  defm : VPatWConvertFP2IVL_V<riscv_vfcvt_rtz_x_f_vl, "PseudoVFWCVT_RTZ_X_F_V">;
  defm : VPatWConvertFP2IVL_V<riscv_vfcvt_rtz_xu_f_vl, "PseudoVFWCVT_RTZ_XU_F_V">;
  defm : VPatWConvertI2FPVL_V<riscv_sint_to_fp_vl, "PseudoVFWCVT_F_X_V">;
  defm : VPatWConvertI2FPVL_V<riscv_uint_to_fp_vl, "PseudoVFWCVT_F_XU_V">;
  foreach fvtiToFWti = AllWidenableFloatVectors in {
    defvar fvti = fvtiToFWti.Vti;
    defvar fwti = fvtiToFWti.Wti;
    def : Pat<(fwti.Vector (riscv_fpextend_vl (fvti.Vector fvti.RegClass:$rs1),
                                              (fvti.Mask V0),
                                              VLOpFrag)),
              (!cast<Instruction>("PseudoVFWCVT_F_F_V_"#fvti.LMul.MX#"_MASK")
                  (fwti.Vector (IMPLICIT_DEF)), fvti.RegClass:$rs1,
                  (fvti.Mask V0), GPR:$vl, fvti.Log2SEW, TAIL_AGNOSTIC)>;
  }

  // 14.19 Narrowing Floating-Point/Integer Type-Convert Instructions
  defm : VPatNConvertFP2IVL_V<riscv_vfcvt_rtz_x_f_vl, "PseudoVFNCVT_RTZ_X_F_W">;
  defm : VPatNConvertFP2IVL_V<riscv_vfcvt_rtz_xu_f_vl, "PseudoVFNCVT_RTZ_XU_F_W">;
  defm : VPatNConvertI2FPVL_V<riscv_sint_to_fp_vl, "PseudoVFNCVT_F_X_W">;
  defm : VPatNConvertI2FPVL_V<riscv_uint_to_fp_vl, "PseudoVFNCVT_F_XU_W">;
  foreach fvtiToFWti = AllWidenableFloatVectors in {
    defvar fvti = fvtiToFWti.Vti;
    defvar fwti = fvtiToFWti.Wti;
    def : Pat<(fvti.Vector (riscv_fpround_vl (fwti.Vector fwti.RegClass:$rs1),
                                             (fwti.Mask V0),
                                             VLOpFrag)),
              (!cast<Instruction>("PseudoVFNCVT_F_F_W_"#fvti.LMul.MX#"_MASK")
                  (fvti.Vector (IMPLICIT_DEF)), fwti.RegClass:$rs1,
                  (fwti.Mask V0), GPR:$vl, fvti.Log2SEW, TAIL_AGNOSTIC)>;

    def : Pat<(fvti.Vector (riscv_fncvt_rod_vl (fwti.Vector fwti.RegClass:$rs1),
                                               (fwti.Mask V0),
                                               VLOpFrag)),
              (!cast<Instruction>("PseudoVFNCVT_ROD_F_F_W_"#fvti.LMul.MX#"_MASK")
                  (fvti.Vector (IMPLICIT_DEF)), fwti.RegClass:$rs1,
                  (fwti.Mask V0), GPR:$vl, fvti.Log2SEW, TAIL_AGNOSTIC)>;
  }
}

} // Predicates = [HasVInstructionsAnyF]

let Predicates = [HasVInstructions] in {

foreach mti = AllMasks in {
  // 16.1 Vector Mask-Register Logical Instructions
  def : Pat<(mti.Mask (riscv_vmset_vl VLOpFrag)),
            (!cast<Instruction>("PseudoVMSET_M_" # mti.BX) GPR:$vl, mti.Log2SEW)>;
  def : Pat<(mti.Mask (riscv_vmclr_vl VLOpFrag)),
            (!cast<Instruction>("PseudoVMCLR_M_" # mti.BX) GPR:$vl, mti.Log2SEW)>;

  def : Pat<(mti.Mask (riscv_vmand_vl VR:$rs1, VR:$rs2, VLOpFrag)),
            (!cast<Instruction>("PseudoVMAND_MM_" # mti.LMul.MX)
                 VR:$rs1, VR:$rs2, GPR:$vl, mti.Log2SEW)>;
  def : Pat<(mti.Mask (riscv_vmor_vl VR:$rs1, VR:$rs2, VLOpFrag)),
            (!cast<Instruction>("PseudoVMOR_MM_" # mti.LMul.MX)
                 VR:$rs1, VR:$rs2, GPR:$vl, mti.Log2SEW)>;
  def : Pat<(mti.Mask (riscv_vmxor_vl VR:$rs1, VR:$rs2, VLOpFrag)),
            (!cast<Instruction>("PseudoVMXOR_MM_" # mti.LMul.MX)
                 VR:$rs1, VR:$rs2, GPR:$vl, mti.Log2SEW)>;

  def : Pat<(mti.Mask (riscv_vmand_vl VR:$rs1,
                                      (riscv_vmnot_vl VR:$rs2, VLOpFrag),
                                      VLOpFrag)),
            (!cast<Instruction>("PseudoVMANDN_MM_" # mti.LMul.MX)
                 VR:$rs1, VR:$rs2, GPR:$vl, mti.Log2SEW)>;
  def : Pat<(mti.Mask (riscv_vmor_vl VR:$rs1,
                                     (riscv_vmnot_vl VR:$rs2, VLOpFrag),
                                     VLOpFrag)),
            (!cast<Instruction>("PseudoVMORN_MM_" # mti.LMul.MX)
                 VR:$rs1, VR:$rs2, GPR:$vl, mti.Log2SEW)>;
  // XOR is associative so we need 2 patterns for VMXNOR.
  def : Pat<(mti.Mask (riscv_vmxor_vl (riscv_vmnot_vl VR:$rs1,
                                                      VLOpFrag),
                                     VR:$rs2, VLOpFrag)),
            (!cast<Instruction>("PseudoVMXNOR_MM_" # mti.LMul.MX)
                 VR:$rs1, VR:$rs2, GPR:$vl, mti.Log2SEW)>;

  def : Pat<(mti.Mask (riscv_vmnot_vl (riscv_vmand_vl VR:$rs1, VR:$rs2,
                                                      VLOpFrag),
                                      VLOpFrag)),
            (!cast<Instruction>("PseudoVMNAND_MM_" # mti.LMul.MX)
                 VR:$rs1, VR:$rs2, GPR:$vl, mti.Log2SEW)>;
  def : Pat<(mti.Mask (riscv_vmnot_vl (riscv_vmor_vl VR:$rs1, VR:$rs2,
                                                     VLOpFrag),
                                      VLOpFrag)),
            (!cast<Instruction>("PseudoVMNOR_MM_" # mti.LMul.MX)
                 VR:$rs1, VR:$rs2, GPR:$vl, mti.Log2SEW)>;
  def : Pat<(mti.Mask (riscv_vmnot_vl (riscv_vmxor_vl VR:$rs1, VR:$rs2,
                                                      VLOpFrag),
                                      VLOpFrag)),
            (!cast<Instruction>("PseudoVMXNOR_MM_" # mti.LMul.MX)
                 VR:$rs1, VR:$rs2, GPR:$vl, mti.Log2SEW)>;

  // Match the not idiom to the vmnot.m pseudo.
  def : Pat<(mti.Mask (riscv_vmnot_vl VR:$rs, VLOpFrag)),
            (!cast<Instruction>("PseudoVMNAND_MM_" # mti.LMul.MX)
                 VR:$rs, VR:$rs, GPR:$vl, mti.Log2SEW)>;

  // 16.2 Vector count population in mask vcpop.m
  def : Pat<(XLenVT (riscv_vcpop_vl (mti.Mask VR:$rs2), (mti.Mask true_mask),
                                    VLOpFrag)),
            (!cast<Instruction>("PseudoVCPOP_M_" # mti.BX)
                 VR:$rs2, GPR:$vl, mti.Log2SEW)>;
  def : Pat<(XLenVT (riscv_vcpop_vl (mti.Mask VR:$rs2), (mti.Mask V0),
                                    VLOpFrag)),
            (!cast<Instruction>("PseudoVCPOP_M_" # mti.BX # "_MASK")
                 VR:$rs2, (mti.Mask V0), GPR:$vl, mti.Log2SEW)>;
}

} // Predicates = [HasVInstructions]

let Predicates = [HasVInstructions] in {
// 17.1. Integer Scalar Move Instructions
// 17.4. Vector Register Gather Instruction
foreach vti = AllIntegerVectors in {
  def : Pat<(vti.Vector (riscv_vmv_s_x_vl (vti.Vector vti.RegClass:$merge),
                                          vti.ScalarRegClass:$rs1,
                                          VLOpFrag)),
            (!cast<Instruction>("PseudoVMV_S_X_"#vti.LMul.MX)
                vti.RegClass:$merge,
                (vti.Scalar vti.ScalarRegClass:$rs1), GPR:$vl, vti.Log2SEW)>;

  def : Pat<(vti.Vector (riscv_vrgather_vv_vl vti.RegClass:$rs2,
                                              vti.RegClass:$rs1,
                                              vti.RegClass:$merge,
                                              (vti.Mask V0),
                                              VLOpFrag)),
            (!cast<Instruction>("PseudoVRGATHER_VV_"# vti.LMul.MX#"_MASK")
                 vti.RegClass:$merge, vti.RegClass:$rs2, vti.RegClass:$rs1,
                 (vti.Mask V0), GPR:$vl, vti.Log2SEW, TAIL_AGNOSTIC)>;
  def : Pat<(vti.Vector (riscv_vrgather_vx_vl vti.RegClass:$rs2, GPR:$rs1,
                                              vti.RegClass:$merge,
                                              (vti.Mask V0),
                                              VLOpFrag)),
            (!cast<Instruction>("PseudoVRGATHER_VX_"# vti.LMul.MX#"_MASK")
                 vti.RegClass:$merge, vti.RegClass:$rs2, GPR:$rs1,
                 (vti.Mask V0), GPR:$vl, vti.Log2SEW, TAIL_AGNOSTIC)>;
  def : Pat<(vti.Vector (riscv_vrgather_vx_vl vti.RegClass:$rs2,
                                              uimm5:$imm,
                                              vti.RegClass:$merge,
                                              (vti.Mask V0),
                                              VLOpFrag)),
            (!cast<Instruction>("PseudoVRGATHER_VI_"# vti.LMul.MX#"_MASK")
                 vti.RegClass:$merge, vti.RegClass:$rs2, uimm5:$imm,
                 (vti.Mask V0), GPR:$vl, vti.Log2SEW, TAIL_AGNOSTIC)>;

  // emul = lmul * 16 / sew
  defvar vlmul = vti.LMul;
  defvar octuple_lmul = vlmul.octuple;
  defvar octuple_emul = !srl(!mul(octuple_lmul, 16), vti.Log2SEW);
  if !and(!ge(octuple_emul, 1), !le(octuple_emul, 64)) then {
    defvar emul_str = octuple_to_str<octuple_emul>.ret;
    defvar ivti = !cast<VTypeInfo>("VI16" # emul_str);
    defvar inst = "PseudoVRGATHEREI16_VV_" # vti.LMul.MX # "_" # emul_str;

    def : Pat<(vti.Vector
               (riscv_vrgatherei16_vv_vl vti.RegClass:$rs2,
                                         (ivti.Vector ivti.RegClass:$rs1),
                                         vti.RegClass:$merge,
                                         (vti.Mask V0),
                                         VLOpFrag)),
              (!cast<Instruction>(inst#"_MASK")
                   vti.RegClass:$merge, vti.RegClass:$rs2, ivti.RegClass:$rs1,
                   (vti.Mask V0), GPR:$vl, vti.Log2SEW, TAIL_AGNOSTIC)>;
  }
}

} // Predicates = [HasVInstructions]

let Predicates = [HasVInstructionsAnyF] in {

// 17.2. Floating-Point Scalar Move Instructions
foreach vti = AllFloatVectors in {
  def : Pat<(vti.Vector (riscv_vfmv_s_f_vl (vti.Vector vti.RegClass:$merge),
                                           (vti.Scalar (fpimm0)),
                                           VLOpFrag)),
            (!cast<Instruction>("PseudoVMV_S_X_"#vti.LMul.MX)
                vti.RegClass:$merge, X0, GPR:$vl, vti.Log2SEW)>;
  def : Pat<(vti.Vector (riscv_vfmv_s_f_vl (vti.Vector vti.RegClass:$merge),
                                           vti.ScalarRegClass:$rs1,
                                           VLOpFrag)),
            (!cast<Instruction>("PseudoVFMV_S_"#vti.ScalarSuffix#"_"#vti.LMul.MX)
                vti.RegClass:$merge,
                (vti.Scalar vti.ScalarRegClass:$rs1), GPR:$vl, vti.Log2SEW)>;
  defvar ivti = GetIntVTypeInfo<vti>.Vti;

  def : Pat<(vti.Vector
             (riscv_vrgather_vv_vl vti.RegClass:$rs2,
                                   (ivti.Vector vti.RegClass:$rs1),
                                   vti.RegClass:$merge,
                                   (vti.Mask V0),
                                   VLOpFrag)),
            (!cast<Instruction>("PseudoVRGATHER_VV_"# vti.LMul.MX#"_MASK")
                 vti.RegClass:$merge, vti.RegClass:$rs2, vti.RegClass:$rs1,
                 (vti.Mask V0), GPR:$vl, vti.Log2SEW, TAIL_AGNOSTIC)>;
  def : Pat<(vti.Vector (riscv_vrgather_vx_vl vti.RegClass:$rs2, GPR:$rs1,
                                              vti.RegClass:$merge,
                                              (vti.Mask V0),
                                              VLOpFrag)),
            (!cast<Instruction>("PseudoVRGATHER_VX_"# vti.LMul.MX#"_MASK")
                 vti.RegClass:$merge, vti.RegClass:$rs2, GPR:$rs1,
                 (vti.Mask V0), GPR:$vl, vti.Log2SEW, TAIL_AGNOSTIC)>;
  def : Pat<(vti.Vector
             (riscv_vrgather_vx_vl vti.RegClass:$rs2,
                                   uimm5:$imm,
                                   vti.RegClass:$merge,
                                   (vti.Mask V0),
                                   VLOpFrag)),
            (!cast<Instruction>("PseudoVRGATHER_VI_"# vti.LMul.MX#"_MASK")
                 vti.RegClass:$merge, vti.RegClass:$rs2, uimm5:$imm,
                 (vti.Mask V0), GPR:$vl, vti.Log2SEW, TAIL_AGNOSTIC)>;

  defvar vlmul = vti.LMul;
  defvar octuple_lmul = vlmul.octuple;
  defvar octuple_emul = !srl(!mul(octuple_lmul, 16), vti.Log2SEW);
  if !and(!ge(octuple_emul, 1), !le(octuple_emul, 64)) then {
    defvar emul_str = octuple_to_str<octuple_emul>.ret;
    defvar ivti = !cast<VTypeInfo>("VI16" # emul_str);
    defvar inst = "PseudoVRGATHEREI16_VV_" # vti.LMul.MX # "_" # emul_str;

    def : Pat<(vti.Vector
               (riscv_vrgatherei16_vv_vl vti.RegClass:$rs2,
                                         (ivti.Vector ivti.RegClass:$rs1),
                                         vti.RegClass:$merge,
                                         (vti.Mask V0),
                                         VLOpFrag)),
              (!cast<Instruction>(inst#"_MASK")
                   vti.RegClass:$merge, vti.RegClass:$rs2, ivti.RegClass:$rs1,
                   (vti.Mask V0), GPR:$vl, vti.Log2SEW, TAIL_AGNOSTIC)>;
  }
}

} // Predicates = [HasVInstructionsAnyF]

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
def SDTRVVSlide1 : SDTypeProfile<1, 5, [
  SDTCisVec<0>, SDTCisSameAs<1, 0>, SDTCisSameAs<2, 0>, SDTCisInt<0>,
  SDTCisVT<3, XLenVT>, SDTCVecEltisVT<4, i1>, SDTCisSameNumEltsAs<0, 4>,
  SDTCisVT<5, XLenVT>
]>;

def riscv_slideup_vl   : SDNode<"RISCVISD::VSLIDEUP_VL", SDTRVVSlide, []>;
def riscv_slide1up_vl  : SDNode<"RISCVISD::VSLIDE1UP_VL", SDTRVVSlide1, []>;
def riscv_slidedown_vl : SDNode<"RISCVISD::VSLIDEDOWN_VL", SDTRVVSlide, []>;
def riscv_slide1down_vl  : SDNode<"RISCVISD::VSLIDE1DOWN_VL", SDTRVVSlide1, []>;

let Predicates = [HasVInstructions] in {

foreach vti = AllIntegerVectors in {
  def : Pat<(vti.Vector (riscv_vid_vl (vti.Mask true_mask),
                                      VLOpFrag)),
            (!cast<Instruction>("PseudoVID_V_"#vti.LMul.MX) GPR:$vl, vti.Log2SEW)>;

  def : Pat<(vti.Vector (riscv_slide1up_vl (vti.Vector undef),
                                           (vti.Vector vti.RegClass:$rs1),
                                           GPR:$rs2, (vti.Mask true_mask),
                                           VLOpFrag)),
            (!cast<Instruction>("PseudoVSLIDE1UP_VX_"#vti.LMul.MX)
                vti.RegClass:$rs1, GPR:$rs2, GPR:$vl, vti.Log2SEW)>;
  def : Pat<(vti.Vector (riscv_slide1up_vl (vti.Vector vti.RegClass:$rd),
                                           (vti.Vector vti.RegClass:$rs1),
                                           GPR:$rs2, (vti.Mask true_mask),
                                           VLOpFrag)),
            (!cast<Instruction>("PseudoVSLIDE1UP_VX_"#vti.LMul.MX#"_TU")
                vti.RegClass:$rd, vti.RegClass:$rs1, GPR:$rs2, GPR:$vl, vti.Log2SEW)>;
  def : Pat<(vti.Vector (riscv_slide1down_vl (vti.Vector undef),
                                             (vti.Vector vti.RegClass:$rs1),
                                             GPR:$rs2, (vti.Mask true_mask),
                                             VLOpFrag)),
            (!cast<Instruction>("PseudoVSLIDE1DOWN_VX_"#vti.LMul.MX)
                vti.RegClass:$rs1, GPR:$rs2, GPR:$vl, vti.Log2SEW)>;
  def : Pat<(vti.Vector (riscv_slide1down_vl (vti.Vector vti.RegClass:$rd),
                                             (vti.Vector vti.RegClass:$rs1),
                                             GPR:$rs2, (vti.Mask true_mask),
                                             VLOpFrag)),
            (!cast<Instruction>("PseudoVSLIDE1DOWN_VX_"#vti.LMul.MX#"_TU")
                vti.RegClass:$rd, vti.RegClass:$rs1, GPR:$rs2, GPR:$vl, vti.Log2SEW)>;
}

foreach vti = !listconcat(AllIntegerVectors, AllFloatVectors) in {
  def : Pat<(vti.Vector (riscv_slideup_vl (vti.Vector vti.RegClass:$rs3),
                                          (vti.Vector vti.RegClass:$rs1),
                                          uimm5:$rs2, (vti.Mask true_mask),
                                          VLOpFrag)),
            (!cast<Instruction>("PseudoVSLIDEUP_VI_"#vti.LMul.MX)
                vti.RegClass:$rs3, vti.RegClass:$rs1, uimm5:$rs2,
                GPR:$vl, vti.Log2SEW, TAIL_UNDISTURBED_MASK_UNDISTURBED)>;

  def : Pat<(vti.Vector (riscv_slideup_vl (vti.Vector vti.RegClass:$rs3),
                                          (vti.Vector vti.RegClass:$rs1),
                                          GPR:$rs2, (vti.Mask true_mask),
                                          VLOpFrag)),
            (!cast<Instruction>("PseudoVSLIDEUP_VX_"#vti.LMul.MX)
                vti.RegClass:$rs3, vti.RegClass:$rs1, GPR:$rs2,
                GPR:$vl, vti.Log2SEW, TAIL_UNDISTURBED_MASK_UNDISTURBED)>;

  def : Pat<(vti.Vector (riscv_slidedown_vl (vti.Vector vti.RegClass:$rs3),
                                            (vti.Vector vti.RegClass:$rs1),
                                            uimm5:$rs2, (vti.Mask true_mask),
                                            VLOpFrag)),
            (!cast<Instruction>("PseudoVSLIDEDOWN_VI_"#vti.LMul.MX)
                vti.RegClass:$rs3, vti.RegClass:$rs1, uimm5:$rs2,
                GPR:$vl, vti.Log2SEW, TAIL_UNDISTURBED_MASK_UNDISTURBED)>;
  def : Pat<(vti.Vector (riscv_slidedown_vl (vti.Vector undef),
                                            (vti.Vector vti.RegClass:$rs1),
                                            uimm5:$rs2, (vti.Mask true_mask),
                                            VLOpFrag)),
            (!cast<Instruction>("PseudoVSLIDEDOWN_VI_"#vti.LMul.MX)
                (vti.Vector (IMPLICIT_DEF)), vti.RegClass:$rs1, uimm5:$rs2,
                GPR:$vl, vti.Log2SEW, TAIL_AGNOSTIC)>;

  def : Pat<(vti.Vector (riscv_slidedown_vl (vti.Vector vti.RegClass:$rs3),
                                            (vti.Vector vti.RegClass:$rs1),
                                            GPR:$rs2, (vti.Mask true_mask),
                                            VLOpFrag)),
            (!cast<Instruction>("PseudoVSLIDEDOWN_VX_"#vti.LMul.MX)
                vti.RegClass:$rs3, vti.RegClass:$rs1, GPR:$rs2,
                GPR:$vl, vti.Log2SEW, TAIL_UNDISTURBED_MASK_UNDISTURBED)>;
  def : Pat<(vti.Vector (riscv_slidedown_vl (vti.Vector undef),
                                            (vti.Vector vti.RegClass:$rs1),
                                            GPR:$rs2, (vti.Mask true_mask),
                                            VLOpFrag)),
            (!cast<Instruction>("PseudoVSLIDEDOWN_VX_"#vti.LMul.MX)
                (vti.Vector (IMPLICIT_DEF)), vti.RegClass:$rs1, GPR:$rs2,
                GPR:$vl, vti.Log2SEW, TAIL_AGNOSTIC)>;
}

} // Predicates = [HasVInstructions]
