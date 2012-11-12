//===- IntrinsicsMips.td - Defines Mips intrinsics ---------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the MIPS-specific intrinsics.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MIPS DSP data types
def mips_v2q15_ty: LLVMType<v2i16>;
def mips_v4q7_ty: LLVMType<v4i8>;
def mips_q31_ty: LLVMType<i32>;

let TargetPrefix = "mips" in {  // All intrinsics start with "llvm.mips.".

//===----------------------------------------------------------------------===//
// MIPS DSP Rev 1

//===----------------------------------------------------------------------===//
// Addition/subtraction

def int_mips_addu_qb : GCCBuiltin<"__builtin_mips_addu_qb">,
  Intrinsic<[llvm_v4i8_ty], [llvm_v4i8_ty, llvm_v4i8_ty], [Commutative]>;
def int_mips_addu_s_qb : GCCBuiltin<"__builtin_mips_addu_s_qb">,
  Intrinsic<[llvm_v4i8_ty], [llvm_v4i8_ty, llvm_v4i8_ty], [Commutative]>;
def int_mips_subu_qb : GCCBuiltin<"__builtin_mips_subu_qb">,
  Intrinsic<[llvm_v4i8_ty], [llvm_v4i8_ty, llvm_v4i8_ty], []>;
def int_mips_subu_s_qb : GCCBuiltin<"__builtin_mips_subu_s_qb">,
  Intrinsic<[llvm_v4i8_ty], [llvm_v4i8_ty, llvm_v4i8_ty], []>;

def int_mips_addq_ph : GCCBuiltin<"__builtin_mips_addq_ph">,
  Intrinsic<[mips_v2q15_ty], [mips_v2q15_ty, mips_v2q15_ty], [Commutative]>;
def int_mips_addq_s_ph : GCCBuiltin<"__builtin_mips_addq_s_ph">,
  Intrinsic<[mips_v2q15_ty], [mips_v2q15_ty, mips_v2q15_ty], [Commutative]>;
def int_mips_subq_ph : GCCBuiltin<"__builtin_mips_subq_ph">,
  Intrinsic<[mips_v2q15_ty], [mips_v2q15_ty, mips_v2q15_ty], []>;
def int_mips_subq_s_ph : GCCBuiltin<"__builtin_mips_subq_s_ph">,
  Intrinsic<[mips_v2q15_ty], [mips_v2q15_ty, mips_v2q15_ty], []>;

def int_mips_madd: GCCBuiltin<"__builtin_mips_madd">,
  Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty],
            [IntrNoMem, Commutative]>;
def int_mips_maddu: GCCBuiltin<"__builtin_mips_maddu">,
  Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty],
            [IntrNoMem, Commutative]>;

def int_mips_msub: GCCBuiltin<"__builtin_mips_msub">,
  Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty],
            [IntrNoMem]>;
def int_mips_msubu: GCCBuiltin<"__builtin_mips_msubu">,
  Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i32_ty, llvm_i32_ty],
            [IntrNoMem]>;

def int_mips_addq_s_w: GCCBuiltin<"__builtin_mips_addq_s_w">,
  Intrinsic<[mips_q31_ty], [mips_q31_ty, mips_q31_ty], [Commutative]>;
def int_mips_subq_s_w: GCCBuiltin<"__builtin_mips_subq_s_w">,
  Intrinsic<[mips_q31_ty], [mips_q31_ty, mips_q31_ty], []>;

def int_mips_addsc: GCCBuiltin<"__builtin_mips_addsc">,
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [Commutative]>;
def int_mips_addwc: GCCBuiltin<"__builtin_mips_addwc">,
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [Commutative]>;

def int_mips_modsub: GCCBuiltin<"__builtin_mips_modsub">,
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;

def int_mips_raddu_w_qb: GCCBuiltin<"__builtin_mips_raddu_w_qb">,
  Intrinsic<[llvm_i32_ty], [llvm_v4i8_ty], [IntrNoMem]>;

//===----------------------------------------------------------------------===//
// Absolute value

def int_mips_absq_s_ph: GCCBuiltin<"__builtin_mips_absq_s_ph">,
  Intrinsic<[mips_v2q15_ty], [mips_v2q15_ty], []>;
def int_mips_absq_s_w: GCCBuiltin<"__builtin_mips_absq_s_w">,
  Intrinsic<[mips_q31_ty], [mips_q31_ty], []>;

//===----------------------------------------------------------------------===//
// Precision reduce/expand

def int_mips_precrq_qb_ph: GCCBuiltin<"__builtin_mips_precrq_qb_ph">,
  Intrinsic<[llvm_v4i8_ty], [mips_v2q15_ty, mips_v2q15_ty], [IntrNoMem]>;
def int_mips_precrqu_s_qb_ph: GCCBuiltin<"__builtin_mips_precrqu_s_qb_ph">,
  Intrinsic<[llvm_v4i8_ty], [mips_v2q15_ty, mips_v2q15_ty], []>;
def int_mips_precrq_ph_w: GCCBuiltin<"__builtin_mips_precrq_ph_w">,
  Intrinsic<[mips_v2q15_ty], [mips_q31_ty, mips_q31_ty], [IntrNoMem]>;
def int_mips_precrq_rs_ph_w: GCCBuiltin<"__builtin_mips_precrq_rs_ph_w">,
  Intrinsic<[mips_v2q15_ty], [mips_q31_ty, mips_q31_ty], []>;
def int_mips_preceq_w_phl: GCCBuiltin<"__builtin_mips_preceq_w_phl">,
  Intrinsic<[mips_q31_ty], [mips_v2q15_ty], [IntrNoMem]>;
def int_mips_preceq_w_phr: GCCBuiltin<"__builtin_mips_preceq_w_phr">,
  Intrinsic<[mips_q31_ty], [mips_v2q15_ty], [IntrNoMem]>;
def int_mips_precequ_ph_qbl: GCCBuiltin<"__builtin_mips_precequ_ph_qbl">,
  Intrinsic<[mips_v2q15_ty], [llvm_v4i8_ty], [IntrNoMem]>;
def int_mips_precequ_ph_qbr: GCCBuiltin<"__builtin_mips_precequ_ph_qbr">,
  Intrinsic<[mips_v2q15_ty], [llvm_v4i8_ty], [IntrNoMem]>;
def int_mips_precequ_ph_qbla: GCCBuiltin<"__builtin_mips_precequ_ph_qbla">,
  Intrinsic<[mips_v2q15_ty], [llvm_v4i8_ty], [IntrNoMem]>;
def int_mips_precequ_ph_qbra: GCCBuiltin<"__builtin_mips_precequ_ph_qbra">,
  Intrinsic<[mips_v2q15_ty], [llvm_v4i8_ty], [IntrNoMem]>;
def int_mips_preceu_ph_qbl: GCCBuiltin<"__builtin_mips_preceu_ph_qbl">,
  Intrinsic<[mips_v2q15_ty], [llvm_v4i8_ty], [IntrNoMem]>;
def int_mips_preceu_ph_qbr: GCCBuiltin<"__builtin_mips_preceu_ph_qbr">,
  Intrinsic<[mips_v2q15_ty], [llvm_v4i8_ty], [IntrNoMem]>;
def int_mips_preceu_ph_qbla: GCCBuiltin<"__builtin_mips_preceu_ph_qbla">,
  Intrinsic<[mips_v2q15_ty], [llvm_v4i8_ty], [IntrNoMem]>;
def int_mips_preceu_ph_qbra: GCCBuiltin<"__builtin_mips_preceu_ph_qbra">,
  Intrinsic<[mips_v2q15_ty], [llvm_v4i8_ty], [IntrNoMem]>;

//===----------------------------------------------------------------------===//
// Shift

def int_mips_shll_qb: GCCBuiltin<"__builtin_mips_shll_qb">,
  Intrinsic<[llvm_v4i8_ty], [llvm_v4i8_ty, llvm_i32_ty], []>;
def int_mips_shrl_qb: GCCBuiltin<"__builtin_mips_shrl_qb">,
  Intrinsic<[llvm_v4i8_ty], [llvm_v4i8_ty, llvm_i32_ty], [IntrNoMem]>;
def int_mips_shll_ph: GCCBuiltin<"__builtin_mips_shll_ph">,
  Intrinsic<[mips_v2q15_ty], [mips_v2q15_ty, llvm_i32_ty], []>;
def int_mips_shll_s_ph: GCCBuiltin<"__builtin_mips_shll_s_ph">,
  Intrinsic<[mips_v2q15_ty], [mips_v2q15_ty, llvm_i32_ty], []>;
def int_mips_shra_ph: GCCBuiltin<"__builtin_mips_shra_ph">,
  Intrinsic<[mips_v2q15_ty], [mips_v2q15_ty, llvm_i32_ty], [IntrNoMem]>;
def int_mips_shra_r_ph: GCCBuiltin<"__builtin_mips_shra_r_ph">,
  Intrinsic<[mips_v2q15_ty], [mips_v2q15_ty, llvm_i32_ty], [IntrNoMem]>;
def int_mips_shll_s_w: GCCBuiltin<"__builtin_mips_shll_s_w">,
  Intrinsic<[mips_q31_ty], [mips_q31_ty, llvm_i32_ty], []>;
def int_mips_shra_r_w: GCCBuiltin<"__builtin_mips_shra_r_w">,
  Intrinsic<[mips_q31_ty], [mips_q31_ty, llvm_i32_ty], [IntrNoMem]>;
def int_mips_shilo: GCCBuiltin<"__builtin_mips_shilo">,
  Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i32_ty], [IntrNoMem]>;

//===----------------------------------------------------------------------===//
// Multiplication

def int_mips_muleu_s_ph_qbl: GCCBuiltin<"__builtin_mips_muleu_s_ph_qbl">,
  Intrinsic<[mips_v2q15_ty], [llvm_v4i8_ty, mips_v2q15_ty], []>;
def int_mips_muleu_s_ph_qbr: GCCBuiltin<"__builtin_mips_muleu_s_ph_qbr">,
  Intrinsic<[mips_v2q15_ty], [llvm_v4i8_ty, mips_v2q15_ty], []>;
def int_mips_mulq_rs_ph: GCCBuiltin<"__builtin_mips_mulq_rs_ph">,
  Intrinsic<[mips_v2q15_ty], [mips_v2q15_ty, mips_v2q15_ty], [Commutative]>;
def int_mips_muleq_s_w_phl: GCCBuiltin<"__builtin_mips_muleq_s_w_phl">,
  Intrinsic<[mips_q31_ty], [mips_v2q15_ty, mips_v2q15_ty], [Commutative]>;
def int_mips_muleq_s_w_phr: GCCBuiltin<"__builtin_mips_muleq_s_w_phr">,
  Intrinsic<[mips_q31_ty], [mips_v2q15_ty, mips_v2q15_ty], [Commutative]>;
def int_mips_mulsaq_s_w_ph: GCCBuiltin<"__builtin_mips_mulsaq_s_w_ph">,
  Intrinsic<[llvm_i64_ty], [llvm_i64_ty, mips_v2q15_ty, mips_v2q15_ty], []>;
def int_mips_maq_s_w_phl: GCCBuiltin<"__builtin_mips_maq_s_w_phl">,
  Intrinsic<[llvm_i64_ty], [llvm_i64_ty, mips_v2q15_ty, mips_v2q15_ty], []>;
def int_mips_maq_s_w_phr: GCCBuiltin<"__builtin_mips_maq_s_w_phr">,
  Intrinsic<[llvm_i64_ty], [llvm_i64_ty, mips_v2q15_ty, mips_v2q15_ty], []>;
def int_mips_maq_sa_w_phl: GCCBuiltin<"__builtin_mips_maq_sa_w_phl">,
  Intrinsic<[llvm_i64_ty], [llvm_i64_ty, mips_v2q15_ty, mips_v2q15_ty], []>;
def int_mips_maq_sa_w_phr: GCCBuiltin<"__builtin_mips_maq_sa_w_phr">,
  Intrinsic<[llvm_i64_ty], [llvm_i64_ty, mips_v2q15_ty, mips_v2q15_ty], []>;
def int_mips_mult: GCCBuiltin<"__builtin_mips_mult">,
  Intrinsic<[llvm_i64_ty], [llvm_i32_ty, llvm_i32_ty],
            [IntrNoMem, Commutative]>;
def int_mips_multu: GCCBuiltin<"__builtin_mips_multu">,
  Intrinsic<[llvm_i64_ty], [llvm_i32_ty, llvm_i32_ty],
            [IntrNoMem, Commutative]>;

//===----------------------------------------------------------------------===//
// Dot product with accumulate/subtract

def int_mips_dpau_h_qbl: GCCBuiltin<"__builtin_mips_dpau_h_qbl">,
  Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_v4i8_ty, llvm_v4i8_ty],
            [IntrNoMem]>;
def int_mips_dpau_h_qbr: GCCBuiltin<"__builtin_mips_dpau_h_qbr">,
  Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_v4i8_ty, llvm_v4i8_ty],
            [IntrNoMem]>;
def int_mips_dpsu_h_qbl: GCCBuiltin<"__builtin_mips_dpsu_h_qbl">,
  Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_v4i8_ty, llvm_v4i8_ty],
            [IntrNoMem]>;
def int_mips_dpsu_h_qbr: GCCBuiltin<"__builtin_mips_dpsu_h_qbr">,
  Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_v4i8_ty, llvm_v4i8_ty],
            [IntrNoMem]>;
def int_mips_dpaq_s_w_ph: GCCBuiltin<"__builtin_mips_dpaq_s_w_ph">,
  Intrinsic<[llvm_i64_ty], [llvm_i64_ty, mips_v2q15_ty, mips_v2q15_ty], []>;
def int_mips_dpsq_s_w_ph: GCCBuiltin<"__builtin_mips_dpsq_s_w_ph">,
  Intrinsic<[llvm_i64_ty], [llvm_i64_ty, mips_v2q15_ty, mips_v2q15_ty], []>;
def int_mips_dpaq_sa_l_w: GCCBuiltin<"__builtin_mips_dpaq_sa_l_w">,
  Intrinsic<[llvm_i64_ty], [llvm_i64_ty, mips_q31_ty, mips_q31_ty], []>;
def int_mips_dpsq_sa_l_w: GCCBuiltin<"__builtin_mips_dpsq_sa_l_w">,
  Intrinsic<[llvm_i64_ty], [llvm_i64_ty, mips_q31_ty, mips_q31_ty], []>;

//===----------------------------------------------------------------------===//
// Comparison

def int_mips_cmpu_eq_qb: GCCBuiltin<"__builtin_mips_cmpu_eq_qb">,
  Intrinsic<[], [llvm_v4i8_ty, llvm_v4i8_ty], [Commutative]>;
def int_mips_cmpu_lt_qb: GCCBuiltin<"__builtin_mips_cmpu_lt_qb">,
  Intrinsic<[], [llvm_v4i8_ty, llvm_v4i8_ty], [Commutative]>;
def int_mips_cmpu_le_qb: GCCBuiltin<"__builtin_mips_cmpu_le_qb">,
  Intrinsic<[], [llvm_v4i8_ty, llvm_v4i8_ty], [Commutative]>;
def int_mips_cmpgu_eq_qb: GCCBuiltin<"__builtin_mips_cmpgu_eq_qb">,
  Intrinsic<[llvm_i32_ty], [llvm_v4i8_ty, llvm_v4i8_ty], [Commutative]>;
def int_mips_cmpgu_lt_qb: GCCBuiltin<"__builtin_mips_cmpgu_lt_qb">,
  Intrinsic<[llvm_i32_ty], [llvm_v4i8_ty, llvm_v4i8_ty], [Commutative]>;
def int_mips_cmpgu_le_qb: GCCBuiltin<"__builtin_mips_cmpgu_le_qb">,
  Intrinsic<[llvm_i32_ty], [llvm_v4i8_ty, llvm_v4i8_ty], [Commutative]>;
def int_mips_cmp_eq_ph: GCCBuiltin<"__builtin_mips_cmp_eq_ph">,
  Intrinsic<[], [mips_v2q15_ty, mips_v2q15_ty], [Commutative]>;
def int_mips_cmp_lt_ph: GCCBuiltin<"__builtin_mips_cmp_lt_ph">,
  Intrinsic<[], [mips_v2q15_ty, mips_v2q15_ty], [Commutative]>;
def int_mips_cmp_le_ph: GCCBuiltin<"__builtin_mips_cmp_le_ph">,
  Intrinsic<[], [mips_v2q15_ty, mips_v2q15_ty], [Commutative]>;

//===----------------------------------------------------------------------===//
// Extracting

def int_mips_extr_s_h: GCCBuiltin<"__builtin_mips_extr_s_h">,
  Intrinsic<[llvm_i32_ty], [llvm_i64_ty, llvm_i32_ty], []>;
def int_mips_extr_w: GCCBuiltin<"__builtin_mips_extr_w">,
  Intrinsic<[llvm_i32_ty], [llvm_i64_ty, llvm_i32_ty], []>;
def int_mips_extr_rs_w: GCCBuiltin<"__builtin_mips_extr_rs_w">,
  Intrinsic<[llvm_i32_ty], [llvm_i64_ty, llvm_i32_ty], []>;
def int_mips_extr_r_w: GCCBuiltin<"__builtin_mips_extr_r_w">,
  Intrinsic<[llvm_i32_ty], [llvm_i64_ty, llvm_i32_ty], []>;
def int_mips_extp: GCCBuiltin<"__builtin_mips_extp">,
  Intrinsic<[llvm_i32_ty], [llvm_i64_ty, llvm_i32_ty], []>;
def int_mips_extpdp: GCCBuiltin<"__builtin_mips_extpdp">,
  Intrinsic<[llvm_i32_ty], [llvm_i64_ty, llvm_i32_ty], []>;

//===----------------------------------------------------------------------===//
// Misc

def int_mips_wrdsp: GCCBuiltin<"__builtin_mips_wrdsp">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], []>;
def int_mips_rddsp: GCCBuiltin<"__builtin_mips_rddsp">,
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrReadMem]>;

def int_mips_insv: GCCBuiltin<"__builtin_mips_insv">,
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrReadMem]>;
def int_mips_bitrev: GCCBuiltin<"__builtin_mips_bitrev">,
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]>;

def int_mips_packrl_ph: GCCBuiltin<"__builtin_mips_packrl_ph">,
  Intrinsic<[mips_v2q15_ty], [mips_v2q15_ty, mips_v2q15_ty], [IntrNoMem]>;

def int_mips_repl_qb: GCCBuiltin<"__builtin_mips_repl_qb">,
  Intrinsic<[llvm_v4i8_ty], [llvm_i32_ty], [IntrNoMem]>;
def int_mips_repl_ph: GCCBuiltin<"__builtin_mips_repl_ph">,
  Intrinsic<[mips_v2q15_ty], [llvm_i32_ty], [IntrNoMem]>;

def int_mips_pick_qb: GCCBuiltin<"__builtin_mips_pick_qb">,
  Intrinsic<[llvm_v4i8_ty], [llvm_v4i8_ty, llvm_v4i8_ty], [IntrReadMem]>;
def int_mips_pick_ph: GCCBuiltin<"__builtin_mips_pick_ph">,
  Intrinsic<[mips_v2q15_ty], [mips_v2q15_ty, mips_v2q15_ty], [IntrReadMem]>;

def int_mips_mthlip: GCCBuiltin<"__builtin_mips_mthlip">,
  Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i32_ty], []>;

def int_mips_bposge32: GCCBuiltin<"__builtin_mips_bposge32">,
  Intrinsic<[llvm_i32_ty], [], [IntrReadMem]>;

def int_mips_lbux: GCCBuiltin<"__builtin_mips_lbux">,
  Intrinsic<[llvm_i32_ty], [llvm_ptr_ty, llvm_i32_ty], [IntrReadArgMem]>;
def int_mips_lhx: GCCBuiltin<"__builtin_mips_lhx">,
  Intrinsic<[llvm_i32_ty], [llvm_ptr_ty, llvm_i32_ty], [IntrReadArgMem]>;
def int_mips_lwx: GCCBuiltin<"__builtin_mips_lwx">,
  Intrinsic<[llvm_i32_ty], [llvm_ptr_ty, llvm_i32_ty], [IntrReadArgMem]>;

//===----------------------------------------------------------------------===//
// MIPS DSP Rev 2

def int_mips_absq_s_qb: GCCBuiltin<"__builtin_mips_absq_s_qb">,
  Intrinsic<[mips_v4q7_ty], [mips_v4q7_ty], []>;

def int_mips_addqh_ph: GCCBuiltin<"__builtin_mips_addqh_ph">,
  Intrinsic<[mips_v2q15_ty], [mips_v2q15_ty, mips_v2q15_ty],
            [IntrNoMem, Commutative]>;
def int_mips_addqh_r_ph: GCCBuiltin<"__builtin_mips_addqh_r_ph">,
  Intrinsic<[mips_v2q15_ty], [mips_v2q15_ty, mips_v2q15_ty],
            [IntrNoMem, Commutative]>;
def int_mips_addqh_w: GCCBuiltin<"__builtin_mips_addqh_w">,
  Intrinsic<[mips_q31_ty], [mips_q31_ty, mips_q31_ty],
            [IntrNoMem, Commutative]>;
def int_mips_addqh_r_w: GCCBuiltin<"__builtin_mips_addqh_r_w">,
  Intrinsic<[mips_q31_ty], [mips_q31_ty, mips_q31_ty],
            [IntrNoMem, Commutative]>;

def int_mips_addu_ph: GCCBuiltin<"__builtin_mips_addu_ph">,
  Intrinsic<[llvm_v2i16_ty], [llvm_v2i16_ty, llvm_v2i16_ty], [Commutative]>;
def int_mips_addu_s_ph: GCCBuiltin<"__builtin_mips_addu_s_ph">,
  Intrinsic<[llvm_v2i16_ty], [llvm_v2i16_ty, llvm_v2i16_ty], [Commutative]>;

def int_mips_adduh_qb: GCCBuiltin<"__builtin_mips_adduh_qb">,
  Intrinsic<[llvm_v4i8_ty], [llvm_v4i8_ty, llvm_v4i8_ty],
            [IntrNoMem, Commutative]>;
def int_mips_adduh_r_qb: GCCBuiltin<"__builtin_mips_adduh_r_qb">,
  Intrinsic<[llvm_v4i8_ty], [llvm_v4i8_ty, llvm_v4i8_ty],
            [IntrNoMem, Commutative]>;

def int_mips_append: GCCBuiltin<"__builtin_mips_append">,
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
  [IntrNoMem]>;
def int_mips_balign: GCCBuiltin<"__builtin_mips_balign">,
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
  [IntrNoMem]>;

def int_mips_cmpgdu_eq_qb: GCCBuiltin<"__builtin_mips_cmpgdu_eq_qb">,
  Intrinsic<[llvm_i32_ty], [llvm_v4i8_ty, llvm_v4i8_ty], [Commutative]>;
def int_mips_cmpgdu_lt_qb: GCCBuiltin<"__builtin_mips_cmpgdu_lt_qb">,
  Intrinsic<[llvm_i32_ty], [llvm_v4i8_ty, llvm_v4i8_ty], [Commutative]>;
def int_mips_cmpgdu_le_qb: GCCBuiltin<"__builtin_mips_cmpgdu_le_qb">,
  Intrinsic<[llvm_i32_ty], [llvm_v4i8_ty, llvm_v4i8_ty], [Commutative]>;

def int_mips_dpa_w_ph: GCCBuiltin<"__builtin_mips_dpa_w_ph">,
  Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_v2i16_ty, llvm_v2i16_ty],
            [IntrNoMem]>;
def int_mips_dps_w_ph: GCCBuiltin<"__builtin_mips_dps_w_ph">,
  Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_v2i16_ty, llvm_v2i16_ty],
            [IntrNoMem]>;

def int_mips_dpaqx_s_w_ph: GCCBuiltin<"__builtin_mips_dpaqx_s_w_ph">,
  Intrinsic<[llvm_i64_ty], [llvm_i64_ty, mips_v2q15_ty, mips_v2q15_ty], []>;
def int_mips_dpaqx_sa_w_ph: GCCBuiltin<"__builtin_mips_dpaqx_sa_w_ph">,
  Intrinsic<[llvm_i64_ty], [llvm_i64_ty, mips_v2q15_ty, mips_v2q15_ty], []>;
def int_mips_dpax_w_ph: GCCBuiltin<"__builtin_mips_dpax_w_ph">,
  Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_v2i16_ty, llvm_v2i16_ty],
            [IntrNoMem]>;
def int_mips_dpsx_w_ph: GCCBuiltin<"__builtin_mips_dpsx_w_ph">,
  Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_v2i16_ty, llvm_v2i16_ty],
            [IntrNoMem]>;
def int_mips_dpsqx_s_w_ph: GCCBuiltin<"__builtin_mips_dpsqx_s_w_ph">,
  Intrinsic<[llvm_i64_ty], [llvm_i64_ty, mips_v2q15_ty, mips_v2q15_ty], []>;
def int_mips_dpsqx_sa_w_ph: GCCBuiltin<"__builtin_mips_dpsqx_sa_w_ph">,
  Intrinsic<[llvm_i64_ty], [llvm_i64_ty, mips_v2q15_ty, mips_v2q15_ty], []>;

def int_mips_mul_ph: GCCBuiltin<"__builtin_mips_mul_ph">,
  Intrinsic<[llvm_v2i16_ty], [llvm_v2i16_ty, llvm_v2i16_ty], [Commutative]>;
def int_mips_mul_s_ph: GCCBuiltin<"__builtin_mips_mul_s_ph">,
  Intrinsic<[llvm_v2i16_ty], [llvm_v2i16_ty, llvm_v2i16_ty], [Commutative]>;

def int_mips_mulq_rs_w: GCCBuiltin<"__builtin_mips_mulq_rs_w">,
  Intrinsic<[mips_q31_ty], [mips_q31_ty, mips_q31_ty], [Commutative]>;
def int_mips_mulq_s_ph: GCCBuiltin<"__builtin_mips_mulq_s_ph">,
  Intrinsic<[mips_v2q15_ty], [mips_v2q15_ty, mips_v2q15_ty], [Commutative]>;
def int_mips_mulq_s_w: GCCBuiltin<"__builtin_mips_mulq_s_w">,
  Intrinsic<[mips_q31_ty], [mips_q31_ty, mips_q31_ty], [Commutative]>;
def int_mips_mulsa_w_ph: GCCBuiltin<"__builtin_mips_mulsa_w_ph">,
  Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_v2i16_ty, llvm_v2i16_ty],
            [IntrNoMem]>;

def int_mips_precr_qb_ph: GCCBuiltin<"__builtin_mips_precr_qb_ph">,
  Intrinsic<[llvm_v4i8_ty], [llvm_v2i16_ty, llvm_v2i16_ty], []>;
def int_mips_precr_sra_ph_w: GCCBuiltin<"__builtin_mips_precr_sra_ph_w">,
  Intrinsic<[llvm_v2i16_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
            [IntrNoMem]>;
def int_mips_precr_sra_r_ph_w: GCCBuiltin<"__builtin_mips_precr_sra_r_ph_w">,
  Intrinsic<[llvm_v2i16_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
            [IntrNoMem]>;

def int_mips_prepend: GCCBuiltin<"__builtin_mips_prepend">,
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
  [IntrNoMem]>;

def int_mips_shra_qb: GCCBuiltin<"__builtin_mips_shra_qb">,
  Intrinsic<[llvm_v4i8_ty], [llvm_v4i8_ty, llvm_i32_ty], [IntrNoMem]>;
def int_mips_shra_r_qb: GCCBuiltin<"__builtin_mips_shra_r_qb">,
  Intrinsic<[llvm_v4i8_ty], [llvm_v4i8_ty, llvm_i32_ty], [IntrNoMem]>;
def int_mips_shrl_ph: GCCBuiltin<"__builtin_mips_shrl_ph">,
  Intrinsic<[llvm_v2i16_ty], [llvm_v2i16_ty, llvm_i32_ty], [IntrNoMem]>;

def int_mips_subqh_ph: GCCBuiltin<"__builtin_mips_subqh_ph">,
  Intrinsic<[mips_v2q15_ty], [mips_v2q15_ty, mips_v2q15_ty], [IntrNoMem]>;
def int_mips_subqh_r_ph: GCCBuiltin<"__builtin_mips_subqh_r_ph">,
  Intrinsic<[mips_v2q15_ty], [mips_v2q15_ty, mips_v2q15_ty], [IntrNoMem]>;
def int_mips_subqh_w: GCCBuiltin<"__builtin_mips_subqh_w">,
  Intrinsic<[mips_q31_ty], [mips_q31_ty, mips_q31_ty], [IntrNoMem]>;
def int_mips_subqh_r_w: GCCBuiltin<"__builtin_mips_subqh_r_w">,
  Intrinsic<[mips_q31_ty], [mips_q31_ty, mips_q31_ty], [IntrNoMem]>;

def int_mips_subu_ph: GCCBuiltin<"__builtin_mips_subu_ph">,
  Intrinsic<[llvm_v2i16_ty], [llvm_v2i16_ty, llvm_v2i16_ty], []>;
def int_mips_subu_s_ph: GCCBuiltin<"__builtin_mips_subu_s_ph">,
  Intrinsic<[llvm_v2i16_ty], [llvm_v2i16_ty, llvm_v2i16_ty], []>;

def int_mips_subuh_qb: GCCBuiltin<"__builtin_mips_subuh_qb">,
  Intrinsic<[llvm_v4i8_ty], [llvm_v4i8_ty, llvm_v4i8_ty], [IntrNoMem]>;
def int_mips_subuh_r_qb: GCCBuiltin<"__builtin_mips_subuh_r_qb">,
  Intrinsic<[llvm_v4i8_ty], [llvm_v4i8_ty, llvm_v4i8_ty], [IntrNoMem]>;
}
