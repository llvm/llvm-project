//===-- AMDGPUIntrinsics.td - Common intrinsics  -*- tablegen -*-----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines intrinsics that are used by all hw codegen targets.
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "AMDGPU", isTarget = 1 in {

  def int_AMDGPU_store_output : Intrinsic<[], [llvm_float_ty, llvm_i32_ty], []>;
  def int_AMDGPU_swizzle : Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_abs : Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_arl : Intrinsic<[llvm_i32_ty], [llvm_float_ty], [IntrNoMem]>;
  def int_AMDGPU_cndlt : Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty, llvm_float_ty], [IntrNoMem]>;
  def int_AMDGPU_div : Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty], [IntrNoMem]>;
  def int_AMDGPU_fract : Intrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem]>;
  def int_AMDGPU_clamp : Intrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>], [IntrNoMem]>;

  // This is named backwards (instead of rsq_legacy) so we don't have
  // to define it with the public builtins intrinsics. This is a
  // workaround for how intrinsic names are parsed. If the name is
  // llvm.AMDGPU.rsq.legacy, the parser assumes that you meant
  // llvm.AMDGPU.rsq.{f32 | f64} and incorrectly mangled the name.
  def int_AMDGPU_legacy_rsq : Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;

  def int_AMDGPU_dp4 : Intrinsic<[llvm_float_ty], [llvm_v4f32_ty, llvm_v4f32_ty], [IntrNoMem]>;
  def int_AMDGPU_kill : Intrinsic<[], [llvm_float_ty], []>;
  def int_AMDGPU_kilp : Intrinsic<[], [], []>;
  def int_AMDGPU_lrp : Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty, llvm_float_ty], [IntrNoMem]>;
  def int_AMDGPU_mul : Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty], [IntrNoMem]>;
  def int_AMDGPU_pow : Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty], [IntrNoMem]>;
  def int_AMDGPU_seq : Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty], [IntrNoMem]>;
  def int_AMDGPU_sgt : Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty], [IntrNoMem]>;
  def int_AMDGPU_sge : Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty], [IntrNoMem]>;
  def int_AMDGPU_sle : Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty], [IntrNoMem]>;
  def int_AMDGPU_sne : Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty], [IntrNoMem]>;
  def int_AMDGPU_mullit : Intrinsic<[llvm_v4f32_ty], [llvm_float_ty, llvm_float_ty, llvm_float_ty], [IntrNoMem]>;
  def int_AMDGPU_tex : Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_txb : Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_txf : Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_txq : Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_txd : Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v4f32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_txl : Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_ddx : Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_ddy : Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_umul24 : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_imul24 : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_imad24 : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_umad24 : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_cvt_f32_ubyte0 : Intrinsic<[llvm_float_ty], [llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_cvt_f32_ubyte1 : Intrinsic<[llvm_float_ty], [llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_cvt_f32_ubyte2 : Intrinsic<[llvm_float_ty], [llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_cvt_f32_ubyte3 : Intrinsic<[llvm_float_ty], [llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_cube : Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty], [IntrNoMem]>;
  def int_AMDGPU_bfi : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_bfe_i32 : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_bfe_u32 : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_bfm : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_brev : Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_flbit_i32 : Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_barrier_local  : Intrinsic<[], [], [IntrConvergent]>;
  def int_AMDGPU_barrier_global  : Intrinsic<[], [], [IntrConvergent]>;
}

// Legacy names for compatibility.
let TargetPrefix = "AMDIL", isTarget = 1 in {
  def int_AMDIL_clamp : Intrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>], [IntrNoMem]>;
  def int_AMDIL_exp : Intrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem]>;
}

let TargetPrefix = "TGSI", isTarget = 1 in {

  def int_TGSI_lit_z : Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty, llvm_float_ty],[IntrNoMem]>;
}

include "SIIntrinsics.td"
