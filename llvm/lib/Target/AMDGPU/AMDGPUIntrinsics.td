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
  def int_AMDGPU_clamp : Intrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>], [IntrNoMem]>;

  def int_AMDGPU_dp4 : Intrinsic<[llvm_float_ty], [llvm_v4f32_ty, llvm_v4f32_ty], [IntrNoMem]>;
  def int_AMDGPU_kill : Intrinsic<[], [llvm_float_ty], []>;
  def int_AMDGPU_kilp : Intrinsic<[], [], []>;
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
  def int_AMDGPU_flbit_i32 : Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]>;

  def int_AMDGPU_rsq_clamped :  Intrinsic<
    [llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem]
  >;

  // Deprecated in favor of llvm.amdgcn.rsq
  def int_AMDGPU_rsq : Intrinsic<
    [llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem]
  >;

  // Deprecated in favor of llvm.bitreverse
  def int_AMDGPU_brev : Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]>;

  // Deprecated in favor of llvm.amdgcn.s.barrier
  def int_AMDGPU_barrier_local  : Intrinsic<[], [], [IntrConvergent]>;
  def int_AMDGPU_barrier_global : Intrinsic<[], [], [IntrConvergent]>;

  // Deprecated in favor of llvm.amdgcn.read.workdim
  def int_AMDGPU_read_workdim : Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>;
}

// Legacy names for compatibility.
let TargetPrefix = "AMDIL", isTarget = 1 in {
  def int_AMDIL_clamp : Intrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>], [IntrNoMem]>;
  def int_AMDIL_exp : Intrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem]>;
}

include "SIIntrinsics.td"
