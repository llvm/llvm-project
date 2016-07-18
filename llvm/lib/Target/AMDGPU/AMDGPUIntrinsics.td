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

  def int_AMDGPU_kill : Intrinsic<[], [llvm_float_ty], []>;
  def int_AMDGPU_kilp : Intrinsic<[], [], []>;
  def int_AMDGPU_flbit_i32 : Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]>;

  // Deprecated in favor of separate int_amdgcn_cube* intrinsics.
  def int_AMDGPU_cube : Intrinsic<[llvm_v4f32_ty], [llvm_v4f32_ty], [IntrNoMem]>;

  // Deprecated in favor of expanded bit operations
  def int_AMDGPU_bfe_i32 : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_AMDGPU_bfe_u32 : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;

  // Deprecated in favor of llvm.amdgcn.rsq
  def int_AMDGPU_rsq : Intrinsic<
    [llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem]
  >;

  // Deprecated in favor of llvm.amdgcn.read.workdim
  def int_AMDGPU_read_workdim : Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>;
}

include "SIIntrinsics.td"
