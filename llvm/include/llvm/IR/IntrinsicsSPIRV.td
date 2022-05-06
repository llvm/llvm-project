//===- IntrinsicsSPIRV.td - Defines SPIRV intrinsics -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the SPIRV-specific intrinsics.
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "spv" in {
  def int_spv_assign_type : Intrinsic<[], [llvm_any_ty, llvm_metadata_ty]>;
  def int_spv_assign_name : Intrinsic<[], [llvm_any_ty, llvm_vararg_ty]>;

  def int_spv_track_constant : Intrinsic<[llvm_any_ty], [llvm_any_ty, llvm_metadata_ty]>;
  def int_spv_init_global : Intrinsic<[], [llvm_any_ty, llvm_any_ty]>;
  def int_spv_unref_global : Intrinsic<[], [llvm_any_ty]>;

  def int_spv_gep : Intrinsic<[llvm_anyptr_ty], [llvm_i1_ty, llvm_any_ty, llvm_vararg_ty], [ImmArg<ArgIndex<0>>]>;
  def int_spv_load : Intrinsic<[llvm_i32_ty], [llvm_anyptr_ty, llvm_i16_ty, llvm_i8_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;
  def int_spv_store : Intrinsic<[], [llvm_i32_ty, llvm_anyptr_ty, llvm_i16_ty, llvm_i8_ty], [ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;
  def int_spv_extractv : Intrinsic<[llvm_any_ty], [llvm_i32_ty, llvm_vararg_ty]>;
  def int_spv_insertv : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_any_ty, llvm_vararg_ty]>;
  def int_spv_extractelt : Intrinsic<[llvm_any_ty], [llvm_any_ty, llvm_anyint_ty]>;
  def int_spv_insertelt : Intrinsic<[llvm_any_ty], [llvm_any_ty, llvm_any_ty, llvm_anyint_ty]>;
  def int_spv_const_composite : Intrinsic<[llvm_i32_ty], [llvm_vararg_ty]>;
  def int_spv_bitcast : Intrinsic<[llvm_any_ty], [llvm_any_ty]>;
  def int_spv_switch : Intrinsic<[], [llvm_any_ty, llvm_vararg_ty]>;
}
