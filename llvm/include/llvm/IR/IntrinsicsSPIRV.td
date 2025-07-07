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

def generic_ptr_ty : LLVMQualPointerType<4>;

let TargetPrefix = "spv" in {
  def int_spv_assign_type : Intrinsic<[], [llvm_any_ty, llvm_metadata_ty]>;
  def int_spv_assign_ptr_type : Intrinsic<[], [llvm_any_ty, llvm_metadata_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;
  def int_spv_assign_name : Intrinsic<[], [llvm_any_ty, llvm_metadata_ty]>;
  def int_spv_assign_decoration : Intrinsic<[], [llvm_any_ty, llvm_metadata_ty]>;
  def int_spv_value_md : Intrinsic<[], [llvm_metadata_ty]>;

  def int_spv_track_constant : Intrinsic<[llvm_any_ty], [llvm_any_ty, llvm_metadata_ty]>;
  def int_spv_init_global : Intrinsic<[], [llvm_any_ty, llvm_any_ty]>;
  def int_spv_unref_global : Intrinsic<[], [llvm_any_ty]>;

  def int_spv_gep : Intrinsic<[llvm_anyptr_ty], [llvm_i1_ty, llvm_any_ty, llvm_vararg_ty], [ImmArg<ArgIndex<0>>]>;
  def int_spv_load : Intrinsic<[llvm_i32_ty], [llvm_anyptr_ty, llvm_i16_ty, llvm_i8_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;
  def int_spv_store : Intrinsic<[], [llvm_any_ty, llvm_anyptr_ty, llvm_i16_ty, llvm_i8_ty], [ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;
  def int_spv_extractv : Intrinsic<[llvm_any_ty], [llvm_i32_ty, llvm_vararg_ty]>;
  def int_spv_insertv : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_any_ty, llvm_vararg_ty]>;
  def int_spv_extractelt : Intrinsic<[llvm_any_ty], [llvm_any_ty, llvm_anyint_ty]>;
  def int_spv_insertelt : Intrinsic<[llvm_any_ty], [llvm_any_ty, llvm_any_ty, llvm_anyint_ty]>;
  def int_spv_const_composite : Intrinsic<[llvm_any_ty], [llvm_vararg_ty]>;
  def int_spv_bitcast : Intrinsic<[llvm_any_ty], [llvm_any_ty]>;
  def int_spv_ptrcast : Intrinsic<[llvm_any_ty], [llvm_any_ty, llvm_metadata_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;
  def int_spv_switch : Intrinsic<[], [llvm_any_ty, llvm_vararg_ty]>;
  def int_spv_loop_merge : Intrinsic<[], [llvm_vararg_ty]>;
  def int_spv_selection_merge : Intrinsic<[], [llvm_any_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;
  def int_spv_cmpxchg : Intrinsic<[llvm_i32_ty], [llvm_any_ty, llvm_vararg_ty]>;
  def int_spv_unreachable : Intrinsic<[], []>;
  def int_spv_alloca : Intrinsic<[llvm_any_ty], [llvm_i8_ty], [ImmArg<ArgIndex<0>>]>;
  def int_spv_alloca_array : Intrinsic<[llvm_any_ty], [llvm_anyint_ty, llvm_i8_ty], [ImmArg<ArgIndex<1>>]>;
  def int_spv_undef : Intrinsic<[llvm_i32_ty], []>;
  def int_spv_inline_asm : Intrinsic<[], [llvm_metadata_ty, llvm_metadata_ty, llvm_vararg_ty]>;

  // Expect, Assume Intrinsics
  def int_spv_assume : Intrinsic<[], [llvm_i1_ty]>;
  def int_spv_expect : Intrinsic<[llvm_anyint_ty], [LLVMMatchType<0>, LLVMMatchType<0>]>;

  // Memory Use Markers
  def int_spv_lifetime_start : Intrinsic<[],
                                [llvm_i64_ty, llvm_anyptr_ty],
                                [IntrArgMemOnly, IntrWillReturn,
                                NoCapture<ArgIndex<1>>,
                                ImmArg<ArgIndex<0>>]>;
  def int_spv_lifetime_end   : Intrinsic<[],
                                [llvm_i64_ty, llvm_anyptr_ty],
                                [IntrArgMemOnly, IntrWillReturn,
                                NoCapture<ArgIndex<1>>,
                                ImmArg<ArgIndex<0>>]>;

  // The following intrinsic(s) are mirrored from IntrinsicsDirectX.td for HLSL support.
  def int_spv_thread_id : Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem, IntrWillReturn]>;
  def int_spv_group_id : Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem, IntrWillReturn]>;
  def int_spv_thread_id_in_group : Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem, IntrWillReturn]>;
  def int_spv_flattened_thread_id_in_group : Intrinsic<[llvm_i32_ty], [], [IntrNoMem, IntrWillReturn]>;
  def int_spv_all : DefaultAttrsIntrinsic<[llvm_i1_ty], [llvm_any_ty], [IntrNoMem]>;
  def int_spv_any : DefaultAttrsIntrinsic<[llvm_i1_ty], [llvm_any_ty], [IntrNoMem]>;
  def int_spv_cross : DefaultAttrsIntrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>, LLVMMatchType<0>], [IntrNoMem]>;
  def int_spv_degrees : DefaultAttrsIntrinsic<[LLVMMatchType<0>], [llvm_anyfloat_ty], [IntrNoMem]>;
  def int_spv_distance : DefaultAttrsIntrinsic<[LLVMVectorElementType<0>], [llvm_anyfloat_ty, LLVMMatchType<0>], [IntrNoMem]>;
  def int_spv_faceforward : DefaultAttrsIntrinsic<[LLVMMatchType<0>], [llvm_anyfloat_ty, LLVMMatchType<0>, LLVMMatchType<0>], [IntrNoMem]>;
  def int_spv_frac : DefaultAttrsIntrinsic<[LLVMMatchType<0>], [llvm_anyfloat_ty], [IntrNoMem]>;
  def int_spv_lerp : DefaultAttrsIntrinsic<[LLVMMatchType<0>], [llvm_anyfloat_ty, LLVMMatchType<0>,LLVMMatchType<0>],
    [IntrNoMem] >;
  def int_spv_length : DefaultAttrsIntrinsic<[LLVMVectorElementType<0>], [llvm_anyfloat_ty], [IntrNoMem]>;
  def int_spv_normalize : DefaultAttrsIntrinsic<[LLVMMatchType<0>], [llvm_anyfloat_ty], [IntrNoMem]>;
  def int_spv_reflect : DefaultAttrsIntrinsic<[LLVMMatchType<0>], [llvm_anyfloat_ty, LLVMMatchType<0>], [IntrNoMem]>;
  def int_spv_rsqrt : DefaultAttrsIntrinsic<[LLVMMatchType<0>], [llvm_anyfloat_ty], [IntrNoMem]>;
  def int_spv_saturate : DefaultAttrsIntrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem]>;
  def int_spv_smoothstep : DefaultAttrsIntrinsic<[LLVMMatchType<0>], [llvm_anyfloat_ty, LLVMMatchType<0>, LLVMMatchType<0>], [IntrNoMem]>;
  def int_spv_step : DefaultAttrsIntrinsic<[LLVMMatchType<0>], [LLVMMatchType<0>, llvm_anyfloat_ty], [IntrNoMem]>;
  def int_spv_fdot :
    DefaultAttrsIntrinsic<[LLVMVectorElementType<0>],
    [llvm_anyfloat_ty, LLVMScalarOrSameVectorWidth<0, LLVMVectorElementType<0>>],
    [IntrNoMem, Commutative] >;
  def int_spv_sdot :
    DefaultAttrsIntrinsic<[LLVMVectorElementType<0>],
    [llvm_anyint_ty, LLVMScalarOrSameVectorWidth<0, LLVMVectorElementType<0>>],
    [IntrNoMem, Commutative] >;
  def int_spv_udot :
    DefaultAttrsIntrinsic<[LLVMVectorElementType<0>],
    [llvm_anyint_ty, LLVMScalarOrSameVectorWidth<0, LLVMVectorElementType<0>>],
    [IntrNoMem, Commutative] >;
  def int_spv_dot4add_i8packed : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_spv_dot4add_u8packed : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  def int_spv_wave_active_countbits : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i1_ty], [IntrConvergent, IntrNoMem]>;
  def int_spv_wave_all : DefaultAttrsIntrinsic<[llvm_i1_ty], [llvm_i1_ty], [IntrConvergent, IntrNoMem]>;
  def int_spv_wave_any : DefaultAttrsIntrinsic<[llvm_i1_ty], [llvm_i1_ty], [IntrConvergent, IntrNoMem]>;
  def int_spv_wave_reduce_umax : DefaultAttrsIntrinsic<[llvm_any_ty], [LLVMMatchType<0>], [IntrConvergent, IntrNoMem]>;
  def int_spv_wave_reduce_max : DefaultAttrsIntrinsic<[llvm_any_ty], [LLVMMatchType<0>], [IntrConvergent, IntrNoMem]>;
  def int_spv_wave_reduce_sum : DefaultAttrsIntrinsic<[llvm_any_ty], [LLVMMatchType<0>], [IntrConvergent, IntrNoMem]>;
  def int_spv_wave_is_first_lane : DefaultAttrsIntrinsic<[llvm_i1_ty], [], [IntrConvergent]>;
  def int_spv_wave_readlane : DefaultAttrsIntrinsic<[llvm_any_ty], [LLVMMatchType<0>, llvm_i32_ty], [IntrConvergent, IntrNoMem]>;
  def int_spv_wave_get_lane_count
      : DefaultAttrsIntrinsic<[llvm_i32_ty], [], [IntrConvergent]>;
  def int_spv_sign : DefaultAttrsIntrinsic<[LLVMScalarOrSameVectorWidth<0, llvm_i32_ty>], [llvm_any_ty], [IntrNoMem]>;
  def int_spv_radians : DefaultAttrsIntrinsic<[LLVMMatchType<0>], [llvm_anyfloat_ty], [IntrNoMem]>;
  def int_spv_group_memory_barrier_with_group_sync : DefaultAttrsIntrinsic<[], [], []>;
  def int_spv_discard : DefaultAttrsIntrinsic<[], [], []>;
  def int_spv_uclamp : DefaultAttrsIntrinsic<[llvm_anyint_ty], [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>], [IntrNoMem]>;
  def int_spv_sclamp : DefaultAttrsIntrinsic<[llvm_anyint_ty], [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>], [IntrNoMem]>;
  def int_spv_nclamp : DefaultAttrsIntrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>], [IntrNoMem]>;

  // Create resource handle given the binding information. Returns a 
  // type appropriate for the kind of resource given the set id, binding id,
  // array size of the binding, as well as an index and an indicator
  // whether that index may be non-uniform.
  def int_spv_resource_handlefrombinding
      : DefaultAttrsIntrinsic<[llvm_any_ty],
                              [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                               llvm_i32_ty, llvm_i1_ty, llvm_ptr_ty],
                              [IntrNoMem]>;
  def int_spv_resource_handlefromimplicitbinding
      : DefaultAttrsIntrinsic<[llvm_any_ty],
                              [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                               llvm_i32_ty, llvm_i1_ty, llvm_ptr_ty],
                              [IntrNoMem]>;

  def int_spv_firstbituhigh : DefaultAttrsIntrinsic<[LLVMScalarOrSameVectorWidth<0, llvm_i32_ty>], [llvm_anyint_ty], [IntrNoMem]>;
  def int_spv_firstbitshigh : DefaultAttrsIntrinsic<[LLVMScalarOrSameVectorWidth<0, llvm_i32_ty>], [llvm_anyint_ty], [IntrNoMem]>;
  def int_spv_firstbitlow : DefaultAttrsIntrinsic<[LLVMScalarOrSameVectorWidth<0, llvm_i32_ty>], [llvm_anyint_ty], [IntrNoMem]>;

  def int_spv_resource_updatecounter
      : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_any_ty, llvm_i8_ty],
                              [IntrInaccessibleMemOrArgMemOnly]>;

  def int_spv_resource_getpointer
      : DefaultAttrsIntrinsic<[llvm_anyptr_ty], [llvm_any_ty, llvm_i32_ty],
                              [IntrNoMem]>;

  // Read a value from the image buffer. It does not translate directly to a
  // single OpImageRead because the result type is not necessarily a 4 element
  // vector.
  def int_spv_resource_load_typedbuffer
      : DefaultAttrsIntrinsic<[llvm_any_ty], [llvm_any_ty, llvm_i32_ty]>;

  // Write a value to the image buffer. Translates directly to a single
  // OpImageWrite.
  def int_spv_resource_store_typedbuffer
    : DefaultAttrsIntrinsic<[], [llvm_any_ty, llvm_i32_ty, llvm_anyvector_ty]>;

  // Memory aliasing intrinsics
  def int_spv_assign_aliasing_decoration : Intrinsic<[], [llvm_any_ty, llvm_i32_ty, llvm_metadata_ty], [ImmArg<ArgIndex<1>>]>;

  // FPMaxErrorDecorationINTEL
  def int_spv_assign_fpmaxerror_decoration: Intrinsic<[], [llvm_any_ty, llvm_metadata_ty]>;

  // Convert between the generic storage class and a concrete one.
  def int_spv_generic_cast_to_ptr_explicit
    : DefaultAttrsIntrinsic<[llvm_anyptr_ty], [generic_ptr_ty],
       [IntrNoMem, NoUndef<RetIndex>]>;
}
