//===- IntrinsicsDirectX.td - Defines DirectX intrinsics ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the DirectX-specific intrinsics.
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "dx" in {

def int_dx_thread_id : Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem, IntrWillReturn]>;
def int_dx_group_id : Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem, IntrWillReturn]>;
def int_dx_thread_id_in_group : Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem, IntrWillReturn]>;
def int_dx_flattened_thread_id_in_group : Intrinsic<[llvm_i32_ty], [], [IntrNoMem, IntrWillReturn]>;

// Create resource handle given binding information. Returns a `target("dx.")`
// type appropriate for the kind of resource given a register space ID, lower
// bound and range size of the binding, as well as an index and an indicator
// whether that index may be non-uniform.
def int_dx_resource_handlefrombinding
    : DefaultAttrsIntrinsic<
          [llvm_any_ty],
          [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_ptr_ty],
          [IntrNoMem]>;

// Create resource handle with implicit binding in given register space.
// Returns a `target("dx.")` type appropriate for the kind of resource and
// the range size and index of the binding.
def int_dx_resource_handlefromimplicitbinding
    : DefaultAttrsIntrinsic<
          [llvm_any_ty],
          [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_ptr_ty],
          [IntrNoMem]>;

def int_dx_resource_getpointer
    : DefaultAttrsIntrinsic<[llvm_anyptr_ty], [llvm_any_ty, llvm_i32_ty],
                            [IntrNoMem]>;

def int_dx_resource_nonuniformindex
    : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]>;

def int_dx_resource_load_typedbuffer
    : DefaultAttrsIntrinsic<[llvm_any_ty, llvm_i1_ty],
                            [llvm_any_ty, llvm_i32_ty], [IntrReadMem]>;
def int_dx_resource_store_typedbuffer
    : DefaultAttrsIntrinsic<[], [llvm_any_ty, llvm_i32_ty, llvm_any_ty],
                            [IntrWriteMem]>;
def int_dx_resource_load_rawbuffer
    : DefaultAttrsIntrinsic<[llvm_any_ty, llvm_i1_ty],
                            [llvm_any_ty, llvm_i32_ty, llvm_i32_ty],
                            [IntrReadMem]>;
def int_dx_resource_store_rawbuffer
    : DefaultAttrsIntrinsic<
          [], [llvm_any_ty, llvm_i32_ty, llvm_i32_ty, llvm_any_ty],
          [IntrWriteMem]>;

// dx.resource.load.cbufferrow encodes the number of elements returned in the
// function name. The total size of the return should always be 128 bits.
def int_dx_resource_load_cbufferrow_8
    : DefaultAttrsIntrinsic<
          [llvm_any_ty, llvm_any_ty, llvm_any_ty, llvm_any_ty,
           llvm_any_ty, llvm_any_ty, llvm_any_ty, llvm_any_ty],
          [llvm_any_ty, llvm_i32_ty], [IntrReadMem]>;
def int_dx_resource_load_cbufferrow_4
    : DefaultAttrsIntrinsic<
          [llvm_any_ty, llvm_any_ty, llvm_any_ty, llvm_any_ty],
          [llvm_any_ty, llvm_i32_ty], [IntrReadMem]>;
def int_dx_resource_load_cbufferrow_2
    : DefaultAttrsIntrinsic<[llvm_any_ty, llvm_any_ty],
                            [llvm_any_ty, llvm_i32_ty], [IntrReadMem]>;

def int_dx_resource_updatecounter
    : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_any_ty, llvm_i8_ty],
                            [IntrInaccessibleMemOrArgMemOnly]>;

def int_dx_resource_getdimensions_x
    : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_any_ty], [IntrReadMem]>;

// Cast between target extension handle types and dxil-style opaque handles
def int_dx_resource_casthandle : Intrinsic<[llvm_any_ty], [llvm_any_ty]>;

def int_dx_all : DefaultAttrsIntrinsic<[llvm_i1_ty], [llvm_any_ty], [IntrNoMem]>;
def int_dx_any : DefaultAttrsIntrinsic<[llvm_i1_ty], [llvm_any_ty], [IntrNoMem]>;
def int_dx_asdouble : DefaultAttrsIntrinsic<[LLVMScalarOrSameVectorWidth<0, llvm_double_ty>], [llvm_anyint_ty, LLVMMatchType<0>], [IntrNoMem]>;
def int_dx_uclamp : DefaultAttrsIntrinsic<[llvm_anyint_ty], [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>], [IntrNoMem]>;
def int_dx_sclamp : DefaultAttrsIntrinsic<[llvm_anyint_ty], [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>], [IntrNoMem]>;
def int_dx_nclamp : DefaultAttrsIntrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>], [IntrNoMem]>;
def int_dx_cross : DefaultAttrsIntrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>, LLVMMatchType<0>], [IntrNoMem]>;
def int_dx_saturate : DefaultAttrsIntrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem]>;

def int_dx_dot2 : DefaultAttrsIntrinsic<[LLVMMatchType<0>],
                                        [
                                          llvm_anyfloat_ty, LLVMMatchType<0>,
                                          LLVMMatchType<0>, LLVMMatchType<0>
                                        ],
                                        [IntrNoMem, Commutative]>;
def int_dx_dot3 : DefaultAttrsIntrinsic<[LLVMMatchType<0>],
                                        [
                                          llvm_anyfloat_ty, LLVMMatchType<0>,
                                          LLVMMatchType<0>, LLVMMatchType<0>,
                                          LLVMMatchType<0>, LLVMMatchType<0>
                                        ],
                                        [IntrNoMem, Commutative]>;
def int_dx_dot4 : DefaultAttrsIntrinsic<[LLVMMatchType<0>],
                                        [
                                          llvm_anyfloat_ty, LLVMMatchType<0>,
                                          LLVMMatchType<0>, LLVMMatchType<0>,
                                          LLVMMatchType<0>, LLVMMatchType<0>,
                                          LLVMMatchType<0>, LLVMMatchType<0>
                                        ],
                                        [IntrNoMem, Commutative]>;
def int_dx_fdot :
    DefaultAttrsIntrinsic<[LLVMVectorElementType<0>],
    [llvm_anyfloat_ty, LLVMScalarOrSameVectorWidth<0, LLVMVectorElementType<0>>],
    [IntrNoMem, Commutative] >;
def int_dx_sdot :
    DefaultAttrsIntrinsic<[LLVMVectorElementType<0>],
    [llvm_anyint_ty, LLVMScalarOrSameVectorWidth<0, LLVMVectorElementType<0>>],
    [IntrNoMem, Commutative] >;
def int_dx_udot :
    DefaultAttrsIntrinsic<[LLVMVectorElementType<0>],
    [llvm_anyint_ty, LLVMScalarOrSameVectorWidth<0, LLVMVectorElementType<0>>],
    [IntrNoMem, Commutative] >;
def int_dx_dot2add :
    DefaultAttrsIntrinsic<[llvm_float_ty],
    [llvm_float_ty, llvm_half_ty, llvm_half_ty, llvm_half_ty, llvm_half_ty],
    [IntrNoMem, Commutative]>;
def int_dx_dot4add_i8packed : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_dx_dot4add_u8packed : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;

def int_dx_frac  : DefaultAttrsIntrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem]>;
def int_dx_degrees : DefaultAttrsIntrinsic<[LLVMMatchType<0>], [llvm_anyfloat_ty], [IntrNoMem]>;

def int_dx_isinf : DefaultAttrsIntrinsic<[LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>],
    [llvm_anyfloat_ty], [IntrNoMem]>;
def int_dx_isnan : DefaultAttrsIntrinsic<[LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>],
    [llvm_anyfloat_ty], [IntrNoMem]>;

def int_dx_lerp : DefaultAttrsIntrinsic<[LLVMMatchType<0>], [llvm_anyfloat_ty, LLVMMatchType<0>,LLVMMatchType<0>],
    [IntrNoMem]>;

def int_dx_imad : DefaultAttrsIntrinsic<[llvm_anyint_ty], [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>], [IntrNoMem]>;
def int_dx_umad : DefaultAttrsIntrinsic<[llvm_anyint_ty], [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>], [IntrNoMem]>;
def int_dx_normalize : DefaultAttrsIntrinsic<[LLVMMatchType<0>], [llvm_anyfloat_ty], [IntrNoMem]>;
def int_dx_rsqrt  : DefaultAttrsIntrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem]>;
def int_dx_wave_active_countbits : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i1_ty], [IntrConvergent, IntrNoMem]>;
def int_dx_wave_all : DefaultAttrsIntrinsic<[llvm_i1_ty], [llvm_i1_ty], [IntrConvergent, IntrNoMem]>;
def int_dx_wave_any : DefaultAttrsIntrinsic<[llvm_i1_ty], [llvm_i1_ty], [IntrConvergent, IntrNoMem]>;
def int_dx_wave_getlaneindex : DefaultAttrsIntrinsic<[llvm_i32_ty], [], [IntrConvergent, IntrNoMem]>;
def int_dx_wave_reduce_max : DefaultAttrsIntrinsic<[llvm_any_ty], [LLVMMatchType<0>], [IntrConvergent, IntrNoMem]>;
def int_dx_wave_reduce_umax : DefaultAttrsIntrinsic<[llvm_anyint_ty], [LLVMMatchType<0>], [IntrConvergent, IntrNoMem]>;
def int_dx_wave_reduce_sum : DefaultAttrsIntrinsic<[llvm_any_ty], [LLVMMatchType<0>], [IntrConvergent, IntrNoMem]>;
def int_dx_wave_reduce_usum : DefaultAttrsIntrinsic<[llvm_anyint_ty], [LLVMMatchType<0>], [IntrConvergent, IntrNoMem]>;
def int_dx_wave_is_first_lane : DefaultAttrsIntrinsic<[llvm_i1_ty], [], [IntrConvergent]>;
def int_dx_wave_readlane : DefaultAttrsIntrinsic<[llvm_any_ty], [LLVMMatchType<0>, llvm_i32_ty], [IntrConvergent, IntrNoMem]>;
def int_dx_wave_get_lane_count
    : DefaultAttrsIntrinsic<[llvm_i32_ty], [], [IntrConvergent]>;
def int_dx_sign : DefaultAttrsIntrinsic<[LLVMScalarOrSameVectorWidth<0, llvm_i32_ty>], [llvm_any_ty], [IntrNoMem]>;
def int_dx_step : DefaultAttrsIntrinsic<[LLVMMatchType<0>], [llvm_anyfloat_ty, LLVMMatchType<0>], [IntrNoMem]>;
def int_dx_splitdouble : DefaultAttrsIntrinsic<[llvm_anyint_ty, LLVMMatchType<0>],
    [LLVMScalarOrSameVectorWidth<0, llvm_double_ty>], [IntrNoMem]>;
def int_dx_radians : DefaultAttrsIntrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem]>;
def int_dx_discard : DefaultAttrsIntrinsic<[], [llvm_i1_ty], []>;
def int_dx_firstbituhigh : DefaultAttrsIntrinsic<[LLVMScalarOrSameVectorWidth<0, llvm_i32_ty>], [llvm_anyint_ty], [IntrNoMem]>;
def int_dx_firstbitshigh : DefaultAttrsIntrinsic<[LLVMScalarOrSameVectorWidth<0, llvm_i32_ty>], [llvm_anyint_ty], [IntrNoMem]>;
def int_dx_firstbitlow : DefaultAttrsIntrinsic<[LLVMScalarOrSameVectorWidth<0, llvm_i32_ty>], [llvm_anyint_ty], [IntrNoMem]>;

def int_dx_group_memory_barrier_with_group_sync
    : DefaultAttrsIntrinsic<[], [], [IntrConvergent]>;
}
