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
          [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i1_ty],
          [IntrNoMem]>;

def int_dx_resource_getpointer
    : DefaultAttrsIntrinsic<[llvm_anyptr_ty], [llvm_any_ty, llvm_i32_ty],
                            [IntrNoMem]>;
def int_dx_resource_load_typedbuffer
    : DefaultAttrsIntrinsic<[llvm_any_ty, llvm_i1_ty],
                            [llvm_any_ty, llvm_i32_ty], [IntrReadMem]>;
def int_dx_resource_store_typedbuffer
    : DefaultAttrsIntrinsic<[], [llvm_any_ty, llvm_i32_ty, llvm_anyvector_ty],
                            [IntrWriteMem]>;
def int_dx_resource_load_rawbuffer
    : DefaultAttrsIntrinsic<[llvm_any_ty, llvm_i1_ty],
                            [llvm_any_ty, llvm_i32_ty, llvm_i32_ty],
                            [IntrReadMem]>;

def int_dx_resource_updatecounter
    : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_any_ty, llvm_i8_ty],
                            [IntrInaccessibleMemOrArgMemOnly]>;

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

def int_dx_dot2 :
    DefaultAttrsIntrinsic<[LLVMVectorElementType<0>],
    [llvm_anyfloat_ty, LLVMScalarOrSameVectorWidth<0, LLVMVectorElementType<0>>],
    [IntrNoMem, Commutative] >;
def int_dx_dot3 :
    DefaultAttrsIntrinsic<[LLVMVectorElementType<0>],
    [llvm_anyfloat_ty, LLVMScalarOrSameVectorWidth<0, LLVMVectorElementType<0>>],
    [IntrNoMem, Commutative] >;
def int_dx_dot4 :
    DefaultAttrsIntrinsic<[LLVMVectorElementType<0>],
    [llvm_anyfloat_ty, LLVMScalarOrSameVectorWidth<0, LLVMVectorElementType<0>>],
    [IntrNoMem, Commutative] >;
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
def int_dx_dot4add_i8packed : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_dx_dot4add_u8packed : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;

def int_dx_frac  : DefaultAttrsIntrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem]>;
def int_dx_degrees : DefaultAttrsIntrinsic<[LLVMMatchType<0>], [llvm_anyfloat_ty], [IntrNoMem]>;

def int_dx_isinf : DefaultAttrsIntrinsic<[LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>],
    [llvm_anyfloat_ty], [IntrNoMem]>;

def int_dx_lerp : DefaultAttrsIntrinsic<[LLVMMatchType<0>], [llvm_anyfloat_ty, LLVMMatchType<0>,LLVMMatchType<0>],
    [IntrNoMem]>;

def int_dx_length : DefaultAttrsIntrinsic<[LLVMVectorElementType<0>], [llvm_anyfloat_ty], [IntrNoMem]>;
def int_dx_imad : DefaultAttrsIntrinsic<[llvm_anyint_ty], [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>], [IntrNoMem]>;
def int_dx_umad : DefaultAttrsIntrinsic<[llvm_anyint_ty], [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>], [IntrNoMem]>;
def int_dx_normalize : DefaultAttrsIntrinsic<[LLVMMatchType<0>], [llvm_anyfloat_ty], [IntrNoMem]>;
def int_dx_rsqrt  : DefaultAttrsIntrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem]>;
def int_dx_wave_active_countbits : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i1_ty], [IntrConvergent, IntrNoMem]>;
def int_dx_wave_all : DefaultAttrsIntrinsic<[llvm_i1_ty], [llvm_i1_ty], [IntrConvergent, IntrNoMem]>;
def int_dx_wave_any : DefaultAttrsIntrinsic<[llvm_i1_ty], [llvm_i1_ty], [IntrConvergent, IntrNoMem]>;
def int_dx_wave_getlaneindex : DefaultAttrsIntrinsic<[llvm_i32_ty], [], [IntrConvergent, IntrNoMem]>;
def int_dx_wave_is_first_lane : DefaultAttrsIntrinsic<[llvm_i1_ty], [], [IntrConvergent]>;
def int_dx_wave_readlane : DefaultAttrsIntrinsic<[llvm_any_ty], [LLVMMatchType<0>, llvm_i32_ty], [IntrConvergent, IntrNoMem]>;
def int_dx_sign : DefaultAttrsIntrinsic<[LLVMScalarOrSameVectorWidth<0, llvm_i32_ty>], [llvm_any_ty], [IntrNoMem]>;
def int_dx_step : DefaultAttrsIntrinsic<[LLVMMatchType<0>], [llvm_anyfloat_ty, LLVMMatchType<0>], [IntrNoMem]>;
def int_dx_splitdouble : DefaultAttrsIntrinsic<[llvm_anyint_ty, LLVMMatchType<0>],
    [LLVMScalarOrSameVectorWidth<0, llvm_double_ty>], [IntrNoMem]>;
def int_dx_radians : DefaultAttrsIntrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem]>;
def int_dx_discard : DefaultAttrsIntrinsic<[], [llvm_i1_ty], []>;
def int_dx_firstbituhigh : DefaultAttrsIntrinsic<[LLVMScalarOrSameVectorWidth<0, llvm_i32_ty>], [llvm_anyint_ty], [IntrNoMem]>;
def int_dx_firstbitshigh : DefaultAttrsIntrinsic<[LLVMScalarOrSameVectorWidth<0, llvm_i32_ty>], [llvm_anyint_ty], [IntrNoMem]>;

def int_dx_group_memory_barrier_with_group_sync : DefaultAttrsIntrinsic<[], [], []>;
}
