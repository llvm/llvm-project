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

def int_dx_create_handle : ClangBuiltin<"__builtin_hlsl_create_handle">,
    Intrinsic<[ llvm_ptr_ty ], [llvm_i8_ty], [IntrWillReturn]>;

// Create resource handle given binding information. Returns a `target("dx.")`
// type appropriate for the kind of resource given a register space ID, lower
// bound and range size of the binding, as well as an index and an indicator
// whether that index may be non-uniform.
def int_dx_handle_fromBinding
    : DefaultAttrsIntrinsic<
          [llvm_any_ty],
          [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i1_ty],
          [IntrNoMem]>;

def int_dx_typedBufferLoad
    : DefaultAttrsIntrinsic<[llvm_any_ty], [llvm_any_ty, llvm_i32_ty]>;
def int_dx_typedBufferLoad_checkbit
    : DefaultAttrsIntrinsic<[llvm_any_ty, llvm_i1_ty],
                            [llvm_any_ty, llvm_i32_ty]>;
def int_dx_typedBufferStore
    : DefaultAttrsIntrinsic<[], [llvm_any_ty, llvm_i32_ty, llvm_anyvector_ty]>;

// Cast between target extension handle types and dxil-style opaque handles
def int_dx_cast_handle : Intrinsic<[llvm_any_ty], [llvm_any_ty]>;

def int_dx_all : DefaultAttrsIntrinsic<[llvm_i1_ty], [llvm_any_ty]>;
def int_dx_any : DefaultAttrsIntrinsic<[llvm_i1_ty], [llvm_any_ty]>;
def int_dx_clamp : DefaultAttrsIntrinsic<[llvm_any_ty], [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>]>;
def int_dx_uclamp : DefaultAttrsIntrinsic<[llvm_anyint_ty], [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>]>;
def int_dx_saturate : DefaultAttrsIntrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>]>;

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

def int_dx_frac  : DefaultAttrsIntrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>]>;

def int_dx_isinf :
    DefaultAttrsIntrinsic<[LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>],
    [llvm_anyfloat_ty]>;

def int_dx_lerp : Intrinsic<[LLVMMatchType<0>], [llvm_anyfloat_ty, LLVMMatchType<0>,LLVMMatchType<0>],
    [IntrNoMem, IntrWillReturn] >;

def int_dx_length : DefaultAttrsIntrinsic<[LLVMVectorElementType<0>], [llvm_anyfloat_ty]>;
def int_dx_imad : DefaultAttrsIntrinsic<[llvm_anyint_ty], [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>]>;
def int_dx_umad : DefaultAttrsIntrinsic<[llvm_anyint_ty], [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>]>;
def int_dx_normalize : DefaultAttrsIntrinsic<[LLVMMatchType<0>], [llvm_anyfloat_ty]>;
def int_dx_rcp  : DefaultAttrsIntrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>]>;
def int_dx_rsqrt  : DefaultAttrsIntrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>]>;

def int_dx_wave_is_first_lane : DefaultAttrsIntrinsic<[llvm_i1_ty], [], [IntrConvergent]>;
def int_dx_sign : DefaultAttrsIntrinsic<[LLVMScalarOrSameVectorWidth<0, llvm_i32_ty>], [llvm_any_ty]>;
}
