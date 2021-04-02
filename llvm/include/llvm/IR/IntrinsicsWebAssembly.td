//===- IntrinsicsWebAssembly.td - Defines wasm intrinsics --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines all of the WebAssembly-specific intrinsics.
///
//===----------------------------------------------------------------------===//

let TargetPrefix = "wasm" in {  // All intrinsics start with "llvm.wasm.".

// Query the current memory size, and increase the current memory size.
// Note that memory.size is not IntrNoMem because it must be sequenced with
// respect to memory.grow calls.
def int_wasm_memory_size : Intrinsic<[llvm_anyint_ty],
                                     [llvm_i32_ty],
                                     [IntrReadMem]>;
def int_wasm_memory_grow : Intrinsic<[llvm_anyint_ty],
                                     [llvm_i32_ty, LLVMMatchType<0>],
                                     []>;

//===----------------------------------------------------------------------===//
// Trapping float-to-int conversions
//===----------------------------------------------------------------------===//

def int_wasm_trunc_signed : Intrinsic<[llvm_anyint_ty],
                                      [llvm_anyfloat_ty],
                                      [IntrNoMem]>;
def int_wasm_trunc_unsigned : Intrinsic<[llvm_anyint_ty],
                                        [llvm_anyfloat_ty],
                                        [IntrNoMem]>;

//===----------------------------------------------------------------------===//
// Saturating float-to-int conversions
//===----------------------------------------------------------------------===//

def int_wasm_trunc_saturate_signed : Intrinsic<[llvm_anyint_ty],
                                               [llvm_anyfloat_ty],
                                               [IntrNoMem, IntrSpeculatable]>;
def int_wasm_trunc_saturate_unsigned : Intrinsic<[llvm_anyint_ty],
                                                 [llvm_anyfloat_ty],
                                                 [IntrNoMem, IntrSpeculatable]>;

//===----------------------------------------------------------------------===//
// Exception handling intrinsics
//===----------------------------------------------------------------------===//

// throw / rethrow
// The immediate argument is an index to a tag, which is 0 for C++.
def int_wasm_throw : Intrinsic<[], [llvm_i32_ty, llvm_ptr_ty],
                               [Throws, IntrNoReturn, ImmArg<ArgIndex<0>>]>;
def int_wasm_rethrow : Intrinsic<[], [], [Throws, IntrNoReturn]>;

// Since wasm does not use landingpad instructions, these instructions return
// exception pointer and selector values until we lower them in WasmEHPrepare.
def int_wasm_get_exception : Intrinsic<[llvm_ptr_ty], [llvm_token_ty],
                                       [IntrHasSideEffects]>;
def int_wasm_get_ehselector : Intrinsic<[llvm_i32_ty], [llvm_token_ty],
                                        [IntrHasSideEffects]>;

// wasm.catch returns the pointer to the exception object caught by wasm 'catch'
// instruction. This returns a single pointer, which is sufficient for C++
// support. The immediate argument is an index to for a tag, which is 0 for C++.
def int_wasm_catch : Intrinsic<[llvm_ptr_ty], [llvm_i32_ty],
                               [IntrHasSideEffects, ImmArg<ArgIndex<0>>]>;

// WebAssembly EH must maintain the landingpads in the order assigned to them
// by WasmEHPrepare pass to generate landingpad table in EHStreamer. This is
// used in order to give them the indices in WasmEHPrepare.
def int_wasm_landingpad_index: Intrinsic<[], [llvm_token_ty, llvm_i32_ty],
                                         [IntrNoMem, ImmArg<ArgIndex<1>>]>;

// Returns LSDA address of the current function.
def int_wasm_lsda : Intrinsic<[llvm_ptr_ty], [], [IntrNoMem]>;

//===----------------------------------------------------------------------===//
// Atomic intrinsics
//===----------------------------------------------------------------------===//

// wait / notify
def int_wasm_memory_atomic_wait32 :
  Intrinsic<[llvm_i32_ty],
            [LLVMPointerType<llvm_i32_ty>, llvm_i32_ty, llvm_i64_ty],
            [IntrInaccessibleMemOrArgMemOnly, ReadOnly<ArgIndex<0>>,
             NoCapture<ArgIndex<0>>, IntrHasSideEffects],
            "", [SDNPMemOperand]>;
def int_wasm_memory_atomic_wait64 :
  Intrinsic<[llvm_i32_ty],
            [LLVMPointerType<llvm_i64_ty>, llvm_i64_ty, llvm_i64_ty],
            [IntrInaccessibleMemOrArgMemOnly, ReadOnly<ArgIndex<0>>,
             NoCapture<ArgIndex<0>>, IntrHasSideEffects],
            "", [SDNPMemOperand]>;
def int_wasm_memory_atomic_notify:
  Intrinsic<[llvm_i32_ty], [LLVMPointerType<llvm_i32_ty>, llvm_i32_ty],
            [IntrInaccessibleMemOnly, NoCapture<ArgIndex<0>>,
             IntrHasSideEffects],
            "", [SDNPMemOperand]>;

//===----------------------------------------------------------------------===//
// SIMD intrinsics
//===----------------------------------------------------------------------===//

def int_wasm_swizzle :
  Intrinsic<[llvm_v16i8_ty],
            [llvm_v16i8_ty, llvm_v16i8_ty],
            [IntrNoMem, IntrSpeculatable]>;
def int_wasm_shuffle :
  Intrinsic<[llvm_v16i8_ty],
            [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty, llvm_i32_ty,
             llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
             llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
             llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
            [IntrNoMem, IntrSpeculatable]>;
def int_wasm_sub_sat_signed :
  Intrinsic<[llvm_anyvector_ty],
            [LLVMMatchType<0>, LLVMMatchType<0>],
            [IntrNoMem, IntrSpeculatable]>;
def int_wasm_sub_sat_unsigned :
  Intrinsic<[llvm_anyvector_ty],
            [LLVMMatchType<0>, LLVMMatchType<0>],
            [IntrNoMem, IntrSpeculatable]>;
def int_wasm_avgr_unsigned :
  Intrinsic<[llvm_anyvector_ty],
            [LLVMMatchType<0>, LLVMMatchType<0>],
            [IntrNoMem, IntrSpeculatable]>;
def int_wasm_bitselect :
  Intrinsic<[llvm_anyvector_ty],
            [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>],
            [IntrNoMem, IntrSpeculatable]>;
def int_wasm_anytrue :
  Intrinsic<[llvm_i32_ty],
            [llvm_anyvector_ty],
            [IntrNoMem, IntrSpeculatable]>;
def int_wasm_alltrue :
  Intrinsic<[llvm_i32_ty],
            [llvm_anyvector_ty],
            [IntrNoMem, IntrSpeculatable]>;
def int_wasm_bitmask :
  Intrinsic<[llvm_i32_ty],
            [llvm_anyvector_ty],
            [IntrNoMem, IntrSpeculatable]>;
def int_wasm_dot :
  Intrinsic<[llvm_v4i32_ty],
            [llvm_v8i16_ty, llvm_v8i16_ty],
            [IntrNoMem, IntrSpeculatable]>;

def int_wasm_narrow_signed :
  Intrinsic<[llvm_anyvector_ty],
            [llvm_anyvector_ty, LLVMMatchType<1>],
            [IntrNoMem, IntrSpeculatable]>;
def int_wasm_narrow_unsigned :
  Intrinsic<[llvm_anyvector_ty],
            [llvm_anyvector_ty, LLVMMatchType<1>],
            [IntrNoMem, IntrSpeculatable]>;

// TODO: Replace these intrinsics with normal ISel patterns once i32x4 to i64x2
// extending is merged to the proposal.
def int_wasm_extend_low_signed :
  Intrinsic<[llvm_v2i64_ty], [llvm_v4i32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_wasm_extend_high_signed :
  Intrinsic<[llvm_v2i64_ty], [llvm_v4i32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_wasm_extend_low_unsigned :
  Intrinsic<[llvm_v2i64_ty], [llvm_v4i32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_wasm_extend_high_unsigned :
  Intrinsic<[llvm_v2i64_ty], [llvm_v4i32_ty], [IntrNoMem, IntrSpeculatable]>;

def int_wasm_q15mulr_sat_signed :
  Intrinsic<[llvm_v8i16_ty],
            [llvm_v8i16_ty, llvm_v8i16_ty],
            [IntrNoMem, IntrSpeculatable]>;

// TODO: Replace these intrinsics with normal ISel patterns
def int_wasm_pmin :
  Intrinsic<[llvm_anyvector_ty],
            [LLVMMatchType<0>, LLVMMatchType<0>],
            [IntrNoMem, IntrSpeculatable]>;
def int_wasm_pmax :
  Intrinsic<[llvm_anyvector_ty],
            [LLVMMatchType<0>, LLVMMatchType<0>],
            [IntrNoMem, IntrSpeculatable]>;

// TODO: Replace these instrinsics with normal ISel patterns once the
// rounding instructions are merged to the proposal
// (https://github.com/WebAssembly/simd/pull/232).
def int_wasm_ceil :
  Intrinsic<[llvm_anyvector_ty],
            [LLVMMatchType<0>],
            [IntrNoMem, IntrSpeculatable]>;
def int_wasm_floor :
  Intrinsic<[llvm_anyvector_ty],
            [LLVMMatchType<0>],
            [IntrNoMem, IntrSpeculatable]>;
def int_wasm_trunc :
  Intrinsic<[llvm_anyvector_ty],
            [LLVMMatchType<0>],
            [IntrNoMem, IntrSpeculatable]>;
def int_wasm_nearest :
  Intrinsic<[llvm_anyvector_ty],
            [LLVMMatchType<0>],
            [IntrNoMem, IntrSpeculatable]>;

// TODO: Replace these intrinsic with normal ISel patterns once the
// load_zero instructions are merged to the proposal.
def int_wasm_load32_zero :
  Intrinsic<[llvm_v4i32_ty],
            [LLVMPointerType<llvm_i32_ty>],
            [IntrReadMem, IntrArgMemOnly],
             "", [SDNPMemOperand]>;

def int_wasm_load64_zero :
  Intrinsic<[llvm_v2i64_ty],
            [LLVMPointerType<llvm_i64_ty>],
            [IntrReadMem, IntrArgMemOnly],
             "", [SDNPMemOperand]>;

// These intrinsics do not mark their lane index arguments as immediate because
// that changes the corresponding SDNode from ISD::Constant to
// ISD::TargetConstant, which would require extra complications in the ISel
// tablegen patterns. TODO: Replace these intrinsic with normal ISel patterns
// once the load_lane instructions are merged to the proposal.
def int_wasm_load8_lane :
  Intrinsic<[llvm_v16i8_ty],
            [LLVMPointerType<llvm_i8_ty>, llvm_v16i8_ty, llvm_i32_ty],
            [IntrReadMem, IntrArgMemOnly],
            "", [SDNPMemOperand]>;
def int_wasm_load16_lane :
  Intrinsic<[llvm_v8i16_ty],
            [LLVMPointerType<llvm_i16_ty>, llvm_v8i16_ty, llvm_i32_ty],
            [IntrReadMem, IntrArgMemOnly],
            "", [SDNPMemOperand]>;
def int_wasm_load32_lane :
  Intrinsic<[llvm_v4i32_ty],
            [LLVMPointerType<llvm_i32_ty>, llvm_v4i32_ty, llvm_i32_ty],
            [IntrReadMem, IntrArgMemOnly],
            "", [SDNPMemOperand]>;
def int_wasm_load64_lane :
  Intrinsic<[llvm_v2i64_ty],
            [LLVMPointerType<llvm_i64_ty>, llvm_v2i64_ty, llvm_i32_ty],
            [IntrReadMem, IntrArgMemOnly],
            "", [SDNPMemOperand]>;
def int_wasm_store8_lane :
  Intrinsic<[],
            [LLVMPointerType<llvm_i8_ty>, llvm_v16i8_ty, llvm_i32_ty],
            [IntrWriteMem, IntrArgMemOnly],
            "", [SDNPMemOperand]>;
def int_wasm_store16_lane :
  Intrinsic<[],
            [LLVMPointerType<llvm_i16_ty>, llvm_v8i16_ty, llvm_i32_ty],
            [IntrWriteMem, IntrArgMemOnly],
            "", [SDNPMemOperand]>;
def int_wasm_store32_lane :
  Intrinsic<[],
            [LLVMPointerType<llvm_i32_ty>, llvm_v4i32_ty, llvm_i32_ty],
            [IntrWriteMem, IntrArgMemOnly],
            "", [SDNPMemOperand]>;
def int_wasm_store64_lane :
  Intrinsic<[],
            [LLVMPointerType<llvm_i64_ty>, llvm_v2i64_ty, llvm_i32_ty],
            [IntrWriteMem, IntrArgMemOnly],
            "", [SDNPMemOperand]>;

// TODO: Replace this intrinsic with normal ISel patterns once popcnt is merged
// to the proposal.
def int_wasm_popcnt :
  Intrinsic<[llvm_v16i8_ty], [llvm_v16i8_ty], [IntrNoMem, IntrSpeculatable]>;

def int_wasm_extmul_low_signed :
  Intrinsic<[llvm_anyvector_ty],
            [LLVMSubdivide2VectorType<0>, LLVMSubdivide2VectorType<0>],
            [IntrNoMem, IntrSpeculatable]>;
def int_wasm_extmul_high_signed :
  Intrinsic<[llvm_anyvector_ty],
            [LLVMSubdivide2VectorType<0>, LLVMSubdivide2VectorType<0>],
            [IntrNoMem, IntrSpeculatable]>;
def int_wasm_extmul_low_unsigned :
  Intrinsic<[llvm_anyvector_ty],
            [LLVMSubdivide2VectorType<0>, LLVMSubdivide2VectorType<0>],
            [IntrNoMem, IntrSpeculatable]>;
def int_wasm_extmul_high_unsigned :
  Intrinsic<[llvm_anyvector_ty],
            [LLVMSubdivide2VectorType<0>, LLVMSubdivide2VectorType<0>],
            [IntrNoMem, IntrSpeculatable]>;

def int_wasm_extadd_pairwise_signed :
  Intrinsic<[llvm_anyvector_ty],
            [LLVMSubdivide2VectorType<0>],
            [IntrNoMem, IntrSpeculatable]>;
def int_wasm_extadd_pairwise_unsigned :
  Intrinsic<[llvm_anyvector_ty],
            [LLVMSubdivide2VectorType<0>],
            [IntrNoMem, IntrSpeculatable]>;

// TODO: Remove these if possible if they are merged to the spec.
def int_wasm_convert_low_signed :
  Intrinsic<[llvm_v2f64_ty], [llvm_v4i32_ty],
            [IntrNoMem, IntrSpeculatable]>;
def int_wasm_convert_low_unsigned :
  Intrinsic<[llvm_v2f64_ty], [llvm_v4i32_ty],
            [IntrNoMem, IntrSpeculatable]>;
def int_wasm_trunc_sat_zero_signed :
  Intrinsic<[llvm_v4i32_ty], [llvm_v2f64_ty],
            [IntrNoMem, IntrSpeculatable]>;
def int_wasm_trunc_sat_zero_unsigned :
  Intrinsic<[llvm_v4i32_ty], [llvm_v2f64_ty],
            [IntrNoMem, IntrSpeculatable]>;
def int_wasm_demote_zero :
  Intrinsic<[llvm_v4f32_ty], [llvm_v2f64_ty],
            [IntrNoMem, IntrSpeculatable]>;
def int_wasm_promote_low :
  Intrinsic<[llvm_v2f64_ty], [llvm_v4f32_ty],
            [IntrNoMem, IntrSpeculatable]>;

//===----------------------------------------------------------------------===//
// Thread-local storage intrinsics
//===----------------------------------------------------------------------===//

def int_wasm_tls_size :
  Intrinsic<[llvm_anyint_ty],
            [],
            [IntrNoMem, IntrSpeculatable]>;

def int_wasm_tls_align :
  Intrinsic<[llvm_anyint_ty],
            [],
            [IntrNoMem, IntrSpeculatable]>;

def int_wasm_tls_base :
  Intrinsic<[llvm_ptr_ty],
            [],
            [IntrReadMem]>;

} // TargetPrefix = "wasm"
