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
def int_wasm_throw : Intrinsic<[], [llvm_i32_ty, llvm_ptr_ty],
                               [Throws, IntrNoReturn, ImmArg<0>]>;
def int_wasm_rethrow_in_catch : Intrinsic<[], [], [Throws, IntrNoReturn]>;

// Since wasm does not use landingpad instructions, these instructions return
// exception pointer and selector values until we lower them in WasmEHPrepare.
def int_wasm_get_exception : Intrinsic<[llvm_ptr_ty], [llvm_token_ty],
                                       [IntrHasSideEffects]>;
def int_wasm_get_ehselector : Intrinsic<[llvm_i32_ty], [llvm_token_ty],
                                        [IntrHasSideEffects]>;
// This is the same as llvm.wasm.get.exception except that it does not take a
// token operand. This is only for instruction selection purpose.
def int_wasm_extract_exception : Intrinsic<[llvm_ptr_ty], [],
                                           [IntrHasSideEffects]>;

// WebAssembly EH must maintain the landingpads in the order assigned to them
// by WasmEHPrepare pass to generate landingpad table in EHStreamer. This is
// used in order to give them the indices in WasmEHPrepare.
def int_wasm_landingpad_index: Intrinsic<[], [llvm_token_ty, llvm_i32_ty],
                                         [IntrNoMem, ImmArg<1>]>;

// Returns LSDA address of the current function.
def int_wasm_lsda : Intrinsic<[llvm_ptr_ty], [], [IntrNoMem]>;

//===----------------------------------------------------------------------===//
// Atomic intrinsics
//===----------------------------------------------------------------------===//

// wait / notify
def int_wasm_atomic_wait_i32 :
  Intrinsic<[llvm_i32_ty],
            [LLVMPointerType<llvm_i32_ty>, llvm_i32_ty, llvm_i64_ty],
            [IntrInaccessibleMemOrArgMemOnly, ReadOnly<0>, NoCapture<0>,
             IntrHasSideEffects],
             "", [SDNPMemOperand]>;
def int_wasm_atomic_wait_i64 :
  Intrinsic<[llvm_i32_ty],
            [LLVMPointerType<llvm_i64_ty>, llvm_i64_ty, llvm_i64_ty],
            [IntrInaccessibleMemOrArgMemOnly, ReadOnly<0>, NoCapture<0>,
             IntrHasSideEffects],
             "", [SDNPMemOperand]>;
def int_wasm_atomic_notify:
  Intrinsic<[llvm_i32_ty], [LLVMPointerType<llvm_i32_ty>, llvm_i32_ty],
            [IntrInaccessibleMemOnly, NoCapture<0>, IntrHasSideEffects], "",
            [SDNPMemOperand]>;

//===----------------------------------------------------------------------===//
// SIMD intrinsics
//===----------------------------------------------------------------------===//

def int_wasm_swizzle :
  Intrinsic<[llvm_v16i8_ty],
            [llvm_v16i8_ty, llvm_v16i8_ty],
            [IntrNoMem, IntrSpeculatable]>;
def int_wasm_sub_saturate_signed :
  Intrinsic<[llvm_anyvector_ty],
            [LLVMMatchType<0>, LLVMMatchType<0>],
            [IntrNoMem, IntrSpeculatable]>;
def int_wasm_sub_saturate_unsigned :
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
def int_wasm_qfma :
  Intrinsic<[llvm_anyvector_ty],
            [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>],
            [IntrNoMem, IntrSpeculatable]>;
def int_wasm_qfms :
  Intrinsic<[llvm_anyvector_ty],
            [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>],
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
def int_wasm_widen_low_signed :
  Intrinsic<[llvm_anyvector_ty],
            [llvm_anyvector_ty],
            [IntrNoMem, IntrSpeculatable]>;
def int_wasm_widen_high_signed :
  Intrinsic<[llvm_anyvector_ty],
            [llvm_anyvector_ty],
            [IntrNoMem, IntrSpeculatable]>;
def int_wasm_widen_low_unsigned :
  Intrinsic<[llvm_anyvector_ty],
            [llvm_anyvector_ty],
            [IntrNoMem, IntrSpeculatable]>;
def int_wasm_widen_high_unsigned :
  Intrinsic<[llvm_anyvector_ty],
            [llvm_anyvector_ty],
            [IntrNoMem, IntrSpeculatable]>;


//===----------------------------------------------------------------------===//
// Bulk memory intrinsics
//===----------------------------------------------------------------------===//

def int_wasm_memory_init :
  Intrinsic<[],
            [llvm_i32_ty, llvm_i32_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
            [IntrWriteMem, IntrInaccessibleMemOrArgMemOnly, WriteOnly<2>,
             IntrHasSideEffects, ImmArg<0>, ImmArg<1>]>;
def int_wasm_data_drop :
  Intrinsic<[],
            [llvm_i32_ty],
            [IntrNoDuplicate, IntrHasSideEffects, ImmArg<0>]>;

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
