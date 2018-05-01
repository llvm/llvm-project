//===- IntrinsicsWebAssembly.td - Defines wasm intrinsics --*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines all of the WebAssembly-specific intrinsics.
///
//===----------------------------------------------------------------------===//

let TargetPrefix = "wasm" in {  // All intrinsics start with "llvm.wasm.".

// Query the current memory size, and increase the current memory size.
// Note that mem.size is not IntrNoMem because it must be sequenced with
// respect to mem.grow calls.
// These are the new proposed names, which aren't yet official. Use at your own
// risk.
def int_wasm_mem_size : Intrinsic<[llvm_anyint_ty],
                                  [llvm_i32_ty],
                                  [IntrReadMem]>;
def int_wasm_mem_grow : Intrinsic<[llvm_anyint_ty],
                                  [llvm_i32_ty, LLVMMatchType<0>],
                                  []>;

// These are the existing names, which are currently official, but expected
// to be deprecated in the future. They also lack the immediate field.
def int_wasm_current_memory : Intrinsic<[llvm_anyint_ty], [], [IntrReadMem]>;
def int_wasm_grow_memory : Intrinsic<[llvm_anyint_ty], [LLVMMatchType<0>], []>;

//===----------------------------------------------------------------------===//
// Exception handling intrinsics
//===----------------------------------------------------------------------===//

// throw / rethrow
def int_wasm_throw : Intrinsic<[], [llvm_i32_ty, llvm_ptr_ty],
                               [Throws, IntrNoReturn]>;
def int_wasm_rethrow : Intrinsic<[], [], [Throws, IntrNoReturn]>;

// Since wasm does not use landingpad instructions, these instructions return
// exception pointer and selector values until we lower them in WasmEHPrepare.
def int_wasm_get_exception : Intrinsic<[llvm_ptr_ty], [], [IntrHasSideEffects]>;
def int_wasm_get_ehselector : Intrinsic<[llvm_i32_ty], [],
                                        [IntrHasSideEffects]>;
}
