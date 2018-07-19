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
/// \brief This file defines all of the WebAssembly-specific intrinsics.
///
//===----------------------------------------------------------------------===//

let TargetPrefix = "wasm" in {  // All intrinsics start with "llvm.wasm.".

// Note that current_memory is not IntrNoMem because it must be sequenced with
// respect to grow_memory calls.
def int_wasm_current_memory : Intrinsic<[llvm_anyint_ty], [], [IntrReadMem]>;
def int_wasm_grow_memory : Intrinsic<[llvm_anyint_ty], [LLVMMatchType<0>], []>;

// Exception handling intrinsics
def int_wasm_throw: Intrinsic<[], [llvm_i32_ty, llvm_ptr_ty], [Throws]>;
def int_wasm_rethrow: Intrinsic<[], [], [Throws]>;

}
