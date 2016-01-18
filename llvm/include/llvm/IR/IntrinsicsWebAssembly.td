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

def int_wasm_memory_size : Intrinsic<[llvm_anyint_ty], [], [IntrNoMem]>;
def int_wasm_grow_memory : Intrinsic<[], [llvm_anyint_ty], []>;

}
