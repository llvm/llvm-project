//===- IntrinsicsDPU.td - Defines DPU intrinsics ---------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the DPU-specific intrinsics.
//
//===----------------------------------------------------------------------===//

def llvm_mram_ptr_ty : LLVMQualPointerType<llvm_i8_ty, 255>;
def llvm_wram_ptr_ty : LLVMQualPointerType<llvm_i8_ty, 0>;

let TargetPrefix = "dpu" in {

def int_dpu_tid : Intrinsic<[llvm_anyint_ty], [], [IntrNoMem]>;
def int_dpu_sdma : Intrinsic<[], [llvm_wram_ptr_ty, llvm_mram_ptr_ty, llvm_i32_ty], [IntrHasSideEffects, ReadOnly<0>, WriteOnly<1>, Throws]>;
def int_dpu_ldma : Intrinsic<[], [llvm_wram_ptr_ty, llvm_mram_ptr_ty, llvm_i32_ty], [IntrHasSideEffects, ReadOnly<1>, WriteOnly<0>, Throws]>;
def int_dpu_sdma_unchecked : Intrinsic<[], [llvm_wram_ptr_ty, llvm_wram_ptr_ty, llvm_i32_ty], [IntrHasSideEffects, ReadOnly<0>, WriteOnly<1>, Throws]>;
def int_dpu_ldma_unchecked : Intrinsic<[], [llvm_wram_ptr_ty, llvm_wram_ptr_ty, llvm_i32_ty], [IntrHasSideEffects, ReadOnly<1>, WriteOnly<0>, Throws]>;

def int_dpu_seqread_get : Intrinsic<[llvm_anyint_ty], [llvm_i32_ty, llvm_i32_ty, llvm_ptr_ty, llvm_i32_ty], [IntrHasSideEffects]>;

}
