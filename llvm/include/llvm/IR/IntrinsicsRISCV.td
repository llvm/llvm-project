//===- IntrinsicsRISCV.td - Defines RISCV intrinsics -------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the RISCV-specific intrinsics.
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "riscv" in {

//===----------------------------------------------------------------------===//
// Atomics

class MaskedAtomicRMW32Intrinsic
    : Intrinsic<[llvm_i32_ty],
                [llvm_anyptr_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrArgMemOnly, NoCapture<0>]>;

class MaskedAtomicRMW32WithSextIntrinsic
    : Intrinsic<[llvm_i32_ty],
                [llvm_anyptr_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                 llvm_i32_ty],
                [IntrArgMemOnly, NoCapture<0>]>;

def int_riscv_masked_atomicrmw_xchg_i32 : MaskedAtomicRMW32Intrinsic;
def int_riscv_masked_atomicrmw_add_i32  : MaskedAtomicRMW32Intrinsic;
def int_riscv_masked_atomicrmw_sub_i32  : MaskedAtomicRMW32Intrinsic;
def int_riscv_masked_atomicrmw_nand_i32 : MaskedAtomicRMW32Intrinsic;
def int_riscv_masked_atomicrmw_max_i32  : MaskedAtomicRMW32WithSextIntrinsic;
def int_riscv_masked_atomicrmw_min_i32  : MaskedAtomicRMW32WithSextIntrinsic;
def int_riscv_masked_atomicrmw_umax_i32 : MaskedAtomicRMW32Intrinsic;
def int_riscv_masked_atomicrmw_umin_i32 : MaskedAtomicRMW32Intrinsic;

def int_riscv_masked_cmpxchg_i32
    : Intrinsic<[llvm_i32_ty], [llvm_anyptr_ty, llvm_i32_ty, llvm_i32_ty,
                                llvm_i32_ty, llvm_i32_ty],
                [IntrArgMemOnly, NoCapture<0>]>;

class MaskedAtomicRMW64Intrinsic
    : Intrinsic<[llvm_i64_ty],
                [llvm_anyptr_ty, llvm_i64_ty, llvm_i64_ty, llvm_i64_ty],
                [IntrArgMemOnly, NoCapture<0>]>;

class MaskedAtomicRMW64WithSextIntrinsic
    : Intrinsic<[llvm_i64_ty],
                [llvm_anyptr_ty, llvm_i64_ty, llvm_i64_ty, llvm_i64_ty,
                 llvm_i64_ty],
                [IntrArgMemOnly, NoCapture<0>]>;

def int_riscv_masked_atomicrmw_xchg_i64 : MaskedAtomicRMW64Intrinsic;
def int_riscv_masked_atomicrmw_add_i64  : MaskedAtomicRMW64Intrinsic;
def int_riscv_masked_atomicrmw_sub_i64  : MaskedAtomicRMW64Intrinsic;
def int_riscv_masked_atomicrmw_nand_i64 : MaskedAtomicRMW64Intrinsic;
def int_riscv_masked_atomicrmw_max_i64  : MaskedAtomicRMW64WithSextIntrinsic;
def int_riscv_masked_atomicrmw_min_i64  : MaskedAtomicRMW64WithSextIntrinsic;
def int_riscv_masked_atomicrmw_umax_i64 : MaskedAtomicRMW64Intrinsic;
def int_riscv_masked_atomicrmw_umin_i64 : MaskedAtomicRMW64Intrinsic;

def int_riscv_masked_cmpxchg_i64
    : Intrinsic<[llvm_i64_ty], [llvm_anyptr_ty, llvm_i64_ty, llvm_i64_ty,
                                llvm_i64_ty, llvm_i64_ty],
                [IntrArgMemOnly, NoCapture<0>]>;

} // TargetPrefix = "riscv"
