//===- IntrinsicsLoongArch.td - Defines LoongArch intrinsics *- tablegen -*===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the LoongArch-specific intrinsics.
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "loongarch" in {

//===----------------------------------------------------------------------===//
// Atomics

// T @llvm.<name>.T.<p>(any*, T, T, T imm);
class MaskedAtomicRMW<LLVMType itype>
    : Intrinsic<[itype], [llvm_anyptr_ty, itype, itype, itype],
                [IntrArgMemOnly, NoCapture<ArgIndex<0>>, ImmArg<ArgIndex<3>>]>;

// We define 32-bit and 64-bit variants of the above, where T stands for i32
// or i64 respectively:
multiclass MaskedAtomicRMWIntrinsics {
  // i32 @llvm.<name>.i32.<p>(any*, i32, i32, i32 imm);
  def _i32 : MaskedAtomicRMW<llvm_i32_ty>;
  // i64 @llvm.<name>.i32.<p>(any*, i64, i64, i64 imm);
  def _i64 : MaskedAtomicRMW<llvm_i64_ty>;
}

multiclass MaskedAtomicRMWFiveOpIntrinsics {
  // TODO: Support cmpxchg on LA32.
  // i64 @llvm.<name>.i64.<p>(any*, i64, i64, i64, i64 imm);
  def _i64 : MaskedAtomicRMWFiveArg<llvm_i64_ty>;
}

defm int_loongarch_masked_atomicrmw_xchg : MaskedAtomicRMWIntrinsics;
defm int_loongarch_masked_atomicrmw_add : MaskedAtomicRMWIntrinsics;
defm int_loongarch_masked_atomicrmw_sub : MaskedAtomicRMWIntrinsics;
defm int_loongarch_masked_atomicrmw_nand : MaskedAtomicRMWIntrinsics;
defm int_loongarch_masked_atomicrmw_umax : MaskedAtomicRMWIntrinsics;
defm int_loongarch_masked_atomicrmw_umin : MaskedAtomicRMWIntrinsics;
defm int_loongarch_masked_atomicrmw_max : MaskedAtomicRMWFiveOpIntrinsics;
defm int_loongarch_masked_atomicrmw_min : MaskedAtomicRMWFiveOpIntrinsics;

// @llvm.loongarch.masked.cmpxchg.i64.<p>(
//   ptr addr, grlen cmpval, grlen newval, grlen mask, grlenimm ordering)
defm int_loongarch_masked_cmpxchg : MaskedAtomicRMWFiveOpIntrinsics;

//===----------------------------------------------------------------------===//
// LoongArch BASE

def int_loongarch_break : Intrinsic<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;
def int_loongarch_dbar : Intrinsic<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;
def int_loongarch_ibar : Intrinsic<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;
def int_loongarch_syscall : Intrinsic<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_loongarch_crc_w_b_w : Intrinsic<[llvm_i32_ty],
                                        [llvm_i32_ty, llvm_i32_ty]>;
def int_loongarch_crc_w_h_w : Intrinsic<[llvm_i32_ty],
                                        [llvm_i32_ty, llvm_i32_ty]>;
def int_loongarch_crc_w_w_w : Intrinsic<[llvm_i32_ty],
                                        [llvm_i32_ty, llvm_i32_ty]>;
def int_loongarch_crc_w_d_w : Intrinsic<[llvm_i32_ty],
                                        [llvm_i64_ty, llvm_i32_ty]>;

def int_loongarch_crcc_w_b_w : Intrinsic<[llvm_i32_ty],
                                         [llvm_i32_ty, llvm_i32_ty]>;
def int_loongarch_crcc_w_h_w : Intrinsic<[llvm_i32_ty],
                                         [llvm_i32_ty, llvm_i32_ty]>;
def int_loongarch_crcc_w_w_w : Intrinsic<[llvm_i32_ty],
                                         [llvm_i32_ty, llvm_i32_ty]>;
def int_loongarch_crcc_w_d_w : Intrinsic<[llvm_i32_ty],
                                         [llvm_i64_ty, llvm_i32_ty]>;

def int_loongarch_csrrd_w : Intrinsic<[llvm_i32_ty], [llvm_i32_ty],
                                      [ImmArg<ArgIndex<0>>]>;
def int_loongarch_csrrd_d : Intrinsic<[llvm_i64_ty], [llvm_i32_ty],
                                      [ImmArg<ArgIndex<0>>]>;
def int_loongarch_csrwr_w : Intrinsic<[llvm_i32_ty],
                                      [llvm_i32_ty, llvm_i32_ty],
                                      [ImmArg<ArgIndex<1>>]>;
def int_loongarch_csrwr_d : Intrinsic<[llvm_i64_ty],
                                      [llvm_i64_ty, llvm_i32_ty],
                                      [ImmArg<ArgIndex<1>>]>;
def int_loongarch_csrxchg_w : Intrinsic<[llvm_i32_ty],
                                        [llvm_i32_ty, llvm_i32_ty,
                                         llvm_i32_ty],
                                        [ImmArg<ArgIndex<2>>]>;
def int_loongarch_csrxchg_d : Intrinsic<[llvm_i64_ty],
                                        [llvm_i64_ty, llvm_i64_ty,
                                         llvm_i32_ty],
                                        [ImmArg<ArgIndex<2>>]>;

def int_loongarch_iocsrrd_b : Intrinsic<[llvm_i32_ty], [llvm_i32_ty]>;
def int_loongarch_iocsrrd_h : Intrinsic<[llvm_i32_ty], [llvm_i32_ty]>;
def int_loongarch_iocsrrd_w : Intrinsic<[llvm_i32_ty], [llvm_i32_ty]>;
def int_loongarch_iocsrrd_d : Intrinsic<[llvm_i64_ty], [llvm_i32_ty]>;

def int_loongarch_iocsrwr_b : Intrinsic<[], [llvm_i32_ty, llvm_i32_ty]>;
def int_loongarch_iocsrwr_h : Intrinsic<[], [llvm_i32_ty, llvm_i32_ty]>;
def int_loongarch_iocsrwr_w : Intrinsic<[], [llvm_i32_ty, llvm_i32_ty]>;
def int_loongarch_iocsrwr_d : Intrinsic<[], [llvm_i64_ty, llvm_i32_ty]>;
} // TargetPrefix = "loongarch"
