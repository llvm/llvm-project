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
  // i64 @llvm.<name>.i32.<p>(any*, i64, i64, i64 imm);
  def _i64 : MaskedAtomicRMW<llvm_i64_ty>;
}

defm int_loongarch_masked_atomicrmw_xchg : MaskedAtomicRMWIntrinsics;
} // TargetPrefix = "loongarch"
