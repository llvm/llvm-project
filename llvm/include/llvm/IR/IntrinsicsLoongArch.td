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

class BaseInt<list<LLVMType> ret_types, list<LLVMType> param_types,
              list<IntrinsicProperty> intr_properties = []>
    : Intrinsic<ret_types, param_types, intr_properties>,
      ClangBuiltin<!subst("int_loongarch", "__builtin_loongarch", NAME)>;

def int_loongarch_break : BaseInt<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;
def int_loongarch_cacop_d : BaseInt<[], [llvm_i64_ty, llvm_i64_ty, llvm_i64_ty],
                                    [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>]>;
def int_loongarch_cacop_w : BaseInt<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                                    [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>]>;
def int_loongarch_dbar : BaseInt<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_loongarch_ibar : BaseInt<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;
def int_loongarch_movfcsr2gr : BaseInt<[llvm_i32_ty], [llvm_i32_ty],
                                       [ImmArg<ArgIndex<0>>]>;
def int_loongarch_movgr2fcsr : BaseInt<[], [llvm_i32_ty, llvm_i32_ty],
                                       [ImmArg<ArgIndex<0>>]>;
def int_loongarch_syscall : BaseInt<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_loongarch_crc_w_b_w : BaseInt<[llvm_i32_ty],
                                      [llvm_i32_ty, llvm_i32_ty]>;
def int_loongarch_crc_w_h_w : BaseInt<[llvm_i32_ty],
                                      [llvm_i32_ty, llvm_i32_ty]>;
def int_loongarch_crc_w_w_w : BaseInt<[llvm_i32_ty],
                                      [llvm_i32_ty, llvm_i32_ty]>;
def int_loongarch_crc_w_d_w : BaseInt<[llvm_i32_ty],
                                      [llvm_i64_ty, llvm_i32_ty]>;

def int_loongarch_crcc_w_b_w : BaseInt<[llvm_i32_ty],
                                       [llvm_i32_ty, llvm_i32_ty]>;
def int_loongarch_crcc_w_h_w : BaseInt<[llvm_i32_ty],
                                       [llvm_i32_ty, llvm_i32_ty]>;
def int_loongarch_crcc_w_w_w : BaseInt<[llvm_i32_ty],
                                       [llvm_i32_ty, llvm_i32_ty]>;
def int_loongarch_crcc_w_d_w : BaseInt<[llvm_i32_ty],
                                       [llvm_i64_ty, llvm_i32_ty]>;

def int_loongarch_csrrd_w : BaseInt<[llvm_i32_ty], [llvm_i32_ty],
                                    [ImmArg<ArgIndex<0>>]>;
def int_loongarch_csrrd_d : BaseInt<[llvm_i64_ty], [llvm_i32_ty],
                                    [ImmArg<ArgIndex<0>>]>;
def int_loongarch_csrwr_w : BaseInt<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
                                    [ImmArg<ArgIndex<1>>]>;
def int_loongarch_csrwr_d : BaseInt<[llvm_i64_ty], [llvm_i64_ty, llvm_i32_ty],
                                    [ImmArg<ArgIndex<1>>]>;
def int_loongarch_csrxchg_w : BaseInt<[llvm_i32_ty],
                                      [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                                      [ImmArg<ArgIndex<2>>]>;
def int_loongarch_csrxchg_d : BaseInt<[llvm_i64_ty],
                                      [llvm_i64_ty, llvm_i64_ty, llvm_i32_ty],
                                      [ImmArg<ArgIndex<2>>]>;

def int_loongarch_iocsrrd_b : BaseInt<[llvm_i32_ty], [llvm_i32_ty]>;
def int_loongarch_iocsrrd_h : BaseInt<[llvm_i32_ty], [llvm_i32_ty]>;
def int_loongarch_iocsrrd_w : BaseInt<[llvm_i32_ty], [llvm_i32_ty]>;
def int_loongarch_iocsrrd_d : BaseInt<[llvm_i64_ty], [llvm_i32_ty]>;

def int_loongarch_iocsrwr_b : BaseInt<[], [llvm_i32_ty, llvm_i32_ty]>;
def int_loongarch_iocsrwr_h : BaseInt<[], [llvm_i32_ty, llvm_i32_ty]>;
def int_loongarch_iocsrwr_w : BaseInt<[], [llvm_i32_ty, llvm_i32_ty]>;
def int_loongarch_iocsrwr_d : BaseInt<[], [llvm_i64_ty, llvm_i32_ty]>;

def int_loongarch_cpucfg : BaseInt<[llvm_i32_ty], [llvm_i32_ty]>;

def int_loongarch_asrtle_d : BaseInt<[], [llvm_i64_ty, llvm_i64_ty]>;
def int_loongarch_asrtgt_d : BaseInt<[], [llvm_i64_ty, llvm_i64_ty]>;

def int_loongarch_lddir_d : BaseInt<[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty],
                                    [ImmArg<ArgIndex<1>>]>;
def int_loongarch_ldpte_d : BaseInt<[], [llvm_i64_ty, llvm_i64_ty],
                                    [ImmArg<ArgIndex<1>>]>;
} // TargetPrefix = "loongarch"
