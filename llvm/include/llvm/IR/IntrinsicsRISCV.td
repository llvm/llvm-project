//===- IntrinsicsRISCV.td - Defines RISCV intrinsics -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the RISCV-specific intrinsics.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Atomics

// Atomic Intrinsics have multiple versions for different access widths, which
// all follow one of the following signatures (depending on how many arguments
// they require). We carefully instantiate only specific versions of these for
// specific integer widths, rather than using `llvm_anyint_ty`.
//
// In fact, as these intrinsics take `llvm_anyptr_ty`, the given names are the
// canonical names, and the intrinsics used in the code will have a name
// suffixed with the pointer type they are specialised for (denoted `<p>` in the
// names below), in order to avoid type conflicts.

let TargetPrefix = "riscv" in {

  // T @llvm.<name>.T.<p>(any*, T, T, T imm);
  class MaskedAtomicRMWFourArg<LLVMType itype>
      : Intrinsic<[itype], [llvm_anyptr_ty, itype, itype, itype],
                  [IntrArgMemOnly, NoCapture<ArgIndex<0>>, ImmArg<ArgIndex<3>>]>;
  // T @llvm.<name>.T.<p>(any*, T, T, T, T imm);
  class MaskedAtomicRMWFiveArg<LLVMType itype>
      : Intrinsic<[itype], [llvm_anyptr_ty, itype, itype, itype, itype],
                  [IntrArgMemOnly, NoCapture<ArgIndex<0>>, ImmArg<ArgIndex<4>>]>;

  // We define 32-bit and 64-bit variants of the above, where T stands for i32
  // or i64 respectively:
  multiclass MaskedAtomicRMWFourArgIntrinsics {
    // i32 @llvm.<name>.i32.<p>(any*, i32, i32, i32 imm);
    def _i32 : MaskedAtomicRMWFourArg<llvm_i32_ty>;
    // i64 @llvm.<name>.i32.<p>(any*, i64, i64, i64 imm);
    def _i64 : MaskedAtomicRMWFourArg<llvm_i64_ty>;
  }

  multiclass MaskedAtomicRMWFiveArgIntrinsics {
    // i32 @llvm.<name>.i32.<p>(any*, i32, i32, i32, i32 imm);
    def _i32 : MaskedAtomicRMWFiveArg<llvm_i32_ty>;
    // i64 @llvm.<name>.i64.<p>(any*, i64, i64, i64, i64 imm);
    def _i64 : MaskedAtomicRMWFiveArg<llvm_i64_ty>;
  }

  // These intrinsics are intended only for internal compiler use (i.e. as
  // part of AtomicExpandpass via the emitMaskedAtomic*Intrinsic hooks). Their
  // names and semantics could change in the future.

  // @llvm.riscv.masked.atomicrmw.*.{i32,i64}.<p>(
  //   ptr addr, ixlen oparg, ixlen mask, ixlenimm ordering)
  defm int_riscv_masked_atomicrmw_xchg : MaskedAtomicRMWFourArgIntrinsics;
  defm int_riscv_masked_atomicrmw_add : MaskedAtomicRMWFourArgIntrinsics;
  defm int_riscv_masked_atomicrmw_sub : MaskedAtomicRMWFourArgIntrinsics;
  defm int_riscv_masked_atomicrmw_nand : MaskedAtomicRMWFourArgIntrinsics;
  defm int_riscv_masked_atomicrmw_umax : MaskedAtomicRMWFourArgIntrinsics;
  defm int_riscv_masked_atomicrmw_umin : MaskedAtomicRMWFourArgIntrinsics;
  // Signed min and max need an extra operand to do sign extension with.
  // @llvm.riscv.masked.atomicrmw.{max,min}.{i32,i64}.<p>(
  //   ptr addr, ixlen oparg, ixlen mask, ixlen shamt, ixlenimm ordering)
  defm int_riscv_masked_atomicrmw_max : MaskedAtomicRMWFiveArgIntrinsics;
  defm int_riscv_masked_atomicrmw_min : MaskedAtomicRMWFiveArgIntrinsics;

  // @llvm.riscv.masked.cmpxchg.{i32,i64}.<p>(
  //   ptr addr, ixlen cmpval, ixlen newval, ixlen mask, ixlenimm ordering)
  defm int_riscv_masked_cmpxchg : MaskedAtomicRMWFiveArgIntrinsics;

} // TargetPrefix = "riscv"

//===----------------------------------------------------------------------===//
// Bitmanip (Bit Manipulation) Extension

let TargetPrefix = "riscv" in {

  class BitManipGPRIntrinsics
      : DefaultAttrsIntrinsic<[llvm_any_ty],
                              [LLVMMatchType<0>],
                              [IntrNoMem, IntrSpeculatable]>;
  class BitManipGPRGPRIntrinsics
      : DefaultAttrsIntrinsic<[llvm_any_ty],
                              [LLVMMatchType<0>, LLVMMatchType<0>],
                              [IntrNoMem, IntrSpeculatable]>;

  // Zbb
  def int_riscv_orc_b : BitManipGPRIntrinsics;

  // Zbc or Zbkc
  def int_riscv_clmul  : BitManipGPRGPRIntrinsics;
  def int_riscv_clmulh : BitManipGPRGPRIntrinsics;

  // Zbc
  def int_riscv_clmulr : BitManipGPRGPRIntrinsics;

  // Zbkb
  def int_riscv_brev8 : BitManipGPRIntrinsics;
  def int_riscv_zip   : BitManipGPRIntrinsics;
  def int_riscv_unzip : BitManipGPRIntrinsics;

  // Zbkx
  def int_riscv_xperm4  : BitManipGPRGPRIntrinsics;
  def int_riscv_xperm8  : BitManipGPRGPRIntrinsics;
} // TargetPrefix = "riscv"

//===----------------------------------------------------------------------===//
// Vectors

// The intrinsic does not have any operand that must be extended.
defvar NoScalarOperand = 0xF;

// The intrinsic does not have a VL operand.
// (e.g., riscv_vmv_x_s and riscv_vfmv_f_s)
defvar NoVLOperand = 0x1F;

class RISCVVIntrinsic {
  // These intrinsics may accept illegal integer values in their llvm_any_ty
  // operand, so they have to be extended.
  Intrinsic IntrinsicID = !cast<Intrinsic>(NAME);
  bits<4> ScalarOperand = NoScalarOperand;
  bits<5> VLOperand = NoVLOperand;
}

let TargetPrefix = "riscv" in {
  // We use anyint here but we only support XLen.
  def int_riscv_vsetvli   : Intrinsic<[llvm_anyint_ty],
                           /* AVL */  [LLVMMatchType<0>,
                           /* VSEW */  LLVMMatchType<0>,
                           /* VLMUL */ LLVMMatchType<0>],
                                      [IntrNoMem, IntrHasSideEffects,
                                       ImmArg<ArgIndex<1>>,
                                       ImmArg<ArgIndex<2>>]>;
  def int_riscv_vsetvlimax : Intrinsic<[llvm_anyint_ty],
                            /* VSEW */ [LLVMMatchType<0>,
                            /* VLMUL */ LLVMMatchType<0>],
                                      [IntrNoMem, IntrHasSideEffects,
                                       ImmArg<ArgIndex<0>>,
                                       ImmArg<ArgIndex<1>>]>;

  // Versions without side effects: better optimizable and usable if only the
  // returned vector length is important.
  def int_riscv_vsetvli_opt   : Intrinsic<[llvm_anyint_ty],
                               /* AVL */  [LLVMMatchType<0>,
                               /* VSEW */  LLVMMatchType<0>,
                               /* VLMUL */ LLVMMatchType<0>],
                                          [IntrNoMem,
                                           ImmArg<ArgIndex<1>>,
                                           ImmArg<ArgIndex<2>>]>;
  def int_riscv_vsetvlimax_opt : Intrinsic<[llvm_anyint_ty],
                                /* VSEW */ [LLVMMatchType<0>,
                                /* VLMUL */ LLVMMatchType<0>],
                                          [IntrNoMem,
                                           ImmArg<ArgIndex<0>>,
                                           ImmArg<ArgIndex<1>>]>;

  // For unit stride mask load
  // Input: (pointer, vl)
  class RISCVUSMLoad
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMPointerType<LLVMMatchType<0>>,
                     llvm_anyint_ty],
                    [NoCapture<ArgIndex<0>>, IntrReadMem]>, RISCVVIntrinsic {
    let VLOperand = 1;
  }
  // For unit stride load
  // Input: (passthru, pointer, vl)
  class RISCVUSLoad
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>,
                     LLVMPointerType<LLVMMatchType<0>>,
                     llvm_anyint_ty],
                    [NoCapture<ArgIndex<1>>, IntrReadMem]>, RISCVVIntrinsic {
    let VLOperand = 2;
  }
  // For unit stride fault-only-first load
  // Input: (passthru, pointer, vl)
  // Output: (data, vl)
  // NOTE: We model this with default memory properties since we model writing
  // VL as a side effect. IntrReadMem, IntrHasSideEffects does not work.
  class RISCVUSLoadFF
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty, llvm_anyint_ty],
                    [LLVMMatchType<0>,
                     LLVMPointerType<LLVMMatchType<0>>, LLVMMatchType<1>],
                    [NoCapture<ArgIndex<1>>]>,
                    RISCVVIntrinsic {
    let VLOperand = 2;
  }
  // For unit stride load with mask
  // Input: (maskedoff, pointer, mask, vl, policy)
  class RISCVUSLoadMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty ],
                    [LLVMMatchType<0>,
                     LLVMPointerType<LLVMMatchType<0>>,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                     llvm_anyint_ty, LLVMMatchType<1>],
                    [NoCapture<ArgIndex<1>>, ImmArg<ArgIndex<4>>, IntrReadMem]>,
                    RISCVVIntrinsic {
    let VLOperand = 3;
  }
  // For unit stride fault-only-first load with mask
  // Input: (maskedoff, pointer, mask, vl, policy)
  // Output: (data, vl)
  // NOTE: We model this with default memory properties since we model writing
  // VL as a side effect. IntrReadMem, IntrHasSideEffects does not work.
  class RISCVUSLoadFFMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty, llvm_anyint_ty],
                    [LLVMMatchType<0>,
                     LLVMPointerType<LLVMMatchType<0>>,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                     LLVMMatchType<1>, LLVMMatchType<1>],
                    [NoCapture<ArgIndex<1>>, ImmArg<ArgIndex<4>>]>, RISCVVIntrinsic {
    let VLOperand = 3;
  }
  // For strided load with passthru operand
  // Input: (passthru, pointer, stride, vl)
  class RISCVSLoad
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>,
                     LLVMPointerType<LLVMMatchType<0>>,
                     llvm_anyint_ty, LLVMMatchType<1>],
                    [NoCapture<ArgIndex<1>>, IntrReadMem]>, RISCVVIntrinsic {
    let VLOperand = 3;
  }
  // For strided load with mask
  // Input: (maskedoff, pointer, stride, mask, vl, policy)
  class RISCVSLoadMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty ],
                    [LLVMMatchType<0>,
                     LLVMPointerType<LLVMMatchType<0>>, llvm_anyint_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, LLVMMatchType<1>,
                     LLVMMatchType<1>],
                    [NoCapture<ArgIndex<1>>, ImmArg<ArgIndex<5>>, IntrReadMem]>,
                    RISCVVIntrinsic {
    let VLOperand = 4;
  }
  // For indexed load with passthru operand
  // Input: (passthru, pointer, index, vl)
  class RISCVILoad
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>,
                     LLVMPointerType<LLVMMatchType<0>>,
                     llvm_anyvector_ty, llvm_anyint_ty],
                    [NoCapture<ArgIndex<1>>, IntrReadMem]>, RISCVVIntrinsic {
    let VLOperand = 3;
  }
  // For indexed load with mask
  // Input: (maskedoff, pointer, index, mask, vl, policy)
  class RISCVILoadMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty ],
                    [LLVMMatchType<0>,
                     LLVMPointerType<LLVMMatchType<0>>, llvm_anyvector_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty,
                     LLVMMatchType<2>],
                    [NoCapture<ArgIndex<1>>, ImmArg<ArgIndex<5>>, IntrReadMem]>,
                    RISCVVIntrinsic {
    let VLOperand = 4;
  }
  // For unit stride store
  // Input: (vector_in, pointer, vl)
  class RISCVUSStore
        : DefaultAttrsIntrinsic<[],
                    [llvm_anyvector_ty,
                     LLVMPointerType<LLVMMatchType<0>>,
                     llvm_anyint_ty],
                    [NoCapture<ArgIndex<1>>, IntrWriteMem]>, RISCVVIntrinsic {
    let VLOperand = 2;
  }
  // For unit stride store with mask
  // Input: (vector_in, pointer, mask, vl)
  class RISCVUSStoreMasked
        : DefaultAttrsIntrinsic<[],
                    [llvm_anyvector_ty,
                     LLVMPointerType<LLVMMatchType<0>>,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                     llvm_anyint_ty],
                    [NoCapture<ArgIndex<1>>, IntrWriteMem]>, RISCVVIntrinsic {
    let VLOperand = 3;
  }
  // For strided store
  // Input: (vector_in, pointer, stride, vl)
  class RISCVSStore
        : DefaultAttrsIntrinsic<[],
                    [llvm_anyvector_ty,
                     LLVMPointerType<LLVMMatchType<0>>,
                     llvm_anyint_ty, LLVMMatchType<1>],
                    [NoCapture<ArgIndex<1>>, IntrWriteMem]>, RISCVVIntrinsic {
    let VLOperand = 3;
  }
  // For stride store with mask
  // Input: (vector_in, pointer, stirde, mask, vl)
  class RISCVSStoreMasked
        : DefaultAttrsIntrinsic<[],
                    [llvm_anyvector_ty,
                     LLVMPointerType<LLVMMatchType<0>>, llvm_anyint_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, LLVMMatchType<1>],
                    [NoCapture<ArgIndex<1>>, IntrWriteMem]>, RISCVVIntrinsic {
    let VLOperand = 4;
  }
  // For indexed store
  // Input: (vector_in, pointer, index, vl)
  class RISCVIStore
        : DefaultAttrsIntrinsic<[],
                    [llvm_anyvector_ty,
                     LLVMPointerType<LLVMMatchType<0>>,
                     llvm_anyint_ty, llvm_anyint_ty],
                    [NoCapture<ArgIndex<1>>, IntrWriteMem]>, RISCVVIntrinsic {
    let VLOperand = 3;
  }
  // For indexed store with mask
  // Input: (vector_in, pointer, index, mask, vl)
  class RISCVIStoreMasked
        : DefaultAttrsIntrinsic<[],
                    [llvm_anyvector_ty,
                     LLVMPointerType<LLVMMatchType<0>>, llvm_anyvector_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty],
                    [NoCapture<ArgIndex<1>>, IntrWriteMem]>, RISCVVIntrinsic {
    let VLOperand = 4;
  }
  // For destination vector type is the same as source vector.
  // Input: (passthru, vector_in, vl)
  class RISCVUnaryAAUnMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>, llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 2;
  }
  // For destination vector type is the same as first source vector (with mask).
  // Input: (vector_in, vector_in, mask, vl, policy)
  class RISCVUnaryAAMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty,
                     LLVMMatchType<1>],
                    [ImmArg<ArgIndex<4>>, IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 3;
  }
  // Input: (passthru, vector_in, vector_in, mask, vl)
  class RISCVCompress
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 3;
  }
  // For destination vector type is the same as first and second source vector.
  // Input: (vector_in, vector_in, vl)
  class RISCVBinaryAAAUnMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>, llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 2;
  }
  // For destination vector type is the same as first and second source vector.
  // Input: (passthru, vector_in, int_vector_in, vl)
  class RISCVRGatherVVUnMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>,
                     LLVMVectorOfBitcastsToInt<0>, llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 3;
  }
  // For destination vector type is the same as first and second source vector.
  // Input: (vector_in, vector_in, int_vector_in, vl, policy)
  class RISCVRGatherVVMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>, LLVMVectorOfBitcastsToInt<0>,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty,
                     LLVMMatchType<1>],
                    [ImmArg<ArgIndex<5>>, IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 4;
  }
  // Input: (passthru, vector_in, int16_vector_in, vl)
  class RISCVRGatherEI16VVUnMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>,
                     LLVMScalarOrSameVectorWidth<0, llvm_i16_ty>,
                     llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 3;
  }
  // For destination vector type is the same as first and second source vector.
  // Input: (vector_in, vector_in, int16_vector_in, vl, policy)
  class RISCVRGatherEI16VVMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>,
                     LLVMScalarOrSameVectorWidth<0, llvm_i16_ty>,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty,
                     LLVMMatchType<1>],
                    [ImmArg<ArgIndex<5>>, IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 4;
  }
  // For destination vector type is the same as first source vector, and the
  // second operand is XLen.
  // Input: (passthru, vector_in, xlen_in, vl)
  class RISCVGatherVXUnMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>, llvm_anyint_ty,
                     LLVMMatchType<1>],
                    [IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 3;
  }
  // For destination vector type is the same as first source vector (with mask).
  // Second operand is XLen.
  // Input: (maskedoff, vector_in, xlen_in, mask, vl, policy)
  class RISCVGatherVXMasked
       : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_anyint_ty,
                    LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, LLVMMatchType<1>,
                    LLVMMatchType<1>],
                   [ImmArg<ArgIndex<5>>, IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 4;
  }
  // For destination vector type is the same as first source vector.
  // Input: (passthru, vector_in, vector_in/scalar_in, vl)
  class RISCVBinaryAAXUnMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>, llvm_any_ty,
                     llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic {
    let ScalarOperand = 2;
    let VLOperand = 3;
  }
  // For destination vector type is the same as first source vector (with mask).
  // Input: (maskedoff, vector_in, vector_in/scalar_in, mask, vl, policy)
  class RISCVBinaryAAXMasked
       : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_any_ty,
                    LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty,
                    LLVMMatchType<2>],
                   [ImmArg<ArgIndex<5>>, IntrNoMem]>, RISCVVIntrinsic {
    let ScalarOperand = 2;
    let VLOperand = 4;
  }
  // For destination vector type is the same as first source vector. The
  // second source operand must match the destination type or be an XLen scalar.
  // Input: (passthru, vector_in, vector_in/scalar_in, vl)
  class RISCVBinaryAAShiftUnMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>, llvm_any_ty,
                     llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 3;
  }
  // For destination vector type is the same as first source vector (with mask).
  // The second source operand must match the destination type or be an XLen scalar.
  // Input: (maskedoff, vector_in, vector_in/scalar_in, mask, vl, policy)
  class RISCVBinaryAAShiftMasked
       : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_any_ty,
                    LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty,
                    LLVMMatchType<2>],
                   [ImmArg<ArgIndex<5>>, IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 4;
  }
  // For destination vector type is NOT the same as first source vector.
  // Input: (passthru, vector_in, vector_in/scalar_in, vl)
  class RISCVBinaryABXUnMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyvector_ty, llvm_any_ty,
                     llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic {
    let ScalarOperand = 2;
    let VLOperand = 3;
  }
  // For destination vector type is NOT the same as first source vector (with mask).
  // Input: (maskedoff, vector_in, vector_in/scalar_in, mask, vl, policy)
  class RISCVBinaryABXMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyvector_ty, llvm_any_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty,
                     LLVMMatchType<3>],
                    [ImmArg<ArgIndex<5>>, IntrNoMem]>, RISCVVIntrinsic {
    let ScalarOperand = 2;
    let VLOperand = 4;
  }
  // For destination vector type is NOT the same as first source vector. The
  // second source operand must match the destination type or be an XLen scalar.
  // Input: (passthru, vector_in, vector_in/scalar_in, vl)
  class RISCVBinaryABShiftUnMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyvector_ty, llvm_any_ty,
                     llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 3;
  }
  // For destination vector type is NOT the same as first source vector (with mask).
  // The second source operand must match the destination type or be an XLen scalar.
  // Input: (maskedoff, vector_in, vector_in/scalar_in, mask, vl, policy)
  class RISCVBinaryABShiftMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyvector_ty, llvm_any_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty,
                     LLVMMatchType<3>],
                    [ImmArg<ArgIndex<5>>, IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 4;
  }
  // For binary operations with V0 as input.
  // Input: (passthru, vector_in, vector_in/scalar_in, V0, vl)
  class RISCVBinaryWithV0
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>, llvm_any_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                     llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic {
    let ScalarOperand = 2;
    let VLOperand = 4;
  }
  // For binary operations with mask type output and V0 as input.
  // Output: (mask type output)
  // Input: (vector_in, vector_in/scalar_in, V0, vl)
  class RISCVBinaryMOutWithV0
        :DefaultAttrsIntrinsic<[LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>],
                   [llvm_anyvector_ty, llvm_any_ty,
                    LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                    llvm_anyint_ty],
                   [IntrNoMem]>, RISCVVIntrinsic {
    let ScalarOperand = 1;
    let VLOperand = 3;
  }
  // For binary operations with mask type output.
  // Output: (mask type output)
  // Input: (vector_in, vector_in/scalar_in, vl)
  class RISCVBinaryMOut
        : DefaultAttrsIntrinsic<[LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>],
                    [llvm_anyvector_ty, llvm_any_ty, llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic {
    let ScalarOperand = 1;
    let VLOperand = 2;
  }
  // For binary operations with mask type output without mask.
  // Output: (mask type output)
  // Input: (vector_in, vector_in/scalar_in, vl)
  class RISCVCompareUnMasked
        : DefaultAttrsIntrinsic<[LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>],
                    [llvm_anyvector_ty, llvm_any_ty, llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic {
    let ScalarOperand = 1;
    let VLOperand = 2;
  }
  // For binary operations with mask type output with mask.
  // Output: (mask type output)
  // Input: (maskedoff, vector_in, vector_in/scalar_in, mask, vl)
  class RISCVCompareMasked
        : DefaultAttrsIntrinsic<[LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>],
                    [LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                     llvm_anyvector_ty, llvm_any_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic {
    let ScalarOperand = 2;
    let VLOperand = 4;
  }
  // For FP classify operations.
  // Output: (bit mask type output)
  // Input: (passthru, vector_in, vl)
  class RISCVClassifyUnMasked
        : DefaultAttrsIntrinsic<[LLVMVectorOfBitcastsToInt<0>],
                    [LLVMVectorOfBitcastsToInt<0>, llvm_anyvector_ty,
                      llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 1;
  }
  // For FP classify operations with mask.
  // Output: (bit mask type output)
  // Input: (maskedoff, vector_in, mask, vl, policy)
  class RISCVClassifyMasked
        : DefaultAttrsIntrinsic<[LLVMVectorOfBitcastsToInt<0>],
                    [LLVMVectorOfBitcastsToInt<0>, llvm_anyvector_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, 
                     llvm_anyint_ty, LLVMMatchType<1>],
                    [IntrNoMem, ImmArg<ArgIndex<4>>]>, RISCVVIntrinsic {
    let VLOperand = 3;
  }
  // For Saturating binary operations.
  // The destination vector type is the same as first source vector.
  // Input: (passthru, vector_in, vector_in/scalar_in, vl)
  class RISCVSaturatingBinaryAAXUnMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>, llvm_any_ty,
                     llvm_anyint_ty],
                    [IntrNoMem, IntrHasSideEffects]>, RISCVVIntrinsic {
    let ScalarOperand = 2;
    let VLOperand = 3;
  }
  // For Saturating binary operations with mask.
  // The destination vector type is the same as first source vector.
  // Input: (maskedoff, vector_in, vector_in/scalar_in, mask, vl, policy)
  class RISCVSaturatingBinaryAAXMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>, llvm_any_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty,
                     LLVMMatchType<2>],
                    [ImmArg<ArgIndex<5>>, IntrNoMem, IntrHasSideEffects]>, RISCVVIntrinsic {
    let ScalarOperand = 2;
    let VLOperand = 4;
  }
  // For Saturating binary operations.
  // The destination vector type is the same as first source vector.
  // The second source operand matches the destination type or is an XLen scalar.
  // Input: (passthru, vector_in, vector_in/scalar_in, vl)
  class RISCVSaturatingBinaryAAShiftUnMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>, llvm_any_ty,
                     llvm_anyint_ty],
                    [IntrNoMem, IntrHasSideEffects]>, RISCVVIntrinsic {
    let VLOperand = 3;
  }
  // For Saturating binary operations with mask.
  // The destination vector type is the same as first source vector.
  // The second source operand matches the destination type or is an XLen scalar.
  // Input: (maskedoff, vector_in, vector_in/scalar_in, mask, vl, policy)
  class RISCVSaturatingBinaryAAShiftMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>, llvm_any_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty,
                     LLVMMatchType<2>],
                    [ImmArg<ArgIndex<5>>, IntrNoMem, IntrHasSideEffects]>, RISCVVIntrinsic {
    let VLOperand = 4;
  }
  // For Saturating binary operations.
  // The destination vector type is NOT the same as first source vector.
  // The second source operand matches the destination type or is an XLen scalar.
  // Input: (passthru, vector_in, vector_in/scalar_in, vl)
  class RISCVSaturatingBinaryABShiftUnMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyvector_ty, llvm_any_ty,
                     llvm_anyint_ty],
                    [IntrNoMem, IntrHasSideEffects]>, RISCVVIntrinsic {
    let VLOperand = 3;
  }
  // For Saturating binary operations with mask.
  // The destination vector type is NOT the same as first source vector (with mask).
  // The second source operand matches the destination type or is an XLen scalar.
  // Input: (maskedoff, vector_in, vector_in/scalar_in, mask, vl, policy)
  class RISCVSaturatingBinaryABShiftMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyvector_ty, llvm_any_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty,
                     LLVMMatchType<3>],
                    [ImmArg<ArgIndex<5>>, IntrNoMem, IntrHasSideEffects]>, RISCVVIntrinsic {
    let VLOperand = 4;
  }
  // Input: (vector_in, vector_in, scalar_in, vl, policy)
  class RVVSlideUnMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>, llvm_anyint_ty,
                     LLVMMatchType<1>, LLVMMatchType<1>],
                    [ImmArg<ArgIndex<4>>, IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 3;
  }
  // Input: (vector_in, vector_in, vector_in/scalar_in, mask, vl, policy)
  class RVVSlideMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>, llvm_anyint_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                     LLVMMatchType<1>, LLVMMatchType<1>],
                    [ImmArg<ArgIndex<5>>, IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 4;
  }
  // UnMasked Vector Multiply-Add operations, its first operand can not be undef.
  // Input: (vector_in, vector_in/scalar, vector_in, vl, policy)
  class RISCVTernaryAAXAUnMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_any_ty, LLVMMatchType<0>,
                     llvm_anyint_ty, LLVMMatchType<2>],
                    [ImmArg<ArgIndex<4>>, IntrNoMem]>, RISCVVIntrinsic {
    let ScalarOperand = 1;
    let VLOperand = 3;
  }
  // Masked Vector Multiply-Add operations, its first operand can not be undef.
  // Input: (vector_in, vector_in/scalar, vector_in, mask, vl, policy
  class RISCVTernaryAAXAMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_any_ty, LLVMMatchType<0>,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                     llvm_anyint_ty, LLVMMatchType<2>],
                    [ImmArg<ArgIndex<5>>, IntrNoMem]>, RISCVVIntrinsic {
    let ScalarOperand = 1;
    let VLOperand = 4;
  }
  // UnMasked Widening Vector Multiply-Add operations, its first operand can not be undef.
  // Input: (vector_in, vector_in/scalar, vector_in, vl, policy)
  class RISCVTernaryWideUnMasked
        : DefaultAttrsIntrinsic< [llvm_anyvector_ty],
                     [LLVMMatchType<0>, llvm_any_ty, llvm_anyvector_ty,
                      llvm_anyint_ty, LLVMMatchType<3>],
                     [ImmArg<ArgIndex<4>>, IntrNoMem] >, RISCVVIntrinsic {
    let ScalarOperand = 1;
    let VLOperand = 3;
  }
  // Masked Widening Vector Multiply-Add operations, its first operand can not be undef.
  // Input: (vector_in, vector_in/scalar, vector_in, mask, vl, policy
  class RISCVTernaryWideMasked
        : DefaultAttrsIntrinsic< [llvm_anyvector_ty],
                     [LLVMMatchType<0>, llvm_any_ty, llvm_anyvector_ty,
                      LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                      llvm_anyint_ty, LLVMMatchType<3>],
                     [ImmArg<ArgIndex<5>>, IntrNoMem]>, RISCVVIntrinsic {
    let ScalarOperand = 1;
    let VLOperand = 4;
  }
  // For Reduction ternary operations.
  // For destination vector type is the same as first and third source vector.
  // Input: (vector_in, vector_in, vector_in, vl)
  class RISCVReductionUnMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyvector_ty, LLVMMatchType<0>,
                     llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 3;
  }
  // For Reduction ternary operations with mask.
  // For destination vector type is the same as first and third source vector.
  // The mask type come from second source vector.
  // Input: (maskedoff, vector_in, vector_in, vector_in, mask, vl)
  class RISCVReductionMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyvector_ty, LLVMMatchType<0>,
                     LLVMScalarOrSameVectorWidth<1, llvm_i1_ty>, llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 4;
  }
  // For unary operations with scalar type output without mask
  // Output: (scalar type)
  // Input: (vector_in, vl)
  class RISCVMaskedUnarySOutUnMasked
        : DefaultAttrsIntrinsic<[LLVMMatchType<1>],
                    [llvm_anyvector_ty, llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 1;
  }
  // For unary operations with scalar type output with mask
  // Output: (scalar type)
  // Input: (vector_in, mask, vl)
  class RISCVMaskedUnarySOutMasked
        : DefaultAttrsIntrinsic<[LLVMMatchType<1>],
                    [llvm_anyvector_ty, LLVMMatchType<0>, llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 2;
  }
  // For destination vector type is NOT the same as source vector.
  // Input: (passthru, vector_in, vl)
  class RISCVUnaryABUnMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyvector_ty, llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 2;
  }
  // For destination vector type is NOT the same as source vector (with mask).
  // Input: (maskedoff, vector_in, mask, vl, policy)
  class RISCVUnaryABMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyvector_ty,
                     LLVMScalarOrSameVectorWidth<1, llvm_i1_ty>,
                     llvm_anyint_ty, LLVMMatchType<2>],
                    [ImmArg<ArgIndex<4>>, IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 3;
  }
  // For unary operations with the same vector type in/out without mask
  // Output: (vector)
  // Input: (vector_in, vl)
  class RISCVUnaryUnMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 1;
  }
  // For mask unary operations with mask type in/out with mask
  // Output: (mask type output)
  // Input: (mask type maskedoff, mask type vector_in, mask, vl)
  class RISCVMaskedUnaryMOutMasked
        : DefaultAttrsIntrinsic<[llvm_anyint_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>,
                     LLVMMatchType<0>, llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 3;
  }
  // Output: (vector)
  // Input: (vl)
  class RISCVNullaryIntrinsic
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [llvm_anyint_ty], [IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 1;
  }
  // Output: (vector)
  // Input: (passthru, vl)
  class RISCVID
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 1;
  }
  // For Conversion unary operations.
  // Input: (passthru, vector_in, vl)
  class RISCVConversionUnMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyvector_ty, llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 2;
  }
  // For Conversion unary operations with mask.
  // Input: (maskedoff, vector_in, mask, vl, policy)
  class RISCVConversionMasked
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyvector_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty,
                     LLVMMatchType<2>],
                    [ImmArg<ArgIndex<4>>, IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 3;
  }

  // For unit stride segment load
  // Input: (passthru, pointer, vl)
  class RISCVUSSegLoad<int nf>
        : DefaultAttrsIntrinsic<!listconcat([llvm_anyvector_ty], !listsplat(LLVMMatchType<0>,
                                !add(nf, -1))),
                    !listconcat(!listsplat(LLVMMatchType<0>, nf),
                                [LLVMPointerToElt<0>, llvm_anyint_ty]),
                    [NoCapture<ArgIndex<nf>>, IntrReadMem]>, RISCVVIntrinsic {
    let VLOperand = !add(nf, 1);
  }
  // For unit stride segment load with mask
  // Input: (maskedoff, pointer, mask, vl, policy)
  class RISCVUSSegLoadMasked<int nf>
        : DefaultAttrsIntrinsic<!listconcat([llvm_anyvector_ty], !listsplat(LLVMMatchType<0>,
                                !add(nf, -1))),
                    !listconcat(!listsplat(LLVMMatchType<0>, nf),
                                [LLVMPointerToElt<0>,
                                 LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                                 llvm_anyint_ty, LLVMMatchType<1>]),
                    [ImmArg<ArgIndex<!add(nf, 3)>>, NoCapture<ArgIndex<nf>>, IntrReadMem]>,
                    RISCVVIntrinsic {
    let VLOperand = !add(nf, 2);
  }

  // For unit stride fault-only-first segment load
  // Input: (passthru, pointer, vl)
  // Output: (data, vl)
  // NOTE: We model this with default memory properties since we model writing
  // VL as a side effect. IntrReadMem, IntrHasSideEffects does not work.
  class RISCVUSSegLoadFF<int nf>
        : DefaultAttrsIntrinsic<!listconcat([llvm_anyvector_ty], !listsplat(LLVMMatchType<0>,
                                !add(nf, -1)), [llvm_anyint_ty]),
                    !listconcat(!listsplat(LLVMMatchType<0>, nf),
                    [LLVMPointerToElt<0>, LLVMMatchType<1>]),
                    [NoCapture<ArgIndex<nf>>]>, RISCVVIntrinsic {
    let VLOperand = !add(nf, 1);
  }
  // For unit stride fault-only-first segment load with mask
  // Input: (maskedoff, pointer, mask, vl, policy)
  // Output: (data, vl)
  // NOTE: We model this with default memory properties since we model writing
  // VL as a side effect. IntrReadMem, IntrHasSideEffects does not work.
  class RISCVUSSegLoadFFMasked<int nf>
        : DefaultAttrsIntrinsic<!listconcat([llvm_anyvector_ty], !listsplat(LLVMMatchType<0>,
                                !add(nf, -1)), [llvm_anyint_ty]),
                    !listconcat(!listsplat(LLVMMatchType<0>, nf),
                     [LLVMPointerToElt<0>,
                      LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                      LLVMMatchType<1>, LLVMMatchType<1>]),
                    [ImmArg<ArgIndex<!add(nf, 3)>>, NoCapture<ArgIndex<nf>>]>,
                    RISCVVIntrinsic {
    let VLOperand = !add(nf, 2);
  }

  // For stride segment load
  // Input: (passthru, pointer, offset, vl)
  class RISCVSSegLoad<int nf>
        : DefaultAttrsIntrinsic<!listconcat([llvm_anyvector_ty], !listsplat(LLVMMatchType<0>,
                                !add(nf, -1))),
                    !listconcat(!listsplat(LLVMMatchType<0>, nf),
                    [LLVMPointerToElt<0>, llvm_anyint_ty, LLVMMatchType<1>]),
                    [NoCapture<ArgIndex<nf>>, IntrReadMem]>, RISCVVIntrinsic {
    let VLOperand = !add(nf, 2);
  }
  // For stride segment load with mask
  // Input: (maskedoff, pointer, offset, mask, vl, policy)
  class RISCVSSegLoadMasked<int nf>
        : DefaultAttrsIntrinsic<!listconcat([llvm_anyvector_ty], !listsplat(LLVMMatchType<0>,
                                !add(nf, -1))),
                    !listconcat(!listsplat(LLVMMatchType<0>, nf),
                                [LLVMPointerToElt<0>,
                                 llvm_anyint_ty,
                                 LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                                 LLVMMatchType<1>, LLVMMatchType<1>]),
                    [ImmArg<ArgIndex<!add(nf, 4)>>, NoCapture<ArgIndex<nf>>, IntrReadMem]>,
                    RISCVVIntrinsic {
    let VLOperand = !add(nf, 3);
  }

  // For indexed segment load
  // Input: (passthru, pointer, index, vl)
  class RISCVISegLoad<int nf>
        : DefaultAttrsIntrinsic<!listconcat([llvm_anyvector_ty], !listsplat(LLVMMatchType<0>,
                                !add(nf, -1))),
                    !listconcat(!listsplat(LLVMMatchType<0>, nf),
                    [LLVMPointerToElt<0>, llvm_anyvector_ty, llvm_anyint_ty]),
                    [NoCapture<ArgIndex<nf>>, IntrReadMem]>, RISCVVIntrinsic {
    let VLOperand = !add(nf, 2);
  }
  // For indexed segment load with mask
  // Input: (maskedoff, pointer, index, mask, vl, policy)
  class RISCVISegLoadMasked<int nf>
        : DefaultAttrsIntrinsic<!listconcat([llvm_anyvector_ty], !listsplat(LLVMMatchType<0>,
                                !add(nf, -1))),
                    !listconcat(!listsplat(LLVMMatchType<0>, nf),
                                [LLVMPointerToElt<0>,
                                 llvm_anyvector_ty,
                                 LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                                 llvm_anyint_ty, LLVMMatchType<2>]),
                    [ImmArg<ArgIndex<!add(nf, 4)>>, NoCapture<ArgIndex<nf>>, IntrReadMem]>,
                    RISCVVIntrinsic {
    let VLOperand = !add(nf, 3);
  }

  // For unit stride segment store
  // Input: (value, pointer, vl)
  class RISCVUSSegStore<int nf>
        : DefaultAttrsIntrinsic<[],
                    !listconcat([llvm_anyvector_ty],
                                !listsplat(LLVMMatchType<0>, !add(nf, -1)),
                                [LLVMPointerToElt<0>, llvm_anyint_ty]),
                    [NoCapture<ArgIndex<nf>>, IntrWriteMem]>, RISCVVIntrinsic {
    let VLOperand = !add(nf, 1);
  }
  // For unit stride segment store with mask
  // Input: (value, pointer, mask, vl)
  class RISCVUSSegStoreMasked<int nf>
        : DefaultAttrsIntrinsic<[],
                    !listconcat([llvm_anyvector_ty],
                                !listsplat(LLVMMatchType<0>, !add(nf, -1)),
                                [LLVMPointerToElt<0>,
                                 LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                                 llvm_anyint_ty]),
                    [NoCapture<ArgIndex<nf>>, IntrWriteMem]>, RISCVVIntrinsic {
    let VLOperand = !add(nf, 2);
  }

  // For stride segment store
  // Input: (value, pointer, offset, vl)
  class RISCVSSegStore<int nf>
        : DefaultAttrsIntrinsic<[],
                    !listconcat([llvm_anyvector_ty],
                                !listsplat(LLVMMatchType<0>, !add(nf, -1)),
                                [LLVMPointerToElt<0>, llvm_anyint_ty,
                                 LLVMMatchType<1>]),
                    [NoCapture<ArgIndex<nf>>, IntrWriteMem]>, RISCVVIntrinsic {
    let VLOperand = !add(nf, 2);
  }
  // For stride segment store with mask
  // Input: (value, pointer, offset, mask, vl)
  class RISCVSSegStoreMasked<int nf>
        : DefaultAttrsIntrinsic<[],
                    !listconcat([llvm_anyvector_ty],
                                !listsplat(LLVMMatchType<0>, !add(nf, -1)),
                                [LLVMPointerToElt<0>, llvm_anyint_ty,
                                 LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                                 LLVMMatchType<1>]),
                    [NoCapture<ArgIndex<nf>>, IntrWriteMem]>, RISCVVIntrinsic {
    let VLOperand = !add(nf, 3);
  }

  // For indexed segment store
  // Input: (value, pointer, offset, vl)
  class RISCVISegStore<int nf>
        : DefaultAttrsIntrinsic<[],
                    !listconcat([llvm_anyvector_ty],
                                !listsplat(LLVMMatchType<0>, !add(nf, -1)),
                                [LLVMPointerToElt<0>, llvm_anyvector_ty,
                                 llvm_anyint_ty]),
                    [NoCapture<ArgIndex<nf>>, IntrWriteMem]>, RISCVVIntrinsic {
    let VLOperand = !add(nf, 2);
  }
  // For indexed segment store with mask
  // Input: (value, pointer, offset, mask, vl)
  class RISCVISegStoreMasked<int nf>
        : DefaultAttrsIntrinsic<[],
                    !listconcat([llvm_anyvector_ty],
                                !listsplat(LLVMMatchType<0>, !add(nf, -1)),
                                [LLVMPointerToElt<0>, llvm_anyvector_ty,
                                 LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                                 llvm_anyint_ty]),
                    [NoCapture<ArgIndex<nf>>, IntrWriteMem]>, RISCVVIntrinsic {
    let VLOperand = !add(nf, 3);
  }

  multiclass RISCVUSLoad {
    def "int_riscv_" # NAME : RISCVUSLoad;
    def "int_riscv_" # NAME # "_mask" : RISCVUSLoadMasked;
  }
  multiclass RISCVUSLoadFF {
    def "int_riscv_" # NAME : RISCVUSLoadFF;
    def "int_riscv_" # NAME # "_mask" : RISCVUSLoadFFMasked;
  }
  multiclass RISCVSLoad {
    def "int_riscv_" # NAME : RISCVSLoad;
    def "int_riscv_" # NAME # "_mask" : RISCVSLoadMasked;
  }
  multiclass RISCVILoad {
    def "int_riscv_" # NAME : RISCVILoad;
    def "int_riscv_" # NAME # "_mask" : RISCVILoadMasked;
  }
  multiclass RISCVUSStore {
    def "int_riscv_" # NAME : RISCVUSStore;
    def "int_riscv_" # NAME # "_mask" : RISCVUSStoreMasked;
  }
  multiclass RISCVSStore {
    def "int_riscv_" # NAME : RISCVSStore;
    def "int_riscv_" # NAME # "_mask" : RISCVSStoreMasked;
  }

  multiclass RISCVIStore {
    def "int_riscv_" # NAME : RISCVIStore;
    def "int_riscv_" # NAME # "_mask" : RISCVIStoreMasked;
  }
  multiclass RISCVUnaryAA {
    def "int_riscv_" # NAME : RISCVUnaryAAUnMasked;
    def "int_riscv_" # NAME # "_mask" : RISCVUnaryAAMasked;
  }
  multiclass RISCVUnaryAB {
    def "int_riscv_" # NAME : RISCVUnaryABUnMasked;
    def "int_riscv_" # NAME # "_mask" : RISCVUnaryABMasked;
  }
  // AAX means the destination type(A) is the same as the first source
  // type(A). X means any type for the second source operand.
  multiclass RISCVBinaryAAX {
    def "int_riscv_" # NAME : RISCVBinaryAAXUnMasked;
    def "int_riscv_" # NAME # "_mask" : RISCVBinaryAAXMasked;
  }
  // Like RISCVBinaryAAX, but the second operand is used a shift amount so it
  // must be a vector or an XLen scalar.
  multiclass RISCVBinaryAAShift {
    def "int_riscv_" # NAME : RISCVBinaryAAShiftUnMasked;
    def "int_riscv_" # NAME # "_mask" : RISCVBinaryAAShiftMasked;
  }
  multiclass RISCVRGatherVV {
    def "int_riscv_" # NAME : RISCVRGatherVVUnMasked;
    def "int_riscv_" # NAME # "_mask" : RISCVRGatherVVMasked;
  }
  multiclass RISCVRGatherVX {
    def "int_riscv_" # NAME : RISCVGatherVXUnMasked;
    def "int_riscv_" # NAME # "_mask" : RISCVGatherVXMasked;
  }
  multiclass RISCVRGatherEI16VV {
    def "int_riscv_" # NAME : RISCVRGatherEI16VVUnMasked;
    def "int_riscv_" # NAME # "_mask" : RISCVRGatherEI16VVMasked;
  }
  // ABX means the destination type(A) is different from the first source
  // type(B). X means any type for the second source operand.
  multiclass RISCVBinaryABX {
    def "int_riscv_" # NAME : RISCVBinaryABXUnMasked;
    def "int_riscv_" # NAME # "_mask" : RISCVBinaryABXMasked;
  }
  // Like RISCVBinaryABX, but the second operand is used a shift amount so it
  // must be a vector or an XLen scalar.
  multiclass RISCVBinaryABShift {
    def "int_riscv_" # NAME : RISCVBinaryABShiftUnMasked;
    def "int_riscv_" # NAME # "_mask" : RISCVBinaryABShiftMasked;
  }
  multiclass RISCVBinaryWithV0 {
    def "int_riscv_" # NAME : RISCVBinaryWithV0;
  }
  multiclass RISCVBinaryMaskOutWithV0 {
    def "int_riscv_" # NAME : RISCVBinaryMOutWithV0;
  }
  multiclass RISCVBinaryMaskOut {
    def "int_riscv_" # NAME : RISCVBinaryMOut;
  }
  multiclass RISCVSaturatingBinaryAAX {
    def "int_riscv_" # NAME : RISCVSaturatingBinaryAAXUnMasked;
    def "int_riscv_" # NAME # "_mask" : RISCVSaturatingBinaryAAXMasked;
  }
  multiclass RISCVSaturatingBinaryAAShift {
    def "int_riscv_" # NAME : RISCVSaturatingBinaryAAShiftUnMasked;
    def "int_riscv_" # NAME # "_mask" : RISCVSaturatingBinaryAAShiftMasked;
  }
  multiclass RISCVSaturatingBinaryABShift {
    def "int_riscv_" # NAME : RISCVSaturatingBinaryABShiftUnMasked;
    def "int_riscv_" # NAME # "_mask" : RISCVSaturatingBinaryABShiftMasked;
  }
  multiclass RVVSlide {
    def "int_riscv_" # NAME : RVVSlideUnMasked;
    def "int_riscv_" # NAME # "_mask" : RVVSlideMasked;
  }
  multiclass RISCVTernaryAAXA {
    def "int_riscv_" # NAME : RISCVTernaryAAXAUnMasked;
    def "int_riscv_" # NAME # "_mask" : RISCVTernaryAAXAMasked;
  }
  multiclass RISCVCompare {
    def "int_riscv_" # NAME : RISCVCompareUnMasked;
    def "int_riscv_" # NAME # "_mask" : RISCVCompareMasked;
  }
  multiclass RISCVClassify {
    def "int_riscv_" # NAME : RISCVClassifyUnMasked;
    def "int_riscv_" # NAME # "_mask" : RISCVClassifyMasked;
  }
  multiclass RISCVTernaryWide {
    def "int_riscv_" # NAME : RISCVTernaryWideUnMasked;
    def "int_riscv_" # NAME # "_mask" : RISCVTernaryWideMasked;
  }
  multiclass RISCVReduction {
    def "int_riscv_" # NAME : RISCVReductionUnMasked;
    def "int_riscv_" # NAME # "_mask" : RISCVReductionMasked;
  }
  multiclass RISCVMaskedUnarySOut {
    def "int_riscv_" # NAME : RISCVMaskedUnarySOutUnMasked;
    def "int_riscv_" # NAME # "_mask" : RISCVMaskedUnarySOutMasked;
  }
  multiclass RISCVMaskedUnaryMOut {
    def "int_riscv_" # NAME : RISCVUnaryUnMasked;
    def "int_riscv_" # NAME # "_mask" : RISCVMaskedUnaryMOutMasked;
  }
  multiclass RISCVConversion {
    def "int_riscv_" #NAME :RISCVConversionUnMasked;
    def "int_riscv_" # NAME # "_mask" : RISCVConversionMasked;
  }
  multiclass RISCVUSSegLoad<int nf> {
    def "int_riscv_" # NAME : RISCVUSSegLoad<nf>;
    def "int_riscv_" # NAME # "_mask" : RISCVUSSegLoadMasked<nf>;
  }
  multiclass RISCVUSSegLoadFF<int nf> {
    def "int_riscv_" # NAME : RISCVUSSegLoadFF<nf>;
    def "int_riscv_" # NAME # "_mask" : RISCVUSSegLoadFFMasked<nf>;
  }
  multiclass RISCVSSegLoad<int nf> {
    def "int_riscv_" # NAME : RISCVSSegLoad<nf>;
    def "int_riscv_" # NAME # "_mask" : RISCVSSegLoadMasked<nf>;
  }
  multiclass RISCVISegLoad<int nf> {
    def "int_riscv_" # NAME : RISCVISegLoad<nf>;
    def "int_riscv_" # NAME # "_mask" : RISCVISegLoadMasked<nf>;
  }
  multiclass RISCVUSSegStore<int nf> {
    def "int_riscv_" # NAME : RISCVUSSegStore<nf>;
    def "int_riscv_" # NAME # "_mask" : RISCVUSSegStoreMasked<nf>;
  }
  multiclass RISCVSSegStore<int nf> {
    def "int_riscv_" # NAME : RISCVSSegStore<nf>;
    def "int_riscv_" # NAME # "_mask" : RISCVSSegStoreMasked<nf>;
  }
  multiclass RISCVISegStore<int nf> {
    def "int_riscv_" # NAME : RISCVISegStore<nf>;
    def "int_riscv_" # NAME # "_mask" : RISCVISegStoreMasked<nf>;
  }

  defm vle : RISCVUSLoad;
  defm vleff : RISCVUSLoadFF;
  defm vse : RISCVUSStore;
  defm vlse: RISCVSLoad;
  defm vsse: RISCVSStore;
  defm vluxei : RISCVILoad;
  defm vloxei : RISCVILoad;
  defm vsoxei : RISCVIStore;
  defm vsuxei : RISCVIStore;

  def int_riscv_vlm : RISCVUSMLoad;
  def int_riscv_vsm : RISCVUSStore;

  defm vadd : RISCVBinaryAAX;
  defm vsub : RISCVBinaryAAX;
  defm vrsub : RISCVBinaryAAX;

  defm vwaddu : RISCVBinaryABX;
  defm vwadd : RISCVBinaryABX;
  defm vwaddu_w : RISCVBinaryAAX;
  defm vwadd_w : RISCVBinaryAAX;
  defm vwsubu : RISCVBinaryABX;
  defm vwsub : RISCVBinaryABX;
  defm vwsubu_w : RISCVBinaryAAX;
  defm vwsub_w : RISCVBinaryAAX;

  defm vzext : RISCVUnaryAB;
  defm vsext : RISCVUnaryAB;

  defm vadc : RISCVBinaryWithV0;
  defm vmadc_carry_in : RISCVBinaryMaskOutWithV0;
  defm vmadc : RISCVBinaryMaskOut;

  defm vsbc : RISCVBinaryWithV0;
  defm vmsbc_borrow_in : RISCVBinaryMaskOutWithV0;
  defm vmsbc : RISCVBinaryMaskOut;

  defm vand : RISCVBinaryAAX;
  defm vor : RISCVBinaryAAX;
  defm vxor : RISCVBinaryAAX;

  defm vsll : RISCVBinaryAAShift;
  defm vsrl : RISCVBinaryAAShift;
  defm vsra : RISCVBinaryAAShift;

  defm vnsrl : RISCVBinaryABShift;
  defm vnsra : RISCVBinaryABShift;

  defm vmseq : RISCVCompare;
  defm vmsne : RISCVCompare;
  defm vmsltu : RISCVCompare;
  defm vmslt : RISCVCompare;
  defm vmsleu : RISCVCompare;
  defm vmsle : RISCVCompare;
  defm vmsgtu : RISCVCompare;
  defm vmsgt : RISCVCompare;
  defm vmsgeu : RISCVCompare;
  defm vmsge : RISCVCompare;

  defm vminu : RISCVBinaryAAX;
  defm vmin : RISCVBinaryAAX;
  defm vmaxu : RISCVBinaryAAX;
  defm vmax : RISCVBinaryAAX;

  defm vmul : RISCVBinaryAAX;
  defm vmulh : RISCVBinaryAAX;
  defm vmulhu : RISCVBinaryAAX;
  defm vmulhsu : RISCVBinaryAAX;

  defm vdivu : RISCVBinaryAAX;
  defm vdiv : RISCVBinaryAAX;
  defm vremu : RISCVBinaryAAX;
  defm vrem : RISCVBinaryAAX;

  defm vwmul : RISCVBinaryABX;
  defm vwmulu : RISCVBinaryABX;
  defm vwmulsu : RISCVBinaryABX;

  defm vmacc : RISCVTernaryAAXA;
  defm vnmsac : RISCVTernaryAAXA;
  defm vmadd : RISCVTernaryAAXA;
  defm vnmsub : RISCVTernaryAAXA;

  defm vwmaccu  : RISCVTernaryWide;
  defm vwmacc   : RISCVTernaryWide;
  defm vwmaccus : RISCVTernaryWide;
  defm vwmaccsu : RISCVTernaryWide;

  defm vfadd : RISCVBinaryAAX;
  defm vfsub : RISCVBinaryAAX;
  defm vfrsub : RISCVBinaryAAX;

  defm vfwadd : RISCVBinaryABX;
  defm vfwsub : RISCVBinaryABX;
  defm vfwadd_w : RISCVBinaryAAX;
  defm vfwsub_w : RISCVBinaryAAX;

  defm vsaddu : RISCVSaturatingBinaryAAX;
  defm vsadd : RISCVSaturatingBinaryAAX;
  defm vssubu : RISCVSaturatingBinaryAAX;
  defm vssub : RISCVSaturatingBinaryAAX;

  defm vmerge : RISCVBinaryWithV0;

  // Output: (vector)
  // Input: (passthru, vector_in, vl)
  def int_riscv_vmv_v_v : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                                                [LLVMMatchType<0>,
                                                 LLVMMatchType<0>,
                                                 llvm_anyint_ty],
                                                [IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 2;
  }
  // Output: (vector)
  // Input: (passthru, scalar, vl)
  def int_riscv_vmv_v_x : DefaultAttrsIntrinsic<[llvm_anyint_ty],
                                                 [LLVMMatchType<0>,
                                                  LLVMVectorElementType<0>,
                                                  llvm_anyint_ty],
                                                 [IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 2;
  }
  // Output: (vector)
  // Input: (passthru, scalar, vl)
  def int_riscv_vfmv_v_f : DefaultAttrsIntrinsic<[llvm_anyfloat_ty],
                                                 [LLVMMatchType<0>,
                                                  LLVMVectorElementType<0>,
                                                  llvm_anyint_ty],
                                                 [IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 2;
  }

  def int_riscv_vmv_x_s : DefaultAttrsIntrinsic<[LLVMVectorElementType<0>],
                                                [llvm_anyint_ty],
                                                [IntrNoMem]>, RISCVVIntrinsic;
  def int_riscv_vmv_s_x : DefaultAttrsIntrinsic<[llvm_anyint_ty],
                                                [LLVMMatchType<0>,
                                                 LLVMVectorElementType<0>,
                                                 llvm_anyint_ty],
                                                [IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 2;
  }

  def int_riscv_vfmv_f_s : DefaultAttrsIntrinsic<[LLVMVectorElementType<0>],
                                                 [llvm_anyfloat_ty],
                                                 [IntrNoMem]>, RISCVVIntrinsic;
  def int_riscv_vfmv_s_f : DefaultAttrsIntrinsic<[llvm_anyfloat_ty],
                                                 [LLVMMatchType<0>,
                                                  LLVMVectorElementType<0>,
                                                  llvm_anyint_ty],
                                                 [IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 2;
  }

  defm vfmul : RISCVBinaryAAX;
  defm vfdiv : RISCVBinaryAAX;
  defm vfrdiv : RISCVBinaryAAX;

  defm vfwmul : RISCVBinaryABX;

  defm vfmacc : RISCVTernaryAAXA;
  defm vfnmacc : RISCVTernaryAAXA;
  defm vfmsac : RISCVTernaryAAXA;
  defm vfnmsac : RISCVTernaryAAXA;
  defm vfmadd : RISCVTernaryAAXA;
  defm vfnmadd : RISCVTernaryAAXA;
  defm vfmsub : RISCVTernaryAAXA;
  defm vfnmsub : RISCVTernaryAAXA;

  defm vfwmacc : RISCVTernaryWide;
  defm vfwnmacc : RISCVTernaryWide;
  defm vfwmsac : RISCVTernaryWide;
  defm vfwnmsac : RISCVTernaryWide;

  defm vfsqrt : RISCVUnaryAA;
  defm vfrsqrt7 : RISCVUnaryAA;
  defm vfrec7 : RISCVUnaryAA;

  defm vfmin : RISCVBinaryAAX;
  defm vfmax : RISCVBinaryAAX;

  defm vfsgnj : RISCVBinaryAAX;
  defm vfsgnjn : RISCVBinaryAAX;
  defm vfsgnjx : RISCVBinaryAAX;

  defm vfclass : RISCVClassify;

  defm vfmerge : RISCVBinaryWithV0;

  defm vslideup : RVVSlide;
  defm vslidedown : RVVSlide;

  defm vslide1up : RISCVBinaryAAX;
  defm vslide1down : RISCVBinaryAAX;
  defm vfslide1up : RISCVBinaryAAX;
  defm vfslide1down : RISCVBinaryAAX;

  defm vrgather_vv : RISCVRGatherVV;
  defm vrgather_vx : RISCVRGatherVX;
  defm vrgatherei16_vv : RISCVRGatherEI16VV;

  def "int_riscv_vcompress" : RISCVCompress;

  defm vaaddu : RISCVSaturatingBinaryAAX;
  defm vaadd : RISCVSaturatingBinaryAAX;
  defm vasubu : RISCVSaturatingBinaryAAX;
  defm vasub : RISCVSaturatingBinaryAAX;

  defm vsmul : RISCVSaturatingBinaryAAX;

  defm vssrl : RISCVSaturatingBinaryAAShift;
  defm vssra : RISCVSaturatingBinaryAAShift;

  defm vnclipu : RISCVSaturatingBinaryABShift;
  defm vnclip : RISCVSaturatingBinaryABShift;

  defm vmfeq : RISCVCompare;
  defm vmfne : RISCVCompare;
  defm vmflt : RISCVCompare;
  defm vmfle : RISCVCompare;
  defm vmfgt : RISCVCompare;
  defm vmfge : RISCVCompare;

  defm vredsum : RISCVReduction;
  defm vredand : RISCVReduction;
  defm vredor : RISCVReduction;
  defm vredxor : RISCVReduction;
  defm vredminu : RISCVReduction;
  defm vredmin : RISCVReduction;
  defm vredmaxu : RISCVReduction;
  defm vredmax : RISCVReduction;

  defm vwredsumu : RISCVReduction;
  defm vwredsum : RISCVReduction;

  defm vfredosum : RISCVReduction;
  defm vfredusum : RISCVReduction;
  defm vfredmin : RISCVReduction;
  defm vfredmax : RISCVReduction;

  defm vfwredusum : RISCVReduction;
  defm vfwredosum : RISCVReduction;

  def int_riscv_vmand: RISCVBinaryAAAUnMasked;
  def int_riscv_vmnand: RISCVBinaryAAAUnMasked;
  def int_riscv_vmandn: RISCVBinaryAAAUnMasked;
  def int_riscv_vmxor: RISCVBinaryAAAUnMasked;
  def int_riscv_vmor: RISCVBinaryAAAUnMasked;
  def int_riscv_vmnor: RISCVBinaryAAAUnMasked;
  def int_riscv_vmorn: RISCVBinaryAAAUnMasked;
  def int_riscv_vmxnor: RISCVBinaryAAAUnMasked;
  def int_riscv_vmclr : RISCVNullaryIntrinsic;
  def int_riscv_vmset : RISCVNullaryIntrinsic;

  defm vcpop : RISCVMaskedUnarySOut;
  defm vfirst : RISCVMaskedUnarySOut;
  defm vmsbf : RISCVMaskedUnaryMOut;
  defm vmsof : RISCVMaskedUnaryMOut;
  defm vmsif : RISCVMaskedUnaryMOut;

  defm vfcvt_xu_f_v : RISCVConversion;
  defm vfcvt_x_f_v : RISCVConversion;
  defm vfcvt_rtz_xu_f_v : RISCVConversion;
  defm vfcvt_rtz_x_f_v : RISCVConversion;
  defm vfcvt_f_xu_v : RISCVConversion;
  defm vfcvt_f_x_v : RISCVConversion;

  defm vfwcvt_f_xu_v : RISCVConversion;
  defm vfwcvt_f_x_v : RISCVConversion;
  defm vfwcvt_xu_f_v : RISCVConversion;
  defm vfwcvt_x_f_v : RISCVConversion;
  defm vfwcvt_rtz_xu_f_v : RISCVConversion;
  defm vfwcvt_rtz_x_f_v : RISCVConversion;
  defm vfwcvt_f_f_v : RISCVConversion;

  defm vfncvt_f_xu_w : RISCVConversion;
  defm vfncvt_f_x_w : RISCVConversion;
  defm vfncvt_xu_f_w : RISCVConversion;
  defm vfncvt_x_f_w : RISCVConversion;
  defm vfncvt_rtz_xu_f_w : RISCVConversion;
  defm vfncvt_rtz_x_f_w : RISCVConversion;
  defm vfncvt_f_f_w : RISCVConversion;
  defm vfncvt_rod_f_f_w : RISCVConversion;

  // Output: (vector)
  // Input: (passthru, mask type input, vl)
  def int_riscv_viota
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                                [LLVMMatchType<0>,
                                 LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                                 llvm_anyint_ty],
                                [IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 2;
  }
  // Output: (vector)
  // Input: (maskedoff, mask type vector_in, mask, vl, policy)
  def int_riscv_viota_mask
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                                [LLVMMatchType<0>,
                                 LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                                 LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                                 llvm_anyint_ty, LLVMMatchType<1>],
                                [ImmArg<ArgIndex<4>>, IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 3;
  }
  // Output: (vector)
  // Input: (passthru, vl)
  def int_riscv_vid : RISCVID;

  // Output: (vector)
  // Input: (maskedoff, mask, vl, policy)
  def int_riscv_vid_mask
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                                [LLVMMatchType<0>,
                                 LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                                 llvm_anyint_ty, LLVMMatchType<1>],
                                [ImmArg<ArgIndex<3>>, IntrNoMem]>, RISCVVIntrinsic {
    let VLOperand = 2;
  }

  foreach nf = [2, 3, 4, 5, 6, 7, 8] in {
    defm vlseg # nf : RISCVUSSegLoad<nf>;
    defm vlseg # nf # ff : RISCVUSSegLoadFF<nf>;
    defm vlsseg # nf : RISCVSSegLoad<nf>;
    defm vloxseg # nf : RISCVISegLoad<nf>;
    defm vluxseg # nf : RISCVISegLoad<nf>;
    defm vsseg # nf : RISCVUSSegStore<nf>;
    defm vssseg # nf : RISCVSSegStore<nf>;
    defm vsoxseg # nf : RISCVISegStore<nf>;
    defm vsuxseg # nf : RISCVISegStore<nf>;
  }

  // Strided loads/stores for fixed vectors.
  def int_riscv_masked_strided_load
        : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                                [LLVMMatchType<0>, llvm_anyptr_ty,
                                 llvm_anyint_ty,
                                 LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>],
                                [NoCapture<ArgIndex<1>>, IntrReadMem]>;
  def int_riscv_masked_strided_store
        : DefaultAttrsIntrinsic<[],
                                [llvm_anyvector_ty, llvm_anyptr_ty,
                                 llvm_anyint_ty,
                                 LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>],
                                [NoCapture<ArgIndex<1>>, IntrWriteMem]>;

  // Segment loads for fixed vectors.
  foreach nf = [2, 3, 4, 5, 6, 7, 8] in {
    def int_riscv_seg # nf # _load
          : DefaultAttrsIntrinsic<!listconcat([llvm_anyvector_ty],
                                              !listsplat(LLVMMatchType<0>,
                                              !add(nf, -1))),
                                  [llvm_anyptr_ty, llvm_anyint_ty],
                                  [NoCapture<ArgIndex<0>>, IntrReadMem]>;
  }

} // TargetPrefix = "riscv"

//===----------------------------------------------------------------------===//
// Scalar Cryptography
//
// These intrinsics will lower directly into the corresponding instructions
// added by the scalar cyptography extension, if the extension is present.

let TargetPrefix = "riscv" in {

class ScalarCryptoGprIntrinsicAny
    : DefaultAttrsIntrinsic<[llvm_anyint_ty],
                [LLVMMatchType<0>],
                [IntrNoMem, IntrSpeculatable]>;

class ScalarCryptoByteSelect32
    : DefaultAttrsIntrinsic<[llvm_i32_ty],
                            [llvm_i32_ty, llvm_i32_ty, llvm_i8_ty],
                            [IntrNoMem, IntrSpeculatable,
                             ImmArg<ArgIndex<2>>]>;

class ScalarCryptoGprGprIntrinsic32
    : DefaultAttrsIntrinsic<[llvm_i32_ty],
                            [llvm_i32_ty, llvm_i32_ty],
                            [IntrNoMem, IntrSpeculatable]>;

class ScalarCryptoGprGprIntrinsic64
    : DefaultAttrsIntrinsic<[llvm_i64_ty],
                            [llvm_i64_ty, llvm_i64_ty],
                            [IntrNoMem, IntrSpeculatable]>;

class ScalarCryptoGprIntrinsic64
    : DefaultAttrsIntrinsic<[llvm_i64_ty],
                            [llvm_i64_ty],
                            [IntrNoMem, IntrSpeculatable]>;

class ScalarCryptoByteSelectAny
    : DefaultAttrsIntrinsic<[llvm_anyint_ty],
                            [LLVMMatchType<0>, LLVMMatchType<0>, llvm_i8_ty],
                            [IntrNoMem, IntrSpeculatable, ImmArg<ArgIndex<2>>]>;

// Zknd
def int_riscv_aes32dsi  : ScalarCryptoByteSelect32;
def int_riscv_aes32dsmi : ScalarCryptoByteSelect32;

def int_riscv_aes64ds   : ScalarCryptoGprGprIntrinsic64;
def int_riscv_aes64dsm  : ScalarCryptoGprGprIntrinsic64;

def int_riscv_aes64im   : ScalarCryptoGprIntrinsic64;

// Zkne
def int_riscv_aes32esi  : ScalarCryptoByteSelect32;
def int_riscv_aes32esmi : ScalarCryptoByteSelect32;

def int_riscv_aes64es   : ScalarCryptoGprGprIntrinsic64;
def int_riscv_aes64esm  : ScalarCryptoGprGprIntrinsic64;

// Zknd & Zkne
def int_riscv_aes64ks2  : ScalarCryptoGprGprIntrinsic64;
def int_riscv_aes64ks1i : DefaultAttrsIntrinsic<[llvm_i64_ty],
                                                [llvm_i64_ty, llvm_i32_ty],
                                                [IntrNoMem, IntrSpeculatable,
                                                 ImmArg<ArgIndex<1>>]>;

// Zknh
def int_riscv_sha256sig0 : ScalarCryptoGprIntrinsicAny;
def int_riscv_sha256sig1 : ScalarCryptoGprIntrinsicAny;
def int_riscv_sha256sum0 : ScalarCryptoGprIntrinsicAny;
def int_riscv_sha256sum1 : ScalarCryptoGprIntrinsicAny;

def int_riscv_sha512sig0l : ScalarCryptoGprGprIntrinsic32;
def int_riscv_sha512sig0h : ScalarCryptoGprGprIntrinsic32;
def int_riscv_sha512sig1l : ScalarCryptoGprGprIntrinsic32;
def int_riscv_sha512sig1h : ScalarCryptoGprGprIntrinsic32;
def int_riscv_sha512sum0r : ScalarCryptoGprGprIntrinsic32;
def int_riscv_sha512sum1r : ScalarCryptoGprGprIntrinsic32;

def int_riscv_sha512sig0 : ScalarCryptoGprIntrinsic64;
def int_riscv_sha512sig1 : ScalarCryptoGprIntrinsic64;
def int_riscv_sha512sum0 : ScalarCryptoGprIntrinsic64;
def int_riscv_sha512sum1 : ScalarCryptoGprIntrinsic64;

// Zksed
def int_riscv_sm4ks      : ScalarCryptoByteSelectAny;
def int_riscv_sm4ed      : ScalarCryptoByteSelectAny;

// Zksh
def int_riscv_sm3p0      : ScalarCryptoGprIntrinsicAny;
def int_riscv_sm3p1      : ScalarCryptoGprIntrinsicAny;
} // TargetPrefix = "riscv"

//===----------------------------------------------------------------------===//
// Vendor extensions
//===----------------------------------------------------------------------===//
include "llvm/IR/IntrinsicsRISCVXTHead.td"
