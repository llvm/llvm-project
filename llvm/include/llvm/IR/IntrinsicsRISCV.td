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

  // @llvm.riscv.masked.atomicrmw.*.{i32,i64}.<p>(...)
  defm int_riscv_masked_atomicrmw_xchg : MaskedAtomicRMWFourArgIntrinsics;
  defm int_riscv_masked_atomicrmw_add : MaskedAtomicRMWFourArgIntrinsics;
  defm int_riscv_masked_atomicrmw_sub : MaskedAtomicRMWFourArgIntrinsics;
  defm int_riscv_masked_atomicrmw_nand : MaskedAtomicRMWFourArgIntrinsics;
  // Signed min and max need an extra operand to do sign extension with.
  defm int_riscv_masked_atomicrmw_max : MaskedAtomicRMWFiveArgIntrinsics;
  defm int_riscv_masked_atomicrmw_min : MaskedAtomicRMWFiveArgIntrinsics;
  // Unsigned min and max don't need the extra operand.
  defm int_riscv_masked_atomicrmw_umax : MaskedAtomicRMWFourArgIntrinsics;
  defm int_riscv_masked_atomicrmw_umin : MaskedAtomicRMWFourArgIntrinsics;

  // @llvm.riscv.masked.cmpxchg.{i32,i64}.<p>(...)
  defm int_riscv_masked_cmpxchg : MaskedAtomicRMWFiveArgIntrinsics;

} // TargetPrefix = "riscv"

//===----------------------------------------------------------------------===//
// Vectors

class RISCVVIntrinsic {
  // These intrinsics may accept illegal integer values in their llvm_any_ty
  // operand, so they have to be extended. If set to zero then the intrinsic
  // does not have any operand that must be extended.
  Intrinsic IntrinsicID = !cast<Intrinsic>(NAME);
  bits<4> ExtendOperand = 0;
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

  // For unit stride load
  // Input: (pointer, vl)
  class RISCVUSLoad
        : Intrinsic<[llvm_anyvector_ty],
                    [LLVMPointerType<LLVMMatchType<0>>,
                     llvm_anyint_ty],
                    [NoCapture<ArgIndex<0>>, IntrReadMem]>, RISCVVIntrinsic;
  // For unit stride load with mask
  // Input: (maskedoff, pointer, mask, vl)
  class RISCVUSLoadMask
        : Intrinsic<[llvm_anyvector_ty ],
                    [LLVMMatchType<0>,
                     LLVMPointerType<LLVMMatchType<0>>,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                     llvm_anyint_ty],
                    [NoCapture<ArgIndex<1>>, IntrReadMem]>, RISCVVIntrinsic;
  // For strided load
  // Input: (pointer, stride, vl)
  class RISCVSLoad
        : Intrinsic<[llvm_anyvector_ty],
                    [LLVMPointerType<LLVMMatchType<0>>,
                     llvm_anyint_ty, LLVMMatchType<1>],
                    [NoCapture<ArgIndex<0>>, IntrReadMem]>, RISCVVIntrinsic;
  // For strided load with mask
  // Input: (maskedoff, pointer, stride, mask, vl)
  class RISCVSLoadMask
        : Intrinsic<[llvm_anyvector_ty ],
                    [LLVMMatchType<0>,
                     LLVMPointerType<LLVMMatchType<0>>, llvm_anyint_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, LLVMMatchType<1>],
                    [NoCapture<ArgIndex<1>>, IntrReadMem]>, RISCVVIntrinsic;
  // For indexed load
  // Input: (pointer, index, vl)
  class RISCVILoad
        : Intrinsic<[llvm_anyvector_ty],
                    [LLVMPointerType<LLVMMatchType<0>>,
                     llvm_anyvector_ty, llvm_anyint_ty],
                    [NoCapture<ArgIndex<0>>, IntrReadMem]>, RISCVVIntrinsic;
  // For indexed load with mask
  // Input: (maskedoff, pointer, index, mask, vl)
  class RISCVILoadMask
        : Intrinsic<[llvm_anyvector_ty ],
                    [LLVMMatchType<0>,
                     LLVMPointerType<LLVMMatchType<0>>, llvm_anyvector_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty],
                    [NoCapture<ArgIndex<1>>, IntrReadMem]>, RISCVVIntrinsic;
  // For unit stride store
  // Input: (vector_in, pointer, vl)
  class RISCVUSStore
        : Intrinsic<[],
                    [llvm_anyvector_ty,
                     LLVMPointerType<LLVMMatchType<0>>,
                     llvm_anyint_ty],
                    [NoCapture<ArgIndex<1>>, IntrWriteMem]>, RISCVVIntrinsic;
  // For unit stride store with mask
  // Input: (vector_in, pointer, mask, vl)
  class RISCVUSStoreMask
        : Intrinsic<[],
                    [llvm_anyvector_ty,
                     LLVMPointerType<LLVMMatchType<0>>,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                     llvm_anyint_ty],
                    [NoCapture<ArgIndex<1>>, IntrWriteMem]>, RISCVVIntrinsic;
  // For strided store
  // Input: (vector_in, pointer, stride, vl)
  class RISCVSStore
        : Intrinsic<[],
                    [llvm_anyvector_ty,
                     LLVMPointerType<LLVMMatchType<0>>,
                     llvm_anyint_ty, LLVMMatchType<1>],
                    [NoCapture<ArgIndex<1>>, IntrWriteMem]>, RISCVVIntrinsic;
  // For stride store with mask
  // Input: (vector_in, pointer, stirde, mask, vl)
  class RISCVSStoreMask
        : Intrinsic<[],
                    [llvm_anyvector_ty,
                     LLVMPointerType<LLVMMatchType<0>>, llvm_anyint_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, LLVMMatchType<1>],
                    [NoCapture<ArgIndex<1>>, IntrWriteMem]>, RISCVVIntrinsic;
  // For indexed store
  // Input: (vector_in, pointer, index, vl)
  class RISCVIStore
        : Intrinsic<[],
                    [llvm_anyvector_ty,
                     LLVMPointerType<LLVMMatchType<0>>,
                     llvm_anyint_ty, llvm_anyint_ty],
                    [NoCapture<ArgIndex<1>>, IntrWriteMem]>, RISCVVIntrinsic;
  // For indexed store with mask
  // Input: (vector_in, pointer, index, mask, vl)
  class RISCVIStoreMask
        : Intrinsic<[],
                    [llvm_anyvector_ty,
                     LLVMPointerType<LLVMMatchType<0>>, llvm_anyvector_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty],
                    [NoCapture<ArgIndex<1>>, IntrWriteMem]>, RISCVVIntrinsic;
  // For destination vector type is the same as source vector.
  // Input: (vector_in, vl)
  class RISCVUnaryAANoMask
        : Intrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic;
  // For destination vector type is the same as first source vector (with mask).
  // Input: (vector_in, mask, vl)
  class RISCVUnaryAAMask
        : Intrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic;
  // For destination vector type is the same as first and second source vector.
  // Input: (vector_in, vector_in, vl)
  class RISCVBinaryAAANoMask
        : Intrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>, llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic;
  // For destination vector type is the same as first and second source vector.
  // Input: (vector_in, vector_in, vl)
  class RISCVBinaryAAAMask
        : Intrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic;
  // For destination vector type is the same as first source vector.
  // Input: (vector_in, vector_in/scalar_in, vl)
  class RISCVBinaryAAXNoMask
        : Intrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_any_ty, llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic {
    let ExtendOperand = 2;
  }
  // For destination vector type is the same as first source vector (with mask).
  // Input: (maskedoff, vector_in, vector_in/scalar_in, mask, vl)
  class RISCVBinaryAAXMask
       : Intrinsic<[llvm_anyvector_ty],
                   [LLVMMatchType<0>, LLVMMatchType<0>, llvm_any_ty,
                    LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty],
                   [IntrNoMem]>, RISCVVIntrinsic {
    let ExtendOperand = 3;
  }
  // For destination vector type is NOT the same as first source vector.
  // Input: (vector_in, vector_in/scalar_in, vl)
  class RISCVBinaryABXNoMask
        : Intrinsic<[llvm_anyvector_ty],
                    [llvm_anyvector_ty, llvm_any_ty, llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic {
    let ExtendOperand = 2;
  }
  // For destination vector type is NOT the same as first source vector (with mask).
  // Input: (maskedoff, vector_in, vector_in/scalar_in, mask, vl)
  class RISCVBinaryABXMask
        : Intrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyvector_ty, llvm_any_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic {
    let ExtendOperand = 3;
  }
  // For binary operations with V0 as input.
  // Input: (vector_in, vector_in/scalar_in, V0, vl)
  class RISCVBinaryWithV0
        : Intrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_any_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                     llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic {
    let ExtendOperand = 2;
  }
  // For binary operations with mask type output and V0 as input.
  // Output: (mask type output)
  // Input: (vector_in, vector_in/scalar_in, V0, vl)
  class RISCVBinaryMOutWithV0
        :Intrinsic<[LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>],
                   [llvm_anyvector_ty, llvm_any_ty,
                    LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                    llvm_anyint_ty],
                   [IntrNoMem]>, RISCVVIntrinsic {
    let ExtendOperand = 2;
  }
  // For binary operations with mask type output.
  // Output: (mask type output)
  // Input: (vector_in, vector_in/scalar_in, vl)
  class RISCVBinaryMOut
        : Intrinsic<[LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>],
                    [llvm_anyvector_ty, llvm_any_ty, llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic {
    let ExtendOperand = 2;
  }
  // For binary operations with mask type output without mask.
  // Output: (mask type output)
  // Input: (vector_in, vector_in/scalar_in, vl)
  class RISCVCompareNoMask
        : Intrinsic<[LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>],
                    [llvm_anyvector_ty, llvm_any_ty, llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic {
    let ExtendOperand = 2;
  }
  // For binary operations with mask type output with mask.
  // Output: (mask type output)
  // Input: (maskedoff, vector_in, vector_in/scalar_in, mask, vl)
  class RISCVCompareMask
        : Intrinsic<[LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>],
                    [LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                     llvm_anyvector_ty, llvm_any_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic {
    let ExtendOperand = 3;
  }
  // For FP classify operations.
  // Output: (bit mask type output)
  // Input: (vector_in, vl)
  class RISCVClassifyNoMask
        : Intrinsic<[LLVMVectorOfBitcastsToInt<0>],
                    [llvm_anyvector_ty, llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic;
  // For FP classify operations with mask.
  // Output: (bit mask type output)
  // Input: (maskedoff, vector_in, mask, vl)
  class RISCVClassifyMask
        : Intrinsic<[LLVMVectorOfBitcastsToInt<0>],
                    [LLVMVectorOfBitcastsToInt<0>, llvm_anyvector_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic;
  // For Saturating binary operations.
  // The destination vector type is the same as first source vector.
  // Input: (vector_in, vector_in/scalar_in, vl)
  class RISCVSaturatingBinaryAAXNoMask
        : Intrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_any_ty, llvm_anyint_ty],
                    [IntrNoMem, IntrHasSideEffects]>, RISCVVIntrinsic {
    let ExtendOperand = 2;
  }
  // For Saturating binary operations with mask.
  // The destination vector type is the same as first source vector.
  // Input: (maskedoff, vector_in, vector_in/scalar_in, mask, vl)
  class RISCVSaturatingBinaryAAXMask
        : Intrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>, llvm_any_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty],
                    [IntrNoMem, IntrHasSideEffects]>, RISCVVIntrinsic {
    let ExtendOperand = 3;
  }
  // For Saturating binary operations.
  // The destination vector type is NOT the same as first source vector.
  // Input: (vector_in, vector_in/scalar_in, vl)
  class RISCVSaturatingBinaryABXNoMask
        : Intrinsic<[llvm_anyvector_ty],
                    [llvm_anyvector_ty, llvm_any_ty, llvm_anyint_ty],
                    [IntrNoMem, IntrHasSideEffects]>, RISCVVIntrinsic {
    let ExtendOperand = 2;
  }
  // For Saturating binary operations with mask.
  // The destination vector type is NOT the same as first source vector (with mask).
  // Input: (maskedoff, vector_in, vector_in/scalar_in, mask, vl)
  class RISCVSaturatingBinaryABXMask
        : Intrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyvector_ty, llvm_any_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty],
                    [IntrNoMem, IntrHasSideEffects]>, RISCVVIntrinsic {
    let ExtendOperand = 3;
  }
  class RISCVTernaryAAAXNoMask
        : Intrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>, llvm_anyint_ty,
                     LLVMMatchType<1>],
                    [IntrNoMem]>, RISCVVIntrinsic;
  class RISCVTernaryAAAXMask
        : Intrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>, llvm_anyint_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, LLVMMatchType<1>],
                    [IntrNoMem]>, RISCVVIntrinsic;
  class RISCVTernaryAAXANoMask
        : Intrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_any_ty, LLVMMatchType<0>,
                     llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic {
    let ExtendOperand = 2;
  }
  class RISCVTernaryAAXAMask
        : Intrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_any_ty, LLVMMatchType<0>,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic {
    let ExtendOperand = 2;
  }
  class RISCVTernaryWideNoMask
        : Intrinsic< [llvm_anyvector_ty],
                     [LLVMMatchType<0>, llvm_any_ty, llvm_anyvector_ty,
                      llvm_anyint_ty],
                     [IntrNoMem] >, RISCVVIntrinsic {
    let ExtendOperand = 2;
  }
  class RISCVTernaryWideMask
        : Intrinsic< [llvm_anyvector_ty],
                     [LLVMMatchType<0>, llvm_any_ty, llvm_anyvector_ty,
                      LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty],
                     [IntrNoMem]>, RISCVVIntrinsic {
    let ExtendOperand = 2;
  }
  // For Reduction ternary operations.
  // For destination vector type is the same as first and third source vector.
  // Input: (vector_in, vector_in, vector_in, vl)
  class RISCVReductionNoMask
        : Intrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyvector_ty, LLVMMatchType<0>,
                     llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic;
  // For Reduction ternary operations with mask.
  // For destination vector type is the same as first and third source vector.
  // The mask type come from second source vector.
  // Input: (maskedoff, vector_in, vector_in, vector_in, mask, vl)
  class RISCVReductionMask
        : Intrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyvector_ty, LLVMMatchType<0>,
                     LLVMScalarOrSameVectorWidth<1, llvm_i1_ty>, llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic;
  // For unary operations with scalar type output without mask
  // Output: (scalar type)
  // Input: (vector_in, vl)
  class RISCVMaskUnarySOutNoMask
        : Intrinsic<[llvm_anyint_ty],
                    [llvm_anyvector_ty, LLVMMatchType<0>],
                    [IntrNoMem]>, RISCVVIntrinsic;
  // For unary operations with scalar type output with mask
  // Output: (scalar type)
  // Input: (vector_in, mask, vl)
  class RISCVMaskUnarySOutMask
        : Intrinsic<[llvm_anyint_ty],
                    [llvm_anyvector_ty, LLVMMatchType<1>, LLVMMatchType<0>],
                    [IntrNoMem]>, RISCVVIntrinsic;
  // For destination vector type is NOT the same as source vector.
  // Input: (vector_in, vl)
  class RISCVUnaryABNoMask
        : Intrinsic<[llvm_anyvector_ty],
                    [llvm_anyvector_ty, llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic;
  // For destination vector type is NOT the same as source vector (with mask).
  // Input: (maskedoff, vector_in, mask, vl)
  class RISCVUnaryABMask
        : Intrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyvector_ty,
                     LLVMScalarOrSameVectorWidth<1, llvm_i1_ty>,
                     llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic;
  // For unary operations with the same vector type in/out without mask
  // Output: (vector)
  // Input: (vector_in, vl)
  class RISCVUnaryNoMask
        : Intrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic;
  // For mask unary operations with mask type in/out with mask
  // Output: (mask type output)
  // Input: (mask type maskedoff, mask type vector_in, mask, vl)
  class RISCVMaskUnaryMOutMask
        : Intrinsic<[llvm_anyint_ty],
                    [LLVMMatchType<0>, LLVMMatchType<0>,
                     LLVMMatchType<0>, llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic;
  // Output: (vector)
  // Input: (vl)
  class RISCVNullaryIntrinsic
        : Intrinsic<[llvm_anyvector_ty],
                    [llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic;
  // For Conversion unary operations.
  // Input: (vector_in, vl)
  class RISCVConversionNoMask
        : Intrinsic<[llvm_anyvector_ty],
                    [llvm_anyvector_ty, llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic;
  // For Conversion unary operations with mask.
  // Input: (maskedoff, vector_in, mask, vl)
  class RISCVConversionMask
        : Intrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_anyvector_ty,
                     LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic;

  multiclass RISCVUSLoad {
    def "int_riscv_" # NAME : RISCVUSLoad;
    def "int_riscv_" # NAME # "_mask" : RISCVUSLoadMask;
  }
  multiclass RISCVSLoad {
    def "int_riscv_" # NAME : RISCVSLoad;
    def "int_riscv_" # NAME # "_mask" : RISCVSLoadMask;
  }
  multiclass RISCVILoad {
    def "int_riscv_" # NAME : RISCVILoad;
    def "int_riscv_" # NAME # "_mask" : RISCVILoadMask;
  }
  multiclass RISCVUSStore {
    def "int_riscv_" # NAME : RISCVUSStore;
    def "int_riscv_" # NAME # "_mask" : RISCVUSStoreMask;
  }
  multiclass RISCVSStore {
    def "int_riscv_" # NAME : RISCVSStore;
    def "int_riscv_" # NAME # "_mask" : RISCVSStoreMask;
  }

  multiclass RISCVIStore {
    def "int_riscv_" # NAME : RISCVIStore;
    def "int_riscv_" # NAME # "_mask" : RISCVIStoreMask;
  }
  multiclass RISCVUnaryAA {
    def "int_riscv_" # NAME : RISCVUnaryAANoMask;
    def "int_riscv_" # NAME # "_mask" : RISCVUnaryAAMask;
  }
  multiclass RISCVUnaryAB {
    def "int_riscv_" # NAME : RISCVUnaryABNoMask;
    def "int_riscv_" # NAME # "_mask" : RISCVUnaryABMask;
  }
  // AAX means the destination type(A) is the same as the first source
  // type(A). X means any type for the second source operand.
  multiclass RISCVBinaryAAX {
    def "int_riscv_" # NAME : RISCVBinaryAAXNoMask;
    def "int_riscv_" # NAME # "_mask" : RISCVBinaryAAXMask;
  }
  // ABX means the destination type(A) is different from the first source
  // type(B). X means any type for the second source operand.
  multiclass RISCVBinaryABX {
    def "int_riscv_" # NAME : RISCVBinaryABXNoMask;
    def "int_riscv_" # NAME # "_mask" : RISCVBinaryABXMask;
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
    def "int_riscv_" # NAME : RISCVSaturatingBinaryAAXNoMask;
    def "int_riscv_" # NAME # "_mask" : RISCVSaturatingBinaryAAXMask;
  }
  multiclass RISCVSaturatingBinaryABX {
    def "int_riscv_" # NAME : RISCVSaturatingBinaryABXNoMask;
    def "int_riscv_" # NAME # "_mask" : RISCVSaturatingBinaryABXMask;
  }
  multiclass RISCVTernaryAAAX {
    def "int_riscv_" # NAME : RISCVTernaryAAAXNoMask;
    def "int_riscv_" # NAME # "_mask" : RISCVTernaryAAAXMask;
  }
  multiclass RISCVTernaryAAXA {
    def "int_riscv_" # NAME : RISCVTernaryAAXANoMask;
    def "int_riscv_" # NAME # "_mask" : RISCVTernaryAAXAMask;
  }
  multiclass RISCVCompare {
    def "int_riscv_" # NAME : RISCVCompareNoMask;
    def "int_riscv_" # NAME # "_mask" : RISCVCompareMask;
  }
  multiclass RISCVClassify {
    def "int_riscv_" # NAME : RISCVClassifyNoMask;
    def "int_riscv_" # NAME # "_mask" : RISCVClassifyMask;
  }
  multiclass RISCVTernaryWide {
    def "int_riscv_" # NAME : RISCVTernaryWideNoMask;
    def "int_riscv_" # NAME # "_mask" : RISCVTernaryWideMask;
  }
  multiclass RISCVReduction {
    def "int_riscv_" # NAME : RISCVReductionNoMask;
    def "int_riscv_" # NAME # "_mask" : RISCVReductionMask;
  }
  multiclass RISCVMaskUnarySOut {
    def "int_riscv_" # NAME : RISCVMaskUnarySOutNoMask;
    def "int_riscv_" # NAME # "_mask" : RISCVMaskUnarySOutMask;
  }
  multiclass RISCVMaskUnaryMOut {
    def "int_riscv_" # NAME : RISCVUnaryNoMask;
    def "int_riscv_" # NAME # "_mask" : RISCVMaskUnaryMOutMask;
  }
  multiclass RISCVConversion {
    def "int_riscv_" #NAME :RISCVConversionNoMask;
    def "int_riscv_" # NAME # "_mask" : RISCVConversionMask;
  }

  defm vle : RISCVUSLoad;
  defm vleff : RISCVUSLoad;
  defm vse : RISCVUSStore;
  defm vlse: RISCVSLoad;
  defm vsse: RISCVSStore;
  defm vlxe: RISCVILoad;
  defm vsxe: RISCVIStore;
  defm vsuxe: RISCVIStore;

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

  defm vsll : RISCVBinaryAAX;
  defm vsrl : RISCVBinaryAAX;
  defm vsra : RISCVBinaryAAX;

  defm vnsrl : RISCVBinaryABX;
  defm vnsra : RISCVBinaryABX;

  defm vmseq : RISCVCompare;
  defm vmsne : RISCVCompare;
  defm vmsltu : RISCVCompare;
  defm vmslt : RISCVCompare;
  defm vmsleu : RISCVCompare;
  defm vmsle : RISCVCompare;
  defm vmsgtu : RISCVCompare;
  defm vmsgt : RISCVCompare;

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

  def int_riscv_vmerge : RISCVBinaryWithV0;

  def int_riscv_vmv_v_v : Intrinsic<[llvm_anyvector_ty],
                                    [LLVMMatchType<0>, llvm_anyint_ty],
                                    [IntrNoMem]>, RISCVVIntrinsic;
  def int_riscv_vmv_v_x : Intrinsic<[llvm_anyint_ty],
                                    [LLVMVectorElementType<0>, llvm_anyint_ty],
                                    [IntrNoMem]>, RISCVVIntrinsic {
    let ExtendOperand = 1;
  }
  def int_riscv_vfmv_v_f : Intrinsic<[llvm_anyfloat_ty],
                                     [LLVMVectorElementType<0>, llvm_anyint_ty],
                                     [IntrNoMem]>, RISCVVIntrinsic;

  def int_riscv_vmv_x_s : Intrinsic<[LLVMVectorElementType<0>],
                                    [llvm_anyint_ty],
                                    [IntrNoMem]>, RISCVVIntrinsic;
  def int_riscv_vmv_s_x : Intrinsic<[llvm_anyint_ty],
                                    [LLVMMatchType<0>, LLVMVectorElementType<0>,
                                     llvm_anyint_ty],
                                    [IntrNoMem]>, RISCVVIntrinsic {
    let ExtendOperand = 2;
  }

  def int_riscv_vfmv_f_s : Intrinsic<[LLVMVectorElementType<0>],
                                     [llvm_anyfloat_ty],
                                     [IntrNoMem]>, RISCVVIntrinsic;
  def int_riscv_vfmv_s_f : Intrinsic<[llvm_anyfloat_ty],
                                     [LLVMMatchType<0>, LLVMVectorElementType<0>,
                                      llvm_anyint_ty],
                                     [IntrNoMem]>, RISCVVIntrinsic;

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

  defm vfmin : RISCVBinaryAAX;
  defm vfmax : RISCVBinaryAAX;

  defm vfsgnj : RISCVBinaryAAX;
  defm vfsgnjn : RISCVBinaryAAX;
  defm vfsgnjx : RISCVBinaryAAX;

  defm vfclass : RISCVClassify;

  defm vfmerge : RISCVBinaryWithV0;

  defm vslideup : RISCVTernaryAAAX;
  defm vslidedown : RISCVTernaryAAAX;

  defm vslide1up : RISCVBinaryAAX;
  defm vslide1down : RISCVBinaryAAX;
  defm vfslide1up : RISCVBinaryAAX;
  defm vfslide1down : RISCVBinaryAAX;

  defm vrgather : RISCVBinaryAAX;

  def "int_riscv_vcompress" : RISCVBinaryAAAMask;

  defm vaaddu : RISCVSaturatingBinaryAAX;
  defm vaadd : RISCVSaturatingBinaryAAX;
  defm vasubu : RISCVSaturatingBinaryAAX;
  defm vasub : RISCVSaturatingBinaryAAX;

  defm vsmul : RISCVSaturatingBinaryAAX;

  defm vssrl : RISCVSaturatingBinaryAAX;
  defm vssra : RISCVSaturatingBinaryAAX;

  defm vnclipu : RISCVSaturatingBinaryABX;
  defm vnclip : RISCVSaturatingBinaryABX;

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
  defm vfredsum : RISCVReduction;
  defm vfredmin : RISCVReduction;
  defm vfredmax : RISCVReduction;

  defm vfwredsum : RISCVReduction;
  defm vfwredosum : RISCVReduction;

  def int_riscv_vmand: RISCVBinaryAAANoMask;
  def int_riscv_vmnand: RISCVBinaryAAANoMask;
  def int_riscv_vmandnot: RISCVBinaryAAANoMask;
  def int_riscv_vmxor: RISCVBinaryAAANoMask;
  def int_riscv_vmor: RISCVBinaryAAANoMask;
  def int_riscv_vmnor: RISCVBinaryAAANoMask;
  def int_riscv_vmornot: RISCVBinaryAAANoMask;
  def int_riscv_vmxnor: RISCVBinaryAAANoMask;
  def int_riscv_vmclr : RISCVNullaryIntrinsic;
  def int_riscv_vmset : RISCVNullaryIntrinsic;

  defm vpopc : RISCVMaskUnarySOut;
  defm vfirst : RISCVMaskUnarySOut;
  defm vmsbf : RISCVMaskUnaryMOut;
  defm vmsof : RISCVMaskUnaryMOut;
  defm vmsif : RISCVMaskUnaryMOut;

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
  // Input: (mask type input, vl)
  def int_riscv_viota : Intrinsic<[llvm_anyvector_ty],
                                  [LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                                   llvm_anyint_ty],
                                  [IntrNoMem]>, RISCVVIntrinsic;
  // Output: (vector)
  // Input: (maskedoff, mask type vector_in, mask, vl)
  def int_riscv_viota_mask : Intrinsic<[llvm_anyvector_ty],
                                       [LLVMMatchType<0>,
                                        LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                                        LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                                        llvm_anyint_ty],
                                       [IntrNoMem]>, RISCVVIntrinsic;
  // Output: (vector)
  // Input: (vl)
  def int_riscv_vid : RISCVNullaryIntrinsic;

  // Output: (vector)
  // Input: (maskedoff, mask, vl)
  def int_riscv_vid_mask : Intrinsic<[llvm_anyvector_ty],
                                     [LLVMMatchType<0>,
                                      LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                                      llvm_anyint_ty],
                                     [IntrNoMem]>, RISCVVIntrinsic;

} // TargetPrefix = "riscv"
