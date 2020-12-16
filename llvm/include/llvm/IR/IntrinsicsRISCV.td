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
                    llvm_anyvector_ty, llvm_anyint_ty],
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
                     llvm_anyvector_ty, llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic {
    let ExtendOperand = 3;
  }
  // For binary operations with V0 as input.
  // Input: (vector_in, vector_in/scalar_in, V0, vl)
  class RISCVBinaryWithV0
        : Intrinsic<[llvm_anyvector_ty],
                    [LLVMMatchType<0>, llvm_any_ty, llvm_anyvector_ty,
                     llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic {
    let ExtendOperand = 2;
  }
  // For binary operations with mask type output and V0 as input.
  // Output: (mask type output)
  // Input: (vector_in, vector_in/scalar_in, V0, vl)
  class RISCVBinaryMOutWithV0
        :Intrinsic<[llvm_anyvector_ty],
                   [llvm_anyvector_ty, llvm_any_ty, LLVMMatchType<0>,
                    llvm_anyint_ty],
                   [IntrNoMem]>, RISCVVIntrinsic {
    let ExtendOperand = 2;
  }
  // For binary operations with mask type output.
  // Output: (mask type output)
  // Input: (vector_in, vector_in/scalar_in, vl)
  class RISCVBinaryMOut
        : Intrinsic<[llvm_anyvector_ty],
                    [llvm_anyvector_ty, llvm_any_ty, llvm_anyint_ty],
                    [IntrNoMem]>, RISCVVIntrinsic {
    let ExtendOperand = 2;
  }

  multiclass RISCVBinaryAAX {
    def "int_riscv_" # NAME : RISCVBinaryAAXNoMask;
    def "int_riscv_" # NAME # "_mask" : RISCVBinaryAAXMask;
  }
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

  defm vadc : RISCVBinaryWithV0;
  defm vmadc_carry_in : RISCVBinaryMaskOutWithV0;
  defm vmadc : RISCVBinaryMaskOut;

  defm vsbc : RISCVBinaryWithV0;
  defm vmsbc_borrow_in : RISCVBinaryMaskOutWithV0;
  defm vmsbc : RISCVBinaryMaskOut;

  defm vsll : RISCVBinaryAAX;
  defm vsrl : RISCVBinaryAAX;
  defm vsra : RISCVBinaryAAX;

  defm vnsrl : RISCVBinaryABX;
  defm vnsra : RISCVBinaryABX;

} // TargetPrefix = "riscv"
