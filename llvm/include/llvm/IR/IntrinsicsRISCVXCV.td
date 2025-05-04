//===- IntrinsicsRISCVXCV.td - CORE-V intrinsics -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the CORE-V vendor intrinsics for RISC-V.
//
//===----------------------------------------------------------------------===//

class ScalarCoreVBitManipGprGprIntrinsic
    : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
                            [IntrNoMem, IntrSpeculatable]>;

class ScalarCoreVBitManipGprIntrinsic
    : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty],
                            [IntrNoMem, IntrSpeculatable]>;

class ScalarCoreVAluGprIntrinsic
  : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty],
                          [IntrNoMem, IntrSpeculatable]>;

class ScalarCoreVAluGprGprIntrinsic
  : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem, IntrSpeculatable]>;

class ScalarCoreVAluGprGprGprIntrinsic
  : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem, IntrSpeculatable]>;

class ScalarCoreVMacGprGprGprIntrinsic
  : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [IntrNoMem, IntrWillReturn, IntrSpeculatable]>;

class ScalarCoreVMacGprGPRImmIntrinsic
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrWillReturn, IntrSpeculatable, ImmArg<ArgIndex<2>>]>;

class ScalarCoreVMacGprGprGprImmIntrinsic
  : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
              [IntrNoMem, IntrWillReturn, IntrSpeculatable, ImmArg<ArgIndex<3>>]>;

class ScalarCoreVSimdGprIntrinsic
    : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty],
                            [IntrNoMem, IntrSpeculatable]>;

class ScalarCoreVSimdGprGprIntrinsic
    : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
                            [IntrNoMem, IntrSpeculatable]>;

class ScalarCoreVSimdGprImmIntrinsic
    : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
                            [IntrNoMem, IntrSpeculatable, ImmArg<ArgIndex<1>>]>;

class ScalarCoreVSimdGprGprGprIntrinsic
    : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                            [IntrNoMem, IntrSpeculatable]>;

class ScalarCoreVSimdGprGprImmIntrinsic
    : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                            [IntrNoMem, IntrSpeculatable, ImmArg<ArgIndex<2>>]>;

class ScalarCoreVSimdGprGprGprImmIntrinsic
    : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                            [IntrNoMem, IntrSpeculatable, ImmArg<ArgIndex<3>>]>;

multiclass ScalarCoreVSimdGprIntrinsicHB {
  def int_riscv_cv_simd_ # NAME # _h : ScalarCoreVSimdGprIntrinsic;
  def int_riscv_cv_simd_ # NAME # _b : ScalarCoreVSimdGprIntrinsic;
}

multiclass ScalarCoreVSimdGprGprIntrinsicHB {
  def int_riscv_cv_simd_ # NAME # _h : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_ # NAME # _b : ScalarCoreVSimdGprGprIntrinsic;
}

multiclass ScalarCoreVSimdGprGprGprIntrinsicHB {
  def int_riscv_cv_simd_ # NAME # _h : ScalarCoreVSimdGprGprGprIntrinsic;
  def int_riscv_cv_simd_ # NAME # _b : ScalarCoreVSimdGprGprGprIntrinsic;
}

multiclass ScalarCoreVSimdGprImmIntrinsicHB {
  def int_riscv_cv_simd_ # NAME # _h : ScalarCoreVSimdGprImmIntrinsic;
  def int_riscv_cv_simd_ # NAME # _b : ScalarCoreVSimdGprImmIntrinsic;
}

multiclass CoreVSimdBinary <bit exclude_h = false> {
  if exclude_h then {
    def int_riscv_cv_simd_ # NAME # _b : ScalarCoreVSimdGprGprIntrinsic;
  } else {
    defm NAME : ScalarCoreVSimdGprGprIntrinsicHB;
  }
  defm NAME # _sc : ScalarCoreVSimdGprGprIntrinsicHB;
}

multiclass CoreVSimdTernary {
  defm NAME : ScalarCoreVSimdGprGprGprIntrinsicHB;
  defm NAME # _sc : ScalarCoreVSimdGprGprGprIntrinsicHB;
}

let TargetPrefix = "riscv" in {
  def int_riscv_cv_bitmanip_extract : ScalarCoreVBitManipGprGprIntrinsic;
  def int_riscv_cv_bitmanip_extractu : ScalarCoreVBitManipGprGprIntrinsic;
  def int_riscv_cv_bitmanip_bclr : ScalarCoreVBitManipGprGprIntrinsic;
  def int_riscv_cv_bitmanip_bset : ScalarCoreVBitManipGprGprIntrinsic;

  def int_riscv_cv_bitmanip_insert
    : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                            [IntrNoMem, IntrSpeculatable]>;

  def int_riscv_cv_bitmanip_clb : ScalarCoreVBitManipGprIntrinsic;

  def int_riscv_cv_bitmanip_bitrev
    : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                            [IntrNoMem, IntrWillReturn, IntrSpeculatable,
                            ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

  def int_riscv_cv_alu_clip   : ScalarCoreVAluGprGprIntrinsic;
  def int_riscv_cv_alu_clipu  : ScalarCoreVAluGprGprIntrinsic;
  def int_riscv_cv_alu_addN   : ScalarCoreVAluGprGprGprIntrinsic;
  def int_riscv_cv_alu_adduN  : ScalarCoreVAluGprGprGprIntrinsic;
  def int_riscv_cv_alu_addRN  : ScalarCoreVAluGprGprGprIntrinsic;
  def int_riscv_cv_alu_adduRN : ScalarCoreVAluGprGprGprIntrinsic;
  def int_riscv_cv_alu_subN   : ScalarCoreVAluGprGprGprIntrinsic;
  def int_riscv_cv_alu_subuN  : ScalarCoreVAluGprGprGprIntrinsic;
  def int_riscv_cv_alu_subRN  : ScalarCoreVAluGprGprGprIntrinsic;
  def int_riscv_cv_alu_subuRN : ScalarCoreVAluGprGprGprIntrinsic;

  def int_riscv_cv_mac_mac : ScalarCoreVMacGprGprGprIntrinsic;
  def int_riscv_cv_mac_msu : ScalarCoreVMacGprGprGprIntrinsic;

  def int_riscv_cv_mac_muluN    : ScalarCoreVMacGprGPRImmIntrinsic;
  def int_riscv_cv_mac_mulhhuN  : ScalarCoreVMacGprGPRImmIntrinsic;
  def int_riscv_cv_mac_mulsN    : ScalarCoreVMacGprGPRImmIntrinsic;
  def int_riscv_cv_mac_mulhhsN  : ScalarCoreVMacGprGPRImmIntrinsic;
  def int_riscv_cv_mac_muluRN   : ScalarCoreVMacGprGPRImmIntrinsic;
  def int_riscv_cv_mac_mulhhuRN : ScalarCoreVMacGprGPRImmIntrinsic;
  def int_riscv_cv_mac_mulsRN   : ScalarCoreVMacGprGPRImmIntrinsic;
  def int_riscv_cv_mac_mulhhsRN : ScalarCoreVMacGprGPRImmIntrinsic;

  def int_riscv_cv_mac_macuN    : ScalarCoreVMacGprGprGprImmIntrinsic;
  def int_riscv_cv_mac_machhuN  : ScalarCoreVMacGprGprGprImmIntrinsic;
  def int_riscv_cv_mac_macsN    : ScalarCoreVMacGprGprGprImmIntrinsic;
  def int_riscv_cv_mac_machhsN  : ScalarCoreVMacGprGprGprImmIntrinsic;
  def int_riscv_cv_mac_macuRN   : ScalarCoreVMacGprGprGprImmIntrinsic;
  def int_riscv_cv_mac_machhuRN : ScalarCoreVMacGprGprGprImmIntrinsic;
  def int_riscv_cv_mac_macsRN   : ScalarCoreVMacGprGprGprImmIntrinsic;
  def int_riscv_cv_mac_machhsRN : ScalarCoreVMacGprGprGprImmIntrinsic;

  defm add : CoreVSimdBinary<true>;
  def int_riscv_cv_simd_add_h : ScalarCoreVSimdGprGprImmIntrinsic;
  defm sub : CoreVSimdBinary<true>;
  def int_riscv_cv_simd_sub_h : ScalarCoreVSimdGprGprImmIntrinsic;
  defm avg : CoreVSimdBinary;
  defm avgu : CoreVSimdBinary;
  defm min : CoreVSimdBinary;
  defm minu : CoreVSimdBinary;
  defm max : CoreVSimdBinary;
  defm maxu : CoreVSimdBinary;
  defm srl : CoreVSimdBinary;
  defm sra : CoreVSimdBinary;
  defm sll : CoreVSimdBinary;
  defm or : CoreVSimdBinary;
  defm xor : CoreVSimdBinary;
  defm and : CoreVSimdBinary;

  defm abs : ScalarCoreVSimdGprIntrinsicHB;

  defm dotup : CoreVSimdBinary;
  defm dotusp : CoreVSimdBinary;
  defm dotsp : CoreVSimdBinary;
  defm sdotup : CoreVSimdTernary;
  defm sdotusp : CoreVSimdTernary;
  defm sdotsp : CoreVSimdTernary;

  defm extract : ScalarCoreVSimdGprImmIntrinsicHB;
  defm extractu : ScalarCoreVSimdGprImmIntrinsicHB;
  def int_riscv_cv_simd_insert_b : ScalarCoreVSimdGprGprImmIntrinsic;
  def int_riscv_cv_simd_insert_h : ScalarCoreVSimdGprGprImmIntrinsic;


  defm shuffle : ScalarCoreVSimdGprGprIntrinsicHB;
  def int_riscv_cv_simd_shuffle_sci_h : ScalarCoreVSimdGprImmIntrinsic;
  def int_riscv_cv_simd_shuffle_sci_b : ScalarCoreVSimdGprImmIntrinsic;
  defm shuffle2 : ScalarCoreVSimdGprGprGprIntrinsicHB;

  def int_riscv_cv_simd_packhi_h : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_packlo_h : ScalarCoreVSimdGprGprIntrinsic;
  def int_riscv_cv_simd_packhi_b : ScalarCoreVSimdGprGprGprIntrinsic;
  def int_riscv_cv_simd_packlo_b : ScalarCoreVSimdGprGprGprIntrinsic;

  defm cmpeq : CoreVSimdBinary;
  defm cmpne : CoreVSimdBinary;
  defm cmpgt : CoreVSimdBinary;
  defm cmpge : CoreVSimdBinary;
  defm cmplt : CoreVSimdBinary;
  defm cmple : CoreVSimdBinary;
  defm cmpgtu : CoreVSimdBinary;
  defm cmpgeu : CoreVSimdBinary;
  defm cmpltu : CoreVSimdBinary;
  defm cmpleu : CoreVSimdBinary;

  def int_riscv_cv_simd_cplxmul_r : ScalarCoreVSimdGprGprGprImmIntrinsic;
  def int_riscv_cv_simd_cplxmul_i : ScalarCoreVSimdGprGprGprImmIntrinsic;

  def int_riscv_cv_simd_cplxconj : ScalarCoreVSimdGprIntrinsic;

  def int_riscv_cv_simd_subrotmj : ScalarCoreVSimdGprGprImmIntrinsic;
} // TargetPrefix = "riscv"
