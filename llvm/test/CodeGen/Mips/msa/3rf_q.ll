; Test the MSA fixed-point intrinsics that are encoded with the 3RF instruction
; format.

; RUN: llc -march=mips -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s

@llvm_mips_mul_q_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_mul_q_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_mul_q_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_mul_q_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_mul_q_h_ARG1
  %1 = load <8 x i16>, <8 x i16>* @llvm_mips_mul_q_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.mul.q.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_mul_q_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.mul.q.h(<8 x i16>, <8 x i16>) nounwind

; CHECK: llvm_mips_mul_q_h_test:
; CHECK: ld.h
; CHECK: ld.h
; CHECK: mul_q.h
; CHECK: st.h
; CHECK: .size llvm_mips_mul_q_h_test
;
@llvm_mips_mul_q_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_mul_q_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_mul_q_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_mul_q_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_mul_q_w_ARG1
  %1 = load <4 x i32>, <4 x i32>* @llvm_mips_mul_q_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.mul.q.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_mul_q_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.mul.q.w(<4 x i32>, <4 x i32>) nounwind

; CHECK: llvm_mips_mul_q_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: mul_q.w
; CHECK: st.w
; CHECK: .size llvm_mips_mul_q_w_test
;
@llvm_mips_mulr_q_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_mulr_q_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_mulr_q_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_mulr_q_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_mulr_q_h_ARG1
  %1 = load <8 x i16>, <8 x i16>* @llvm_mips_mulr_q_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.mulr.q.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_mulr_q_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.mulr.q.h(<8 x i16>, <8 x i16>) nounwind

; CHECK: llvm_mips_mulr_q_h_test:
; CHECK: ld.h
; CHECK: ld.h
; CHECK: mulr_q.h
; CHECK: st.h
; CHECK: .size llvm_mips_mulr_q_h_test
;
@llvm_mips_mulr_q_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_mulr_q_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_mulr_q_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_mulr_q_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_mulr_q_w_ARG1
  %1 = load <4 x i32>, <4 x i32>* @llvm_mips_mulr_q_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.mulr.q.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_mulr_q_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.mulr.q.w(<4 x i32>, <4 x i32>) nounwind

; CHECK: llvm_mips_mulr_q_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: mulr_q.w
; CHECK: st.w
; CHECK: .size llvm_mips_mulr_q_w_test
;
