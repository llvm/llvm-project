; Test the MSA intrinsics that are encoded with the BIT instruction format.

; RUN: llc -mtriple=mips-elf -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s
; RUN: llc -mtriple=mipsel-elf -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s

@llvm_mips_sat_s_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_sat_s_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_sat_s_b_test() nounwind {
entry:
  %0 = load <16 x i8>, ptr @llvm_mips_sat_s_b_ARG1
  %1 = tail call <16 x i8> @llvm.mips.sat.s.b(<16 x i8> %0, i32 7)
  store <16 x i8> %1, ptr @llvm_mips_sat_s_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.sat.s.b(<16 x i8>, i32) nounwind

; CHECK: llvm_mips_sat_s_b_test:
; CHECK: ld.b
; CHECK: sat_s.b
; CHECK: st.b
; CHECK: .size llvm_mips_sat_s_b_test
;
@llvm_mips_sat_s_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_sat_s_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_sat_s_h_test() nounwind {
entry:
  %0 = load <8 x i16>, ptr @llvm_mips_sat_s_h_ARG1
  %1 = tail call <8 x i16> @llvm.mips.sat.s.h(<8 x i16> %0, i32 7)
  store <8 x i16> %1, ptr @llvm_mips_sat_s_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.sat.s.h(<8 x i16>, i32) nounwind

; CHECK: llvm_mips_sat_s_h_test:
; CHECK: ld.h
; CHECK: sat_s.h
; CHECK: st.h
; CHECK: .size llvm_mips_sat_s_h_test
;
@llvm_mips_sat_s_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_sat_s_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_sat_s_w_test() nounwind {
entry:
  %0 = load <4 x i32>, ptr @llvm_mips_sat_s_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.sat.s.w(<4 x i32> %0, i32 7)
  store <4 x i32> %1, ptr @llvm_mips_sat_s_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.sat.s.w(<4 x i32>, i32) nounwind

; CHECK: llvm_mips_sat_s_w_test:
; CHECK: ld.w
; CHECK: sat_s.w
; CHECK: st.w
; CHECK: .size llvm_mips_sat_s_w_test
;
@llvm_mips_sat_s_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_sat_s_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_sat_s_d_test() nounwind {
entry:
  %0 = load <2 x i64>, ptr @llvm_mips_sat_s_d_ARG1
  %1 = tail call <2 x i64> @llvm.mips.sat.s.d(<2 x i64> %0, i32 7)
  store <2 x i64> %1, ptr @llvm_mips_sat_s_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.sat.s.d(<2 x i64>, i32) nounwind

; CHECK: llvm_mips_sat_s_d_test:
; CHECK: ld.d
; CHECK: sat_s.d
; CHECK: st.d
; CHECK: .size llvm_mips_sat_s_d_test
;
@llvm_mips_sat_u_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_sat_u_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_sat_u_b_test() nounwind {
entry:
  %0 = load <16 x i8>, ptr @llvm_mips_sat_u_b_ARG1
  %1 = tail call <16 x i8> @llvm.mips.sat.u.b(<16 x i8> %0, i32 7)
  store <16 x i8> %1, ptr @llvm_mips_sat_u_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.sat.u.b(<16 x i8>, i32) nounwind

; CHECK: llvm_mips_sat_u_b_test:
; CHECK: ld.b
; CHECK: sat_u.b
; CHECK: st.b
; CHECK: .size llvm_mips_sat_u_b_test
;
@llvm_mips_sat_u_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_sat_u_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_sat_u_h_test() nounwind {
entry:
  %0 = load <8 x i16>, ptr @llvm_mips_sat_u_h_ARG1
  %1 = tail call <8 x i16> @llvm.mips.sat.u.h(<8 x i16> %0, i32 7)
  store <8 x i16> %1, ptr @llvm_mips_sat_u_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.sat.u.h(<8 x i16>, i32) nounwind

; CHECK: llvm_mips_sat_u_h_test:
; CHECK: ld.h
; CHECK: sat_u.h
; CHECK: st.h
; CHECK: .size llvm_mips_sat_u_h_test
;
@llvm_mips_sat_u_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_sat_u_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_sat_u_w_test() nounwind {
entry:
  %0 = load <4 x i32>, ptr @llvm_mips_sat_u_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.sat.u.w(<4 x i32> %0, i32 7)
  store <4 x i32> %1, ptr @llvm_mips_sat_u_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.sat.u.w(<4 x i32>, i32) nounwind

; CHECK: llvm_mips_sat_u_w_test:
; CHECK: ld.w
; CHECK: sat_u.w
; CHECK: st.w
; CHECK: .size llvm_mips_sat_u_w_test
;
@llvm_mips_sat_u_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_sat_u_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_sat_u_d_test() nounwind {
entry:
  %0 = load <2 x i64>, ptr @llvm_mips_sat_u_d_ARG1
  %1 = tail call <2 x i64> @llvm.mips.sat.u.d(<2 x i64> %0, i32 7)
  store <2 x i64> %1, ptr @llvm_mips_sat_u_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.sat.u.d(<2 x i64>, i32) nounwind

; CHECK: llvm_mips_sat_u_d_test:
; CHECK: ld.d
; CHECK: sat_u.d
; CHECK: st.d
; CHECK: .size llvm_mips_sat_u_d_test
;
@llvm_mips_slli_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_slli_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_slli_b_test() nounwind {
entry:
  %0 = load <16 x i8>, ptr @llvm_mips_slli_b_ARG1
  %1 = tail call <16 x i8> @llvm.mips.slli.b(<16 x i8> %0, i32 7)
  store <16 x i8> %1, ptr @llvm_mips_slli_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.slli.b(<16 x i8>, i32) nounwind

; CHECK: llvm_mips_slli_b_test:
; CHECK: ld.b
; CHECK: slli.b
; CHECK: st.b
; CHECK: .size llvm_mips_slli_b_test
;
@llvm_mips_slli_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_slli_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_slli_h_test() nounwind {
entry:
  %0 = load <8 x i16>, ptr @llvm_mips_slli_h_ARG1
  %1 = tail call <8 x i16> @llvm.mips.slli.h(<8 x i16> %0, i32 7)
  store <8 x i16> %1, ptr @llvm_mips_slli_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.slli.h(<8 x i16>, i32) nounwind

; CHECK: llvm_mips_slli_h_test:
; CHECK: ld.h
; CHECK: slli.h
; CHECK: st.h
; CHECK: .size llvm_mips_slli_h_test
;
@llvm_mips_slli_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_slli_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_slli_w_test() nounwind {
entry:
  %0 = load <4 x i32>, ptr @llvm_mips_slli_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.slli.w(<4 x i32> %0, i32 7)
  store <4 x i32> %1, ptr @llvm_mips_slli_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.slli.w(<4 x i32>, i32) nounwind

; CHECK: llvm_mips_slli_w_test:
; CHECK: ld.w
; CHECK: slli.w
; CHECK: st.w
; CHECK: .size llvm_mips_slli_w_test
;
@llvm_mips_slli_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_slli_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_slli_d_test() nounwind {
entry:
  %0 = load <2 x i64>, ptr @llvm_mips_slli_d_ARG1
  %1 = tail call <2 x i64> @llvm.mips.slli.d(<2 x i64> %0, i32 7)
  store <2 x i64> %1, ptr @llvm_mips_slli_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.slli.d(<2 x i64>, i32) nounwind

; CHECK: llvm_mips_slli_d_test:
; CHECK: ld.d
; CHECK: slli.d
; CHECK: st.d
; CHECK: .size llvm_mips_slli_d_test
;
@llvm_mips_srai_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_srai_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_srai_b_test() nounwind {
entry:
  %0 = load <16 x i8>, ptr @llvm_mips_srai_b_ARG1
  %1 = tail call <16 x i8> @llvm.mips.srai.b(<16 x i8> %0, i32 7)
  store <16 x i8> %1, ptr @llvm_mips_srai_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.srai.b(<16 x i8>, i32) nounwind

; CHECK: llvm_mips_srai_b_test:
; CHECK: ld.b
; CHECK: srai.b
; CHECK: st.b
; CHECK: .size llvm_mips_srai_b_test
;
@llvm_mips_srai_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_srai_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_srai_h_test() nounwind {
entry:
  %0 = load <8 x i16>, ptr @llvm_mips_srai_h_ARG1
  %1 = tail call <8 x i16> @llvm.mips.srai.h(<8 x i16> %0, i32 7)
  store <8 x i16> %1, ptr @llvm_mips_srai_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.srai.h(<8 x i16>, i32) nounwind

; CHECK: llvm_mips_srai_h_test:
; CHECK: ld.h
; CHECK: srai.h
; CHECK: st.h
; CHECK: .size llvm_mips_srai_h_test
;
@llvm_mips_srai_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_srai_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_srai_w_test() nounwind {
entry:
  %0 = load <4 x i32>, ptr @llvm_mips_srai_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.srai.w(<4 x i32> %0, i32 7)
  store <4 x i32> %1, ptr @llvm_mips_srai_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.srai.w(<4 x i32>, i32) nounwind

; CHECK: llvm_mips_srai_w_test:
; CHECK: ld.w
; CHECK: srai.w
; CHECK: st.w
; CHECK: .size llvm_mips_srai_w_test
;
@llvm_mips_srai_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_srai_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_srai_d_test() nounwind {
entry:
  %0 = load <2 x i64>, ptr @llvm_mips_srai_d_ARG1
  %1 = tail call <2 x i64> @llvm.mips.srai.d(<2 x i64> %0, i32 7)
  store <2 x i64> %1, ptr @llvm_mips_srai_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.srai.d(<2 x i64>, i32) nounwind

; CHECK: llvm_mips_srai_d_test:
; CHECK: ld.d
; CHECK: srai.d
; CHECK: st.d
; CHECK: .size llvm_mips_srai_d_test
;
@llvm_mips_srari_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_srari_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_srari_b_test() nounwind {
entry:
  %0 = load <16 x i8>, ptr @llvm_mips_srari_b_ARG1
  %1 = tail call <16 x i8> @llvm.mips.srari.b(<16 x i8> %0, i32 7)
  store <16 x i8> %1, ptr @llvm_mips_srari_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.srari.b(<16 x i8>, i32) nounwind

; CHECK: llvm_mips_srari_b_test:
; CHECK: ld.b
; CHECK: srari.b
; CHECK: st.b
; CHECK: .size llvm_mips_srari_b_test
;
@llvm_mips_srari_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_srari_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_srari_h_test() nounwind {
entry:
  %0 = load <8 x i16>, ptr @llvm_mips_srari_h_ARG1
  %1 = tail call <8 x i16> @llvm.mips.srari.h(<8 x i16> %0, i32 7)
  store <8 x i16> %1, ptr @llvm_mips_srari_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.srari.h(<8 x i16>, i32) nounwind

; CHECK: llvm_mips_srari_h_test:
; CHECK: ld.h
; CHECK: srari.h
; CHECK: st.h
; CHECK: .size llvm_mips_srari_h_test
;
@llvm_mips_srari_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_srari_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_srari_w_test() nounwind {
entry:
  %0 = load <4 x i32>, ptr @llvm_mips_srari_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.srari.w(<4 x i32> %0, i32 7)
  store <4 x i32> %1, ptr @llvm_mips_srari_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.srari.w(<4 x i32>, i32) nounwind

; CHECK: llvm_mips_srari_w_test:
; CHECK: ld.w
; CHECK: srari.w
; CHECK: st.w
; CHECK: .size llvm_mips_srari_w_test
;
@llvm_mips_srari_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_srari_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_srari_d_test() nounwind {
entry:
  %0 = load <2 x i64>, ptr @llvm_mips_srari_d_ARG1
  %1 = tail call <2 x i64> @llvm.mips.srari.d(<2 x i64> %0, i32 7)
  store <2 x i64> %1, ptr @llvm_mips_srari_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.srari.d(<2 x i64>, i32) nounwind

; CHECK: llvm_mips_srari_d_test:
; CHECK: ld.d
; CHECK: srari.d
; CHECK: st.d
; CHECK: .size llvm_mips_srari_d_test
;
@llvm_mips_srli_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_srli_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_srli_b_test() nounwind {
entry:
  %0 = load <16 x i8>, ptr @llvm_mips_srli_b_ARG1
  %1 = tail call <16 x i8> @llvm.mips.srli.b(<16 x i8> %0, i32 7)
  store <16 x i8> %1, ptr @llvm_mips_srli_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.srli.b(<16 x i8>, i32) nounwind

; CHECK: llvm_mips_srli_b_test:
; CHECK: ld.b
; CHECK: srli.b
; CHECK: st.b
; CHECK: .size llvm_mips_srli_b_test
;
@llvm_mips_srli_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_srli_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_srli_h_test() nounwind {
entry:
  %0 = load <8 x i16>, ptr @llvm_mips_srli_h_ARG1
  %1 = tail call <8 x i16> @llvm.mips.srli.h(<8 x i16> %0, i32 7)
  store <8 x i16> %1, ptr @llvm_mips_srli_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.srli.h(<8 x i16>, i32) nounwind

; CHECK: llvm_mips_srli_h_test:
; CHECK: ld.h
; CHECK: srli.h
; CHECK: st.h
; CHECK: .size llvm_mips_srli_h_test
;
@llvm_mips_srli_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_srli_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_srli_w_test() nounwind {
entry:
  %0 = load <4 x i32>, ptr @llvm_mips_srli_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.srli.w(<4 x i32> %0, i32 7)
  store <4 x i32> %1, ptr @llvm_mips_srli_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.srli.w(<4 x i32>, i32) nounwind

; CHECK: llvm_mips_srli_w_test:
; CHECK: ld.w
; CHECK: srli.w
; CHECK: st.w
; CHECK: .size llvm_mips_srli_w_test
;
@llvm_mips_srli_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_srli_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_srli_d_test() nounwind {
entry:
  %0 = load <2 x i64>, ptr @llvm_mips_srli_d_ARG1
  %1 = tail call <2 x i64> @llvm.mips.srli.d(<2 x i64> %0, i32 7)
  store <2 x i64> %1, ptr @llvm_mips_srli_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.srli.d(<2 x i64>, i32) nounwind

; CHECK: llvm_mips_srli_d_test:
; CHECK: ld.d
; CHECK: srli.d
; CHECK: st.d
; CHECK: .size llvm_mips_srli_d_test
;
@llvm_mips_srlri_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_srlri_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_srlri_b_test() nounwind {
entry:
  %0 = load <16 x i8>, ptr @llvm_mips_srlri_b_ARG1
  %1 = tail call <16 x i8> @llvm.mips.srlri.b(<16 x i8> %0, i32 7)
  store <16 x i8> %1, ptr @llvm_mips_srlri_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.srlri.b(<16 x i8>, i32) nounwind

; CHECK: llvm_mips_srlri_b_test:
; CHECK: ld.b
; CHECK: srlri.b
; CHECK: st.b
; CHECK: .size llvm_mips_srlri_b_test
;
@llvm_mips_srlri_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_srlri_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_srlri_h_test() nounwind {
entry:
  %0 = load <8 x i16>, ptr @llvm_mips_srlri_h_ARG1
  %1 = tail call <8 x i16> @llvm.mips.srlri.h(<8 x i16> %0, i32 7)
  store <8 x i16> %1, ptr @llvm_mips_srlri_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.srlri.h(<8 x i16>, i32) nounwind

; CHECK: llvm_mips_srlri_h_test:
; CHECK: ld.h
; CHECK: srlri.h
; CHECK: st.h
; CHECK: .size llvm_mips_srlri_h_test
;
@llvm_mips_srlri_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_srlri_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_srlri_w_test() nounwind {
entry:
  %0 = load <4 x i32>, ptr @llvm_mips_srlri_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.srlri.w(<4 x i32> %0, i32 7)
  store <4 x i32> %1, ptr @llvm_mips_srlri_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.srlri.w(<4 x i32>, i32) nounwind

; CHECK: llvm_mips_srlri_w_test:
; CHECK: ld.w
; CHECK: srlri.w
; CHECK: st.w
; CHECK: .size llvm_mips_srlri_w_test
;
@llvm_mips_srlri_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_srlri_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_srlri_d_test() nounwind {
entry:
  %0 = load <2 x i64>, ptr @llvm_mips_srlri_d_ARG1
  %1 = tail call <2 x i64> @llvm.mips.srlri.d(<2 x i64> %0, i32 7)
  store <2 x i64> %1, ptr @llvm_mips_srlri_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.srlri.d(<2 x i64>, i32) nounwind

; CHECK: llvm_mips_srlri_d_test:
; CHECK: ld.d
; CHECK: srlri.d
; CHECK: st.d
; CHECK: .size llvm_mips_srlri_d_test
;
