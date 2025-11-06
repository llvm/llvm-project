; Test the MSA intrinsics that are encoded with the 3R instruction format.
; There are lots of these so this covers those beginning with 'c'

; RUN: llc -mtriple=mips-elf -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s
; RUN: llc -mtriple=mipsel-elf -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s

@llvm_mips_ceq_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_ceq_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_ceq_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_ceq_b_test() nounwind {
entry:
  %0 = load <16 x i8>, ptr @llvm_mips_ceq_b_ARG1
  %1 = load <16 x i8>, ptr @llvm_mips_ceq_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.ceq.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, ptr @llvm_mips_ceq_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.ceq.b(<16 x i8>, <16 x i8>) nounwind

; CHECK: llvm_mips_ceq_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: ceq.b
; CHECK: st.b
; CHECK: .size llvm_mips_ceq_b_test
;
@llvm_mips_ceq_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_ceq_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_ceq_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_ceq_h_test() nounwind {
entry:
  %0 = load <8 x i16>, ptr @llvm_mips_ceq_h_ARG1
  %1 = load <8 x i16>, ptr @llvm_mips_ceq_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.ceq.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, ptr @llvm_mips_ceq_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.ceq.h(<8 x i16>, <8 x i16>) nounwind

; CHECK: llvm_mips_ceq_h_test:
; CHECK: ld.h
; CHECK: ld.h
; CHECK: ceq.h
; CHECK: st.h
; CHECK: .size llvm_mips_ceq_h_test
;
@llvm_mips_ceq_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_ceq_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_ceq_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_ceq_w_test() nounwind {
entry:
  %0 = load <4 x i32>, ptr @llvm_mips_ceq_w_ARG1
  %1 = load <4 x i32>, ptr @llvm_mips_ceq_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.ceq.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, ptr @llvm_mips_ceq_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.ceq.w(<4 x i32>, <4 x i32>) nounwind

; CHECK: llvm_mips_ceq_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: ceq.w
; CHECK: st.w
; CHECK: .size llvm_mips_ceq_w_test
;
@llvm_mips_ceq_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_ceq_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_ceq_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_ceq_d_test() nounwind {
entry:
  %0 = load <2 x i64>, ptr @llvm_mips_ceq_d_ARG1
  %1 = load <2 x i64>, ptr @llvm_mips_ceq_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.ceq.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, ptr @llvm_mips_ceq_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.ceq.d(<2 x i64>, <2 x i64>) nounwind

; CHECK: llvm_mips_ceq_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: ceq.d
; CHECK: st.d
; CHECK: .size llvm_mips_ceq_d_test
;
@llvm_mips_cle_s_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_cle_s_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_cle_s_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_cle_s_b_test() nounwind {
entry:
  %0 = load <16 x i8>, ptr @llvm_mips_cle_s_b_ARG1
  %1 = load <16 x i8>, ptr @llvm_mips_cle_s_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.cle.s.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, ptr @llvm_mips_cle_s_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.cle.s.b(<16 x i8>, <16 x i8>) nounwind

; CHECK: llvm_mips_cle_s_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: cle_s.b
; CHECK: st.b
; CHECK: .size llvm_mips_cle_s_b_test
;
@llvm_mips_cle_s_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_cle_s_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_cle_s_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_cle_s_h_test() nounwind {
entry:
  %0 = load <8 x i16>, ptr @llvm_mips_cle_s_h_ARG1
  %1 = load <8 x i16>, ptr @llvm_mips_cle_s_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.cle.s.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, ptr @llvm_mips_cle_s_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.cle.s.h(<8 x i16>, <8 x i16>) nounwind

; CHECK: llvm_mips_cle_s_h_test:
; CHECK: ld.h
; CHECK: ld.h
; CHECK: cle_s.h
; CHECK: st.h
; CHECK: .size llvm_mips_cle_s_h_test
;
@llvm_mips_cle_s_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_cle_s_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_cle_s_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_cle_s_w_test() nounwind {
entry:
  %0 = load <4 x i32>, ptr @llvm_mips_cle_s_w_ARG1
  %1 = load <4 x i32>, ptr @llvm_mips_cle_s_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.cle.s.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, ptr @llvm_mips_cle_s_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.cle.s.w(<4 x i32>, <4 x i32>) nounwind

; CHECK: llvm_mips_cle_s_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: cle_s.w
; CHECK: st.w
; CHECK: .size llvm_mips_cle_s_w_test
;
@llvm_mips_cle_s_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_cle_s_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_cle_s_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_cle_s_d_test() nounwind {
entry:
  %0 = load <2 x i64>, ptr @llvm_mips_cle_s_d_ARG1
  %1 = load <2 x i64>, ptr @llvm_mips_cle_s_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.cle.s.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, ptr @llvm_mips_cle_s_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.cle.s.d(<2 x i64>, <2 x i64>) nounwind

; CHECK: llvm_mips_cle_s_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: cle_s.d
; CHECK: st.d
; CHECK: .size llvm_mips_cle_s_d_test
;
@llvm_mips_cle_u_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_cle_u_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_cle_u_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_cle_u_b_test() nounwind {
entry:
  %0 = load <16 x i8>, ptr @llvm_mips_cle_u_b_ARG1
  %1 = load <16 x i8>, ptr @llvm_mips_cle_u_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.cle.u.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, ptr @llvm_mips_cle_u_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.cle.u.b(<16 x i8>, <16 x i8>) nounwind

; CHECK: llvm_mips_cle_u_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: cle_u.b
; CHECK: st.b
; CHECK: .size llvm_mips_cle_u_b_test
;
@llvm_mips_cle_u_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_cle_u_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_cle_u_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_cle_u_h_test() nounwind {
entry:
  %0 = load <8 x i16>, ptr @llvm_mips_cle_u_h_ARG1
  %1 = load <8 x i16>, ptr @llvm_mips_cle_u_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.cle.u.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, ptr @llvm_mips_cle_u_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.cle.u.h(<8 x i16>, <8 x i16>) nounwind

; CHECK: llvm_mips_cle_u_h_test:
; CHECK: ld.h
; CHECK: ld.h
; CHECK: cle_u.h
; CHECK: st.h
; CHECK: .size llvm_mips_cle_u_h_test
;
@llvm_mips_cle_u_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_cle_u_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_cle_u_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_cle_u_w_test() nounwind {
entry:
  %0 = load <4 x i32>, ptr @llvm_mips_cle_u_w_ARG1
  %1 = load <4 x i32>, ptr @llvm_mips_cle_u_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.cle.u.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, ptr @llvm_mips_cle_u_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.cle.u.w(<4 x i32>, <4 x i32>) nounwind

; CHECK: llvm_mips_cle_u_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: cle_u.w
; CHECK: st.w
; CHECK: .size llvm_mips_cle_u_w_test
;
@llvm_mips_cle_u_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_cle_u_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_cle_u_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_cle_u_d_test() nounwind {
entry:
  %0 = load <2 x i64>, ptr @llvm_mips_cle_u_d_ARG1
  %1 = load <2 x i64>, ptr @llvm_mips_cle_u_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.cle.u.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, ptr @llvm_mips_cle_u_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.cle.u.d(<2 x i64>, <2 x i64>) nounwind

; CHECK: llvm_mips_cle_u_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: cle_u.d
; CHECK: st.d
; CHECK: .size llvm_mips_cle_u_d_test
;
@llvm_mips_clt_s_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_clt_s_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_clt_s_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_clt_s_b_test() nounwind {
entry:
  %0 = load <16 x i8>, ptr @llvm_mips_clt_s_b_ARG1
  %1 = load <16 x i8>, ptr @llvm_mips_clt_s_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.clt.s.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, ptr @llvm_mips_clt_s_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.clt.s.b(<16 x i8>, <16 x i8>) nounwind

; CHECK: llvm_mips_clt_s_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: clt_s.b
; CHECK: st.b
; CHECK: .size llvm_mips_clt_s_b_test
;
@llvm_mips_clt_s_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_clt_s_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_clt_s_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_clt_s_h_test() nounwind {
entry:
  %0 = load <8 x i16>, ptr @llvm_mips_clt_s_h_ARG1
  %1 = load <8 x i16>, ptr @llvm_mips_clt_s_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.clt.s.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, ptr @llvm_mips_clt_s_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.clt.s.h(<8 x i16>, <8 x i16>) nounwind

; CHECK: llvm_mips_clt_s_h_test:
; CHECK: ld.h
; CHECK: ld.h
; CHECK: clt_s.h
; CHECK: st.h
; CHECK: .size llvm_mips_clt_s_h_test
;
@llvm_mips_clt_s_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_clt_s_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_clt_s_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_clt_s_w_test() nounwind {
entry:
  %0 = load <4 x i32>, ptr @llvm_mips_clt_s_w_ARG1
  %1 = load <4 x i32>, ptr @llvm_mips_clt_s_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.clt.s.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, ptr @llvm_mips_clt_s_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.clt.s.w(<4 x i32>, <4 x i32>) nounwind

; CHECK: llvm_mips_clt_s_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: clt_s.w
; CHECK: st.w
; CHECK: .size llvm_mips_clt_s_w_test
;
@llvm_mips_clt_s_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_clt_s_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_clt_s_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_clt_s_d_test() nounwind {
entry:
  %0 = load <2 x i64>, ptr @llvm_mips_clt_s_d_ARG1
  %1 = load <2 x i64>, ptr @llvm_mips_clt_s_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.clt.s.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, ptr @llvm_mips_clt_s_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.clt.s.d(<2 x i64>, <2 x i64>) nounwind

; CHECK: llvm_mips_clt_s_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: clt_s.d
; CHECK: st.d
; CHECK: .size llvm_mips_clt_s_d_test
;
@llvm_mips_clt_u_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_clt_u_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_clt_u_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_clt_u_b_test() nounwind {
entry:
  %0 = load <16 x i8>, ptr @llvm_mips_clt_u_b_ARG1
  %1 = load <16 x i8>, ptr @llvm_mips_clt_u_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.clt.u.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, ptr @llvm_mips_clt_u_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.clt.u.b(<16 x i8>, <16 x i8>) nounwind

; CHECK: llvm_mips_clt_u_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: clt_u.b
; CHECK: st.b
; CHECK: .size llvm_mips_clt_u_b_test
;
@llvm_mips_clt_u_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_clt_u_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_clt_u_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_clt_u_h_test() nounwind {
entry:
  %0 = load <8 x i16>, ptr @llvm_mips_clt_u_h_ARG1
  %1 = load <8 x i16>, ptr @llvm_mips_clt_u_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.clt.u.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, ptr @llvm_mips_clt_u_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.clt.u.h(<8 x i16>, <8 x i16>) nounwind

; CHECK: llvm_mips_clt_u_h_test:
; CHECK: ld.h
; CHECK: ld.h
; CHECK: clt_u.h
; CHECK: st.h
; CHECK: .size llvm_mips_clt_u_h_test
;
@llvm_mips_clt_u_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_clt_u_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_clt_u_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_clt_u_w_test() nounwind {
entry:
  %0 = load <4 x i32>, ptr @llvm_mips_clt_u_w_ARG1
  %1 = load <4 x i32>, ptr @llvm_mips_clt_u_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.clt.u.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, ptr @llvm_mips_clt_u_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.clt.u.w(<4 x i32>, <4 x i32>) nounwind

; CHECK: llvm_mips_clt_u_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: clt_u.w
; CHECK: st.w
; CHECK: .size llvm_mips_clt_u_w_test
;
@llvm_mips_clt_u_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_clt_u_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_clt_u_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_clt_u_d_test() nounwind {
entry:
  %0 = load <2 x i64>, ptr @llvm_mips_clt_u_d_ARG1
  %1 = load <2 x i64>, ptr @llvm_mips_clt_u_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.clt.u.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, ptr @llvm_mips_clt_u_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.clt.u.d(<2 x i64>, <2 x i64>) nounwind

; CHECK: llvm_mips_clt_u_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: clt_u.d
; CHECK: st.d
; CHECK: .size llvm_mips_clt_u_d_test
;
