; Test the MSA intrinsics that are encoded with the 3R instruction format.
; There are lots of these so this covers those beginning with 'm'

; RUN: llc -march=mips -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s

@llvm_mips_max_a_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_max_a_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_max_a_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_max_a_b_test() nounwind {
entry:
  %0 = load <16 x i8>, ptr @llvm_mips_max_a_b_ARG1
  %1 = load <16 x i8>, ptr @llvm_mips_max_a_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.max.a.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, ptr @llvm_mips_max_a_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.max.a.b(<16 x i8>, <16 x i8>) nounwind

; CHECK: llvm_mips_max_a_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: max_a.b
; CHECK: st.b
; CHECK: .size llvm_mips_max_a_b_test
;
@llvm_mips_max_a_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_max_a_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_max_a_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_max_a_h_test() nounwind {
entry:
  %0 = load <8 x i16>, ptr @llvm_mips_max_a_h_ARG1
  %1 = load <8 x i16>, ptr @llvm_mips_max_a_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.max.a.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, ptr @llvm_mips_max_a_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.max.a.h(<8 x i16>, <8 x i16>) nounwind

; CHECK: llvm_mips_max_a_h_test:
; CHECK: ld.h
; CHECK: ld.h
; CHECK: max_a.h
; CHECK: st.h
; CHECK: .size llvm_mips_max_a_h_test
;
@llvm_mips_max_a_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_max_a_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_max_a_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_max_a_w_test() nounwind {
entry:
  %0 = load <4 x i32>, ptr @llvm_mips_max_a_w_ARG1
  %1 = load <4 x i32>, ptr @llvm_mips_max_a_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.max.a.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, ptr @llvm_mips_max_a_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.max.a.w(<4 x i32>, <4 x i32>) nounwind

; CHECK: llvm_mips_max_a_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: max_a.w
; CHECK: st.w
; CHECK: .size llvm_mips_max_a_w_test
;
@llvm_mips_max_a_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_max_a_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_max_a_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_max_a_d_test() nounwind {
entry:
  %0 = load <2 x i64>, ptr @llvm_mips_max_a_d_ARG1
  %1 = load <2 x i64>, ptr @llvm_mips_max_a_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.max.a.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, ptr @llvm_mips_max_a_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.max.a.d(<2 x i64>, <2 x i64>) nounwind

; CHECK: llvm_mips_max_a_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: max_a.d
; CHECK: st.d
; CHECK: .size llvm_mips_max_a_d_test
;
@llvm_mips_max_s_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_max_s_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_max_s_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_max_s_b_test() nounwind {
entry:
  %0 = load <16 x i8>, ptr @llvm_mips_max_s_b_ARG1
  %1 = load <16 x i8>, ptr @llvm_mips_max_s_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.max.s.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, ptr @llvm_mips_max_s_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.max.s.b(<16 x i8>, <16 x i8>) nounwind

; CHECK: llvm_mips_max_s_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: max_s.b
; CHECK: st.b
; CHECK: .size llvm_mips_max_s_b_test
;
@llvm_mips_max_s_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_max_s_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_max_s_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_max_s_h_test() nounwind {
entry:
  %0 = load <8 x i16>, ptr @llvm_mips_max_s_h_ARG1
  %1 = load <8 x i16>, ptr @llvm_mips_max_s_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.max.s.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, ptr @llvm_mips_max_s_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.max.s.h(<8 x i16>, <8 x i16>) nounwind

; CHECK: llvm_mips_max_s_h_test:
; CHECK: ld.h
; CHECK: ld.h
; CHECK: max_s.h
; CHECK: st.h
; CHECK: .size llvm_mips_max_s_h_test
;
@llvm_mips_max_s_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_max_s_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_max_s_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_max_s_w_test() nounwind {
entry:
  %0 = load <4 x i32>, ptr @llvm_mips_max_s_w_ARG1
  %1 = load <4 x i32>, ptr @llvm_mips_max_s_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.max.s.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, ptr @llvm_mips_max_s_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.max.s.w(<4 x i32>, <4 x i32>) nounwind

; CHECK: llvm_mips_max_s_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: max_s.w
; CHECK: st.w
; CHECK: .size llvm_mips_max_s_w_test
;
@llvm_mips_max_s_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_max_s_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_max_s_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_max_s_d_test() nounwind {
entry:
  %0 = load <2 x i64>, ptr @llvm_mips_max_s_d_ARG1
  %1 = load <2 x i64>, ptr @llvm_mips_max_s_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.max.s.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, ptr @llvm_mips_max_s_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.max.s.d(<2 x i64>, <2 x i64>) nounwind

; CHECK: llvm_mips_max_s_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: max_s.d
; CHECK: st.d
; CHECK: .size llvm_mips_max_s_d_test
;
@llvm_mips_max_u_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_max_u_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_max_u_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_max_u_b_test() nounwind {
entry:
  %0 = load <16 x i8>, ptr @llvm_mips_max_u_b_ARG1
  %1 = load <16 x i8>, ptr @llvm_mips_max_u_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.max.u.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, ptr @llvm_mips_max_u_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.max.u.b(<16 x i8>, <16 x i8>) nounwind

; CHECK: llvm_mips_max_u_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: max_u.b
; CHECK: st.b
; CHECK: .size llvm_mips_max_u_b_test
;
@llvm_mips_max_u_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_max_u_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_max_u_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_max_u_h_test() nounwind {
entry:
  %0 = load <8 x i16>, ptr @llvm_mips_max_u_h_ARG1
  %1 = load <8 x i16>, ptr @llvm_mips_max_u_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.max.u.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, ptr @llvm_mips_max_u_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.max.u.h(<8 x i16>, <8 x i16>) nounwind

; CHECK: llvm_mips_max_u_h_test:
; CHECK: ld.h
; CHECK: ld.h
; CHECK: max_u.h
; CHECK: st.h
; CHECK: .size llvm_mips_max_u_h_test
;
@llvm_mips_max_u_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_max_u_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_max_u_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_max_u_w_test() nounwind {
entry:
  %0 = load <4 x i32>, ptr @llvm_mips_max_u_w_ARG1
  %1 = load <4 x i32>, ptr @llvm_mips_max_u_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.max.u.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, ptr @llvm_mips_max_u_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.max.u.w(<4 x i32>, <4 x i32>) nounwind

; CHECK: llvm_mips_max_u_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: max_u.w
; CHECK: st.w
; CHECK: .size llvm_mips_max_u_w_test
;
@llvm_mips_max_u_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_max_u_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_max_u_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_max_u_d_test() nounwind {
entry:
  %0 = load <2 x i64>, ptr @llvm_mips_max_u_d_ARG1
  %1 = load <2 x i64>, ptr @llvm_mips_max_u_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.max.u.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, ptr @llvm_mips_max_u_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.max.u.d(<2 x i64>, <2 x i64>) nounwind

; CHECK: llvm_mips_max_u_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: max_u.d
; CHECK: st.d
; CHECK: .size llvm_mips_max_u_d_test
;
@llvm_mips_min_a_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_min_a_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_min_a_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_min_a_b_test() nounwind {
entry:
  %0 = load <16 x i8>, ptr @llvm_mips_min_a_b_ARG1
  %1 = load <16 x i8>, ptr @llvm_mips_min_a_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.min.a.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, ptr @llvm_mips_min_a_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.min.a.b(<16 x i8>, <16 x i8>) nounwind

; CHECK: llvm_mips_min_a_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: min_a.b
; CHECK: st.b
; CHECK: .size llvm_mips_min_a_b_test
;
@llvm_mips_min_a_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_min_a_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_min_a_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_min_a_h_test() nounwind {
entry:
  %0 = load <8 x i16>, ptr @llvm_mips_min_a_h_ARG1
  %1 = load <8 x i16>, ptr @llvm_mips_min_a_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.min.a.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, ptr @llvm_mips_min_a_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.min.a.h(<8 x i16>, <8 x i16>) nounwind

; CHECK: llvm_mips_min_a_h_test:
; CHECK: ld.h
; CHECK: ld.h
; CHECK: min_a.h
; CHECK: st.h
; CHECK: .size llvm_mips_min_a_h_test
;
@llvm_mips_min_a_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_min_a_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_min_a_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_min_a_w_test() nounwind {
entry:
  %0 = load <4 x i32>, ptr @llvm_mips_min_a_w_ARG1
  %1 = load <4 x i32>, ptr @llvm_mips_min_a_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.min.a.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, ptr @llvm_mips_min_a_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.min.a.w(<4 x i32>, <4 x i32>) nounwind

; CHECK: llvm_mips_min_a_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: min_a.w
; CHECK: st.w
; CHECK: .size llvm_mips_min_a_w_test
;
@llvm_mips_min_a_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_min_a_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_min_a_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_min_a_d_test() nounwind {
entry:
  %0 = load <2 x i64>, ptr @llvm_mips_min_a_d_ARG1
  %1 = load <2 x i64>, ptr @llvm_mips_min_a_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.min.a.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, ptr @llvm_mips_min_a_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.min.a.d(<2 x i64>, <2 x i64>) nounwind

; CHECK: llvm_mips_min_a_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: min_a.d
; CHECK: st.d
; CHECK: .size llvm_mips_min_a_d_test
;
@llvm_mips_min_s_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_min_s_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_min_s_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_min_s_b_test() nounwind {
entry:
  %0 = load <16 x i8>, ptr @llvm_mips_min_s_b_ARG1
  %1 = load <16 x i8>, ptr @llvm_mips_min_s_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.min.s.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, ptr @llvm_mips_min_s_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.min.s.b(<16 x i8>, <16 x i8>) nounwind

; CHECK: llvm_mips_min_s_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: min_s.b
; CHECK: st.b
; CHECK: .size llvm_mips_min_s_b_test
;
@llvm_mips_min_s_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_min_s_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_min_s_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_min_s_h_test() nounwind {
entry:
  %0 = load <8 x i16>, ptr @llvm_mips_min_s_h_ARG1
  %1 = load <8 x i16>, ptr @llvm_mips_min_s_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.min.s.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, ptr @llvm_mips_min_s_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.min.s.h(<8 x i16>, <8 x i16>) nounwind

; CHECK: llvm_mips_min_s_h_test:
; CHECK: ld.h
; CHECK: ld.h
; CHECK: min_s.h
; CHECK: st.h
; CHECK: .size llvm_mips_min_s_h_test
;
@llvm_mips_min_s_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_min_s_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_min_s_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_min_s_w_test() nounwind {
entry:
  %0 = load <4 x i32>, ptr @llvm_mips_min_s_w_ARG1
  %1 = load <4 x i32>, ptr @llvm_mips_min_s_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.min.s.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, ptr @llvm_mips_min_s_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.min.s.w(<4 x i32>, <4 x i32>) nounwind

; CHECK: llvm_mips_min_s_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: min_s.w
; CHECK: st.w
; CHECK: .size llvm_mips_min_s_w_test
;
@llvm_mips_min_s_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_min_s_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_min_s_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_min_s_d_test() nounwind {
entry:
  %0 = load <2 x i64>, ptr @llvm_mips_min_s_d_ARG1
  %1 = load <2 x i64>, ptr @llvm_mips_min_s_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.min.s.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, ptr @llvm_mips_min_s_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.min.s.d(<2 x i64>, <2 x i64>) nounwind

; CHECK: llvm_mips_min_s_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: min_s.d
; CHECK: st.d
; CHECK: .size llvm_mips_min_s_d_test
;
@llvm_mips_min_u_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_min_u_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_min_u_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_min_u_b_test() nounwind {
entry:
  %0 = load <16 x i8>, ptr @llvm_mips_min_u_b_ARG1
  %1 = load <16 x i8>, ptr @llvm_mips_min_u_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.min.u.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, ptr @llvm_mips_min_u_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.min.u.b(<16 x i8>, <16 x i8>) nounwind

; CHECK: llvm_mips_min_u_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: min_u.b
; CHECK: st.b
; CHECK: .size llvm_mips_min_u_b_test
;
@llvm_mips_min_u_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_min_u_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_min_u_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_min_u_h_test() nounwind {
entry:
  %0 = load <8 x i16>, ptr @llvm_mips_min_u_h_ARG1
  %1 = load <8 x i16>, ptr @llvm_mips_min_u_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.min.u.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, ptr @llvm_mips_min_u_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.min.u.h(<8 x i16>, <8 x i16>) nounwind

; CHECK: llvm_mips_min_u_h_test:
; CHECK: ld.h
; CHECK: ld.h
; CHECK: min_u.h
; CHECK: st.h
; CHECK: .size llvm_mips_min_u_h_test
;
@llvm_mips_min_u_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_min_u_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_min_u_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_min_u_w_test() nounwind {
entry:
  %0 = load <4 x i32>, ptr @llvm_mips_min_u_w_ARG1
  %1 = load <4 x i32>, ptr @llvm_mips_min_u_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.min.u.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, ptr @llvm_mips_min_u_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.min.u.w(<4 x i32>, <4 x i32>) nounwind

; CHECK: llvm_mips_min_u_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: min_u.w
; CHECK: st.w
; CHECK: .size llvm_mips_min_u_w_test
;
@llvm_mips_min_u_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_min_u_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_min_u_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_min_u_d_test() nounwind {
entry:
  %0 = load <2 x i64>, ptr @llvm_mips_min_u_d_ARG1
  %1 = load <2 x i64>, ptr @llvm_mips_min_u_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.min.u.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, ptr @llvm_mips_min_u_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.min.u.d(<2 x i64>, <2 x i64>) nounwind

; CHECK: llvm_mips_min_u_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: min_u.d
; CHECK: st.d
; CHECK: .size llvm_mips_min_u_d_test
;
@llvm_mips_mod_s_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_mod_s_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_mod_s_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_mod_s_b_test() nounwind {
entry:
  %0 = load <16 x i8>, ptr @llvm_mips_mod_s_b_ARG1
  %1 = load <16 x i8>, ptr @llvm_mips_mod_s_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.mod.s.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, ptr @llvm_mips_mod_s_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.mod.s.b(<16 x i8>, <16 x i8>) nounwind

; CHECK: llvm_mips_mod_s_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: mod_s.b
; CHECK: st.b
; CHECK: .size llvm_mips_mod_s_b_test
;
@llvm_mips_mod_s_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_mod_s_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_mod_s_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_mod_s_h_test() nounwind {
entry:
  %0 = load <8 x i16>, ptr @llvm_mips_mod_s_h_ARG1
  %1 = load <8 x i16>, ptr @llvm_mips_mod_s_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.mod.s.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, ptr @llvm_mips_mod_s_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.mod.s.h(<8 x i16>, <8 x i16>) nounwind

; CHECK: llvm_mips_mod_s_h_test:
; CHECK: ld.h
; CHECK: ld.h
; CHECK: mod_s.h
; CHECK: st.h
; CHECK: .size llvm_mips_mod_s_h_test
;
@llvm_mips_mod_s_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_mod_s_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_mod_s_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_mod_s_w_test() nounwind {
entry:
  %0 = load <4 x i32>, ptr @llvm_mips_mod_s_w_ARG1
  %1 = load <4 x i32>, ptr @llvm_mips_mod_s_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.mod.s.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, ptr @llvm_mips_mod_s_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.mod.s.w(<4 x i32>, <4 x i32>) nounwind

; CHECK: llvm_mips_mod_s_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: mod_s.w
; CHECK: st.w
; CHECK: .size llvm_mips_mod_s_w_test
;
@llvm_mips_mod_s_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_mod_s_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_mod_s_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_mod_s_d_test() nounwind {
entry:
  %0 = load <2 x i64>, ptr @llvm_mips_mod_s_d_ARG1
  %1 = load <2 x i64>, ptr @llvm_mips_mod_s_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.mod.s.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, ptr @llvm_mips_mod_s_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.mod.s.d(<2 x i64>, <2 x i64>) nounwind

; CHECK: llvm_mips_mod_s_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: mod_s.d
; CHECK: st.d
; CHECK: .size llvm_mips_mod_s_d_test
;
@llvm_mips_mod_u_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_mod_u_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_mod_u_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_mod_u_b_test() nounwind {
entry:
  %0 = load <16 x i8>, ptr @llvm_mips_mod_u_b_ARG1
  %1 = load <16 x i8>, ptr @llvm_mips_mod_u_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.mod.u.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, ptr @llvm_mips_mod_u_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.mod.u.b(<16 x i8>, <16 x i8>) nounwind

; CHECK: llvm_mips_mod_u_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: mod_u.b
; CHECK: st.b
; CHECK: .size llvm_mips_mod_u_b_test
;
@llvm_mips_mod_u_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_mod_u_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_mod_u_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_mod_u_h_test() nounwind {
entry:
  %0 = load <8 x i16>, ptr @llvm_mips_mod_u_h_ARG1
  %1 = load <8 x i16>, ptr @llvm_mips_mod_u_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.mod.u.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, ptr @llvm_mips_mod_u_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.mod.u.h(<8 x i16>, <8 x i16>) nounwind

; CHECK: llvm_mips_mod_u_h_test:
; CHECK: ld.h
; CHECK: ld.h
; CHECK: mod_u.h
; CHECK: st.h
; CHECK: .size llvm_mips_mod_u_h_test
;
@llvm_mips_mod_u_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_mod_u_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_mod_u_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_mod_u_w_test() nounwind {
entry:
  %0 = load <4 x i32>, ptr @llvm_mips_mod_u_w_ARG1
  %1 = load <4 x i32>, ptr @llvm_mips_mod_u_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.mod.u.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, ptr @llvm_mips_mod_u_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.mod.u.w(<4 x i32>, <4 x i32>) nounwind

; CHECK: llvm_mips_mod_u_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: mod_u.w
; CHECK: st.w
; CHECK: .size llvm_mips_mod_u_w_test
;
@llvm_mips_mod_u_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_mod_u_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_mod_u_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_mod_u_d_test() nounwind {
entry:
  %0 = load <2 x i64>, ptr @llvm_mips_mod_u_d_ARG1
  %1 = load <2 x i64>, ptr @llvm_mips_mod_u_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.mod.u.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, ptr @llvm_mips_mod_u_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.mod.u.d(<2 x i64>, <2 x i64>) nounwind

; CHECK: llvm_mips_mod_u_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: mod_u.d
; CHECK: st.d
; CHECK: .size llvm_mips_mod_u_d_test
;
@llvm_mips_mulv_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_mulv_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_mulv_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_mulv_b_test() nounwind {
entry:
  %0 = load <16 x i8>, ptr @llvm_mips_mulv_b_ARG1
  %1 = load <16 x i8>, ptr @llvm_mips_mulv_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.mulv.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, ptr @llvm_mips_mulv_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.mulv.b(<16 x i8>, <16 x i8>) nounwind

; CHECK: llvm_mips_mulv_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: mulv.b
; CHECK: st.b
; CHECK: .size llvm_mips_mulv_b_test
;
@llvm_mips_mulv_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_mulv_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_mulv_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_mulv_h_test() nounwind {
entry:
  %0 = load <8 x i16>, ptr @llvm_mips_mulv_h_ARG1
  %1 = load <8 x i16>, ptr @llvm_mips_mulv_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.mulv.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, ptr @llvm_mips_mulv_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.mulv.h(<8 x i16>, <8 x i16>) nounwind

; CHECK: llvm_mips_mulv_h_test:
; CHECK: ld.h
; CHECK: ld.h
; CHECK: mulv.h
; CHECK: st.h
; CHECK: .size llvm_mips_mulv_h_test
;
@llvm_mips_mulv_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_mulv_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_mulv_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_mulv_w_test() nounwind {
entry:
  %0 = load <4 x i32>, ptr @llvm_mips_mulv_w_ARG1
  %1 = load <4 x i32>, ptr @llvm_mips_mulv_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.mulv.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, ptr @llvm_mips_mulv_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.mulv.w(<4 x i32>, <4 x i32>) nounwind

; CHECK: llvm_mips_mulv_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: mulv.w
; CHECK: st.w
; CHECK: .size llvm_mips_mulv_w_test
;
@llvm_mips_mulv_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_mulv_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_mulv_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_mulv_d_test() nounwind {
entry:
  %0 = load <2 x i64>, ptr @llvm_mips_mulv_d_ARG1
  %1 = load <2 x i64>, ptr @llvm_mips_mulv_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.mulv.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, ptr @llvm_mips_mulv_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.mulv.d(<2 x i64>, <2 x i64>) nounwind

; CHECK: llvm_mips_mulv_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: mulv.d
; CHECK: st.d
; CHECK: .size llvm_mips_mulv_d_test

define void @mulv_b_test() nounwind {
entry:
  %0 = load <16 x i8>, ptr @llvm_mips_mulv_b_ARG1
  %1 = load <16 x i8>, ptr @llvm_mips_mulv_b_ARG2
  %2 = mul <16 x i8> %0, %1
  store <16 x i8> %2, ptr @llvm_mips_mulv_b_RES
  ret void
}

; CHECK: mulv_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: mulv.b
; CHECK: st.b
; CHECK: .size mulv_b_test

define void @mulv_h_test() nounwind {
entry:
  %0 = load <8 x i16>, ptr @llvm_mips_mulv_h_ARG1
  %1 = load <8 x i16>, ptr @llvm_mips_mulv_h_ARG2
  %2 = mul <8 x i16> %0, %1
  store <8 x i16> %2, ptr @llvm_mips_mulv_h_RES
  ret void
}

; CHECK: mulv_h_test:
; CHECK: ld.h
; CHECK: ld.h
; CHECK: mulv.h
; CHECK: st.h
; CHECK: .size mulv_h_test

define void @mulv_w_test() nounwind {
entry:
  %0 = load <4 x i32>, ptr @llvm_mips_mulv_w_ARG1
  %1 = load <4 x i32>, ptr @llvm_mips_mulv_w_ARG2
  %2 = mul <4 x i32> %0, %1
  store <4 x i32> %2, ptr @llvm_mips_mulv_w_RES
  ret void
}

; CHECK: mulv_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: mulv.w
; CHECK: st.w
; CHECK: .size mulv_w_test

define void @mulv_d_test() nounwind {
entry:
  %0 = load <2 x i64>, ptr @llvm_mips_mulv_d_ARG1
  %1 = load <2 x i64>, ptr @llvm_mips_mulv_d_ARG2
  %2 = mul <2 x i64> %0, %1
  store <2 x i64> %2, ptr @llvm_mips_mulv_d_RES
  ret void
}

; CHECK: mulv_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: mulv.d
; CHECK: st.d
; CHECK: .size mulv_d_test
;
