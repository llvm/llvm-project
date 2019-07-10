; Test the MSA intrinsics that are encoded with the 3R instruction format and
; use the result as a third operand and results in wider elements than the
; operands had.

; RUN: llc -march=mips -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s

@llvm_mips_dpadd_s_h_ARG2 = global <16 x i8> <i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15, i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23>, align 16
@llvm_mips_dpadd_s_h_ARG3 = global <16 x i8> <i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31, i8 32, i8 33, i8 34, i8 35, i8 36, i8 37, i8 38, i8 39>, align 16
@llvm_mips_dpadd_s_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_dpadd_s_h_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_dpadd_s_h_ARG2
  %1 = load <16 x i8>, <16 x i8>* @llvm_mips_dpadd_s_h_ARG3
  %2 = tail call <8 x i16> @llvm.mips.dpadd.s.h(<8 x i16> <i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4>, <16 x i8> %0, <16 x i8> %1)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_dpadd_s_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.dpadd.s.h(<8 x i16>, <16 x i8>, <16 x i8>) nounwind

; CHECK: llvm_mips_dpadd_s_h_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: ldi.h [[R1:\$w[0-9]+]],
; CHECK: dpadd_s.h [[R1]],
; CHECK: st.h
; CHECK: .size llvm_mips_dpadd_s_h_test
;
@llvm_mips_dpadd_s_w_ARG2 = global <8 x i16> <i16 4, i16 5, i16 6, i16 7, i16 8, i16 9, i16 10, i16 11>, align 16
@llvm_mips_dpadd_s_w_ARG3 = global <8 x i16> <i16 12, i16 13, i16 14, i16 15, i16 16, i16 17, i16 18, i16 19>, align 16
@llvm_mips_dpadd_s_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_dpadd_s_w_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_dpadd_s_w_ARG2
  %1 = load <8 x i16>, <8 x i16>* @llvm_mips_dpadd_s_w_ARG3
  %2 = tail call <4 x i32> @llvm.mips.dpadd.s.w(<4 x i32> <i32 4, i32 4, i32 4, i32 4>, <8 x i16> %0, <8 x i16> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_dpadd_s_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.dpadd.s.w(<4 x i32>, <8 x i16>, <8 x i16>) nounwind

; CHECK: llvm_mips_dpadd_s_w_test:
; CHECK: ld.h
; CHECK: ld.h
; CHECK: ldi.w [[R1:\$w[0-9]+]],
; CHECK: dpadd_s.w [[R1]],
; CHECK: st.w
; CHECK: .size llvm_mips_dpadd_s_w_test
;
@llvm_mips_dpadd_s_d_ARG2 = global <4 x i32> <i32 2, i32 3, i32 4, i32 5>, align 16
@llvm_mips_dpadd_s_d_ARG3 = global <4 x i32> <i32 6, i32 7, i32 8, i32 9>, align 16
@llvm_mips_dpadd_s_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_dpadd_s_d_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_dpadd_s_d_ARG2
  %1 = load <4 x i32>, <4 x i32>* @llvm_mips_dpadd_s_d_ARG3
  %2 = tail call <2 x i64> @llvm.mips.dpadd.s.d(<2 x i64> <i64 4, i64 4>, <4 x i32> %0, <4 x i32> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_dpadd_s_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.dpadd.s.d(<2 x i64>, <4 x i32>, <4 x i32>) nounwind

; CHECK: llvm_mips_dpadd_s_d_test:
; CHECK: ldi.d [[R1:\$w[0-9]+]],
; CHECK: ld.w
; CHECK: ld.w
; CHECK: dpadd_s.d [[R1]],
; CHECK: st.d
; CHECK: .size llvm_mips_dpadd_s_d_test
;
@llvm_mips_dpadd_u_h_ARG2 = global <16 x i8> <i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15, i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23>, align 16
@llvm_mips_dpadd_u_h_ARG3 = global <16 x i8> <i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31, i8 32, i8 33, i8 34, i8 35, i8 36, i8 37, i8 38, i8 39>, align 16
@llvm_mips_dpadd_u_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_dpadd_u_h_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_dpadd_u_h_ARG2
  %1 = load <16 x i8>, <16 x i8>* @llvm_mips_dpadd_u_h_ARG3
  %2 = tail call <8 x i16> @llvm.mips.dpadd.u.h(<8 x i16> <i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4>, <16 x i8> %0, <16 x i8> %1)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_dpadd_u_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.dpadd.u.h(<8 x i16>, <16 x i8>, <16 x i8>) nounwind

; CHECK: llvm_mips_dpadd_u_h_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: ldi.h [[R1:\$w[0-9]+]],
; CHECK: dpadd_u.h [[R1]],
; CHECK: st.h
; CHECK: .size llvm_mips_dpadd_u_h_test
;
@llvm_mips_dpadd_u_w_ARG2 = global <8 x i16> <i16 4, i16 5, i16 6, i16 7, i16 8, i16 9, i16 10, i16 11>, align 16
@llvm_mips_dpadd_u_w_ARG3 = global <8 x i16> <i16 12, i16 13, i16 14, i16 15, i16 16, i16 17, i16 18, i16 19>, align 16
@llvm_mips_dpadd_u_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_dpadd_u_w_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_dpadd_u_w_ARG2
  %1 = load <8 x i16>, <8 x i16>* @llvm_mips_dpadd_u_w_ARG3
  %2 = tail call <4 x i32> @llvm.mips.dpadd.u.w(<4 x i32> <i32 4, i32 4, i32 4, i32 4>, <8 x i16> %0, <8 x i16> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_dpadd_u_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.dpadd.u.w(<4 x i32>, <8 x i16>, <8 x i16>) nounwind

; CHECK: llvm_mips_dpadd_u_w_test:
; CHECK: ld.h
; CHECK: ld.h
; CHECK: ldi.w [[R1:\$w[0-9]+]],
; CHECK: dpadd_u.w [[R1]],
; CHECK: st.w
; CHECK: .size llvm_mips_dpadd_u_w_test
;
@llvm_mips_dpadd_u_d_ARG2 = global <4 x i32> <i32 2, i32 3, i32 4, i32 5>, align 16
@llvm_mips_dpadd_u_d_ARG3 = global <4 x i32> <i32 6, i32 7, i32 8, i32 9>, align 16
@llvm_mips_dpadd_u_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_dpadd_u_d_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_dpadd_u_d_ARG2
  %1 = load <4 x i32>, <4 x i32>* @llvm_mips_dpadd_u_d_ARG3
  %2 = tail call <2 x i64> @llvm.mips.dpadd.u.d(<2 x i64> <i64 4, i64 4>, <4 x i32> %0, <4 x i32> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_dpadd_u_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.dpadd.u.d(<2 x i64>, <4 x i32>, <4 x i32>) nounwind

; CHECK: llvm_mips_dpadd_u_d_test:
; CHECK: ldi.d [[R1:\$w[0-9]+]],
; CHECK: ld.w
; CHECK: ld.w
; CHECK: dpadd_u.d [[R1]],
; CHECK: st.d
; CHECK: .size llvm_mips_dpadd_u_d_test
;
@llvm_mips_dpsub_s_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_dpsub_s_h_ARG2 = global <16 x i8> <i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15, i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23>, align 16
@llvm_mips_dpsub_s_h_ARG3 = global <16 x i8> <i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31, i8 32, i8 33, i8 34, i8 35, i8 36, i8 37, i8 38, i8 39>, align 16
@llvm_mips_dpsub_s_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_dpsub_s_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_dpsub_s_h_ARG1
  %1 = load <16 x i8>, <16 x i8>* @llvm_mips_dpsub_s_h_ARG2
  %2 = load <16 x i8>, <16 x i8>* @llvm_mips_dpsub_s_h_ARG3
  %3 = tail call <8 x i16> @llvm.mips.dpsub.s.h(<8 x i16> %0, <16 x i8> %1, <16 x i8> %2)
  store <8 x i16> %3, <8 x i16>* @llvm_mips_dpsub_s_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.dpsub.s.h(<8 x i16>, <16 x i8>, <16 x i8>) nounwind

; CHECK: llvm_mips_dpsub_s_h_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: ld.h
; CHECK: dpsub_s.h
; CHECK: st.h
; CHECK: .size llvm_mips_dpsub_s_h_test
;
@llvm_mips_dpsub_s_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_dpsub_s_w_ARG2 = global <8 x i16> <i16 4, i16 5, i16 6, i16 7, i16 8, i16 9, i16 10, i16 11>, align 16
@llvm_mips_dpsub_s_w_ARG3 = global <8 x i16> <i16 12, i16 13, i16 14, i16 15, i16 16, i16 17, i16 18, i16 19>, align 16
@llvm_mips_dpsub_s_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_dpsub_s_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_dpsub_s_w_ARG1
  %1 = load <8 x i16>, <8 x i16>* @llvm_mips_dpsub_s_w_ARG2
  %2 = load <8 x i16>, <8 x i16>* @llvm_mips_dpsub_s_w_ARG3
  %3 = tail call <4 x i32> @llvm.mips.dpsub.s.w(<4 x i32> %0, <8 x i16> %1, <8 x i16> %2)
  store <4 x i32> %3, <4 x i32>* @llvm_mips_dpsub_s_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.dpsub.s.w(<4 x i32>, <8 x i16>, <8 x i16>) nounwind

; CHECK: llvm_mips_dpsub_s_w_test:
; CHECK: ld.h
; CHECK: ld.h
; CHECK: ld.w
; CHECK: dpsub_s.w
; CHECK: st.w
; CHECK: .size llvm_mips_dpsub_s_w_test
;
@llvm_mips_dpsub_s_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_dpsub_s_d_ARG2 = global <4 x i32> <i32 2, i32 3, i32 4, i32 5>, align 16
@llvm_mips_dpsub_s_d_ARG3 = global <4 x i32> <i32 6, i32 7, i32 8, i32 9>, align 16
@llvm_mips_dpsub_s_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_dpsub_s_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_dpsub_s_d_ARG1
  %1 = load <4 x i32>, <4 x i32>* @llvm_mips_dpsub_s_d_ARG2
  %2 = load <4 x i32>, <4 x i32>* @llvm_mips_dpsub_s_d_ARG3
  %3 = tail call <2 x i64> @llvm.mips.dpsub.s.d(<2 x i64> %0, <4 x i32> %1, <4 x i32> %2)
  store <2 x i64> %3, <2 x i64>* @llvm_mips_dpsub_s_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.dpsub.s.d(<2 x i64>, <4 x i32>, <4 x i32>) nounwind

; CHECK: llvm_mips_dpsub_s_d_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: ld.d
; CHECK: dpsub_s.d
; CHECK: st.d
; CHECK: .size llvm_mips_dpsub_s_d_test
;
@llvm_mips_dpsub_u_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_dpsub_u_h_ARG2 = global <16 x i8> <i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15, i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23>, align 16
@llvm_mips_dpsub_u_h_ARG3 = global <16 x i8> <i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31, i8 32, i8 33, i8 34, i8 35, i8 36, i8 37, i8 38, i8 39>, align 16
@llvm_mips_dpsub_u_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_dpsub_u_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_dpsub_u_h_ARG1
  %1 = load <16 x i8>, <16 x i8>* @llvm_mips_dpsub_u_h_ARG2
  %2 = load <16 x i8>, <16 x i8>* @llvm_mips_dpsub_u_h_ARG3
  %3 = tail call <8 x i16> @llvm.mips.dpsub.u.h(<8 x i16> %0, <16 x i8> %1, <16 x i8> %2)
  store <8 x i16> %3, <8 x i16>* @llvm_mips_dpsub_u_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.dpsub.u.h(<8 x i16>, <16 x i8>, <16 x i8>) nounwind

; CHECK: llvm_mips_dpsub_u_h_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: ld.h
; CHECK: dpsub_u.h
; CHECK: st.h
; CHECK: .size llvm_mips_dpsub_u_h_test
;
@llvm_mips_dpsub_u_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_dpsub_u_w_ARG2 = global <8 x i16> <i16 4, i16 5, i16 6, i16 7, i16 8, i16 9, i16 10, i16 11>, align 16
@llvm_mips_dpsub_u_w_ARG3 = global <8 x i16> <i16 12, i16 13, i16 14, i16 15, i16 16, i16 17, i16 18, i16 19>, align 16
@llvm_mips_dpsub_u_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_dpsub_u_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_dpsub_u_w_ARG1
  %1 = load <8 x i16>, <8 x i16>* @llvm_mips_dpsub_u_w_ARG2
  %2 = load <8 x i16>, <8 x i16>* @llvm_mips_dpsub_u_w_ARG3
  %3 = tail call <4 x i32> @llvm.mips.dpsub.u.w(<4 x i32> %0, <8 x i16> %1, <8 x i16> %2)
  store <4 x i32> %3, <4 x i32>* @llvm_mips_dpsub_u_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.dpsub.u.w(<4 x i32>, <8 x i16>, <8 x i16>) nounwind

; CHECK: llvm_mips_dpsub_u_w_test:
; CHECK: ld.h
; CHECK: ld.h
; CHECK: ld.w
; CHECK: dpsub_u.w
; CHECK: st.w
; CHECK: .size llvm_mips_dpsub_u_w_test
;
@llvm_mips_dpsub_u_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_dpsub_u_d_ARG2 = global <4 x i32> <i32 2, i32 3, i32 4, i32 5>, align 16
@llvm_mips_dpsub_u_d_ARG3 = global <4 x i32> <i32 6, i32 7, i32 8, i32 9>, align 16
@llvm_mips_dpsub_u_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_dpsub_u_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_dpsub_u_d_ARG1
  %1 = load <4 x i32>, <4 x i32>* @llvm_mips_dpsub_u_d_ARG2
  %2 = load <4 x i32>, <4 x i32>* @llvm_mips_dpsub_u_d_ARG3
  %3 = tail call <2 x i64> @llvm.mips.dpsub.u.d(<2 x i64> %0, <4 x i32> %1, <4 x i32> %2)
  store <2 x i64> %3, <2 x i64>* @llvm_mips_dpsub_u_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.dpsub.u.d(<2 x i64>, <4 x i32>, <4 x i32>) nounwind

; CHECK: llvm_mips_dpsub_u_d_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: ld.d
; CHECK: dpsub_u.d
; CHECK: st.d
; CHECK: .size llvm_mips_dpsub_u_d_test
;
