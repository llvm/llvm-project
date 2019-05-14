; Test the MSA intrinsics that are encoded with the I5 instruction format.
; There are lots of these so this covers those beginning with 'c'

; RUN: llc -march=mips -mattr=+msa,+fp64 < %s | FileCheck %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64 < %s | FileCheck %s

@llvm_mips_ceqi_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_ceqi_b_RES1 = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16
@llvm_mips_ceqi_b_RES2 = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_ceqi_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_ceqi_b_ARG1
  %1 = tail call <16 x i8> @llvm.mips.ceqi.b(<16 x i8> %0, i32 14)
  store <16 x i8> %1, <16 x i8>* @llvm_mips_ceqi_b_RES1
  %2 = tail call <16 x i8> @llvm.mips.ceqi.b(<16 x i8> %0, i32 -14)
  store <16 x i8> %2, <16 x i8>* @llvm_mips_ceqi_b_RES2
  ret void
}

declare <16 x i8> @llvm.mips.ceqi.b(<16 x i8>, i32) nounwind

; CHECK: llvm_mips_ceqi_b_test:
; CHECK: ld.b [[RS:\$w[0-9]+]]
; CHECK: ceqi.b [[RD1:\$w[0-9]]], [[RS]], 14
; CHECK: st.b [[RD1]]
; CHECK: ceqi.b [[RD2:\$w[0-9]]], [[RS]], -14
; CHECK: st.b [[RD2]]
; CHECK: .size llvm_mips_ceqi_b_test
;
@llvm_mips_ceqi_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_ceqi_h_RES1 = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16
@llvm_mips_ceqi_h_RES2 = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_ceqi_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_ceqi_h_ARG1
  %1 = tail call <8 x i16> @llvm.mips.ceqi.h(<8 x i16> %0, i32 14)
  store <8 x i16> %1, <8 x i16>* @llvm_mips_ceqi_h_RES1
  %2 = tail call <8 x i16> @llvm.mips.ceqi.h(<8 x i16> %0, i32 -14)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_ceqi_h_RES2
  ret void
}

declare <8 x i16> @llvm.mips.ceqi.h(<8 x i16>, i32) nounwind

; CHECK: llvm_mips_ceqi_h_test:
; CHECK: ld.h [[RS:\$w[0-9]+]]
; CHECK: ceqi.h [[RD1:\$w[0-9]]], [[RS]], 14
; CHECK: st.h [[RD1]]
; CHECK: ceqi.h [[RD2:\$w[0-9]]], [[RS]], -14
; CHECK: st.h [[RD2]]
; CHECK: .size llvm_mips_ceqi_h_test
;
@llvm_mips_ceqi_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_ceqi_w_RES1 = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16
@llvm_mips_ceqi_w_RES2 = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_ceqi_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_ceqi_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.ceqi.w(<4 x i32> %0, i32 14)
  store <4 x i32> %1, <4 x i32>* @llvm_mips_ceqi_w_RES1
  %2 = tail call <4 x i32> @llvm.mips.ceqi.w(<4 x i32> %0, i32 -14)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_ceqi_w_RES2
  ret void
}

declare <4 x i32> @llvm.mips.ceqi.w(<4 x i32>, i32) nounwind

; CHECK: llvm_mips_ceqi_w_test:
; CHECK: ld.w [[RS:\$w[0-9]+]]
; CHECK: ceqi.w [[RD1:\$w[0-9]]], [[RS]], 14
; CHECK: st.w [[RD1]]
; CHECK: ceqi.w [[RD2:\$w[0-9]]], [[RS]], -14
; CHECK: st.w [[RD2]]
; CHECK: .size llvm_mips_ceqi_w_test
;
@llvm_mips_ceqi_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_ceqi_d_RES1 = global <2 x i64> <i64 0, i64 0>, align 16
@llvm_mips_ceqi_d_RES2 = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_ceqi_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_ceqi_d_ARG1
  %1 = tail call <2 x i64> @llvm.mips.ceqi.d(<2 x i64> %0, i32 14)
  store <2 x i64> %1, <2 x i64>* @llvm_mips_ceqi_d_RES1
  %2 = tail call <2 x i64> @llvm.mips.ceqi.d(<2 x i64> %0, i32 -14)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_ceqi_d_RES2
  ret void
}

declare <2 x i64> @llvm.mips.ceqi.d(<2 x i64>, i32) nounwind

; CHECK: llvm_mips_ceqi_d_test:
; CHECK: ld.d [[RS:\$w[0-9]+]]
; CHECK: ceqi.d [[RD1:\$w[0-9]]], [[RS]], 14
; CHECK: st.d [[RD1]]
; CHECK: ceqi.d [[RD2:\$w[0-9]]], [[RS]], -14
; CHECK: st.d [[RD2]]
; CHECK: .size llvm_mips_ceqi_d_test
;
@llvm_mips_clei_s_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_clei_s_b_RES1 = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16
@llvm_mips_clei_s_b_RES2 = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_clei_s_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_clei_s_b_ARG1
  %1 = tail call <16 x i8> @llvm.mips.clei.s.b(<16 x i8> %0, i32 14)
  store <16 x i8> %1, <16 x i8>* @llvm_mips_clei_s_b_RES1
  %2 = tail call <16 x i8> @llvm.mips.clei.s.b(<16 x i8> %0, i32 -14)
  store <16 x i8> %2, <16 x i8>* @llvm_mips_clei_s_b_RES2
  ret void
}

declare <16 x i8> @llvm.mips.clei.s.b(<16 x i8>, i32) nounwind

; CHECK: llvm_mips_clei_s_b_test:
; CHECK: ld.b [[RS:\$w[0-9]+]]
; CHECK: clei_s.b [[RD1:\$w[0-9]]], [[RS]], 14
; CHECK: st.b [[RD1]]
; CHECK: clei_s.b [[RD2:\$w[0-9]]], [[RS]], -14
; CHECK: st.b [[RD2]]
; CHECK: .size llvm_mips_clei_s_b_test
;
@llvm_mips_clei_s_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_clei_s_h_RES1 = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16
@llvm_mips_clei_s_h_RES2 = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_clei_s_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_clei_s_h_ARG1
  %1 = tail call <8 x i16> @llvm.mips.clei.s.h(<8 x i16> %0, i32 14)
  store <8 x i16> %1, <8 x i16>* @llvm_mips_clei_s_h_RES1
  %2 = tail call <8 x i16> @llvm.mips.clei.s.h(<8 x i16> %0, i32 -14)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_clei_s_h_RES2
  ret void
}

declare <8 x i16> @llvm.mips.clei.s.h(<8 x i16>, i32) nounwind

; CHECK: llvm_mips_clei_s_h_test:
; CHECK: ld.h [[RS:\$w[0-9]+]]
; CHECK: clei_s.h [[RD1:\$w[0-9]]], [[RS]], 14
; CHECK: st.h [[RD1]]
; CHECK: clei_s.h [[RD2:\$w[0-9]]], [[RS]], -14
; CHECK: st.h [[RD2]]
; CHECK: .size llvm_mips_clei_s_h_test
;
@llvm_mips_clei_s_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_clei_s_w_RES1 = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16
@llvm_mips_clei_s_w_RES2 = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_clei_s_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_clei_s_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.clei.s.w(<4 x i32> %0, i32 14)
  store <4 x i32> %1, <4 x i32>* @llvm_mips_clei_s_w_RES1
  %2 = tail call <4 x i32> @llvm.mips.clei.s.w(<4 x i32> %0, i32 -14)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_clei_s_w_RES2
  ret void
}

declare <4 x i32> @llvm.mips.clei.s.w(<4 x i32>, i32) nounwind

; CHECK: llvm_mips_clei_s_w_test:
; CHECK: ld.w [[RS:\$w[0-9]+]]
; CHECK: clei_s.w [[RD1:\$w[0-9]]], [[RS]], 14
; CHECK: st.w [[RD1]]
; CHECK: clei_s.w [[RD2:\$w[0-9]]], [[RS]], -14
; CHECK: st.w [[RD2]]
; CHECK: .size llvm_mips_clei_s_w_test
;
@llvm_mips_clei_s_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_clei_s_d_RES1 = global <2 x i64> <i64 0, i64 0>, align 16
@llvm_mips_clei_s_d_RES2 = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_clei_s_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_clei_s_d_ARG1
  %1 = tail call <2 x i64> @llvm.mips.clei.s.d(<2 x i64> %0, i32 14)
  store <2 x i64> %1, <2 x i64>* @llvm_mips_clei_s_d_RES1
  %2 = tail call <2 x i64> @llvm.mips.clei.s.d(<2 x i64> %0, i32 -14)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_clei_s_d_RES2
  ret void
}

declare <2 x i64> @llvm.mips.clei.s.d(<2 x i64>, i32) nounwind

; CHECK: llvm_mips_clei_s_d_test:
; CHECK: ld.d [[RS:\$w[0-9]+]]
; CHECK: clei_s.d [[RD1:\$w[0-9]]], [[RS]], 14
; CHECK: st.d [[RD1]]
; CHECK: clei_s.d [[RD2:\$w[0-9]]], [[RS]], -14
; CHECK: st.d [[RD2]]
; CHECK: .size llvm_mips_clei_s_d_test
;
@llvm_mips_clei_u_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_clei_u_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_clei_u_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_clei_u_b_ARG1
  %1 = tail call <16 x i8> @llvm.mips.clei.u.b(<16 x i8> %0, i32 14)
  store <16 x i8> %1, <16 x i8>* @llvm_mips_clei_u_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.clei.u.b(<16 x i8>, i32) nounwind

; CHECK: llvm_mips_clei_u_b_test:
; CHECK: ld.b
; CHECK: clei_u.b
; CHECK: st.b
; CHECK: .size llvm_mips_clei_u_b_test
;
@llvm_mips_clei_u_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_clei_u_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_clei_u_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_clei_u_h_ARG1
  %1 = tail call <8 x i16> @llvm.mips.clei.u.h(<8 x i16> %0, i32 14)
  store <8 x i16> %1, <8 x i16>* @llvm_mips_clei_u_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.clei.u.h(<8 x i16>, i32) nounwind

; CHECK: llvm_mips_clei_u_h_test:
; CHECK: ld.h
; CHECK: clei_u.h
; CHECK: st.h
; CHECK: .size llvm_mips_clei_u_h_test
;
@llvm_mips_clei_u_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_clei_u_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_clei_u_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_clei_u_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.clei.u.w(<4 x i32> %0, i32 14)
  store <4 x i32> %1, <4 x i32>* @llvm_mips_clei_u_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.clei.u.w(<4 x i32>, i32) nounwind

; CHECK: llvm_mips_clei_u_w_test:
; CHECK: ld.w
; CHECK: clei_u.w
; CHECK: st.w
; CHECK: .size llvm_mips_clei_u_w_test
;
@llvm_mips_clei_u_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_clei_u_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_clei_u_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_clei_u_d_ARG1
  %1 = tail call <2 x i64> @llvm.mips.clei.u.d(<2 x i64> %0, i32 14)
  store <2 x i64> %1, <2 x i64>* @llvm_mips_clei_u_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.clei.u.d(<2 x i64>, i32) nounwind

; CHECK: llvm_mips_clei_u_d_test:
; CHECK: ld.d
; CHECK: clei_u.d
; CHECK: st.d
; CHECK: .size llvm_mips_clei_u_d_test
;
@llvm_mips_clti_s_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_clti_s_b_RES1 = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16
@llvm_mips_clti_s_b_RES2 = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_clti_s_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_clti_s_b_ARG1
  %1 = tail call <16 x i8> @llvm.mips.clti.s.b(<16 x i8> %0, i32 14)
  store <16 x i8> %1, <16 x i8>* @llvm_mips_clti_s_b_RES1
  %2 = tail call <16 x i8> @llvm.mips.clti.s.b(<16 x i8> %0, i32 -14)
  store <16 x i8> %2, <16 x i8>* @llvm_mips_clti_s_b_RES2
  ret void
}

declare <16 x i8> @llvm.mips.clti.s.b(<16 x i8>, i32) nounwind

; CHECK: llvm_mips_clti_s_b_test:
; CHECK: ld.b [[RS:\$w[0-9]+]]
; CHECK: clti_s.b [[RD1:\$w[0-9]]], [[RS]], 14
; CHECK: st.b [[RD1]]
; CHECK: clti_s.b [[RD2:\$w[0-9]]], [[RS]], -14
; CHECK: st.b [[RD2]]
; CHECK: .size llvm_mips_clti_s_b_test
;
@llvm_mips_clti_s_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_clti_s_h_RES1 = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16
@llvm_mips_clti_s_h_RES2 = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_clti_s_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_clti_s_h_ARG1
  %1 = tail call <8 x i16> @llvm.mips.clti.s.h(<8 x i16> %0, i32 14)
  store <8 x i16> %1, <8 x i16>* @llvm_mips_clti_s_h_RES1
  %2 = tail call <8 x i16> @llvm.mips.clti.s.h(<8 x i16> %0, i32 -14)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_clti_s_h_RES2
  ret void
}

declare <8 x i16> @llvm.mips.clti.s.h(<8 x i16>, i32) nounwind

; CHECK: llvm_mips_clti_s_h_test:
; CHECK: ld.h [[RS:\$w[0-9]+]]
; CHECK: clti_s.h [[RD1:\$w[0-9]]], [[RS]], 14
; CHECK: st.h [[RD1]]
; CHECK: clti_s.h [[RD2:\$w[0-9]]], [[RS]], -14
; CHECK: st.h [[RD2]]
; CHECK: .size llvm_mips_clti_s_h_test
;
@llvm_mips_clti_s_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_clti_s_w_RES1 = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16
@llvm_mips_clti_s_w_RES2 = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_clti_s_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_clti_s_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.clti.s.w(<4 x i32> %0, i32 14)
  store <4 x i32> %1, <4 x i32>* @llvm_mips_clti_s_w_RES1
  %2 = tail call <4 x i32> @llvm.mips.clti.s.w(<4 x i32> %0, i32 -14)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_clti_s_w_RES2
  ret void
}

declare <4 x i32> @llvm.mips.clti.s.w(<4 x i32>, i32) nounwind

; CHECK: llvm_mips_clti_s_w_test:
; CHECK: ld.w [[RS:\$w[0-9]+]]
; CHECK: clti_s.w [[RD1:\$w[0-9]]], [[RS]], 14
; CHECK: st.w [[RD1]]
; CHECK: clti_s.w [[RD2:\$w[0-9]]], [[RS]], -14
; CHECK: st.w [[RD2]]
; CHECK: .size llvm_mips_clti_s_w_test
;
@llvm_mips_clti_s_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_clti_s_d_RES1 = global <2 x i64> <i64 0, i64 0>, align 16
@llvm_mips_clti_s_d_RES2 = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_clti_s_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_clti_s_d_ARG1
  %1 = tail call <2 x i64> @llvm.mips.clti.s.d(<2 x i64> %0, i32 14)
  store <2 x i64> %1, <2 x i64>* @llvm_mips_clti_s_d_RES1
  %2 = tail call <2 x i64> @llvm.mips.clti.s.d(<2 x i64> %0, i32 -14)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_clti_s_d_RES2
  ret void
}

declare <2 x i64> @llvm.mips.clti.s.d(<2 x i64>, i32) nounwind

; CHECK: llvm_mips_clti_s_d_test:
; CHECK: ld.d [[RS:\$w[0-9]+]]
; CHECK: clti_s.d [[RD1:\$w[0-9]]], [[RS]], 14
; CHECK: st.d [[RD1]]
; CHECK: clti_s.d [[RD2:\$w[0-9]]], [[RS]], -14
; CHECK: st.d [[RD2]]
; CHECK: .size llvm_mips_clti_s_d_test
;
@llvm_mips_clti_u_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_clti_u_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_clti_u_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_clti_u_b_ARG1
  %1 = tail call <16 x i8> @llvm.mips.clti.u.b(<16 x i8> %0, i32 14)
  store <16 x i8> %1, <16 x i8>* @llvm_mips_clti_u_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.clti.u.b(<16 x i8>, i32) nounwind

; CHECK: llvm_mips_clti_u_b_test:
; CHECK: ld.b
; CHECK: clti_u.b
; CHECK: st.b
; CHECK: .size llvm_mips_clti_u_b_test
;
@llvm_mips_clti_u_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_clti_u_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_clti_u_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_clti_u_h_ARG1
  %1 = tail call <8 x i16> @llvm.mips.clti.u.h(<8 x i16> %0, i32 14)
  store <8 x i16> %1, <8 x i16>* @llvm_mips_clti_u_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.clti.u.h(<8 x i16>, i32) nounwind

; CHECK: llvm_mips_clti_u_h_test:
; CHECK: ld.h
; CHECK: clti_u.h
; CHECK: st.h
; CHECK: .size llvm_mips_clti_u_h_test
;
@llvm_mips_clti_u_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_clti_u_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_clti_u_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_clti_u_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.clti.u.w(<4 x i32> %0, i32 14)
  store <4 x i32> %1, <4 x i32>* @llvm_mips_clti_u_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.clti.u.w(<4 x i32>, i32) nounwind

; CHECK: llvm_mips_clti_u_w_test:
; CHECK: ld.w
; CHECK: clti_u.w
; CHECK: st.w
; CHECK: .size llvm_mips_clti_u_w_test
;
@llvm_mips_clti_u_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_clti_u_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_clti_u_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_clti_u_d_ARG1
  %1 = tail call <2 x i64> @llvm.mips.clti.u.d(<2 x i64> %0, i32 14)
  store <2 x i64> %1, <2 x i64>* @llvm_mips_clti_u_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.clti.u.d(<2 x i64>, i32) nounwind

; CHECK: llvm_mips_clti_u_d_test:
; CHECK: ld.d
; CHECK: clti_u.d
; CHECK: st.d
; CHECK: .size llvm_mips_clti_u_d_test
;
