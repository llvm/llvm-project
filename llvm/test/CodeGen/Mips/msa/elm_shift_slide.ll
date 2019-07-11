; Test the MSA intrinsics that are encoded with the ELM instruction format and
; are either shifts or slides.

; RUN: llc -march=mips -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s

@llvm_mips_sldi_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_sldi_b_ARG2 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_sldi_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_sldi_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_sldi_b_ARG1
  %1 = load <16 x i8>, <16 x i8>* @llvm_mips_sldi_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.sldi.b(<16 x i8> %0, <16 x i8> %1, i32 1)
  store <16 x i8> %2, <16 x i8>* @llvm_mips_sldi_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.sldi.b(<16 x i8>, <16 x i8>, i32) nounwind

; CHECK: llvm_mips_sldi_b_test:
; CHECK: ld.b
; CHECK: sldi.b
; CHECK: st.b
; CHECK: .size llvm_mips_sldi_b_test
;
@llvm_mips_sldi_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_sldi_h_ARG2 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_sldi_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_sldi_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_sldi_h_ARG1
  %1 = load <8 x i16>, <8 x i16>* @llvm_mips_sldi_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.sldi.h(<8 x i16> %0, <8 x i16> %1, i32 1)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_sldi_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.sldi.h(<8 x i16>, <8 x i16>, i32) nounwind

; CHECK: llvm_mips_sldi_h_test:
; CHECK: ld.h
; CHECK: sldi.h
; CHECK: st.h
; CHECK: .size llvm_mips_sldi_h_test
;
@llvm_mips_sldi_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_sldi_w_ARG2 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_sldi_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_sldi_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_sldi_w_ARG1
  %1 = load <4 x i32>, <4 x i32>* @llvm_mips_sldi_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.sldi.w(<4 x i32> %0, <4 x i32> %1, i32 1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_sldi_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.sldi.w(<4 x i32>, <4 x i32>, i32) nounwind

; CHECK: llvm_mips_sldi_w_test:
; CHECK: ld.w
; CHECK: sldi.w
; CHECK: st.w
; CHECK: .size llvm_mips_sldi_w_test
;
@llvm_mips_sldi_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_sldi_d_ARG2 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_sldi_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_sldi_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_sldi_d_ARG1
  %1 = load <2 x i64>, <2 x i64>* @llvm_mips_sldi_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.sldi.d(<2 x i64> %0, <2 x i64> %1, i32 1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_sldi_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.sldi.d(<2 x i64>, <2 x i64>, i32) nounwind

; CHECK: llvm_mips_sldi_d_test:
; CHECK: ld.d
; CHECK: sldi.d
; CHECK: st.d
; CHECK: .size llvm_mips_sldi_d_test
;
@llvm_mips_splati_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_splati_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_splati_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_splati_b_ARG1
  %1 = tail call <16 x i8> @llvm.mips.splati.b(<16 x i8> %0, i32 1)
  store <16 x i8> %1, <16 x i8>* @llvm_mips_splati_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.splati.b(<16 x i8>, i32) nounwind

; CHECK: llvm_mips_splati_b_test:
; CHECK: ld.b
; CHECK: splati.b
; CHECK: st.b
; CHECK: .size llvm_mips_splati_b_test
;
@llvm_mips_splati_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_splati_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_splati_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_splati_h_ARG1
  %1 = tail call <8 x i16> @llvm.mips.splati.h(<8 x i16> %0, i32 1)
  store <8 x i16> %1, <8 x i16>* @llvm_mips_splati_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.splati.h(<8 x i16>, i32) nounwind

; CHECK: llvm_mips_splati_h_test:
; CHECK: ld.h
; CHECK: splati.h
; CHECK: st.h
; CHECK: .size llvm_mips_splati_h_test
;
@llvm_mips_splati_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_splati_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_splati_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_splati_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.splati.w(<4 x i32> %0, i32 1)
  store <4 x i32> %1, <4 x i32>* @llvm_mips_splati_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.splati.w(<4 x i32>, i32) nounwind

; CHECK: llvm_mips_splati_w_test:
; CHECK: ld.w
; CHECK: splati.w
; CHECK: st.w
; CHECK: .size llvm_mips_splati_w_test
;
@llvm_mips_splati_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_splati_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_splati_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_splati_d_ARG1
  %1 = tail call <2 x i64> @llvm.mips.splati.d(<2 x i64> %0, i32 1)
  store <2 x i64> %1, <2 x i64>* @llvm_mips_splati_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.splati.d(<2 x i64>, i32) nounwind

; CHECK: llvm_mips_splati_d_test:
; CHECK: ld.d
; CHECK: splati.d
; CHECK: st.d
; CHECK: .size llvm_mips_splati_d_test
;
