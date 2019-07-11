; Test the MSA intrinsics that are encoded with the 3R instruction format.
; There are lots of these so this covers those beginning with 'i'

; RUN: llc -march=mips -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s

@llvm_mips_ilvev_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_ilvev_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_ilvev_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_ilvev_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_ilvev_b_ARG1
  %1 = load <16 x i8>, <16 x i8>* @llvm_mips_ilvev_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.ilvev.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, <16 x i8>* @llvm_mips_ilvev_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.ilvev.b(<16 x i8>, <16 x i8>) nounwind

; CHECK: llvm_mips_ilvev_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: ilvev.b
; CHECK: st.b
; CHECK: .size llvm_mips_ilvev_b_test
;
@llvm_mips_ilvev_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_ilvev_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_ilvev_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_ilvev_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_ilvev_h_ARG1
  %1 = load <8 x i16>, <8 x i16>* @llvm_mips_ilvev_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.ilvev.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_ilvev_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.ilvev.h(<8 x i16>, <8 x i16>) nounwind

; CHECK: llvm_mips_ilvev_h_test:
; CHECK: ld.h
; CHECK: ld.h
; CHECK: ilvev.h
; CHECK: st.h
; CHECK: .size llvm_mips_ilvev_h_test
;
@llvm_mips_ilvev_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_ilvev_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_ilvev_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_ilvev_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_ilvev_w_ARG1
  %1 = load <4 x i32>, <4 x i32>* @llvm_mips_ilvev_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.ilvev.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_ilvev_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.ilvev.w(<4 x i32>, <4 x i32>) nounwind

; CHECK: llvm_mips_ilvev_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: ilvev.w
; CHECK: st.w
; CHECK: .size llvm_mips_ilvev_w_test
;
@llvm_mips_ilvev_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_ilvev_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_ilvev_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_ilvev_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_ilvev_d_ARG1
  %1 = load <2 x i64>, <2 x i64>* @llvm_mips_ilvev_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.ilvev.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_ilvev_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.ilvev.d(<2 x i64>, <2 x i64>) nounwind

; CHECK: llvm_mips_ilvev_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: ilvev.d
; CHECK: st.d
; CHECK: .size llvm_mips_ilvev_d_test
;
@llvm_mips_ilvl_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_ilvl_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_ilvl_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_ilvl_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_ilvl_b_ARG1
  %1 = load <16 x i8>, <16 x i8>* @llvm_mips_ilvl_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.ilvl.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, <16 x i8>* @llvm_mips_ilvl_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.ilvl.b(<16 x i8>, <16 x i8>) nounwind

; CHECK: llvm_mips_ilvl_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: ilvl.b
; CHECK: st.b
; CHECK: .size llvm_mips_ilvl_b_test
;
@llvm_mips_ilvl_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_ilvl_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_ilvl_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_ilvl_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_ilvl_h_ARG1
  %1 = load <8 x i16>, <8 x i16>* @llvm_mips_ilvl_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.ilvl.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_ilvl_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.ilvl.h(<8 x i16>, <8 x i16>) nounwind

; CHECK: llvm_mips_ilvl_h_test:
; CHECK: ld.h
; CHECK: ld.h
; CHECK: ilvl.h
; CHECK: st.h
; CHECK: .size llvm_mips_ilvl_h_test
;
@llvm_mips_ilvl_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_ilvl_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_ilvl_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_ilvl_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_ilvl_w_ARG1
  %1 = load <4 x i32>, <4 x i32>* @llvm_mips_ilvl_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.ilvl.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_ilvl_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.ilvl.w(<4 x i32>, <4 x i32>) nounwind

; CHECK: llvm_mips_ilvl_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: ilvl.w
; CHECK: st.w
; CHECK: .size llvm_mips_ilvl_w_test
;
@llvm_mips_ilvl_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_ilvl_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_ilvl_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_ilvl_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_ilvl_d_ARG1
  %1 = load <2 x i64>, <2 x i64>* @llvm_mips_ilvl_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.ilvl.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_ilvl_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.ilvl.d(<2 x i64>, <2 x i64>) nounwind

; CHECK: llvm_mips_ilvl_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: ilvl.d
; CHECK: st.d
; CHECK: .size llvm_mips_ilvl_d_test
;
@llvm_mips_ilvod_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_ilvod_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_ilvod_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_ilvod_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_ilvod_b_ARG1
  %1 = load <16 x i8>, <16 x i8>* @llvm_mips_ilvod_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.ilvod.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, <16 x i8>* @llvm_mips_ilvod_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.ilvod.b(<16 x i8>, <16 x i8>) nounwind

; CHECK: llvm_mips_ilvod_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: ilvod.b
; CHECK: st.b
; CHECK: .size llvm_mips_ilvod_b_test
;
@llvm_mips_ilvod_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_ilvod_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_ilvod_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_ilvod_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_ilvod_h_ARG1
  %1 = load <8 x i16>, <8 x i16>* @llvm_mips_ilvod_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.ilvod.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_ilvod_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.ilvod.h(<8 x i16>, <8 x i16>) nounwind

; CHECK: llvm_mips_ilvod_h_test:
; CHECK: ld.h
; CHECK: ld.h
; CHECK: ilvod.h
; CHECK: st.h
; CHECK: .size llvm_mips_ilvod_h_test
;
@llvm_mips_ilvod_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_ilvod_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_ilvod_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_ilvod_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_ilvod_w_ARG1
  %1 = load <4 x i32>, <4 x i32>* @llvm_mips_ilvod_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.ilvod.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_ilvod_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.ilvod.w(<4 x i32>, <4 x i32>) nounwind

; CHECK: llvm_mips_ilvod_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: ilvod.w
; CHECK: st.w
; CHECK: .size llvm_mips_ilvod_w_test
;
@llvm_mips_ilvod_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_ilvod_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_ilvod_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_ilvod_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_ilvod_d_ARG1
  %1 = load <2 x i64>, <2 x i64>* @llvm_mips_ilvod_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.ilvod.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_ilvod_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.ilvod.d(<2 x i64>, <2 x i64>) nounwind

; CHECK: llvm_mips_ilvod_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: ilvod.d
; CHECK: st.d
; CHECK: .size llvm_mips_ilvod_d_test
;
@llvm_mips_ilvr_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_ilvr_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_ilvr_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_ilvr_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_ilvr_b_ARG1
  %1 = load <16 x i8>, <16 x i8>* @llvm_mips_ilvr_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.ilvr.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, <16 x i8>* @llvm_mips_ilvr_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.ilvr.b(<16 x i8>, <16 x i8>) nounwind

; CHECK: llvm_mips_ilvr_b_test:
; CHECK: ld.b
; CHECK: ld.b
; CHECK: ilvr.b
; CHECK: st.b
; CHECK: .size llvm_mips_ilvr_b_test
;
@llvm_mips_ilvr_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_ilvr_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_ilvr_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_ilvr_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_ilvr_h_ARG1
  %1 = load <8 x i16>, <8 x i16>* @llvm_mips_ilvr_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.ilvr.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_ilvr_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.ilvr.h(<8 x i16>, <8 x i16>) nounwind

; CHECK: llvm_mips_ilvr_h_test:
; CHECK: ld.h
; CHECK: ld.h
; CHECK: ilvr.h
; CHECK: st.h
; CHECK: .size llvm_mips_ilvr_h_test
;
@llvm_mips_ilvr_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_ilvr_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_ilvr_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_ilvr_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_ilvr_w_ARG1
  %1 = load <4 x i32>, <4 x i32>* @llvm_mips_ilvr_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.ilvr.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_ilvr_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.ilvr.w(<4 x i32>, <4 x i32>) nounwind

; CHECK: llvm_mips_ilvr_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: ilvr.w
; CHECK: st.w
; CHECK: .size llvm_mips_ilvr_w_test
;
@llvm_mips_ilvr_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_ilvr_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_ilvr_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_ilvr_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_ilvr_d_ARG1
  %1 = load <2 x i64>, <2 x i64>* @llvm_mips_ilvr_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.ilvr.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_ilvr_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.ilvr.d(<2 x i64>, <2 x i64>) nounwind

; CHECK: llvm_mips_ilvr_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: ilvr.d
; CHECK: st.d
; CHECK: .size llvm_mips_ilvr_d_test
;
