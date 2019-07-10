; Test the absence of the andi.b / and.v instructions

; RUN: llc -march=mips -mattr=+msa,+fp64,+mips32r2 -relocation-model=pic < %s | FileCheck %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64,+mips32r2 -relocation-model=pic < %s | FileCheck %s

@llvm_mips_bclr_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_bclr_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_bclr_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_bclr_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_bclr_b_ARG1
  %1 = load <16 x i8>, <16 x i8>* @llvm_mips_bclr_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.bclr.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, <16 x i8>* @llvm_mips_bclr_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.bclr.b(<16 x i8>, <16 x i8>) nounwind

; CHECK-LABEL: llvm_mips_bclr_b_test:
; CHECK-NOT: andi.b
; CHECK: bclr.b

@llvm_mips_bclr_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_bclr_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_bclr_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_bclr_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_bclr_h_ARG1
  %1 = load <8 x i16>, <8 x i16>* @llvm_mips_bclr_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.bclr.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_bclr_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.bclr.h(<8 x i16>, <8 x i16>) nounwind

; CHECK-LABEL: llvm_mips_bclr_h_test:
; CHECK-NOT: and.v
; CHECK: bclr.h

@llvm_mips_bclr_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_bclr_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_bclr_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_bclr_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_bclr_w_ARG1
  %1 = load <4 x i32>, <4 x i32>* @llvm_mips_bclr_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.bclr.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_bclr_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.bclr.w(<4 x i32>, <4 x i32>) nounwind

; CHECK-LABEL: llvm_mips_bclr_w_test:
; CHECK-NOT: and.v
; CHECK: bclr.w

@llvm_mips_bclr_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_bclr_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_bclr_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_bclr_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_bclr_d_ARG1
  %1 = load <2 x i64>, <2 x i64>* @llvm_mips_bclr_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.bclr.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_bclr_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.bclr.d(<2 x i64>, <2 x i64>) nounwind

; CHECK-LABEL: llvm_mips_bclr_d_test:
; CHECK-NOT: and.v
; CHECK: bclr.d

@llvm_mips_bneg_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_bneg_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_bneg_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_bneg_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_bneg_b_ARG1
  %1 = load <16 x i8>, <16 x i8>* @llvm_mips_bneg_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.bneg.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, <16 x i8>* @llvm_mips_bneg_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.bneg.b(<16 x i8>, <16 x i8>) nounwind

; CHECK-LABEL: llvm_mips_bneg_b_test:
; CHECK-NOT: andi.b
; CHECK: bneg.b

@llvm_mips_bneg_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_bneg_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_bneg_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_bneg_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_bneg_h_ARG1
  %1 = load <8 x i16>, <8 x i16>* @llvm_mips_bneg_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.bneg.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_bneg_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.bneg.h(<8 x i16>, <8 x i16>) nounwind

; CHECK-LABEL: llvm_mips_bneg_h_test:
; CHECK-NOT: and.v
; CHECK: bneg.h

@llvm_mips_bneg_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_bneg_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_bneg_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_bneg_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_bneg_w_ARG1
  %1 = load <4 x i32>, <4 x i32>* @llvm_mips_bneg_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.bneg.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_bneg_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.bneg.w(<4 x i32>, <4 x i32>) nounwind

; CHECK-LABEL: llvm_mips_bneg_w_test:
; CHECK-NOT: and.v
; CHECK: bneg.w

@llvm_mips_bneg_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_bneg_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_bneg_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_bneg_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_bneg_d_ARG1
  %1 = load <2 x i64>, <2 x i64>* @llvm_mips_bneg_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.bneg.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_bneg_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.bneg.d(<2 x i64>, <2 x i64>) nounwind

; CHECK-LABEL: llvm_mips_bneg_d_test:
; CHECK-NOT: and.v
; CHECK: bneg.d

@llvm_mips_bset_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_bset_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_bset_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_bset_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_bset_b_ARG1
  %1 = load <16 x i8>, <16 x i8>* @llvm_mips_bset_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.bset.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, <16 x i8>* @llvm_mips_bset_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.bset.b(<16 x i8>, <16 x i8>) nounwind

; CHECK-LABEL: llvm_mips_bset_b_test:
; CHECK-NOT: andi.b
; CHECK: bset.b

@llvm_mips_bset_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_bset_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_bset_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_bset_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_bset_h_ARG1
  %1 = load <8 x i16>, <8 x i16>* @llvm_mips_bset_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.bset.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_bset_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.bset.h(<8 x i16>, <8 x i16>) nounwind

; CHECK-LABEL: llvm_mips_bset_h_test:
; CHECK-NOT: and.v
; CHECK: bset.h

@llvm_mips_bset_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_bset_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_bset_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_bset_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_bset_w_ARG1
  %1 = load <4 x i32>, <4 x i32>* @llvm_mips_bset_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.bset.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_bset_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.bset.w(<4 x i32>, <4 x i32>) nounwind

; CHECK-LABEL: llvm_mips_bset_w_test:
; CHECK-NOT: and.v
; CHECK: bset.w

@llvm_mips_bset_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_bset_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_bset_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_bset_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_bset_d_ARG1
  %1 = load <2 x i64>, <2 x i64>* @llvm_mips_bset_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.bset.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_bset_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.bset.d(<2 x i64>, <2 x i64>) nounwind

; CHECK-LABEL: llvm_mips_bset_d_test:
; CHECK-NOT: and.v
; CHECK: bset.d

@llvm_mips_sll_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_sll_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_sll_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_sll_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_sll_b_ARG1
  %1 = load <16 x i8>, <16 x i8>* @llvm_mips_sll_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.sll.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, <16 x i8>* @llvm_mips_sll_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.sll.b(<16 x i8>, <16 x i8>) nounwind

; CHECK-LABEL: llvm_mips_sll_b_test:
; CHECK-NOT: andi.b
; CHECK: sll.b

@llvm_mips_sll_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_sll_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_sll_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_sll_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_sll_h_ARG1
  %1 = load <8 x i16>, <8 x i16>* @llvm_mips_sll_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.sll.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_sll_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.sll.h(<8 x i16>, <8 x i16>) nounwind

; CHECK-LABEL: llvm_mips_sll_h_test:
; CHECK-NOT: and.v
; CHECK: sll.h

@llvm_mips_sll_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_sll_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_sll_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_sll_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_sll_w_ARG1
  %1 = load <4 x i32>, <4 x i32>* @llvm_mips_sll_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.sll.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_sll_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.sll.w(<4 x i32>, <4 x i32>) nounwind

; CHECK-LABEL: llvm_mips_sll_w_test:
; CHECK-NOT: and.v
; CHECK: sll.w

@llvm_mips_sll_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_sll_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_sll_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_sll_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_sll_d_ARG1
  %1 = load <2 x i64>, <2 x i64>* @llvm_mips_sll_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.sll.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_sll_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.sll.d(<2 x i64>, <2 x i64>) nounwind

; CHECK-LABEL: llvm_mips_sll_d_test:
; CHECK-NOT: and.v
; CHECK: sll.d

@llvm_mips_sra_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_sra_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_sra_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_sra_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_sra_b_ARG1
  %1 = load <16 x i8>, <16 x i8>* @llvm_mips_sra_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.sra.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, <16 x i8>* @llvm_mips_sra_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.sra.b(<16 x i8>, <16 x i8>) nounwind

; CHECK-LABEL: llvm_mips_sra_b_test:
; CHECK-NOT: andi.b
; CHECK: sra.b

@llvm_mips_sra_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_sra_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_sra_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_sra_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_sra_h_ARG1
  %1 = load <8 x i16>, <8 x i16>* @llvm_mips_sra_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.sra.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_sra_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.sra.h(<8 x i16>, <8 x i16>) nounwind

; CHECK-LABEL: llvm_mips_sra_h_test:
; CHECK-NOT: and.v
; CHECK: sra.h

@llvm_mips_sra_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_sra_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_sra_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_sra_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_sra_w_ARG1
  %1 = load <4 x i32>, <4 x i32>* @llvm_mips_sra_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.sra.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_sra_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.sra.w(<4 x i32>, <4 x i32>) nounwind

; CHECK-LABEL: llvm_mips_sra_w_test:
; CHECK-NOT: and.v
; CHECK: sra.w

@llvm_mips_sra_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_sra_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_sra_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_sra_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_sra_d_ARG1
  %1 = load <2 x i64>, <2 x i64>* @llvm_mips_sra_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.sra.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_sra_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.sra.d(<2 x i64>, <2 x i64>) nounwind

; CHECK-LABEL: llvm_mips_sra_d_test:
; CHECK-NOT: and.v
; CHECK: sra.d

@llvm_mips_srl_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_srl_b_ARG2 = global <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, align 16
@llvm_mips_srl_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_srl_b_test() nounwind {
entry:
  %0 = load <16 x i8>, <16 x i8>* @llvm_mips_srl_b_ARG1
  %1 = load <16 x i8>, <16 x i8>* @llvm_mips_srl_b_ARG2
  %2 = tail call <16 x i8> @llvm.mips.srl.b(<16 x i8> %0, <16 x i8> %1)
  store <16 x i8> %2, <16 x i8>* @llvm_mips_srl_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.srl.b(<16 x i8>, <16 x i8>) nounwind

; CHECK-LABEL: llvm_mips_srl_b_test:
; CHECK-NOT: andi.b
; CHECK: srl.b

@llvm_mips_srl_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_srl_h_ARG2 = global <8 x i16> <i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, align 16
@llvm_mips_srl_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_srl_h_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_srl_h_ARG1
  %1 = load <8 x i16>, <8 x i16>* @llvm_mips_srl_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.srl.h(<8 x i16> %0, <8 x i16> %1)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_srl_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.srl.h(<8 x i16>, <8 x i16>) nounwind

; CHECK-LABEL: llvm_mips_srl_h_test:
; CHECK-NOT: and.v
; CHECK: srl.h

@llvm_mips_srl_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_srl_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_srl_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_srl_w_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_srl_w_ARG1
  %1 = load <4 x i32>, <4 x i32>* @llvm_mips_srl_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.srl.w(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_srl_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.srl.w(<4 x i32>, <4 x i32>) nounwind

; CHECK-LABEL: llvm_mips_srl_w_test:
; CHECK-NOT: and.v
; CHECK: srl.w

@llvm_mips_srl_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_srl_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_srl_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_srl_d_test() nounwind {
entry:
  %0 = load <2 x i64>, <2 x i64>* @llvm_mips_srl_d_ARG1
  %1 = load <2 x i64>, <2 x i64>* @llvm_mips_srl_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.srl.d(<2 x i64> %0, <2 x i64> %1)
  store <2 x i64> %2, <2 x i64>* @llvm_mips_srl_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.srl.d(<2 x i64>, <2 x i64>) nounwind

; CHECK-LABEL: llvm_mips_srl_d_test:
; CHECK-NOT: and.v
; CHECK: srl.d
