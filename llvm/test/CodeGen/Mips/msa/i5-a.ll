; Test the MSA intrinsics that are encoded with the I5 instruction format.
; There are lots of these so this covers those beginning with 'a'

; RUN: llc -mtriple=mips-elf -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s
; RUN: llc -mtriple=mipsel-elf -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s

@llvm_mips_addvi_b_ARG1 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, align 16
@llvm_mips_addvi_b_RES  = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, align 16

define void @llvm_mips_addvi_b_test() nounwind {
entry:
  %0 = load <16 x i8>, ptr @llvm_mips_addvi_b_ARG1
  %1 = tail call <16 x i8> @llvm.mips.addvi.b(<16 x i8> %0, i32 14)
  store <16 x i8> %1, ptr @llvm_mips_addvi_b_RES
  ret void
}

declare <16 x i8> @llvm.mips.addvi.b(<16 x i8>, i32) nounwind

; CHECK: llvm_mips_addvi_b_test:
; CHECK: ld.b
; CHECK: addvi.b
; CHECK: st.b
; CHECK: .size llvm_mips_addvi_b_test
;
@llvm_mips_addvi_h_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_addvi_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_addvi_h_test() nounwind {
entry:
  %0 = load <8 x i16>, ptr @llvm_mips_addvi_h_ARG1
  %1 = tail call <8 x i16> @llvm.mips.addvi.h(<8 x i16> %0, i32 14)
  store <8 x i16> %1, ptr @llvm_mips_addvi_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.addvi.h(<8 x i16>, i32) nounwind

; CHECK: llvm_mips_addvi_h_test:
; CHECK: ld.h
; CHECK: addvi.h
; CHECK: st.h
; CHECK: .size llvm_mips_addvi_h_test
;
@llvm_mips_addvi_w_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_addvi_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_addvi_w_test() nounwind {
entry:
  %0 = load <4 x i32>, ptr @llvm_mips_addvi_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.addvi.w(<4 x i32> %0, i32 14)
  store <4 x i32> %1, ptr @llvm_mips_addvi_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.addvi.w(<4 x i32>, i32) nounwind

; CHECK: llvm_mips_addvi_w_test:
; CHECK: ld.w
; CHECK: addvi.w
; CHECK: st.w
; CHECK: .size llvm_mips_addvi_w_test
;
@llvm_mips_addvi_d_ARG1 = global <2 x i64> <i64 0, i64 1>, align 16
@llvm_mips_addvi_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_addvi_d_test() nounwind {
entry:
  %0 = load <2 x i64>, ptr @llvm_mips_addvi_d_ARG1
  %1 = tail call <2 x i64> @llvm.mips.addvi.d(<2 x i64> %0, i32 14)
  store <2 x i64> %1, ptr @llvm_mips_addvi_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.addvi.d(<2 x i64>, i32) nounwind

; CHECK: llvm_mips_addvi_d_test:
; CHECK: ld.d
; CHECK: addvi.d
; CHECK: st.d
; CHECK: .size llvm_mips_addvi_d_test
;
