; RUN: not llc --mtriple=loongarch64 --mattr=+lsx < %s 2>&1 | FileCheck %s

declare <16 x i8> @llvm.loongarch.lsx.vbitclri.b(<16 x i8>, i32)

define <16 x i8> @lsx_vbitclri_b_lo(<16 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vbitclri.b: argument out of range
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vbitclri.b(<16 x i8> %va, i32 -1)
  ret <16 x i8> %res
}

define <16 x i8> @lsx_vbitclri_b_hi(<16 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vbitclri.b: argument out of range
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vbitclri.b(<16 x i8> %va, i32 8)
  ret <16 x i8> %res
}

declare <8 x i16> @llvm.loongarch.lsx.vbitclri.h(<8 x i16>, i32)

define <8 x i16> @lsx_vbitclri_h_lo(<8 x i16> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vbitclri.h: argument out of range
entry:
  %res = call <8 x i16> @llvm.loongarch.lsx.vbitclri.h(<8 x i16> %va, i32 -1)
  ret <8 x i16> %res
}

define <8 x i16> @lsx_vbitclri_h_hi(<8 x i16> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vbitclri.h: argument out of range
entry:
  %res = call <8 x i16> @llvm.loongarch.lsx.vbitclri.h(<8 x i16> %va, i32 16)
  ret <8 x i16> %res
}

declare <4 x i32> @llvm.loongarch.lsx.vbitclri.w(<4 x i32>, i32)

define <4 x i32> @lsx_vbitclri_w_lo(<4 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vbitclri.w: argument out of range
entry:
  %res = call <4 x i32> @llvm.loongarch.lsx.vbitclri.w(<4 x i32> %va, i32 -1)
  ret <4 x i32> %res
}

define <4 x i32> @lsx_vbitclri_w_hi(<4 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vbitclri.w: argument out of range
entry:
  %res = call <4 x i32> @llvm.loongarch.lsx.vbitclri.w(<4 x i32> %va, i32 32)
  ret <4 x i32> %res
}

declare <2 x i64> @llvm.loongarch.lsx.vbitclri.d(<2 x i64>, i32)

define <2 x i64> @lsx_vbitclri_d_lo(<2 x i64> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vbitclri.d: argument out of range
entry:
  %res = call <2 x i64> @llvm.loongarch.lsx.vbitclri.d(<2 x i64> %va, i32 -1)
  ret <2 x i64> %res
}

define <2 x i64> @lsx_vbitclri_d_hi(<2 x i64> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vbitclri.d: argument out of range
entry:
  %res = call <2 x i64> @llvm.loongarch.lsx.vbitclri.d(<2 x i64> %va, i32 64)
  ret <2 x i64> %res
}
