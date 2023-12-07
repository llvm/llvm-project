; RUN: not llc --mtriple=loongarch64 --mattr=+lsx < %s 2>&1 | FileCheck %s

declare <16 x i8> @llvm.loongarch.lsx.vldrepl.b(i8*, i32)

define <16 x i8> @lsx_vldrepl_b_lo(i8* %p) nounwind {
; CHECK: llvm.loongarch.lsx.vldrepl.b: argument out of range
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vldrepl.b(i8* %p, i32 -2049)
  ret <16 x i8> %res
}

define <16 x i8> @lsx_vldrepl_b_hi(i8* %p) nounwind {
; CHECK: llvm.loongarch.lsx.vldrepl.b: argument out of range
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vldrepl.b(i8* %p, i32 2048)
  ret <16 x i8> %res
}

declare <8 x i16> @llvm.loongarch.lsx.vldrepl.h(i8*, i32)

define <8 x i16> @lsx_vldrepl_h_lo(i8* %p) nounwind {
; CHECK: llvm.loongarch.lsx.vldrepl.h: argument out of range or not a multiple of 2.
entry:
  %res = call <8 x i16> @llvm.loongarch.lsx.vldrepl.h(i8* %p, i32 -2050)
  ret <8 x i16> %res
}

define <8 x i16> @lsx_vldrepl_h_hi(i8* %p) nounwind {
; CHECK: llvm.loongarch.lsx.vldrepl.h: argument out of range or not a multiple of 2.
entry:
  %res = call <8 x i16> @llvm.loongarch.lsx.vldrepl.h(i8* %p, i32 2048)
  ret <8 x i16> %res
}

declare <4 x i32> @llvm.loongarch.lsx.vldrepl.w(i8*, i32)

define <4 x i32> @lsx_vldrepl_w_lo(i8* %p) nounwind {
; CHECK: llvm.loongarch.lsx.vldrepl.w: argument out of range or not a multiple of 4.
entry:
  %res = call <4 x i32> @llvm.loongarch.lsx.vldrepl.w(i8* %p, i32 -2052)
  ret <4 x i32> %res
}

define <4 x i32> @lsx_vldrepl_w_hi(i8* %p) nounwind {
; CHECK: llvm.loongarch.lsx.vldrepl.w: argument out of range or not a multiple of 4.
entry:
  %res = call <4 x i32> @llvm.loongarch.lsx.vldrepl.w(i8* %p, i32 2048)
  ret <4 x i32> %res
}

declare <2 x i64> @llvm.loongarch.lsx.vldrepl.d(i8*, i32)

define <2 x i64> @lsx_vldrepl_d_lo(i8* %p) nounwind {
; CHECK: llvm.loongarch.lsx.vldrepl.d: argument out of range or not a multiple of 8.
entry:
  %res = call <2 x i64> @llvm.loongarch.lsx.vldrepl.d(i8* %p, i32 -2056)
  ret <2 x i64> %res
}

define <2 x i64> @lsx_vldrepl_d_hi(i8* %p) nounwind {
; CHECK: llvm.loongarch.lsx.vldrepl.d: argument out of range or not a multiple of 8.
entry:
  %res = call <2 x i64> @llvm.loongarch.lsx.vldrepl.d(i8* %p, i32 2048)
  ret <2 x i64> %res
}
