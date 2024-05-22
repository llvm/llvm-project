; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare <32 x i8> @llvm.loongarch.lasx.xvldrepl.b(i8*, i32)

define <32 x i8> @lasx_xvldrepl_b_lo(i8* %p) nounwind {
; CHECK: llvm.loongarch.lasx.xvldrepl.b: argument out of range
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvldrepl.b(i8* %p, i32 -2049)
  ret <32 x i8> %res
}

define <32 x i8> @lasx_xvldrepl_b_hi(i8* %p) nounwind {
; CHECK: llvm.loongarch.lasx.xvldrepl.b: argument out of range
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvldrepl.b(i8* %p, i32 2048)
  ret <32 x i8> %res
}

declare <16 x i16> @llvm.loongarch.lasx.xvldrepl.h(i8*, i32)

define <16 x i16> @lasx_xvldrepl_h_lo(i8* %p) nounwind {
; CHECK: llvm.loongarch.lasx.xvldrepl.h: argument out of range or not a multiple of 2.
entry:
  %res = call <16 x i16> @llvm.loongarch.lasx.xvldrepl.h(i8* %p, i32 -2050)
  ret <16 x i16> %res
}

define <16 x i16> @lasx_xvldrepl_h_hi(i8* %p) nounwind {
; CHECK: llvm.loongarch.lasx.xvldrepl.h: argument out of range or not a multiple of 2.
entry:
  %res = call <16 x i16> @llvm.loongarch.lasx.xvldrepl.h(i8* %p, i32 2048)
  ret <16 x i16> %res
}

declare <8 x i32> @llvm.loongarch.lasx.xvldrepl.w(i8*, i32)

define <8 x i32> @lasx_xvldrepl_w_lo(i8* %p) nounwind {
; CHECK: llvm.loongarch.lasx.xvldrepl.w: argument out of range or not a multiple of 4.
entry:
  %res = call <8 x i32> @llvm.loongarch.lasx.xvldrepl.w(i8* %p, i32 -2052)
  ret <8 x i32> %res
}

define <8 x i32> @lasx_xvldrepl_w_hi(i8* %p) nounwind {
; CHECK: llvm.loongarch.lasx.xvldrepl.w: argument out of range or not a multiple of 4.
entry:
  %res = call <8 x i32> @llvm.loongarch.lasx.xvldrepl.w(i8* %p, i32 2048)
  ret <8 x i32> %res
}

declare <4 x i64> @llvm.loongarch.lasx.xvldrepl.d(i8*, i32)

define <4 x i64> @lasx_xvldrepl_d_lo(i8* %p) nounwind {
; CHECK: llvm.loongarch.lasx.xvldrepl.d: argument out of range or not a multiple of 8.
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvldrepl.d(i8* %p, i32 -2056)
  ret <4 x i64> %res
}

define <4 x i64> @lasx_xvldrepl_d_hi(i8* %p) nounwind {
; CHECK: llvm.loongarch.lasx.xvldrepl.d: argument out of range or not a multiple of 8.
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvldrepl.d(i8* %p, i32 2048)
  ret <4 x i64> %res
}
