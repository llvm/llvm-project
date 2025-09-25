; RUN: not llc --mtriple=loongarch32 --mattr=+32s,+lasx < %s 2>&1 | FileCheck %s
; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare <4 x i64> @llvm.loongarch.lasx.xvldi(i32)

define <4 x i64> @lasx_xvldi_lo() nounwind {
; CHECK: llvm.loongarch.lasx.xvldi: argument out of range
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvldi(i32 -4097)
  ret <4 x i64> %res
}

define <4 x i64> @lasx_xvldi_hi() nounwind {
; CHECK: llvm.loongarch.lasx.xvldi: argument out of range
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvldi(i32 4096)
  ret <4 x i64> %res
}

declare <32 x i8> @llvm.loongarch.lasx.xvrepli.b(i32)

define <32 x i8> @lasx_xvrepli_b_lo() nounwind {
; CHECK: llvm.loongarch.lasx.xvrepli.b: argument out of range
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvrepli.b(i32 -513)
  ret <32 x i8> %res
}

define <32 x i8> @lasx_xvrepli_b_hi() nounwind {
; CHECK: llvm.loongarch.lasx.xvrepli.b: argument out of range
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvrepli.b(i32 512)
  ret <32 x i8> %res
}

declare <16 x i16> @llvm.loongarch.lasx.xvrepli.h(i32)

define <16 x i16> @lasx_xvrepli_h_lo() nounwind {
; CHECK: llvm.loongarch.lasx.xvrepli.h: argument out of range
entry:
  %res = call <16 x i16> @llvm.loongarch.lasx.xvrepli.h(i32 -513)
  ret <16 x i16> %res
}

define <16 x i16> @lasx_xvrepli_h_hi() nounwind {
; CHECK: llvm.loongarch.lasx.xvrepli.h: argument out of range
entry:
  %res = call <16 x i16> @llvm.loongarch.lasx.xvrepli.h(i32 512)
  ret <16 x i16> %res
}

declare <8 x i32> @llvm.loongarch.lasx.xvrepli.w(i32)

define <8 x i32> @lasx_xvrepli_w_lo() nounwind {
; CHECK: llvm.loongarch.lasx.xvrepli.w: argument out of range
entry:
  %res = call <8 x i32> @llvm.loongarch.lasx.xvrepli.w(i32 -513)
  ret <8 x i32> %res
}

define <8 x i32> @lasx_xvrepli_w_hi() nounwind {
; CHECK: llvm.loongarch.lasx.xvrepli.w: argument out of range
entry:
  %res = call <8 x i32> @llvm.loongarch.lasx.xvrepli.w(i32 512)
  ret <8 x i32> %res
}

declare <4 x i64> @llvm.loongarch.lasx.xvrepli.d(i32)

define <4 x i64> @lasx_xvrepli_d_lo() nounwind {
; CHECK: llvm.loongarch.lasx.xvrepli.d: argument out of range
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvrepli.d(i32 -513)
  ret <4 x i64> %res
}

define <4 x i64> @lasx_xvrepli_d_hi() nounwind {
; CHECK: llvm.loongarch.lasx.xvrepli.d: argument out of range
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvrepli.d(i32 512)
  ret <4 x i64> %res
}
