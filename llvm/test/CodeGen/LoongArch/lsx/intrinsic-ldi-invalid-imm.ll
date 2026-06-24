; RUN: not llc --mtriple=loongarch32 --mattr=+32s,+lsx < %s 2>&1 | FileCheck %s
; RUN: not llc --mtriple=loongarch64 --mattr=+lsx < %s 2>&1 | FileCheck %s

declare <2 x i64> @llvm.loongarch.lsx.vldi(i32)

define <2 x i64> @lsx_vldi_lo() nounwind {
; CHECK: llvm.loongarch.lsx.vldi: argument out of range
entry:
  %res = call <2 x i64> @llvm.loongarch.lsx.vldi(i32 -4097)
  ret <2 x i64> %res
}

define <2 x i64> @lsx_vldi_hi() nounwind {
; CHECK: llvm.loongarch.lsx.vldi: argument out of range
entry:
  %res = call <2 x i64> @llvm.loongarch.lsx.vldi(i32 4096)
  ret <2 x i64> %res
}

declare <16 x i8> @llvm.loongarch.lsx.vrepli.b(i32)

define <16 x i8> @lsx_vrepli_b_lo() nounwind {
; CHECK: llvm.loongarch.lsx.vrepli.b: argument out of range
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vrepli.b(i32 -513)
  ret <16 x i8> %res
}

define <16 x i8> @lsx_vrepli_b_hi() nounwind {
; CHECK: llvm.loongarch.lsx.vrepli.b: argument out of range
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vrepli.b(i32 512)
  ret <16 x i8> %res
}

declare <8 x i16> @llvm.loongarch.lsx.vrepli.h(i32)

define <8 x i16> @lsx_vrepli_h_lo() nounwind {
; CHECK: llvm.loongarch.lsx.vrepli.h: argument out of range
entry:
  %res = call <8 x i16> @llvm.loongarch.lsx.vrepli.h(i32 -513)
  ret <8 x i16> %res
}

define <8 x i16> @lsx_vrepli_h_hi() nounwind {
; CHECK: llvm.loongarch.lsx.vrepli.h: argument out of range
entry:
  %res = call <8 x i16> @llvm.loongarch.lsx.vrepli.h(i32 512)
  ret <8 x i16> %res
}

declare <4 x i32> @llvm.loongarch.lsx.vrepli.w(i32)

define <4 x i32> @lsx_vrepli_w_lo() nounwind {
; CHECK: llvm.loongarch.lsx.vrepli.w: argument out of range
entry:
  %res = call <4 x i32> @llvm.loongarch.lsx.vrepli.w(i32 -513)
  ret <4 x i32> %res
}

define <4 x i32> @lsx_vrepli_w_hi() nounwind {
; CHECK: llvm.loongarch.lsx.vrepli.w: argument out of range
entry:
  %res = call <4 x i32> @llvm.loongarch.lsx.vrepli.w(i32 512)
  ret <4 x i32> %res
}

declare <2 x i64> @llvm.loongarch.lsx.vrepli.d(i32)

define <2 x i64> @lsx_vrepli_d_lo() nounwind {
; CHECK: llvm.loongarch.lsx.vrepli.d: argument out of range
entry:
  %res = call <2 x i64> @llvm.loongarch.lsx.vrepli.d(i32 -513)
  ret <2 x i64> %res
}

define <2 x i64> @lsx_vrepli_d_hi() nounwind {
; CHECK: llvm.loongarch.lsx.vrepli.d: argument out of range
entry:
  %res = call <2 x i64> @llvm.loongarch.lsx.vrepli.d(i32 512)
  ret <2 x i64> %res
}
