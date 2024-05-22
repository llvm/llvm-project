; RUN: not llc --mtriple=loongarch64 --mattr=+lsx < %s 2>&1 | FileCheck %s

declare <8 x i16> @llvm.loongarch.lsx.vsllwil.h.b(<16 x i8>, i32)

define <8 x i16> @lsx_vsllwil_h_b_lo(<16 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vsllwil.h.b: argument out of range
entry:
  %res = call <8 x i16> @llvm.loongarch.lsx.vsllwil.h.b(<16 x i8> %va, i32 -1)
  ret <8 x i16> %res
}

define <8 x i16> @lsx_vsllwil_h_b_hi(<16 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vsllwil.h.b: argument out of range
entry:
  %res = call <8 x i16> @llvm.loongarch.lsx.vsllwil.h.b(<16 x i8> %va, i32 8)
  ret <8 x i16> %res
}

declare <4 x i32> @llvm.loongarch.lsx.vsllwil.w.h(<8 x i16>, i32)

define <4 x i32> @lsx_vsllwil_w_h_lo(<8 x i16> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vsllwil.w.h: argument out of range
entry:
  %res = call <4 x i32> @llvm.loongarch.lsx.vsllwil.w.h(<8 x i16> %va, i32 -1)
  ret <4 x i32> %res
}

define <4 x i32> @lsx_vsllwil_w_h_hi(<8 x i16> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vsllwil.w.h: argument out of range
entry:
  %res = call <4 x i32> @llvm.loongarch.lsx.vsllwil.w.h(<8 x i16> %va, i32 16)
  ret <4 x i32> %res
}

declare <2 x i64> @llvm.loongarch.lsx.vsllwil.d.w(<4 x i32>, i32)

define <2 x i64> @lsx_vsllwil_d_w_lo(<4 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vsllwil.d.w: argument out of range
entry:
  %res = call <2 x i64> @llvm.loongarch.lsx.vsllwil.d.w(<4 x i32> %va, i32 -1)
  ret <2 x i64> %res
}

define <2 x i64> @lsx_vsllwil_d_w_hi(<4 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vsllwil.d.w: argument out of range
entry:
  %res = call <2 x i64> @llvm.loongarch.lsx.vsllwil.d.w(<4 x i32> %va, i32 32)
  ret <2 x i64> %res
}

declare <8 x i16> @llvm.loongarch.lsx.vsllwil.hu.bu(<16 x i8>, i32)

define <8 x i16> @lsx_vsllwil_hu_bu_lo(<16 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vsllwil.hu.bu: argument out of range
entry:
  %res = call <8 x i16> @llvm.loongarch.lsx.vsllwil.hu.bu(<16 x i8> %va, i32 -1)
  ret <8 x i16> %res
}

define <8 x i16> @lsx_vsllwil_hu_bu_hi(<16 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vsllwil.hu.bu: argument out of range
entry:
  %res = call <8 x i16> @llvm.loongarch.lsx.vsllwil.hu.bu(<16 x i8> %va, i32 8)
  ret <8 x i16> %res
}

declare <4 x i32> @llvm.loongarch.lsx.vsllwil.wu.hu(<8 x i16>, i32)

define <4 x i32> @lsx_vsllwil_wu_hu_lo(<8 x i16> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vsllwil.wu.hu: argument out of range
entry:
  %res = call <4 x i32> @llvm.loongarch.lsx.vsllwil.wu.hu(<8 x i16> %va, i32 -1)
  ret <4 x i32> %res
}

define <4 x i32> @lsx_vsllwil_wu_hu_hi(<8 x i16> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vsllwil.wu.hu: argument out of range
entry:
  %res = call <4 x i32> @llvm.loongarch.lsx.vsllwil.wu.hu(<8 x i16> %va, i32 16)
  ret <4 x i32> %res
}

declare <2 x i64> @llvm.loongarch.lsx.vsllwil.du.wu(<4 x i32>, i32)

define <2 x i64> @lsx_vsllwil_du_wu_lo(<4 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vsllwil.du.wu: argument out of range
entry:
  %res = call <2 x i64> @llvm.loongarch.lsx.vsllwil.du.wu(<4 x i32> %va, i32 -1)
  ret <2 x i64> %res
}

define <2 x i64> @lsx_vsllwil_du_wu_hi(<4 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vsllwil.du.wu: argument out of range
entry:
  %res = call <2 x i64> @llvm.loongarch.lsx.vsllwil.du.wu(<4 x i32> %va, i32 32)
  ret <2 x i64> %res
}
