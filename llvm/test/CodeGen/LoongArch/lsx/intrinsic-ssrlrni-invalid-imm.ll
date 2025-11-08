; RUN: not llc --mtriple=loongarch32 --mattr=+32s,+lsx < %s 2>&1 | FileCheck %s
; RUN: not llc --mtriple=loongarch64 --mattr=+lsx < %s 2>&1 | FileCheck %s

declare <16 x i8> @llvm.loongarch.lsx.vssrlrni.b.h(<16 x i8>, <16 x i8>, i32)

define <16 x i8> @lsx_vssrlrni_b_h_lo(<16 x i8> %va, <16 x i8> %vb) nounwind {
; CHECK: llvm.loongarch.lsx.vssrlrni.b.h: argument out of range
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vssrlrni.b.h(<16 x i8> %va, <16 x i8> %vb, i32 -1)
  ret <16 x i8> %res
}

define <16 x i8> @lsx_vssrlrni_b_h_hi(<16 x i8> %va, <16 x i8> %vb) nounwind {
; CHECK: llvm.loongarch.lsx.vssrlrni.b.h: argument out of range
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vssrlrni.b.h(<16 x i8> %va, <16 x i8> %vb, i32 16)
  ret <16 x i8> %res
}

declare <8 x i16> @llvm.loongarch.lsx.vssrlrni.h.w(<8 x i16>, <8 x i16>, i32)

define <8 x i16> @lsx_vssrlrni_h_w_lo(<8 x i16> %va, <8 x i16> %vb) nounwind {
; CHECK: llvm.loongarch.lsx.vssrlrni.h.w: argument out of range
entry:
  %res = call <8 x i16> @llvm.loongarch.lsx.vssrlrni.h.w(<8 x i16> %va, <8 x i16> %vb, i32 -1)
  ret <8 x i16> %res
}

define <8 x i16> @lsx_vssrlrni_h_w_hi(<8 x i16> %va, <8 x i16> %vb) nounwind {
; CHECK: llvm.loongarch.lsx.vssrlrni.h.w: argument out of range
entry:
  %res = call <8 x i16> @llvm.loongarch.lsx.vssrlrni.h.w(<8 x i16> %va, <8 x i16> %vb, i32 32)
  ret <8 x i16> %res
}

declare <4 x i32> @llvm.loongarch.lsx.vssrlrni.w.d(<4 x i32>, <4 x i32>, i32)

define <4 x i32> @lsx_vssrlrni_w_d_lo(<4 x i32> %va, <4 x i32> %vb) nounwind {
; CHECK: llvm.loongarch.lsx.vssrlrni.w.d: argument out of range
entry:
  %res = call <4 x i32> @llvm.loongarch.lsx.vssrlrni.w.d(<4 x i32> %va, <4 x i32> %vb, i32 -1)
  ret <4 x i32> %res
}

define <4 x i32> @lsx_vssrlrni_w_d_hi(<4 x i32> %va, <4 x i32> %vb) nounwind {
; CHECK: llvm.loongarch.lsx.vssrlrni.w.d: argument out of range
entry:
  %res = call <4 x i32> @llvm.loongarch.lsx.vssrlrni.w.d(<4 x i32> %va, <4 x i32> %vb, i32 64)
  ret <4 x i32> %res
}

declare <2 x i64> @llvm.loongarch.lsx.vssrlrni.d.q(<2 x i64>, <2 x i64>, i32)

define <2 x i64> @lsx_vssrlrni_d_q_lo(<2 x i64> %va, <2 x i64> %vb) nounwind {
; CHECK: llvm.loongarch.lsx.vssrlrni.d.q: argument out of range
entry:
  %res = call <2 x i64> @llvm.loongarch.lsx.vssrlrni.d.q(<2 x i64> %va, <2 x i64> %vb, i32 -1)
  ret <2 x i64> %res
}

define <2 x i64> @lsx_vssrlrni_d_q_hi(<2 x i64> %va, <2 x i64> %vb) nounwind {
; CHECK: llvm.loongarch.lsx.vssrlrni.d.q: argument out of range
entry:
  %res = call <2 x i64> @llvm.loongarch.lsx.vssrlrni.d.q(<2 x i64> %va, <2 x i64> %vb, i32 128)
  ret <2 x i64> %res
}

declare <16 x i8> @llvm.loongarch.lsx.vssrlrni.bu.h(<16 x i8>, <16 x i8>, i32)

define <16 x i8> @lsx_vssrlrni_bu_h_lo(<16 x i8> %va, <16 x i8> %vb) nounwind {
; CHECK: llvm.loongarch.lsx.vssrlrni.bu.h: argument out of range
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vssrlrni.bu.h(<16 x i8> %va, <16 x i8> %vb, i32 -1)
  ret <16 x i8> %res
}

define <16 x i8> @lsx_vssrlrni_bu_h_hi(<16 x i8> %va, <16 x i8> %vb) nounwind {
; CHECK: llvm.loongarch.lsx.vssrlrni.bu.h: argument out of range
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vssrlrni.bu.h(<16 x i8> %va, <16 x i8> %vb, i32 16)
  ret <16 x i8> %res
}

declare <8 x i16> @llvm.loongarch.lsx.vssrlrni.hu.w(<8 x i16>, <8 x i16>, i32)

define <8 x i16> @lsx_vssrlrni_hu_w_lo(<8 x i16> %va, <8 x i16> %vb) nounwind {
; CHECK: llvm.loongarch.lsx.vssrlrni.hu.w: argument out of range
entry:
  %res = call <8 x i16> @llvm.loongarch.lsx.vssrlrni.hu.w(<8 x i16> %va, <8 x i16> %vb, i32 -1)
  ret <8 x i16> %res
}

define <8 x i16> @lsx_vssrlrni_hu_w_hi(<8 x i16> %va, <8 x i16> %vb) nounwind {
; CHECK: llvm.loongarch.lsx.vssrlrni.hu.w: argument out of range
entry:
  %res = call <8 x i16> @llvm.loongarch.lsx.vssrlrni.hu.w(<8 x i16> %va, <8 x i16> %vb, i32 32)
  ret <8 x i16> %res
}

declare <4 x i32> @llvm.loongarch.lsx.vssrlrni.wu.d(<4 x i32>, <4 x i32>, i32)

define <4 x i32> @lsx_vssrlrni_wu_d_lo(<4 x i32> %va, <4 x i32> %vb) nounwind {
; CHECK: llvm.loongarch.lsx.vssrlrni.wu.d: argument out of range
entry:
  %res = call <4 x i32> @llvm.loongarch.lsx.vssrlrni.wu.d(<4 x i32> %va, <4 x i32> %vb, i32 -1)
  ret <4 x i32> %res
}

define <4 x i32> @lsx_vssrlrni_wu_d_hi(<4 x i32> %va, <4 x i32> %vb) nounwind {
; CHECK: llvm.loongarch.lsx.vssrlrni.wu.d: argument out of range
entry:
  %res = call <4 x i32> @llvm.loongarch.lsx.vssrlrni.wu.d(<4 x i32> %va, <4 x i32> %vb, i32 64)
  ret <4 x i32> %res
}

declare <2 x i64> @llvm.loongarch.lsx.vssrlrni.du.q(<2 x i64>, <2 x i64>, i32)

define <2 x i64> @lsx_vssrlrni_du_q_lo(<2 x i64> %va, <2 x i64> %vb) nounwind {
; CHECK: llvm.loongarch.lsx.vssrlrni.du.q: argument out of range
entry:
  %res = call <2 x i64> @llvm.loongarch.lsx.vssrlrni.du.q(<2 x i64> %va, <2 x i64> %vb, i32 -1)
  ret <2 x i64> %res
}

define <2 x i64> @lsx_vssrlrni_du_q_hi(<2 x i64> %va, <2 x i64> %vb) nounwind {
; CHECK: llvm.loongarch.lsx.vssrlrni.du.q: argument out of range
entry:
  %res = call <2 x i64> @llvm.loongarch.lsx.vssrlrni.du.q(<2 x i64> %va, <2 x i64> %vb, i32 128)
  ret <2 x i64> %res
}
