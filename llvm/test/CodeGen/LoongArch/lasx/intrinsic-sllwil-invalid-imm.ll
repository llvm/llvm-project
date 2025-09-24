; RUN: not llc --mtriple=loongarch32 --mattr=+32s,+lasx < %s 2>&1 | FileCheck %s
; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare <16 x i16> @llvm.loongarch.lasx.xvsllwil.h.b(<32 x i8>, i32)

define <16 x i16> @lasx_xvsllwil_h_b_lo(<32 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvsllwil.h.b: argument out of range
entry:
  %res = call <16 x i16> @llvm.loongarch.lasx.xvsllwil.h.b(<32 x i8> %va, i32 -1)
  ret <16 x i16> %res
}

define <16 x i16> @lasx_xvsllwil_h_b_hi(<32 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvsllwil.h.b: argument out of range
entry:
  %res = call <16 x i16> @llvm.loongarch.lasx.xvsllwil.h.b(<32 x i8> %va, i32 8)
  ret <16 x i16> %res
}

declare <8 x i32> @llvm.loongarch.lasx.xvsllwil.w.h(<16 x i16>, i32)

define <8 x i32> @lasx_xvsllwil_w_h_lo(<16 x i16> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvsllwil.w.h: argument out of range
entry:
  %res = call <8 x i32> @llvm.loongarch.lasx.xvsllwil.w.h(<16 x i16> %va, i32 -1)
  ret <8 x i32> %res
}

define <8 x i32> @lasx_xvsllwil_w_h_hi(<16 x i16> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvsllwil.w.h: argument out of range
entry:
  %res = call <8 x i32> @llvm.loongarch.lasx.xvsllwil.w.h(<16 x i16> %va, i32 16)
  ret <8 x i32> %res
}

declare <4 x i64> @llvm.loongarch.lasx.xvsllwil.d.w(<8 x i32>, i32)

define <4 x i64> @lasx_xvsllwil_d_w_lo(<8 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvsllwil.d.w: argument out of range
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvsllwil.d.w(<8 x i32> %va, i32 -1)
  ret <4 x i64> %res
}

define <4 x i64> @lasx_xvsllwil_d_w_hi(<8 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvsllwil.d.w: argument out of range
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvsllwil.d.w(<8 x i32> %va, i32 32)
  ret <4 x i64> %res
}

declare <16 x i16> @llvm.loongarch.lasx.xvsllwil.hu.bu(<32 x i8>, i32)

define <16 x i16> @lasx_xvsllwil_hu_bu_lo(<32 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvsllwil.hu.bu: argument out of range
entry:
  %res = call <16 x i16> @llvm.loongarch.lasx.xvsllwil.hu.bu(<32 x i8> %va, i32 -1)
  ret <16 x i16> %res
}

define <16 x i16> @lasx_xvsllwil_hu_bu_hi(<32 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvsllwil.hu.bu: argument out of range
entry:
  %res = call <16 x i16> @llvm.loongarch.lasx.xvsllwil.hu.bu(<32 x i8> %va, i32 8)
  ret <16 x i16> %res
}

declare <8 x i32> @llvm.loongarch.lasx.xvsllwil.wu.hu(<16 x i16>, i32)

define <8 x i32> @lasx_xvsllwil_wu_hu_lo(<16 x i16> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvsllwil.wu.hu: argument out of range
entry:
  %res = call <8 x i32> @llvm.loongarch.lasx.xvsllwil.wu.hu(<16 x i16> %va, i32 -1)
  ret <8 x i32> %res
}

define <8 x i32> @lasx_xvsllwil_wu_hu_hi(<16 x i16> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvsllwil.wu.hu: argument out of range
entry:
  %res = call <8 x i32> @llvm.loongarch.lasx.xvsllwil.wu.hu(<16 x i16> %va, i32 16)
  ret <8 x i32> %res
}

declare <4 x i64> @llvm.loongarch.lasx.xvsllwil.du.wu(<8 x i32>, i32)

define <4 x i64> @lasx_xvsllwil_du_wu_lo(<8 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvsllwil.du.wu: argument out of range
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvsllwil.du.wu(<8 x i32> %va, i32 -1)
  ret <4 x i64> %res
}

define <4 x i64> @lasx_xvsllwil_du_wu_hi(<8 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lasx.xvsllwil.du.wu: argument out of range
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvsllwil.du.wu(<8 x i32> %va, i32 32)
  ret <4 x i64> %res
}
