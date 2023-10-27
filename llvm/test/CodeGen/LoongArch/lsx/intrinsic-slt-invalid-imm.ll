; RUN: not llc --mtriple=loongarch64 --mattr=+lsx < %s 2>&1 | FileCheck %s

declare <16 x i8> @llvm.loongarch.lsx.vslti.b(<16 x i8>, i32)

define <16 x i8> @lsx_vslti_b_lo(<16 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vslti.b: argument out of range
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vslti.b(<16 x i8> %va, i32 -17)
  ret <16 x i8> %res
}

define <16 x i8> @lsx_vslti_b_hi(<16 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vslti.b: argument out of range
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vslti.b(<16 x i8> %va, i32 16)
  ret <16 x i8> %res
}

declare <8 x i16> @llvm.loongarch.lsx.vslti.h(<8 x i16>, i32)

define <8 x i16> @lsx_vslti_h_lo(<8 x i16> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vslti.h: argument out of range
entry:
  %res = call <8 x i16> @llvm.loongarch.lsx.vslti.h(<8 x i16> %va, i32 -17)
  ret <8 x i16> %res
}

define <8 x i16> @lsx_vslti_h_hi(<8 x i16> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vslti.h: argument out of range
entry:
  %res = call <8 x i16> @llvm.loongarch.lsx.vslti.h(<8 x i16> %va, i32 16)
  ret <8 x i16> %res
}

declare <4 x i32> @llvm.loongarch.lsx.vslti.w(<4 x i32>, i32)

define <4 x i32> @lsx_vslti_w_lo(<4 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vslti.w: argument out of range
entry:
  %res = call <4 x i32> @llvm.loongarch.lsx.vslti.w(<4 x i32> %va, i32 -17)
  ret <4 x i32> %res
}

define <4 x i32> @lsx_vslti_w_hi(<4 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vslti.w: argument out of range
entry:
  %res = call <4 x i32> @llvm.loongarch.lsx.vslti.w(<4 x i32> %va, i32 16)
  ret <4 x i32> %res
}

declare <2 x i64> @llvm.loongarch.lsx.vslti.d(<2 x i64>, i32)

define <2 x i64> @lsx_vslti_d_lo(<2 x i64> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vslti.d: argument out of range
entry:
  %res = call <2 x i64> @llvm.loongarch.lsx.vslti.d(<2 x i64> %va, i32 -17)
  ret <2 x i64> %res
}

define <2 x i64> @lsx_vslti_d_hi(<2 x i64> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vslti.d: argument out of range
entry:
  %res = call <2 x i64> @llvm.loongarch.lsx.vslti.d(<2 x i64> %va, i32 16)
  ret <2 x i64> %res
}

declare <16 x i8> @llvm.loongarch.lsx.vslti.bu(<16 x i8>, i32)

define <16 x i8> @lsx_vslti_bu_lo(<16 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vslti.bu: argument out of range
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vslti.bu(<16 x i8> %va, i32 -1)
  ret <16 x i8> %res
}

define <16 x i8> @lsx_vslti_bu_hi(<16 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vslti.bu: argument out of range
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vslti.bu(<16 x i8> %va, i32 32)
  ret <16 x i8> %res
}

declare <8 x i16> @llvm.loongarch.lsx.vslti.hu(<8 x i16>, i32)

define <8 x i16> @lsx_vslti_hu_lo(<8 x i16> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vslti.hu: argument out of range
entry:
  %res = call <8 x i16> @llvm.loongarch.lsx.vslti.hu(<8 x i16> %va, i32 -1)
  ret <8 x i16> %res
}

define <8 x i16> @lsx_vslti_hu_hi(<8 x i16> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vslti.hu: argument out of range
entry:
  %res = call <8 x i16> @llvm.loongarch.lsx.vslti.hu(<8 x i16> %va, i32 32)
  ret <8 x i16> %res
}

declare <4 x i32> @llvm.loongarch.lsx.vslti.wu(<4 x i32>, i32)

define <4 x i32> @lsx_vslti_wu_lo(<4 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vslti.wu: argument out of range
entry:
  %res = call <4 x i32> @llvm.loongarch.lsx.vslti.wu(<4 x i32> %va, i32 -1)
  ret <4 x i32> %res
}

define <4 x i32> @lsx_vslti_wu_hi(<4 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vslti.wu: argument out of range
entry:
  %res = call <4 x i32> @llvm.loongarch.lsx.vslti.wu(<4 x i32> %va, i32 32)
  ret <4 x i32> %res
}

declare <2 x i64> @llvm.loongarch.lsx.vslti.du(<2 x i64>, i32)

define <2 x i64> @lsx_vslti_du_lo(<2 x i64> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vslti.du: argument out of range
entry:
  %res = call <2 x i64> @llvm.loongarch.lsx.vslti.du(<2 x i64> %va, i32 -1)
  ret <2 x i64> %res
}

define <2 x i64> @lsx_vslti_du_hi(<2 x i64> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vslti.du: argument out of range
entry:
  %res = call <2 x i64> @llvm.loongarch.lsx.vslti.du(<2 x i64> %va, i32 32)
  ret <2 x i64> %res
}
