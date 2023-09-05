; RUN: not llc --mtriple=loongarch64 --mattr=+lsx < %s 2>&1 | FileCheck %s

declare <16 x i8> @llvm.loongarch.lsx.vslei.b(<16 x i8>, i32)

define <16 x i8> @lsx_vslei_b_lo(<16 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vslei.b: argument out of range
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vslei.b(<16 x i8> %va, i32 -17)
  ret <16 x i8> %res
}

define <16 x i8> @lsx_vslei_b_hi(<16 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vslei.b: argument out of range
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vslei.b(<16 x i8> %va, i32 16)
  ret <16 x i8> %res
}

declare <8 x i16> @llvm.loongarch.lsx.vslei.h(<8 x i16>, i32)

define <8 x i16> @lsx_vslei_h_lo(<8 x i16> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vslei.h: argument out of range
entry:
  %res = call <8 x i16> @llvm.loongarch.lsx.vslei.h(<8 x i16> %va, i32 -17)
  ret <8 x i16> %res
}

define <8 x i16> @lsx_vslei_h_hi(<8 x i16> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vslei.h: argument out of range
entry:
  %res = call <8 x i16> @llvm.loongarch.lsx.vslei.h(<8 x i16> %va, i32 16)
  ret <8 x i16> %res
}

declare <4 x i32> @llvm.loongarch.lsx.vslei.w(<4 x i32>, i32)

define <4 x i32> @lsx_vslei_w_lo(<4 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vslei.w: argument out of range
entry:
  %res = call <4 x i32> @llvm.loongarch.lsx.vslei.w(<4 x i32> %va, i32 -17)
  ret <4 x i32> %res
}

define <4 x i32> @lsx_vslei_w_hi(<4 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vslei.w: argument out of range
entry:
  %res = call <4 x i32> @llvm.loongarch.lsx.vslei.w(<4 x i32> %va, i32 16)
  ret <4 x i32> %res
}

declare <2 x i64> @llvm.loongarch.lsx.vslei.d(<2 x i64>, i32)

define <2 x i64> @lsx_vslei_d_lo(<2 x i64> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vslei.d: argument out of range
entry:
  %res = call <2 x i64> @llvm.loongarch.lsx.vslei.d(<2 x i64> %va, i32 -17)
  ret <2 x i64> %res
}

define <2 x i64> @lsx_vslei_d_hi(<2 x i64> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vslei.d: argument out of range
entry:
  %res = call <2 x i64> @llvm.loongarch.lsx.vslei.d(<2 x i64> %va, i32 16)
  ret <2 x i64> %res
}

declare <16 x i8> @llvm.loongarch.lsx.vslei.bu(<16 x i8>, i32)

define <16 x i8> @lsx_vslei_bu_lo(<16 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vslei.bu: argument out of range
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vslei.bu(<16 x i8> %va, i32 -1)
  ret <16 x i8> %res
}

define <16 x i8> @lsx_vslei_bu_hi(<16 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vslei.bu: argument out of range
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vslei.bu(<16 x i8> %va, i32 32)
  ret <16 x i8> %res
}

declare <8 x i16> @llvm.loongarch.lsx.vslei.hu(<8 x i16>, i32)

define <8 x i16> @lsx_vslei_hu_lo(<8 x i16> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vslei.hu: argument out of range
entry:
  %res = call <8 x i16> @llvm.loongarch.lsx.vslei.hu(<8 x i16> %va, i32 -1)
  ret <8 x i16> %res
}

define <8 x i16> @lsx_vslei_hu_hi(<8 x i16> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vslei.hu: argument out of range
entry:
  %res = call <8 x i16> @llvm.loongarch.lsx.vslei.hu(<8 x i16> %va, i32 32)
  ret <8 x i16> %res
}

declare <4 x i32> @llvm.loongarch.lsx.vslei.wu(<4 x i32>, i32)

define <4 x i32> @lsx_vslei_wu_lo(<4 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vslei.wu: argument out of range
entry:
  %res = call <4 x i32> @llvm.loongarch.lsx.vslei.wu(<4 x i32> %va, i32 -1)
  ret <4 x i32> %res
}

define <4 x i32> @lsx_vslei_wu_hi(<4 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vslei.wu: argument out of range
entry:
  %res = call <4 x i32> @llvm.loongarch.lsx.vslei.wu(<4 x i32> %va, i32 32)
  ret <4 x i32> %res
}

declare <2 x i64> @llvm.loongarch.lsx.vslei.du(<2 x i64>, i32)

define <2 x i64> @lsx_vslei_du_lo(<2 x i64> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vslei.du: argument out of range
entry:
  %res = call <2 x i64> @llvm.loongarch.lsx.vslei.du(<2 x i64> %va, i32 -1)
  ret <2 x i64> %res
}

define <2 x i64> @lsx_vslei_du_hi(<2 x i64> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vslei.du: argument out of range
entry:
  %res = call <2 x i64> @llvm.loongarch.lsx.vslei.du(<2 x i64> %va, i32 32)
  ret <2 x i64> %res
}
