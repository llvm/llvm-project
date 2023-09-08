; RUN: not llc --mtriple=loongarch64 --mattr=+lsx < %s 2>&1 | FileCheck %s

declare <16 x i8> @llvm.loongarch.lsx.vmini.b(<16 x i8>, i32)

define <16 x i8> @lsx_vmini_b_lo(<16 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vmini.b: argument out of range
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vmini.b(<16 x i8> %va, i32 -17)
  ret <16 x i8> %res
}

define <16 x i8> @lsx_vmini_b_hi(<16 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vmini.b: argument out of range
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vmini.b(<16 x i8> %va, i32 16)
  ret <16 x i8> %res
}

declare <8 x i16> @llvm.loongarch.lsx.vmini.h(<8 x i16>, i32)

define <8 x i16> @lsx_vmini_h_lo(<8 x i16> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vmini.h: argument out of range
entry:
  %res = call <8 x i16> @llvm.loongarch.lsx.vmini.h(<8 x i16> %va, i32 -17)
  ret <8 x i16> %res
}

define <8 x i16> @lsx_vmini_h_hi(<8 x i16> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vmini.h: argument out of range
entry:
  %res = call <8 x i16> @llvm.loongarch.lsx.vmini.h(<8 x i16> %va, i32 16)
  ret <8 x i16> %res
}

declare <4 x i32> @llvm.loongarch.lsx.vmini.w(<4 x i32>, i32)

define <4 x i32> @lsx_vmini_w_lo(<4 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vmini.w: argument out of range
entry:
  %res = call <4 x i32> @llvm.loongarch.lsx.vmini.w(<4 x i32> %va, i32 -17)
  ret <4 x i32> %res
}

define <4 x i32> @lsx_vmini_w_hi(<4 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vmini.w: argument out of range
entry:
  %res = call <4 x i32> @llvm.loongarch.lsx.vmini.w(<4 x i32> %va, i32 16)
  ret <4 x i32> %res
}

declare <2 x i64> @llvm.loongarch.lsx.vmini.d(<2 x i64>, i32)

define <2 x i64> @lsx_vmini_d_lo(<2 x i64> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vmini.d: argument out of range
entry:
  %res = call <2 x i64> @llvm.loongarch.lsx.vmini.d(<2 x i64> %va, i32 -17)
  ret <2 x i64> %res
}

define <2 x i64> @lsx_vmini_d_hi(<2 x i64> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vmini.d: argument out of range
entry:
  %res = call <2 x i64> @llvm.loongarch.lsx.vmini.d(<2 x i64> %va, i32 16)
  ret <2 x i64> %res
}

declare <16 x i8> @llvm.loongarch.lsx.vmini.bu(<16 x i8>, i32)

define <16 x i8> @lsx_vmini_bu_lo(<16 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vmini.bu: argument out of range
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vmini.bu(<16 x i8> %va, i32 -1)
  ret <16 x i8> %res
}

define <16 x i8> @lsx_vmini_bu_hi(<16 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vmini.bu: argument out of range
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vmini.bu(<16 x i8> %va, i32 32)
  ret <16 x i8> %res
}

declare <8 x i16> @llvm.loongarch.lsx.vmini.hu(<8 x i16>, i32)

define <8 x i16> @lsx_vmini_hu_lo(<8 x i16> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vmini.hu: argument out of range
entry:
  %res = call <8 x i16> @llvm.loongarch.lsx.vmini.hu(<8 x i16> %va, i32 -1)
  ret <8 x i16> %res
}

define <8 x i16> @lsx_vmini_hu_hi(<8 x i16> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vmini.hu: argument out of range
entry:
  %res = call <8 x i16> @llvm.loongarch.lsx.vmini.hu(<8 x i16> %va, i32 32)
  ret <8 x i16> %res
}

declare <4 x i32> @llvm.loongarch.lsx.vmini.wu(<4 x i32>, i32)

define <4 x i32> @lsx_vmini_wu_lo(<4 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vmini.wu: argument out of range
entry:
  %res = call <4 x i32> @llvm.loongarch.lsx.vmini.wu(<4 x i32> %va, i32 -1)
  ret <4 x i32> %res
}

define <4 x i32> @lsx_vmini_wu_hi(<4 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vmini.wu: argument out of range
entry:
  %res = call <4 x i32> @llvm.loongarch.lsx.vmini.wu(<4 x i32> %va, i32 32)
  ret <4 x i32> %res
}

declare <2 x i64> @llvm.loongarch.lsx.vmini.du(<2 x i64>, i32)

define <2 x i64> @lsx_vmini_du_lo(<2 x i64> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vmini.du: argument out of range
entry:
  %res = call <2 x i64> @llvm.loongarch.lsx.vmini.du(<2 x i64> %va, i32 -1)
  ret <2 x i64> %res
}

define <2 x i64> @lsx_vmini_du_hi(<2 x i64> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vmini.du: argument out of range
entry:
  %res = call <2 x i64> @llvm.loongarch.lsx.vmini.du(<2 x i64> %va, i32 32)
  ret <2 x i64> %res
}
