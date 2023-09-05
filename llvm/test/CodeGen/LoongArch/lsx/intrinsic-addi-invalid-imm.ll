; RUN: not llc --mtriple=loongarch64 --mattr=+lsx < %s 2>&1 | FileCheck %s

declare <16 x i8> @llvm.loongarch.lsx.vaddi.bu(<16 x i8>, i32)

define <16 x i8> @lsx_vaddi_bu_lo(<16 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vaddi.bu: argument out of range
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vaddi.bu(<16 x i8> %va, i32 -1)
  ret <16 x i8> %res
}

define <16 x i8> @lsx_vaddi_bu_hi(<16 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vaddi.bu: argument out of range
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vaddi.bu(<16 x i8> %va, i32 32)
  ret <16 x i8> %res
}

declare <8 x i16> @llvm.loongarch.lsx.vaddi.hu(<8 x i16>, i32)

define <8 x i16> @lsx_vaddi_hu_lo(<8 x i16> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vaddi.hu: argument out of range
entry:
  %res = call <8 x i16> @llvm.loongarch.lsx.vaddi.hu(<8 x i16> %va, i32 -1)
  ret <8 x i16> %res
}

define <8 x i16> @lsx_vaddi_hu_hi(<8 x i16> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vaddi.hu: argument out of range
entry:
  %res = call <8 x i16> @llvm.loongarch.lsx.vaddi.hu(<8 x i16> %va, i32 32)
  ret <8 x i16> %res
}

declare <4 x i32> @llvm.loongarch.lsx.vaddi.wu(<4 x i32>, i32)

define <4 x i32> @lsx_vaddi_wu_lo(<4 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vaddi.wu: argument out of range
entry:
  %res = call <4 x i32> @llvm.loongarch.lsx.vaddi.wu(<4 x i32> %va, i32 -1)
  ret <4 x i32> %res
}

define <4 x i32> @lsx_vaddi_wu_hi(<4 x i32> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vaddi.wu: argument out of range
entry:
  %res = call <4 x i32> @llvm.loongarch.lsx.vaddi.wu(<4 x i32> %va, i32 32)
  ret <4 x i32> %res
}

declare <2 x i64> @llvm.loongarch.lsx.vaddi.du(<2 x i64>, i32)

define <2 x i64> @lsx_vaddi_du_lo(<2 x i64> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vaddi.du: argument out of range
entry:
  %res = call <2 x i64> @llvm.loongarch.lsx.vaddi.du(<2 x i64> %va, i32 -1)
  ret <2 x i64> %res
}

define <2 x i64> @lsx_vaddi_du_hi(<2 x i64> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vaddi.du: argument out of range
entry:
  %res = call <2 x i64> @llvm.loongarch.lsx.vaddi.du(<2 x i64> %va, i32 32)
  ret <2 x i64> %res
}
