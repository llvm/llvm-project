; RUN: not llc --mtriple=loongarch32 --mattr=+32s,+lsx < %s 2>&1 | FileCheck %s
; RUN: not llc --mtriple=loongarch64 --mattr=+lsx < %s 2>&1 | FileCheck %s

declare <16 x i8> @llvm.loongarch.lsx.vnori.b(<16 x i8>, i32)

define <16 x i8> @lsx_vnori_b_lo(<16 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vnori.b: argument out of range
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vnori.b(<16 x i8> %va, i32 -1)
  ret <16 x i8> %res
}

define <16 x i8> @lsx_vnori_b_hi(<16 x i8> %va) nounwind {
; CHECK: llvm.loongarch.lsx.vnori.b: argument out of range
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vnori.b(<16 x i8> %va, i32 256)
  ret <16 x i8> %res
}
