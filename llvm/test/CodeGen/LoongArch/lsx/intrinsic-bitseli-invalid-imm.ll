; RUN: not llc --mtriple=loongarch64 --mattr=+lsx < %s 2>&1 | FileCheck %s

declare <16 x i8> @llvm.loongarch.lsx.vbitseli.b(<16 x i8>, <16 x i8>, i32)

define <16 x i8> @lsx_vbitseli_b_lo(<16 x i8> %va, <16 x i8> %vb) nounwind {
; CHECK: llvm.loongarch.lsx.vbitseli.b: argument out of range
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vbitseli.b(<16 x i8> %va, <16 x i8> %vb, i32 -1)
  ret <16 x i8> %res
}

define <16 x i8> @lsx_vbitseli_b_hi(<16 x i8> %va, <16 x i8> %vb) nounwind {
; CHECK: llvm.loongarch.lsx.vbitseli.b: argument out of range
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vbitseli.b(<16 x i8> %va, <16 x i8> %vb, i32 256)
  ret <16 x i8> %res
}
