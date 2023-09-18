; RUN: not llc --mtriple=loongarch64 --mattr=+lsx < %s 2>&1 | FileCheck %s

declare <4 x i32> @llvm.loongarch.lsx.vpermi.w(<4 x i32>, <4 x i32>, i32)

define <4 x i32> @lsx_vpermi_w_lo(<4 x i32> %va, <4 x i32> %vb) nounwind {
; CHECK: llvm.loongarch.lsx.vpermi.w: argument out of range
entry:
  %res = call <4 x i32> @llvm.loongarch.lsx.vpermi.w(<4 x i32> %va, <4 x i32> %vb, i32 -1)
  ret <4 x i32> %res
}

define <4 x i32> @lsx_vpermi_w_hi(<4 x i32> %va, <4 x i32> %vb) nounwind {
; CHECK: llvm.loongarch.lsx.vpermi.w: argument out of range
entry:
  %res = call <4 x i32> @llvm.loongarch.lsx.vpermi.w(<4 x i32> %va, <4 x i32> %vb, i32 256)
  ret <4 x i32> %res
}
