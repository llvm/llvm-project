; RUN: not llc --mtriple=loongarch32 --mattr=+32s,+lsx < %s 2>&1 | FileCheck %s
; RUN: not llc --mtriple=loongarch64 --mattr=+lsx < %s 2>&1 | FileCheck %s

declare <4 x i32> @llvm.loongarch.lsx.vpermi.w(<4 x i32>, <4 x i32>, i32)

define <4 x i32> @lsx_vpermi_w(<4 x i32> %va, <4 x i32> %vb, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <4 x i32> @llvm.loongarch.lsx.vpermi.w(<4 x i32> %va, <4 x i32> %vb, i32 %c)
  ret <4 x i32> %res
}
