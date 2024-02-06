; RUN: not llc --mtriple=loongarch64 --mattr=+lsx < %s 2>&1 | FileCheck %s

declare <16 x i8> @llvm.loongarch.lsx.vld(ptr, i32)

define <16 x i8> @lsx_vld(ptr %p, i32 %a) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vld(ptr %p, i32 %a)
  ret <16 x i8> %res
}
