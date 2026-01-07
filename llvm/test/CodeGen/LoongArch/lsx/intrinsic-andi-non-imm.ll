; RUN: not llc --mtriple=loongarch32 --mattr=+32s,+lsx < %s 2>&1 | FileCheck %s
; RUN: not llc --mtriple=loongarch64 --mattr=+lsx < %s 2>&1 | FileCheck %s

declare <16 x i8> @llvm.loongarch.lsx.vandi.b(<16 x i8>, i32)

define <16 x i8> @lsx_vandi_b(<16 x i8> %va, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vandi.b(<16 x i8> %va, i32 %b)
  ret <16 x i8> %res
}
