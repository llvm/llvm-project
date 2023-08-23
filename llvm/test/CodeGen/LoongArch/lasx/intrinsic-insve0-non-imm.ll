; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare <8 x i32> @llvm.loongarch.lasx.xvinsve0.w(<8 x i32>, <8 x i32>, i32)

define <8 x i32> @lasx_xvinsve0_w(<8 x i32> %va, <8 x i32> %vb, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <8 x i32> @llvm.loongarch.lasx.xvinsve0.w(<8 x i32> %va, <8 x i32> %vb, i32 %c)
  ret <8 x i32> %res
}

declare <4 x i64> @llvm.loongarch.lasx.xvinsve0.d(<4 x i64>, <4 x i64>, i32)

define <4 x i64> @lasx_xvinsve0_d(<4 x i64> %va, <4 x i64> %vb, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvinsve0.d(<4 x i64> %va, <4 x i64> %vb, i32 %c)
  ret <4 x i64> %res
}
