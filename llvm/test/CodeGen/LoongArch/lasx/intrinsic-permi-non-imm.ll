; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare <8 x i32> @llvm.loongarch.lasx.xvpermi.w(<8 x i32>, <8 x i32>, i32)

define <8 x i32> @lasx_xvpermi_w(<8 x i32> %va, <8 x i32> %vb, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <8 x i32> @llvm.loongarch.lasx.xvpermi.w(<8 x i32> %va, <8 x i32> %vb, i32 %c)
  ret <8 x i32> %res
}

declare <4 x i64> @llvm.loongarch.lasx.xvpermi.d(<4 x i64>, i32)

define <4 x i64> @lasx_xvpermi_d(<4 x i64> %va, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvpermi.d(<4 x i64> %va, i32 %b)
  ret <4 x i64> %res
}

declare <32 x i8> @llvm.loongarch.lasx.xvpermi.q(<32 x i8>, <32 x i8>, i32)

define <32 x i8> @lasx_xvpermi_q(<32 x i8> %va, <32 x i8> %vb, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvpermi.q(<32 x i8> %va, <32 x i8> %vb, i32 %c)
  ret <32 x i8> %res
}
