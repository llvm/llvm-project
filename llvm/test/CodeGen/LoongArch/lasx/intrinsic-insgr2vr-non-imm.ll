; RUN: not llc --mtriple=loongarch32 --mattr=+32s,+lasx < %s 2>&1 | FileCheck %s
; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare <8 x i32> @llvm.loongarch.lasx.xvinsgr2vr.w(<8 x i32>, i32, i32)

define <8 x i32> @lasx_xvinsgr2vr_w(<8 x i32> %va, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <8 x i32> @llvm.loongarch.lasx.xvinsgr2vr.w(<8 x i32> %va, i32 1, i32 %b)
  ret <8 x i32> %res
}

declare <4 x i64> @llvm.loongarch.lasx.xvinsgr2vr.d(<4 x i64>, i64, i32)

define <4 x i64> @lasx_xvinsgr2vr_d(<4 x i64> %va, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvinsgr2vr.d(<4 x i64> %va, i64 1, i32 %b)
  ret <4 x i64> %res
}
