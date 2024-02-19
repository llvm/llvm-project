; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare <32 x i8> @llvm.loongarch.lasx.xvextrins.b(<32 x i8>, <32 x i8>, i32)

define <32 x i8> @lasx_xvextrins_b(<32 x i8> %va, <32 x i8> %vb, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvextrins.b(<32 x i8> %va, <32 x i8> %vb, i32 %c)
  ret <32 x i8> %res
}

declare <16 x i16> @llvm.loongarch.lasx.xvextrins.h(<16 x i16>, <16 x i16>, i32)

define <16 x i16> @lasx_xvextrins_h(<16 x i16> %va, <16 x i16> %vb, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <16 x i16> @llvm.loongarch.lasx.xvextrins.h(<16 x i16> %va, <16 x i16> %vb, i32 %c)
  ret <16 x i16> %res
}

declare <8 x i32> @llvm.loongarch.lasx.xvextrins.w(<8 x i32>, <8 x i32>, i32)

define <8 x i32> @lasx_xvextrins_w(<8 x i32> %va, <8 x i32> %vb, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <8 x i32> @llvm.loongarch.lasx.xvextrins.w(<8 x i32> %va, <8 x i32> %vb, i32 %c)
  ret <8 x i32> %res
}

declare <4 x i64> @llvm.loongarch.lasx.xvextrins.d(<4 x i64>, <4 x i64>, i32)

define <4 x i64> @lasx_xvextrins_d(<4 x i64> %va, <4 x i64> %vb, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvextrins.d(<4 x i64> %va, <4 x i64> %vb, i32 %c)
  ret <4 x i64> %res
}
