; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare <32 x i8> @llvm.loongarch.lasx.xvrepl128vei.b(<32 x i8>, i32)

define <32 x i8> @lasx_xvrepl128vei_b(<32 x i8> %va, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvrepl128vei.b(<32 x i8> %va, i32 %b)
  ret <32 x i8> %res
}

declare <16 x i16> @llvm.loongarch.lasx.xvrepl128vei.h(<16 x i16>, i32)

define <16 x i16> @lasx_xvrepl128vei_h(<16 x i16> %va, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <16 x i16> @llvm.loongarch.lasx.xvrepl128vei.h(<16 x i16> %va, i32 %b)
  ret <16 x i16> %res
}

declare <8 x i32> @llvm.loongarch.lasx.xvrepl128vei.w(<8 x i32>, i32)

define <8 x i32> @lasx_xvrepl128vei_w(<8 x i32> %va, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <8 x i32> @llvm.loongarch.lasx.xvrepl128vei.w(<8 x i32> %va, i32 %b)
  ret <8 x i32> %res
}

declare <4 x i64> @llvm.loongarch.lasx.xvrepl128vei.d(<4 x i64>, i32)

define <4 x i64> @lasx_xvrepl128vei_d(<4 x i64> %va, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvrepl128vei.d(<4 x i64> %va, i32 %b)
  ret <4 x i64> %res
}
