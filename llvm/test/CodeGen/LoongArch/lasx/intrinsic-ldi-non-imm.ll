; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare <4 x i64> @llvm.loongarch.lasx.xvldi(i32)

define <4 x i64> @lasx_xvldi(i32 %a) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvldi(i32 %a)
  ret <4 x i64> %res
}

declare <32 x i8> @llvm.loongarch.lasx.xvrepli.b(i32)

define <32 x i8> @lasx_xvrepli_b(i32 %a) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvrepli.b(i32 %a)
  ret <32 x i8> %res
}

declare <16 x i16> @llvm.loongarch.lasx.xvrepli.h(i32)

define <16 x i16> @lasx_xvrepli_h(i32 %a) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <16 x i16> @llvm.loongarch.lasx.xvrepli.h(i32 %a)
  ret <16 x i16> %res
}

declare <8 x i32> @llvm.loongarch.lasx.xvrepli.w(i32)

define <8 x i32> @lasx_xvrepli_w(i32 %a) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <8 x i32> @llvm.loongarch.lasx.xvrepli.w(i32 %a)
  ret <8 x i32> %res
}

declare <4 x i64> @llvm.loongarch.lasx.xvrepli.d(i32)

define <4 x i64> @lasx_xvrepli_d(i32 %a) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvrepli.d(i32 %a)
  ret <4 x i64> %res
}
