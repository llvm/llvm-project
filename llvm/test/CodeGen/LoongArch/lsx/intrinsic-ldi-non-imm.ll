; RUN: not llc --mtriple=loongarch64 --mattr=+lsx < %s 2>&1 | FileCheck %s

declare <2 x i64> @llvm.loongarch.lsx.vldi(i32)

define <2 x i64> @lsx_vldi(i32 %a) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <2 x i64> @llvm.loongarch.lsx.vldi(i32 %a)
  ret <2 x i64> %res
}

declare <16 x i8> @llvm.loongarch.lsx.vrepli.b(i32)

define <16 x i8> @lsx_vrepli_b(i32 %a) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vrepli.b(i32 %a)
  ret <16 x i8> %res
}

declare <8 x i16> @llvm.loongarch.lsx.vrepli.h(i32)

define <8 x i16> @lsx_vrepli_h(i32 %a) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <8 x i16> @llvm.loongarch.lsx.vrepli.h(i32 %a)
  ret <8 x i16> %res
}

declare <4 x i32> @llvm.loongarch.lsx.vrepli.w(i32)

define <4 x i32> @lsx_vrepli_w(i32 %a) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <4 x i32> @llvm.loongarch.lsx.vrepli.w(i32 %a)
  ret <4 x i32> %res
}

declare <2 x i64> @llvm.loongarch.lsx.vrepli.d(i32)

define <2 x i64> @lsx_vrepli_d(i32 %a) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <2 x i64> @llvm.loongarch.lsx.vrepli.d(i32 %a)
  ret <2 x i64> %res
}
