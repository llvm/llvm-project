; RUN: not llc --mtriple=loongarch64 --mattr=+lsx < %s 2>&1 | FileCheck %s

declare <16 x i8> @llvm.loongarch.lsx.vldrepl.b(i8*, i32)

define <16 x i8> @lsx_vldrepl_b(i8* %p, i32 %a) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vldrepl.b(i8* %p, i32 %a)
  ret <16 x i8> %res
}

declare <8 x i16> @llvm.loongarch.lsx.vldrepl.h(i8*, i32)

define <8 x i16> @lsx_vldrepl_h(i8* %p, i32 %a) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <8 x i16> @llvm.loongarch.lsx.vldrepl.h(i8* %p, i32 %a)
  ret <8 x i16> %res
}

declare <4 x i32> @llvm.loongarch.lsx.vldrepl.w(i8*, i32)

define <4 x i32> @lsx_vldrepl_w(i8* %p, i32 %a) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <4 x i32> @llvm.loongarch.lsx.vldrepl.w(i8* %p, i32 %a)
  ret <4 x i32> %res
}

declare <2 x i64> @llvm.loongarch.lsx.vldrepl.d(i8*, i32)

define <2 x i64> @lsx_vldrepl_d(i8* %p, i32 %a) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <2 x i64> @llvm.loongarch.lsx.vldrepl.d(i8* %p, i32 %a)
  ret <2 x i64> %res
}
