; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare <32 x i8> @llvm.loongarch.lasx.xvldrepl.b(i8*, i32)

define <32 x i8> @lasx_xvldrepl_b(i8* %p, i32 %a) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvldrepl.b(i8* %p, i32 %a)
  ret <32 x i8> %res
}

declare <16 x i16> @llvm.loongarch.lasx.xvldrepl.h(i8*, i32)

define <16 x i16> @lasx_xvldrepl_h(i8* %p, i32 %a) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <16 x i16> @llvm.loongarch.lasx.xvldrepl.h(i8* %p, i32 %a)
  ret <16 x i16> %res
}

declare <8 x i32> @llvm.loongarch.lasx.xvldrepl.w(i8*, i32)

define <8 x i32> @lasx_xvldrepl_w(i8* %p, i32 %a) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <8 x i32> @llvm.loongarch.lasx.xvldrepl.w(i8* %p, i32 %a)
  ret <8 x i32> %res
}

declare <4 x i64> @llvm.loongarch.lasx.xvldrepl.d(i8*, i32)

define <4 x i64> @lasx_xvldrepl_d(i8* %p, i32 %a) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <4 x i64> @llvm.loongarch.lasx.xvldrepl.d(i8* %p, i32 %a)
  ret <4 x i64> %res
}
