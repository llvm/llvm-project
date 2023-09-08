; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare <32 x i8> @llvm.loongarch.lasx.xvld(i8*, i32)

define <32 x i8> @lasx_xvld(i8* %p, i32 %a) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvld(i8* %p, i32 %a)
  ret <32 x i8> %res
}
