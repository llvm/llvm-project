; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare <32 x i8> @llvm.loongarch.lasx.xvxori.b(<32 x i8>, i32)

define <32 x i8> @lasx_xvxori_b(<32 x i8> %va, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvxori.b(<32 x i8> %va, i32 %b)
  ret <32 x i8> %res
}
