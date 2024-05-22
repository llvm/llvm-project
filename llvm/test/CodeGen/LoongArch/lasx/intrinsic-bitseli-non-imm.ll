; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare <32 x i8> @llvm.loongarch.lasx.xvbitseli.b(<32 x i8>, <32 x i8>, i32)

define <32 x i8> @lasx_xvbitseli_b(<32 x i8> %va, <32 x i8> %vb, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvbitseli.b(<32 x i8> %va, <32 x i8> %vb, i32 %c)
  ret <32 x i8> %res
}
